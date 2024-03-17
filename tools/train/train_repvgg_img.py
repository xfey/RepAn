import os
import torch
import torch.nn as nn
import torch.optim as optim

# DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

import core.config as config
import logger.meter as meter
import logger.logging as logging
import logger.checkpoint as checkpoint
from core.builder import setup_env
from core.config import cfg
from datasets.loader import get_normal_dataloader
from logger.meter import TrainMeter, TestMeter
from runner.criterion import KD_Loss, CrossEntropyLoss_label_smoothed

from net.repvgg_IMG import RepVGG_A1


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)

writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))


def main(local_rank, world_size):
    setup_env()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    
    # Networks
    student_net = RepVGG_A1(num_classes=1000).to(local_rank)
    student_net._train()
    # student_net.set_attach_rate(1.)
    # student_net.turn_only310(True)
    # student_net.load_state_dict(ckpt['model_state'])
    
    # Dataloaders
    [train_loader, valid_loader] = get_normal_dataloader()
    
    # Optim & Loss & LR
    # criterion = nn.CrossEntropyLoss()
    # criterion = KD_Loss(alpha=cfg.POST.ALPHA, temperature=cfg.POST.TEMPERATURE)
    # criterion = DKD_Loss(alpha=cfg.DECO.ALPHA, temperature=cfg.DECO.TEMPERATURE)
    net_params = [
        {"params": student_net.weights(rep=False), "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
        {"params": student_net.weights(rep=True), "weight_decay": 0},
    ]
    optimizer = optim.SGD(net_params, cfg.OPTIM.BASE_LR, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.OPTIM.MAX_EPOCH)
    
    # Meters
    train_meter = TrainMeter(len(train_loader))
    test_meter = TestMeter(len(valid_loader))
    best_top1 = 100.1
    
    student_net = DDP(student_net, device_ids=[local_rank])
    
    # Resume
    try:
        last_checkpoint = checkpoint.get_last_checkpoint(best=False)
        ckpt_epoch, ckpt_dict = checkpoint.load_checkpoint(last_checkpoint, student_net)
        ckpt_epoch = max(ckpt_epoch, 0)
        for _ in range(ckpt_epoch):
            scheduler.step()
    except:
        ckpt_epoch = 0
    
    dist.barrier()
    for cur_epoch in range(ckpt_epoch, cfg.OPTIM.MAX_EPOCH):
        train_epoch(cur_epoch, student_net, train_loader, train_meter, optimizer, scheduler, rank=local_rank)
        if local_rank == 0:
            if (cur_epoch + 1) % cfg.EVAL_PERIOD == 0 or (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH:
                top1err, top5err = test_epoch(cur_epoch, student_net, valid_loader, test_meter)
                if top1err < best_top1:
                    best_top1 = top1err
                    checkpoint.save_checkpoint(student_net, cur_epoch, best=True)
    # dist.barrier()
    torch.cuda.empty_cache()


def train_epoch(cur_epoch, student_net, train_loader, train_meter, optimizer, scheduler, rank):
    student_net.train()
    
    lr = scheduler.get_last_lr()[0]
    cur_step = cur_epoch * len(train_loader)
    train_meter.iter_tic()
    train_loader.sampler.set_epoch(cur_epoch)
    if rank == 0:
        writer.add_scalar('train/lr', lr, cur_epoch)
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Forward
        inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
        # with torch.no_grad():
        #     raw_teacher_preds = teacher_net(inputs)
        #     teacher_preds = raw_teacher_preds.clone().detach()
        student_preds = student_net(inputs)
        loss = CrossEntropyLoss_label_smoothed(student_preds, labels, cfg.TRAIN.LABEL_SMOOTH)
        # loss = criterion(student_preds, teacher_preds, labels)
        # loss = criterion(student_preds, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(student_net.weights(rep=False), cfg.OPTIM.GRAD_CLIP)
        optimizer.step()
        
        # Compute the errors
        top1_err, top5_err = meter.topk_errors(student_preds, labels, [1, 5])
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        
        # Update and log stats
        train_meter.update_stats(top1_err, top5_err, loss, lr, inputs.size(0) * cfg.NUM_GPUS)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        if rank == 0 and (cur_iter%10)==0:
            writer.add_scalar('train/loss', loss, cur_step)
            writer.add_scalar('train/top1_err', top1_err, cur_step)
            writer.add_scalar('train/top5_err', top5_err, cur_step)
        cur_step += 1
    # Log epoch stats
    top1_err = train_meter.get_epoch_stats(cur_epoch)["top1_err"]
    top5_err = train_meter.get_epoch_stats(cur_epoch)["top5_err"]
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    scheduler.step()
    # Saving checkpoint
    if (cur_epoch + 1) % cfg.SAVE_PERIOD == 0:
        checkpoint.save_checkpoint(student_net, cur_epoch, best=False)
    return top1_err, top5_err


@torch.no_grad()
def test_epoch(cur_epoch, net, test_loader, test_meter):
    net.eval()
    test_meter.reset()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = net(inputs)
        top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
        top1_err, top5_err = top1_err.item(), top5_err.item()

        test_meter.iter_toc()
        test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # top1_err = test_meter.mb_top1_err.get_global_avg()
    # top5_err = test_meter.mb_top5_err.get_global_avg()
    top1_err = test_meter.get_epoch_stats(cur_epoch)["top1_err"]
    top5_err = test_meter.get_epoch_stats(cur_epoch)["top5_err"]
    writer.add_scalar('val/top1_err', top1_err, cur_epoch)
    writer.add_scalar('val/top5_err', top5_err, cur_epoch)
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    return top1_err, top5_err


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23333'
    
    if torch.cuda.is_available():
        cfg.NUM_GPUS = torch.cuda.device_count()
    
    mp.spawn(main, nprocs=cfg.NUM_GPUS, args=(cfg.NUM_GPUS,), join=True)

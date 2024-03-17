import os
import time
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
from runner.criterion import KD_Loss

# from net.resnet_c100 import resnet101
from net.repvgg_IMG import RepVGG_A1

import timm


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)

writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

# manual barrier file
barrier_file = os.path.join(cfg.OUT_DIR, ".tmp_barrier")


def manual_barrier(rank):
    res_map = {'F': False, 'T': True}
    if rank != 0:
        while True:
            with open(barrier_file, 'r') as f:
                barr_res = f.readline()
                if res_map[barr_res[0]]:
                    return
                else:
                    time.sleep(5)
    else:
        with open(barrier_file, 'w') as f:
            f.write('T')
            return

def close_barrier(rank, current=False):
    if rank == 0:
        if not current:
            time.sleep(10)
        with open(barrier_file, 'w') as f:
            f.write('F')
            return


def main(local_rank, world_size):

    setup_env()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    
    # Networks
    teacher_net = timm.create_model(cfg.POST.TEACHER, pretrained=True)
    teacher_net.to(local_rank)
    teacher_net.eval()
    
    student_net = RepVGG_A1(num_classes=1000)
    student_net.to(local_rank)
    
    # Dataloaders
    [train_loader, valid_loader] = get_normal_dataloader()
    
    DEBUG_FLAG = True   # [tmp]
    cycle_time = 1      # [tmp] continue training
    start_epoch = 0
    
    # not used when debugging.
    cfg.POST.WARMUP = min(cfg.POST.WARMUP, cfg.OPTIM.MAX_EPOCH-1)
    
    while True:
        # for cycle_time in range(cfg.POST.CYCLE):
        if DEBUG_FLAG:  # [tmp]
            DEBUG_FLAG = False  # [tmp]
            # Optim & Loss & LR
            criterion = KD_Loss(alpha=cfg.POST.ALPHA, temperature=cfg.POST.TEMPERATURE)
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
            
        if cycle_time == 0:
            close_barrier(local_rank, current=True)
            student_net.load_official_checkpoints(cfg.POST.PATH)
            student_net._reparam(first=True)    # rep branches into Rep_conv
            student_net.inverse_turn_all(1.)    # inversion turn
            
            # Optim & Loss & LR
            criterion = KD_Loss(alpha=cfg.POST.ALPHA, temperature=cfg.POST.TEMPERATURE)
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
            
            # # [tmp]
            # from logger.checkpoint import load_checkpoint
            # # start_epoch, _ = load_checkpoint("exp/img/1020_cycle_lr05e120/checkpoints/model_epoch_0058.pyth", student_net)
            # # del _
            # ckpt = torch.load("exp/img/1020_cycle_lr05e120/checkpoints/best_model_epoch_0058.pyth", map_location="cpu")["model_state"]
            # start_epoch = 58
            # student_net.load_state_dict(ckpt)
            # del ckpt
            # for _ in range(start_epoch):
            #     scheduler.step()
            
            # DDP
            student_net = DDP(student_net, device_ids=[local_rank])
        else:
            logger.info('waiting barrier (rank={})'.format(local_rank))
            manual_barrier(local_rank)
            logger.info('passing barrier (rank={})'.format(local_rank))
            close_barrier(local_rank)
            ckpt = torch.load(os.path.join(cfg.OUT_DIR, "checkpoints", "cycle_"+str(cycle_time-1)+".pyth"), map_location='cpu')
            student_net.module.load_state_dict(ckpt['model_state'])
            student_net.module._reparam(first=True) # rep branches into Rep_conv
            student_net.module.inverse_turn_all(1.) # inversion turn
            
            # Optim & Loss & LR
            # # lr_decay
            # if hasattr(cfg.POST, "LR_DECAY"):
            #     cfg.OPTIM.BASE_LR = cfg.OPTIM.BASE_LR * (cfg.POST.LR_DECAY ** cycle_times)
            #     logger.info("Decayed LR for cycle {}: {}".format(cycle_times, cfg.OPTIM.BASE_LR))
            
            # clear
            del criterion, optimizer, scheduler
            torch.cuda.empty_cache()
            
            criterion = KD_Loss(alpha=cfg.POST.ALPHA, temperature=cfg.POST.TEMPERATURE)
            net_params = [
                {"params": student_net.module.weights(rep=False), "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
                {"params": student_net.module.weights(rep=True), "weight_decay": 0},
            ]
            optimizer = optim.SGD(net_params, cfg.OPTIM.BASE_LR, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.OPTIM.MAX_EPOCH)
            # Meters
            train_meter = TrainMeter(len(train_loader))
            test_meter = TestMeter(len(valid_loader))
            best_top1 = 100.1
        
        # dist.barrier()
        for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
            # if cur_epoch > 0:
            #     student_net.module.set_attach_rate(min(((cur_epoch + 1.) / cfg.POST.WARMUP), 1.) * 1.)
            train_epoch(cur_epoch, teacher_net, student_net, train_loader, train_meter, optimizer, scheduler, criterion, rank=local_rank, cycle=cycle_time)
            if local_rank == 0 and ((cur_epoch + 1) % cfg.EVAL_PERIOD == 0 or (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH):
                top1err, top5err = test_epoch(cur_epoch, student_net, valid_loader, test_meter, cycle=cycle_time)
                if top1err < best_top1:
                    best_top1 = top1err
                    checkpoint.save_checkpoint(student_net, cur_epoch, best=True)
                # checkpoint.save_checkpoint(student_net, cur_epoch, best=False)
            # dist.barrier()

        # save model for cycling
        if local_rank == 0:
            logger.info("Time_cycle:{} Best_top1:{}".format(cycle_time, test_meter.min_top1_err))
            checkpoint_net = {
                # "epoch": cur_epoch,
                "model_state": student_net.module.state_dict(),
            }
            torch.save(checkpoint_net, os.path.join(cfg.OUT_DIR, "checkpoints", "cycle_"+str(cycle_time)+".pyth"))
            del checkpoint_net
        
        cycle_time += 1
    
    torch.cuda.empty_cache()
    exit(0)


def train_epoch(cur_epoch, teacher_net, student_net, train_loader, train_meter, optimizer, scheduler, criterion, rank, cycle):
    teacher_net.eval()
    student_net.train()
    
    lr = scheduler.get_last_lr()[0]
    cur_step = cur_epoch * len(train_loader)
    train_loader.sampler.set_epoch(cur_epoch)
    train_meter.iter_tic()
    if rank == 0:
        writer.add_scalar('train/lr', lr, cur_epoch)
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        
        # Forward
        inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
        with torch.no_grad():
            raw_teacher_preds = teacher_net(inputs)
            teacher_preds = raw_teacher_preds.clone().detach()
        student_preds = student_net(inputs)
        loss = criterion(student_preds, teacher_preds, labels)
        # loss = criterion(student_preds, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student_net.module.weights(rep=False), cfg.OPTIM.GRAD_CLIP)
        optimizer.step()
        
        # Compute the errors
        top1_err, top5_err = meter.topk_errors(student_preds, labels, [1, 5])
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        
        # Update and log stats
        train_meter.update_stats(top1_err, top5_err, loss, lr, inputs.size(0))
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        if rank == 0 and (cur_iter%10)==0:
            writer.add_scalar('train/loss_cycle'+str(cycle), loss, cur_step)
            writer.add_scalar('train/top1_err_cycle'+str(cycle), top1_err, cur_step)
            writer.add_scalar('train/top5_err_cycle'+str(cycle), top5_err, cur_step)
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
def test_epoch(cur_epoch, net, test_loader, test_meter, cycle):
    net.eval()
    test_meter.reset()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = net(inputs)
        top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
        top1_err, top5_err = top1_err.item(), top5_err.item()

        test_meter.iter_toc()
        test_meter.update_stats(top1_err, top5_err, inputs.size(0))
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # top1_err = test_meter.mb_top1_err.get_global_avg()
    # top5_err = test_meter.mb_top5_err.get_global_avg()
    top1_err = test_meter.get_epoch_stats(cur_epoch)["top1_err"]
    top5_err = test_meter.get_epoch_stats(cur_epoch)["top5_err"]
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    writer.add_scalar('test/top1err_cycle'+str(cycle), top1_err, cur_epoch)
    writer.add_scalar('test/top5err_cycle'+str(cycle), top5_err, cur_epoch)
    return top1_err, top5_err


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23333'
    
    if torch.cuda.is_available():
        cfg.NUM_GPUS = torch.cuda.device_count()
    
    mp.spawn(main, nprocs=cfg.NUM_GPUS, args=(cfg.NUM_GPUS,), join=True)

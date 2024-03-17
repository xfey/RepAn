"""
    Synflow metric.
    Implemented from https://github.com/ganguli-lab/Synaptic-Flow
"""


import torch
import torch.nn as nn
from logger.meter import AverageMeter


def synflow_metric(net):
    scores = {}
    
    @torch.no_grad()
    def linearize(model):
        # model.double()
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        # model.float()
        for name, param in model.state_dict().items():
            param.mul_(signs[name])
    
    signs = linearize(net)
    # (data, _) = next(iter(dataloader))
    # input_dim = list(data[0,:].shape)
    # input = torch.ones([1] + input_dim).cuda()#, dtype=torch.float64).to(device)
    # output = net(input)
    # torch.sum(output).backward()
    
    for n, p in net.named_parameters():
        if "conv_3" in n:
            if p.grad is not None:
                scores[n[:-14]] = torch.clone(p.grad * p).detach().abs_().sum().item()
            else:
                scores[n[:-14]] = None
        elif "conv_1" in n:
            if p.grad is not None:
                if scores[n[:-14]] is not None:
                    scores[n[:-14]] = scores[n[:-14]] + torch.clone(p.grad * p).detach().abs_().sum().item()
                else:
                    scores[n[:-14]] = torch.clone(p.grad * p).detach().abs_().sum().item()
        
        if p.grad is not None:
            p.grad.data.zero_()

    nonlinearize(net, signs)
    return scores


def record_metrics(net:nn.Module):
    metric_records = {}
    
    for n,m in net.named_modules():
        if not "conv_3" in n:
            continue
        
        if m.weight.grad is None:
            metric_records[n] = None
            continue
        
        grad = m.weight.grad.data.reshape(-1)
        weight = m.weight.data.reshape(-1)
        
        metric_val = (grad * weight).sum().item()
        
        if n not in metric_records:
            metric_records[n] = AverageMeter()
        metric_records[n].update(metric_val)
    return metric_records

def calculate_records(net:nn.Module):
    metric_records = record_metrics(net)
    records = {}
    for key,value in metric_records.items():
        if value is not None:
            records[key] = metric_records[key].avg
        else:
            records[key] = None
    return records

# def calculate_records_all(metric_records):
#     records = calculate_records(metric_records)
#     return sum(records.values()) / len(records)

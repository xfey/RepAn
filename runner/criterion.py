"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.config import cfg


def _label_smooth(target, n_classes: int, label_smoothing):
    # convert to one-hot
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target


def CrossEntropyLoss_soft_target(pred, soft_target):
    """CELoss with soft target, mainly used during KD"""
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), dim=1))


def CrossEntropyLoss_label_smoothed(pred, target, label_smoothing=0.1):
    label_smoothing = cfg.TRAIN.LABEL_SMOOTH if label_smoothing == 0.1 else label_smoothing
    soft_target = _label_smooth(target, pred.size(1), label_smoothing)
    return CrossEntropyLoss_soft_target(pred, soft_target)


# class KLLossSoft(torch.nn.modules.loss._Loss):
#     """ inplace distillation for image classification 
#             output: output logits of the student network
#             target: output logits of the teacher network
#             T: temperature
#             KL(p||q) = Ep \log p - \Ep log q
#     """
#     def forward(self, output, soft_logits, target=None, temperature=4., alpha=0.1):     # NOTE: updated
#         output, soft_logits = output / temperature, soft_logits / temperature
#         soft_target_prob = F.softmax(soft_logits, dim=1)
#         output_log_prob = F.log_softmax(output, dim=1)
#         kd_loss = -torch.sum(soft_target_prob * output_log_prob, dim=1)
#         if target is not None:
#             n_class = output.size(1)
#             target = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
#             target = target.unsqueeze(1)
#             output_log_prob = output_log_prob.unsqueeze(2)
#             ce_loss = -torch.bmm(target, output_log_prob).squeeze()
#             loss = alpha * temperature * temperature * kd_loss + (1.0 - alpha) * ce_loss
#         else:
#             loss = kd_loss 
        
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         return loss


class KD_Loss(nn.Module):
    def __init__(self, alpha, temperature):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        
    def forward(self, output, soft_logits, target=None):
        kldivloss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output/self.temperature, dim=1),
                                                        F.softmax(soft_logits/self.temperature, dim=1))
        if target is not None:
            celoss =  F.cross_entropy(output, target)
            total_loss = self.alpha * (self.temperature**2) * kldivloss + (1. - self.alpha) * celoss
        else:
            total_loss = kldivloss
        return total_loss



class FD_loss(nn.Module):
	'''
	Pay Attention to Features, Transfer Learn Faster CNNs
	https://openreview.net/pdf?id=ryxyCeHtPB
	'''
	def __init__(self):
		super(FD_loss, self).__init__()
		
	def forward(self, fm_s, fm_t, eps=1e-6):
		fm_s_norm = torch.norm(fm_s, dim=(2,3), keepdim=True)
		fm_s      = torch.div(fm_s, fm_s_norm+eps)
		fm_t_norm = torch.norm(fm_t, dim=(2,3), keepdim=True)
		fm_t      = torch.div(fm_t, fm_t_norm+eps)

		loss = torch.pow(fm_s-fm_t, 2).mean(dim=(2,3))
		loss = loss.sum(1).mean(0)

		return loss

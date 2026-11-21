"""Original Student objective with global masked denominators for DDP gradients."""
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path('/data/adminad/workspace/FineVLA_clean')
sys.path.insert(0, str(ROOT / 'experiments/pi05_native_student_20261016'))
from model import iou


def objective(net, pred, batch, distributed=False, train_validity=True):
    """DDP averages gradients, so each local numerator is scaled by world size.

    Evaluation uses distributed=False on rank zero. Its semantics then match
    the original single-process objective, including absent-class masking.
    """
    world = dist.get_world_size() if distributed else 1
    target, positive = batch['boxes'], batch['loc_valid']
    present = batch['supply'] == 1
    has_positive = present.any(-1)
    counts = torch.stack([
        positive.sum(), has_positive.sum(),
        (batch['supply'] == 0).sum(), (batch['supply'] == 1).sum(),
        (batch['candidate_labels'] == 0).sum(), (batch['candidate_labels'] == 1).sum(),
    ]).to(dtype=pred['heat'].dtype)
    if distributed:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    low, high = net.centers - .5 / 8, net.centers + .5 / 8
    area = (torch.minimum(target[..., None, 2:], high) -
            torch.maximum(target[..., None, :2], low)).clamp_min(0).prod(-1)
    heat = area / area.sum(-1, keepdim=True).clamp_min(1e-8)
    location = -(heat * pred['heat'].log_softmax(-1)).sum(-1)
    regression = (5 * F.smooth_l1_loss(pred['boxes'], target, reduction='none').sum(-1)
                  + 1 - iou(pred['boxes'], target, True))
    localization = ((location + regression) * positive).sum() * world / counts[0].clamp_min(1)

    def balanced(logits, labels, class_counts):
        ce = F.cross_entropy(logits.reshape(-1, 2), labels.clamp_min(0).reshape(-1),
                             reduction='none').reshape_as(labels)
        active = (class_counts > 0).to(dtype=ce.dtype)
        numerators = torch.stack([(ce * (labels == cls)).sum() for cls in (0, 1)])
        return (numerators * world / class_counts.clamp_min(1) * active).sum() / active.sum().clamp_min(1)

    supply = balanced(pred['supply'], batch['supply'], counts[2:4]) if train_validity else pred['supply'].sum() * 0
    candidate = balanced(pred['candidate_verification'], batch['candidate_labels'], counts[4:6])
    known = batch['supply'] >= 0
    target_weight = present.float() / present.sum(-1, keepdim=True).clamp_min(1)
    scores = pred['supply'].softmax(-1)[..., 1].clamp_min(1e-6).log()
    log_probability = scores.masked_fill(~known, -1e4).log_softmax(-1)
    view = ((-(target_weight * log_probability).sum(-1) * has_positive).sum() * world / counts[1].clamp_min(1)
            if train_validity else pred['supply'].sum() * 0)
    return {
        'loss': localization + supply + candidate + .1 * view,
        'localization': localization, 'view_validity': supply,
        'box_verification': candidate, 'view_distribution': view,
    }


def mean_metrics(losses):
    keys = list(losses)
    values = torch.stack([losses[key].detach() for key in keys])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return dict(zip(keys, values.cpu().tolist()))

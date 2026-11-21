"""Language-conditioned, shared-view Student. No task IDs or fixed object heads."""
import torch
from torch import nn
import torch.nn.functional as F


def iou(a, b, generalized=False):
    lo, hi = torch.maximum(a[..., :2], b[..., :2]), torch.minimum(a[..., 2:], b[..., 2:])
    inter = (hi - lo).clamp_min(0).prod(-1)
    union = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1) + (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1) - inter
    value = inter / union.clamp_min(1e-7)
    if generalized:
        cover = (torch.maximum(a[..., 2:], b[..., 2:]) - torch.minimum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
        value = value - (cover - union) / cover.clamp_min(1e-7)
    return value


class Student(nn.Module):
    def __init__(self, dim=2560, width=128, grid_size=8):
        super().__init__()
        self.config = dict(dim=dim, width=width, grid_size=grid_size)
        self.visual = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU())
        self.language = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, width), nn.GELU())
        self.position = nn.Linear(2, width, bias=False)
        self.local = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        self.condition = nn.Sequential(nn.Linear(3 * width, width), nn.GELU())
        self.first_heat = nn.Linear(width, 1)
        self.cross = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.refine = nn.Sequential(nn.Linear(2 * width, width), nn.GELU())
        self.heat = nn.Linear(width, 1)
        self.box = nn.Linear(width, 4)
        self.supply = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 2))
        self.verify = nn.Sequential(nn.Linear(3 * width + 4, width), nn.GELU(), nn.Linear(width, 2))
        axis = (torch.arange(grid_size).float() + .5) / grid_size
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('centers', torch.stack([xx, yy], -1).reshape(-1, 2))

    def verify_boxes(self, local, context, language, boxes):
        # Continuous overlap pooling works for small/edge boxes, without a size cutoff.
        side = self.config['grid_size']
        low = self.centers - .5 / side
        high = self.centers + .5 / side
        area = (torch.minimum(boxes[..., None, 2:], high) - torch.maximum(boxes[..., None, :2], low)).clamp_min(0).prod(-1)
        weights = area / area.sum(-1, keepdim=True).clamp_min(1e-8)
        roi = torch.einsum('bvkn,bvnd->bvkd', weights, local)
        shape = roi.shape
        return self.verify(torch.cat([roi, context[:, :, None].expand(*shape), language[:, None, None].expand(*shape), boxes], -1))

    def forward(self, grid, category, colors, candidates=None):
        b, v, h, w, _ = grid.shape
        x = self.visual(grid.float())
        x = x + self.local(x.reshape(b*v, h, w, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape_as(x)
        x = x.flatten(2, 3) + self.position(self.centers)
        lang = self.language(torch.cat([category.float(), colors.float()], -1))
        language = lang[:, None, None].expand_as(x)
        local = self.condition(torch.cat([x, language, x * language], -1))
        coarse = self.first_heat(local).squeeze(-1).softmax(-1)
        token = (local * coarse[..., None]).sum(-2)
        context = self.cross_norm(token + self.cross(token, token, token, need_weights=False)[0])
        refined = self.refine(torch.cat([local, context[:, :, None].expand_as(local)], -1))
        logits = self.heat(refined).squeeze(-1)
        parameters = self.box(refined)
        center = (self.centers + parameters[..., :2].tanh() / self.config['grid_size']).clamp(0, 1)
        wh = parameters[..., 2:].sigmoid()
        box = torch.cat([(center - wh/2).clamp(0, 1), (center + wh/2).clamp(0, 1)], -1)
        box = (box * logits.softmax(-1)[..., None]).sum(-2)
        foreground = (local * logits.softmax(-1)[..., None]).sum(-2)
        supply = self.supply(torch.cat([foreground, context, lang[:, None].expand_as(context)], -1))
        own = self.verify_boxes(local, context, lang, box[:, :, None]).squeeze(2)
        out = dict(boxes=box, heat=logits, supply=supply, own_verification=own)
        if candidates is not None:
            out['candidate_verification'] = self.verify_boxes(local, context, lang, candidates)
        return out


def decode(pred):
    # Learned two-class decisions, no task-specific score cutoff or forced best camera.
    accepted = (pred['supply'].argmax(-1) == 1) & (pred['own_verification'].argmax(-1) == 1)
    score = pred['supply'].softmax(-1)[..., 1] * pred['own_verification'].softmax(-1)[..., 1]
    weight = score * accepted
    weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
    return dict(boxes=pred['boxes'], accepted=accepted, score=score, view_weights=weight,
                all_rejected=~accepted.any(-1))


def balanced_ce(logits, labels, valid):
    ce = F.cross_entropy(logits.flatten(0, -2), labels.clamp_min(0).flatten(), reduction='none').reshape_as(labels)
    terms = []
    for cls in [0, 1]:
        mask = valid & (labels == cls)
        terms.append((ce * mask).sum() / mask.sum().clamp_min(1))
    return sum(terms) / 2


def losses(model, pred, batch):
    present = batch['supply'] == 1
    target = batch['boxes']
    side = model.config['grid_size']
    # Box-derived soft location supervision; no claims of segmentation ground truth.
    lo, hi = model.centers - .5/side, model.centers + .5/side
    area = (torch.minimum(target[..., None, 2:], hi) - torch.maximum(target[..., None, :2], lo)).clamp_min(0).prod(-1)
    heat = area / area.sum(-1, keepdim=True).clamp_min(1e-8)
    loc = -(heat * pred['heat'].log_softmax(-1)).sum(-1)
    regression = 5 * F.smooth_l1_loss(pred['boxes'], target, reduction='none').sum(-1) + 1 - iou(pred['boxes'], target, True)
    positive = present.sum().clamp_min(1)
    localization = ((loc + regression) * present).sum() / positive
    supply = balanced_ce(pred['supply'], batch['supply'], batch['supply'] >= 0)
    candidate = balanced_ce(pred['candidate_verification'], batch['candidate_labels'], batch['candidate_labels'] >= 0)
    # Equal usable-view target is deliberate: ordinal Qwen scores are not calibrated trust.
    logp = pred['supply'].softmax(-1)[..., 1].clamp_min(1e-6).log().log_softmax(-1)
    view_target = present.float() / present.sum(-1, keepdim=True).clamp_min(1)
    any_positive = present.any(-1)
    views = (-(view_target * logp).sum(-1) * any_positive).sum() / any_positive.sum().clamp_min(1)
    total = localization + supply + candidate + .1 * views
    return dict(loss=total, localization=localization, supply=supply, candidate=candidate, view_consistency=views)

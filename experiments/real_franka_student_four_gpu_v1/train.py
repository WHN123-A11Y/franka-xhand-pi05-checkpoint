"""Four-rank Student training with globally normalized masked objectives."""
from pathlib import Path
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path('/data/adminad/workspace/FineVLA_clean')
sys.path.insert(0, str(ROOT / 'experiments/pi05_native_student_20261016'))
from model import Student, decode, iou
from ddp_objective import objective, mean_metrics

FULL_HEADS = ['localization', 'view_validity', 'box_verification', 'view_distribution']
PARTIAL_HEADS = ['localization', 'box_verification']


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda_current_device': torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None}


def restore_rng(value, device):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu())
    if value.get('cuda_current_device') is not None:
        torch.cuda.set_rng_state(value['cuda_current_device'].cpu(), device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/real_franka_full_training_20261117/student')
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--eval-interval', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=24, help='Global batch size, divided across ranks')
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--allow-unidentifiable-validity', action='store_true',
                        help='Freeze supply and omit validity/distribution losses if explicit negatives are absent')
    args = parser.parse_args()
    rank, local_rank, world = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    if world != 4 or args.batch_size % world:
        raise ValueError('This workflow requires four ranks and a global batch divisible by four')
    torch.set_num_threads(4)
    device = torch.device('cuda', local_rank) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=datetime.timedelta(minutes=30))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = None
    if rank == 0:
        lock = (args.out / 'train.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dist.barrier()

    def status(stage, **values):
        if rank == 0:
            atomic(args.out / 'training_status.json', {'stage': stage, 'time': time.time(),
                   'world_size': world, 'global_batch_size': args.batch_size,
                   'batch_per_rank': args.batch_size // world, **values})
            print(json.dumps({'stage': stage, **values}), flush=True)

    for name in ('LABELS_READY.json', 'FEATURES_READY.json', 'feature_config.json'):
        if not (args.out / name).is_file():
            raise FileNotFoundError(name)
    ready = json.loads((args.out / 'LABELS_READY.json').read_text())
    train_validity = bool(ready['train_supply_identifiable'])
    if not ready['train_candidate_identifiable']:
        raise ValueError('Candidate verification needs both explicit positive and negative training labels')
    if not train_validity and not args.allow_unidentifiable_validity:
        raise ValueError('View validity is unidentifiable without explicit presence negatives')
    trained_heads = FULL_HEADS if train_validity else PARTIAL_HEADS
    disabled_heads = [] if train_validity else ['view_validity', 'view_distribution']
    samples = json.loads((args.out / 'samples.json').read_text())
    feature_config = json.loads((args.out / 'feature_config.json').read_text())
    parts = {split: [sample for sample in samples if sample['split'] == split] for split in ('train', 'val', 'test')}
    episodes = {split: {sample['episode'] for sample in rows} for split, rows in parts.items()}
    assert not (episodes['train'] & episodes['val'] or episodes['train'] & episodes['test'] or episodes['val'] & episodes['test'])
    assert all(parts.values())
    bank = torch.load(args.out / 'language.pt', map_location='cpu', weights_only=False)
    features = {}
    status('loading_features', step=0, total=args.steps, trained_heads=trained_heads, disabled_heads=disabled_heads)
    for sample in samples:
        if sample['scene'] not in features:
            value = torch.load(args.out / 'features' / (sample['scene'] + '.pt'), map_location='cpu', weights_only=False)
            assert value['selection_sha256'] == feature_config['policy_selection_sha256']
            features[sample['scene']] = value['grid']
    dimension = next(iter(features.values())).shape[-1]
    net = Student(dim=dimension, width=args.width, grid_size=8).to(device)
    if not train_validity:
        net.supply.requires_grad_(False)
    ddp = DistributedDataParallel(net, device_ids=[local_rank] if device.type == 'cuda' else None,
                                  find_unused_parameters=False, broadcast_buffers=False)
    optimizer = torch.optim.AdamW([parameter for parameter in net.parameters() if parameter.requires_grad],
                                 lr=args.lr, weight_decay=1e-4)
    identity = {
        'samples_sha256': sha(args.out / 'samples.json'), 'feature_config_sha256': sha(args.out / 'feature_config.json'),
        'language_sha256': sha(args.out / 'language.pt'), 'source_sha256': sha(__file__),
        'objective_source_sha256': sha(Path(__file__).with_name('ddp_objective.py')),
        'model_source_sha256': feature_config['student_model_source_sha256'],
        'seed': args.seed, 'steps': args.steps, 'batch_size': args.batch_size,
        'width': args.width, 'lr': args.lr, 'world_size': world,
        'trained_heads': trained_heads, 'disabled_heads': disabled_heads,
        'sampling': 'identical global Python choices on every rank, then contiguous rank partition',
    }
    counts = collections.Counter()
    for sample in parts['train']:
        for label in sample['supply']:
            counts[{1: 'supply_positive', 0: 'supply_negative', -1: 'supply_unknown'}[label]] += 1
        for labels in sample['candidate_labels']:
            for label in labels:
                if label >= 0:
                    counts['candidate_positive' if label == 1 else 'candidate_negative'] += 1
        counts['localization_positive'] += sum(sample['loc_valid'])

    def batch(rows):
        value = {
            'grid': torch.stack([features[sample['scene']] for sample in rows]).to(device),
            'category': torch.stack([bank[sample['expression']] for sample in rows]).to(device),
            'colors': torch.stack([bank[''] for _ in rows]).to(device),
        }
        for key, dtype in [('boxes', torch.float32), ('loc_valid', torch.bool), ('supply', torch.long),
                           ('candidate_boxes', torch.float32), ('candidate_labels', torch.long)]:
            value[key] = torch.tensor([sample[key] for sample in rows], dtype=dtype, device=device)
        return value

    def decode_for_scope(prediction):
        if train_validity:
            return decode(prediction)
        accepted = prediction['own_verification'].argmax(-1) == 1
        score = prediction['own_verification'].softmax(-1)[..., 1]
        display_weight = score * accepted
        display_weight = display_weight / display_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        return {'boxes': prediction['boxes'], 'accepted': accepted, 'score': score,
                'candidate_display_weights': display_weight, 'all_rejected': ~accepted.any(-1)}

    @torch.inference_mode()
    def evaluate(rows, with_predictions=False):
        # Deliberately bypass DDP: only rank zero evaluates, so no collectives here.
        net.eval()
        total, predictions = collections.Counter(), []
        for start in range(0, len(rows), args.batch_size):
            selected_rows = rows[start:start + args.batch_size]
            value = batch(selected_rows)
            prediction = net(value['grid'], value['category'], value['colors'], value['candidate_boxes'])
            losses = objective(net, prediction, value, distributed=False, train_validity=train_validity)
            decoded = decode_for_scope(prediction)
            for key, loss in losses.items():
                total[key] += float(loss) * len(selected_rows)
            overlap = iou(prediction['boxes'], value['boxes'])
            total['localization_iou_sum'] += float((overlap * value['loc_valid']).sum())
            total['localization_views'] += int(value['loc_valid'].sum())
            for key, logits, labels in [('view_validity', prediction['supply'], value['supply']),
                                        ('box_verification', prediction['candidate_verification'], value['candidate_labels'])]:
                if key == 'view_validity' and not train_validity:
                    continue
                decisions = logits.argmax(-1)
                for cls in (0, 1):
                    mask = labels == cls
                    total[f'{key}_class_{cls}_correct'] += int(((decisions == labels) & mask).sum())
                    total[f'{key}_class_{cls}_count'] += int(mask.sum())
            known = value['supply'] >= 0
            total['accepted_known_views'] += int((decoded['accepted'] & known).sum())
            total['accepted_known_negative_views'] += int((decoded['accepted'] & (value['supply'] == 0)).sum())
            if with_predictions:
                for index, sample in enumerate(selected_rows):
                    entry = {key: sample[key] for key in ('episode', 'frame', 'role', 'instruction', 'expression', 'scene', 'split')}
                    entry.update(images=sample['images'], boxes=decoded['boxes'][index].cpu().tolist(),
                                 accepted=decoded['accepted'][index].cpu().tolist(), score=decoded['score'][index].cpu().tolist(),
                                 view_weights=decoded['view_weights'][index].cpu().tolist() if train_validity else None,
                                 all_rejected=bool(decoded['all_rejected'][index]),
                                 teacher_presence_labels=sample['supply'], teacher_box_labels=sample['boxes'],
                                 teacher_box_valid=sample['loc_valid'], view_validity_trained=train_validity,
                                 view_validity=None if not train_validity else prediction['supply'][index].softmax(-1)[..., 1].cpu().tolist(),
                                 acceptance_source='trained supply and candidate verification' if train_validity else 'candidate verification only; view validity untrained')
                    if not train_validity:
                        entry['candidate_display_weights'] = decoded['candidate_display_weights'][index].cpu().tolist()
                    predictions.append(entry)
        result = {key: total[key] / max(len(rows), 1) for key in ('loss', 'localization', 'view_validity', 'box_verification', 'view_distribution')}
        result.update(samples=len(rows), mean_iou_to_teacher=total['localization_iou_sum'] / max(total['localization_views'], 1),
                      localization_views=total['localization_views'], trained_heads=trained_heads, disabled_heads=disabled_heads)
        for key in ('view_validity', 'box_verification'):
            accuracies = [total[f'{key}_class_{cls}_correct'] / total[f'{key}_class_{cls}_count']
                          for cls in (0, 1) if total[f'{key}_class_{cls}_count']]
            result[key + '_balanced_accuracy_to_teacher'] = (sum(accuracies) / len(accuracies)) if accuracies else None
            result[key + '_counts'] = {str(cls): total[f'{key}_class_{cls}_count'] for cls in (0, 1)}
        result['accepted_known_negative_views'] = total['accepted_known_negative_views']
        result['accepted_known_views'] = total['accepted_known_views']
        return result, predictions

    start, best, best_step, history = 0, float('inf'), 0, []
    last = args.out / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Student resume identity changed')
        net.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng_by_rank'][rank], device)
        start, best, best_step = saved['step'], saved['best_validation_loss'], saved['best_step']
        history = json.loads((args.out / 'history.json').read_text())
    if rank == 0:
        atomic(args.out / 'training_inputs.json', {'identity': identity, 'counts': dict(counts),
               'splits': {split: len(rows) for split, rows in parts.items()}, 'trained_heads': trained_heads,
               'disabled_heads': disabled_heads, 'unknown_labels_masked': True, 'test_used_for_selection': False,
               'feature_cache_cpu_bytes_per_rank': sum(grid.numel() * grid.element_size() for grid in features.values())})
    status('training', step=start, total=args.steps, trained_heads=trained_heads, disabled_heads=disabled_heads)

    def packet(step, rng_by_rank):
        return {'config': net.config, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                'rng': rng_by_rank[0], 'rng_by_rank': rng_by_rank, 'step': step, 'best_step': best_step,
                'best_validation_loss': best, 'identity': identity, 'feature_config': feature_config,
                'trained_heads': trained_heads, 'disabled_heads': disabled_heads, 'supervision_counts': dict(counts),
                'scope': 'Teacher pseudo supervision; partial identifiable heads only' if not train_validity else 'all-head Student Teacher pseudo supervision; no human truth or robot success claim'}

    local_batch_size = args.batch_size // world
    for step in range(start + 1, args.steps + 1):
        ddp.train()
        global_indices = random.choices(range(len(parts['train'])), k=args.batch_size)
        indices = global_indices[rank * local_batch_size:(rank + 1) * local_batch_size]
        value = batch([parts['train'][index] for index in indices])
        prediction = ddp(value['grid'], value['category'], value['colors'], value['candidate_boxes'])
        losses = objective(net, prediction, value, distributed=True, train_validity=train_validity)
        finite = torch.isfinite(losses['loss']).to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError('Nonfinite objective on at least one rank')
        optimizer.zero_grad(set_to_none=True)
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step()
        report = step % 20 == 0 or step % args.eval_interval == 0 or step == args.steps
        metrics = mean_metrics(losses) if report else None
        if step % 20 == 0:
            status('training', step=step, total=args.steps, best_step=best_step,
                   train_loss=metrics['loss'], train=metrics, trained_heads=trained_heads, disabled_heads=disabled_heads)
        if step % args.eval_interval == 0 or step == args.steps:
            dist.barrier()
            validation = None
            improved = False
            if rank == 0:
                validation, _ = evaluate(parts['val'])
                history.append({'step': step, 'train': metrics, 'val': validation})
                if validation['loss'] < best:
                    best, best_step, improved = validation['loss'], step, True
            selected = torch.tensor([best, best_step], dtype=torch.float64, device=device)
            dist.broadcast(selected, src=0)
            best, best_step = float(selected[0]), int(selected[1])
            rank_states = [None] * world
            dist.all_gather_object(rank_states, rng(device))
            if rank == 0:
                if improved:
                    save(args.out / 'best.pt', packet(step, rank_states))
                save(last, packet(step, rank_states))
                atomic(args.out / 'history.json', history)
                status('training', step=step, total=args.steps, best_step=best_step,
                       train_loss=metrics['loss'], validation=validation,
                       trained_heads=trained_heads, disabled_heads=disabled_heads)
            dist.barrier()

    dist.barrier()
    if rank == 0:
        selected = torch.load(args.out / 'best.pt', map_location='cpu', weights_only=False)
        net.load_state_dict(selected['state_dict'])
        validation, _ = evaluate(parts['val'])
        test, test_predictions = evaluate(parts['test'], True)
        _, all_predictions = evaluate(samples, True)
        metadata = {'selected_step': selected['step'], 'checkpoint_sha256': sha(args.out / 'best.pt'),
                    'trained_heads': trained_heads, 'disabled_heads': disabled_heads, 'supervision_counts': dict(counts),
                    'label_source': 'Teacher pseudo supervision, not human ground truth',
                    'native_policy_checkpoint': feature_config['native_policy_checkpoint'],
                    'policy_selection_sha256': feature_config['policy_selection_sha256'],
                    'view_validity_trained': train_validity, 'view_distribution_trained': train_validity,
                    'candidate_display_weights_are_learned_view_weights': False if not train_validity else None}
        atomic(args.out / 'predictions_all.json', {'metadata': metadata, 'samples': all_predictions})
        atomic(args.out / 'test_predictions.json', test_predictions)
        atomic(args.out / 'training_summary.json', {'best_step': selected['step'], 'validation': validation, 'test': test,
               'trained_heads': trained_heads, 'disabled_heads': disabled_heads, 'supervision_counts': dict(counts),
               'test_used_for_selection': False, 'scope': 'Teacher agreement; partial identifiable heads only' if not train_validity else 'Teacher agreement, no real-robot success claim'})
        result = {'checkpoint': str((args.out / 'best.pt').resolve()), 'checkpoint_sha256': sha(args.out / 'best.pt'),
                  'step': selected['step'], 'trained_heads': trained_heads, 'disabled_heads': disabled_heads,
                  'supervision_counts': dict(counts), 'feature_config': str((args.out / 'feature_config.json').resolve()),
                  'native_policy_checkpoint': feature_config['native_policy_checkpoint'],
                  'policy_selection_sha256': feature_config['policy_selection_sha256'],
                  'language': str((args.out / 'language.pt').resolve()), 'expressions': str((args.out / 'expressions.json').resolve()),
                  'full_student_acceptance_ready': train_validity, 'world_size': world}
        atomic(args.out / ('READY.json' if train_validity else 'PARTIAL_READY.json'), result)
        if not train_validity:
            atomic(args.out / 'LOCALIZATION_READY.json', result)
        status('finished', best_step=selected['step'], validation=validation, test=test,
               trained_heads=trained_heads, disabled_heads=disabled_heads, full_student_acceptance_ready=train_validity)
    dist.barrier()
    dist.destroy_process_group()
    if lock:
        lock.close()


if __name__ == '__main__':
    main()

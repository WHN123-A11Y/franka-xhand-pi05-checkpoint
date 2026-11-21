"""Freeze existing Teacher supervision; preserve unknown labels without new calls."""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import sys
import time

ROOT = Path('/data/adminad/workspace/FineVLA_clean')
ORIGINAL = ROOT / 'experiments/real_franka_full_training_v1/student/prepare.py'
DEFAULT_OUT = ROOT / 'runs/real_franka_full_training_20261117/student'


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'last.pt').exists():
        raise RuntimeError('Do not regenerate frozen supervision during an existing Student training attempt')
    sys.argv = [str(ORIGINAL), '--out', str(args.out), '--allow-incomplete']
    runpy.run_path(str(ORIGINAL), run_name='__main__')
    audit = json.loads((args.out / 'supervision_audit.json').read_text())
    counts = audit['counts']
    if counts.get('teacher_error', 0):
        raise RuntimeError('Existing Teacher exceptions require an explicit exclusion audit')
    samples = json.loads((args.out / 'samples.json').read_text())
    scenes = json.loads((args.out / 'scenes.json').read_text())
    train_candidate_identifiable = counts.get('train_candidate_0', 0) > 0 and counts.get('train_candidate_1', 0) > 0
    train_supply_identifiable = counts.get('train_supply_0', 0) > 0 and counts.get('train_supply_1', 0) > 0
    if not train_candidate_identifiable or not counts.get('train_localization_views', 0):
        raise RuntimeError('Existing supervision does not identify localization and candidate verification')
    selection = args.out.parent / 'policy/selected.json'
    if not (selection.parent / 'READY.json').exists():
        raise RuntimeError('Selected OFT checkpoint is not ready')
    result = {
        'time': time.time(), 'samples': len(samples), 'scenes': len(scenes),
        'source_digest': audit['source_digest'],
        'train_supply_identifiable': train_supply_identifiable,
        'train_candidate_identifiable': train_candidate_identifiable,
        'teacher_complete': counts.get('teacher_complete', 0),
        'teacher_missing_excluded': counts.get('teacher_missing', 0),
        'teacher_total_expected': counts.get('teacher_complete', 0) + counts.get('teacher_missing', 0),
        'teacher_source_frozen_at_user_request': True,
        'new_teacher_or_presence_calls_allowed': False,
        'missing_and_unverified_views': 'unknown -1; excluded from supervised losses; not negative labels',
        'supervision_scope': 'localization and candidate verification; no identifiable full-view validity negatives',
        'planned_trained_heads': ['localization', 'box_verification'],
        'inactive_heads': ['view_validity', 'view_distribution'],
        'samples_sha256': hashlib.sha256((args.out / 'samples.json').read_bytes()).hexdigest(),
        'policy_selection_sha256': hashlib.sha256(selection.read_bytes()).hexdigest(),
        'is_human_ground_truth': False,
    }
    atomic(args.out / 'LABELS_READY.json', result)
    control = args.out / 'four_gpu'
    control.mkdir(exist_ok=True)
    atomic(control / 'supervision_frozen.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

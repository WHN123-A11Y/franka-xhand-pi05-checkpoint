"""Resume the unchanged Franka SFT implementation on an explicit eight-device mesh."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

ROOT = Path('/data/adminad/workspace/FineVLA_clean')
ORIGINAL = ROOT / 'experiments/real_franka_full_training_v1/policy/train.py'


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--topology-audit-dir', type=Path, required=True)
    args, original_args = parser.parse_known_args()
    args.topology_audit_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ORIGINAL.parent))
    spec = importlib.util.spec_from_file_location('franka_sft_original', ORIGINAL)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    import jax
    import orbax.checkpoint as ocp
    from orbax.checkpoint import checkpoint_utils
    from openpi.shared import array_typing

    devices = jax.devices()
    if len(devices) != 8:
        raise RuntimeError(f'Expected exactly eight visible devices, got {devices}')
    original_init = train.nt.init_train_state
    cached = {}

    def init_with_target_sharding(*init_args, **init_kwargs):
        state, state_sharding = original_init(*init_args, **init_kwargs)
        cached['state_sharding'] = state_sharding
        return state, state_sharding

    def restore_on_current_mesh(manager, state, data_loader, step=None):
        del data_loader
        expected_sharding = cached['state_sharding']
        with array_typing.disable_typechecking():
            train_state, params = train.checkpoints._split_params(state)
            train_sharding, params_sharding = train.checkpoints._split_params(expected_sharding)
            items = {'train_state': train_state, 'params': {'params': params}}
            restore_kwargs = {
                'train_state': {'restore_args': checkpoint_utils.construct_restore_args(train_state, train_sharding)},
                'params': {'restore_args': checkpoint_utils.construct_restore_args({'params': params}, {'params': params_sharding})},
            }
            restored_items = manager.restore(step, items=items, restore_kwargs=restore_kwargs)
            restored = train.checkpoints._merge_params(restored_items['train_state'], restored_items['params'])
        jax.block_until_ready(restored)
        expected_leaves, expected_tree = jax.tree.flatten(state)
        actual_leaves, actual_tree = jax.tree.flatten(restored)
        if expected_tree != actual_tree:
            raise RuntimeError('Restored state tree differs from the requested state tree')
        array_count = 0
        device_counts = set()
        for expected, actual in zip(expected_leaves, actual_leaves):
            if not hasattr(expected, 'shape') or not hasattr(expected, 'dtype'):
                continue
            if not isinstance(actual, jax.Array):
                raise RuntimeError(f'Expected restored JAX array, got {type(actual)}')
            if tuple(expected.shape) != tuple(actual.shape) or expected.dtype != actual.dtype:
                raise RuntimeError(f'Restored shape/dtype mismatch: {expected} versus {actual.shape}/{actual.dtype}')
            device_counts.add(len(actual.sharding.device_set))
            if actual.sharding.device_set != set(devices):
                raise RuntimeError(f'Restored leaf uses an unexpected device set: {actual.sharding}')
            array_count += 1
        if not array_count:
            raise RuntimeError('No restored arrays were verified')
        result = {
            'stage': 'topology_restored', 'time': time.time(), 'step': int(restored.step),
            'visible_devices': [str(device) for device in devices],
            'mesh': {'batch': 4, 'fsdp': 2}, 'global_batch_size': 8,
            'restored_array_count': array_count, 'array_device_counts': sorted(device_counts),
            'shape_dtype_and_device_set_verified': True, 'explicit_restore_sharding': True,
            'optimizer_restored': True, 'original_script': str(ORIGINAL),
            'original_script_sha256': hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
            'orbax_version': ocp.__version__,
        }
        target = args.topology_audit_dir / 'RESTORE_VERIFIED.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(target)
        print(json.dumps(result), flush=True)
        return restored

    train.nt.init_train_state = init_with_target_sharding
    train.checkpoints.restore_state = restore_on_current_mesh
    sys.argv = [str(ORIGINAL), *original_args]
    train.main()


if __name__ == '__main__':
    main()

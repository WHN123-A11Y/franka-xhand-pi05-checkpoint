"""Extract disjoint scene shards using the selected frozen native policy."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import types

ROOT = Path('/data/adminad/workspace/FineVLA_clean')
VIEWS = ('head_view', 'front_left_view', 'wrist_view')


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_feature(path, identity):
    import torch
    value = torch.load(path, map_location='cpu', weights_only=False)
    grid = value['grid']
    if value['selection_sha256'] != identity:
        raise ValueError('Feature checkpoint identity mismatch: ' + str(path))
    if grid.ndim != 4 or tuple(grid.shape[:3]) != (3, 8, 8) or not torch.isfinite(grid).all():
        raise ValueError('Invalid feature shape or values: ' + str(path))
    native_shape = value['native_shape']
    if len(native_shape) != 3 or native_shape[0] != 3 or native_shape[2] != grid.shape[-1]:
        raise ValueError('Native feature shape mismatch: ' + str(path))
    side = int(native_shape[1] ** .5)
    if side * side != native_shape[1]:
        raise ValueError('Non-square native patch grid: ' + str(path))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/real_franka_full_training_20261117/student')
    parser.add_argument('--policy-selection', type=Path, default=ROOT / 'runs/real_franka_full_training_20261117/policy/selected.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'runs/real_franka_full_training_20261117/data')
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid extraction shard')
    if not (args.out / 'LABELS_READY.json').is_file():
        raise FileNotFoundError('LABELS_READY.json')
    selection = json.loads(args.policy_selection.read_text())
    identity = sha(args.policy_selection)
    native_root = Path(selection['native_root'])
    checkpoint = Path(selection['checkpoint'])
    feat = args.out / 'features'
    expected_provenance = dict(selection_sha256=identity, checkpoint=str(checkpoint), native_root=str(native_root), media_ready_sha256=sha(args.data_dir / 'READY.json'))
    if json.loads((feat / 'policy_identity.json').read_text()) != expected_provenance:
        raise ValueError('Launcher feature provenance does not match selected policy/media')
    sys.path.insert(0, str(native_root / 'src'))
    sys.path.insert(0, str(native_root / 'packages/openpi-client/src'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    import torch.nn.functional as F
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import nnx_utils
    from openpi_client.image_tools import resize_with_pad

    torch.set_num_threads(2)
    if jax.device_count() != 1:
        raise ValueError('Each extraction worker must see exactly one GPU')
    status_path = args.out / 'four_gpu' / f'extraction_status_shard_{args.shard}.json'
    scenes = json.loads((args.out / 'scenes.json').read_text())
    assigned = [(index, scene) for index, scene in enumerate(scenes) if index % args.shards == args.shard]
    def status(stage, **kwargs):
        value = dict(stage=stage, time=time.time(), shard=args.shard, shards=args.shards, gpu=os.getenv('CUDA_VISIBLE_DEVICES'), total=len(assigned), **kwargs)
        atomic(status_path, value)
        print(json.dumps(value), flush=True)

    status('loading_native_policy', checkpoint=str(checkpoint), completed=0)
    policy = policy_config.create_trained_policy(config.get_config('pi05_xhand'), checkpoint)
    model = policy._model
    def vision(self, images):
        return self.PaliGemma.img(images, train=False)[0]
    def language(self, ids):
        return self.PaliGemma.llm(ids, method='embed')
    vision = nnx_utils.module_jit(types.MethodType(vision, model))
    language = nnx_utils.module_jit(types.MethodType(language, model))
    completed = 0
    for _, scene in assigned:
        destination = feat / (scene['scene'] + '.pt')
        if destination.exists():
            validate_feature(destination, identity)
            completed += 1
            status('extracting', completed=completed, resumed=True, current=scene['scene'])
            continue
        images = [np.asarray(np.load(args.data_dir / 'media' / f"episode_{scene['episode']:06d}" / (view + '.npy'), mmap_mode='r')[scene['frame']]) for view in VIEWS]
        assert all(image.shape == (168, 224, 3) for image in images)
        padded = np.stack([resize_with_pad(image, 224, 224) for image in images])
        tokens = np.asarray(vision(jnp.asarray(padded.astype(np.float32) / 255.0 * 2.0 - 1.0)), dtype=np.float32)
        side = int(tokens.shape[1] ** .5)
        assert side * side == tokens.shape[1] and tokens.shape[0] == 3
        native_shape = list(tokens.shape)
        grids = []
        for image, token in zip(images, tokens):
            height, width = image.shape[:2]
            ratio = max(width / 224, height / 224)
            resized_width, resized_height = int(width / ratio), int(height / ratio)
            px, py = (224 - resized_width) // 2, (224 - resized_height) // 2
            axis = (torch.arange(8).float() + .5) / 8
            yy, xx = torch.meshgrid(axis, axis, indexing='ij')
            coords = torch.stack([2 * (px + xx * resized_width) / 224 - 1, 2 * (py + yy * resized_height) / 224 - 1], -1)[None]
            grids.append(F.grid_sample(torch.from_numpy(token).reshape(side, side, -1).permute(2, 0, 1)[None], coords, mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0))
        grid = torch.stack(grids).half()
        assert torch.isfinite(grid).all()
        temporary = destination.with_suffix('.tmp')
        torch.save(dict(grid=grid, native_shape=native_shape, selection_sha256=identity), temporary)
        temporary.replace(destination)
        completed += 1
        status('extracting', completed=completed, resumed=False, current=scene['scene'])

    if args.shard == 0:
        status('extracting_language', completed=completed)
        samples = json.loads((args.out / 'samples.json').read_text())
        tokenizer = PaligemmaTokenizer(max_len=48)
        bank = {}
        for text in sorted({sample['expression'] for sample in samples} | {''}):
            ids, mask = tokenizer.tokenize(text)
            embedding = np.asarray(language(jnp.asarray(ids[None])), dtype=np.float32)[0]
            bank[text] = torch.from_numpy(embedding[mask].mean(0))
            assert torch.isfinite(bank[text]).all()
        for destination in (args.out / 'language.pt', feat / 'language.pt'):
            temporary = destination.with_suffix('.tmp')
            torch.save(bank, temporary)
            temporary.replace(destination)
        example = validate_feature(feat / (scenes[0]['scene'] + '.pt'), identity)
        source_model = ROOT / 'experiments/pi05_native_student_20261016/model.py'
        norm = checkpoint / 'assets/xhand_local/norm_stats.json'
        atomic(args.out / 'feature_config.json', {
            'dim': int(example['grid'].shape[-1]), 'grid_size': 8, 'native_shape': example['native_shape'],
            'native_policy_checkpoint': str(checkpoint), 'native_root': str(native_root),
            'policy_selection_sha256': identity, 'native_norm_sha256': sha(norm), 'media_ready_sha256': expected_provenance['media_ready_sha256'],
            'input_media': str(args.data_dir / 'media'),
            'input_preprocessing': 'shared FFmpeg bicubic original640x480->224x168 RGB cache; same native PIL letterbox224x224 as policy and K4; no second source resize',
            'feature_source': 'selected task-trained frozen pi05 PaliGemma image tokens',
            'language': 'selected policy frozen token embeddings, masked mean',
            'language_inputs': 'raw role expression as category; empty string as colors; no spelling/category correction',
            'coordinates': 'original RGB normalized (same aspect224x168); native padded16x16 tokens resampled at8x8 unpadded-image cell centers; bilinear align_corners=False; ypad=y*0.75+0.125',
            'student_model_source': str(source_model), 'student_model_source_sha256': sha(source_model),
            'camera_order': list(VIEWS), 'model_slots': ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'],
            'scenes_sha256': sha(args.out / 'scenes.json'), 'samples_sha256': sha(args.out / 'samples.json'),
            'extraction_parallelism': {'workers': args.shards, 'partition': 'scene index modulo worker count', 'transform_unchanged': True},
        })
    status('finished', completed=completed)


if __name__ == '__main__':
    main()

"""Shared native final-hidden K4 primitives; no robot control interface."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import types

import numpy as np
import torch


REGION_MODE = 'candidate-verification-only'
FUSION_SOURCE = 'normalized own-box candidate-verification confidence among accepted views'


def validate_partial_student(saved, ready, *, checkpoint, baseline_checkpoint):
    """Validate a genuine partial checkpoint without inventing validity training."""
    required = {'localization', 'box_verification'}
    disabled = {'view_validity', 'view_distribution'}
    if ready.get('full_student_acceptance_ready') is not False:
        raise RuntimeError('This mode requires explicit partial Student completion')
    if not required.issubset(set(saved.get('trained_heads', []))):
        raise RuntimeError('Student lacks trained localization/candidate-verification heads')
    if not required.issubset(set(ready.get('trained_heads', []))):
        raise RuntimeError('Partial completion record lacks the required trained heads')
    if not disabled.issubset(set(saved.get('disabled_heads', []))) or not disabled.issubset(set(ready.get('disabled_heads', []))):
        raise RuntimeError('Partial Student must explicitly mark validity/distribution inactive')
    if disabled.intersection(set(saved.get('trained_heads', []))):
        raise RuntimeError('Contradictory Student trained/disabled head declarations')
    counts = saved.get('supervision_counts', {})
    if counts != ready.get('supervision_counts'):
        raise RuntimeError('Checkpoint and partial completion supervision counts differ')
    for key in ('localization_positive', 'candidate_positive', 'candidate_negative'):
        if int(counts.get(key, 0)) <= 0:
            raise RuntimeError(f'Missing actual Student supervision for {key}')
    checkpoint = Path(checkpoint)
    if Path(ready.get('checkpoint', '')).resolve() != checkpoint.resolve():
        raise RuntimeError('Partial completion record points to another Student checkpoint')
    if ready.get('checkpoint_sha256') != sha256(checkpoint):
        raise RuntimeError('Student checkpoint checksum differs from partial completion')
    if Path(ready.get('native_policy_checkpoint', '')).resolve() != Path(baseline_checkpoint).resolve():
        raise RuntimeError('Partial Student and K4 baseline policy differ')
    return {'trained_heads': sorted(required), 'disabled_heads': sorted(disabled),
            'candidate_positive': int(counts['candidate_positive']),
            'candidate_negative': int(counts['candidate_negative']),
            'full_view_validity_trained': False, 'view_distribution_trained': False,
            'fusion_weight_source': FUSION_SOURCE}


def decode_candidate_only(prediction):
    """Use only the trained own-box verifier; supply logits have no influence."""
    logits = prediction['own_verification']
    boxes = prediction['boxes']
    if logits.shape[:-1] != boxes.shape[:-1] or logits.shape[-1] != 2:
        raise ValueError('Expected per-view boxes and two-class own-box verification')
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
        raise FloatingPointError('Nonfinite candidate verifier logits or boxes')
    accepted = logits.argmax(-1) == 1
    confidence = logits.softmax(-1)[..., 1]
    weights = confidence * accepted
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    return {'boxes': boxes, 'accepted': accepted, 'score': confidence,
            'candidate_confidence_weights': weights,
            'view_validity': None, 'learned_view_weights': None,
            'all_rejected': ~accepted.any(-1)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def atomic_torch(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


class Bridge(torch.nn.Module):
    """Same biased Linear + zero-initialized tanh gate as simulation K4."""
    def __init__(self, feature_dim, hidden_dim=1024):
        super().__init__()
        self.projection = torch.nn.Linear(feature_dim, hidden_dim, bias=True)
        self.gate = torch.nn.Parameter(torch.zeros(()))

    def forward(self, feature, valid):
        if feature.ndim != 2 or valid.shape != feature.shape[:1] or valid.dtype != torch.bool:
            raise ValueError('Expected features [B,D], Boolean valid [B]')
        delta = self.gate.tanh() * self.projection(feature)
        return torch.where(valid[:, None], delta, torch.zeros_like(delta))


def mesh_shardings(mesh):
    """Return replicated state/scalars and batch-sharded data for the mesh."""
    from jax.sharding import NamedSharding, PartitionSpec
    return NamedSharding(mesh, PartitionSpec()), NamedSharding(mesh, PartitionSpec('data'))


def frozen_jit(meth, mesh, in_shardings, out_shardings):
    """Freeze an NNX method with explicit state and positional argument sharding.

    ``in_shardings`` describes only method arguments; replicated state is
    prepended internally. Arguments are placed explicitly before invocation,
    so committed single-device input arrays cannot select a stale topology.
    """
    import flax.nnx as nnx
    import jax
    if not inspect.ismethod(meth) or not isinstance(meth.__self__, nnx.Module):
        raise ValueError('frozen_jit expects a bound NNX module method')
    replicated, _ = mesh_shardings(mesh)
    graphdef, state = nnx.split(meth.__self__)
    state = jax.device_put(state, replicated)

    def fun(state, *args):
        module = nnx.merge(graphdef, state)
        return meth.__func__(module, *args)

    compiled = jax.jit(fun, in_shardings=(replicated, *tuple(in_shardings)),
                       out_shardings=out_shardings)

    def wrapper(*args):
        if len(args) != len(in_shardings):
            raise ValueError(f'Expected {len(in_shardings)} positional arguments; got {len(args)}')
        placed = tuple(jax.device_put(value, sharding)
                       for value, sharding in zip(args, in_shardings))
        return compiled(state, *placed)

    wrapper.frozen_state = state
    wrapper.argument_shardings = tuple(in_shardings)
    wrapper.output_shardings = out_shardings
    wrapper.compiled = compiled
    return wrapper


def native_projection(model, mesh=None):
    """Actual frozen native head; globally masked loss over the batch axis."""
    import jax.numpy as jnp

    def projection_loss(self, hidden, target, delta, valid_time):
        v = self(hidden + delta[:, None, :].astype(hidden.dtype))
        err = jnp.square(v[..., :19].astype(jnp.float32) - target[..., :19])
        return (err * valid_time[..., None]).sum() / jnp.maximum(valid_time.sum() * 19, 1)

    method = types.MethodType(projection_loss, model.action_out_proj)
    if mesh is None:
        from openpi.shared import nnx_utils
        return nnx_utils.module_jit(method)
    replicated, data = mesh_shardings(mesh)
    return frozen_jit(method, mesh, (data, data, data, data), replicated)


def native_sampler(model, mesh):
    """Original native sampler on replicated inputs, including a one-row batch.

    This is intentionally separate from the batch-sharded capture path:
    batch1 parity/test samples must not be assigned PartitionSpec('data').
    """
    replicated, _ = mesh_shardings(mesh)
    original = type(model).sample_actions

    def sample_fixed(self, rng, observation, noise):
        return original(self, rng, observation, num_steps=10, noise=noise)

    compiled = frozen_jit(types.MethodType(sample_fixed, model), mesh,
                          (replicated, replicated, replicated), replicated)

    def sample(rng, observation, *, num_steps=10, noise=None):
        if num_steps != 10:
            raise ValueError('This fixed native parity wrapper uses the actual ten-step sampler')
        return compiled(rng, observation, noise)

    sample.frozen_state = compiled.frozen_state
    sample.compiled = compiled
    return sample


def native_functions(model, output, mesh=None):
    """Native capture/project/sampler with four-GPU data-parallel capture."""
    source = textwrap.dedent(inspect.getsource(type(model).compute_loss))
    source = source[source.index('def compute_loss'):]
    source = source.replace('def compute_loss(', 'def capture_features(', 1)
    final = 'return jnp.sum(squared_error, axis=-1) / self.action_loss_dim'
    if source.count(final) != 1:
        raise RuntimeError('Unexpected native XHand compute_loss source')
    source = source.replace(final, 'return suffix_out[:, -self.action_horizon :], u_t, prefix_tokens[:, :768]')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'native_capture_source.py').write_text(source)
    ns = dict(type(model).compute_loss.__globals__)
    exec(compile(source, '<franka_four_gpu_k4_capture>', 'exec'), ns)
    capture_method = types.MethodType(ns['capture_features'], model)
    if mesh is None:
        from openpi.shared import nnx_utils
        capture = nnx_utils.module_jit(capture_method)
    else:
        replicated, data = mesh_shardings(mesh)
        capture = frozen_jit(capture_method, mesh,
                             (replicated, data, data), (data, data, data))

    sampler = textwrap.dedent(inspect.getsource(type(model).sample_actions))
    sampler = sampler[sampler.index('def sample_actions'):]
    sampler = sampler.replace('def sample_actions(', 'def sample_delta(', 1)
    before = '    num_steps:'
    if sampler.count(before) != 1:
        raise RuntimeError('Unexpected native sampler argument list')
    sampler = sampler.replace(before, '    delta=None,\n    num_steps:', 1)
    before = '        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])'
    if sampler.count(before) != 1:
        raise RuntimeError('Unexpected native sampler projection')
    sampler = sampler.replace(before, '        hidden = suffix_out[:, -self.action_horizon :]\n'
                             '        v_t = self.action_out_proj(hidden + delta[:, None, :].astype(hidden.dtype))')
    (output / 'native_sampler_source.py').write_text(sampler)
    ns = dict(type(model).sample_actions.__globals__)
    exec(compile(sampler, '<franka_four_gpu_k4_sampler>', 'exec'), ns)
    if mesh is None:
        sample = nnx_utils.module_jit(types.MethodType(ns['sample_delta'], model))
    else:
        modified = ns['sample_delta']

        def sample_fixed(self, rng, observation, delta, noise):
            return modified(self, rng, observation, delta=delta, num_steps=10, noise=noise)

        sample_compiled = frozen_jit(types.MethodType(sample_fixed, model), mesh,
                                     (replicated, replicated, replicated, replicated), replicated)

        def sample(rng, observation, *, delta=None, num_steps=10, noise=None):
            if delta is None:
                raise ValueError('Explicit final-hidden delta is required')
            if num_steps != 10:
                raise ValueError('This fixed residual sampler uses ten native steps')
            return sample_compiled(rng, observation, delta, noise)

        sample.frozen_state = sample_compiled.frozen_state
        sample.compiled = sample_compiled

    project = native_projection(model, mesh=mesh)
    return capture, project, sample


def unpad_grid(image_tokens):
    """16x16 padded native image tokens -> 8x8 original 4:3 image coordinates."""
    import torch.nn.functional as F
    if image_tokens.ndim != 3 or image_tokens.shape[1] != 768:
        raise ValueError(f'Expected three 16x16 visual grids; got {image_tokens.shape}')
    b, _, d = image_tokens.shape
    grid = torch.from_numpy(np.asarray(image_tokens, dtype=np.float32).copy()).reshape(b * 3, 16, 16, d)
    a = (torch.arange(8).float() + .5) / 8
    yy, xx = torch.meshgrid(a, a, indexing='ij')
    coords = torch.stack([2 * xx - 1, 2 * (yy * .75 + .125) - 1], -1)
    coords = coords[None].expand(b * 3, -1, -1, -1)
    sampled = F.grid_sample(grid.permute(0, 3, 1, 2), coords, align_corners=False)
    return sampled.permute(0, 2, 3, 1).reshape(b, 3, 8, 8, d)


@torch.inference_mode()
def student_regions(student, language_bank, image_tokens, prompts, expressions, decode=None):
    """Frozen Student candidate-only decisions and raw native policy ROI values.

    Any supplied decoder is deliberately ignored: this explicit partial mode
    must never consume the untrained supply/learned-view branch.
    """
    grid = unpad_grid(image_tokens)
    b, _, _, _, d = grid.shape
    names = []
    for prompt in prompts:
        if prompt not in expressions:
            raise ValueError(f'No verbatim role expressions for {prompt!r}')
        names.extend([expressions[prompt]['object'], expressions[prompt]['destination']])
    if any(name not in language_bank for name in names) or '' not in language_bank:
        raise ValueError('Student language bank is incomplete for current instructions')
    native = torch.from_numpy(np.asarray(image_tokens, dtype=np.float32).copy()).reshape(b, 3, 16, 16, d)
    batched = grid[:, None].expand(b, 2, 3, 8, 8, d).reshape(b * 2, 3, 8, 8, d)
    text = torch.stack([language_bank[name].float() for name in names])
    empty = language_bank[''].float()[None].expand(b * 2, -1)
    pred = decode_candidate_only(student(batched, text, empty))
    boxes = pred['boxes'].reshape(b, 2, 3, 4)
    accepted = pred['accepted'].reshape(b, 2, 3)
    view_weights = pred['candidate_confidence_weights'].reshape(b, 2, 3)
    if not torch.isfinite(boxes).all() or not torch.isfinite(view_weights).all():
        raise FloatingPointError('Nonfinite Student boxes or candidate confidence weights')
    padded = boxes.clone()
    padded[..., [1, 3]] = boxes[..., [1, 3]] * .75 + .125
    axis = (torch.arange(16).float() + .5) / 16
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    low = torch.stack([xx, yy], -1) - .5 / 16
    high = low + 1 / 16
    area = (torch.minimum(padded[..., None, None, 2:], high) -
            torch.maximum(padded[..., None, None, :2], low)).clamp_min(0).prod(-1)
    # Empty/clipped ROIs have no usable visual cells. This geometric guard is
    # not a fabricated confidence label or a replacement view selector.
    student_accepted = accepted.clone()
    accepted = accepted & (area.sum((-1, -2)) > 0)
    view_weights = view_weights * accepted
    view_weights = view_weights / view_weights.sum(-1,keepdim=True).clamp_min(1e-8)
    weights = area / area.sum((-1, -2), keepdim=True).clamp_min(1e-8)
    per_view = torch.einsum('brvhw,bvhwd->brvd', weights, native)
    fused = (per_view * view_weights[..., None]).sum(2)
    # Rejected roles keep exactly zero vectors, and a completely unsupported
    # instruction bypasses the bridge (including its learned projection bias).
    valid_roles = accepted.any(-1)
    fused = torch.where(valid_roles[..., None], fused, torch.zeros_like(fused))
    feature = fused.flatten(1)
    valid = valid_roles.any(-1)
    return feature, valid, {'boxes': boxes.numpy(), 'accepted': accepted.numpy(),
                            'student_accepted': student_accepted.numpy(),
                            'candidate_confidence_weights': view_weights.numpy(),
                            'valid_roles': valid_roles.numpy(),
                            'region_mode': REGION_MODE,
                            'fusion_weight_source': FUSION_SOURCE,
                            'view_validity_trained': False,
                            'view_distribution_trained': False}

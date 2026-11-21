"""Offline Franka π0.5 + partial Student + K4 replay; never controls hardware."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import torch

from common import Bridge, native_functions, student_regions, sha256
from student_model import Student


def args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native-source', 'checkpoint', 'student', 'language', 'expressions', 'k4', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--sample', type=Path, help='portable NPZ saved by this program')
    g.add_argument('--data-root', type=Path, help='prepared dataset root')
    p.add_argument('--split', default='test', choices=['train','val','test'])
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--gpu', default='4', help='physical GPU index, set before JAX import')
    p.add_argument('--tokenizer-cache', type=Path, help='directory containing big_vision/paligemma_tokenizer.model')
    p.add_argument('--reference', type=Path, help='previous sampler_check npz, for an honest numerical comparison')
    return p.parse_args()


def main():
    a = args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    if a.tokenizer_cache: os.environ['OPENPI_DATA_HOME'] = str(a.tokenizer_cache.resolve())
    sys.path[:0] = [str(a.native_source/'src'), str(a.native_source/'packages/openpi-client/src')]
    import jax
    import jax.numpy as jnp
    from openpi.models import model as mm
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.shared import nnx_utils
    a.out.mkdir(parents=True, exist_ok=True)
    if a.data_root:
        from data import FullDataset
        raw = FullDataset(a.data_root, a.split, horizon=10)[a.index]
        image, mask, state, prompt = raw['image'], raw['image_mask'], raw['state'], str(raw['prompt'])
        episode, frame = int(raw['episode']), int(raw['frame'])
        noise = np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(900913), a.index),(1,10,32)))[0]
    else:
        with np.load(a.sample,allow_pickle=False) as z:
            names = ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')
            image = {k: z['image_'+k].copy() for k in names}
            mask = {k: z['mask_'+k].copy() for k in names}
            state, prompt, noise = z['state'].copy(), str(z['prompt']), z['noise'].copy()
            episode, frame = int(z['episode']), int(z['frame'])
    if np.asarray(state).shape != (19,) or np.asarray(noise).shape != (10,32):
        raise ValueError('Expected state [19] and stored noise [10,32]')
    expression_obj=json.loads(a.expressions.read_text())
    expressions=expression_obj.get('expressions', expression_obj)
    if prompt not in expressions: raise ValueError('Instruction is absent from exact-match expression table')
    saved=torch.load(a.student,map_location='cpu',weights_only=False)
    if not {'localization','box_verification'}.issubset(saved.get('trained_heads',[])) or not {'view_validity','view_distribution'}.issubset(saved.get('disabled_heads',[])):
        raise ValueError('Expected partially supervised Student checkpoint with disabled view heads')
    student=Student(**saved['config']).eval(); student.load_state_dict(saved['state_dict'],strict=True)
    language=torch.load(a.language,map_location='cpu',weights_only=False)
    packet=torch.load(a.k4,map_location='cpu',weights_only=False)
    identity=packet.get('identity',{})
    for name,path,key in (('Student',a.student,'student_sha256'),('language bank',a.language,'language_sha256')):
        if identity.get(key) != sha256(path): raise ValueError(f'{name} does not match the selected K4 training identity')
    if packet.get('student_region_mode') != 'candidate-verification-only':
        raise ValueError('K4 checkpoint does not declare candidate-verification-only mode')
    bridge=Bridge(**packet['bridge_config']).eval(); bridge.load_state_dict(packet['bridge'],strict=True)
    policy=policy_config.create_trained_policy(config.get_config('pi05_xhand'),a.checkpoint)
    model=policy._model
    if (model.action_dim,model.action_horizon,model.action_loss_dim)!=(32,10,19): raise ValueError('Unexpected policy architecture')
    inp=policy._input_transform({'image':image,'image_mask':mask,'state':state,'prompt':prompt})
    inp={k:inp[k] for k in ('image','image_mask','state','tokenized_prompt','tokenized_prompt_mask')}
    def fingerprint(value):
        arr=np.ascontiguousarray(value)
        return {'sha256':hashlib.sha256(arr.tobytes()).hexdigest(),'dtype':str(arr.dtype),'shape':list(arr.shape)}
    input_checks={'state':fingerprint(inp['state']),'tokenized_prompt':fingerprint(inp['tokenized_prompt']),
                  'tokenized_prompt_mask':fingerprint(inp['tokenized_prompt_mask']),
                  'images':{k:fingerprint(v) for k,v in inp['image'].items()},
                  'masks':{k:fingerprint(v) for k,v in inp['image_mask'].items()},'noise':fingerprint(noise)}
    obs=mm.Observation.from_dict(jax.tree.map(lambda x:np.expand_dims(x,0),inp))
    # Exactly the prefix_tokens[:768] read by training's compute_loss, with no future action input.
    def visual(self, observation):
        observation=mm.preprocess_observation(None,observation,train=False)
        return self.embed_prefix(observation)[0][:,:768]
    get_visual=nnx_utils.module_jit(types.MethodType(visual,model))
    tokens=np.asarray(get_visual(obs),dtype=np.float32)
    region,valid,grounding=student_regions(student,language,tokens,[prompt],expressions)
    with torch.no_grad(): delta=bridge(region,valid).numpy()
    _,_,sampler=native_functions(model,a.out/'generated_native_sources')
    rng=jax.random.key(0); noise=jnp.asarray(noise[None])
    base=np.asarray(policy._sample_actions(rng,obs,noise=noise),np.float32)
    zero=np.asarray(sampler(rng,obs,delta=jnp.zeros_like(jnp.asarray(delta)),noise=noise),np.float32)
    adapted=np.asarray(sampler(rng,obs,delta=jnp.asarray(delta),noise=noise),np.float32)
    def inverse(x):
        return np.asarray(policy._output_transform({'state':np.asarray(obs.state)[0],'actions':x[0].copy()})['actions'],np.float32)[:,:19]
    outputs={'native':base[0],'zero':zero[0],'k4':adapted[0],
             'native_recorded_units19':inverse(base),'zero_recorded_units19':inverse(zero),'k4_recorded_units19':inverse(adapted),
             'delta_hidden':delta[0],'noise':np.asarray(noise)[0], 'student_boxes':grounding['boxes'][0],
             'accepted_views':grounding['accepted'][0], 'candidate_confidence_weights':grounding['candidate_confidence_weights'][0]}
    np.savez_compressed(a.out/'outputs.npz',**outputs)
    np.savez_compressed(a.out/'input_sample.npz',**{'image_'+k:v for k,v in image.items()},**{'mask_'+k:v for k,v in mask.items()},state=state,prompt=np.array(prompt),episode=episode,frame=frame,noise=np.asarray(noise)[0])
    comparison={}
    if a.reference:
        with np.load(a.reference,allow_pickle=False) as ref:
            for key in outputs.keys() & set(ref.files):
                if outputs[key].dtype.kind in 'bif' and outputs[key].shape == ref[key].shape:
                    comparison[key]=float(np.max(np.abs(outputs[key].astype(np.float64)-ref[key].astype(np.float64))))
    result={'episode':episode,'frame':frame,'prompt':prompt,'visible_gpu':a.gpu,'jax_devices':list(map(str,jax.devices())),
            'student_sha256':sha256(a.student),'k4_sha256':sha256(a.k4),'k4_step':packet['step'],
            'native_zero_max_abs':float(np.max(np.abs(base-zero))),'reference_max_abs':comparison,
            'region_valid':bool(valid[0]),'roles':['object','destination'], 'view_validity_trained':False,'transformed_input_checks':input_checks,
            'scope':'Offline recorded sample and model predictions only; no robot controller, no closed-loop test.'}
    (a.out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)

if __name__ == '__main__': main()

"""Full-frame online native-flow K4 training with frozen policy and Student.

Every epoch visits all TRAIN frames. The frozen policy is evaluated online
with fresh native beta-distributed time and normal noise at every update.
Only the final-hidden residual bridge is optimized. No hidden-state cache is
read and no robot command interface is imported.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path('/data/adminad/workspace/FineVLA_clean')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (Bridge, atomic_json, atomic_torch, native_functions, sha256,
                    student_regions, validate_partial_student, REGION_MODE, FUSION_SOURCE, native_sampler)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--baseline-checkpoint', type=Path, required=True)
    p.add_argument('--policy-ready', type=Path, required=True)
    p.add_argument('--student-checkpoint', type=Path, required=True)
    p.add_argument('--student-ready', type=Path, required=True, help='Actual PARTIAL_READY.json; no full Student READY is fabricated')
    p.add_argument('--student-region-mode', required=True, choices=[REGION_MODE])
    p.add_argument('--student-language-bank', type=Path, required=True)
    p.add_argument('--expressions', type=Path, required=True)
    p.add_argument('--student-feature-config', type=Path, required=True)
    p.add_argument('--student-model-source', type=Path, default=ROOT/'experiments/pi05_native_student_20261016/model.py')
    p.add_argument('--data-source', type=Path, default=ROOT/'experiments/real_franka_full_training_v1/policy/data.py')
    p.add_argument('--native-source', type=Path, default=Path('/data/fangchuan/workspace/openpi_pi05_xhand_20260923'))
    p.add_argument('--out', type=Path, default=ROOT/'runs/real_franka_full_training_20261117/k4')
    p.add_argument('--gpus', default='4,5,6,7', help='Four physical GPUs used for frozen-policy data parallelism')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--validation-draws', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-interval', type=int, default=100)
    p.add_argument('--log-interval', type=int, default=25)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-only', action='store_true')
    p.add_argument('--sampler-cases-per-episode', type=int, default=1)
    return p.parse_args()


def main():
    a = arguments()
    if min(a.epochs, a.batch_size, a.validation_draws, a.save_interval) < 1:
        raise ValueError('Positive epoch/batch/draw/save counts required')
    a.out.mkdir(parents=True, exist_ok=True)
    owner = (a.out/'trainer.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner.seek(0); owner.truncate(); owner.write(str(os.getpid())); owner.flush()
    for p in (a.policy_ready, a.student_ready, a.student_checkpoint,
              a.student_language_bank, a.expressions, a.student_feature_config):
        if not p.is_file():
            raise FileNotFoundError(f'Full training prerequisite missing: {p}')
    if not (a.baseline_checkpoint/'params').is_dir():
        raise FileNotFoundError('Expected a native policy export with params/ and assets/')
    policy_ready=json.loads(a.policy_ready.read_text())
    if not policy_ready.get('full_model_finetune') or 'smoke' in str(policy_ready.get('scope','')).lower():
        raise RuntimeError('K4 requires the completed full current-task policy stage, not a smoke/pilot checkpoint')
    if Path(policy_ready['checkpoint']).resolve()!=a.baseline_checkpoint.resolve():
        raise RuntimeError('Policy READY points to another selected checkpoint')
    native_norm=a.baseline_checkpoint/'assets/xhand_local/norm_stats.json'
    train_norm=a.data_root/'assets/xhand_local/norm_stats.json'
    if sha256(native_norm)!=sha256(train_norm):
        raise RuntimeError('Selected policy does not use shared TRAIN-only state/action normalization')
    if (a.out/'last.pt').exists() and not a.resume:
        raise RuntimeError('Existing full K4 run: use --resume; no overwrite')

    if a.gpus != '4,5,6,7':
        raise ValueError('This attempt is assigned exactly GPUs4,5,6,7')
    if a.batch_size % 4:
        raise ValueError('Global batch must be divisible by four')
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpus
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','0.8')

    sys.path.insert(0, str(a.native_source/'src'))
    sys.path.insert(0, str(a.native_source/'packages/openpi-client/src'))
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import numpy as np
    import torch
    from openpi.models import model as mm
    from openpi.policies import policy_config
    from openpi.training import config
    torch.set_num_threads(4)
    devices=jax.devices()
    if len(devices)!=4:raise RuntimeError(f'Expected four devices, got {devices}')
    mesh=jax.make_mesh((4,),('data',))
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    data_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec('data'))
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    def status(phase, **kw):
        obj = dict(phase=phase, time=time.time(), pid=os.getpid(), **kw)
        atomic_json(a.out/'status.json', obj)
        print(json.dumps(obj, ensure_ascii=False), flush=True)

    status('checking_student_supervision')
    student_saved = torch.load(a.student_checkpoint, map_location='cpu', weights_only=False)
    ready = json.loads(a.student_ready.read_text())
    # This explicitly selected partial mode uses only the heads that actually
    # received supervision. Rejected proposals never become absence labels.
    partial_supervision = validate_partial_student(
        student_saved, ready, checkpoint=a.student_checkpoint,
        baseline_checkpoint=a.baseline_checkpoint)
    atomic_json(a.out/'student_supervision_check.json', {
        'student_region_mode': a.student_region_mode, **partial_supervision,
        'partial_ready_path': str(a.student_ready),
        'student_sha256': sha256(a.student_checkpoint),
        'new_teacher_calls': False})
    feature_cfg = json.loads(a.student_feature_config.read_text())
    # Explicit path check is conservative: a Student trained on a historical
    # or RoboTwin encoder is incompatible with these newly fine-tuned tokens.
    feature_checkpoint = feature_cfg.get('native_policy_checkpoint',feature_cfg.get('checkpoint', feature_cfg.get('baseline_checkpoint')))
    if feature_checkpoint is None or Path(feature_checkpoint).resolve() != a.baseline_checkpoint.resolve():
        raise RuntimeError('Student native feature checkpoint differs from the selected current-task baseline')
    selection_path=a.baseline_checkpoint.parent/'selected.json'
    if feature_cfg.get('policy_selection_sha256'):
        if not selection_path.is_file() or feature_cfg['policy_selection_sha256']!=sha256(selection_path):
            raise RuntimeError('Student policy selection hash differs from baseline selected.json')
    if feature_cfg.get('media_ready_sha256'):
        media_ready=a.data_root/'READY.json'
        if not media_ready.is_file() or feature_cfg['media_ready_sha256']!=sha256(media_ready):
            raise RuntimeError('Student and K4 use different shared decoded media identities')
    if feature_cfg.get('student_model_source_sha256') != sha256(a.student_model_source):
        raise RuntimeError('Student model source differs from its feature/training configuration')
    student_module = load_module('full_franka_k4_student', a.student_model_source)
    student = student_module.Student(**student_saved['config']).eval()
    student.load_state_dict(student_saved['state_dict'], strict=True)
    student.requires_grad_(False)
    language = torch.load(a.student_language_bank, map_location='cpu', weights_only=False)
    expressions = json.loads(a.expressions.read_text())
    if 'expressions' in expressions: expressions = expressions['expressions']
    dataset_module = load_module('full_franka_k4_data', a.data_source)
    datasets = {sp: dataset_module.FullDataset(a.data_root, sp, horizon=10) for sp in ('train','val','test')}
    if not all(len(ds) for ds in datasets.values()): raise RuntimeError('Empty split')
    status('loading_native_policy', samples={sp:len(ds) for sp,ds in datasets.items()})
    policy = policy_config.create_trained_policy(config.get_config('pi05_xhand'), a.baseline_checkpoint)
    model = policy._model
    replicated_state=jax.device_put(nnx.state(model),replicated)
    jax.block_until_ready(replicated_state)
    nnx.update(model,replicated_state)
    state_arrays=[x for x in jax.tree.leaves(replicated_state) if isinstance(x,jax.Array)]
    if not state_arrays or any(x.sharding.device_set!=set(devices) for x in state_arrays):
        raise RuntimeError('Frozen policy parameters were not replicated onto all four GPUs')
    atomic_json(a.out/'parallelism.json',{'gpus':a.gpus,'mesh':{'data':4},'global_batch_size':a.batch_size,
        'local_batch_size':a.batch_size//4,'replicated_parameter_arrays':len(state_arrays),
        'replicated_parameter_device_counts':sorted({len(x.sharding.device_set) for x in state_arrays}),
        'trainable_module':'one CPU residual bridge; global native loss and delta gradient',
        'sampler_batch_one':'replicated inputs and outputs, preserving native batch1 computation',
        'scope':'actual full selected policy parameters replicated before native forward compilation'})
    if (model.action_dim, model.action_horizon, model.action_loss_dim) != (32,10,19):
        raise ValueError('Full K4 requires native Franka 19D supervision and horizon10')
    capture, project, sampler = native_functions(model, a.out/'sources',mesh=mesh)
    policy._sample_actions=native_sampler(model,mesh)
    value_gradient = jax.jit(jax.value_and_grad(project, argnums=2),
        in_shardings=(data_sharding,data_sharding,data_sharding,data_sharding),
        out_shardings=(replicated,data_sharding))
    dim = int(student_saved['config']['dim'])
    bridge = Bridge(2*dim,1024)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=a.lr, weight_decay=1e-4)
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('resume','smoke_only')}
    identity = {'settings': settings, 'student_sha256':sha256(a.student_checkpoint),
                'language_sha256':sha256(a.student_language_bank),
                'feature_config_sha256':sha256(a.student_feature_config),
                'policy_ready_sha256':sha256(a.policy_ready),
                'policy_selection_sha256':sha256(selection_path),
                'student_ready_sha256':sha256(a.student_ready),
                'scripts':{p.name:sha256(p) for p in (Path(__file__),HERE/'common.py',a.data_source)}}
    atomic_json(a.out/'protocol.json', {'identity':identity,
        'samples':{sp:len(ds) for sp,ds in datasets.items()},
        'supervised_action_dims':19,'padded_dims_ignored':13,'horizon':10,
        'intervention':'biased Linear + tanh zero gate into final action hidden before frozen native action_out_proj',
        'training':'all TRAIN frames once per epoch, fresh native random noise and beta time per update; no H cache',
        'images':'no augmentation, preserving exact Student ROI/image coordinates; native input resize+pad',
        'region_normalization':'none; raw native frozen policy ROI features, as simulation K4 bridge',
        'selection':'validation only, three fixed noise/time draws, includes zero residual baseline',
        'parallelism':{'gpus':a.gpus,'data_parallel_devices':4,'global_batch':a.batch_size,'local_batch':a.batch_size//4},
        'view_fusion':FUSION_SOURCE,
        'student_region_mode':a.student_region_mode,
        'view_validity_trained':False,'view_distribution_trained':False,
        'student_supervision':partial_supervision,
        'scope':'offline full training; physical closed-loop success not measured'})

    def raw_batch(split, ids):
        records = [datasets[split][int(i)] for i in ids]
        actual = len(records)
        records.extend([records[-1]]*(a.batch_size-actual))
        transformed=[];prompts=[];valid_time=[]
        for i,r in enumerate(records):
            inp = policy._input_transform({k:r[k] for k in ('image','image_mask','state','actions','prompt')})
            inp = {k:inp[k] for k in ('image','image_mask','state','actions','tokenized_prompt','tokenized_prompt_mask')}
            transformed.append(inp); prompts.append(str(r['prompt']))
            valid_time.append(~np.asarray(r['action_is_pad'],bool) if i<actual else np.zeros(10,bool))
        batch = jax.tree.map(lambda *xs:np.stack(xs), *transformed)
        actions = batch.pop('actions')
        observation = mm.Observation.from_dict(batch)
        observation=jax.device_put(observation,data_sharding)
        actions=jax.device_put(actions,data_sharding)
        valid_mask=jax.device_put(np.stack(valid_time),data_sharding)
        return observation,actions,valid_mask,prompts,records[:actual]

    def feature_batch(split, ids, key):
        obs,target,mask,prompts,records=raw_batch(split,ids)
        hidden,velocity,visual=capture(key,obs,target)
        if not (a.out/'DATA_PARALLEL_VERIFIED.json').exists():
            jax.block_until_ready((hidden,velocity,visual))
            arrays={'hidden':hidden,'velocity':velocity,'visual':visual}
            checks={}
            for name,array in arrays.items():
                if array.sharding.device_set!=set(devices) or array.sharding.spec!=jax.sharding.PartitionSpec('data'):
                    raise RuntimeError(f'Native {name} not data-sharded across four GPUs: {array.sharding}')
                checks[name]={'global_shape':list(array.shape),'shard_shapes':[list(shard.data.shape) for shard in array.addressable_shards],
                    'devices':[str(shard.device) for shard in array.addressable_shards],'partition':str(array.sharding.spec)}
            atomic_json(a.out/'DATA_PARALLEL_VERIFIED.json',{'time':time.time(),'gpus':a.gpus,'checks':checks,'source':'real recorded native-policy forward'})
        region,valid,grounding=student_regions(student,language,np.asarray(visual,dtype=np.float32),prompts,expressions,None)
        return obs,target,mask,records,hidden,velocity,region.clone(),valid.clone(),grounding

    def evaluate(split, module, seed=101):
        sums={'baseline':0.,'k4':0.};count=0;accepted=0;frames=0
        for start in range(0,len(datasets[split]),a.batch_size):
            ids=list(range(start,min(start+a.batch_size,len(datasets[split]))))
            for draw in range(a.validation_draws):
                key=jax.random.fold_in(jax.random.fold_in(jax.random.key(seed),start),draw)
                _,_,mask,records,h,u,region,valid,_=feature_batch(split,ids,key)
                with torch.no_grad():delta=module(region,valid).numpy()
                scalar_count=int(np.asarray(mask).sum())*19
                b=float(project(h,u,jnp.zeros_like(jnp.asarray(delta)),mask))
                k=float(project(h,u,jnp.asarray(delta),mask))
                sums['baseline']+=b*scalar_count;sums['k4']+=k*scalar_count;count+=scalar_count
                if draw==0:accepted+=int(valid[:len(ids)].sum());frames+=len(ids)
            if (start//a.batch_size)%25==0:
                status('validation' if split=='val' else 'test_after_selection',
                       split=split,completed_frames=start+len(ids),total_frames=len(datasets[split]),
                       fixed_noise_time_draws=a.validation_draws,accepted_region_frames=accepted)
        return {'baseline_flow_mse19':sums['baseline']/count,'k4_flow_mse19':sums['k4']/count,
                'valid_scalar_count':count,'frames':frames,'accepted_region_frames':accepted,
                'fixed_noise_time_draws':a.validation_draws,'noise_root_seed':seed}

    # Real source data, native sampler, actual trained Student outputs.
    first_ids=list(range(min(a.batch_size,len(datasets['train']))))
    diagnostic_train_ids=first_ids
    obs,target,mask,records,h,u,region,valid,grounding=feature_batch('train',first_ids,jax.random.key(991))
    single=jax.tree.map(lambda x:x[:1],obs)
    noise=jax.random.normal(jax.random.key(712),(1,10,32))
    native=policy._sample_actions(jax.random.key(71),single,noise=noise)
    zero=sampler(jax.random.key(71),single,delta=jnp.zeros((1,1024)),noise=noise)
    sampler_error=float(jnp.abs(native-zero).max())
    if sampler_error != 0:raise AssertionError(f'Native zero sampler parity {sampler_error}')
    # Find a genuine supported TRAIN batch for the gradient diagnostic. Never
    # force acceptance merely to make an otherwise untrainable bridge pass.
    if not bool(valid.any()):
        for offset in range(a.batch_size,min(len(datasets['train']),512),a.batch_size):
            ids=list(range(offset,min(offset+a.batch_size,len(datasets['train']))))
            _,_,mask,_,h,u,region,valid,grounding=feature_batch('train',ids,jax.random.fold_in(jax.random.key(992),offset))
            if bool(valid.any()):
                diagnostic_train_ids=ids
                break
        if not bool(valid.any()):
            raise RuntimeError('Student accepts no ROI in first512 TRAIN frames; inspect trained localization/candidate verifier')
    zero_delta=bridge(region,valid)
    base_loss=float(project(h,u,jnp.zeros_like(jnp.asarray(zero_delta.detach().numpy())),mask))
    delta_array=jax.device_put(zero_delta.detach().numpy(),data_sharding)
    direct_zero_loss=float(project(h,u,delta_array,mask))
    if direct_zero_loss!=base_loss:raise AssertionError('Zero bridge direct native loss parity')
    loss,gradient=value_gradient(h,u,delta_array,mask)
    value_gradient_loss_difference=abs(float(loss)-base_loss)
    if not np.isclose(float(loss),base_loss,rtol=1e-6,atol=1e-8):
        raise AssertionError(f'Native differentiated projection loss mismatch {value_gradient_loss_difference}')
    zero_delta.backward(torch.from_numpy(np.array(gradient,dtype=np.float32)))
    gate_gradient=float(bridge.gate.grad.abs()) if bridge.gate.grad is not None else 0.
    if not math.isfinite(gate_gradient) or gate_gradient<=0:
        raise RuntimeError('No finite nonzero initial gate gradient on genuine accepted TRAIN regions')
    bridge.zero_grad(set_to_none=True)
    invalid_delta=bridge(region,torch.zeros_like(valid))
    if not torch.equal(invalid_delta,torch.zeros_like(invalid_delta)):raise AssertionError('Invalid bypass')
    smoke={'native_zero_sampler_max_abs':sampler_error,'native_zero_flow_loss_equal':direct_zero_loss==base_loss,
           'native_value_gradient_loss_difference':value_gradient_loss_difference,
           'native_value_gradient_loss_tolerance':{'rtol':1e-6,'atol':1e-8},
           'invalid_region_exact_zero':True,'initial_gate_gradient_abs':gate_gradient,
           'accepted_initial_train_frames':int(valid.sum()),'region_dim':region.shape[-1],
           'native_hidden_shape':list(h.shape),'native_visual_model':str(a.baseline_checkpoint),
           'data_parallel_devices':4,'gpus':a.gpus,
           'diagnostic_train_indices':list(map(int,diagnostic_train_ids)),
           'student_region_mode':a.student_region_mode,
           'fusion_weight_source':FUSION_SOURCE,
           'source':'actual recorded TRAIN batch; no synthetic tensors or forced acceptances'}
    atomic_json(a.out/'smoke_checks.json',smoke)
    atomic_json(a.out/'smoke_grounding.json',{k:v.tolist() if hasattr(v,'tolist') else v for k,v in grounding.items()})
    np.savez_compressed(a.out/'smoke_native_actions.npz',native=np.asarray(native),zero=np.asarray(zero),noise=np.asarray(noise))
    if a.smoke_only:
        status('smoke_finished',**smoke);return

    steps_per_epoch=math.ceil(len(datasets['train'])/a.batch_size)
    epoch=0;batch_index=0;step=0;optimizer_updates=0;best_step=0;best_score=None
    history=[]
    last_path=a.out/'last.pt';best_path=a.out/'best.pt'
    def packet():
        return {'identity':identity,'bridge':bridge.state_dict(),'bridge_config':{'feature_dim':2*dim,'hidden_dim':1024},
                'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                'epoch':epoch,'batch_index':batch_index,'step':step,'optimizer_updates':optimizer_updates,
                'best_step':best_step,'best_validation':best_score,
                'student_region_mode':a.student_region_mode,
                'view_validity_trained':False,'view_distribution_trained':False,
                'fusion_weight_source':FUSION_SOURCE,
                'scope':'candidate-only partial Student; full-frame online native-flow K4; physical execution not performed'}
    if a.resume:
        saved=torch.load(last_path,map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('Resume identity differs from frozen training inputs')
        bridge.load_state_dict(saved['bridge']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        epoch,batch_index,step,optimizer_updates=[saved[k] for k in ('epoch','batch_index','step','optimizer_updates')]
        best=torch.load(best_path,map_location='cpu',weights_only=False)
        if best['identity']!=identity or best['step']>step:raise RuntimeError('Best checkpoint incompatible with last commit')
        best_step,best_score=best['step'],best['validation']['k4_flow_mse19']
        history=json.loads((a.out/'history.json').read_text())
    else:
        status('baseline_validation',total_frames=len(datasets['val']))
        baseline_validation=evaluate('val',bridge)
        best_score=baseline_validation['baseline_flow_mse19']
        if abs(baseline_validation['k4_flow_mse19']-best_score)>1e-8:raise AssertionError('Initial bridge not baseline')
        obj=packet();obj['validation']=baseline_validation;atomic_torch(last_path,obj);atomic_torch(best_path,obj)
        history=[{'epoch':0,'step':0,'validation':baseline_validation,'zero_residual_candidate':True}]
        atomic_json(a.out/'history.json',history)

    status('training',step=step,epoch=epoch,epochs=a.epochs,total_updates=steps_per_epoch*a.epochs)
    while epoch<a.epochs:
        order=np.random.default_rng(a.seed+1000003*(epoch+1)).permutation(len(datasets['train']))
        while batch_index<steps_per_epoch:
            ids=order[batch_index*a.batch_size:(batch_index+1)*a.batch_size].tolist()
            key=jax.random.fold_in(jax.random.key(314159+a.seed),step)
            _,_,mask,records,h,u,region,valid,_=feature_batch('train',ids,key)
            bridge.train();optimizer.zero_grad(set_to_none=True)
            delta=bridge(region,valid)
            loss,gradient=value_gradient(h,u,jnp.asarray(delta.detach().numpy()),mask)
            value=float(loss)
            if not math.isfinite(value):raise FloatingPointError('Nonfinite native flow loss')
            if bool(valid[:len(ids)].any()):
                delta.backward(torch.from_numpy(np.array(gradient,dtype=np.float32)))
                torch.nn.utils.clip_grad_norm_(bridge.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();optimizer_updates+=1
            step+=1;batch_index+=1
            if step%a.log_interval==0:
                status('training',step=step,total_updates=steps_per_epoch*a.epochs,epoch=epoch+1,batch_index=batch_index,
                       epoch_batches=steps_per_epoch,loss=value,accepted_region_frames=int(valid[:len(ids)].sum()),
                       gate=float(bridge.gate.detach().tanh()),optimizer_updates=optimizer_updates)
            if step%a.save_interval==0:atomic_torch(last_path,packet())
        # Commit the completed update state before choosing a new checkpoint.
        atomic_torch(last_path,packet())
        status('validation',epoch=epoch+1,step=step)
        bridge.eval();validation=evaluate('val',bridge)
        if validation['k4_flow_mse19']<best_score:
            best_score=validation['k4_flow_mse19'];best_step=step
            obj=packet();obj['validation']=validation;atomic_torch(best_path,obj)
        history.append({'epoch':epoch+1,'step':step,'validation':validation})
        atomic_json(a.out/'history.json',history)
        epoch+=1;batch_index=0;atomic_torch(last_path,packet())
        status('epoch_complete',epoch=epoch,step=step,best_step=best_step,validation=validation)
    if optimizer_updates==0:raise RuntimeError('No training ROI accepted: adapter received no updates')
    selected=torch.load(best_path,map_location='cpu',weights_only=False)
    bridge.load_state_dict(selected['bridge']);bridge.eval()
    status('test_after_selection',step=step,selected_step=selected['step'])
    test=evaluate('test',bridge,seed=900101)
    summary={'scope':'full-frame online native-flow training; physical execution not performed',
             'epochs':a.epochs,'optimizer_updates':optimizer_updates,'steps':step,
             'selected_step':selected['step'],'zero_residual_selected':selected['step']==0,
             'validation':selected['validation'],'test':test,'test_used_for_selection':False,
             'student_checkpoint':str(a.student_checkpoint),'baseline_checkpoint':str(a.baseline_checkpoint),
             'student_region_mode':a.student_region_mode,
             'view_validity_trained':False,'view_distribution_trained':False,
             'fusion_weight_source':FUSION_SOURCE,
             'smoke_checks':smoke}
    atomic_json(a.out/'summary.json',summary)
    sampler_results=[]
    # One predeclared first valid frame per held-out episode by default. These
    # samples are never used to select the K4 checkpoint or tune a hyperparameter.
    ep_counts={}
    (a.out/'sampler_check').mkdir(exist_ok=True)
    for i in range(len(datasets['test'])):
        raw=datasets['test'][i];ep=int(raw['episode'])
        if ep_counts.get(ep,0)>=a.sampler_cases_per_episode or np.asarray(raw['action_is_pad']).any():continue
        obs,target,mask,records,_,_,region,valid,grounding=feature_batch('test',[i],jax.random.fold_in(jax.random.key(912),i))
        single=jax.tree.map(lambda x:x[:1],obs)
        with torch.no_grad():delta=bridge(region,valid).numpy()[:1]
        key=jax.random.fold_in(jax.random.key(900913),i);noise=jax.random.normal(key,(1,10,32))
        baseline=np.asarray(policy._sample_actions(key,single,noise=noise),np.float32)
        zero=np.asarray(sampler(key,single,delta=jnp.zeros((1,1024)),noise=noise),np.float32)
        adapted=np.asarray(sampler(key,single,delta=jnp.asarray(delta),noise=noise),np.float32)
        parity=float(np.abs(baseline-zero).max())
        if parity!=0.:raise AssertionError(f'Native sampler changed on test frame {ep}/{raw["frame"]}: {parity}')
        target=np.asarray(target)[0,:,:19]
        recorded={}
        for label,actions in (('native',baseline),('zero',zero),('k4',adapted)):
            output=policy._output_transform({'state':np.asarray(single.state)[0],
                                             'actions':actions[0].copy()})
            recorded[label]=np.asarray(output['actions'],np.float32)[:,:19]
            if not np.isfinite(recorded[label]).all():raise FloatingPointError('Nonfinite inverse-normalized sampled actions')
        result={'episode':ep,'frame':int(raw['frame']),'native_zero_max_abs':parity,
                'baseline_normalized_mae19':float(np.abs(baseline[0,:,:19]-target).mean()),
                'k4_normalized_mae19':float(np.abs(adapted[0,:,:19]-target).mean()),
                'baseline_recorded_units_mae19':float(np.abs(recorded['native']-raw['actions']).mean()),
                'k4_recorded_units_mae19':float(np.abs(recorded['k4']-raw['actions']).mean()),
                'accepted_region':bool(valid[0]),'sampling_noise_seed':[900913,i]}
        np.savez_compressed(a.out/'sampler_check'/f'ep{ep:06d}_f{int(raw["frame"]):06d}.npz',
                            native=baseline[0],zero=zero[0],k4=adapted[0],target_normalized19=target,
                            native_recorded_units19=recorded['native'],zero_recorded_units19=recorded['zero'],
                            k4_recorded_units19=recorded['k4'],
                            target_recorded19=raw['actions'],state_recorded19=raw['state'],
                            delta_hidden=delta[0],noise=np.asarray(noise)[0],
                            student_boxes=grounding['boxes'][0],accepted_views=grounding['accepted'][0],
                            candidate_confidence_weights=grounding['candidate_confidence_weights'][0],
                            student_region_mode=np.array(a.student_region_mode),
                            instruction=np.array(str(raw['prompt'])))
        sampler_results.append(result);ep_counts[ep]=ep_counts.get(ep,0)+1
        status('sampler_check',completed_cases=len(sampler_results),episode=ep,
               frame=int(raw['frame']),native_zero_max_abs=parity,
               baseline_normalized_mae19=result['baseline_normalized_mae19'],
               k4_normalized_mae19=result['k4_normalized_mae19'])
    if len(ep_counts)!=18:raise RuntimeError(f'Expected sampled check from 18 held-out episodes, got {len(ep_counts)}')
    atomic_json(a.out/'sampler_check/summary.json',{'cases':sampler_results,'scope':'offline predicted chunks only; no physical execution'})
    atomic_json(a.out/'READY.json',{'selected_checkpoint':str(best_path),'selected_sha256':sha256(best_path),
                                 'selected_step':selected['step'],'steps':step,'epochs':a.epochs,
                                 'ready_for_adapted_evaluation':selected['step']>0,
                                 'student_region_mode':a.student_region_mode,
                                 'view_validity_trained':False,'view_distribution_trained':False,
                                 'fusion_weight_source':FUSION_SOURCE,
                                 'scope':'full training plus offline native sampler checks; not physical deployment approval',
                                 'summary':str(a.out/'summary.json'),'sampler_summary':str(a.out/'sampler_check/summary.json')})
    status('finished',**summary)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        # Keep traceback/exit code visible for the orchestrator. An old status
        # claiming "training" is not left behind after a process failure.
        import traceback
        parser_args=arguments()
        parser_args.out.mkdir(parents=True,exist_ok=True)
        atomic_json(parser_args.out/'status.json',{'phase':'failed','time':time.time(),
                    'pid':os.getpid(),'error':str(exc),'traceback':traceback.format_exc()})
        raise

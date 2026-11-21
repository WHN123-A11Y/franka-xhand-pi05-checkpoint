"""Full-frame native pi05 fine-tuning with episode-held-out model selection."""
from pathlib import Path
import argparse,dataclasses,functools,hashlib,importlib.util,json,os,shutil,sys,time
ROOT=Path('/data/adminad/workspace/FineVLA_clean');NATIVE=Path('/data/fangchuan/workspace/openpi_pi05_xhand_20260923')
sys.path.insert(0,str(NATIVE/'src'));sys.path.insert(0,str(NATIVE/'packages/openpi-client/src'));sys.path.insert(0,str(Path(__file__).parent))
import jax,jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import optax
from openpi.models import model as mm
from openpi.training import config as cfg,optimizer as optim,sharding,checkpoints,weight_loaders
from data import FullDataset,make_data_config,make_transform
spec=importlib.util.spec_from_file_location('native_train',NATIVE/'scripts/train.py');nt=importlib.util.module_from_spec(spec);spec.loader.exec_module(nt)
INIT=NATIVE/'checkpoints_multitask_20260925/pi05_xhand/xhand_multitask_99_full_ft_b32_20260925/21000'
def save(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
class AssetProvider:
 def __init__(self,config):self.config=config
 def data_config(self):return self.config
def train_step(config,rng,state,batch):
 model=nnx.merge(state.model_def,state.params);model.train();obs,actions,valid=batch;key=jax.random.fold_in(rng,state.step)
 def objective(m):
  loss=m.compute_loss(key,obs,actions,train=True);return (loss*valid).sum()/valid.sum().clip(1)
 loss,grads=nnx.value_and_grad(objective,argnums=nnx.DiffState(0,config.trainable_filter))(model)
 params=state.params.filter(config.trainable_filter);updates,os_=state.tx.update(grads,state.opt_state,params);nnx.update(model,optax.apply_updates(params,updates))
 state=dataclasses.replace(state,step=state.step+1,params=nnx.state(model),opt_state=os_)
 return state,{'loss':loss,'grad_norm':optax.global_norm(grads)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=ROOT/'runs/real_franka_full_training_20261117/policy');p.add_argument('--data-root',type=Path,default=ROOT/'runs/real_franka_full_training_20261117/data');p.add_argument('--steps',type=int,default=30000);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--fsdp-devices',type=int,default=2);p.add_argument('--val-interval',type=int,default=500);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();a.run_dir.mkdir(parents=True,exist_ok=True)
 def status(stage,**kw):obj=dict(stage=stage,time=time.time(),**kw);save(a.run_dir/'status.json',obj);print(json.dumps(obj),flush=True)
 status('loading_data',scope='native full-model fine-tune; no robot control')
 train=FullDataset(a.data_root,'train');val=FullDataset(a.data_root,'val');base=cfg.get_config('pi05_xhand')
 c=dataclasses.replace(base,exp_name='full_frames',name='pi05_franka_full',checkpoint_base_dir=str(a.run_dir/'checkpoints'),assets_base_dir=str(a.data_root/'assets'),num_train_steps=a.steps,batch_size=a.batch_size,fsdp_devices=a.fsdp_devices,num_workers=0,ema_decay=None,weight_loader=weight_loaders.CheckpointWeightLoader(str(INIT/'params')),lr_schedule=optim.CosineDecaySchedule(warmup_steps=500,peak_lr=1e-5,decay_steps=max(30000,a.steps),decay_lr=1e-6),resume=a.resume,wandb_enabled=False)
 dc=make_data_config(a.data_root,c.model);transform=make_transform(a.data_root,c.model);assets=AssetProvider(dc);mesh=sharding.make_mesh(a.fsdp_devices);ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS));rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
 normfile=a.data_root/'assets/xhand_local/norm_stats.json';identity={'source':str(NATIVE),'source_revision':'215abfb217dbac7d5f1273282331b9b1866c0479','init_checkpoint':str(INIT),'normalization_sha256':hashlib.sha256(normfile.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'batch_size':a.batch_size,'fsdp_devices':a.fsdp_devices,'steps':a.steps,'val_interval':a.val_interval,'dataset_manifest_sha256':hashlib.sha256((a.data_root/'manifest.json').read_bytes()).hexdigest(),'action_horizon':10,'real_action_dims':19,'terminal_repeat':'masked in train and validation loss','ema':None,'seed':42}
 identpath=a.run_dir/'identity.json'
 if identpath.exists() and json.loads(identpath.read_text())!=identity:raise RuntimeError('Resume identity mismatch')
 save(identpath,identity)
 if not a.smoke and (a.run_dir/'READY.json').exists():
  status('finished',**json.loads((a.run_dir/'READY.json').read_text()));return
 manager,resuming=checkpoints.initialize_checkpoint_dir(c.checkpoint_dir,keep_period=5000,overwrite=False,resume=a.resume)
 state,ss=nt.init_train_state(c,jax.random.key(42),mesh,resume=resuming)
 if resuming:state=checkpoints.restore_state(manager,state,assets)
 jax.block_until_ready(state);status('model_initialized',step=int(state.step),devices=[str(x) for x in jax.devices()])
 stepfn=jax.jit(functools.partial(train_step,c),in_shardings=(rep,ss,ds),out_shardings=(ss,rep),donate_argnums=(1,))
 def batch(dataset,indices,weights=None):
  vals=[];masks=[]
  for n,idx in enumerate(indices):
   raw=dataset[int(idx)];valid=~raw.pop('action_is_pad');raw.pop('episode');raw.pop('frame');item=transform(raw);vals.append(item);masks.append(valid.astype(np.float32)*(1 if weights is None else weights[n]))
  b=jax.tree.map(lambda *x:np.stack(x),*vals);obs=mm.Observation.from_dict({k:v for k,v in b.items() if k!='actions'})
  return jax.device_put((obs,b['actions'],np.stack(masks)),ds)
 def eval_loss(params,obs,actions,valid,keys):
  m=nnx.merge(state.model_def,params);m.eval();loss=m.compute_loss(keys,obs,actions,train=False);return (loss*valid).sum(),valid.sum()
 ev=jax.jit(eval_loss,in_shardings=(ss.params,ds,ds,ds,rep),out_shardings=rep)
 def validate():
  total=count=0.;n=min(16,len(val)) if a.smoke else len(val)
  for start in range(0,n,a.batch_size):
   ids=list(range(start,min(start+a.batch_size,n)));w=[1.]*len(ids)
   while len(ids)<a.batch_size:ids.append(ids[-1]);w.append(0.)
   obs,ac,mask=batch(val,ids,w);value,num=ev(state.params,obs,ac,mask,jax.random.fold_in(jax.random.key(1701),start));total+=float(value);count+=float(num)
  return total/count
 selection_path=a.run_dir/'selection.json';selection=json.loads(selection_path.read_text()) if selection_path.exists() else None
 def commit(step,metric,is_best):
  checkpoints.save_state(manager,state,assets,step);manager.wait_until_finished()
  if is_best:
   src=Path(c.checkpoint_dir)/str(step);tmp=a.run_dir/'best_export.tmp'
   if tmp.exists():shutil.rmtree(tmp)
   tmp.mkdir()
   for name in ('params','assets'):shutil.copytree(src/name,tmp/name,copy_function=os.link)
   best=a.run_dir/'best_export';old=a.run_dir/'best_export.old'
   if old.exists():shutil.rmtree(old)
   if best.exists():best.rename(old)
   tmp.rename(best)
   if old.exists():shutil.rmtree(old)
   save(selection_path,{'step':step,'validation_native_flow_mse19':metric,'selection_scope':'all validation episode frames; fixed noise seed1701; no test used','checkpoint':str(best),'identity':identity})
 if selection is None:
  status('initial_validation',step=int(state.step));metric=validate();selection={'step':int(state.step),'validation_native_flow_mse19':metric};commit(int(state.step),metric,True)
  status('initial_checkpoint_saved',step=int(state.step),validation=metric)
 best=float(selection['validation_native_flow_mse19']);history=json.loads((a.run_dir/'history.json').read_text()) if (a.run_dir/'history.json').exists() else [];start=int(state.step);rng=jax.random.key(42);permutation={}
 for step in range(start,a.steps):
  ids=[]
  for k in range(a.batch_size):
   pos=step*a.batch_size+k;epoch=pos//len(train)
   if epoch not in permutation:
    permutation={epoch:np.random.default_rng(42+epoch).permutation(len(train))}
   ids.append(int(permutation[epoch][pos%len(train)]))
  b=batch(train,ids)
  with sharding.set_mesh(mesh):state,inf=stepfn(rng,state,b)
  actual=step+1
  if actual%10==0 or actual==1:
   metrics={k:float(v) for k,v in jax.device_get(inf).items()};status('training',step=actual,total=a.steps,**metrics)
  if actual%a.val_interval==0 or actual==a.steps:
   status('validation',step=actual);metric=validate();is_best=metric<best
   if is_best:best=metric
   commit(actual,metric,is_best);history.append({'step':actual,'validation_native_flow_mse19':metric,'best':best});save(a.run_dir/'history.json',history);status('checkpointed',step=actual,validation=metric,best=best)
 manager.wait_until_finished()
 selected=json.loads(selection_path.read_text());export=a.run_dir/'selected'
 if export.exists():raise RuntimeError('selected export already exists')
 shutil.copytree(a.run_dir/'best_export',export,copy_function=os.link)
 result={'checkpoint':str(export),'native_root':str(NATIVE),'native_config':'pi05_xhand','step':selected['step'],'validation_native_flow_mse19':selected['validation_native_flow_mse19'],'normalization_sha256':identity['normalization_sha256'],'trained_steps':a.steps,'full_model_finetune':True,'scope':'smoke only' if a.smoke else 'all TRAIN frames, validation selection only; no real robot execution'}
 save(a.run_dir/'selected.json',result)
 if not a.smoke:save(a.run_dir/'READY.json',result)
 status('finished',**result)
if __name__=='__main__':main()

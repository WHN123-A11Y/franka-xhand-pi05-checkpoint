"""Shared full-frame, episode-isolated Franka/XHand dataset API."""
from pathlib import Path
from collections import OrderedDict
import json
import numpy as np

CAMERA_MAP={'head_view':'base_0_rgb','front_left_view':'left_wrist_0_rgb','wrist_view':'right_wrist_0_rgb'}
DEFAULT_ROOT=Path('/data/adminad/workspace/FineVLA_clean/runs/real_franka_full_training_20261117/data')
class FullDataset:
 def __init__(self,data_root=DEFAULT_ROOT,split='train',horizon=10):
  self.root=Path(data_root);self.split=split;self.horizon=horizon
  if not (self.root/'READY.json').exists():raise RuntimeError('full dataset not READY')
  z=np.load(self.root/'state_action.npz',allow_pickle=False);self.state=z['state'];self.action=z['action'];self.episode=z['episode'];self.frame=z['frame'];self.prompts=z['prompts'];self.lengths=z['lengths'];self.starts=np.r_[0,np.cumsum(self.lengths)[:-1]]
  sp=json.loads((self.root/'split.json').read_text());self.episode_ids=set(sp['episodes'][split]);self.indices=np.flatnonzero(np.isin(self.episode,list(self.episode_ids)));self.cache=OrderedDict()
 def __len__(self):return len(self.indices)
 def _media(self,ep):
  if ep in self.cache:x=self.cache.pop(ep);self.cache[ep]=x;return x
  x={v:np.load(self.root/'media'/f'episode_{ep:06d}'/(v+'.npy'),mmap_mode='r') for v in CAMERA_MAP};self.cache[ep]=x
  while len(self.cache)>8:self.cache.popitem(last=False)
  return x
 def __getitem__(self,index):
  idx=int(self.indices[int(index)]);ep=int(self.episode[idx]);f=int(self.frame[idx]);local=np.arange(f,f+self.horizon);valid=local<self.lengths[ep];future=self.starts[ep]+np.minimum(local,self.lengths[ep]-1);media=self._media(ep)
  return {'image':{k:media[v][f] for v,k in CAMERA_MAP.items()},'image_mask':{k:np.asarray(True) for k in CAMERA_MAP.values()},'state':self.state[idx].copy(),'actions':self.action[future].copy(),'action_is_pad':~valid,'prompt':str(self.prompts[ep]),'episode':ep,'frame':f}

def make_data_config(data_root,model_config):
 from openpi.training import config
 from openpi.shared import normalize
 from openpi import transforms
 return config.DataConfig(repo_id='franka_full_local',asset_id='xhand_local',norm_stats=normalize.load(Path(data_root)/'assets/xhand_local'),model_transforms=config.ModelTransformFactory()(model_config),use_quantile_norm=True)

def make_transform(data_root,model_config):
 from openpi import transforms
 cfg=make_data_config(data_root,model_config)
 return transforms.compose([transforms.Normalize(cfg.norm_stats,use_quantiles=True),*cfg.model_transforms.inputs])

"""Materialize full real-robot trajectories and train-only normalization."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from zipfile import ZipFile
import hashlib,json,os,subprocess,time
import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq

ROOT=Path('/data/adminad/workspace/FineVLA_clean')
OUT=ROOT/'runs/real_franka_full_training_20261117/data'
ZIP=ROOT/'real_data/franka_xhand_pi05_pick_place.zip'
SPLIT=ROOT/'runs/franka_teacher_64_handoff_20261116/split_validation.json'
VIEWS=['head_view','front_left_view','wrist_view']
def save(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def status(phase,**kw):save(OUT/'status.json',dict(phase=phase,time=time.time(),**kw));print(phase,kw,flush=True)
def decode(job):
 ep,n,v=job;dest=OUT/'media'/f'episode_{ep:06d}'/(v+'.npy')
 if dest.exists():
  a=np.load(dest,mmap_mode='r');assert a.shape==(n,168,224,3) and a.dtype==np.uint8;return ep,v,'exists'
 src=OUT/'source/videos/chunk-000'/('observation.images.'+v)/f'episode_{ep:06d}.mp4'
 ff=imageio_ffmpeg.get_ffmpeg_exe();cmd=[ff,'-v','error','-threads','1','-i',str(src),'-vf','scale=224:168:flags=bicubic','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-']
 raw=subprocess.check_output(cmd);a=np.frombuffer(raw,np.uint8);assert a.size==n*168*224*3,(src,a.size,n)
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp')
 with tmp.open('wb') as f:np.save(f,a.reshape(n,168,224,3))
 tmp.replace(dest);return ep,v,'written'
def main():
 OUT.mkdir(parents=True,exist_ok=True);split=json.loads(SPLIT.read_text());prefix='franka_xhand_pi05_pick_place/'
 status('extracting_source')
 with ZipFile(ZIP) as z:
  for info in z.infolist():
   rel=info.filename.removeprefix(prefix)
   if not rel.startswith(('meta/','data/','videos/')) or info.is_dir():continue
   dest=OUT/'source'/rel
   if dest.exists():assert dest.stat().st_size==info.file_size;continue
   dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(info))
 info=json.loads((OUT/'source/meta/info.json').read_text());episodes=[json.loads(x) for x in (OUT/'source/meta/episodes.jsonl').read_text().splitlines() if x]
 byep={e:s for s,ids in split['episodes'].items() for e in ids};states=[];actions=[];eps=[];frames=[];prompts=[];train=[]
 for ep in episodes:
  i=ep['episode_index'];t=pq.ParquetFile(OUT/'source/data/chunk-000'/f'episode_{i:06d}.parquet').read()
  s=np.asarray(t['observation.state'].to_pylist(),np.float32);a=np.asarray(t['action'].to_pylist(),np.float32);n=len(s);assert n==ep['length'] and s.shape==a.shape==(n,19) and np.isfinite(s).all() and np.isfinite(a).all()
  states.append(s);actions.append(a);eps.extend([i]*n);frames.extend(range(n));prompts.append(ep['tasks'][0]);train.extend([byep[i]=='train']*n)
 states=np.concatenate(states);actions=np.concatenate(actions);train=np.asarray(train)
 np.savez(OUT/'state_action.npz',state=states,action=actions,episode=np.asarray(eps,np.int32),frame=np.asarray(frames,np.int32),prompts=np.asarray(prompts),lengths=np.asarray([x['length'] for x in episodes],np.int32))
 def stats(a):return {'mean':a.mean(0,dtype=np.float64).tolist(),'std':a.std(0,dtype=np.float64).tolist(),'q01':np.quantile(a,.01,axis=0).tolist(),'q99':np.quantile(a,.99,axis=0).tolist()}
 norms={'norm_stats':{'state':stats(states[train]),'actions':stats(actions[train])}}
 save(OUT/'assets/xhand_local/norm_stats.json',norms);save(OUT/'split.json',split)
 jobs=[(e['episode_index'],e['length'],v) for e in episodes for v in VIEWS];status('decoding_media',completed=0,total=len(jobs))
 with ThreadPoolExecutor(max_workers=6) as pool:
  for i,f in enumerate(as_completed([pool.submit(decode,j) for j in jobs]),1):
   f.result()
   if i%12==0 or i==len(jobs):status('decoding_media',completed=i,total=len(jobs))
 counts={sp:int(sum(1 for e in eps if byep[e]==sp)) for sp in ('train','val','test')}
 meta={'dataset':str(ZIP),'dataset_sha256':split['dataset_sha256'],'frames':len(states),'counts':counts,'episodes':{s:len(v) for s,v in split['episodes'].items()},'normalization_fit':'all TRAIN episode rows only, exact np quantile0.01/0.99; existing held-out splits unchanged','frame_shape':[168,224,3],'media':'ffmpeg bicubic resize of original640x480 AV1 to224x168, aspect preserved; model later letterboxes224x224','action_channels':info['features']['action']['names'],'action_semantics':'19 recorded absolute positions in original order; same-row future10 chunk; terminal repeat accompanied by mask','source_fps':info['fps']}
 save(OUT/'manifest.json',meta);save(OUT/'READY.json',meta);status('finished',**counts)
if __name__=='__main__':main()

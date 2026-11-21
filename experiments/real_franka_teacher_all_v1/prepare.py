"""Extract five exact, evenly spaced frames per episode for all three cameras.

Source instructions and noun spans are copied verbatim. No teacher or labels
are generated here. Any mismatch in video/parquet length fails preflight.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import hashlib,json,re,subprocess,zipfile
import numpy as np
import pyarrow.parquet as pq

ROOT=Path('/data/adminad/workspace/FineVLA_clean')
ZIP=ROOT/'real_data/franka_xhand_pi05_pick_place.zip'
OUT=ROOT/'runs/real_franka_teacher_all_20261116'
FF=Path('/data/fangchuan/tools/conda-envs/robotwin-openwam/bin/ffmpeg')
FP=FF.with_name('ffprobe')
PREFIX='franka_xhand_pi05_pick_place/'
VIEWS=['head_view','front_left_view','wrist_view']
OUT.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
 eps=[json.loads(x) for x in z.read(PREFIX+'meta/episodes.jsonl').splitlines()]
 tasks=[json.loads(x) for x in z.read(PREFIX+'meta/tasks.jsonl').splitlines()]
source_hash=hashlib.sha256()
with ZIP.open('rb') as f:
 for block in iter(lambda:f.read(8<<20),b''):source_hash.update(block)

def extract(ep):
 idx=ep['episode_index'];length=ep['length'];text=ep['tasks'][0]
 m=re.fullmatch(r'Pick up the (.+?) and put it into the (.+?)\.',text)
 if not m:raise ValueError(f'episode {idx}: cannot parse exact noun spans: {text}')
 frames=np.linspace(0,length-1,5).round().astype(int).tolist()
 assert len(set(frames))==5 and frames[0]==0 and frames[-1]==length-1
 work=OUT/'inputs'/f'episode_{idx:06d}';work.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(ZIP) as z:
  table=pq.read_table(BytesIO(z.read(PREFIX+f'data/chunk-000/episode_{idx:06d}.parquet')),columns=['timestamp','frame_index','episode_index','task_index'])
  assert table.num_rows==length
  times=np.asarray(table['timestamp'].to_pylist());ti=np.asarray(table['task_index'].to_pylist());assert np.all(ti==ti[0])
  outputs=[]
  for view in VIEWS:
   arc=PREFIX+f'videos/chunk-000/observation.images.{view}/episode_{idx:06d}.mp4'
   video=work/(view+'.source.mp4');video.write_bytes(z.read(arc))
   try:
    raw=subprocess.check_output([str(FP),'-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height,avg_frame_rate','-of','json',str(video)])
    stream=json.loads(raw)['streams'][0]
    if int(stream.get('nb_frames',-1))!=length:raise ValueError(f'{idx} {view}: video {stream.get("nb_frames")} frames != metadata {length}')
    if (stream.get('width'),stream.get('height'))!=(640,480):raise ValueError(f'{idx} {view}: unexpected dimensions')
    folder=work/view;folder.mkdir(exist_ok=True)
    if not all((folder/f'frame_{f:06d}.png').exists() for f in frames):
     filt='select='+ '+'.join(f'eq(n\\,{f})' for f in frames)
     for existing in folder.glob('pending_*.png'):existing.unlink()
     subprocess.run([str(FF),'-loglevel','error','-threads','2','-y','-i',str(video),'-vf',filt,'-vsync','0','-start_number','0',str(folder/'pending_%02d.png')],check=True,stderr=subprocess.PIPE)
     pending=sorted(folder.glob('pending_*.png'))
     if len(pending)!=5:raise ValueError(f'{idx} {view}: decoded {len(pending)} selected frames')
     for temp,frame in zip(pending,frames):temp.replace(folder/f'frame_{frame:06d}.png')
    for frame in frames:
     img=folder/f'frame_{frame:06d}.png';digest=hashlib.sha256(img.read_bytes()).hexdigest()
     for role,phrase in [('object',m.group(1)),('destination',m.group(2))]:
      outputs.append({'id':f'ep{idx:06d}_{view}_f{frame:06d}_{role}','episode':idx,'view':view,'frame_index':frame,'timestamp':float(times[frame]),'task_index':int(ti[0]),'original_instruction':text,'role':role,'expression_verbatim':phrase,'rgb_path':str(img),'rgb_sha256':digest,'flag_instruction_typo':'grew' in text,'shard':idx%8,'frame_count_metadata':length})
   finally:video.unlink(missing_ok=True)
 return outputs

all_jobs=[]
with ThreadPoolExecutor(max_workers=8) as pool:
 futures={pool.submit(extract,e):e['episode_index'] for e in eps}
 for fut in as_completed(futures):
  idx=futures[fut];jobs=fut.result();all_jobs.extend(jobs);print(f'episode {idx:03d}: {len(jobs)} role-view samples ready',flush=True)
all_jobs.sort(key=lambda x:(x['episode'],x['frame_index'],VIEWS.index(x['view']),x['role']))
assert len(eps)==108 and len(all_jobs)==108*5*3*2
for shard in range(8):
 (OUT/f'manifest_shard_{shard}.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in all_jobs if j['shard']==shard))
meta={'source_zip':str(ZIP),'source_sha256':source_hash.hexdigest(),'episodes':len(eps),'frames_per_episode':5,'frame_selection':'numpy.linspace(0, episode_length-1, 5).round(), includes first and last','frame_groups':len(eps)*5,'views':VIEWS,'roles':['object','destination'],'role_view_samples':len(all_jobs),'shards':8,'instruction_parsing':'exact string spans from original instruction; no spelling or category correction; grew explicitly flagged','timestamp':'copied from same frame index in parquet; metadata does not establish camera sensor hardware synchronization','status':'prepared only; no teacher predictions implied'}
(OUT/'manifest_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print(json.dumps(meta,ensure_ascii=False),flush=True)

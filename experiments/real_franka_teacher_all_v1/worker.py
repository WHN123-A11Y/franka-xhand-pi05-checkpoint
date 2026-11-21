"""Frozen original Qwen+DINO per-view Teacher, sharded by episode.

No human labels or simulator masks are read. Results are predictions, not
verified ground truth. Completed outputs and failures are atomically persisted.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback
ROOT=Path('/data/adminad/workspace/FineVLA_clean')
OUT=ROOT/'runs/real_franka_teacher_all_20261116'
sys.path.insert(0,str(ROOT/'experiments/refcoco_eval'))
from teacher_adapter import QwenRuntime,TeacherAdapter
from color_contract import install as install_literal_color_contract
install_literal_color_contract()
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,choices=range(8),required=True);p.add_argument('--retry-errors',action='store_true');a=p.parse_args()
jobs=[json.loads(x) for x in (OUT/f'manifest_shard_{a.shard}.jsonl').read_text().splitlines()]
dest=OUT/'predictions';dest.mkdir(parents=True,exist_ok=True)
status=OUT/'workers'/f'shard_{a.shard}.json';status.parent.mkdir(exist_ok=True)
def atomic(path,data):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));os.replace(tmp,path)
def report(state,current=None):
 complete=failed=valid=abstained=0;last_done=None
 for j in jobs:
  f=dest/(j['id']+'.json')
  if f.exists():
   try:
    r=json.loads(f.read_text());complete+=1;failed+=r['status']=='error';valid+=r['status']=='complete' and bool(r['result']['valid']);abstained+=r['status']=='complete' and not r['result']['valid'];last_done=j['id']
   except (OSError,ValueError,KeyError):pass
 atomic(status,{'shard':a.shard,'state':state,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'total':len(jobs),'completed':complete,'valid_proposals':valid,'no_proposal':abstained,'exceptions':failed,'current':current,'last_done':last_done,'updated_unix':time.time()})
report('loading_models')
runtime=QwenRuntime(model_path='/data/adminad/models/Qwen3.5-4B')
teacher=TeacherAdapter(runtime,OUT/'worker_runtime'/f'shard_{a.shard}')
metadata={'shard':a.shard,'method':'TeacherAdapter with literal-role-color preservation; frozen Qwen3.5-4B plus Grounding DINO; per-image/per-view, not cross-view fused','source_code':str(ROOT/'experiments/refcoco_eval/teacher_adapter.py'),'color_override':str(ROOT/'experiments/real_franka_teacher_all_v1/color_contract.py'),'manifest_sha256':hashlib.sha256((OUT/f'manifest_shard_{a.shard}.jsonl').read_bytes()).hexdigest(),'teacher_metadata':teacher.metadata,'limitations':['Verifier checks category and color, not destination shape or spatial relations.','The exact recorded task phrase is used, including bowling ball and grew; source expressions may not describe visible objects accurately.','No independent instance truth on this real dataset; valid is Teacher decision, not verified correctness.']}
atomic(OUT/'workers'/f'shard_{a.shard}_config.json',metadata)
for j in jobs:
 f=dest/(j['id']+'.json')
 if f.exists():
  try:
   old=json.loads(f.read_text())
   if old.get('rgb_sha256')!=j['rgb_sha256'] or old.get('expression_verbatim')!=j['expression_verbatim']:raise ValueError('resume identity mismatch '+j['id'])
   if old.get('status')=='complete' or not a.retry_errors:continue
  except json.JSONDecodeError:raise RuntimeError('corrupted output '+str(f))
 report('running',j['id']);start=time.monotonic();rec={**j,'status':'running','teacher_method':'Qwen3.5-4B + Grounding DINO TeacherAdapter with literal-role-color preservation, independent view','is_verified_ground_truth':False}
 try:
  if hashlib.sha256(Path(j['rgb_path']).read_bytes()).hexdigest()!=j['rgb_sha256']:raise ValueError('source image hash mismatch')
  rec['result']=teacher.predict(j['rgb_path'],j['expression_verbatim'],j['id']);rec['status']='complete'
 except Exception as e:
  rec['status']='error';rec['error']=repr(e);rec['traceback']=traceback.format_exc()
 rec['elapsed_seconds']=time.monotonic()-start;atomic(f,rec)
 print(json.dumps({'id':j['id'],'status':rec['status'],'valid':rec.get('result',{}).get('valid'),'failure':rec.get('result',{}).get('failure_reason'),'elapsed_sec':round(rec['elapsed_seconds'],2)},ensure_ascii=False),flush=True)
report('finished')

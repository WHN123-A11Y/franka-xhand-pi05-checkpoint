"""Build separately masked localization, view-presence, and candidate labels."""
from pathlib import Path
import argparse,collections,hashlib,json
ROOT=Path('/data/adminad/workspace/FineVLA_clean');VIEWS=['head_view','front_left_view','wrist_view'];K=4
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
def box_ok(b):return isinstance(b,list) and len(b)==4 and all(isinstance(x,(int,float)) for x in b) and 0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1
def main():
 p=argparse.ArgumentParser();p.add_argument('--teacher-dir',type=Path,default=ROOT/'runs/real_franka_teacher_all_20261116');p.add_argument('--presence-dir',type=Path,default=ROOT/'runs/real_franka_full_training_20261117/student/presence');p.add_argument('--out',type=Path,default=ROOT/'runs/real_franka_full_training_20261117/student');p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args()
 split_path=ROOT/'runs/franka_teacher_64_handoff_20261116/split_validation.json';split=json.loads(split_path.read_text());splits={ep:s for s,eps in split['episodes'].items() for ep in eps};groups={};scenes={};counts=collections.Counter();all_sources=[]
 for mf in sorted(a.teacher_dir.glob('manifest_shard_*.jsonl')):
  for line in mf.read_text().splitlines():
   j=json.loads(line);key=(j['episode'],j['frame_index'],j['role']);v=VIEWS.index(j['view']);name=f"ep{j['episode']:06d}_f{j['frame_index']:06d}"
   s=groups.setdefault(key,{'episode':j['episode'],'frame':j['frame_index'],'role':j['role'],'instruction':j['original_instruction'],'expression':j['expression_verbatim'],'split':splits[j['episode']],'scene':name,'images':[None]*3,'boxes':[[0.,0.,0.,0.] for _ in VIEWS],'loc_valid':[False]*3,'supply':[-1]*3,'candidate_boxes':[[[0.,0.,0.,0.] for _ in range(K)] for _ in VIEWS],'candidate_labels':[[-1]*K for _ in VIEWS],'sources':[None]*3})
   assert s['expression']==j['expression_verbatim'];s['images'][v]=j['rgb_path'];scene=scenes.setdefault(name,{'scene':name,'episode':j['episode'],'frame':j['frame_index'],'images':[None]*3});scene['images'][v]=j['rgb_path']
   source={'teacher_id':j['id'],'rgb_sha256':j['rgb_sha256']};path=a.teacher_dir/'predictions'/(j['id']+'.json');result=None
   if path.exists():
    r=json.loads(path.read_text());assert r['rgb_sha256']==j['rgb_sha256'] and r['expression_verbatim']==j['expression_verbatim'];source['teacher_sha256']=sha(path);all_sources.append(source['teacher_sha256']);result=r.get('result',{})
    if r.get('status')=='complete':counts['teacher_complete']+=1
    else:counts['teacher_error']+=1
   else:counts['teacher_missing']+=1
   accepted=bool(result and result.get('valid') and result.get('box_xyxy'))
   candidates=[]
   if accepted:
    b=[float(x)/d for x,d in zip(result['box_xyxy'],(640,480,640,480))]
    if not box_ok(b):raise ValueError('invalid final Teacher box '+j['id'])
    s['boxes'][v]=b;s['loc_valid'][v]=True;s['supply'][v]=1;source['supply_source']='accepted_teacher_box_visible_in_view';candidates.append({'box':b,'label':1,'source':'final_accepted_teacher_box'})
   else:
    pp=a.presence_dir/'predictions'/(j['id']+'.json')
    if pp.exists():
     pr=json.loads(pp.read_text());assert pr['rgb_sha256']==j['rgb_sha256'] and pr['expression_verbatim']==j['expression_verbatim'];label=pr.get('label',-1);assert label in (-1,0,1);s['supply'][v]=label;source.update(presence_sha256=sha(pp),supply_source='explicit_full_image_presence_teacher',presence_status=pr.get('status'));all_sources.append(source['presence_sha256']);counts['presence_records']+=1
    else:counts['presence_missing_for_unknown_view']+=1;source['supply_source']='unknown_masked'
   for candidate in (result or {}).get('trace',{}).get('candidates',[]):
    b=candidate.get('box');decision=(candidate.get('verification') or {}).get('decision')
    if box_ok(b) and decision in ('yes','no'):candidates.append({'box':list(map(float,b)),'label':1 if decision=='yes' else 0,'source':'explicit_candidate_verification_'+decision})
   # Retain genuine candidate labels only. Duplicate coordinates with conflicting
   # judgments are excluded, rather than silently choosing a favorable label.
   by_box={}
   for c in candidates:
    ident=tuple(round(x,6) for x in c['box']);prev=by_box.get(ident)
    if prev is None:by_box[ident]=c
    elif prev['label']!=c['label']:prev['label']=-1;counts['candidate_conflicts_masked']+=1
   choices=[c for c in by_box.values() if c['label']>=0];choices.sort(key=lambda c:(c['label']!=0,c['source']!='final_accepted_teacher_box'))
   # Include final positive whenever available, then explicit hard negatives and
   # remaining verified candidates, all in one fixed-size padded bank.
   if accepted:
    positive=next((c for c in choices if c['source']=='final_accepted_teacher_box'),None)
    if positive:choices=[positive]+[c for c in choices if c is not positive]
   choices=choices[:K]
   for k,c in enumerate(choices):s['candidate_boxes'][v][k]=c['box'];s['candidate_labels'][v][k]=c['label']
   source['candidates']=choices;s['sources'][v]=source
 samples=sorted(groups.values(),key=lambda s:(s['episode'],s['frame'],s['role']));scene_list=sorted(scenes.values(),key=lambda s:s['scene']);assert len(samples)==1080 and len(scene_list)==540
 for s in samples:
  assert all(p and Path(p).is_file() for p in s['images'])
  counts[s['split']+'_samples']+=1
  for x in s['supply']:counts[s['split']+'_supply_'+str(x)]+=1
  for labels in s['candidate_labels']:
   for x in labels:counts[s['split']+'_candidate_'+str(x)]+=1
  counts[s['split']+'_localization_views']+=sum(s['loc_valid'])
 expressions={}
 for s in samples:expressions.setdefault(s['instruction'],{})[s['role']]=s['expression']
 assert all(set(v)=={'object','destination'} for v in expressions.values())
 a.out.mkdir(parents=True,exist_ok=True);atomic(a.out/'samples.json',samples);atomic(a.out/'scenes.json',scene_list);atomic(a.out/'expressions.json',expressions)
 audit={'counts':dict(counts),'split_sha256':sha(split_path),'source_digest':hashlib.sha256(''.join(sorted(all_sources)).encode()).hexdigest(),'coordinates':'original full RGB normalized xyxy; 640x480; no source crop','localization_positive':'accepted original Teacher box only','supply_positive':'accepted Teacher region or explicit whole-image presence yes','supply_negative':'explicit WHOLE-IMAGE presence no only; rejected bbox never implies absent target','candidate_labels':'actual per-region verification yes/no only; uncertain, parse error and conflicting duplicate judgments masked','unknown_value':-1,'is_human_ground_truth':False,'teacher_limits':['Teacher category/color checks are pseudo supervision; size/shape/relation correctness not guaranteed.','Original expressions including grew and bowling ball are retained unchanged.']}
 atomic(a.out/'supervision_audit.json',audit)
 missing=counts['teacher_missing']+counts['teacher_error']+counts['presence_missing_for_unknown_view']
 if missing and not a.allow_incomplete:raise SystemExit('Incomplete supervision sources: '+str(missing)+'; saved audit but did not write labels READY')
 if not a.allow_incomplete:atomic(a.out/'LABELS_READY.json',{'samples':len(samples),'scenes':len(scene_list),'source_digest':audit['source_digest'],'train_supply_identifiable':counts['train_supply_0']>0 and counts['train_supply_1']>0,'train_candidate_identifiable':counts['train_candidate_0']>0 and counts['train_candidate_1']>0})
 print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()

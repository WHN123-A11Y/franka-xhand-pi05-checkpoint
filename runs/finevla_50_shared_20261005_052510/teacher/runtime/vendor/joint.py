"""Verify three synchronized camera proposals in one Qwen call, offline only."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

VIEWS=['cam_high','cam_left_wrist','cam_right_wrist']
METHODS=['teacher','dino','teacher_dino']
PROMPT_VERSION='category_color_synchronized_three_views_v1'
PROMPT='''These three complete camera images show the SAME scene at the SAME instant, from the main, left-wrist and right-wrist cameras. Each image has one thin black/white rectangle marking a proposal for the same requested object. The rectangles are annotations, not part of the object.
Requested object category: {category}
Requested colors: {colors}
{presence}

Use the three views TOGETHER to recognize the object: an object unclear or partly hidden in one view may be recognizable in another. Compare actual visible appearance across cameras. Image scale, orientation and position differ across cameras; never transfer pixel coordinates or left/right relations between them.
For EACH camera, decide whether ITS OWN rectangle contains a visible part of the requested physical object. Seeing the target in another image helps recognize it, but does not validate an empty box, a robot part, or a different object in this image. A box may be loose or show only part of the object if the visible content can be identified using the synchronized views. Do not force all cameras to match. All proposals may be wrong even if the object exists in the scene.
Evaluate ONLY object category and the stated colors. An empty color list means no color requirement. Color can belong to a visible component, not necessarily the entire object. Do not add material, texture, pattern, brand, size, detailed shape, action, pose or location requirements.
For each view: yes = visible target inside this box; no = another object, clear color mismatch, robot part, empty/background or target only outside this box; uncertain = insufficient visible evidence. Describe the visible evidence and how another view helps, without inventing an unseen match.
Return only JSON with exactly these three camera keys. Each value has exactly four string fields:
{{"cam_high":{{"observed_object":"actual boxed object or background","observed_color":"visible color or unclear","reason":"brief local evidence and any support from another view","decision":"yes|no|uncertain"}},"cam_left_wrist":{{"observed_object":"actual boxed object or background","observed_color":"visible color or unclear","reason":"brief evidence","decision":"yes|no|uncertain"}},"cam_right_wrist":{{"observed_object":"actual boxed object or background","observed_color":"visible color or unclear","reason":"brief evidence","decision":"yes|no|uncertain"}}}}
'''


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def parse(raw):
    lines=raw.strip().splitlines();fence=len(lines)>=3 and lines[0] in {'```','```json'} and lines[-1]=='```'
    try:
        answer=json.loads('\n'.join(lines[1:-1]) if fence else raw)
        if not isinstance(answer,dict) or set(answer)!=set(VIEWS):raise ValueError('missing or extra camera key')
        for v,a in answer.items():
            if not isinstance(a,dict) or set(a)!={'observed_object','observed_color','reason','decision'}:raise ValueError('invalid camera schema')
            if any(not isinstance(x,str) or not x.strip() for x in a.values()):raise ValueError('empty/nonstring field')
            if a['decision'] not in {'yes','no','uncertain'}:raise ValueError('invalid decision')
        return answer,None
    except (ValueError,TypeError) as e:return None,str(e)


def prepare(a):
    source=a.source.resolve();root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    for n in ['runtime','marked','results','attempts','logs','final/results','final/scoring_results','final/marked']:(root/n).mkdir(parents=True)
    shutil.copy2(__file__,root/'runtime/run.py')
    if read(source/'status.json')['phase']!='DONE':raise ValueError('Source incomplete')
    rows=read(source/'ranked_candidates.json')
    if a.episodes is not None:rows=[r for r in rows if r['episode'] in a.episodes]
    if a.frames is not None:rows=[r for r in rows if r['frame'] in a.frames]
    if not rows:raise ValueError('Empty selection')
    grouped=defaultdict(list)
    for r in rows:grouped[(r['scene'],r['role'])].append(r)
    cases=[];groups=[];by={}
    for r in rows:
        group={k:r[k] for k in ['name','scene','role','view','domain','episode','frame','instruction','target','role_description','image','image_sha256','candidate_indices']}
        groups.append(group)
        for c in r['candidates']:
            if c['rank']!=1:raise ValueError('Expected method Top-1 only')
            if sha(c['image'])!=c['image_sha256'] or sha(c['marked_image'])!=c['marked_sha256']:raise ValueError('Source image changed')
            item={k:v for k,v in c.items() if k not in ['verification','result_path','score_result_path']}
            path=root/'marked'/f'{c["index"]:04d}.png';shutil.copy2(c['marked_image'],path)
            item['marked_image']=str(path);cases.append(item);by[(r['name'],c['method'])]=item
            shutil.copy2(path,root/'final/marked'/path.name)
            score=source/'scoring_results'/f'{c["index"]:04d}.json'
            assert read(score)['answer']==c['scores'];shutil.copy2(score,root/'final/scoring_results'/score.name)
    queries=[]
    presence=('The task setup guarantees that the requested object is physically present in this scene. It may be occluded or outside a particular camera, and it may be outside all proposed rectangles.' if a.presence_prior=='known_in_scene' else 'Do not assume the requested object is visible or present without visual evidence.')
    for (scene,role),local in sorted(grouped.items()):
        if {r['view'] for r in local}!=set(VIEWS) or len(local)!=3:raise ValueError('Missing/duplicate synchronized camera')
        local=sorted(local,key=lambda r:VIEWS.index(r['view']))
        if len({(r['domain'],r['episode'],r['frame'],r['target']) for r in local})!=1:raise ValueError('Unsynchronized input')
        for method in METHODS:
            cs=[by[(r['name'],method)] for r in local]
            requirements=[c['verification_contract'] for c in cs]
            if any(x!=requirements[0] for x in requirements):raise ValueError('Mismatched object descriptions')
            requirement=requirements[0]
            if set(requirement)!={'category','colors'}:raise ValueError('Only category and colors allowed')
            prompt=PROMPT.format(**dict(requirement,colors=json.dumps(requirement['colors']),presence=presence))
            images=[dict(view=c['view'],image=c['marked_image'],sha256=c['marked_sha256'],candidate_index=c['index']) for c in cs]
            queries.append(dict(index=len(queries),name=scene+'_'+role+'_'+method,scene=scene,role=role,method=method,images=images,prompt=prompt,input_sha256=digest([(i['view'],i['sha256']) for i in images]),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
    for f,data in [('cases.json',cases),('groups.json',groups),('queries.json',queries)]:save(root/f,data)
    cfg=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=str(source),source_sha256=sha(source/'ranked_candidates.json'),model=read(source/'experiment.json')['model'],gpus=a.gpus.split(','),prompt_version=PROMPT_VERSION,presence_prior=a.presence_prior,queries=len(queries),candidate_count=len(cases),target_views=len(groups),target_frames=len(grouped),native_images_per_call=3,max_new_tokens=512,retry_max_new_tokens=768,selection='Keep yes candidates, maximize unchanged category/color score; stable method-order tie',reference_images=False,same_time_context=True,numeric_threshold=None,approved_for_training=False,scope='Only verification changed; boxes, full images, category/color descriptions, numeric scores and model remain fixed',manifest_hashes={f:sha(root/f) for f in ['queries.json','groups.json','cases.json']},runtime_sha256=sha(root/'runtime/run.py'),git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    save(root/'experiment.json',cfg);save(root/'status.json',dict(phase='PREPARED',completed=0,total=len(queries),candidate_count=len(cases)))
    return root


def valid(root,q):
    p=root/'results'/f'{q["index"]:04d}.json'
    if not p.exists():return None
    r=read(p)
    if r['input_sha256']!=q['input_sha256'] or r['prompt_sha256']!=q['prompt_sha256']:raise ValueError('Result input mismatch')
    answer,error=parse(r['raw'])
    return r if error is None and answer==r['answer'] else None


def worker(root,index,workers):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    cfg=read(root/'experiment.json');assigned=read(root/'queries.json')[index::workers];pending=[q for q in assigned if not valid(root,q)]
    if not pending:return
    torch.manual_seed(0)
    processor=AutoProcessor.from_pretrained(cfg['model'],local_files_only=True)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(cfg['model'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    save(root/f'worker_{index}_environment.json',dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'],python=sys.version,model=cfg['model'],do_sample=False,enable_thinking=False))
    for q in pending:
        content=[]
        for n,item in enumerate(q['images']):
            if sha(item['image'])!=item['sha256']:raise ValueError('Image changed')
            with Image.open(item['image']) as im:im=im.convert('RGB')
            content.extend([dict(type='text',text=f'Image {n+1}: {item["view"]}'),dict(type='image',image=im)])
        content.append(dict(type='text',text=q['prompt']))
        inputs=processor.apply_chat_template([[dict(role='user',content=content)]],tokenize=True,padding=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to(model.device)
        if inputs['image_grid_thw'].shape[0]!=3:raise ValueError('Processor did not receive exactly three images')
        for limit in [cfg['max_new_tokens'],cfg['retry_max_new_tokens']]:
            torch.cuda.synchronize();start=time.monotonic()
            with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=limit,do_sample=False)
            torch.cuda.synchronize();raw=processor.tokenizer.decode(output[0,inputs['input_ids'].shape[-1]:],skip_special_tokens=True).strip();answer,error=parse(raw)
            result=dict(index=q['index'],name=q['name'],input_sha256=q['input_sha256'],prompt_sha256=q['prompt_sha256'],raw=raw,answer=answer,parse_error=error,elapsed_sec=time.monotonic()-start,max_new_tokens=limit,input_tokens=inputs['input_ids'].shape[-1],generated_tokens=output.shape[-1]-inputs['input_ids'].shape[-1],image_grid_thw=inputs['image_grid_thw'].tolist(),finished_utc=datetime.now(timezone.utc).isoformat())
            save(root/'attempts'/f'{q["index"]:04d}_{limit}.json',result)
            if error is None:break
        save(root/'results'/f'{q["index"]:04d}.json',result)
        print(json.dumps(dict(index=q['index'],name=q['name'],error=error,decisions={v:x['decision'] for v,x in answer.items()} if answer else None),ensure_ascii=False),flush=True)


def collect(root):
    cases={c['index']:c for c in read(root/'cases.json')};verifications={};valid_count=0
    for q in read(root/'queries.json'):
        r=valid(root,q)
        if r is None:continue
        valid_count+=1
        for image in q['images']:
            idx=image['candidate_index'];answer=r['answer'][image['view']];verifications[idx]=answer
            save(root/'final/results'/f'{idx:04d}.json',dict(index=idx,answer=answer,raw=json.dumps(answer,ensure_ascii=False),parse_error=None,joint_result=str(root/'results'/f'{q["index"]:04d}.json'),joint_input_sha256=q['input_sha256'],view=image['view']))
    rows=[]
    for g in read(root/'groups.json'):
        cs=[dict(**cases[i],verification=verifications.get(i),result_path=f'results/{i:04d}.json',score_result_path=f'scoring_results/{i:04d}.json') for i in g['candidate_indices']]
        yes=[c for c in cs if c['verification'] and c['verification']['decision']=='yes']
        complete=all(c['verification'] for c in cs);chosen=max(yes,key=lambda c:c['scores']['overall_score']) if complete and yes else None
        state='INCOMPLETE' if not complete else ('KEPT' if chosen else ('UNCERTAIN' if any(c['verification']['decision']=='uncertain' for c in cs) else 'REJECTED'))
        ties=[c['index'] for c in yes if c['scores']['overall_score']==chosen['scores']['overall_score']] if chosen else []
        rows.append(dict(**g,candidates=cs,chosen=chosen,state=state,tied_indices=ties,final_box=chosen['box'] if chosen else None,accepted=bool(chosen) if complete else None,approved_for_training=False))
    out=root/'final';save(out/'ranked_candidates.json',rows)
    save(out/'cases.json',list(cases.values()));save(out/'groups.json',read(root/'groups.json'))
    summary=dict(target_views=len(rows),complete_groups=sum(r['state']!='INCOMPLETE' for r in rows),candidates=len(cases),valid_scores=len(cases),valid_verifications=len(verifications),view_states=dict(Counter(r['state'] for r in rows)),complete_joint_calls=valid_count,approved_for_training=False,numeric_threshold=None)
    save(out/'summary.json',summary);return summary


def supervise(root):
    cfg=read(root/'experiment.json')
    for f,h in cfg['manifest_hashes'].items():
        if sha(root/f)!=h:raise ValueError('Manifest changed')
    if sha(root/'runtime/run.py')!=cfg['runtime_sha256']:raise ValueError('Runtime changed')
    procs=[]
    for i,gpu in enumerate(cfg['gpus']):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
        with (root/'logs'/f'worker_{i}.log').open('a') as log:
            procs.append(subprocess.Popen([sys.executable,'-u',str(root/'runtime/run.py'),'--out',str(root),'--worker',str(i),'--workers',str(len(cfg['gpus']))],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env))
    save(root/'processes.json',dict(supervisor_pid=os.getpid(),pgid=os.getpgrp(),workers=[p.pid for p in procs]));last=0
    while True:
        exits=[p.poll() for p in procs];done=all(x is not None for x in exits)
        save(root/'status.json',dict(phase='COLLECTING' if done else 'THREE_VIEW_VERIFY',completed=len(list((root/'results').glob('*.json'))),total=cfg['queries'],worker_exit=exits,updated_utc=datetime.now(timezone.utc).isoformat()))
        if done or time.monotonic()-last>30:summary=collect(root);last=time.monotonic()
        if done:break
        time.sleep(3)
    assert sha(Path(cfg['source'])/'ranked_candidates.json')==cfg['source_sha256']
    ok=all(x==0 for x in exits) and summary['valid_verifications']==cfg['candidate_count']
    save(root/'status.json',dict(phase='DONE' if ok else 'FAILED',completed=summary['complete_joint_calls'],total=cfg['queries'],candidate_verdicts=summary['valid_verifications'],worker_exit=exits))
    save(root/'final/status.json',read(root/'status.json'))
    if not ok:raise RuntimeError('Missing/invalid joint responses; not treated as rejections')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--episodes',type=int,nargs='+');p.add_argument('--frames',type=int,nargs='+')
    p.add_argument('--presence-prior',choices=['known_in_scene','unknown'],default='known_in_scene')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();root=a.out.resolve()
    if a.worker is not None:worker(root,a.worker,a.workers)
    else:
        if not a.resume:
            if a.source is None:p.error('--source required')
            root=prepare(a)
        if not a.prepare_only:supervise(root)


if __name__=='__main__':main()

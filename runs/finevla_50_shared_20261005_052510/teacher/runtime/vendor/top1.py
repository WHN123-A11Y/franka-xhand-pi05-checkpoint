"""Score only Teacher, full DINO and Teacher-padded-DINO Top-1, then keep the best."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path):
    spec = importlib.util.spec_from_file_location('fullframe_rating', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge(raw):
    # Exactly one representative per method. No rank-2/3 proposal, reference gate
    # or IoU merge may replace these original Top-1 coordinates.
    proposals = [dict(box=raw['teacher_box'],source='teacher',rank=1,
                      detector_score=None,method='teacher',fallback=False)]
    for branch, method in (('full','dino'),('padded','teacher_dino')):
        result=raw['detections'][branch]
        if result['candidates']:
            top=max(result['candidates'],key=lambda c:c['detector_score'])
            if top['box'] != result['output_box']:
                raise ValueError(f'Top-1 does not match branch output: {raw["name"]}/{branch}')
            proposals.append(dict(box=top['box'],source=branch,rank=1,
                                  detector_score=top['detector_score'],method=method,fallback=False))
        elif branch=='padded':
            # Preserve the existing local-DINO fallback explicitly. An empty
            # full-image DINO has no candidate and never borrows Teacher's box.
            if result['output_box'] != raw['teacher_box'] or result['output_source'] != 'teacher_fallback':
                raise ValueError('Unknown empty local-DINO fallback')
            proposals.append(dict(box=raw['teacher_box'],source='teacher_fallback',rank=1,
                                  detector_score=None,method=method,fallback=True))
    for candidate in proposals:
        b = candidate['box']
        if not (len(b) == 4 and all(0 <= v <= 1 for v in b) and b[0] < b[2] and b[1] < b[3]):
            raise ValueError(f'Invalid candidate: {raw["name"]}: {b}')
    return proposals


def select(candidates):
    """Qwen overall-score maximum; equal scores keep Teacher/DINO/combined order."""
    if any(c.get('scores') is None for c in candidates):
        return None, []
    ranked = sorted(candidates, key=lambda c: -c['scores']['overall_score'])
    if not ranked:
        return None, []
    best = ranked[0]
    ties = [c['index'] for c in ranked if c['scores']['overall_score'] == best['scores']['overall_score']]
    return best, ties


CONTEXT_KEYS = ('name', 'scene', 'role', 'view', 'domain', 'episode', 'frame',
                'instruction', 'target', 'role_description', 'image', 'image_sha256')


def prepare(args):
    from PIL import Image, ImageDraw
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'experiment.json').exists():
        raise FileExistsError('Existing run: use --resume instead of preparing it again')
    for sub in ('runtime', 'marked', 'results', 'attempts', 'logs'):
        (root/sub).mkdir(exist_ok=True)
    helper = Path(__file__).resolve().parents[1]/'qwen_fullframe_box_scoring/run.py'
    sources = {'run.py': Path(__file__).resolve(), 'report.py': Path(__file__).with_name('report.py'), 'scorer.py': helper}
    source_hashes = {}
    for name, path in sources.items():
        shutil.copy2(path, root/'runtime'/name)
        source_hashes[name] = {'source': str(path), 'sha256': sha(path)}
    scorer = module(root/'runtime/scorer.py')
    template = scorer.PROMPT.replace('Original robot instruction: {instruction}\n', '')
    # Project only language/frame fields; historical chosen boxes, decisions, reference
    # similarities and manual review annotations never enter this new pipeline.
    contexts = [{k: c[k] for k in CONTEXT_KEYS} for c in read(args.context)
                if c['domain'] == args.domain
                and (args.episodes is None or c['episode'] in args.episodes)
                and (args.frames is None or c['frame'] in args.frames)]
    contexts.sort(key=lambda c: (c['episode'], c['frame'], c['role'], c['view']))
    if not contexts or len({c['name'] for c in contexts}) != len(contexts):
        raise ValueError('Empty or duplicate contexts')
    save(root/'contexts.json', contexts)
    raw = {}
    input_hashes = {str(args.context.resolve()): sha(args.context)}
    # Reuse exact full-frame marked model inputs where available, solely by image
    # hash and box geometry. This is an input-pixel cache, not a reference image for
    # visual comparison, and cannot add/remove/prioritize a candidate.
    marked_cache = {}
    if args.marked_input_cache:
        input_hashes[str(args.marked_input_cache.resolve())] = sha(args.marked_input_cache)
        for item in read(args.marked_input_cache):
            key = (item['image_sha256'], tuple(item['box']))
            if key in marked_cache and marked_cache[key]['marked_sha256'] != item['marked_sha256']:
                raise ValueError('Conflicting marked-input pixels for identical geometry')
            marked_cache[key] = {k:item[k] for k in ('marked_image','marked_sha256')}
    cached_marked = []
    for path in sorted(args.source.glob('shards/*/dino/results.json')):
        input_hashes[str(path.resolve())] = sha(path)
        for c in read(path)['cases']:
            if c['name'] in raw:
                raise ValueError('Duplicate raw DINO case')
            raw[c['name']] = c
    groups, cases, queries = [], [], []
    image_hashes = {}
    for context in contexts:
        original = raw[context['name']]
        for key in ('target', 'view', 'image_sha256'):
            if context[key] != original[key]:
                raise ValueError(f'Context mismatch: {key}: {context["name"]}')
        path = context['image']
        image_hashes.setdefault(path, sha(path))
        if image_hashes[path] != context['image_sha256']:
            raise ValueError('Source image changed')
        with Image.open(path) as image:
            image = image.convert('RGB')
        pool = merge(original)
        ids = []
        for candidate in pool:
            index = len(cases)
            ids.append(index)
            marked = image.copy()
            d = ImageDraw.Draw(marked)
            xy = [round(v*s) for v, s in zip(candidate['box'], [marked.width, marked.height]*2)]
            d.rectangle(xy, outline='black', width=3)
            if xy[2]-xy[0] > 2 and xy[3]-xy[1] > 2:
                d.rectangle((xy[0]+1, xy[1]+1, xy[2]-1, xy[3]-1), outline='white', width=1)
            dest = root/'marked'/f'{index:04d}.png'
            cached = marked_cache.get((context['image_sha256'], tuple(candidate['box'])))
            if cached:
                if sha(cached['marked_image']) != cached['marked_sha256']:
                    raise ValueError('Cached marked image changed')
                with Image.open(cached['marked_image']) as prior:
                    if prior.size != image.size:
                        raise ValueError('Cached marked input is not full frame')
                shutil.copy2(cached['marked_image'],dest)
                cached_marked.append(dict(index=index, **cached))
            else:
                marked.save(dest)
            case = dict(**context, **candidate, index=index, marked_image=str(dest), marked_sha256=sha(dest))
            cases.append(case)
            prompt = template.format(**{k: context[k] for k in ('target', 'role_description', 'view')})
            queries.append(dict(index=index, name=context['name'], image=str(dest), image_sha256=sha(dest),
                                prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        full = original['detections']['full']['candidates']
        groups.append(dict(**context, candidate_indices=ids, teacher_box=original['teacher_box'],
                           dino_top1=max(full, key=lambda c: c['detector_score']) if full else None))
    for filename, data in [('groups.json', groups), ('cases.json', cases), ('queries.json', queries)]:
        save(root/filename, data)
    config = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=str(args.source.resolve()),
                  context_source=str(args.context.resolve()), domain=args.domain, target_views=len(groups),
                  candidate_count=len(cases), model=args.model, gpus=args.gpus.split(','),
                  max_new_tokens=384, format_retry_max_new_tokens=768, source_hashes=source_hashes,
                  input_hashes=input_hashes, image_hashes=image_hashes, prompt_version=scorer.PROMPT_VERSION,
                  marked_input_cache=str(args.marked_input_cache.resolve()) if args.marked_input_cache else None,
                  selection='max Qwen overall_score among 3 method Top-1s; ties: Teacher, DINO, Teacher+DINO',
                  reference_images_used=False, acceptance_threshold=None, approved_for_training=False,
                  scope='Offline three method Top-1 comparison; no Student/action changes',
                  candidate_policy='Teacher Top-1, full DINO Top-1, Teacher+20% crop DINO Top-1; no IoU merging',
                  manifest_hashes={f:sha(root/f) for f in ('groups.json','cases.json','queries.json','contexts.json')},
                  git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    save(root/'experiment.json', config)
    save(root/'marked_input_cache.json',cached_marked)
    save(root/'status.json', dict(phase='PREPARED', completed=0, total=len(cases), target_views=len(groups)))
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>三个方法第一名评分准备完成</h1>'
        f'<p>{len(groups)} 组目标视角，{len(cases)} 个候选。完成后此页自动生成图片对比。</p><a href="status.json">查看进度</a>')
    return root


def valid_result(root, query, scorer):
    path = root/'results'/f'{query["index"]:04d}.json'
    if not path.exists():
        return None
    result = read(path)
    if result['image_sha256'] != query['image_sha256'] or result['prompt_sha256'] != query['prompt_sha256']:
        raise ValueError('Cached result input mismatch')
    answer, error, _ = scorer.parse(result['raw'])
    if error is None and answer == result['answer']:
        return result
    return None


def worker(root, worker_id, workers):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    cfg = read(root/'experiment.json')
    scorer = module(root/'runtime/scorer.py')
    assigned = read(root/'queries.json')[worker_id::workers]
    pending = [q for q in assigned if valid_result(root, q, scorer) is None]
    if not pending:
        return
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(cfg['model'], local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(cfg['model'], torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    save(root/f'worker_{worker_id}_environment.json', dict(torch=torch.__version__,transformers=transformers.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        python=sys.version,model=cfg['model'],do_sample=False,enable_thinking=False))
    for q in pending:
        if sha(q['image']) != q['image_sha256']:
            raise ValueError('Marked input changed')
        with Image.open(q['image']) as im:
            im = im.convert('RGB')
        messages = [[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':q['prompt']}]}]]
        inputs = processor.apply_chat_template(messages, tokenize=True, padding=True, add_generation_prompt=True,
                    enable_thinking=False, return_dict=True, return_tensors='pt').to(model.device)
        result = None
        for limit in (cfg['max_new_tokens'], cfg['format_retry_max_new_tokens']):
            torch.cuda.synchronize();start=time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=limit, do_sample=False)
            torch.cuda.synchronize()
            raw = processor.tokenizer.decode(generated[0,inputs['input_ids'].shape[-1]:],skip_special_tokens=True).strip()
            answer, error, wrapper = scorer.parse(raw)
            result = dict(index=q['index'],name=q['name'],raw=raw,answer=answer,parse_error=error,
                          image_sha256=q['image_sha256'],prompt_sha256=q['prompt_sha256'],
                          max_new_tokens=limit,elapsed_sec=time.monotonic()-start,json_fence_unwrapped=wrapper,
                          generated_tokens=generated.shape[-1]-inputs['input_ids'].shape[-1],
                          finished_utc=datetime.now(timezone.utc).isoformat())
            save(root/'attempts'/f'{q["index"]:04d}_{limit}_{time.time_ns()}.json',result)
            if error is None:
                break
        save(root/'results'/f'{q["index"]:04d}.json',result)
        print(json.dumps(dict(index=q['index'],name=q['name'],scores=result['answer'],error=result['parse_error']),ensure_ascii=False),flush=True)


def collect(root):
    scorer = module(root/'runtime/scorer.py')
    queries = read(root/'queries.json')
    results = {q['index']:valid_result(root,q,scorer) for q in queries}
    cases = {c['index']:c for c in read(root/'cases.json')}
    rows = []
    for group in read(root/'groups.json'):
        candidates = []
        for index in group['candidate_indices']:
            result = results[index]
            candidates.append(dict(**cases[index], scores=result['answer'] if result else None,
                                   result_path=f'results/{index:04d}.json'))
        chosen, ties = select(candidates)
        rows.append(dict(**group,candidates=candidates,chosen=chosen,tied_indices=ties,
                         state='SCORED_TOP_CANDIDATE' if chosen else 'SCORING_INCOMPLETE',
                         accepted=None,approved_for_training=False))
    save(root/'ranked_candidates.json',rows)
    summary = dict(target_views=len(rows),candidates=len(cases),valid_scores=sum(v is not None for v in results.values()),
                   complete_groups=sum(r['chosen'] is not None for r in rows),
                   top_score_counts=dict(Counter(r['chosen']['scores']['overall_score'] for r in rows if r['chosen'])),
                   tied_groups=sum(len(r['tied_indices'])>1 for r in rows),
                   highest_method_counts=dict(Counter(r['chosen']['method'] for r in rows if r['chosen'])),
                   method_candidate_counts=dict(Counter(c['method'] for c in cases.values())),
                   fallback_count=sum(c['fallback'] for c in cases.values()),
                   top_zero_groups=sum(r['chosen']['scores']['overall_score']==0 for r in rows if r['chosen']),
                   reference_images_used=False,acceptance_threshold=None,approved_for_training=False,
                   scope='Highest-rated candidate only, not an accepted training label or measured localization accuracy')
    save(root/'summary.json',summary)
    return summary


def hydrate_score_cache(root, cache):
    """Reuse only exact image+prompt+model results; record each original artifact."""
    cfg = read(root/'experiment.json');previous = read(cache/'experiment.json')
    for key in ('model','prompt_version','max_new_tokens','format_retry_max_new_tokens'):
        if cfg[key] != previous[key]:
            raise ValueError('Score cache configuration differs: '+key)
    if cfg['source_hashes']['scorer.py']['sha256'] != previous['source_hashes']['scorer.py']['sha256']:
        raise ValueError('Score schema/prompt helper changed')
    scorer = module(root/'runtime/scorer.py')
    old_queries = {(q['image_sha256'],q['prompt_sha256']):q for q in read(cache/'queries.json')}
    reused = []
    for q in read(root/'queries.json'):
        old = old_queries.get((q['image_sha256'],q['prompt_sha256']))
        if old is None:
            continue
        result = valid_result(cache,old,scorer)
        if result is None:
            continue
        source = cache/'results'/f'{old["index"]:04d}.json'
        copied = dict(result,index=q['index'],name=q['name'],reused_from=str(source),reused_sha256=sha(source))
        save(root/'results'/f'{q["index"]:04d}.json',copied)
        reused.append(dict(index=q['index'],source=str(source),source_sha256=sha(source)))
    save(root/'score_cache.json',dict(source=str(cache),reused=reused,exact_image_prompt_match=True))


def supervise(root):
    cfg = read(root/'experiment.json')
    for name, digest in cfg['manifest_hashes'].items():
        if sha(root/name) != digest:
            raise ValueError('Manifest changed: '+name)
    for name, entry in cfg['source_hashes'].items():
        if sha(root/'runtime'/name) != entry['sha256']:
            raise ValueError('Runtime changed: '+name)
    procs, streams = [], []
    for i, gpu in enumerate(cfg['gpus']):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
        log = (root/'logs'/f'worker_{i}.log').open('a');streams.append(log)
        p = subprocess.Popen([sys.executable,'-u',str(root/'runtime/run.py'),'--out',str(root),
             '--worker',str(i),'--workers',str(len(cfg['gpus']))],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        procs.append(p)
    save(root/'processes.json',dict(supervisor_pid=os.getpid(),worker_pids=[p.pid for p in procs],pgid=os.getpgrp()))
    last_report = 0
    while True:
        exits = [p.poll() for p in procs]
        completed = len(list((root/'results').glob('*.json')))
        done = all(v is not None for v in exits)
        save(root/'status.json',dict(phase='SCORING' if not done else 'RENDERING',completed=completed,
             total=cfg['candidate_count'],target_views=cfg['target_views'],worker_exit=exits,
             updated_utc=datetime.now(timezone.utc).isoformat()))
        if done or (completed and time.monotonic()-last_report > 60):
            collect(root)
            subprocess.run([sys.executable,str(root/'runtime/report.py'),'--run',str(root)],check=True)
            last_report=time.monotonic()
        if done:
            break
        time.sleep(3)
    for log in streams:
        log.close()
    summary = collect(root)
    for path, digest in cfg['input_hashes'].items():
        if sha(path) != digest:
            raise ValueError('Source artifact changed: '+path)
    success = all(v == 0 for v in exits) and summary['valid_scores'] == cfg['candidate_count']
    save(root/'status.json',dict(phase='DONE' if success else 'INCOMPLETE',completed=summary['valid_scores'],
          total=cfg['candidate_count'],target_views=cfg['target_views'],worker_exit=exits))
    if not success:
        raise RuntimeError('Incomplete model scores; inspect logs. Missing scores are not replaced.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--context',type=Path)
    p.add_argument('--marked-input-cache',type=Path)
    p.add_argument('--score-cache',type=Path)
    p.add_argument('--domain',default='Clean')
    p.add_argument('--episodes',type=int,nargs='+')
    p.add_argument('--frames',type=int,nargs='+')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--model',default='/data/adminad/models/Qwen3.5-4B')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--worker',type=int)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();root=a.out.resolve()
    if a.worker is not None:
        worker(root,a.worker,a.workers)
    else:
        if not a.resume:
            if a.source is None or a.context is None:
                p.error('--source and --context required for a new run')
            root=prepare(a)
            if a.score_cache:
                hydrate_score_cache(root,a.score_cache.resolve())
        if not a.prepare_only:
            supervise(root)


if __name__ == '__main__':
    main()

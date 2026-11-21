"""Offline single-episode multiview smoke; snapshot/adapt uploaded Teacher paths only."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

VERSION_DIR = Path(__file__).resolve().parent
ROOT = VERSION_DIR.parents[2]
SOURCE = VERSION_DIR / 'source'
REL = Path('experiments/delta_grounding')
FILES = [
    'success_teacher_multiview_v1.py', 'teacher_crossview_bt.py',
    'teacher_stage2_fast_local_bradley_terry_v1.py',
    'test_generic_hierarchical_semantic_screen.py', 'test_qwen_auto_proposal_teacher.py',
    'FROZEN_TEACHER_1691_V1_20260816/teacher_stage1_1691_phaseaware_twoschedule.py',
    'LOCKED_MULTIVIEW_TEACHER_BT_V1_20260903/teacher_per_view.py',
]
VIEWS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']


def replace_constants(text, replacements):
    lines = text.splitlines(keepends=True)
    edits = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in replacements:
                edits.append((node.lineno-1, node.end_lineno, f'{name} = Path({str(replacements[name])!r})\n'))
    for start, end, replacement in reversed(edits):
        lines[start:end] = [replacement]
    return ''.join(lines)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', choices=['RoboTwin-Clean','RoboTwin-Randomized'], default='RoboTwin-Clean')
    p.add_argument('--task', default='adjust_bottle')
    p.add_argument('--task-index', type=int, default=8)
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--model', type=Path, default=Path('/data/adminad/models/Qwen3.5-4B'))
    p.add_argument('--language-results', type=Path, default=VERSION_DIR/'fixtures/language_results.jsonl')
    args=p.parse_args()
    import av
    import pyarrow.parquet as pq
    import torch
    dataset=Path('/data/adminad/datasets')/args.dataset/args.task
    tasks=[json.loads(x) for x in (dataset/'meta/tasks.jsonl').read_text().splitlines()]
    instruction=next(r['task'] for r in tasks if r['task_index']==args.task_index)
    episodes=[json.loads(x) for x in (dataset/'meta/episodes.jsonl').read_text().splitlines()]
    episode=next(r for r in episodes if instruction in r['tasks'])
    index=episode['episode_index']
    if not 0 <= args.frame < episode['length']: raise ValueError('Frame outside episode')
    info=json.loads((dataset/'meta/info.json').read_text())
    chunk=index//info['chunks_size']
    table=pq.read_table(dataset/info['data_path'].format(episode_chunk=chunk,episode_index=index),
                        columns=['task_index','frame_index','timestamp']).to_pydict()
    if int(table['task_index'][args.frame]) != args.task_index or int(table['frame_index'][args.frame]) != args.frame:
        raise ValueError('Frame does not match requested task/index')
    language=next(r for r in [json.loads(x) for x in args.language_results.read_text().splitlines()]
                  if r['dataset']==args.dataset and r['task']==args.task and r['task_index']==args.task_index)
    if language['status']!='parsed' or language['instruction']!=instruction: raise ValueError('Unverified language plan')
    step=language['plan']['steps'][0]
    if len(step['targets'])!=1 or step['locations']: raise ValueError('Smoke requires one appearance-only target')
    entity=next(e for e in language['plan']['entities'] if e['id']==step['targets'][0]['entity_id'])
    target=entity['mention']
    out=ROOT/'runs'/('grounding_teacher_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    print(f'Output: {out}',flush=True)
    vendor=out/'teacher_source'
    changes=[]
    for name in FILES:
        original=SOURCE/REL/name
        destination=vendor/REL/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        data=original.read_text()
        adapted=replace_constants(data,{'ROOT':vendor,'PROJECT':vendor,'MODEL_PATH':args.model})
        destination.write_text(adapted)
        changes.append(dict(source=str(original),sha256=hashlib.sha256(data.encode()).hexdigest(),
                            adapted_sha256=hashlib.sha256(adapted.encode()).hexdigest()))
    image_dir=out/'images';image_dir.mkdir()
    group=f'{args.dataset}_{args.task}_ep{index}_frame{args.frame}_{entity["id"]}'
    rows=[]
    for view in VIEWS:
        key=f'observation.images.{view}'
        video=dataset/info['video_path'].format(episode_chunk=chunk,episode_index=index,video_key=key)
        image=image_dir/f'{view}.png'
        with av.open(str(video)) as container:
            stream=container.streams.video[0]
            if abs(float(stream.average_rate)-info['fps'])>0.01: raise ValueError('Video FPS mismatch')
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index == args.frame:
                    frame.to_image().save(image)
                    break
            else:
                raise RuntimeError(f'Cannot decode frame {args.frame}: {video}')
        rows.append(dict(sample_id=view,group_id=group,view_id=view,episode_index=index,
                         frame_index=args.frame,image_path=str(image),target_text=instruction,
                         grounding_target_text=target))
    manifest=out/'manifest.tsv'
    with manifest.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    (out/'config.json').write_text(json.dumps(dict(dataset=args.dataset,task=args.task,episode=index,
         task_index=args.task_index,frame=args.frame,timestamp=table['timestamp'][args.frame],
         instruction=instruction,target=target,language_source=str(args.language_results),model=str(args.model),
         adaptations=changes,notes='Original candidate/ranking algorithm; path-only adaptation. No resume, no visibility guarantee.'),indent=2))
    sys.path.insert(0,str(SOURCE))
    sys.path.insert(0,str(vendor/REL))
    entry=vendor/REL/'success_teacher_multiview_v1.py'
    spec=importlib.util.spec_from_file_location('smoke_multiview',entry)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.argv=[str(entry),'--manifest',str(manifest),'--output-root',str(out/'teacher_output'),'--limit-groups','1']
    module.main()
    result=out/'teacher_output'/group/'multiview_results_bt_v1.pt'
    if not result.exists(): raise RuntimeError('Teacher failed: final result not written; inspect log')
    print(f'FINAL_RESULT={result}',flush=True)


if __name__=='__main__': main()

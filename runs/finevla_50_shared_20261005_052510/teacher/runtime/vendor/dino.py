"""Compare Teacher Top1 -> cropped Grounding DINO against matched full-image DINO.

No identity gate, rejection, or cross-view winner. Empty local detections retain
Teacher Top1. Frozen Teacher artifacts are read-only; all outputs go under runs/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
VIEWS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
MODEL = Path('/data/adminad/models/grounding-dino-base')
MODES = ('full', 'strict', 'padded')
COLORS = ('#ff6577', '#3dd9ff', '#75f08f', '#ffd166')


def checked_box(values):
    box = [float(v) for v in values]
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError(f'Invalid box: {box}')
    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError(f'Expected normalized xyxy box: {box}')
    return box


def crop_bounds(box, size, padding):
    width, height = size
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return (max(0, math.floor((x1 - dx) * width)),
            max(0, math.floor((y1 - dy) * height)),
            min(width, math.ceil((x2 + dx) * width)),
            min(height, math.ceil((y2 + dy) * height)))


def map_box_to_original(values, bounds, size):
    x0, y0, x3, y3 = bounds
    width, height = size
    cw, ch = x3 - x0, y3 - y0
    x1, y1, x2, y2 = [float(v) for v in values]
    local = [min(cw, max(0, x1)), min(ch, max(0, y1)),
             min(cw, max(0, x2)), min(ch, max(0, y2))]
    if local[0] >= local[2] or local[1] >= local[3]:
        return None
    return checked_box([(x0 + local[0]) / width, (y0 + local[1]) / height,
                        (x0 + local[2]) / width, (y0 + local[3]) / height])


def geometry_check():
    assert crop_bounds([.25, .25, .75, .75], (640, 480), 0) == (160, 120, 480, 360)
    assert crop_bounds([0, .25, .5, .75], (640, 480), .2) == (0, 72, 384, 408)
    assert map_box_to_original([0, 0, 320, 240], (160, 120, 480, 360), (640, 480)) == [.25, .25, .75, .75]
    assert map_box_to_original([-10, -10, 330, 250], (160, 120, 480, 360), (640, 480)) == [.25, .25, .75, .75]


def prepare(args):
    import torch
    run = args.teacher_run.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else run / 'manifest.tsv'
    rows = list(csv.DictReader(manifest_path.open(), delimiter='\t'))
    selected = [r for r in rows if int(r['episode_index']) in args.episodes
                and int(r['frame_index']) in args.frames]
    cache, records, seen = {}, [], set()
    for row in selected:
        group, view = row['group_id'], row['view_id']
        if view not in VIEWS or not re.fullmatch(r'[A-Za-z0-9_.-]+', group):
            raise ValueError(f'Invalid group/view: {group}/{view}')
        if (group, view) in seen:
            raise ValueError(f'Duplicate group/view: {group}/{view}')
        seen.add((group, view))
        if group not in cache:
            path = run / 'teacher_output' / group / 'multiview_results_bt_v1.pt'
            result = torch.load(path, map_location='cpu', weights_only=False)
            if result['group_id'] != group:
                raise ValueError(f'Teacher group mismatch: {path}')
            cache[group] = (path, result)
        path, result = cache[group]
        teacher_view = result['views'][view]
        image = Path(row['image_path']).resolve()
        if not image.is_file() or image != Path(teacher_view['image_path']).resolve():
            raise ValueError(f'Teacher image mismatch: {group}/{view}')
        target = result['target_text']
        if target != row['grounding_target_text'] or not target.strip():
            raise ValueError(f'Teacher target mismatch: {group}/{view}')
        with Image.open(image) as im:
            size = list(im.size)
        teacher = teacher_view['topk'][0]
        records.append({'name': f'{group}_{view}', 'group_id': group, 'view': view,
                        'image': str(image), 'image_size': size, 'target': target,
                        'prompt': target.strip().lower().rstrip('.') + '.',
                        'teacher_box': checked_box(teacher['box']),
                        'teacher_rank': int(teacher['rank']),
                        'teacher_result': str(path),
                        'teacher_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()})
    if not records:
        raise ValueError('No matching Teacher records')
    for group in cache:
        if {r['view'] for r in records if r['group_id'] == group} != set(VIEWS):
            raise ValueError(f'Missing synchronized view for {group}')
    return records


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def render_row(case):
    with Image.open(case['image']) as source:
        source_image = source.convert('RGB')
    width, height = source_image.size
    canvas = Image.new('RGB', (4 * width, height + 72), '#20252b')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    entries = [('A Teacher Top1', case['teacher_box'], None, 'teacher')]
    for mode, title in zip(MODES, ('B DINO full image', 'C Teacher crop -> DINO', 'D Teacher +20% -> DINO')):
        result = case['detections'][mode]
        entries.append((title, result['output_box'], result['top1_score'], result['output_source']))
    for index, (title, box, score, source) in enumerate(entries):
        panel = source_image.copy()
        draw = ImageDraw.Draw(panel)
        if box is not None:
            draw.rectangle(tuple(round(v * scale) for v, scale in zip(box, (width, height, width, height))),
                           outline=COLORS[index], width=5)
        canvas.paste(panel, (index * width, 72))
        draw = ImageDraw.Draw(canvas)
        draw.text((index * width + 8, 6), f'{case["view"]} | {title}', fill=COLORS[index], font=font)
        detail = source if score is None else f'{source} | DINO score={score:.3f}'
        draw.text((index * width + 8, 30), detail, fill='white', font=font)
        draw.text((index * width + 8, 50), case['target'][:65], fill='#dddddd', font=font)
    return canvas


def render_groups(cases, out):
    grouped = {}
    for case in cases:
        grouped.setdefault(case['group_id'], {})[case['view']] = case
    links = []
    for group, views in grouped.items():
        if set(views) != set(VIEWS):
            continue
        rows = [render_row(views[v]) for v in VIEWS]
        image = Image.new('RGB', (rows[0].width, sum(r.height for r in rows)), '#20252b')
        y = 0
        for row in rows:
            image.paste(row, (0, y))
            y += row.height
        image.save(out / 'visualizations' / f'{group}.jpg', quality=93)
        links.append(f'<section><h2>{html.escape(group)}</h2><img src="{html.escape(group)}.jpg"></section>')
    (out / 'visualizations' / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Teacher first then DINO</title>'
        '<style>body{background:#161b22;color:white;font:16px sans-serif}img{width:100%}</style>'
        '<h1>A Teacher | B DINO full | C Teacher crop | D Teacher crop +20%</h1>'
        + ''.join(links))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='Optional source manifest when Teacher outputs were resumed into another directory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, nargs='+', default=[0])
    parser.add_argument('--frames', type=int, nargs='+', default=[0])
    parser.add_argument('--padding', type=float, default=.2, choices=[.2])
    parser.add_argument('--box-threshold', type=float, default=.12)
    parser.add_argument('--text-threshold', type=float, default=.12)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.box_threshold <= 1 or not 0 <= args.text_threshold <= 1:
        raise ValueError('Thresholds must be in [0,1]')
    if not args.out.resolve().is_relative_to((ROOT / 'runs').resolve()) or args.out.resolve() == (ROOT / 'runs').resolve():
        raise ValueError('Output must be under runs/<run_id>')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    geometry_check()
    records = prepare(args)
    print(f'PREFLIGHT_PASS cases={len(records)} groups={len(records)//3} GEOMETRY_PASS', flush=True)
    if args.preflight:
        return
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ('crops', 'visualizations'):
        (args.out / name).mkdir()
    protocol = {'method': 'Teacher per-view Top1 -> cropped DINO; no rejection or cross-view selection',
                'modes': ['Teacher unchanged', 'DINO full image with matched Teacher text',
                          'DINO within exact Teacher crop', 'DINO within Teacher crop expanded 20% each side'],
                'empty_local_detection': 'retain original Teacher box; explicitly recorded as fallback',
                'selection': 'DINO maximum detector score within each input crop',
                'padding_each_side': args.padding, 'box_threshold': args.box_threshold,
                'text_threshold': args.text_threshold, 'teacher_run': str(args.teacher_run.resolve()),
                'manifest': str(args.manifest.resolve() if args.manifest else args.teacher_run.resolve() / 'manifest.tsv'),
                'model': str(MODEL), 'episodes': args.episodes, 'frames': args.frames,
                'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__,
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'warning': 'Scores and boxes are exploratory; no automatic correctness labels or action use.'}
    dump(args.out / 'protocol.json', protocol)
    dump(args.out / 'input_manifest.json', records)
    dump(args.out / 'status.json', {'phase': 'MODEL_LOAD', 'completed': 0, 'total': len(records)})
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL, local_files_only=True).to('cuda').eval()
    report = {'protocol': protocol, 'cases': []}
    for case in records:
        with Image.open(case['image']) as source:
            image = source.convert('RGB')
        case = dict(case)
        case['detections'] = {}
        for mode in MODES:
            bounds = ((0, 0, *image.size) if mode == 'full' else
                      crop_bounds(case['teacher_box'], image.size, 0 if mode == 'strict' else args.padding))
            crop = image.crop(bounds)
            crop_file = None
            if mode != 'full':
                crop_file = f'crops/{case["name"]}_{mode}.png'
                crop.save(args.out / crop_file)
            inputs = processor(images=crop, text=case['prompt'], return_tensors='pt').to(model.device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(**inputs)
            detection = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=args.box_threshold, text_threshold=args.text_threshold,
                target_sizes=[crop.size[::-1]])[0]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            candidates = []
            for box, score, label in zip(detection['boxes'], detection['scores'], detection['text_labels']):
                mapped = map_box_to_original(box.tolist(), bounds, image.size)
                if mapped is not None:
                    candidates.append({'box': mapped, 'crop_box_pixels': box.tolist(),
                                       'detector_score': float(score), 'detector_label': str(label)})
            candidates.sort(key=lambda c: c['detector_score'], reverse=True)
            if candidates:
                output_box, output_source = candidates[0]['box'], 'dino'
            elif mode != 'full':
                output_box, output_source = case['teacher_box'], 'teacher_fallback'
            else:
                output_box, output_source = None, 'no_detection'
            case['detections'][mode] = {'bounds_pixels': list(bounds), 'crop_image': crop_file,
                'candidate_count': len(candidates), 'candidates': candidates,
                'output_box': output_box, 'output_source': output_source,
                'top1_score': candidates[0]['detector_score'] if candidates else None,
                'forward_and_postprocess_sec': seconds}
        row = render_row(case)
        row.save(args.out / 'visualizations' / f'{case["name"]}.jpg', quality=93)
        report['cases'].append(case)
        dump(args.out / 'results.json', report)
        dump(args.out / 'status.json', {'phase': 'DETECT', 'completed': len(report['cases']), 'total': len(records)})
        print(case['name'] + ': ' + ', '.join(f'{m}={case["detections"][m]["candidate_count"]}' for m in MODES), flush=True)
    render_groups(report['cases'], args.out)
    dump(args.out / 'status.json', {'phase': 'DONE', 'completed': len(records), 'total': len(records)})
    print(f'OUTPUT={args.out.resolve()}', flush=True)


if __name__ == '__main__':
    main()

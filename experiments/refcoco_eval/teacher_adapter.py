"""GT-free RefCOCO adapters reusing the Clean-50 Teacher's frozen local code.

Only image_path, raw expression and an opaque sample_id enter prediction. No
annotation/dataset readers live here. All returned boxes are ORIGINAL PIXEL xyxy.
"""
from __future__ import annotations
import contextlib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path('/data/adminad/models/Qwen3.5-4B')
DETECTOR = Path('/data/adminad/models/grounding-dino-base')
FROZEN = ROOT / 'validated_scripts/grounding_teacher_multiview/v1_20260928'
CLEAN = ROOT / 'runs/finevla_50_shared_20261005_052510/teacher'
VENDOR = CLEAN / 'runtime/vendor'
DIRECT_PROMPT = '''Locate exactly the single physical object referred to by this expression in the image:
{expression}
Return ONLY JSON {{"bbox_xyxy_1000":[x_min,y_min,x_max,y_max]}}.
Coordinates MUST be normalized to 0..1000 relative to the complete supplied image: left/top = 0 and right/bottom = 1000. Give a tight rectangle around the whole referred object, not a related reference object. Do not give multiple boxes. If no object can be located return {{"bbox_xyxy_1000":null}}.
'''
HEAD_PROMPT = '''Read this referring expression and extract its TARGET noun span (the object being referred to, not another reference object):
{expression}
Return ONLY JSON {{"head":"exact nonempty noun substring from expression","target_phrase":"exact substring describing target including its directly attached color words"}}.
Do not invent a category, color, attribute or invisible fact. Keep another reference object's color out of target_phrase. If no explicit target noun exists, use an exact noun/pronoun span from the expression without resolving it using an image.
'''
VERIFY_PROMPT = '''This is ONE complete image with a thin black/white rectangle marking a candidate region. The rectangle is an annotation, not part of the object.
Requested object category: {category}
Requested colors: {colors}
Decide whether THIS rectangle contains a visible part of the requested physical object. Use only visible evidence in this one image. A target outside the rectangle does not validate it. A loose or partly occluded box can match if the object inside is identifiable. All proposals may be wrong.
Evaluate ONLY object category and stated colors. An empty color list means no color requirement. A color can belong to a visible component. Do not add material, texture, pattern, brand, size, detailed shape, action, pose or location requirements.
Return ONLY JSON with exactly four string fields:
{{"observed_object":"actual boxed object or background","observed_color":"visible color or unclear","reason":"brief local visible evidence","decision":"yes|no|uncertain"}}
'''


def weight_fingerprints(directory):
    paths = sorted(Path(directory).glob('*.safetensors')) + sorted(Path(directory).glob('*.json'))
    result = {}
    for path in paths:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                h.update(block)
        result[path.name] = h.hexdigest()
    return result


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def json_object(raw):
    raw = raw.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = '\n'.join(raw.splitlines()[1:-1])
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value


def valid_box(box, width, height):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in box)
            and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class QwenRuntime:
    """Shared model so Direct and Teacher use EXACTLY the same local weights/settings."""
    def __init__(self, model_path=MODEL, device='cuda'):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.model_path = str(Path(model_path).resolve())
        torch.manual_seed(42)
        torch.set_num_threads(4)
        start = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_path, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to(device).eval()
        synchronize()
        self.load_seconds = time.perf_counter() - start
        self.calls = 0
        self.trace = []
        self.metadata = dict(model=self.model_path, precision='bfloat16', attention='sdpa',
                             do_sample=False, enable_thinking=False, batch_size=1,
                             image_processor=self.processor.image_processor.to_dict(),
                             torch=torch.__version__, transformers=transformers.__version__,
                             cpu_threads=torch.get_num_threads(), device=str(device), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
                             load_seconds=self.load_seconds, cross_sample_prediction_cache=False,
                             model_files_sha256=weight_fingerprints(self.model_path),
                             adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def generate(self, prompt, image=None, limit=384, stage='generate'):
        import torch
        content = []
        if image is not None:
            content.append(dict(type='image', image=image))
        content.append(dict(type='text', text=prompt))
        inputs = self.processor.apply_chat_template(
            [dict(role='user', content=content)], tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors='pt').to(self.model.device)
        synchronize()
        start = time.perf_counter()
        self.calls += 1
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=limit, do_sample=False)
        synchronize()
        raw = self.processor.tokenizer.decode(output[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        self.trace.append(dict(stage=stage, prompt=prompt, raw=raw, max_new_tokens=limit,
                               generated_tokens=int(output.shape[-1]-inputs['input_ids'].shape[-1]),
                               image_grid_thw=inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else None,
                               seconds=time.perf_counter()-start))
        return raw

    def warmup(self):
        from PIL import Image
        self.generate('Describe the visible colors briefly.', Image.new('RGB', (256,256), 'gray'), 16, 'warmup')
        self.trace.clear()
        self.calls = 0


def expression_contract(runtime, expression):
    """Expression-only adaptation of existing entity-head/category/color projection.

    Explicitly not a semantic ground-truth parser. No image or annotation used.
    Relation/shape information is retained by proposal search but NOT verifier.
    """
    projection = module(VENDOR/'projection.py', 'refcoco_retained_projection')
    first = json_object(runtime.generate(HEAD_PROMPT.format(expression=expression), limit=192, stage='target_head'))
    if set(first) != {'head','target_phrase'}:
        raise ValueError('Invalid target-head schema')
    for key in first:
        if not isinstance(first[key], str) or not first[key].strip() or first[key] not in expression:
            raise ValueError('Target head/phrase is not an exact expression substring')
    if first['head'] not in first['target_phrase']:
        raise ValueError('Target phrase must contain target head')
    raw = runtime.generate(projection.PROMPT.format(head=first['head'], role=expression), limit=64, stage='category_projection')
    category = json_object(raw)
    if set(category) != {'category'} or not isinstance(category['category'],str) or not category['category'].strip():
        raise ValueError('Invalid category schema')
    return dict(category=category['category'], colors=projection.colors_in(first['target_phrase']),
                head=first['head'], target_phrase=first['target_phrase'],
                ignored_by_verifier=['relations','location','size','shape','material','pattern','action'])


class QwenDirectAdapter:
    def __init__(self, runtime):
        self.runtime = runtime
        self.metadata = dict(runtime.metadata, method='qwen_direct', prompt=DIRECT_PROMPT,
                             coordinates='explicit requested normalized xyxy 0..1000; one bbox_xyxy_1000 dict or one native bbox_2d item; no inferred scale',
                             max_new_tokens=384, format_retries=0)

    def predict(self, image_path, expression, sample_id):
        from PIL import Image
        synchronize()
        start = time.perf_counter()
        self.runtime.trace = []
        calls = self.runtime.calls
        with Image.open(image_path) as im:
            image = im.convert('RGB')
        box = None
        failure = None
        raw = self.runtime.generate(DIRECT_PROMPT.format(expression=expression), image, 384, 'direct_box')
        try:
            payload = raw.strip()
            if payload.startswith('```') and payload.endswith('```'):
                payload = '\n'.join(payload.splitlines()[1:-1])
            obj = json.loads(payload)
            if isinstance(obj, list) and len(obj) == 1:
                obj = obj[0]
            if isinstance(obj, dict) and set(obj) == {'bbox_xyxy_1000'}:
                norm = obj['bbox_xyxy_1000']
            elif isinstance(obj, dict) and 'bbox_2d' in obj and set(obj) <= {'bbox_2d', 'label'}:
                norm = obj['bbox_2d']
            else:
                raise ValueError('Expected exactly one direct box')
            if norm is None:
                failure = 'no_box'
            elif not valid_box(norm, 1000,1000):
                failure = 'invalid_box'
            else:
                box = [float(v)*s/1000 for v,s in zip(norm, (image.width,image.height)*2)]
        except (ValueError, TypeError):
            failure = 'parse_error'
        synchronize()
        return dict(sample_id=sample_id, box_xyxy=box, valid=box is not None, failure_reason=failure,
                    elapsed_sec=time.perf_counter()-start, qwen_calls=self.runtime.calls-calls,
                    detector_calls=0, trace=list(self.runtime.trace))


class TeacherAdapter:
    def __init__(self, runtime, work_dir):
        self.runtime = runtime
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        adapter = module(FROZEN/'smoke.py', 'refcoco_teacher_snapshot_adapter')
        vendor = self.work_dir/'teacher_source'
        hashes = {}
        for name in adapter.FILES:
            src = FROZEN/'source'/adapter.REL/name
            dst = vendor/adapter.REL/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            original = src.read_text()
            adapted = adapter.replace_constants(original, dict(ROOT=vendor, PROJECT=vendor, MODEL_PATH=Path(runtime.model_path)))
            dst.write_text(adapted)
            hashes[str(src)] = hashlib.sha256(src.read_bytes()).hexdigest()
        sys.path.insert(0, str(FROZEN/'source'))
        sys.path.insert(0, str(vendor/adapter.REL))
        self.core = module(vendor/adapter.REL/'LOCKED_MULTIVIEW_TEACHER_BT_V1_20260903/teacher_per_view.py', 'refcoco_per_view_core')
        self.core._SHARED_PROCESSOR = runtime.processor
        self.core._SHARED_MODEL = runtime.model
        self.dino = module(VENDOR/'dino.py', 'refcoco_retained_dino')
        self.scorer = module(VENDOR/'scorer.py', 'refcoco_retained_scorer')
        for source_name in ('dino.py', 'scorer.py', 'projection.py', 'joint.py', 'top1.py'):
            source = VENDOR/source_name
            hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        start = time.perf_counter()
        self.dino_processor = AutoProcessor.from_pretrained(DETECTOR, local_files_only=True, use_fast=False)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR, local_files_only=True).to(runtime.model.device).eval()
        synchronize()
        self.detector_load_seconds = time.perf_counter()-start
        self.metadata = dict(runtime.metadata, method='our_teacher_single_image', detector=str(DETECTOR),
            detector_files_sha256=weight_fingerprints(DETECTOR),
            detector_precision='float32', detector_load_seconds=self.detector_load_seconds,
            detector_threshold=.12, detector_text_threshold=.12, crop_padding=.2,
            retained=['coarse fixed grids','two schedules','local search','pre-BT screening',
                      'cached local Bradley-Terry ranking','full-image DINO','predicted padded-crop DINO',
                      'category/color scores','yes-only max score stable method tie order'],
            modified=['expression-only head extraction before retained category projection',
                      'single-image visible-evidence final verification'],
            disabled=['cross-camera BT','three-view contextual verification','robot scene presence prior'],
            verifier_ignores=['relations','shape','size','location','material','patterns'],
            sources_sha256=hashes, direct_prompt=DIRECT_PROMPT, verification_prompt=VERIFY_PROMPT,
            head_prompt=HEAD_PROMPT, category_prompt=(VENDOR/'projection.py').read_text(),
            scorer_prompt=self.scorer.PROMPT, cache='within-sample vision KV cache retained; no cross-sample outputs/features',
            timing='includes original target-dependent and first-use core warmups; model loads excluded; CUDA synchronized')

    def _detect(self, image, expression, bounds):
        import torch
        crop = image.crop(bounds)
        inputs = self.dino_processor(images=crop, text=expression.strip().lower().rstrip('.')+'.',
                                     return_tensors='pt').to(self.dino_model.device)
        with torch.inference_mode():
            out = self.dino_model(**inputs)
        det = self.dino_processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=.12, text_threshold=.12, target_sizes=[crop.size[::-1]])[0]
        rows = []
        for box, score, label in zip(det['boxes'],det['scores'],det['text_labels']):
            mapped = self.dino.map_box_to_original(box.tolist(), bounds, image.size)
            if mapped is not None:
                rows.append(dict(box=mapped, detector_score=float(score), detector_label=str(label),
                                 crop_box_pixels=box.tolist()))
        return sorted(rows,key=lambda c:-c['detector_score'])

    def _parsed_generate(self, prompt, image, stage, parse, limits):
        for limit in limits:
            raw = self.runtime.generate(prompt, image, limit, stage)
            try:
                result = parse(raw)
                return result
            except (ValueError, TypeError, KeyError):
                pass
        return None

    def predict(self, image_path, expression, sample_id):
        import torch
        from PIL import Image, ImageDraw
        synchronize()
        start = time.perf_counter()
        self.runtime.trace = []
        calls = self.runtime.calls
        detector_calls = 0
        core_calls = 0
        trace = dict(expression=expression)
        with Image.open(image_path) as im:
            image = im.convert('RGB')
        case_dir = self.work_dir/'cases'/hashlib.sha256(str(sample_id).encode()).hexdigest()[:24]
        # No cached result read: runner owns configuration-bound completed-record cache.
        case_dir.mkdir(parents=True, exist_ok=True)
        forward_calls = [0]
        vision_calls = [0]
        def count_forward(*args):
            forward_calls[0] += 1
        hook = self.runtime.model.register_forward_pre_hook(count_forward)
        def count_vision(*args):
            vision_calls[0] += 1
        visual = getattr(self.runtime.model.model, 'visual', None)
        vision_hook = visual.register_forward_pre_hook(count_vision) if visual is not None else None
        failure = None
        box = None
        try:
            try:
                contract = expression_contract(self.runtime, expression)
            except (ValueError, TypeError):
                contract = None
                failure = 'language_contract_parse_error'
            if contract is not None:
                trace['contract'] = contract
                argv = sys.argv
                try:
                    sys.argv = ['teacher_per_view.py', '--image', str(image_path), '--target', expression,
                                '--output-dir', str(case_dir)]
                    with (case_dir/'search.log').open('w', buffering=1) as stream, contextlib.redirect_stdout(stream):
                        self.core.main()
                finally:
                    sys.argv = argv
                result = torch.load(case_dir/'fast_teacher_results.pt', map_location='cpu', weights_only=False)
                core_calls = int(result['total_qwen_calls'])
                # Original counters omit warmup; report separately rather than silently discard.
                core_warmups = int(result['timing']['warmup_sec']>0)+int(result['timing']['bt_language_warmup_sec']>0)
                core_calls += core_warmups
                trace['search'] = dict(ranking=result['ranking'], qwen_call_counts=result['qwen_call_counts'],
                                       counted_warmup_qwen_calls=core_warmups,
                                       timing=result['timing'], artifact=str(case_dir/'fast_teacher_results.pt'))
                teacher_box = result['ranking'][0]['box'] if result['ranking'] else None
                if teacher_box is None:
                    failure = 'search_no_box'
                else:
                    pool = [dict(method='teacher', box=teacher_box, detector_score=None, fallback=False)]
                    detections = {}
                    for branch, bounds in [('full',(0,0,*image.size)),
                                           ('padded',self.dino.crop_bounds(teacher_box,image.size,.2))]:
                        detector_calls += 1
                        rows = self._detect(image, expression, bounds)
                        detections[branch] = dict(bounds_pixels=list(bounds), candidates=rows)
                        if rows:
                            pool.append(dict(method='dino' if branch=='full' else 'teacher_dino',
                                             box=rows[0]['box'], detector_score=rows[0]['detector_score'], fallback=False))
                        elif branch=='padded':
                            pool.append(dict(method='teacher_dino', box=teacher_box, detector_score=None, fallback=True))
                    trace['detections'] = detections
                    for index,candidate in enumerate(pool):
                        marked = image.copy()
                        draw = ImageDraw.Draw(marked)
                        xy = [round(v*s) for v,s in zip(candidate['box'],image.size*2)]
                        draw.rectangle(xy, outline='black', width=3)
                        if xy[2]-xy[0]>2 and xy[3]-xy[1]>2:
                            draw.rectangle((xy[0]+1,xy[1]+1,xy[2]-1,xy[3]-1),outline='white',width=1)
                        marked.save(case_dir/f'candidate_{index}.png')
                        prompt = self.scorer.PROMPT.replace('robot-camera image','image').format(
                            category=contract['category'],colors=json.dumps(contract['colors']))
                        def parse_score(raw):
                            obj,error,_ = self.scorer.parse(raw)
                            if error:
                                raise ValueError(error)
                            return obj
                        candidate['scores'] = self._parsed_generate(prompt,marked,'score_'+candidate['method'],parse_score,(384,768))
                        def parse_verify(raw):
                            obj=json_object(raw)
                            if set(obj)!={'observed_object','observed_color','reason','decision'}:
                                raise ValueError('verification fields')
                            if any(not isinstance(v,str) or not v.strip() for v in obj.values()) or obj['decision'] not in {'yes','no','uncertain'}:
                                raise ValueError('verification values')
                            return obj
                        candidate['verification'] = self._parsed_generate(
                            VERIFY_PROMPT.format(category=contract['category'],colors=json.dumps(contract['colors'])),
                            marked,'verify_'+candidate['method'],parse_verify,(512,768))
                    trace['candidates'] = pool
                    if any(c['scores'] is None or c['verification'] is None for c in pool):
                        failure = 'candidate_score_or_verification_parse_error'
                    else:
                        accepted = [c for c in pool if c['verification']['decision']=='yes']
                        if accepted:
                            chosen = max(accepted,key=lambda c:c['scores']['overall_score'])
                            trace['chosen_method'] = chosen['method']
                            box = [float(v)*s for v,s in zip(chosen['box'],image.size*2)]
                            if not valid_box(box,*image.size):
                                box=None
                                failure='invalid_box'
                        else:
                            failure = 'teacher_uncertain' if any(c['verification']['decision']=='uncertain' for c in pool) else 'teacher_rejected'
        finally:
            hook.remove()
            if vision_hook is not None:
                vision_hook.remove()
        synchronize()
        trace['generation_calls'] = list(self.runtime.trace)
        trace['qwen_model_forward_calls'] = forward_calls[0]
        trace['qwen_vision_encoder_calls'] = vision_calls[0]
        return dict(sample_id=sample_id,box_xyxy=box,valid=box is not None,failure_reason=failure,
                    elapsed_sec=time.perf_counter()-start,qwen_calls=self.runtime.calls-calls+core_calls,
                    detector_calls=detector_calls,trace=trace)

"""Project existing entity spans into category/color; no re-parsing the whole instruction."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

PROMPT='''Name the physical object denoted by this noun in this context.
Noun from the entity parser: {head}
Meaning in the instruction: {role}
Return only JSON {{"category":"short unambiguous physical object category"}}.
Do not include color, shape, material, pattern, size, position, action, or body-part details. Name the whole object. If the noun is a named destination color, name the destination surface instead. Use the meaning to disambiguate homonyms. Do not refer to other objects.
'''


def key(instruction,target,role):
    return hashlib.sha256(json.dumps([instruction,target,role],ensure_ascii=False).encode()).hexdigest()


def colors_in(mention):
    from PIL import ImageColor
    # Shared color vocabulary, not per-task tuning. Only literal source words.
    names=set(ImageColor.colormap)|{'golden'}
    return list(dict.fromkeys(m.group().lower() for m in re.finditer(r'\b[a-zA-Z]+\b',mention) if m.group().lower() in names))


def main():
    import torch
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--entities',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--model',default='/data/adminad/models/Qwen3.5-4B');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,a.out/'project_snapshot.py')
    items={};lang={r['id']:r for r in json.loads(a.entities.read_text())}
    for r in json.loads(a.source.read_text()):
        scene=r['scene'].rsplit('_f',1)[0];lr=lang[scene]
        assert lr['schema_valid'] and lr['instruction']==r['instruction']
        role_id=next(k for k,v in lr['role_key_map'].items() if v==r['role'])
        graph=lr['entity_graph'];ids=graph['roles'][role_id]
        matches=[o for o in graph['objects'] if o['id'] in ids and o['grounding_text']==r['target']]
        if len(matches)!=1:raise ValueError('Entity cannot be matched unambiguously')
        obj=matches[0];k=key(r['instruction'],r['target'],r['role_description'])
        items[k]=dict(instruction=r['instruction'],target=r['target'],role_description=r['role_description'],head=obj['head'],source=obj['source'],colors=colors_in(r['target']))
    (a.out/'input_projection.json').write_text(json.dumps(items,ensure_ascii=False,indent=2))
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    results={};cache={};torch.manual_seed(0)
    for k,item in items.items():
        prompt=PROMPT.format(head=item['head'],role=item['role_description'])
        # A color-only source is an ellipsis, not an object noun to reclassify.
        # Resolve only an explicitly supplied physical noun from its role metadata.
        # This rule uses no task names, episode IDs, images or visual labels.
        head_words=item['head'].lower().split()
        color_only=bool(head_words) and all(w in colors_in(item['head']) for w in head_words)
        explicit_surface=bool(re.search(r'\bsurface\b',item['role_description'],re.I))
        if color_only:
            if not explicit_surface:raise ValueError('Color-only target lacks explicit physical category; needs language review')
            cache[prompt]=('surface',json.dumps({'category':'surface'}))
        if prompt not in cache:
            inputs=processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':prompt}]}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to(model.device)
            with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=64,do_sample=False)
            raw=processor.tokenizer.decode(out[0,inputs['input_ids'].shape[-1]:],skip_special_tokens=True).strip()
            parsed=json.loads(raw)
            if set(parsed)!={'category'} or not isinstance(parsed['category'],str) or not parsed['category'].strip():raise ValueError(raw)
            cache[prompt]=(parsed['category'],raw)
        category,raw=cache[prompt]
        results[k]=dict(**{f:item[f] for f in ['instruction','target','role_description','head','source']},answer=dict(category=category,colors=item['colors']),prompt=prompt,raw=raw,model=a.model,category_origin='explicit_surface_noun_in_role' if color_only else 'qwen_noun_disambiguation')
        (a.out/'contracts.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(target=item['target'],**results[k]['answer']),ensure_ascii=False),flush=True)
    (a.out/'status.json').write_text(json.dumps(dict(phase='DONE',completed=len(results),total=len(items))))
    (a.out/'provenance.json').write_text(json.dumps(dict(model=a.model,prompt=PROMPT,entities=str(a.entities.resolve()),entities_sha256=hashlib.sha256(a.entities.read_bytes()).hexdigest(),source=str(a.source.resolve()),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),color_vocabulary='Pillow CSS names plus golden; exact words in entity mention only',scope='Category disambiguation from existing parsed noun and role, colors projected from own source span; no images, old scores or manual visual labels'),indent=2))


if __name__=='__main__':main()

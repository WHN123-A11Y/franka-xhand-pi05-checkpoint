"""Score only category and color in a fixed full-frame candidate."""
import json
import math

PROMPT_VERSION='fullframe_category_color_scores_v1'
PROMPT='''Inspect the complete robot-camera image. The thin black/white outline is an annotation of a candidate region, not part of the object.
Requested object category: {category}
Requested colors: {colors}

Rate visible evidence that the object INSIDE this rectangle matches this category and these colors. Ignore all other properties: size, shape details, patterns, material, brand, location, pose, action and functional role. An empty color list means no color requirement. A color may belong to a visible component, not necessarily the whole object. Never infer an unstated color. Use the full scene to distinguish the physical object from robot components or background. An object outside the rectangle does not validate it. A partly visible target or loose box can match if the object is identifiable inside.
Rate each box independently from 0 to 100: 0 is a clear mismatch/no visible support, 50 is ambiguous evidence, 100 is clear visible support. Scores are not calibrated probabilities. Do not force one candidate or camera to be correct.
Return ONLY JSON with exactly these fields:
{{"object_score":0,"attribute_score":null,"overall_score":0,"observed_object":"actual visible object","observed_color":"visible color or unclear","reason":"brief category/color evidence"}}
object_score and overall_score must be numbers 0..100. attribute_score rates ONLY the requested colors, 0..100, or null if the color list is empty. Do not score other attributes or roles.
'''


def parse(raw):
    lines=raw.strip().splitlines();wrapper=len(lines)>=3 and lines[0] in {'```','```json'} and lines[-1]=='```'
    try:
        answer=json.loads('\n'.join(lines[1:-1]) if wrapper else raw)
        if not isinstance(answer,dict) or set(answer)!={'object_score','attribute_score','overall_score','observed_object','observed_color','reason'}:raise ValueError('wrong fields')
        for k in ['object_score','attribute_score','overall_score']:
            n=answer[k]
            if k=='attribute_score' and n is None:continue
            if isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or not 0<=n<=100:raise ValueError('invalid score')
        for k in ['observed_object','observed_color','reason']:
            if not isinstance(answer[k],str) or not answer[k].strip():raise ValueError('invalid text')
        return answer,None,wrapper
    except (ValueError,TypeError) as e:return None,str(e),wrapper

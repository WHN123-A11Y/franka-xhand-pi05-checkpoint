"""Preserve literal single-role color words that the teacher head parser drops.

The role expression is a single exact noun phrase extracted from the dataset.
This wrapper does not correct unknown words such as 'grew', invent categories,
or consult any ground-truth boxes/images. Raw parser output is retained.
"""
import re
from PIL import ImageColor
import teacher_adapter

_original=teacher_adapter.expression_contract

def expression_contract(runtime,expression):
    result=_original(runtime,expression)
    raw=dict(result)
    names=set(ImageColor.colormap)|{'golden'}
    literal=list(dict.fromkeys(m.group().lower() for m in re.finditer(r'\b[a-zA-Z]+\b',expression) if m.group().lower() in names))
    result['original_contract']=raw
    result['literal_role_colors']=literal
    result['color_rule']='literal color words from the supplied single-role phrase; unknown words remain unchanged'
    result['colors']=literal
    result['target_phrase']=expression
    return result

def install():
    teacher_adapter.expression_contract=expression_contract

"""Export Teacher candidate boxes for human review; scores are not visibility probabilities."""
import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

VIEWS=['cam_high','cam_left_wrist','cam_right_wrist']
COLORS=['red','orange','cyan']

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    results=list((args.run/'teacher_output').glob('*/multiview_results_bt_v1.pt'))
    if len(results)!=1:raise ValueError('Expected one completed group')
    result=torch.load(results[0],map_location='cpu',weights_only=False)
    summary={'warning':'Relative ranking only, not calibrated visibility or cross-view identity.',
             'target':json.loads((args.run/'config.json').read_text())['target'],
             'cross_view_reliability':result['cross_view_reliability'],
             'final_multiview_ranking':result['final_multiview_ranking'],'views':{}}
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,17) if Path(font_path).exists() else ImageFont.load_default()
    panels=[]
    for view in VIEWS:
        local=torch.load(results[0].parent/f'sample_{view}/fast_teacher_results.pt',map_location='cpu',weights_only=False)
        ranking=local['ranking'][:3]
        image=Image.open(args.run/'images'/f'{view}.png').convert('RGB')
        draw=ImageDraw.Draw(image)
        for i,r in reversed(list(enumerate(ranking))):
            box=[float(v) for v in r['box']]
            if len(box)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in box):raise ValueError('Invalid box')
            if not math.isfinite(r['bt_score']):raise ValueError('Nonfinite score')
            x1,y1,x2,y2=[round(v*s) for v,s in zip(box,[image.width,image.height,image.width,image.height])]
            draw.rectangle((x1,y1,x2,y2),outline=COLORS[i],width=3)
            draw.text((x1+2,y1+2),f'Top {i+1}',fill=COLORS[i],font=font)
        panel=Image.new('RGB',(image.width,image.height+65),'white');panel.paste(image,(0,65))
        d=ImageDraw.Draw(panel);d.text((8,5),view,fill='black',font=font)
        d.text((8,29),'Local Top1 red / Top2 orange / Top3 cyan',fill='black',font=font)
        panel.save(args.run/f'{view}_top3.png');panels.append(panel)
        summary['views'][view]={'local_top3':ranking,'timing':local['timing']}
    sheet=Image.new('RGB',(sum(p.width for p in panels),max(p.height for p in panels)),'white')
    x=0
    for panel in panels:sheet.paste(panel,(x,0));x+=panel.width
    sheet.save(args.run/'three_view_top3.png')
    best=result['final_multiview_ranking'][0]
    best_image=Image.open(args.run/'images'/f"{best['view_id']}.png").convert('RGB')
    box=[round(float(v)*s) for v,s in zip(best['box'],[best_image.width,best_image.height]*2)]
    draw=ImageDraw.Draw(best_image)
    draw.rectangle(box,outline='red',width=3)
    draw.text((8,8),f"Cross-view best: {best['view_id']}",fill='black',font=font)
    best_image.save(args.run/'best_candidate.png')
    def encode(value):
        if torch.is_tensor(value):return value.tolist()
        if isinstance(value,Path):return str(value)
        raise TypeError(type(value).__name__)
    (args.run/'summary.json').write_text(json.dumps(summary,default=encode,ensure_ascii=False,indent=2,allow_nan=False))
    print(args.run/'three_view_top3.png')

if __name__=='__main__':main()

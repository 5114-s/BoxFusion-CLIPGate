import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
ROOT=Path('/data/ZhaoX/BoxFusion')
OUT=ROOT/'reports/three_module_scene_search_20261115'
parser=argparse.ArgumentParser()
parser.add_argument('--scenes',nargs='+',default=['scene0644_00','scene0591_01','scene0535_00','scene0077_00','scene0518_00','scene0169_00'])
parser.add_argument('--output',default='shortlist_rgb.jpg')
args=parser.parse_args()
scenes=args.scenes
font=ImageFont.truetype('DejaVuSans.ttf',17)
canvas=Image.new('RGB',(1280,len(scenes)*270),'white')
draw=ImageDraw.Draw(canvas)
for row,scene in enumerate(scenes):
    cache=ROOT/'development/final_controls/experiments/mvsr_multiview_audit_20261012/native_prefix'/f'{scene}.npz'
    with np.load(cache) as z: frames=z['frame_ids']
    for col,fraction in enumerate([.10,.35,.60,.85]):
        frame=int(frames[round((len(frames)-1)*fraction)])
        path=ROOT/'data/scannet_val_rgbfix'/scene/'frames/color'/f'{frame}.jpg'
        draw.text((col*320+6,row*270+4),f'{scene} | f{frame}',font=font,fill='black')
        if path.exists():
            im=ImageOps.contain(Image.open(path).convert('RGB'),(314,234))
            canvas.paste(im,(col*320+3,row*270+30))
target=OUT/args.output
canvas.save(target,quality=95)
print(target)

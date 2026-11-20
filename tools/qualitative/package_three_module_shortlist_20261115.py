from pathlib import Path
import zipfile
from PIL import Image, ImageDraw, ImageOps
import build_framework_scene0030_real_assets as source

ROOT=source.ROOT
OUT=ROOT/'figure_assets/three_module_scene_shortlist_20261115'
cases=[('scene0591_01','-170','Office | scene0591_01',
        ['HSR: sofa | rank 16 -> 12','PTR: chair | IoU 0.639','VAR: cabinet | IoU 0.596'],279),
       ('scene0169_00','-75','Conference room | scene0169_00',
        ['HSR: chair | rank 2 -> 1','PTR: other furniture | IoU 0.657','VAR: table | IoU 0.677'],335)]
canvas=Image.new('RGB',(2000,1070),'white')
draw=ImageDraw.Draw(canvas)
for col,(scene,az,title,labels,total) in enumerate(cases):
    x=col*1000
    im=Image.open(OUT/f'{scene}_final_az{az}_white.png')
    im=ImageOps.contain(im,(980,784),Image.Resampling.LANCZOS)
    canvas.paste(im,(x+(1000-im.width)//2,70))
    draw.text((x+30,22),title,font=source.font(28,True),fill=source.INK)
    for row,(text,color) in enumerate(zip(labels,['#2779B5','#EF7624','#00A695'])):
        draw.text((x+30,855+row*38),text,font=source.font(25),fill=color)
    draw.text((x+30,978),f'GT-assisted selection: 3 / {total} predictions shown.',font=source.font(22),fill='#536273')
draw.text((30,1030),'Real mesh and unchanged predicted boxes. Mesh used for visualization only.',font=source.font(22),fill='#536273')
canvas.save(OUT/'recommended_two_scenes_preview.jpg',quality=96)
pack=ROOT/'figure_assets/carr3d_three_module_verified_scenes_20261115.zip'
with zipfile.ZipFile(pack,'w',zipfile.ZIP_DEFLATED) as z:
    for scene,az,*_ in cases:
        for suffix in ['white.png','transparent.png']:
            path=OUT/f'{scene}_final_az{az}_{suffix}'
            z.write(path,arcname=path.name)
        path=OUT/f'{scene}_final_selection.json'
        z.write(path,arcname=path.name)
        audit=ROOT/'reports/three_module_scene_search_20261115'/scene
        for name in ['audit_summary.json','cache_replay_gt_coverage.csv','hsr_terminal_examples.json','branch_lineage.json']:
            z.write(audit/name,arcname=f'audit/{scene}/{name}')
    for name in ['使用说明.txt','recommended_two_scenes_preview.jpg']:
        z.write(OUT/name,arcname=name)
print(pack)

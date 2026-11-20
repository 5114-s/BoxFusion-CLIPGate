#!/usr/bin/env python3
"""Render only genuine cached predictions, with explicit selected-target disclosure."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import build_framework_scene0030_real_assets as source

ROOT=source.ROOT
AUDIT=ROOT/'reports/three_module_scene_search_20261115'
OUT=ROOT/'figure_assets/three_module_scene_shortlist_20261115'
COLORS={'hsr':'#2779B5','ptr':'#EF7624','var':'#00A695'}

def camera_for(renderer,xyz,faces,boxes,azimuth,elevation,width,height,framing='scene'):
    valid=np.unique(faces)
    lo,hi=xyz[valid].min(0),xyz[valid].max(0)
    look=(lo+hi)/2
    look[2]=.95
    if framing=='targets': look=boxes.mean((0,1))
    az,el=np.deg2rad([azimuth,elevation])
    direction=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    position=look+max(12.,np.linalg.norm(hi-lo)*2)*direction
    principal=np.array([width/2,height/2],float)
    focal=float(width*1.5)
    points=np.concatenate([xyz[valid],boxes.reshape(-1,3)]) if framing=='scene' else boxes.reshape(-1,3)
    uv=renderer.project(points,position,look,focal,principal)[:,:2]
    factor=.75 if framing=='targets' else .92
    focal*=min(width*factor/np.ptp(uv[:,0]),height*factor/np.ptp(uv[:,1]))
    uv=renderer.project(points,position,look,focal,principal)[:,:2]
    principal+=np.array([width/2,height/2])-(uv.min(0)+uv.max(0))/2
    return position,look,focal,principal

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenes',nargs='+',default=['scene0591_01','scene0644_00','scene0535_00','scene0169_00','scene0518_00'])
    parser.add_argument('--azimuths',nargs='+',type=float,default=[-135,-45,45,135])
    parser.add_argument('--elevation',type=float,default=30)
    parser.add_argument('--width',type=int,default=1000)
    parser.add_argument('--hsr-id',type=int)
    parser.add_argument('--ptr-gt',type=int)
    parser.add_argument('--var-gt',type=int)
    parser.add_argument('--output-suffix',default='')
    parser.add_argument('--framing',choices=['scene','targets'],default='scene')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    renderer=source.common.load_renderer()
    for scene in args.scenes:
        audit=json.loads((AUDIT/scene/'audit_summary.json').read_text())
        z=np.load(AUDIT/scene/'verified_geometry.npz')
        matrix=z['axis_alignment']
        boxes=z['boxes_world']@matrix[:3,:3].T+matrix[:3,3]
        pair=dict(audit['best_distinct_target_pair'])
        if args.ptr_gt is not None: pair['ptr_gt']=args.ptr_gt
        if args.var_gt is not None: pair['var_gt']=args.var_gt
        pg,vg=pair['ptr_gt'],pair['var_gt']
        assert pg!=vg
        pair['ptr_iou']=audit['gt_coverage'][pg]['m1p_max_iou']
        pair['var_iou']=audit['gt_coverage'][vg]['m1a_max_iou']
        pair['minimum_iou']=min(pair['ptr_iou'],pair['var_iou'])
        assert audit['gt_coverage'][pg]['native_max_iou']<.15
        assert audit['gt_coverage'][vg]['native_max_iou']<.15
        choices=[r for r in audit['hsr_terminal_examples'] if r['gt_index'] not in [pg,vg] and r['gt_iou']>=.5]
        if args.hsr_id is not None: choices=[r for r in choices if r['native_id']==args.hsr_id]
        hsr=max(choices,key=lambda r:(r['rank_gain'],r['gt_iou']))
        pi=audit['gt_coverage'][pg]['m1p_best_index']
        vi=audit['gt_coverage'][vg]['m1a_best_index']
        ids=[hsr['prediction_index'],pi,vi]
        selected=boxes[ids]
        objects=json.loads((source.common.RAW/scene/f'{scene}.aggregation.json').read_text())['segGroups']
        walls=[int(o['objectId']) for o in objects if o['label']=='wall']
        xyz,rgb,display_rgb,faces,mesh_info=renderer.load_mesh(scene,matrix,walls)
        data=dict(scene=scene,pair=pair,hsr=hsr,selected_prediction_indices=ids,
                  selected_gt_indices=[hsr['gt_index'],pg,vg],counts=audit['cached_replay_counts'],
                  geometry_modified=False,mesh=mesh_info,colors=COLORS,
                  framing=args.framing,
                  selection='GT-assisted illustrative selection: three predictions, not complete output',
                  source=str(AUDIT/scene/'verified_geometry.npz'),source_sha256=source.common.sha(AUDIT/scene/'verified_geometry.npz'),
                  camera_views=[])
        previews=[]
        for az in args.azimuths:
            width,height=args.width,int(args.width*.8)
            projection=camera_for(renderer,xyz,faces,selected,az,args.elevation,width,height,args.framing)
            projected=renderer.project(xyz,*projection)
            rgba=renderer.rasterize(projected,display_rgb,faces,width,height)
            im=Image.fromarray(rgba)
            draw=ImageDraw.Draw(im)
            for role,box in zip(['hsr','ptr','var'],selected):
                uv=renderer.project(box,*projection)[:,:2]
                for a,b in renderer.box_edges(box):
                    draw.line([tuple(uv[a]),tuple(uv[b])],fill=COLORS[role],width=max(3,round(width/260)))
            stem=f'{scene}{args.output_suffix}_az{int(az)}'
            im.save(OUT/f'{stem}_transparent.png',dpi=(400,400))
            white=Image.new('RGB',im.size,'white'); white.paste(im,mask=im.getchannel('A'))
            white.save(OUT/f'{stem}_white.png',dpi=(400,400))
            preview=Image.new('RGB',(width,height+122),'white'); preview.paste(white,(0,0))
            d=ImageDraw.Draw(preview)
            d.text((15,height+2),f'{scene} | azimuth {az:g} | 3 selected / {len(boxes)} predictions',font=source.font(20,True),fill=source.INK)
            d.text((15,height+32),f"HSR {hsr['gt_class']}: rank {hsr['native_rank']} -> {hsr['hsr_rank']} | IoU {hsr['gt_iou']:.3f}",font=source.font(19),fill=COLORS['hsr'])
            d.text((15,height+59),f"PTR {audit['gt_coverage'][pg]['class']}: IoU {pair['ptr_iou']:.3f}",font=source.font(19),fill=COLORS['ptr'])
            d.text((15,height+86),f"VAR {audit['gt_coverage'][vg]['class']}: IoU {pair['var_iou']:.3f}",font=source.font(19),fill=COLORS['var'])
            preview.save(OUT/f'{stem}_audit.jpg',quality=95)
            previews.append(preview)
            pos,look,focal,principal=projection
            data['camera_views'].append(dict(azimuth=az,elevation=args.elevation,position=pos,look_at=look,
                focal_pixels=focal,principal_point_pixels=principal,image_size=[width,height],asset_stem=stem))
        cols=2 if len(previews)>1 else 1
        sheet=Image.new('RGB',(previews[0].width*cols,previews[0].height*((len(previews)+cols-1)//cols)),'white')
        for i,im in enumerate(previews): sheet.paste(im,((i%cols)*im.width,(i//cols)*im.height))
        sheet.save(OUT/f'{scene}{args.output_suffix}_views.jpg',quality=95)
        source.save_json(OUT/f'{scene}{args.output_suffix}_selection.json',data)
        print(scene, 'HSR',hsr['native_id'],hsr['native_rank'],hsr['hsr_rank'],'PTR',pair['ptr_iou'],'VAR',pair['var_iou'],flush=True)

if __name__=='__main__': main()

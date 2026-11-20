#!/usr/bin/env python3
"""Reference-like composition from original high-resolution scan, never inpainted."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from plyfile import PlyData
import build_framework_scene0030_real_assets as source

ROOT=source.ROOT
OLD=ROOT/'figure_assets/three_module_scene_shortlist_20261115'
OUT=ROOT/'figure_assets/carr3d_reference_style_real_20261115'
COLORS=['#2499D0','#F28C12','#00A88E']

def high_mesh(scene,matrix):
    directory=source.common.RAW/scene
    path=directory/f'{scene}_vh_clean.ply'
    p=PlyData.read(path,known_list_len={'face':{'vertex_indices':3}})
    v=p['vertex'].data
    xyz=np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float64)
    xyz=xyz@matrix[:3,:3].T+matrix[:3,3]
    rgb=np.column_stack([v[k] for k in ['red','green','blue']]).astype(np.float64)/255.
    normals=np.column_stack([v[k] for k in ['nx','ny','nz']]).astype(np.float64)@matrix[:3,:3].T
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
    light=np.array([-.35,-.55,.76]); light/=np.linalg.norm(light)
    illumination=.88+.12*np.abs(normals@light)
    rgb=np.clip(rgb**.92*illumination[:,None]+.015,0,1)
    del normals
    faces=np.ascontiguousarray(p['face'].data['vertex_indices'],dtype=np.int32)
    segments=np.asarray(json.loads((directory/f'{scene}_vh_clean.segs.json').read_text())['segIndices'],np.int32)
    objects=json.loads((directory/f'{scene}_vh_clean.aggregation.json').read_text())['segGroups']
    walls=[]
    for obj in objects:
        if obj['label']!='wall': continue
        mask=np.isin(segments,obj['segments'])
        pts=xyz[mask]
        walls.append(dict(id=int(obj['objectId']),segments=obj['segments'],center=(pts.min(0)+pts.max(0))/2))
    return xyz,rgb,faces,segments,objects,walls,path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',default='scene0591_01')
    parser.add_argument('--width',type=int,default=1200)
    parser.add_argument('--azimuths',type=float,nargs='+',default=[-170,-150,-130])
    parser.add_argument('--elevation',type=float,default=22)
    parser.add_argument('--distance',type=float,default=8)
    parser.add_argument('--fill',type=float,default=.73)
    parser.add_argument('--tag',default='probe')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    scene=args.scene
    selected=json.loads((OLD/f'{scene}_final_selection.json').read_text())
    z=np.load(selected['source'])
    matrix=z['axis_alignment']
    boxes=z['boxes_world'][selected['selected_prediction_indices']]@matrix[:3,:3].T+matrix[:3,3]
    renderer=source.common.load_renderer()
    xyz,rgb,faces,segments,objects,walls,path=high_mesh(scene,matrix)
    print('Loaded original mesh',scene,len(xyz),len(faces),flush=True)
    views=[]; previews=[]
    for az in args.azimuths:
        width,height=args.width,args.width
        look=boxes.mean((0,1)); azr,elr=np.deg2rad([az,args.elevation])
        direction=np.array([np.cos(elr)*np.cos(azr),np.cos(elr)*np.sin(azr),np.sin(elr)])
        pos=look+args.distance*direction
        # Remove near-side walls only; original rear walls and all objects remain.
        removed=[w for w in walls if np.dot(w['center'][:2]-look[:2],direction[:2])>0]
        remove_segments=[s for w in removed for s in w['segments']]
        remove_segments += [s for o in objects if o['label']=='ceiling' for s in o['segments']]
        rm=np.isin(segments,remove_segments)
        keep=~(rm[faces[:,0]]|rm[faces[:,1]]|rm[faces[:,2]])
        # Crop architectural upper border only; no objects are altered or removed.
        max_z=max(2.7,float(boxes[:,:,2].max())+.1)
        keep &= np.minimum(np.minimum(xyz[faces[:,0],2],xyz[faces[:,1],2]),xyz[faces[:,2],2])<max_z
        ff=faces[keep]
        principal=np.array([width/2,height/2],float); focal=float(width)
        uv=renderer.project(boxes.reshape(-1,3),pos,look,focal,principal)[:,:2]
        focal*=min(width*args.fill/np.ptp(uv[:,0]),height*args.fill/np.ptp(uv[:,1]))
        uv=renderer.project(boxes.reshape(-1,3),pos,look,focal,principal)[:,:2]
        principal+=np.array([width/2,height*.54])-(uv.min(0)+uv.max(0))/2
        projection=(pos,look,focal,principal)
        rgba=renderer.rasterize(renderer.project(xyz,*projection),rgb,ff,width,height)
        image=Image.fromarray(rgba); draw=ImageDraw.Draw(image)
        for box,color in zip(boxes,COLORS):
            uv=renderer.project(box,*projection)[:,:2]
            for a,b in renderer.box_edges(box):
                draw.line([tuple(uv[a]),tuple(uv[b])],fill=color,width=max(3,round(width*.004)))
        stem=f'{scene}_{args.tag}_az{int(az)}'
        image.save(OUT/f'{stem}_transparent.png',dpi=(400,400))
        white=Image.new('RGB',image.size,'white');white.paste(image,mask=image.getchannel('A'))
        white.save(OUT/f'{stem}_white.png',dpi=(400,400))
        # Typography and frame are separate from the unmodified data rendering.
        s=width/1200
        card=Image.new('RGB',(width+round(60*s),height+round(240*s)),'white')
        d=ImageDraw.Draw(card)
        d.rounded_rectangle((round(6*s),round(6*s),card.width-round(7*s),card.height-round(7*s)),radius=round(24*s),outline='#9EA6AC',width=max(1,round(2*s)))
        d.text((card.width/2,round(42*s)),'Online 3D Detections',anchor='mt',font=source.font(round(48*s),True),fill='#1B2634')
        card.paste(white,(round(30*s),round(110*s)))
        labels=['Native / HSR','Proposal / PTR','Anchor / VAR']
        for i,(label,color) in enumerate(zip(labels,COLORS)):
            x=round((45+390*i)*s); y=height+round(140*s)
            d.line((x,y+round(13*s),x+round(44*s),y+round(13*s)),fill=color,width=max(2,round(5*s)))
            d.text((x+round(54*s),y),label,font=source.font(round(27*s)),fill='#243142')
        d.text((card.width/2,height+round(192*s)),f'{scene} | Selected predictions; mesh for visualization only',anchor='mt',font=source.font(round(21*s)),fill='#5B6570')
        card.save(OUT/f'{stem}_panel.png',dpi=(400,400))
        previews.append((f'azimuth {az:g}, original high-resolution mesh',card))
        views.append(dict(azimuth=az,elevation=args.elevation,camera_position=pos,look_at=look,
            focal_pixels=focal,principal_point_pixels=principal,image_size=[width,height],
            removed_near_side_wall_ids=[w['id'] for w in removed],retained_faces=len(ff),
            upper_height_crop=max_z,output_stem=stem))
        print('Rendered',stem,flush=True)
    source.sheet(previews,OUT/f'{scene}_{args.tag}_views.jpg',cols=len(previews),tw=600,th=690)
    source.save_json(OUT/f'{scene}_{args.tag}_manifest.json',dict(scene=scene,selected_records=selected,
        mesh_source=path,mesh_sha256=source.common.sha(path),vertices=len(xyz),faces=len(faces),
        prediction_geometry_modified=False,mesh_inpainting=False,furniture_added=False,
        generated_imagery=False,views=views,notes=[
        'Original high-resolution ScanNet mesh, no generated surfaces or hole filling.',
        'View-dependent cutaway removes foreground walls; image framing crops background.',
        'GT-assisted selected output, not the complete detection list.']))

if __name__=='__main__':main()

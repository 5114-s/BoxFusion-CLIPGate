#!/usr/bin/env python3
"""Render authentic pre-NMS anchors and post-NMS proposals from one final-run keyframe."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = ['#FF7A1A', '#16B86A', '#1677FF', '#A855F7']

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def frame_slice(frame_ids, lengths, target):
    start=0
    for frame,length in zip(frame_ids,lengths):
        length=int(length)
        if int(frame)==target: return slice(start,start+length)
        start+=length
    raise KeyError(target)

def project(boxes, pose, intrinsic, width, height):
    inverse=np.linalg.inv(pose)
    results=[]
    for box in boxes:
        world=np.c_[box,np.ones(8)]
        camera=(inverse@world.T).T[:,:3]
        if np.sum(camera[:,2]>.05)<4:
            results.append(None); continue
        camera=camera[camera[:,2]>.05]
        uvw=(intrinsic@camera.T).T
        uv=uvw[:,:2]/uvw[:,2:3]
        lo=uv.min(0); hi=uv.max(0)
        clipped_lo=np.maximum(lo,[0,0]); clipped_hi=np.minimum(hi,[width-1,height-1])
        if np.any(clipped_hi-clipped_lo<6) or np.prod(clipped_hi-clipped_lo)<120:
            results.append(None); continue
        results.append([*clipped_lo.tolist(),*clipped_hi.tolist()])
    return results

def draw_box(draw, box, color, width, alpha=None):
    if box is None: return
    draw.rectangle(tuple(map(float,box)),outline=color,width=width)

def add_badge(image, text):
    draw=ImageDraw.Draw(image,'RGBA')
    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',34)
    except OSError: font=ImageFont.load_default()
    bb=draw.textbbox((0,0),text,font=font)
    pad=14
    draw.rounded_rectangle((18,18,18+bb[2]+2*pad,18+bb[3]+2*pad),radius=10,fill=(0,0,0,165))
    draw.text((18+pad,18+pad),text,font=font,fill=(255,255,255,255))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--raw-root',type=Path,required=True)
    ap.add_argument('--scene',required=True)
    ap.add_argument('--frame',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rgb_path=args.raw_root/args.scene/'color'/f'{args.frame}.jpg'
    pose_path=args.raw_root/args.scene/'pose'/f'{args.frame}.txt'
    intr_path=args.raw_root/args.scene/'intrinsic'/'intrinsic_color.txt'
    image=Image.open(rgb_path).convert('RGB')
    w,h=image.size
    pose=np.loadtxt(pose_path); intrinsic=np.loadtxt(intr_path)[:3,:3]
    with np.load(args.cache,allow_pickle=False) as z:
        ps=frame_slice(z['frame_ids'],z['proposal_lengths'],args.frame)
        ans=frame_slice(z['frame_ids'],z['anchor_lengths'],args.frame)
        proposals=z['proposal_boxes_2d'][ps].astype(float)
        proposal_scores=z['proposal_scores'][ps].astype(float)
        anchors=z['anchor_corners'][ans].astype(float)
        anchor_scores=z['anchor_scores'][ans].astype(float)
    # Cached 2D proposals use the 640x480 inference canvas.
    proposals[:,[0,2]]*=w/640.0; proposals[:,[1,3]]*=h/480.0
    anchor_rects=project(anchors,pose,intrinsic,w,h)
    valid=[i for i,b in enumerate(anchor_rects) if b is not None]
    valid.sort(key=lambda i:(-anchor_scores[i],i))

    # Dense pre-NMS view: all valid anchors remain visible, with top-ranked anchors emphasized.
    pre=image.copy().convert('RGBA')
    faint=Image.new('RGBA',pre.size,(0,0,0,0)); fd=ImageDraw.Draw(faint,'RGBA')
    for i in reversed(valid):
        fd.rectangle(anchor_rects[i],outline=(22,119,255,42),width=2)
    pre=Image.alpha_composite(pre,faint)
    pd=ImageDraw.Draw(pre,'RGBA')
    for rank,i in enumerate(valid[:36]):
        draw_box(pd,anchor_rects[i],PALETTE[rank%len(PALETTE)],4)
    pre_clean=pre.copy()
    add_badge(pre,f'Pre-NMS Anchors  |  {len(anchors)} candidates')

    # Sparse post-NMS view: draw every retained proposal from exactly the same keyframe.
    post=image.copy().convert('RGBA'); qd=ImageDraw.Draw(post,'RGBA')
    order=np.argsort(-proposal_scores,kind='stable')
    for rank,i in enumerate(order):
        draw_box(qd,proposals[i],PALETTE[rank%len(PALETTE)],5)
    post_clean=post.copy()
    add_badge(post,f'Post-NMS Proposals  |  {len(proposals)} retained')

    pre_clean_path=args.output/'pre_nms_anchors_clean.png'; post_clean_path=args.output/'post_nms_proposals_clean.png'
    pre_clean.convert('RGB').save(pre_clean_path,quality=95)
    post_clean.convert('RGB').save(post_clean_path,quality=95)
    pre_path=args.output/'pre_nms_anchors.png'; post_path=args.output/'post_nms_proposals.png'
    pre.convert('RGB').save(pre_path,quality=95)
    post.convert('RGB').save(post_path,quality=95)
    meta={
      'schema':'boxfusion.recar3d.candidate_inputs.v1','scene':args.scene,'frame_id':args.frame,
      'same_keyframe':True,'strict_online_final_run_cache':str(args.cache.resolve()),
      'pre_nms_anchor_count':int(len(anchors)),'projectable_anchor_count':int(len(valid)),
      'emphasized_anchor_count':int(min(36,len(valid))),
      'post_nms_proposal_count':int(len(proposals)),
      'rgb_path':str(rgb_path),'rgb_sha256':sha(rgb_path),'cache_sha256':sha(args.cache),
      'outputs':{'pre_nms_anchors':{'path':str(pre_path.resolve()),'sha256':sha(pre_path)},
                 'post_nms_proposals':{'path':str(post_path.resolve()),'sha256':sha(post_path)},
                 'pre_nms_anchors_clean':{'path':str(pre_clean_path.resolve()),'sha256':sha(pre_clean_path)},
                 'post_nms_proposals_clean':{'path':str(post_clean_path.resolve()),'sha256':sha(post_clean_path)}}
    }
    (args.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__': main()

#!/usr/bin/env python3
"""Render matched multi-view pre-/post-NMS candidate assets for scene0432_01."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path('/data/ZhaoX/BoxFusion')
SCENE='scene0432_01'; FRAMES=(0,25,50)
RAW=Path('/extra/ZhaoX/scannet_data/scans')/SCENE
CACHE=ROOT/'development/final_controls/evidence/scannet_recar3d_key_controls/scene0432_01.npz'
OUT=ROOT/'figure_assets/pipeline_candidate_visuals_scene0432_01'
COLORS=('#F28C00','#18B957','#3478F6')
TARGET_PROPOSALS={0:(0,3,2),25:(1,0,2),50:(2,1,0)}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def slices(ids,lengths):
 o=0; d={}
 for f,n in zip(ids,lengths): d[int(f)]=slice(o,o+int(n)); o+=int(n)
 return d

def project(boxes,pose,K,w,h):
 inv=np.linalg.inv(pose); out=[]
 for b in boxes:
  cam=(inv@np.c_[b,np.ones(8)].T).T[:,:3]
  cam=cam[cam[:,2]>.05]
  if len(cam)<4: out.append(None); continue
  uvw=(K@cam.T).T; uv=uvw[:,:2]/uvw[:,2:3]
  lo=np.maximum(uv.min(0),[0,0]); hi=np.minimum(uv.max(0),[w-1,h-1])
  if np.any(hi-lo<6) or np.prod(hi-lo)<100: out.append(None)
  else: out.append([*lo,*hi])
 return out

def rect_iou(rects, targets):
 rects=np.asarray(rects,float); targets=np.asarray(targets,float)
 lo=np.maximum(rects[:,None,:2],targets[None,:,:2]); hi=np.minimum(rects[:,None,2:],targets[None,:,2:])
 inter=np.prod(np.maximum(hi-lo,0),axis=2)
 ra=np.prod(rects[:,2:]-rects[:,:2],axis=1)[:,None]; ta=np.prod(targets[:,2:]-targets[:,:2],axis=1)[None]
 return inter/np.maximum(ra+ta-inter,1e-12)

def render_frame(frame,kind,z,plook,alook):
 rgb=Image.open(RAW/'color'/f'{frame}.jpg').convert('RGBA'); w,h=rgb.size
 draw=ImageDraw.Draw(rgb,'RGBA')
 psl=plook[frame]
 proposal_boxes=z['proposal_boxes_2d'][psl].astype(float)
 proposal_scores=z['proposal_scores'][psl]
 proposal_boxes[:,[0,2]]*=w/640.; proposal_boxes[:,[1,3]]*=h/480.
 if kind=='proposal':
  order=np.argsort(-proposal_scores,kind='stable')
  for rank,i in enumerate(order): draw.rectangle(proposal_boxes[i].tolist(),outline=COLORS[rank%3],width=8)
  return rgb,int(len(proposal_boxes)),int(len(proposal_boxes))
 sl=alook[frame]; boxes=z['anchor_corners'][sl].astype(float); scores=z['anchor_scores'][sl]
 pose=np.loadtxt(RAW/'pose'/f'{frame}.txt'); K=np.loadtxt(RAW/'intrinsic'/'intrinsic_color.txt')[:3,:3]
 rects=project(boxes,pose,K,w,h); valid=[i for i,b in enumerate(rects) if b is not None]
 targets=proposal_boxes[list(TARGET_PROPOSALS[frame])]
 overlaps=rect_iou([rects[i] for i in valid],targets)
 assignments=np.argmax(overlaps,axis=1); maxima=np.max(overlaps,axis=1)
 faint=Image.new('RGBA',rgb.size,(0,0,0,0)); fd=ImageDraw.Draw(faint,'RGBA')
 for local,i in enumerate(valid):
  if maxima[local]>=.03:
   c=tuple(int(COLORS[assignments[local]][j:j+2],16) for j in (1,3,5))+(48,)
  else: c=(120,132,150,22)
  fd.rectangle(rects[i],outline=c,width=3)
 rgb=Image.alpha_composite(rgb,faint); draw=ImageDraw.Draw(rgb,'RGBA')
 # Emphasize the strongest anchors assigned to each of the same three objects.
 for target in range(3):
  candidates=[(scores[i],i) for local,i in enumerate(valid) if assignments[local]==target and maxima[local]>=.03]
  candidates.sort(key=lambda row:(-row[0],row[1]))
  for _,i in candidates[:12]: draw.rectangle(rects[i],outline=COLORS[target],width=7)
 return rgb,int(len(boxes)),int(len(valid))

def compose(kind,z,plook,alook):
 canvas=Image.new('RGB',(615,385),'white')
 offsets=((0,25),(65,12),(130,0)); counts=[]
 for frame,(x,y) in zip(FRAMES,offsets):
  im,total,valid=render_frame(frame,kind,z,plook,alook)
  im=im.convert('RGB').resize((485,360),Image.Resampling.LANCZOS)
  canvas.paste(im,(x,y)); counts.append({'frame_id':frame,'count':total,'projectable':valid})
 return canvas,counts

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with np.load(CACHE,allow_pickle=False) as z:
  plook=slices(z['frame_ids'],z['proposal_lengths']); alook=slices(z['frame_ids'],z['anchor_lengths'])
  pre,ac=compose('anchor',z,plook,alook); post,pc=compose('proposal',z,plook,alook)
 pre_path=OUT/'05a_pre_nms_anchors_multiview.png'; post_path=OUT/'05b_post_nms_proposals_multiview.png'
 pre.save(pre_path); post.save(post_path)
 manifest={'schema':'boxfusion.scene0432_candidate_pair.v1','scene':SCENE,'frames':list(FRAMES),
  'same_views':True,'cache':str(CACHE),'cache_sha256':sha(CACHE),'anchor_counts':ac,'proposal_counts':pc,
  'rendering_note':'Anchor colors preserve the original proposal identities: orange left table, green right table, blue ottoman. All anchors are faintly drawn and the top 12 associated anchors per object/view are emphasized.',
  'outputs':{'pre_nms_anchors':{'path':str(pre_path),'sha256':sha(pre_path)},'post_nms_proposals':{'path':str(post_path),'sha256':sha(post_path)}}}
 (OUT/'candidate_pair_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__': main()

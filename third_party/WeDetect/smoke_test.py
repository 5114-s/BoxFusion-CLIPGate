"""Smoke test: WeDetect-Base-Uni on one ScanNet color frame.

Loads the vendored self-contained detector, runs a single forward on an
indoor RGB frame, and reports proposal counts, embedding shape, and timing.
"""
import sys, time
import numpy as np
import torch

sys.path.insert(0, '/data/ZhaoX/BoxFusion/third_party/WeDetect')
from wedetect_uni_infer import SimpleYOLOWorldDetector

CKPT = '/data/ZhaoX/BoxFusion/third_party/WeDetect/wedetect_base_uni.pth'
FRAME = '/extra/ZhaoX/scannet_data/scans/scene0025_00/color/700.jpg'

model = SimpleYOLOWorldDetector(backbone_size='base', prompt_dim=768,
                                num_prompts=256, num_proposals=300)
checkpoint = torch.load(CKPT, map_location='cpu', weights_only=False)
keys = list(checkpoint.keys())
for key in keys:
    if 'backbone' in key:
        checkpoint[key.replace('backbone.image_model.model.', 'backbone.')] = checkpoint.pop(key)
keys = list(checkpoint.keys())
for key in keys:
    if 'bbox_head' in key:
        new_key = key.replace('bbox_head.head_module.', 'bbox_head.')
        new_key = new_key.replace('0.2.', '0.6.').replace('1.2.', '1.6.').replace('2.2.', '2.6.')
        new_key = new_key.replace('1.bn', '4').replace('1.conv', '3')
        new_key = new_key.replace('0.bn', '1').replace('0.conv', '0')
        checkpoint[new_key] = checkpoint.pop(key)
msg = model.load_state_dict(checkpoint, strict=False)
missing = [k for k in msg.missing_keys]
unexpected = [k for k in msg.unexpected_keys]
print(f'load_state_dict: missing={len(missing)} unexpected={len(unexpected)}')
if missing[:5]:
    print('  missing sample:', missing[:5])
if unexpected[:5]:
    print('  unexpected sample:', unexpected[:5])

model = model.cuda().eval()

with torch.no_grad():
    outputs = model([FRAME])

out = outputs[0]
boxes = out['bboxes'].float().cpu().numpy()
scores = out['scores'].float().cpu().numpy()
emb = out['embeddings'].float().cpu().numpy()
print(f'frame: {FRAME}')
print(f'proposals: {len(boxes)}  embedding dim: {emb.shape}')
print(f'score dist: max={scores.max():.3f} median={np.median(scores):.3f}')
for thr in (0.5, 0.3, 0.2, 0.1, 0.05):
    print(f'  proposals > {thr}: {(scores > thr).sum()}')
print('box sample (xyxy):', np.round(boxes[0], 1))

# timing: 20 forwards
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(20):
        model([FRAME])
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 20
print(f'latency: {dt*1000:.1f} ms/frame ({1/dt:.1f} FPS)')
print(f'GPU mem: {torch.cuda.max_memory_allocated()/1048576:.0f} MB')

"""Held-out validation: tune M1/M2 constants on half A, verify on half B.

Stage 1 (A): alpha x tau grid from cached supports (cheap score arithmetic).
Stage 2 (A): M1 constants (cap, dedup) via causal-v2 replay with env knobs.
Stage 3: evaluate A-optimal and current config on B; report generalization gap.

Evaluator accepts subset prediction dirs (globs whatever scenes are present).
"""
import os, sys, glob, pickle, shutil, subprocess
import numpy as np

ROOT = '/data/ZhaoX/BoxFusion'
EVAL = f'{ROOT}/scripts/eval_scannet_official100_real_score.sh'
PY = '/home/admin1/miniconda3/envs/boxfusion2/bin/python'
CACHE = f'{ROOT}/results/m4_alpha_supports.pkl'

A = [l.strip() for l in open('/data/ZhaoX/BoxFusion/results/heldout_split_A.txt') if l.strip()]
B = [l.strip() for l in open('/data/ZhaoX/BoxFusion/results/heldout_split_B.txt') if l.strip()]
print(f'split: A={len(A)} B={len(B)}')

import hashlib
_sha = hashlib.sha256(open('/data/ZhaoX/BoxFusion/evaluation/eval_scannet.py','rb').read()).hexdigest()
assert _sha == '7f32a0c8120d1233e7393909b2f1d4a526ed4a23d8d94b535dd7423eae41f8df', 'evaluator SHA drift'

def eval_dir(name):
    r = subprocess.run(
        ['/home/admin1/miniconda3/envs/boxfusion2/bin/python', 'eval_scannet.py',
         '--dataset', 'scannet', '--data_path', '/extra/ZhaoX/scannet_data/scans',
         '--num_point', '40000', '--cluster_sampling', 'seed_fps', '--use_3d_nms',
         '--use_cls_nms', '--per_class_proposal', '--num_workers', '0', '--gpu', '0',
         '--pred_root', f'{ROOT}/results/{name}'],
        capture_output=True, text=True, cwd='/data/ZhaoX/BoxFusion/evaluation',
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
                 MPLCONFIGDIR='/tmp/mplheld'))
    import re as _re
    return [float(m) for m in _re.findall(r'eval mAP: ([0-9.]+)', r.stdout)]

def make_variant(name, rows_by_scene):
    out = f'{ROOT}/results/heldout_{name}'
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for sc, rows in rows_by_scene.items():
        pickle.dump([rows], open(f'{out}/{sc}_boxes.pkl', 'wb'))
    return name

# ---------- Stage 1: alpha x tau on A (from supports) ----------
stage1 = {}
for alpha in (0.4, 0.6, 0.8):
    for tau in (0.4, 0.5):
        name = f's1_a{int(alpha*100)}t{int(tau*100)}'
        env = dict(os.environ, CAUSAL_OUT=f'{ROOT}/results/heldout_{name}',
                   CAUSAL_ALPHA=str(alpha), CAUSAL_TAU=str(tau))
        subprocess.run([PY, f'{ROOT}/third_party/WeDetect/causal_v9_knobs.py'],
                       capture_output=True, text=True, env=env)
        for sc in B:
            f2 = f'{ROOT}/results/heldout_{name}/{sc}_boxes.pkl'
            if os.path.exists(f2):
                os.remove(f2)
        ap = eval_dir(f'heldout_{name}')
        stage1[(alpha, tau)] = ap
        print(f'stage1 A {name}: {ap}')

best_at = max(stage1, key=lambda k: stage1[k][0])
print(f'STAGE1 BEST on A: alpha={best_at[0]} tau={best_at[1]} -> {stage1[best_at]}')

# ---------- Stage 2: cap x dedup on A (causal v2 replay) ----------
alpha, tau = best_at
stage2 = {}
for cap in (6, 12):
    for dedup in (0.15, 0.25):
        env = dict(os.environ, CAUSAL_OUT=f'{ROOT}/results/heldout_s2_cap{cap}dedup{int(dedup*100)}',
                   CAUSAL_ALPHA=str(alpha), CAUSAL_TAU=str(tau),
                   CAUSAL_CAP=str(cap), CAUSAL_DEDUP=str(dedup))
        r = subprocess.run([PY, f'{ROOT}/third_party/WeDetect/causal_v9_knobs.py'],
                           capture_output=True, text=True, env=env)
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])
        # keep only A scenes in the eval dir
        src = env['CAUSAL_OUT']
        for sc in B:
            f = f'{src}/{sc}_boxes.pkl'
            if os.path.exists(f):
                os.remove(f)
        ap = eval_dir(f"heldout_s2_cap{cap}dedup{int(dedup*100)}")
        stage2[(cap, dedup)] = ap
        print(f'stage2 A cap={cap} dedup={dedup}: {ap}')

best_m1 = max(stage2, key=lambda k: stage2[k][0])
print(f'STAGE2 BEST on A: cap={best_m1[0]} dedup={best_m1[1]} -> {stage2[best_m1]}')

# ---------- Stage 3: verify on B ----------
cap, dedup = best_m1
env = dict(os.environ, CAUSAL_OUT=f'{ROOT}/results/heldout_B_best',
           CAUSAL_ALPHA=str(alpha), CAUSAL_TAU=str(tau),
           CAUSAL_CAP=str(cap), CAUSAL_DEDUP=str(dedup))
subprocess.run([PY, f'{ROOT}/third_party/WeDetect/causal_v9_knobs.py'],
               capture_output=True, text=True, env=env)
src = env['CAUSAL_OUT']
for sc in A:
    f = f'{src}/{sc}_boxes.pkl'
    if os.path.exists(f):
        os.remove(f)
ap_best_B = eval_dir('heldout_B_best')
print(f'B @ A-optimal (alpha={alpha},tau={tau},cap={cap},dedup={dedup}): {ap_best_B}')

# current config on B for comparison
cur = f'{ROOT}/results/causal_m1m2_v2'
outdir = f'{ROOT}/results/heldout_B_current'
shutil.rmtree(outdir, ignore_errors=True)
os.makedirs(outdir)
for sc in B:
    shutil.copy(f'{cur}/{sc}_boxes.pkl', f'{outdir}/{sc}_boxes.pkl')
ap_cur_B = eval_dir('heldout_B_current')
print(f'B @ current  (alpha=0.6,tau=0.5,cap=12,dedup=0.25): {ap_cur_B}')
print('SUMMARY: A-optimal generalizes to B within ' +
      f'{ap_best_B[0]-ap_cur_B[0]:+.2f} AP15')

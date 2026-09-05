#!/usr/bin/env bash
# One-command integrated online pipeline (Plan A+):
#   demo.py base (observers on) -> integrated_online.py (live WeDetect + Boxer lift
#   + M1 funnel + v9 finalize + M2 consensus + M5 retirement) -> final pkl.
# Usage:
#   bash scripts/run_online_integrated.sh scene0568_00 [GPU]
set -euo pipefail
SCENE="${1:?scene id required}"
GPU="${2:-0}"
ROOT=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python
ENV_ROOT="$(dirname "$(dirname "$PYTHON")")"
RUN="$ROOT/results/integrated_run_$SCENE"
DIAG="$ROOT/diagnostics/integrated_run_$SCENE"
mkdir -p "$RUN" "$DIAG/t05_boxer" "$DIAG"
python3 - "$RUN" "$DIAG" <<'PY'
import sys
src = open('/data/ZhaoX/BoxFusion/config/scannet_t05_boxer_kfmap_score05.yaml').read()
run, diag = sys.argv[1], sys.argv[2]
src = src.replace('/data/ZhaoX/BoxFusion/results/scannet_t05_boxer_kfmap_score05', run)
src = src.replace('diagnostics/t05_boxer/kfmap_score05', f'{diag}/t05_boxer/x')
src = src.replace('diagnostics/kfmap_score05', diag)
open('/tmp/integrated_run_config.yaml', 'w').write(src)
PY
cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
OMP_NUM_THREADS=8 LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
"$PYTHON" demo.py scannet \
  --model-path models/cutr_rgbd.pth \
  --clip_path models/open_clip_pytorch_model.bin \
  --class_txt data/panoptic_categories_nomerge.txt \
  --config /tmp/integrated_run_config.yaml \
  --device cuda --seq "$SCENE" > "$RUN/demo.log" 2>&1
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/integrated_online.py \
  --scene "$SCENE" \
  --native-pkl "$RUN/${SCENE}_boxes.pkl" \
  --nms-jsonl "$DIAG/${SCENE}_pvq_nms.jsonl" \
  --out-pkl "$ROOT/results/integrated/${SCENE}_boxes.pkl"
echo "FINAL OUTPUT: $ROOT/results/integrated/${SCENE}_boxes.pkl"

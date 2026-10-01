#!/usr/bin/env bash
# Live-CuTR deployable benchmark: 10 scenes, solo GPU, no observers, production Cbest config.
set -u
ROOT=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python
ENV_ROOT="$(dirname "$(dirname "$PYTHON")")"
cd "$ROOT"
while pgrep -f 'fpsbench' > /dev/null 2>&1; do sleep 60; done   # wait for fpsbench GPU 0 to finish
while pgrep -f 'm4_alpha' > /dev/null 2>&1; do sleep 60; done   # and the M4 chain
echo "=== livebench start $(date) ==="
while read -r scene; do
  [[ -s "results/scannet_t05_boxer_livebench/${scene}_boxes.pkl" ]] && { echo "skip $scene (done)"; continue; }
  echo "--- $scene $(date '+%T')"
  CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
  OMP_NUM_THREADS=8 LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
  "$PYTHON" demo.py scannet \
    --model-path models/cutr_rgbd.pth \
    --clip_path models/open_clip_pytorch_model.bin \
    --class_txt data/panoptic_categories_nomerge.txt \
    --config config/scannet_t05_boxer_livebench.yaml \
    --device cuda --seq "$scene" > "logs/livebench/${scene}.log" 2>&1
  grep -E 'Average FPS' "logs/livebench/${scene}.log" | tail -1
done < /tmp/livebench_scenes.txt
echo "=== livebench done $(date) ==="

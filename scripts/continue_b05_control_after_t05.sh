#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
T05_LOG="$ROOT/logs/cgf_score05_topk/official100_queue.log"
B05_EXPERIMENT=scannet_b05_control_score05
B05_ROOT="$ROOT/results/$B05_EXPERIMENT"
LOG_ROOT="$ROOT/logs/cgf_score05_topk"
QUEUE_LOG="$LOG_ROOT/b05_current_after_t05.log"
PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$QUEUE_LOG") 2>&1

verify_sha() {
    local expected="$1"
    local relative="$2"
    local actual
    actual=$(sha256sum "$ROOT/$relative" | awk '{print $1}')
    if [[ "$actual" != "$expected" ]]; then
        echo "SHA256 mismatch for $relative: expected=$expected actual=$actual" >&2
        return 1
    fi
}

verify_pair() {
    verify_sha 6e18e061c4153e6bbd8864cdee687eee63bb71adea98ca2a00813d4d8d49994a demo.py
    verify_sha 0fc5ff77f9fcbe55cfd79501066e4eb5f1d87abb0a3f6df38d2f5b651202d42b boxfusion/instances.py
    verify_sha dc66d26a09555bf3f95684d5a9d8c2d73811469e328a5c400af40283298fe2c8 boxfusion/box_manager.py
    verify_sha 76a1be9d2202527e50fc8e0d2c598367309812a45ff6cd0ca6405bfe19bcea23 boxfusion/box_fusion.py
    verify_sha e5cd196edba19dd92379d3fc865f48dbb656e4a684c4525e93610b9749c7231a boxfusion/reliable_views.py
    verify_sha f27d123bc4f470a5e434e1516447d88376626a14113b6ccbdd59a8b0e7838942 tools/utils.py
    verify_sha a2bfadbe1ac1ec6bf54eca9c7fd01ee67c611b0b8d52966d874ae82c9274b25a boxfusion/capture_stream.py
    verify_sha 596b42b22828360aa780a95f188244fcef4ef69d4ee0096a37c7b8094daebe4c config/scannet_topk_fusion_score05.yaml
    verify_sha e9b8de94c49348230614a6604e6b548b05bf69f8a4f472524cca98d89c4739da config/scannet_b05_control_score05.yaml
    verify_sha 4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5 evaluation/data_util/meta_data/scannetv2_val.txt
    verify_sha 856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217 models/cutr_rgbd.pth
    verify_sha 9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4 models/open_clip_pytorch_model.bin
    verify_sha aea2a72940b7cc53ee273f9f235e2efc848e1994e22da5f439af9751e1e27c27 upstream_clean/BoxFusion_shallow/evaluation/eval_scannet.py
}

echo "[$(date '+%F %T')] Waiting for the frozen T05 official100 completion marker"
ready=0
for _ in $(seq 1 1440); do
    if [[ -f "$T05_LOG" ]] && rg -q \
        "T05 official100 inference and constant-score evaluation completed" \
        "$T05_LOG"; then
        ready=1
        break
    fi
    sleep 30
done
if [[ "$ready" -ne 1 ]]; then
    echo "T05 did not complete within the 12-hour bounded wait" >&2
    exit 1
fi

echo "[$(date '+%F %T')] Preflight: same-root B05/T05 semantic pair"
verify_pair
"$PYTHON" - "$ROOT" <<'PY'
import copy
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
b05 = yaml.safe_load((root / "config/scannet_b05_control_score05.yaml").read_text())
t05 = yaml.safe_load((root / "config/scannet_topk_fusion_score05.yaml").read_text())
b05_output = b05["data"].pop("output_dir")
t05_output = t05["data"].pop("output_dir")
b05_enabled = b05["box_fusion"]["reliable_views"].pop("enabled")
t05_enabled = t05["box_fusion"]["reliable_views"].pop("enabled")
if b05 != t05:
    raise SystemExit("B05/T05 configs differ outside output_dir and reliable_views.enabled")
if b05_enabled is not False or t05_enabled is not True:
    raise SystemExit("B05/T05 reliable-view switches are invalid")
print("semantic pair passed", b05_output, t05_output)
PY

existing=0
if [[ -d "$B05_ROOT" ]]; then
    existing=$(find "$B05_ROOT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)
fi
if [[ "$existing" -ne 0 ]]; then
    echo "Fresh B05 output root is not empty: $existing predictions" >&2
    exit 1
fi

echo "[$(date '+%F %T')] Starting fresh same-root B05 official100 on GPUs 0,1"
bash "$ROOT/scripts/run_scannet_b05_control_score05.sh" 0,1

echo "[$(date '+%F %T')] Postflight: B05/T05 sources remain frozen"
verify_pair
found=$(find "$B05_ROOT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)
if [[ "$found" -ne 100 ]]; then
    echo "Fresh B05 expected 100 predictions, found $found" >&2
    exit 1
fi
if rg -q "Appearance gate summary|Reliable-view fusion summary" \
    "$ROOT/logs/$B05_EXPERIMENT/scenes"; then
    echo "An optional module unexpectedly ran in fresh B05" >&2
    exit 1
fi

echo "[$(date '+%F %T')] Evaluating fresh B05 with score=1.0"
bash "$ROOT/scripts/eval_scannet_cgf_paper100_constant_score.sh" "$B05_EXPERIMENT"

BOOTSTRAP="$ROOT/tools/scannet_b05_t05_paired_bootstrap.py"
BOOTSTRAP_OUT="$LOG_ROOT/scannet_b05_t05_paired_bootstrap_10000.json"
if [[ ! -f "$BOOTSTRAP" ]]; then
    echo "Missing audited paired bootstrap tool: $BOOTSTRAP" >&2
    exit 1
fi
echo "[$(date '+%F %T')] Starting 10,000-replicate paired-scene bootstrap"
"$PYTHON" "$BOOTSTRAP" \
    --baseline "$B05_ROOT" \
    --treatment "$ROOT/results/scannet_topk_fusion_score05" \
    --replicates 10000 \
    --seed 20260822 \
    --out "$BOOTSTRAP_OUT"

echo "[$(date '+%F %T')] Fresh B05, T05 and paired bootstrap completed"

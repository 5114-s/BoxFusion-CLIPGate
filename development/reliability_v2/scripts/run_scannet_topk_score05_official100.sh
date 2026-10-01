#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
EXPERIMENT=scannet_topk_fusion_score05
BASELINE_ROOT="$ROOT/upstream_clean/scorefix_results/scannet"
RESULT_ROOT="$ROOT/results/$EXPERIMENT"
LOG_ROOT="$ROOT/logs/cgf_score05_topk"
QUEUE_LOG="$LOG_ROOT/official100_queue.log"
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

verify_frozen_inputs() {
    verify_sha 6e18e061c4153e6bbd8864cdee687eee63bb71adea98ca2a00813d4d8d49994a demo.py
    verify_sha 0fc5ff77f9fcbe55cfd79501066e4eb5f1d87abb0a3f6df38d2f5b651202d42b boxfusion/instances.py
    verify_sha dc66d26a09555bf3f95684d5a9d8c2d73811469e328a5c400af40283298fe2c8 boxfusion/box_manager.py
    verify_sha 76a1be9d2202527e50fc8e0d2c598367309812a45ff6cd0ca6405bfe19bcea23 boxfusion/box_fusion.py
    verify_sha e5cd196edba19dd92379d3fc865f48dbb656e4a684c4525e93610b9749c7231a boxfusion/reliable_views.py
    verify_sha f27d123bc4f470a5e434e1516447d88376626a14113b6ccbdd59a8b0e7838942 tools/utils.py
    verify_sha a2bfadbe1ac1ec6bf54eca9c7fd01ee67c611b0b8d52966d874ae82c9274b25a boxfusion/capture_stream.py
    verify_sha 596b42b22828360aa780a95f188244fcef4ef69d4ee0096a37c7b8094daebe4c config/scannet_topk_fusion_score05.yaml
    verify_sha 0f30e897c780807adb5d312e2ad1b9ab9ef520cf36a525c83b2f394a9f6d0ff3 scripts/run_scannet_topk_fusion_score05.sh
    verify_sha bbb800d5436b2671ba461db1cb6ff4d888c1bdf17e47c7f5479dd362d2810a4f scripts/run_scannet_clip_gate_topk_fusion_scorefix.sh
    verify_sha b223e61b7db2c94dea588fea87394f76ba678765107610354efe163d363d82b5 scripts/eval_scannet_cgf_paper100_constant_score.sh
    verify_sha 4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5 evaluation/data_util/meta_data/scannetv2_val.txt
    verify_sha 856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217 models/cutr_rgbd.pth
    verify_sha 9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4 models/open_clip_pytorch_model.bin
    verify_sha 0d628e3140d491acfce107268fe51233e1df44f84581f582fe253842fc6557c9 data/panoptic_categories_nomerge.txt
    verify_sha 49ab2384fbc01406eb7eb24ce89403bbfa9516bc213e11e8cd2014fa8eeea197 data/class_features.pt
    verify_sha aea2a72940b7cc53ee273f9f235e2efc848e1994e22da5f439af9751e1e27c27 upstream_clean/BoxFusion_shallow/evaluation/eval_scannet.py
    verify_sha c2b08890cf6b6497165d7d7af0bf16f9205a65698c197639db70adf702f27d6f upstream_clean/BoxFusion_shallow/evaluation/utils/ap_helper.py
    verify_sha 6ef54c395e46716e364547115090bae96643bf346b3e8eb1b859719781a557dd upstream_clean/BoxFusion_shallow/evaluation/utils/eval_det.py
}

echo "[$(date '+%F %T')] Preflight: frozen code, models, protocol and B05 baseline"
verify_frozen_inputs

"$PYTHON" - "$ROOT" "$BASELINE_ROOT" <<'PY'
import hashlib
import pickle
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline = Path(sys.argv[2])
scenes = [
    line.strip()
    for line in (root / "evaluation/data_util/meta_data/scannetv2_val.txt")
    .read_text()
    .splitlines()
    if line.strip()
]
if len(scenes) != 100 or len(set(scenes)) != 100:
    raise SystemExit("official scene list is not exactly 100 unique scenes")
expected_files = {f"{scene}_boxes.pkl" for scene in scenes}
actual_files = {path.name for path in baseline.glob("scene*_boxes.pkl")}
if actual_files != expected_files:
    raise SystemExit("B05 baseline file set differs from the official scene list")
digest = hashlib.sha256()
box_count = 0
for path in sorted(baseline.glob("scene*_boxes.pkl")):
    payload = path.read_bytes()
    digest.update(path.name.encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
    rows = pickle.loads(payload)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
        raise SystemExit(f"invalid B05 prediction container: {path}")
    box_count += len(rows[0])
if digest.hexdigest() != "9132319261b38c920b63f6968ba5d1cc7f8675758a7d2c7ac05aca8a3c4ebca4":
    raise SystemExit("B05 baseline aggregate digest mismatch")
if box_count != 1786:
    raise SystemExit(f"B05 baseline expected 1786 boxes, found {box_count}")

import yaml
cfg = yaml.safe_load((root / "config/scannet_topk_fusion_score05.yaml").read_text())
assert cfg["detection"]["score_thresh"] == 0.5
assert cfg["association"]["appearance_gate"]["enabled"] is False
assert cfg["box_fusion"]["reliable_views"]["enabled"] is True
assert cfg["box_fusion"]["reliable_views"]["top_k"] == 3
print("B05 baseline and T05 configuration audit passed")
PY

echo "[$(date '+%F %T')] Re-evaluating frozen B05 with score=1.0"
bash "$ROOT/scripts/eval_scannet_cgf_paper100_constant_score.sh" \
    scannet_b05_frozen "$BASELINE_ROOT"

echo "[$(date '+%F %T')] Starting T05 Reliable-View Top-K official100 on GPUs 0,1"
bash "$ROOT/scripts/run_scannet_topk_fusion_score05.sh" 0,1

echo "[$(date '+%F %T')] Postflight: code and protocol must still be frozen"
verify_frozen_inputs

expected=$(awk 'END {print NR}' "$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt")
found=$(find "$RESULT_ROOT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)
if [[ "$expected" -ne 100 || "$found" -ne "$expected" ]]; then
    echo "T05 output completeness failure: expected=$expected found=$found" >&2
    exit 1
fi
if rg -q "Appearance gate summary" "$ROOT/logs/$EXPERIMENT/scenes"; then
    echo "Appearance gate unexpectedly ran in T05" >&2
    exit 1
fi
summary_count=$(rg -l "Reliable-view fusion summary" "$ROOT/logs/$EXPERIMENT/scenes" | wc -l)
if [[ "$summary_count" -ne 100 ]]; then
    echo "Expected 100 Reliable-view summaries, found $summary_count" >&2
    exit 1
fi

echo "[$(date '+%F %T')] Evaluating T05 with score=1.0"
bash "$ROOT/scripts/eval_scannet_cgf_paper100_constant_score.sh" "$EXPERIMENT"

echo "[$(date '+%F %T')] T05 official100 inference and constant-score evaluation completed"

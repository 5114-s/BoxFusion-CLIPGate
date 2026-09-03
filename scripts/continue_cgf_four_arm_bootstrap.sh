#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
UPSTREAM_SESSION=cgf_topk_after_p0
QUEUE_LOG="$ROOT/logs/cgf_paper100_constant_score/p0_to_topk_only_queue.log"
BOOTSTRAP_ROOT="$ROOT/logs/cgf_paper100_constant_score/bootstrap"
SCRIPT="$BOOTSTRAP_ROOT/scannet_four_arm_paired_bootstrap.py"
OUT="$BOOTSTRAP_ROOT/scannet_four_arm_paired_bootstrap_10000.json"
STDOUT_LOG="$BOOTSTRAP_ROOT/scannet_four_arm_paired_bootstrap_10000.log"
PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python

mkdir -p "$BOOTSTRAP_ROOT"
exec > >(tee -a "$STDOUT_LOG") 2>&1

echo "[$(date '+%F %T')] Waiting for $UPSTREAM_SESSION"
while tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; do
    sleep 30
done

if ! rg -q "P0 and Top-K-only queue completed" "$QUEUE_LOG"; then
    echo "Upstream P0/Top-K-only queue did not complete successfully" >&2
    exit 1
fi

echo "[$(date '+%F %T')] Starting four-arm paired bootstrap"
"$PYTHON" "$SCRIPT" \
    --p0 "$ROOT/results/scannet_cgf_paper100_score04" \
    --p1 "$ROOT/results/scannet_clip_gate_score04" \
    --t "$ROOT/results/scannet_topk_fusion_score04" \
    --p2 "$ROOT/results/scannet_clip_gate_topk_fusion_score04" \
    --replicates 10000 \
    --seed 20260822 \
    --out "$OUT"
echo "[$(date '+%F %T')] Four-arm paired bootstrap completed: $OUT"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
P0_SESSION=cgf_p0_score04
P0_EXPERIMENT=scannet_cgf_paper100_score04
T_EXPERIMENT=scannet_topk_fusion_score04
META="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
HASH_MANIFEST="$ROOT/logs/cgf_paper100_constant_score/root_code_before_p0_t.sha256"
QUEUE_LOG="$ROOT/logs/cgf_paper100_constant_score/p0_to_topk_only_queue.log"

mkdir -p "$(dirname "$QUEUE_LOG")"
exec > >(tee -a "$QUEUE_LOG") 2>&1

echo "[$(date '+%F %T')] Waiting for $P0_SESSION"
while tmux has-session -t "$P0_SESSION" 2>/dev/null; do
    sleep 30
done

validate_prediction_set() {
    local experiment="$1"
    local prediction_root="$ROOT/results/$experiment"
    local expected
    local found
    expected=$(awk 'END {print NR}' "$META")
    found=$(find "$prediction_root" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)
    if [[ "$found" -ne "$expected" ]]; then
        echo "Expected $expected predictions for $experiment, found $found" >&2
        return 1
    fi
    while IFS= read -r scene || [[ -n "$scene" ]]; do
        if [[ ! -s "$prediction_root/${scene}_boxes.pkl" ]]; then
            echo "Missing official scene for $experiment: $scene" >&2
            return 1
        fi
    done < "$META"
}

if ! rg -q "inference and evaluation completed" "$ROOT/logs/$P0_EXPERIMENT/driver.log"; then
    echo "$P0_EXPERIMENT did not reach its completion marker" >&2
    exit 1
fi
validate_prediction_set "$P0_EXPERIMENT"

echo "[$(date '+%F %T')] Evaluating P0 with the frozen constant-score evaluator"
bash "$ROOT/scripts/eval_scannet_cgf_paper100_constant_score.sh" "$P0_EXPERIMENT"

echo "[$(date '+%F %T')] Verifying frozen P0/T provenance"
(
    cd "$ROOT"
    sha256sum -c "$HASH_MANIFEST"
)

existing_topk=$(find "$ROOT/results/$T_EXPERIMENT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' 2>/dev/null | wc -l)
if [[ "$existing_topk" -ne 0 ]]; then
    echo "Refusing to mix $existing_topk pre-existing Top-K-only predictions" >&2
    exit 1
fi

echo "[$(date '+%F %T')] Starting Top-K-only on GPUs 0,1"
bash "$ROOT/scripts/run_scannet_topk_fusion_score04.sh" 0,1
validate_prediction_set "$T_EXPERIMENT"

echo "[$(date '+%F %T')] Evaluating Top-K-only with the frozen constant-score evaluator"
bash "$ROOT/scripts/eval_scannet_cgf_paper100_constant_score.sh" "$T_EXPERIMENT"
echo "[$(date '+%F %T')] P0 and Top-K-only queue completed"

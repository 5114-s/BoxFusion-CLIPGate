#!/usr/bin/env bash
set -euo pipefail

# Isolated B6 + Selective-Boxer + terminal-p100 R3 cache-replay runner.
#
# Usage:
#   BOXFUSION_YOLOE_CHECKPOINT=/path/yoloe-11s-seg-pf.pt \
#     bash scripts/run_scannet_tr3d_terminal_active.sh 0
#
#   # Two independent scene shards:
#   BOXFUSION_YOLOE_CHECKPOINT=/path/yoloe-11s-seg-pf.pt \
#     bash scripts/run_scannet_tr3d_terminal_active.sh 0,1
#
# Optional learned heads:
#   BOXFUSION_REFINER_CHECKPOINT=/path/box_refiner.pt
#   BOXFUSION_QUALITY_CHECKPOINT=/path/quality_linear.npz
#   BOXFUSION_QUALITY_MODE=linear  # or iou_mlp

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
CONFIG="${BOXFUSION_ONLINE_CONFIG:-$ROOT/config/scannet_online_refinement.yaml}"
META="${BOXFUSION_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
PRED_ROOT="${BOXFUSION_ONLINE_PRED_ROOT:-$ROOT/results/scannet_online_refinement}"
LOG_ROOT="${BOXFUSION_ONLINE_LOG_ROOT:-$ROOT/logs/scannet_online_refinement}"
DIAGNOSTICS_ROOT="${BOXFUSION_DIAGNOSTICS_ROOT:-$ROOT/results/scannet_online_refinement_diagnostics}"
R3_SAME_RUN_BASELINE_ROOT="${BOXFUSION_R3_SAME_RUN_BASELINE_ROOT:-${PRED_ROOT}_same_run_baseline}"
R3_PREFIX_MANIFEST="${BOXFUSION_R3_PREFIX_MANIFEST:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/data/tr3d_prefix_val100_boxfusion_causal_p100_v3/manifests/trajectory_prefix_val100_boxfusion_causal_p100_v3.jsonl}"
R3_PARENT_CACHE_ROOT="${BOXFUSION_R3_PARENT_CACHE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/cache/tr3d_prefix_boxfusion_causal_p100_full100_v3}"
R3_DIAGNOSTICS_ROOT="${BOXFUSION_R3_DIAGNOSTICS_ROOT:-$DIAGNOSTICS_ROOT/tr3d_terminal}"
R3_FROZEN_G0_ROOT="${BOXFUSION_R3_FROZEN_G0_ROOT:-$ROOT/results/b6_selective_boxer/s1_selective/scannetv2_val-4b18fc586f7a}"
R3_SHADOW_GOLD_ROOT="${BOXFUSION_R3_SHADOW_GOLD_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/results/tr3d_r3_shadow_active/r3_shadow_active_full100_v1}"
R3_FROZEN_MANIFEST="${BOXFUSION_R3_FROZEN_MANIFEST:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/manifests/frozen_g0_selective_boxer_full100.json}"
R3_SHADOW_MANIFEST="${BOXFUSION_R3_SHADOW_MANIFEST:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/reports/tr3d_r3_shadow_active/r3_shadow_active_full100_v1/materialize_manifest.json}"
C3_ONLINE_ENABLED="${BOXFUSION_C3_ONLINE_ENABLED:-0}"
C3_ONLINE_C2_CACHE_ROOT="${BOXFUSION_C3_ONLINE_C2_CACHE_ROOT:-$ROOT/artifacts/tr3d_c2_maskrgbd/c2_c1top10_full100_v1/cache}"
C3_ONLINE_PARENT_CACHE_ROOT="${BOXFUSION_C3_ONLINE_PARENT_CACHE_ROOT:-$R3_PARENT_CACHE_ROOT}"
C3_ONLINE_CANDIDATE_SOURCE="${BOXFUSION_C3_ONLINE_CANDIDATE_SOURCE:-c2}"
C3_ONLINE_DIAGNOSTICS_ROOT="${BOXFUSION_C3_ONLINE_DIAGNOSTICS_ROOT:-$DIAGNOSTICS_ROOT/tr3d_c3_online_identity}"
C3_ONLINE_AUDIT_REPORT="${BOXFUSION_C3_ONLINE_AUDIT_REPORT:-$LOG_ROOT/c3_online_identity_audit.json}"
C3_ACTIVE_POLICY="${BOXFUSION_C3_ONLINE_ACTIVE_POLICY:-}"
C3_ACTIVE_OUTPUT_ROOT="${BOXFUSION_C3_ONLINE_ACTIVE_OUTPUT_ROOT:-${PRED_ROOT}_c3_active}"
C3_ACTIVE_DIAGNOSTICS_ROOT="${BOXFUSION_C3_ONLINE_ACTIVE_DIAGNOSTICS_ROOT:-${DIAGNOSTICS_ROOT}/tr3d_c3_online_active}"
INCREMENTAL_TR3D_ENABLED="${BOXFUSION_INCREMENTAL_TR3D_OBSERVER:-0}"
SKIP_EVALUATION="${BOXFUSION_SKIP_EVALUATION:-0}"
PROPOSAL_CACHE_MODE_OVERRIDE="${BOXFUSION_PROPOSAL_CACHE_MODE_OVERRIDE:-}"
INCREMENTAL_TR3D_DIAGNOSTICS_ROOT="${BOXFUSION_INCREMENTAL_TR3D_DIAGNOSTICS_ROOT:-${DIAGNOSTICS_ROOT}/tr3d_incremental}"
TR3D_WORKER_PYTHON="${BOXFUSION_TR3D_WORKER_PYTHON:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/.conda/boxfusion-tr3d/bin/python}"
TR3D_WORKER_RUNTIME_ROOT="${BOXFUSION_TR3D_WORKER_RUNTIME_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev}"
TR3D_WORKER_CONFIG="${BOXFUSION_TR3D_WORKER_CONFIG:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/config/tr3d/tr3d_scannet_foreground.py}"
TR3D_WORKER_CHECKPOINT="${BOXFUSION_TR3D_WORKER_CHECKPOINT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/work_dirs/tr3d/tr3d_fg_full_seed0_fp32_v1/epoch_12.pth}"
TR3D_WORKER_PROJECT_ROOT="${BOXFUSION_TR3D_WORKER_PROJECT_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev}"
TR3D_WORKER_VENDOR_ROOT="${BOXFUSION_TR3D_WORKER_VENDOR_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/third_party/mmdetection3d}"
TR3D_INCREMENTAL_INTERVAL="${BOXFUSION_TR3D_INCREMENTAL_EVERY_KEYFRAMES:-5}"
TR3D_LIGHTWEIGHT_FUSION="${BOXFUSION_TR3D_LIGHTWEIGHT_FUSION:-0}"
TR3D_LIGHTWEIGHT_STAGE="${BOXFUSION_TR3D_LIGHTWEIGHT_STAGE:-6}"
TR3D_LIGHTWEIGHT_TOP_K="${BOXFUSION_TR3D_LIGHTWEIGHT_TOP_K:-5}"
TR3D_LIGHTWEIGHT_DIVERSITY="${BOXFUSION_TR3D_LIGHTWEIGHT_DIVERSITY_WEIGHT:-0.30}"
TR3D_LIGHTWEIGHT_MIN_ANGLE="${BOXFUSION_TR3D_LIGHTWEIGHT_MIN_VIEW_ANGLE_DEG:-12.0}"
TR3D_LIGHTWEIGHT_DEPTH_STRIDE="${BOXFUSION_TR3D_LIGHTWEIGHT_DEPTH_STRIDE:-6}"
TR3D_LIGHTWEIGHT_DRAIN="${BOXFUSION_TR3D_LIGHTWEIGHT_DRAIN_FINALIZE:-0}"
BOXER_DIAGNOSTICS_OVERRIDE="${BOXFUSION_BOXER_DIAGNOSTICS_ROOT:-}"
BOXER_GATE_CENTER="${BOXFUSION_BOXER_GATE_MAX_CENTER_SHIFT_M:-}"
BOXER_GATE_MIN_VOLUME="${BOXFUSION_BOXER_GATE_MIN_VOLUME_RATIO:-}"
BOXER_GATE_MAX_VOLUME="${BOXFUSION_BOXER_GATE_MAX_VOLUME_RATIO:-}"
GT_ROOT="$LIVE_ROOT/evaluation/data_util/scannet_train_detection_data"
FRAMES_ROOT="${BOXFUSION_SCANNET_FRAMES_ROOT:-$LIVE_ROOT/upstream_clean/scannet_readme_frames}"
EVAL_ROOT="${BOXFUSION_EVAL_ROOT:-$ROOT/evaluation/scannet_online_refinement}"
R3_SAME_RUN_EVAL_ROOT="${BOXFUSION_R3_SAME_RUN_EVAL_ROOT:-${EVAL_ROOT}_same_run_baseline}"
EVAL_SEED="${BOXFUSION_EVAL_SEED:-0}"
INFERENCE_SEED="${BOXFUSION_INFERENCE_SEED:-0}"
YOLOE_CHECKPOINT="${BOXFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
REFINER_CHECKPOINT="${BOXFUSION_REFINER_CHECKPOINT:-}"
QUALITY_CHECKPOINT="${BOXFUSION_QUALITY_CHECKPOINT:-}"
QUALITY_MODE="${BOXFUSION_QUALITY_MODE:-linear}"
QUALITY_DETECTOR_BLEND="${BOXFUSION_QUALITY_DETECTOR_BLEND:-}"
SCANNET_MIN_EXTENT="${BOXFUSION_SCANNET_MIN_EXTENT:-}"
PROPOSAL_INTERVAL="${BOXFUSION_PROPOSAL_INTERVAL:-5}"
CANDIDATE_TTL_CLOCK="${BOXFUSION_CANDIDATE_TTL_CLOCK:-}"
CANDIDATE_TRACK_TTL="${BOXFUSION_CANDIDATE_TRACK_TTL:-}"
ARCHIVE_CONFIRMED="${BOXFUSION_ARCHIVE_CONFIRMED_TRACKS:-}"
DISABLE_ONLINE="${BOXFUSION_DISABLE_ONLINE_REFINEMENT:-0}"
ABLATION_PROFILE="${BOXFUSION_ONLINE_ABLATION_PROFILE:-}"
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"
MIN_FREE_KB="${BOXFUSION_TERMINAL_MIN_FREE_KB:-1048576}"

if [[ ! "$MIN_FREE_KB" =~ ^[1-9][0-9]*$ ]]; then
    echo "BOXFUSION_TERMINAL_MIN_FREE_KB must be a positive integer" >&2
    exit 1
fi
if [[ "$C3_ONLINE_ENABLED" != "0" && "$C3_ONLINE_ENABLED" != "1" ]]; then
    echo "BOXFUSION_C3_ONLINE_ENABLED must be 0 or 1" >&2
    exit 1
fi
if [[ "$INCREMENTAL_TR3D_ENABLED" != "0" && "$INCREMENTAL_TR3D_ENABLED" != "1" ]]; then
    echo "BOXFUSION_INCREMENTAL_TR3D_OBSERVER must be 0 or 1" >&2
    exit 1
fi
for binary_value in "$TR3D_LIGHTWEIGHT_FUSION" "$TR3D_LIGHTWEIGHT_DRAIN"; do
    if [[ "$binary_value" != "0" && "$binary_value" != "1" ]]; then
        echo "lightweight fusion/drain switches must be 0 or 1" >&2
        exit 1
    fi
done
if [[ "$TR3D_LIGHTWEIGHT_FUSION" == "1" && "$INCREMENTAL_TR3D_ENABLED" != "1" ]]; then
    echo "BOXFUSION_TR3D_LIGHTWEIGHT_FUSION=1 requires incremental observer=1" >&2
    exit 1
fi
if [[ ! "$TR3D_LIGHTWEIGHT_STAGE" =~ ^[1-6]$ ]]; then
    echo "BOXFUSION_TR3D_LIGHTWEIGHT_STAGE must be an integer from 1 to 6" >&2
    exit 1
fi
if [[ "$SKIP_EVALUATION" != "0" && "$SKIP_EVALUATION" != "1" ]]; then
    echo "BOXFUSION_SKIP_EVALUATION must be 0 or 1" >&2
    exit 1
fi
if [[ -n "$PROPOSAL_CACHE_MODE_OVERRIDE" \
      && "$PROPOSAL_CACHE_MODE_OVERRIDE" != "disabled" \
      && "$PROPOSAL_CACHE_MODE_OVERRIDE" != "replay" ]]; then
    echo "BOXFUSION_PROPOSAL_CACHE_MODE_OVERRIDE must be disabled or replay" >&2
    exit 1
fi

pred_root_canonical="$(readlink -m "$PRED_ROOT")"
baseline_root_canonical="$(readlink -m "$R3_SAME_RUN_BASELINE_ROOT")"
if [[ -n "$C3_ACTIVE_POLICY" ]]; then
    if [[ "$C3_ONLINE_ENABLED" != "1" ]]; then
        echo "C3 online active requires BOXFUSION_C3_ONLINE_ENABLED=1" >&2
        exit 1
    fi
    if [[ ! -f "$C3_ACTIVE_POLICY" || -L "$C3_ACTIVE_POLICY" \
          || -w "$C3_ACTIVE_POLICY" ]]; then
        echo "C3 active policy must be an immutable regular file: $C3_ACTIVE_POLICY" >&2
        exit 1
    fi
    for root in "$C3_ACTIVE_OUTPUT_ROOT" "$C3_ACTIVE_DIAGNOSTICS_ROOT"; do
        if [[ "$(readlink -m "$root")" == "$pred_root_canonical" \
              || "$(readlink -m "$root")" == "$baseline_root_canonical" ]]; then
            echo "C3 active roots must be isolated from R3 roots: $root" >&2
            exit 1
        fi
    done
fi
case "$baseline_root_canonical/" in
    "$pred_root_canonical/"* )
        echo "Same-run baseline root must not equal or be nested below the active root" >&2
        exit 1
        ;;
esac
case "$pred_root_canonical/" in
    "$baseline_root_canonical/"* )
        echo "Active root must not be nested below the same-run baseline root" >&2
        exit 1
        ;;
esac
if [[ "$(readlink -m "$EVAL_ROOT")" == "$(readlink -m "$R3_SAME_RUN_EVAL_ROOT")" ]]; then
    echo "Active and same-run baseline evaluation roots must be different" >&2
    exit 1
fi
for immutable_reference_root in "$R3_FROZEN_G0_ROOT" "$R3_SHADOW_GOLD_ROOT"; do
    reference_canonical="$(readlink -m "$immutable_reference_root")"
    if [[ "$reference_canonical" == "$pred_root_canonical" \
          || "$reference_canonical" == "$baseline_root_canonical" ]]; then
        echo "Writable active/baseline roots must differ from immutable audit references: $immutable_reference_root" >&2
        exit 1
    fi
done

for write_target in \
    "$PRED_ROOT" \
    "$R3_SAME_RUN_BASELINE_ROOT" \
    "$LOG_ROOT" \
    "$DIAGNOSTICS_ROOT" \
    "$R3_DIAGNOSTICS_ROOT" \
    "$EVAL_ROOT" \
    "$R3_SAME_RUN_EVAL_ROOT"; do
    probe="$write_target"
    while [[ ! -e "$probe" ]]; do
        parent_probe="$(dirname "$probe")"
        if [[ "$parent_probe" == "$probe" ]]; then
            break
        fi
        probe="$parent_probe"
    done
    available_kb="$(df -Pk "$probe" | awk 'NR == 2 {print $4}')"
    if [[ ! "$available_kb" =~ ^[0-9]+$ ]]; then
        echo "Could not determine free space for $write_target" >&2
        exit 1
    fi
    if (( available_kb < MIN_FREE_KB )); then
        echo "Refusing terminal-R3 run: only ${available_kb} KiB are free for $write_target." >&2
        echo "At least ${MIN_FREE_KB} KiB are required to avoid partial predictions/logs." >&2
        echo "Choose a spacious artifact root or free disk space; do not lower the guard for a formal run." >&2
        exit 1
    fi
done
if [[ -n "$C3_ACTIVE_POLICY" ]]; then
    for write_target in "$C3_ACTIVE_OUTPUT_ROOT" "$C3_ACTIVE_DIAGNOSTICS_ROOT"; do
        probe="$write_target"
        while [[ ! -e "$probe" ]]; do
            parent_probe="$(dirname "$probe")"
            [[ "$parent_probe" == "$probe" ]] && break
            probe="$parent_probe"
        done
        available_kb="$(df -Pk "$probe" | awk 'NR == 2 {print $4}')"
        if [[ ! "$available_kb" =~ ^[0-9]+$ ]] || (( available_kb < MIN_FREE_KB )); then
            echo "Insufficient free space for C3 active output: $write_target" >&2
            exit 1
        fi
    done
fi
if [[ "$C3_ONLINE_ENABLED" == "1" ]]; then
    c3_immutable_roots=("$C3_ONLINE_PARENT_CACHE_ROOT")
    if [[ "$C3_ONLINE_CANDIDATE_SOURCE" == "c2" ]]; then
        c3_immutable_roots+=("$C3_ONLINE_C2_CACHE_ROOT")
    elif [[ "$C3_ONLINE_CANDIDATE_SOURCE" != "parent_score" ]]; then
        echo "Invalid C3 online candidate source: $C3_ONLINE_CANDIDATE_SOURCE" >&2
        exit 1
    fi
    for immutable_root in "${c3_immutable_roots[@]}"; do
        if [[ ! -d "$immutable_root" || -L "$immutable_root" ]]; then
            echo "Missing/non-regular C3 online immutable root: $immutable_root" >&2
            exit 1
        fi
    done
    if [[ -e "$C3_ONLINE_AUDIT_REPORT" || -L "$C3_ONLINE_AUDIT_REPORT" ]]; then
        echo "Refusing existing C3 online audit report: $C3_ONLINE_AUDIT_REPORT" >&2
        exit 1
    fi
fi

mkdir -p \
    "$PRED_ROOT" \
    "$R3_SAME_RUN_BASELINE_ROOT" \
    "$LOG_ROOT/scenes" \
    "$LOG_ROOT/mplconfig" \
    "$DIAGNOSTICS_ROOT" \
    "$R3_DIAGNOSTICS_ROOT"
if [[ "$C3_ONLINE_ENABLED" == "1" ]]; then
    mkdir -p "$C3_ONLINE_DIAGNOSTICS_ROOT"
fi
if [[ -n "$C3_ACTIVE_POLICY" ]]; then
    mkdir -p "$C3_ACTIVE_OUTPUT_ROOT" "$C3_ACTIVE_DIAGNOSTICS_ROOT"
fi
exec 8>"$PRED_ROOT/.terminal_r3_run.lock"
if ! flock -n 8; then
    echo "Another terminal-R3 driver holds $PRED_ROOT/.terminal_r3_run.lock" >&2
    exit 1
fi
exec 9>"$R3_SAME_RUN_BASELINE_ROOT/.terminal_r3_same_run_baseline.lock"
if ! flock -n 9; then
    echo "Another terminal-R3 driver holds $R3_SAME_RUN_BASELINE_ROOT/.terminal_r3_same_run_baseline.lock" >&2
    exit 1
fi
exec 7>"$LOG_ROOT/run.lock"
if ! flock -n 7; then
    echo "Another online-refinement driver holds $LOG_ROOT/run.lock" >&2
    exit 1
fi
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

require_immutable_file() {
    local path="$1"
    local label="$2"
    local mode
    if [[ ! -f "$path" || -L "$path" || ! -s "$path" ]]; then
        echo "$label is missing, empty, non-regular, or a symlink: $path" >&2
        return 1
    fi
    mode="$(stat -c '%a' "$path")"
    if (( (8#$mode & 8#222) != 0 )); then
        echo "$label is still writable (mode $mode): $path" >&2
        return 1
    fi
}

marker_value() {
    local path="$1"
    local key="$2"
    awk -F= -v wanted="$key" '
        $1 == wanted { value = substr($0, length($1) + 2); count += 1 }
        END { if (count != 1) exit 1; print value }
    ' "$path"
}

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
    echo "No GPU was specified" >&2
    exit 1
fi
for gpu in "${GPUS[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU index: $gpu" >&2
        exit 1
    fi
done
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne "${#GPUS[@]}" ]]; then
    echo "GPU list contains duplicate indices: $GPU_SPEC" >&2
    exit 1
fi
if [[ ! "$PROPOSAL_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "BOXFUSION_PROPOSAL_INTERVAL must be a positive integer" >&2
    exit 1
fi
if [[ -n "$CANDIDATE_TTL_CLOCK" \
      && "$CANDIDATE_TTL_CLOCK" != "keyframe" \
      && "$CANDIDATE_TTL_CLOCK" != "provider_call" ]]; then
    echo "BOXFUSION_CANDIDATE_TTL_CLOCK must be keyframe or provider_call" >&2
    exit 1
fi
if [[ -n "$CANDIDATE_TRACK_TTL" \
      && ! "$CANDIDATE_TRACK_TTL" =~ ^[0-9]+$ ]]; then
    echo "BOXFUSION_CANDIDATE_TRACK_TTL must be a non-negative integer" >&2
    exit 1
fi
if [[ -n "$ARCHIVE_CONFIRMED" \
      && "$ARCHIVE_CONFIRMED" != "0" \
      && "$ARCHIVE_CONFIRMED" != "1" ]]; then
    echo "BOXFUSION_ARCHIVE_CONFIRMED_TRACKS must be 0 or 1" >&2
    exit 1
fi
if [[ "$QUALITY_MODE" != "linear" \
      && "$QUALITY_MODE" != "mlp" \
      && "$QUALITY_MODE" != "iou_mlp" ]]; then
    echo "BOXFUSION_QUALITY_MODE must be linear, mlp, or iou_mlp" >&2
    exit 1
fi
if [[ -n "$SCANNET_MIN_EXTENT" \
      && ! "$SCANNET_MIN_EXTENT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "BOXFUSION_SCANNET_MIN_EXTENT must be non-negative" >&2
    exit 1
fi
if [[ -n "$QUALITY_DETECTOR_BLEND" ]]; then
    if [[ ! "$QUALITY_DETECTOR_BLEND" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
        echo "BOXFUSION_QUALITY_DETECTOR_BLEND must be numeric" >&2
        exit 1
    fi
    if ! "$PYTHON" -c \
        "value=float('$QUALITY_DETECTOR_BLEND'); assert 0.0 <= value <= 1.0" \
        >/dev/null 2>&1; then
        echo "BOXFUSION_QUALITY_DETECTOR_BLEND must lie in [0, 1]" >&2
        exit 1
    fi
fi
boxer_gate_value_count=0
for boxer_gate_value in \
    "$BOXER_GATE_CENTER" "$BOXER_GATE_MIN_VOLUME" "$BOXER_GATE_MAX_VOLUME"; do
    if [[ -n "$boxer_gate_value" ]]; then
        boxer_gate_value_count=$((boxer_gate_value_count + 1))
    fi
done
if [[ "$boxer_gate_value_count" != "0" \
      && "$boxer_gate_value_count" != "3" ]]; then
    echo "All three BOXFUSION_BOXER_GATE_* overrides must be set together" >&2
    exit 1
fi
if [[ "$boxer_gate_value_count" == "3" ]]; then
    if ! "$PYTHON" -c \
        "import math; c=float('$BOXER_GATE_CENTER'); lo=float('$BOXER_GATE_MIN_VOLUME'); hi=float('$BOXER_GATE_MAX_VOLUME'); assert math.isfinite(c) and c >= 0.0; assert math.isfinite(lo) and lo > 0.0; assert math.isfinite(hi) and hi >= lo" \
        >/dev/null 2>&1; then
        echo "Invalid selective-Boxer gate: require finite center>=0 and 0<min<=max" >&2
        exit 1
    fi
fi
if [[ "$DISABLE_ONLINE" != "0" && "$DISABLE_ONLINE" != "1" ]]; then
    echo "BOXFUSION_DISABLE_ONLINE_REFINEMENT must be 0 or 1" >&2
    exit 1
fi
if [[ -n "$ABLATION_PROFILE" ]]; then
    case "$ABLATION_PROFILE" in
        observer|quality_observer|refit_only|supplemental_only|supplemental_conservative|quality_only|full) ;;
        *)
            echo "Invalid BOXFUSION_ONLINE_ABLATION_PROFILE: $ABLATION_PROFILE" >&2
            exit 1
            ;;
    esac
    if [[ "$DISABLE_ONLINE" == "1" ]]; then
        echo "Online ablation profile conflicts with disabled refinement" >&2
        exit 1
    fi
fi
if [[ ! "$EVAL_SEED" =~ ^[0-9]+$ ]]; then
    echo "BOXFUSION_EVAL_SEED must be a non-negative integer" >&2
    exit 1
fi
if [[ ! "$INFERENCE_SEED" =~ ^[0-9]+$ ]]; then
    echo "BOXFUSION_INFERENCE_SEED must be a non-negative integer" >&2
    exit 1
fi

required_files=(
    "$PYTHON"
    "$CONFIG"
    "$META"
    "$LIVE_ROOT/models/cutr_rgbd.pth"
    "$LIVE_ROOT/models/open_clip_pytorch_model.bin"
    "$LIVE_ROOT/data/panoptic_categories_nomerge.txt"
    "$LIVE_ROOT/data/class_features.pt"
    "$LIVE_ROOT/data/pst_1024_0.tiff"
    "$R3_PREFIX_MANIFEST"
    "$R3_FROZEN_MANIFEST"
    "$R3_SHADOW_MANIFEST"
    "$ROOT/demo_tr3d_terminal_active.py"
    "$ROOT/boxfusion/tr3d_terminal_active.py"
    "$ROOT/tools/audit_tr3d_terminal_active.py"
    "$ROOT/evaluation/eval_scannet.py"
)
if [[ "$DISABLE_ONLINE" == "0" ]]; then
    required_files+=("$YOLOE_CHECKPOINT")
fi
if [[ "$INCREMENTAL_TR3D_ENABLED" == "1" ]]; then
    required_files+=(
        "$TR3D_WORKER_PYTHON" "$TR3D_WORKER_CONFIG"
        "$TR3D_WORKER_CHECKPOINT" "$ROOT/tools/tr3d_online_worker.py"
    )
fi
for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        if [[ "$path" == "$YOLOE_CHECKPOINT" ]]; then
            echo "Set BOXFUSION_YOLOE_CHECKPOINT to the local YOLOE segmentation checkpoint." >&2
        fi
        exit 1
    fi
done
if [[ -n "$REFINER_CHECKPOINT" && ! -f "$REFINER_CHECKPOINT" ]]; then
    echo "Missing BoxRefiner checkpoint: $REFINER_CHECKPOINT" >&2
    exit 1
fi
if [[ -n "$QUALITY_CHECKPOINT" && ! -f "$QUALITY_CHECKPOINT" ]]; then
    echo "Missing quality checkpoint: $QUALITY_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -d "$FRAMES_ROOT" ]]; then
    echo "Missing ScanNet frames root: $FRAMES_ROOT" >&2
    exit 1
fi
if [[ ! -d "$R3_PARENT_CACHE_ROOT" ]]; then
    echo "Missing immutable TR3D parent cache: $R3_PARENT_CACHE_ROOT" >&2
    exit 1
fi
for r3_reference_root in "$R3_FROZEN_G0_ROOT" "$R3_SHADOW_GOLD_ROOT"; do
    if [[ ! -d "$r3_reference_root" ]]; then
        echo "Missing terminal R3 audit reference: $r3_reference_root" >&2
        exit 1
    fi
done
if [[ ! -d "$GT_ROOT" ]]; then
    echo "Missing ScanNet evaluation ground truth: $GT_ROOT" >&2
    exit 1
fi

CACHE_MODE="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("proposal_cache", {}).get("mode", "disabled"))
PY
)"
if [[ -n "$PROPOSAL_CACHE_MODE_OVERRIDE" ]]; then
    CACHE_MODE="$PROPOSAL_CACHE_MODE_OVERRIDE"
fi
CACHE_ROOT="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("proposal_cache", {}).get("root", ""))
PY
)"
CACHE_NAMESPACE="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("proposal_cache", {}).get("namespace", ""))
PY
)"
CACHE_BASELINE_ROOT="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("proposal_cache", {}).get("baseline_prediction_root", ""))
PY
)"
BOXER_DIAGNOSTICS_ROOT="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("boxer", {}).get("diagnostics_dir", ""))
PY
)"
if [[ -n "$BOXER_DIAGNOSTICS_OVERRIDE" ]]; then
    BOXER_DIAGNOSTICS_ROOT="$BOXER_DIAGNOSTICS_OVERRIDE"
fi
if [[ -n "$BOXER_DIAGNOSTICS_ROOT" ]]; then
    probe="$BOXER_DIAGNOSTICS_ROOT"
    while [[ ! -e "$probe" ]]; do
        parent_probe="$(dirname "$probe")"
        if [[ "$parent_probe" == "$probe" ]]; then
            break
        fi
        probe="$parent_probe"
    done
    available_kb="$(df -Pk "$probe" | awk 'NR == 2 {print $4}')"
    if [[ ! "$available_kb" =~ ^[0-9]+$ \
          || "$available_kb" -lt "$MIN_FREE_KB" ]]; then
        echo "Insufficient free space for Boxer diagnostics: $BOXER_DIAGNOSTICS_ROOT" >&2
        exit 1
    fi
    mkdir -p "$BOXER_DIAGNOSTICS_ROOT"
fi
if [[ "$CACHE_MODE" != "disabled" && "$CACHE_MODE" != "replay" ]]; then
    echo "This runner supports disabled or replay proposal cache, got: $CACHE_MODE" >&2
    exit 1
fi
if [[ "$CACHE_MODE" == "replay" ]]; then
    for cache_path in "$CACHE_ROOT" "$CACHE_BASELINE_ROOT"; do
        if [[ -z "$cache_path" || ! -d "$cache_path" ]]; then
            echo "Missing replay cache path: $cache_path" >&2
            exit 1
        fi
    done
    if [[ -z "$CACHE_NAMESPACE" ]]; then
        echo "Replay cache namespace is required" >&2
        exit 1
    fi
fi

LIST_SHA256="$(sha256sum "$META" | awk '{print $1}')"
CODE_FINGERPRINT="$(
    {
        sha256sum \
            "$CONFIG" \
            "$ROOT/demo_tr3d_terminal_active.py" \
            "$ROOT/boxfusion/tr3d_terminal_active.py" \
            "$ROOT/boxfusion/tr3d_c3_online_identity.py" \
            "$ROOT/tools/audit_tr3d_c3_online_identity.py" \
            "$ROOT/tools/audit_tr3d_terminal_active.py" \
            "$ROOT/evaluation/eval_scannet.py" \
            "$ROOT/scripts/run_scannet_tr3d_terminal_active.sh" \
            "$ROOT/scripts/run_scannet_b6_g0_tr3d_terminal_active.sh" \
            "$R3_PREFIX_MANIFEST" \
            "$R3_FROZEN_MANIFEST" \
            "$R3_SHADOW_MANIFEST" \
            "$ROOT/boxfusion/boxer_lifter.py" \
            "$ROOT/boxfusion/proposal_cache.py" \
            "$ROOT/boxfusion/online_refinement.py" \
            "$ROOT/boxfusion/online_ablation.py" \
            "$ROOT/boxfusion/quality_score.py" \
            "$ROOT/boxfusion/box_fusion.py" \
            "$ROOT/boxfusion/box_manager.py" \
            "$ROOT/boxfusion/tr3d_incremental_online.py" \
            "$ROOT/boxfusion/tr3d_lightweight_fusion.py" \
            "$ROOT/boxfusion/tr3d_worker_client.py" \
            "$ROOT/tools/tr3d_online_worker.py" \
            "$LIVE_ROOT/models/cutr_rgbd.pth" \
            "$LIVE_ROOT/models/open_clip_pytorch_model.bin" \
            "$YOLOE_CHECKPOINT"
        if [[ -n "$QUALITY_CHECKPOINT" ]]; then
            sha256sum "$QUALITY_CHECKPOINT"
        fi
        if [[ "$INCREMENTAL_TR3D_ENABLED" == "1" ]]; then
            sha256sum "$TR3D_WORKER_CONFIG" "$TR3D_WORKER_CHECKPOINT"
        fi
        printf '%s\n' \
            "scene_list_sha256=$LIST_SHA256" \
            "python=$(readlink -f "$PYTHON")" \
            "python_version=$("$PYTHON" --version 2>&1)" \
            "quality_mode=$QUALITY_MODE" \
            "quality_blend=$QUALITY_DETECTOR_BLEND" \
            "min_extent=$SCANNET_MIN_EXTENT" \
            "boxer_gate_center=$BOXER_GATE_CENTER" \
            "boxer_gate_min_volume=$BOXER_GATE_MIN_VOLUME" \
            "boxer_gate_max_volume=$BOXER_GATE_MAX_VOLUME" \
            "ablation_profile=$ABLATION_PROFILE" \
            "proposal_interval=$PROPOSAL_INTERVAL" \
            "candidate_ttl_clock=$CANDIDATE_TTL_CLOCK" \
            "candidate_track_ttl=$CANDIDATE_TRACK_TTL" \
            "archive_confirmed=$ARCHIVE_CONFIRMED" \
            "inference_seed=$INFERENCE_SEED" \
            "evaluation_seed=$EVAL_SEED" \
            "proposal_cache_mode=$CACHE_MODE" \
            "proposal_cache_namespace=$CACHE_NAMESPACE" \
            "same_run_baseline_root=$(readlink -m "$R3_SAME_RUN_BASELINE_ROOT")"
        printf '%s\n' \
            "incremental_tr3d_enabled=$INCREMENTAL_TR3D_ENABLED" \
            "incremental_tr3d_interval=$TR3D_INCREMENTAL_INTERVAL"
        printf '%s\n' \
            "tr3d_lightweight_fusion=$TR3D_LIGHTWEIGHT_FUSION" \
            "tr3d_lightweight_stage=$TR3D_LIGHTWEIGHT_STAGE" \
            "tr3d_lightweight_top_k=$TR3D_LIGHTWEIGHT_TOP_K" \
            "tr3d_lightweight_diversity=$TR3D_LIGHTWEIGHT_DIVERSITY" \
            "tr3d_lightweight_min_angle=$TR3D_LIGHTWEIGHT_MIN_ANGLE" \
            "tr3d_lightweight_depth_stride=$TR3D_LIGHTWEIGHT_DEPTH_STRIDE" \
            "tr3d_lightweight_drain=$TR3D_LIGHTWEIGHT_DRAIN"
        printf '%s\n' \
            "c3_online_enabled=$C3_ONLINE_ENABLED" \
            "c3_online_c2_cache_root=$(readlink -m "$C3_ONLINE_C2_CACHE_ROOT")" \
            "c3_online_parent_cache_root=$(readlink -m "$C3_ONLINE_PARENT_CACHE_ROOT")"
    } | sha256sum | awk '{print $1}'
)"

if [[ "$DISABLE_ONLINE" == "0" ]]; then
    PREFLIGHT_IMPORTS="import torch, torchvision, open_clip, ultralytics"
else
    PREFLIGHT_IMPORTS="import torch, torchvision, open_clip"
fi
CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
"$PYTHON" -c \
    "$PREFLIGHT_IMPORTS; assert torch.cuda.is_available(); print('Online environment OK:', torch.__version__, torch.version.cuda)"

total="$(awk 'END {print NR}' "$META")"
worker_count="${#GPUS[@]}"
echo "[$(date '+%F %T')] Starting isolated online-refinement inference"
echo "[$(date '+%F %T')] Code root: $ROOT"
echo "[$(date '+%F %T')] Live assets (read-only): $LIVE_ROOT"
echo "[$(date '+%F %T')] GPUs: $GPU_SPEC; workers: $worker_count; scenes: $total"
echo "[$(date '+%F %T')] Proposal interval: every $PROPOSAL_INTERVAL keyframes"
echo "[$(date '+%F %T')] Candidate TTL clock: ${CANDIDATE_TTL_CLOCK:-config-default}"
echo "[$(date '+%F %T')] Candidate track TTL: ${CANDIDATE_TRACK_TTL:-config-default}"
echo "[$(date '+%F %T')] Archive confirmed tracks: ${ARCHIVE_CONFIRMED:-config-default}"
echo "[$(date '+%F %T')] Inference/evaluation seeds: $INFERENCE_SEED/$EVAL_SEED"
echo "[$(date '+%F %T')] Online refinement disabled: $DISABLE_ONLINE"
echo "[$(date '+%F %T')] Online ablation profile: ${ABLATION_PROFILE:-config-default}"
echo "[$(date '+%F %T')] ScanNet minimum extent: ${SCANNET_MIN_EXTENT:-config-default}"
echo "[$(date '+%F %T')] Detector/quality blend: ${QUALITY_DETECTOR_BLEND:-config-default}"
if [[ "$boxer_gate_value_count" == "3" ]]; then
    echo "[$(date '+%F %T')] Selective Boxer gate: center<=$BOXER_GATE_CENTER m; volume=[$BOXER_GATE_MIN_VOLUME,$BOXER_GATE_MAX_VOLUME]"
else
    echo "[$(date '+%F %T')] Selective Boxer gate: config-default"
fi
echo "[$(date '+%F %T')] Prediction root: $PRED_ROOT"
echo "[$(date '+%F %T')] Same-run post-B6/pre-R3 baseline root: $R3_SAME_RUN_BASELINE_ROOT"
echo "[$(date '+%F %T')] Diagnostics root: $DIAGNOSTICS_ROOT"
echo "[$(date '+%F %T')] ScanNet frames root: $FRAMES_ROOT"
echo "[$(date '+%F %T')] Proposal cache: $CACHE_MODE:${CACHE_NAMESPACE:-disabled}"
echo "[$(date '+%F %T')] TR3D terminal mode: immutable p100 parent-cache replay"
echo "[$(date '+%F %T')] TR3D prefix manifest: $R3_PREFIX_MANIFEST"
echo "[$(date '+%F %T')] TR3D parent cache: $R3_PARENT_CACHE_ROOT"
echo "[$(date '+%F %T')] TR3D diagnostics: $R3_DIAGNOSTICS_ROOT"
echo "[$(date '+%F %T')] C3 online identity observer: $C3_ONLINE_ENABLED"
echo "[$(date '+%F %T')] Lightweight fusion observer: $TR3D_LIGHTWEIGHT_FUSION (stage=L$TR3D_LIGHTWEIGHT_STAGE, top-k=$TR3D_LIGHTWEIGHT_TOP_K, diversity=$TR3D_LIGHTWEIGHT_DIVERSITY, min-angle=$TR3D_LIGHTWEIGHT_MIN_ANGLE, depth-stride=$TR3D_LIGHTWEIGHT_DEPTH_STRIDE, drain=$TR3D_LIGHTWEIGHT_DRAIN)"
if [[ "$C3_ONLINE_ENABLED" == "1" ]]; then
    echo "[$(date '+%F %T')] C3 online C2 cache: $C3_ONLINE_C2_CACHE_ROOT"
    echo "[$(date '+%F %T')] C3 online diagnostics: $C3_ONLINE_DIAGNOSTICS_ROOT"
fi
if [[ -n "$C3_ACTIVE_POLICY" ]]; then
    echo "[$(date '+%F %T')] C3 online active policy: $C3_ACTIVE_POLICY"
    echo "[$(date '+%F %T')] C3 online active predictions: $C3_ACTIVE_OUTPUT_ROOT"
    echo "[$(date '+%F %T')] C3 online active diagnostics: $C3_ACTIVE_DIAGNOSTICS_ROOT"
fi
echo "[$(date '+%F %T')] Frozen G0 audit root: $R3_FROZEN_G0_ROOT"
echo "[$(date '+%F %T')] Shadow-gold audit root: $R3_SHADOW_GOLD_ROOT"
echo "[$(date '+%F %T')] Same-run baseline evaluation root: $R3_SAME_RUN_EVAL_ROOT"
echo "[$(date '+%F %T')] Active evaluation root: $EVAL_ROOT"
echo "[$(date '+%F %T')] Code fingerprint: $CODE_FINGERPRINT"
echo "[$(date '+%F %T')] Scene-list SHA256: $LIST_SHA256"

run_worker() {
    local gpu="$1"
    local shard="$2"
    local shards="$3"
    local index=0
    local completed=0

    while IFS= read -r scene || [[ -n "$scene" ]]; do
        if (( index % shards != shard )); then
            index=$((index + 1))
            continue
        fi
        local prediction="$PRED_ROOT/${scene}_boxes.pkl"
        local marker="$PRED_ROOT/${scene}.run_fingerprint"
        local same_run_baseline_prediction="$R3_SAME_RUN_BASELINE_ROOT/${scene}_boxes.pkl"
        local same_run_baseline_marker="$R3_SAME_RUN_BASELINE_ROOT/${scene}.run_fingerprint"
        local scene_log="$LOG_ROOT/scenes/${scene}.log"
        local boxer_diagnostic=""
        local r3_diagnostic="$R3_DIAGNOSTICS_ROOT/${scene}_tr3d_terminal.json"
        local r3_parent_cache="$R3_PARENT_CACHE_ROOT/$scene/p100.npz"
        local c3_online_diagnostic="$C3_ONLINE_DIAGNOSTICS_ROOT/${scene}_c3_online_identity.json"
        local c3_online_c2_cache="$C3_ONLINE_C2_CACHE_ROOT/$scene/p100.c2-maskrgbd.npz"
        local c3_online_parent_cache="$C3_ONLINE_PARENT_CACHE_ROOT/$scene/p100.npz"
        local c3_active_prediction="$C3_ACTIVE_OUTPUT_ROOT/${scene}_boxes.pkl"
        local c3_active_report="$C3_ACTIVE_DIAGNOSTICS_ROOT/${scene}_c3_online_active.json"
        local cache_manifest=""
        local cache_expected_fingerprint=""
        local scene_frames_root="$FRAMES_ROOT/$scene/frames"
        local scene_input_fingerprint
        local scene_fingerprint
        if [[ ! -d "$scene_frames_root" ]]; then
            echo "Missing ScanNet frame directory: $scene_frames_root" >&2
            return 1
        fi
        scene_input_fingerprint="$(
            find "$scene_frames_root" -type f \
                -printf '%P\t%s\t%T@\n' \
                | LC_ALL=C sort \
                | sha256sum \
                | awk '{print $1}'
        )"
        scene_fingerprint="$(
            printf '%s\n%s\n' "$CODE_FINGERPRINT" "$scene_input_fingerprint" \
                | sha256sum \
                | awk '{print $1}'
        )"
        local immutable_scene_artifacts=("$r3_parent_cache")
        if [[ "$SKIP_EVALUATION" == "0" ]]; then
            immutable_scene_artifacts+=(
                "$R3_FROZEN_G0_ROOT/${scene}_boxes.pkl"
                "$R3_SHADOW_GOLD_ROOT/${scene}_boxes.pkl"
            )
        fi
        for immutable_scene_artifact in "${immutable_scene_artifacts[@]}"; do
            if [[ ! -f "$immutable_scene_artifact" || -L "$immutable_scene_artifact" ]]; then
                echo "Missing/non-regular immutable scene artifact: $immutable_scene_artifact" >&2
                return 1
            fi
            scene_fingerprint="$(
                printf '%s\n%s\n' \
                    "$scene_fingerprint" \
                    "$(sha256sum "$immutable_scene_artifact" | awk '{print $1}')" \
                    | sha256sum \
                    | awk '{print $1}'
            )"
        done
        if [[ "$C3_ONLINE_ENABLED" == "1" ]]; then
            c3_scene_artifacts=("$c3_online_parent_cache")
            if [[ "$C3_ONLINE_CANDIDATE_SOURCE" == "c2" ]]; then
                c3_scene_artifacts+=("$c3_online_c2_cache")
            fi
            for immutable_scene_artifact in "${c3_scene_artifacts[@]}"; do
                if [[ ! -f "$immutable_scene_artifact" || -L "$immutable_scene_artifact" ]]; then
                    echo "Missing/non-regular C3 online scene artifact: $immutable_scene_artifact" >&2
                    return 1
                fi
                scene_fingerprint="$(
                    printf '%s\n%s\n' \
                        "$scene_fingerprint" \
                        "$(sha256sum "$immutable_scene_artifact" | awk '{print $1}')" \
                        | sha256sum \
                        | awk '{print $1}'
                )"
            done
        fi
        if [[ -n "$C3_ACTIVE_POLICY" ]]; then
            scene_fingerprint="$(
                printf '%s\n%s\n' \
                    "$scene_fingerprint" \
                    "$(sha256sum "$C3_ACTIVE_POLICY" | awk '{print $1}')" \
                    | sha256sum \
                    | awk '{print $1}'
            )"
        fi
        if [[ -n "$BOXER_DIAGNOSTICS_ROOT" ]]; then
            boxer_diagnostic="$BOXER_DIAGNOSTICS_ROOT/${scene}_boxer_lifting.jsonl"
        fi
        if [[ "$CACHE_MODE" == "replay" ]]; then
            cache_manifest="$CACHE_ROOT/$CACHE_NAMESPACE/$scene/manifest.json"
            local cache_baseline_marker="$CACHE_BASELINE_ROOT/${scene}.run_fingerprint"
            if [[ ! -s "$cache_baseline_marker" || ! -s "$cache_manifest" ]]; then
                echo "Missing frozen CuTR marker/cache for replay: $scene" >&2
                return 1
            fi
            cache_expected_fingerprint="$(tr -d '\n' < "$cache_baseline_marker")"
            local manifest_producer
            manifest_producer="$(
                "$PYTHON" - "$cache_manifest" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["producer_fingerprint"])
PY
            )"
            if [[ "$manifest_producer" != "$cache_expected_fingerprint" ]]; then
                echo "Frozen cache does not match CuTR baseline marker: $scene" >&2
                return 1
            fi
            scene_fingerprint="$(
                printf '%s\n%s\n' \
                    "$scene_fingerprint" \
                    "$(sha256sum "$cache_manifest" | awk '{print $1}')" \
                    | sha256sum \
                    | awk '{print $1}'
                )"
        fi
        local paired_artifact_exists=0
        local paired_artifact
        for paired_artifact in \
            "$prediction" \
            "$same_run_baseline_prediction" \
            "$marker" \
            "$same_run_baseline_marker" \
            "$r3_diagnostic"; do
            if [[ -e "$paired_artifact" || -L "$paired_artifact" ]]; then
                paired_artifact_exists=1
            fi
        done
        if [[ -n "$boxer_diagnostic" \
              && ( -e "$boxer_diagnostic" || -L "$boxer_diagnostic" ) ]]; then
            paired_artifact_exists=1
        fi
        if [[ "$C3_ONLINE_ENABLED" == "1" \
              && ( -e "$c3_online_diagnostic" || -L "$c3_online_diagnostic" ) ]]; then
            paired_artifact_exists=1
        fi
        if [[ -n "$C3_ACTIVE_POLICY" \
              && ( -e "$c3_active_prediction" || -L "$c3_active_prediction" \
                   || -e "$c3_active_report" || -L "$c3_active_report" ) ]]; then
            paired_artifact_exists=1
        fi
        if (( paired_artifact_exists )); then
            require_immutable_file "$prediction" "active prediction" || return 1
            require_immutable_file \
                "$same_run_baseline_prediction" \
                "same-run baseline prediction" || return 1
            require_immutable_file "$marker" "active completion marker" || return 1
            require_immutable_file \
                "$same_run_baseline_marker" \
                "same-run baseline completion marker" || return 1
            require_immutable_file "$r3_diagnostic" "terminal R3 diagnostic" || return 1
            if [[ -n "$boxer_diagnostic" ]]; then
                require_immutable_file "$boxer_diagnostic" "Boxer diagnostic" || return 1
            fi
            if [[ "$C3_ONLINE_ENABLED" == "1" ]]; then
                require_immutable_file "$c3_online_diagnostic" "C3 online identity diagnostic" || return 1
            fi
            if [[ -n "$C3_ACTIVE_POLICY" ]]; then
                require_immutable_file "$c3_active_prediction" "C3 online active prediction" || return 1
                require_immutable_file "$c3_active_report" "C3 online active report" || return 1
            fi
            if ! cmp -s "$marker" "$same_run_baseline_marker"; then
                echo "Paired active/baseline completion markers differ: $scene" >&2
                return 1
            fi
            local recorded_schema recorded_fingerprint
            local recorded_prediction_sha recorded_baseline_sha recorded_r3_sha
            local recorded_boxer_sha actual_boxer_sha
            local recorded_c3_online_sha actual_c3_online_sha
            recorded_schema="$(marker_value "$marker" schema)" || {
                echo "Malformed completion marker schema: $marker" >&2
                return 1
            }
            recorded_fingerprint="$(marker_value "$marker" scene_fingerprint)" || {
                echo "Malformed completion marker fingerprint: $marker" >&2
                return 1
            }
            recorded_prediction_sha="$(marker_value "$marker" active_prediction_sha256)" || {
                echo "Malformed active prediction hash: $marker" >&2
                return 1
            }
            recorded_baseline_sha="$(marker_value "$marker" same_run_baseline_sha256)" || {
                echo "Malformed same-run baseline hash: $marker" >&2
                return 1
            }
            recorded_r3_sha="$(marker_value "$marker" r3_diagnostic_sha256)" || {
                echo "Malformed R3 diagnostic hash: $marker" >&2
                return 1
            }
            recorded_boxer_sha="$(marker_value "$marker" boxer_diagnostic_sha256)" || {
                echo "Malformed Boxer diagnostic hash: $marker" >&2
                return 1
            }
            actual_boxer_sha="none"
            if [[ -n "$boxer_diagnostic" ]]; then
                actual_boxer_sha="$(sha256sum "$boxer_diagnostic" | awk '{print $1}')"
            fi
            recorded_c3_online_sha="none"
            actual_c3_online_sha="none"
            if [[ "$C3_ONLINE_ENABLED" == "1" ]]; then
                recorded_c3_online_sha="$(marker_value "$marker" c3_online_diagnostic_sha256)" || {
                    echo "Malformed C3 online diagnostic hash: $marker" >&2
                    return 1
                }
                actual_c3_online_sha="$(sha256sum "$c3_online_diagnostic" | awk '{print $1}')"
            fi
            if [[ "$recorded_schema" != "boxfusion.tr3d_terminal_completion.v2" \
                  || "$recorded_fingerprint" != "$scene_fingerprint" \
                  || "$recorded_prediction_sha" != "$(sha256sum "$prediction" | awk '{print $1}')" \
                  || "$recorded_baseline_sha" != "$(sha256sum "$same_run_baseline_prediction" | awk '{print $1}')" \
                  || "$recorded_r3_sha" != "$(sha256sum "$r3_diagnostic" | awk '{print $1}')" \
                  || "$recorded_boxer_sha" != "$actual_boxer_sha" \
                  || "$recorded_c3_online_sha" != "$actual_c3_online_sha" ]]; then
                echo "Paired completion manifest/hash mismatch: $scene" >&2
                return 1
            fi
            completed=$((completed + 1))
            echo "[$(date '+%F %T')] [GPU $gpu] $scene paired active/baseline artifacts already complete"
            index=$((index + 1))
            continue
        fi

        local optional_args=()
        if [[ -n "$PROPOSAL_CACHE_MODE_OVERRIDE" ]]; then
            optional_args+=(--proposal-cache-mode "$PROPOSAL_CACHE_MODE_OVERRIDE")
        fi
        optional_args+=(
            --tr3d-terminal-prefix-manifest "$R3_PREFIX_MANIFEST"
            --tr3d-terminal-parent-cache-root "$R3_PARENT_CACHE_ROOT"
            --tr3d-terminal-diagnostics-root "$R3_DIAGNOSTICS_ROOT"
            --tr3d-terminal-same-run-baseline-root "$R3_SAME_RUN_BASELINE_ROOT"
        )
        if [[ "$C3_ONLINE_ENABLED" == "1" ]]; then
            optional_args+=(
                --tr3d-c3-online-parent-cache-root "$C3_ONLINE_PARENT_CACHE_ROOT"
                --tr3d-c3-online-diagnostics-root "$C3_ONLINE_DIAGNOSTICS_ROOT"
                --tr3d-c3-online-prefix-id p100
                --tr3d-c3-online-candidate-source "$C3_ONLINE_CANDIDATE_SOURCE"
            )
            if [[ "$C3_ONLINE_CANDIDATE_SOURCE" == "c2" ]]; then
                optional_args+=(--tr3d-c3-online-c2-cache-root "$C3_ONLINE_C2_CACHE_ROOT")
            fi
        fi
        if [[ -n "$C3_ACTIVE_POLICY" ]]; then
            optional_args+=(
                --tr3d-c3-online-active-policy "$C3_ACTIVE_POLICY"
                --tr3d-c3-online-active-output-root "$C3_ACTIVE_OUTPUT_ROOT"
                --tr3d-c3-online-active-diagnostics-root "$C3_ACTIVE_DIAGNOSTICS_ROOT"
            )
        fi
        if [[ -n "$BOXER_DIAGNOSTICS_ROOT" ]]; then
            optional_args+=(
                --boxer-diagnostics-root "$BOXER_DIAGNOSTICS_ROOT"
            )
        fi
        if [[ "$boxer_gate_value_count" == "3" ]]; then
            optional_args+=(
                --boxer-selective-max-center-shift-m "$BOXER_GATE_CENTER"
                --boxer-selective-min-volume-ratio "$BOXER_GATE_MIN_VOLUME"
                --boxer-selective-max-volume-ratio "$BOXER_GATE_MAX_VOLUME"
            )
        fi
        if [[ -n "$REFINER_CHECKPOINT" ]]; then
            optional_args+=(--online-refiner-checkpoint "$REFINER_CHECKPOINT")
        fi
        if [[ -n "$QUALITY_CHECKPOINT" ]]; then
            optional_args+=(
                --online-quality-checkpoint "$QUALITY_CHECKPOINT"
                --online-quality-mode "$QUALITY_MODE"
            )
        fi
        if [[ -n "$QUALITY_DETECTOR_BLEND" ]]; then
            optional_args+=(
                --online-quality-detector-blend "$QUALITY_DETECTOR_BLEND"
            )
        fi
        if [[ "$DISABLE_ONLINE" == "1" ]]; then
            optional_args+=(--disable-online-refinement)
        fi
        if [[ -n "$ABLATION_PROFILE" ]]; then
            optional_args+=(--online-ablation-profile "$ABLATION_PROFILE")
        fi
        if [[ -n "$CANDIDATE_TTL_CLOCK" ]]; then
            optional_args+=(
                --online-candidate-ttl-clock "$CANDIDATE_TTL_CLOCK"
            )
        fi
        if [[ -n "$CANDIDATE_TRACK_TTL" ]]; then
            optional_args+=(
                --online-candidate-track-ttl "$CANDIDATE_TRACK_TTL"
            )
        fi
        if [[ -n "$SCANNET_MIN_EXTENT" ]]; then
            optional_args+=(
                --scannet-min-extent "$SCANNET_MIN_EXTENT"
            )
        fi
        if [[ "$ARCHIVE_CONFIRMED" == "1" ]]; then
            optional_args+=(--online-archive-confirmed-tracks)
        elif [[ "$ARCHIVE_CONFIRMED" == "0" ]]; then
            optional_args+=(--no-online-archive-confirmed-tracks)
        fi
        if [[ "$INCREMENTAL_TR3D_ENABLED" == "1" ]]; then
            optional_args+=(
                --tr3d-incremental-observer
                --tr3d-incremental-diagnostics-root "$INCREMENTAL_TR3D_DIAGNOSTICS_ROOT"
                --tr3d-incremental-prefix-manifest "$R3_PREFIX_MANIFEST"
                --tr3d-worker-python "$TR3D_WORKER_PYTHON"
                --tr3d-worker-runtime-root "$TR3D_WORKER_RUNTIME_ROOT"
                --tr3d-worker-config "$TR3D_WORKER_CONFIG"
                --tr3d-worker-checkpoint "$TR3D_WORKER_CHECKPOINT"
                --tr3d-worker-project-root "$TR3D_WORKER_PROJECT_ROOT"
                --tr3d-worker-vendor-root "$TR3D_WORKER_VENDOR_ROOT"
                --tr3d-worker-device cuda:0
                --tr3d-incremental-every-keyframes "$TR3D_INCREMENTAL_INTERVAL"
            )
            if [[ "$TR3D_LIGHTWEIGHT_FUSION" == "1" ]]; then
                optional_args+=(
                    --tr3d-lightweight-fusion
                    --tr3d-lightweight-stage "$TR3D_LIGHTWEIGHT_STAGE"
                    --tr3d-lightweight-top-k "$TR3D_LIGHTWEIGHT_TOP_K"
                    --tr3d-lightweight-diversity-weight "$TR3D_LIGHTWEIGHT_DIVERSITY"
                    --tr3d-lightweight-min-view-angle-deg "$TR3D_LIGHTWEIGHT_MIN_ANGLE"
                    --tr3d-lightweight-depth-stride "$TR3D_LIGHTWEIGHT_DEPTH_STRIDE"
                )
                if [[ "$TR3D_LIGHTWEIGHT_DRAIN" == "1" ]]; then
                    optional_args+=(--tr3d-lightweight-drain-finalize)
                fi
            fi
        fi

        echo "[$(date '+%F %T')] [GPU $gpu] Running $scene (list index $((index + 1))/$total)"
        if ! (
            cd "$ROOT"
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONHASHSEED="$INFERENCE_SEED" \
            CUBLAS_WORKSPACE_CONFIG=:4096:8 \
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONUNBUFFERED=1 \
            BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT="$cache_expected_fingerprint" \
            BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT="$cache_expected_fingerprint" \
            MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
            XDG_CACHE_HOME="$LOG_ROOT/model_cache_gpu${gpu}" \
            LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
            "$PYTHON" demo_tr3d_terminal_active.py scannet \
                --model-path "$LIVE_ROOT/models/cutr_rgbd.pth" \
                --clip_path "$LIVE_ROOT/models/open_clip_pytorch_model.bin" \
                --class_txt "$LIVE_ROOT/data/panoptic_categories_nomerge.txt" \
                --class-features "$LIVE_ROOT/data/class_features.pt" \
                --config "$CONFIG" \
                --output-dir "$PRED_ROOT" \
                --diagnostics-root "$DIAGNOSTICS_ROOT" \
                --scannet-frames-root "$FRAMES_ROOT" \
                --online-proposal-checkpoint "$YOLOE_CHECKPOINT" \
                --online-proposal-every-keyframes "$PROPOSAL_INTERVAL" \
                --device cuda \
                --seed "$INFERENCE_SEED" \
                --seq "$scene" \
                "${optional_args[@]}"
        ) >"$scene_log" 2>&1; then
            echo "[$(date '+%F %T')] ERROR: GPU $gpu failed on $scene" >&2
            tail -n 40 "$scene_log" >&2 || true
            return 1
        fi

        if [[ ! -s "$prediction" ]]; then
            echo "[$(date '+%F %T')] ERROR: GPU $gpu did not produce $prediction" >&2
            tail -n 40 "$scene_log" >&2 || true
            return 1
        fi
        if [[ ! -s "$same_run_baseline_prediction" ]]; then
            echo "[$(date '+%F %T')] ERROR: GPU $gpu did not produce paired same-run baseline $same_run_baseline_prediction" >&2
            tail -n 40 "$scene_log" >&2 || true
            return 1
        fi
        if [[ -n "$boxer_diagnostic" && ! -s "$boxer_diagnostic" ]]; then
            echo "[$(date '+%F %T')] ERROR: Boxer diagnostic missing: $boxer_diagnostic" >&2
            return 1
        fi
        if [[ ! -s "$r3_diagnostic" ]]; then
            echo "[$(date '+%F %T')] ERROR: terminal R3 diagnostic missing: $r3_diagnostic" >&2
            return 1
        fi
        if [[ "$C3_ONLINE_ENABLED" == "1" && ! -s "$c3_online_diagnostic" ]]; then
            echo "[$(date '+%F %T')] ERROR: C3 online identity diagnostic missing: $c3_online_diagnostic" >&2
            return 1
        fi
        local prediction_sha baseline_sha r3_sha boxer_sha c3_online_sha
        local marker_tmp baseline_marker_tmp
        prediction_sha="$(sha256sum "$prediction" | awk '{print $1}')"
        baseline_sha="$(sha256sum "$same_run_baseline_prediction" | awk '{print $1}')"
        r3_sha="$(sha256sum "$r3_diagnostic" | awk '{print $1}')"
        boxer_sha="none"
        if [[ -n "$boxer_diagnostic" ]]; then
            boxer_sha="$(sha256sum "$boxer_diagnostic" | awk '{print $1}')"
        fi
        c3_online_sha="none"
        if [[ "$C3_ONLINE_ENABLED" == "1" ]]; then
            c3_online_sha="$(sha256sum "$c3_online_diagnostic" | awk '{print $1}')"
            chmod 0444 "$c3_online_diagnostic"
        fi
        chmod 0444 "$prediction" "$same_run_baseline_prediction" "$r3_diagnostic"
        if [[ -n "$boxer_diagnostic" ]]; then
            chmod 0444 "$boxer_diagnostic"
        fi
        marker_tmp="$(mktemp "$PRED_ROOT/.${scene}.completion.XXXXXX")"
        {
            printf 'schema=boxfusion.tr3d_terminal_completion.v2\n'
            printf 'scene_fingerprint=%s\n' "$scene_fingerprint"
            printf 'active_prediction_sha256=%s\n' "$prediction_sha"
            printf 'same_run_baseline_sha256=%s\n' "$baseline_sha"
            printf 'r3_diagnostic_sha256=%s\n' "$r3_sha"
            printf 'boxer_diagnostic_sha256=%s\n' "$boxer_sha"
            if [[ "$C3_ONLINE_ENABLED" == "1" ]]; then
                printf 'c3_online_diagnostic_sha256=%s\n' "$c3_online_sha"
            fi
        } > "$marker_tmp"
        chmod 0444 "$marker_tmp"
        baseline_marker_tmp="$(mktemp "$R3_SAME_RUN_BASELINE_ROOT/.${scene}.completion.XXXXXX")"
        if ! cp --preserve=mode "$marker_tmp" "$baseline_marker_tmp"; then
            unlink "$marker_tmp" 2>/dev/null || true
            unlink "$baseline_marker_tmp" 2>/dev/null || true
            echo "Could not stage paired completion markers: $scene" >&2
            return 1
        fi
        chmod 0444 "$baseline_marker_tmp"
        if ! ln "$marker_tmp" "$marker"; then
            unlink "$marker_tmp" 2>/dev/null || true
            unlink "$baseline_marker_tmp" 2>/dev/null || true
            echo "Refusing to overwrite completion marker: $marker" >&2
            return 1
        fi
        if ! ln "$baseline_marker_tmp" "$same_run_baseline_marker"; then
            unlink "$marker_tmp" 2>/dev/null || true
            unlink "$baseline_marker_tmp" 2>/dev/null || true
            echo "Refusing to overwrite paired baseline completion marker: $same_run_baseline_marker" >&2
            return 1
        fi
        unlink "$marker_tmp"
        unlink "$baseline_marker_tmp"
        chmod 0444 "$marker" "$same_run_baseline_marker"
        completed=$((completed + 1))
        local summary
        summary="$(
            grep -E 'Online refinement summary|Boxer lifting summary|TR3D terminal active summary|C3 online identity summary|C3 online active summary' "$scene_log" \
                | tail -n 5 \
                | tr '\n' ' ' \
                || true
        )"
        echo "[$(date '+%F %T')] [GPU $gpu] Completed $scene $summary"
        index=$((index + 1))
    done < "$META"
    echo "[$(date '+%F %T')] [GPU $gpu] Worker completed $completed scenes"
}

child_pids=()
cleanup() {
    local pid
    for pid in "${child_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM

for shard in "${!GPUS[@]}"; do
    run_worker "${GPUS[$shard]}" "$shard" "$worker_count" &
    child_pids+=("$!")
done

worker_status=0
for pid in "${child_pids[@]}"; do
    if ! wait "$pid"; then
        worker_status=1
    fi
done
trap - INT TERM
if [[ "$worker_status" -ne 0 ]]; then
    echo "At least one worker failed; evaluation was not started" >&2
    exit 1
fi

expected_scene_count=0
while IFS= read -r scene || [[ -n "$scene" ]]; do
    expected_scene_count=$((expected_scene_count + 1))
    require_immutable_file \
        "$PRED_ROOT/${scene}_boxes.pkl" \
        "active prediction" || exit 1
    require_immutable_file \
        "$R3_SAME_RUN_BASELINE_ROOT/${scene}_boxes.pkl" \
        "same-run baseline prediction" || exit 1
    require_immutable_file \
        "$PRED_ROOT/${scene}.run_fingerprint" \
        "active completion marker" || exit 1
    require_immutable_file \
        "$R3_SAME_RUN_BASELINE_ROOT/${scene}.run_fingerprint" \
        "same-run baseline completion marker" || exit 1
    require_immutable_file \
        "$R3_DIAGNOSTICS_ROOT/${scene}_tr3d_terminal.json" \
        "terminal R3 diagnostic" || exit 1
    if [[ -n "$BOXER_DIAGNOSTICS_ROOT" ]]; then
        require_immutable_file \
            "$BOXER_DIAGNOSTICS_ROOT/${scene}_boxer_lifting.jsonl" \
            "Boxer diagnostic" || exit 1
    fi
    if [[ "$C3_ONLINE_ENABLED" == "1" ]]; then
        require_immutable_file \
            "$C3_ONLINE_DIAGNOSTICS_ROOT/${scene}_c3_online_identity.json" \
            "C3 online identity diagnostic" || exit 1
    fi
    if [[ -n "$C3_ACTIVE_POLICY" ]]; then
        require_immutable_file \
            "$C3_ACTIVE_OUTPUT_ROOT/${scene}_boxes.pkl" \
            "C3 online active prediction" || exit 1
        require_immutable_file \
            "$C3_ACTIVE_DIAGNOSTICS_ROOT/${scene}_c3_online_active.json" \
            "C3 online active report" || exit 1
    fi
done < "$META"

count_exact_artifacts() {
    local root="$1"
    local pattern="$2"
    find "$root" -maxdepth 1 -type f -name "$pattern" | wc -l
}

active_prediction_count="$(count_exact_artifacts "$PRED_ROOT" 'scene*_boxes.pkl')"
baseline_prediction_count="$(count_exact_artifacts "$R3_SAME_RUN_BASELINE_ROOT" 'scene*_boxes.pkl')"
active_marker_count="$(count_exact_artifacts "$PRED_ROOT" 'scene*.run_fingerprint')"
baseline_marker_count="$(count_exact_artifacts "$R3_SAME_RUN_BASELINE_ROOT" 'scene*.run_fingerprint')"
r3_diagnostic_count="$(count_exact_artifacts "$R3_DIAGNOSTICS_ROOT" 'scene*_tr3d_terminal.json')"
c3_online_diagnostic_count="$total"
if [[ "$C3_ONLINE_ENABLED" == "1" ]]; then
    c3_online_diagnostic_count="$(count_exact_artifacts "$C3_ONLINE_DIAGNOSTICS_ROOT" 'scene*_c3_online_identity.json')"
fi
incremental_diagnostic_count="$total"
if [[ "$INCREMENTAL_TR3D_ENABLED" == "1" ]]; then
    incremental_diagnostic_count="$(count_exact_artifacts "$INCREMENTAL_TR3D_DIAGNOSTICS_ROOT" 'scene*_tr3d_incremental.json')"
fi
if [[ "$expected_scene_count" -ne "$total" \
      || "$active_prediction_count" -ne "$total" \
      || "$baseline_prediction_count" -ne "$total" \
      || "$active_marker_count" -ne "$total" \
      || "$baseline_marker_count" -ne "$total" \
      || "$r3_diagnostic_count" -ne "$total" \
      || "$c3_online_diagnostic_count" -ne "$total" \
      || "$incremental_diagnostic_count" -ne "$total" ]]; then
    echo "Paired artifact count mismatch: expected=$total active=$active_prediction_count baseline=$baseline_prediction_count active_markers=$active_marker_count baseline_markers=$baseline_marker_count r3_diagnostics=$r3_diagnostic_count c3_online_diagnostics=$c3_online_diagnostic_count incremental_diagnostics=$incremental_diagnostic_count" >&2
    exit 1
fi
if [[ -n "$BOXER_DIAGNOSTICS_ROOT" ]]; then
    boxer_diagnostic_count="$(count_exact_artifacts "$BOXER_DIAGNOSTICS_ROOT" 'scene*_boxer_lifting.jsonl')"
    if [[ "$boxer_diagnostic_count" -ne "$total" ]]; then
        echo "Expected $total Boxer diagnostics, found $boxer_diagnostic_count" >&2
        exit 1
    fi
fi

if [[ "$SKIP_EVALUATION" == "1" ]]; then
    echo "[$(date '+%F %T')] Completed train-only paired diagnostic collection; terminal audit and evaluation skipped"
    exit 0
fi

AUDIT_REPORT="$LOG_ROOT/terminal_active_audit.json"
AUDIT_STDOUT="$LOG_ROOT/terminal_active_audit_stdout.json"
if [[ -e "$AUDIT_REPORT" || -e "$AUDIT_STDOUT" ]]; then
    require_immutable_file "$AUDIT_REPORT" "terminal R3 audit report" || exit 1
    require_immutable_file "$AUDIT_STDOUT" "terminal R3 audit stdout" || exit 1
    if ! "$PYTHON" - "$AUDIT_REPORT" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("ok") is not True:
    raise SystemExit("stored terminal R3 audit did not pass")
PY
    then
        exit 1
    fi
    echo "[$(date '+%F %T')] Reusing immutable successful no-GT terminal R3 audit"
else
    echo "[$(date '+%F %T')] Running no-GT terminal R3 identity/gold/same-run audit"
    audit_stdout_tmp="$(mktemp "$LOG_ROOT/.terminal_active_audit_stdout.XXXXXX")"
    if ! PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
        "$ROOT/tools/audit_tr3d_terminal_active.py" \
        --scene-list "$META" \
        --frozen-root "$R3_FROZEN_G0_ROOT" \
        --same-run-baseline-root "$R3_SAME_RUN_BASELINE_ROOT" \
        --active-root "$PRED_ROOT" \
        --shadow-root "$R3_SHADOW_GOLD_ROOT" \
        --frozen-manifest "$R3_FROZEN_MANIFEST" \
        --shadow-manifest "$R3_SHADOW_MANIFEST" \
        --prefix-manifest "$R3_PREFIX_MANIFEST" \
        --parent-cache-root "$R3_PARENT_CACHE_ROOT" \
        --diagnostics-root "$R3_DIAGNOSTICS_ROOT" \
        --report "$AUDIT_REPORT" \
        > "$audit_stdout_tmp"; then
        unlink "$audit_stdout_tmp" 2>/dev/null || true
        exit 1
    fi
    chmod 0444 "$AUDIT_REPORT" "$audit_stdout_tmp"
    if ! ln "$audit_stdout_tmp" "$AUDIT_STDOUT"; then
        unlink "$audit_stdout_tmp" 2>/dev/null || true
        echo "Refusing to overwrite terminal R3 audit stdout: $AUDIT_STDOUT" >&2
        exit 1
    fi
    unlink "$audit_stdout_tmp"
    chmod 0444 "$AUDIT_STDOUT"
fi

if [[ "$C3_ONLINE_ENABLED" == "1" ]]; then
    echo "[$(date '+%F %T')] Running GT-free C3 online identity audit"
    c3_audit_stdout_tmp="$(mktemp "$LOG_ROOT/.c3_online_identity_audit.XXXXXX")"
    if ! TMPDIR="${BOXFUSION_C3_ONLINE_TMPDIR:-/dev/shm}" \
        PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
        "$ROOT/tools/audit_tr3d_c3_online_identity.py" \
        --scene-list "$META" \
        --diagnostics-root "$C3_ONLINE_DIAGNOSTICS_ROOT" \
        --output "$C3_ONLINE_AUDIT_REPORT" \
        > "$c3_audit_stdout_tmp"; then
        unlink "$c3_audit_stdout_tmp" 2>/dev/null || true
        exit 1
    fi
    chmod 0444 "$C3_ONLINE_AUDIT_REPORT" "$c3_audit_stdout_tmp"
    c3_audit_stdout="$LOG_ROOT/c3_online_identity_audit_stdout.json"
    if ! ln "$c3_audit_stdout_tmp" "$c3_audit_stdout"; then
        unlink "$c3_audit_stdout_tmp" 2>/dev/null || true
        echo "Refusing to overwrite C3 online audit stdout: $c3_audit_stdout" >&2
        exit 1
    fi
    unlink "$c3_audit_stdout_tmp"
    chmod 0444 "$c3_audit_stdout"
fi

run_standard_evaluation() {
    local label="$1"
    local pred_root="$2"
    local dump_root="$3"
    local stdout_log="$4"
    local eval_tmp map_count
    if [[ -e "$stdout_log" ]]; then
        require_immutable_file "$stdout_log" "$label evaluation log" || return 1
        if [[ ! -d "$dump_root" || -L "$dump_root" ]]; then
            echo "$label evaluation log exists without a regular dump directory: $dump_root" >&2
            return 1
        fi
        map_count="$(grep -c '^eval mAP:' "$stdout_log" || true)"
        if [[ "$map_count" -ne 3 ]]; then
            echo "$label evaluation log is incomplete: $stdout_log" >&2
            return 1
        fi
        echo "[$(date '+%F %T')] Reusing immutable $label standard evaluation"
    else
        if [[ -e "$dump_root" || -L "$dump_root" ]]; then
            echo "Refusing orphan $label evaluation directory: $dump_root" >&2
            return 1
        fi
        eval_tmp="$(mktemp "$LOG_ROOT/.${label//[^A-Za-z0-9_.-]/_}_eval.XXXXXX")"
        if ! (
            cd "$ROOT/evaluation"
            CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
            PYTHONDONTWRITEBYTECODE=1 \
            MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
            LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
            "$PYTHON" eval_scannet.py \
                --dataset scannet \
                --data_path /extra/ZhaoX/scannet_data/scans \
                --gt_root "$GT_ROOT" \
                --dump_dir "$dump_root" \
                --num_point 40000 \
                --cluster_sampling seed_fps \
                --use_3d_nms \
                --use_cls_nms \
                --per_class_proposal \
                --num_workers 0 \
                --gpu 0 \
                --seed "$EVAL_SEED" \
                --scene_list "$META" \
                --pred_root "$pred_root"
        ) >"$eval_tmp" 2>&1; then
            echo "$label standard evaluation failed; inspect $eval_tmp" >&2
            return 1
        fi
        map_count="$(grep -c '^eval mAP:' "$eval_tmp" || true)"
        if [[ "$map_count" -ne 3 || ! -d "$dump_root" ]]; then
            echo "$label standard evaluation produced incomplete artifacts: $eval_tmp" >&2
            return 1
        fi
        chmod 0444 "$eval_tmp"
        if ! ln "$eval_tmp" "$stdout_log"; then
            unlink "$eval_tmp" 2>/dev/null || true
            echo "Refusing to overwrite $label evaluation log: $stdout_log" >&2
            return 1
        fi
        unlink "$eval_tmp"
        chmod 0444 "$stdout_log"
        chmod -R a-w "$dump_root"
    fi
    echo "=== $label standard ScanNet metrics ==="
    grep -E '^eval mAP:|^eval APrec:|^eval ARecall:' "$stdout_log"
}

echo "[$(date '+%F %T')] Completed paired inference; evaluating same-run baseline and active outputs"
run_standard_evaluation \
    "same-run baseline (post-B6/pre-R3)" \
    "$R3_SAME_RUN_BASELINE_ROOT" \
    "$R3_SAME_RUN_EVAL_ROOT" \
    "$LOG_ROOT/eval_same_run_baseline_stdout.log"
run_standard_evaluation \
    "terminal R3 active" \
    "$PRED_ROOT" \
    "$EVAL_ROOT" \
    "$LOG_ROOT/eval_active_stdout.log"
echo "[$(date '+%F %T')] Paired same-run baseline/active inference, audit, and standard evaluation completed"

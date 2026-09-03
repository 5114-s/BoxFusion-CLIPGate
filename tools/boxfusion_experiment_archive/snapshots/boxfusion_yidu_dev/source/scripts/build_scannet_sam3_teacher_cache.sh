#!/usr/bin/env bash
set -euo pipefail

# Build an immutable SAM3 proposal cache for the exact ScanNet frames queried
# by the online Mask Graph provider.
#
# A provenance namespace is intentionally mandatory:
#   BOXFUSION_SAM3_TEACHER_NAMESPACE=sam3-scannet-val10-v1 \
#     bash scripts/build_scannet_sam3_teacher_cache.sh 0,1
#
# Resume only an unchanged interrupted build:
#   BOXFUSION_SAM3_TEACHER_ALLOW_RESUME=1 \
#   BOXFUSION_SAM3_TEACHER_NAMESPACE=sam3-scannet-val10-v1 \
#     bash scripts/build_scannet_sam3_teacher_cache.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FULL100="${BOXFUSION_SAM3_TEACHER_FULL100:-0}"
ALLOW_RESUME="${BOXFUSION_SAM3_TEACHER_ALLOW_RESUME:-0}"
NAMESPACE="${BOXFUSION_SAM3_TEACHER_NAMESPACE:-}"
RUN_TAG="${BOXFUSION_SAM3_TEACHER_RUN_TAG:-sam3_teacher_ablation10_v1}"

ENV_ROOT="${BOXFUSION_SAM3_ENV_ROOT:-/home/admin1/miniconda3/envs/sam3}"
PYTHON="$ENV_ROOT/bin/python"
RUNTIME_ENV_ROOT="${BOXFUSION_RUNTIME_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion2}"
RUNTIME_PYTHON="$RUNTIME_ENV_ROOT/bin/python"
SAM3_ROOT="${BOXFUSION_SAM3_SOURCE:-/data/ZhaoX/Group3D/third_party/sam3}"
CHECKPOINT="${BOXFUSION_SAM3_CHECKPOINT:-/data/ZhaoX/Group3D/checkpoints/sam3/sam3.pt}"
BPE_PATH="${BOXFUSION_SAM3_BPE_PATH:-$SAM3_ROOT/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
FRAMES_ROOT="${BOXFUSION_SAM3_FRAMES_ROOT:-/data/ZhaoX/BoxFusion/upstream_clean/scannet_readme_frames}"

# The caller is commonly inside boxfusion2. Never let its Python 3.10
# site-packages or torch shared libraries leak into the Python 3.12 SAM3
# interpreter. In particular, appending the ambient PYTHONPATH can make SAM3
# import boxfusion2/torch and fail before the cache builder starts.
unset PYTHONHOME
SAM3_PYTHONPATH="$ROOT:$SAM3_ROOT"
SAM3_LD_LIBRARY_PATH="$ENV_ROOT/lib"

CONFIDENCE_THRESHOLD="${BOXFUSION_SAM3_CONFIDENCE_THRESHOLD:-0.50}"
DUPLICATE_MASK_IOU="${BOXFUSION_SAM3_DUPLICATE_MASK_IOU:-0.90}"
MASK_THRESHOLD="${BOXFUSION_SAM3_MASK_THRESHOLD:-0.50}"
MIN_MASK_PIXELS="${BOXFUSION_SAM3_MIN_MASK_PIXELS:-100}"
MAX_PER_CLASS="${BOXFUSION_SAM3_MAX_PER_CLASS:-10}"
MAX_PROPOSALS="${BOXFUSION_SAM3_MAX_PROPOSALS:-64}"
RESOLUTION="${BOXFUSION_SAM3_RESOLUTION:-1008}"
PRECISION="${BOXFUSION_SAM3_PRECISION:-bf16}"
GAP="${BOXFUSION_SAM3_GAP:-25}"
PROPOSAL_INTERVAL="${BOXFUSION_SAM3_PROPOSAL_INTERVAL:-5}"
MAX_FRAMES="${BOXFUSION_SAM3_MAX_FRAMES:-}"

if [[ -n "${BOXFUSION_SAM3_SCENE_LIST:-}" ]]; then
    SCENE_LIST="$BOXFUSION_SAM3_SCENE_LIST"
    SCOPE_TAG="custom"
elif [[ "$FULL100" == "1" ]]; then
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
    SCOPE_TAG="full100"
else
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
    SCOPE_TAG="ablation10"
fi

if [[ -n "${BOXFUSION_SAM3_OBSERVER_LOG_ROOT:-}" ]]; then
    OBSERVER_LOG_ROOT="$BOXFUSION_SAM3_OBSERVER_LOG_ROOT"
elif [[ "$SCOPE_TAG" == "ablation10" \
        && -d "$ROOT/logs/maskgraph_observer_ablation10_v1/scenes" ]]; then
    # This makes the default ten-scene build fail closed if the simulated
    # provider schedule ever differs from the completed observer run.
    OBSERVER_LOG_ROOT="$ROOT/logs/maskgraph_observer_ablation10_v1"
else
    OBSERVER_LOG_ROOT=""
fi

if [[ -z "${BOXFUSION_SAM3_TEACHER_RUN_TAG:-}" ]]; then
    RUN_TAG="sam3_teacher_${SCOPE_TAG}_v1"
fi
CACHE_ROOT="${BOXFUSION_SAM3_TEACHER_CACHE_ROOT:-$ROOT/cache/sam3_teacher/$RUN_TAG}"
LOG_ROOT="${BOXFUSION_SAM3_TEACHER_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
METADATA_ROOT="${BOXFUSION_SAM3_TEACHER_METADATA_ROOT:-$LOG_ROOT/metadata}"
RUNTIME_RGB_ROOT="${BOXFUSION_SAM3_RUNTIME_RGB_ROOT:-$CACHE_ROOT/runtime_rgb}"

if [[ "$FULL100" != "0" && "$FULL100" != "1" ]]; then
    echo "BOXFUSION_SAM3_TEACHER_FULL100 must be 0 or 1" >&2
    exit 2
fi
if [[ "$ALLOW_RESUME" != "0" && "$ALLOW_RESUME" != "1" ]]; then
    echo "BOXFUSION_SAM3_TEACHER_ALLOW_RESUME must be 0 or 1" >&2
    exit 2
fi
if [[ -z "$NAMESPACE" || "$NAMESPACE" =~ ^[[:space:]]+$ ]]; then
    echo "BOXFUSION_SAM3_TEACHER_NAMESPACE is required." >&2
    echo "Use an immutable provenance name, for example:" >&2
    echo "  BOXFUSION_SAM3_TEACHER_NAMESPACE=sam3-scannet-val10-v1" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing SAM3 Python environment: $PYTHON" >&2
    echo "Set BOXFUSION_SAM3_ENV_ROOT to the SAM3 conda environment root." >&2
    exit 1
fi
if [[ ! -x "$RUNTIME_PYTHON" ]]; then
    echo "Missing BoxFusion runtime Python: $RUNTIME_PYTHON" >&2
    echo "Set BOXFUSION_RUNTIME_ENV_ROOT to the boxfusion2 environment root." >&2
    exit 1
fi
if [[ ! -f "$SAM3_ROOT/sam3/model_builder.py" ]]; then
    echo "Missing official SAM3 source tree: $SAM3_ROOT" >&2
    echo "Set BOXFUSION_SAM3_SOURCE to the directory containing sam3/." >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing SAM3 checkpoint: $CHECKPOINT" >&2
    echo "Set BOXFUSION_SAM3_CHECKPOINT to the official SAM3 checkpoint." >&2
    exit 1
fi
if [[ ! -f "$BPE_PATH" ]]; then
    echo "Missing SAM3 BPE vocabulary: $BPE_PATH" >&2
    echo "Set BOXFUSION_SAM3_BPE_PATH to bpe_simple_vocab_16e6.txt.gz." >&2
    exit 1
fi
if [[ ! -s "$SCENE_LIST" ]]; then
    echo "Missing or empty ScanNet scene list: $SCENE_LIST" >&2
    exit 1
fi
if [[ ! -d "$FRAMES_ROOT" ]]; then
    echo "Missing ScanNet frames root: $FRAMES_ROOT" >&2
    exit 1
fi
if [[ ! -f "$ROOT/tools/build_scannet_sam3_proposal_cache.py" ]]; then
    echo "Missing SAM3 ScanNet cache builder:" >&2
    echo "  $ROOT/tools/build_scannet_sam3_proposal_cache.py" >&2
    exit 1
fi

for name_value in \
    "BOXFUSION_SAM3_MIN_MASK_PIXELS=$MIN_MASK_PIXELS" \
    "BOXFUSION_SAM3_MAX_PER_CLASS=$MAX_PER_CLASS" \
    "BOXFUSION_SAM3_MAX_PROPOSALS=$MAX_PROPOSALS" \
    "BOXFUSION_SAM3_RESOLUTION=$RESOLUTION" \
    "BOXFUSION_SAM3_GAP=$GAP" \
    "BOXFUSION_SAM3_PROPOSAL_INTERVAL=$PROPOSAL_INTERVAL"; do
    name="${name_value%%=*}"
    value="${name_value#*=}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer" >&2
        exit 2
    fi
done
if [[ -n "$MAX_FRAMES" && ! "$MAX_FRAMES" =~ ^[1-9][0-9]*$ ]]; then
    echo "BOXFUSION_SAM3_MAX_FRAMES must be a positive integer" >&2
    exit 2
fi
if [[ "$PRECISION" != "bf16" && "$PRECISION" != "fp32" ]]; then
    echo "BOXFUSION_SAM3_PRECISION must be bf16 or fp32" >&2
    exit 2
fi
if [[ -n "$OBSERVER_LOG_ROOT" && ! -d "$OBSERVER_LOG_ROOT" ]]; then
    echo "Missing observer log root: $OBSERVER_LOG_ROOT" >&2
    exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
    echo "No GPU was specified" >&2
    exit 2
fi
for gpu in "${GPUS[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU index: $gpu" >&2
        exit 2
    fi
done

# Never silently mix cache entries or logs from different parameters. Builder
# resume additionally verifies its immutable per-shard metadata.
if [[ "$ALLOW_RESUME" != "1" ]]; then
    for directory in "$CACHE_ROOT" "$LOG_ROOT"; do
        if [[ -d "$directory" ]] \
            && [[ -n "$(find "$directory" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            echo "Refusing to reuse non-empty SAM3 teacher directory:" >&2
            echo "  $directory" >&2
            echo "Choose a fresh BOXFUSION_SAM3_TEACHER_RUN_TAG, or set" >&2
            echo "BOXFUSION_SAM3_TEACHER_ALLOW_RESUME=1 only for an unchanged interrupted build." >&2
            exit 1
        fi
    done
fi

mkdir -p "$CACHE_ROOT" "$RUNTIME_RGB_ROOT" "$LOG_ROOT/shards" "$METADATA_ROOT"
exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
    echo "Another SAM3 cache builder holds $LOG_ROOT/run.lock" >&2
    exit 1
fi
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

# Decode ScanNet JPEGs only in the exact BoxFusion runtime. OpenCV JPEG
# implementations are not byte-stable across the boxfusion2 and SAM3
# environments, while strict cache keys intentionally bind to every RGB byte.
CONDA_PREFIX="$RUNTIME_ENV_ROOT" \
CONDA_DEFAULT_ENV="boxfusion2" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONNOUSERSITE=1 \
PYTHONSAFEPATH=1 \
PYTHONPATH="$ROOT" \
LD_LIBRARY_PATH="$RUNTIME_ENV_ROOT/lib" \
BOXFUSION_EXPECTED_RUNTIME_ENV="$RUNTIME_ENV_ROOT" \
"$RUNTIME_PYTHON" -c \
    "import os, sys, cv2; expected=os.path.realpath(os.environ['BOXFUSION_EXPECTED_RUNTIME_ENV']); assert os.path.realpath(sys.prefix) == expected, f'wrong runtime Python prefix: {sys.prefix}'; assert cv2.__version__.startswith('4.6.'), f'expected OpenCV 4.6, loaded {cv2.__version__} from {cv2.__file__}'; print('BoxFusion runtime RGB exporter OK:', sys.version.split()[0], cv2.__version__, cv2.__file__)"

# Fail before launching workers if the selected environment cannot import the
# official source. CUDA is checked in the same visibility context as a worker.
CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
CONDA_PREFIX="$ENV_ROOT" \
CONDA_DEFAULT_ENV="sam3" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONNOUSERSITE=1 \
PYTHONSAFEPATH=1 \
PYTHONPATH="$SAM3_PYTHONPATH" \
LD_LIBRARY_PATH="$SAM3_LD_LIBRARY_PATH" \
BOXFUSION_EXPECTED_SAM3_ENV="$ENV_ROOT" \
"$PYTHON" -c \
    "import os, sys, torch; expected=os.path.realpath(os.environ['BOXFUSION_EXPECTED_SAM3_ENV']); torch_path=os.path.realpath(torch.__file__); assert os.path.realpath(sys.prefix) == expected, f'wrong Python prefix: {sys.prefix}'; assert os.path.commonpath((torch_path, expected)) == expected, f'wrong torch imported: {torch_path}'; from sam3.model.sam3_image_processor import Sam3Processor; from sam3.model_builder import build_sam3_image_model; assert torch.cuda.is_available(), 'SAM3 environment cannot see CUDA'; print('SAM3 environment OK:', torch.__version__, torch.version.cuda, torch_path)"

scene_count="$(awk 'NF { count += 1 } END { print count + 0 }' "$SCENE_LIST")"
worker_count="${#GPUS[@]}"
echo "[$(date '+%F %T')] Building immutable ScanNet SAM3 teacher cache"
echo "[$(date '+%F %T')] scenes: $scene_count from $SCENE_LIST"
echo "[$(date '+%F %T')] GPUs: $GPU_SPEC; workers: $worker_count"
echo "[$(date '+%F %T')] namespace: $NAMESPACE"
echo "[$(date '+%F %T')] output: $CACHE_ROOT"
echo "[$(date '+%F %T')] runtime RGB staging: $RUNTIME_RGB_ROOT"
echo "[$(date '+%F %T')] schedule: gap=$GAP, proposal_interval=$PROPOSAL_INTERVAL"
echo "[$(date '+%F %T')] schedule oracle: ${OBSERVER_LOG_ROOT:-simulated-only}"
echo "[$(date '+%F %T')] SAM3: resolution=$RESOLUTION, precision=$PRECISION"
echo "[$(date '+%F %T')] thresholds: confidence=$CONFIDENCE_THRESHOLD, mask=$MASK_THRESHOLD, duplicate_iou=$DUPLICATE_MASK_IOU"
echo "[$(date '+%F %T')] bounds: min_mask_pixels=$MIN_MASK_PIXELS, max_per_class=$MAX_PER_CLASS, max_proposals=$MAX_PROPOSALS"
echo "[$(date '+%F %T')] resume: $ALLOW_RESUME"

# Phase 1: export the exact online-runtime RGB arrays losslessly. SAM3 workers
# never decode the JPEGs and fail closed unless this manifest is present.
echo "[$(date '+%F %T')] Staging lossless runtime RGB with boxfusion2/OpenCV 4.6"
for shard in "${!GPUS[@]}"; do
    stage_optional_args=()
    if [[ "$ALLOW_RESUME" == "1" ]]; then
        stage_optional_args+=(--resume)
    fi
    if [[ -n "$MAX_FRAMES" ]]; then
        stage_optional_args+=(--max-frames "$MAX_FRAMES")
    fi
    if [[ -n "$OBSERVER_LOG_ROOT" ]]; then
        stage_optional_args+=(--observer-log-root "$OBSERVER_LOG_ROOT")
    fi
    (
        cd "$ROOT"
        CONDA_PREFIX="$RUNTIME_ENV_ROOT" \
        CONDA_DEFAULT_ENV="boxfusion2" \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONNOUSERSITE=1 \
        PYTHONSAFEPATH=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH="$ROOT" \
        LD_LIBRARY_PATH="$RUNTIME_ENV_ROOT/lib" \
        "$RUNTIME_PYTHON" tools/build_scannet_sam3_proposal_cache.py \
            --scene-list "$SCENE_LIST" \
            --frames-root "$FRAMES_ROOT" \
            --output-dir "$CACHE_ROOT" \
            --runtime-rgb-dir "$RUNTIME_RGB_ROOT" \
            --namespace "$NAMESPACE" \
            --gap "$GAP" \
            --proposal-interval "$PROPOSAL_INTERVAL" \
            --num-shards "$worker_count" \
            --shard-index "$shard" \
            "${stage_optional_args[@]}" \
            --stage-runtime-rgb
    ) | tee "$LOG_ROOT/runtime_rgb_shard${shard}.log"
done

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
    gpu="${GPUS[$shard]}"
    optional_args=()
    if [[ "$ALLOW_RESUME" == "1" ]]; then
        optional_args+=(--resume)
    fi
    if [[ -n "$MAX_FRAMES" ]]; then
        optional_args+=(--max-frames "$MAX_FRAMES")
    fi
    if [[ -n "$OBSERVER_LOG_ROOT" ]]; then
        optional_args+=(--observer-log-root "$OBSERVER_LOG_ROOT")
    fi
    echo "[$(date '+%F %T')] [GPU $gpu] launching shard $shard/$worker_count"
    (
        cd "$ROOT"
        CUDA_VISIBLE_DEVICES="$gpu" \
        CONDA_PREFIX="$ENV_ROOT" \
        CONDA_DEFAULT_ENV="sam3" \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONNOUSERSITE=1 \
        PYTHONSAFEPATH=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH="$SAM3_PYTHONPATH" \
        LD_LIBRARY_PATH="$SAM3_LD_LIBRARY_PATH" \
        "$PYTHON" tools/build_scannet_sam3_proposal_cache.py \
            --scene-list "$SCENE_LIST" \
            --frames-root "$FRAMES_ROOT" \
            --output-dir "$CACHE_ROOT" \
            --runtime-rgb-dir "$RUNTIME_RGB_ROOT" \
            --namespace "$NAMESPACE" \
            --checkpoint "$CHECKPOINT" \
            --sam3-root "$SAM3_ROOT" \
            --bpe-path "$BPE_PATH" \
            --device cuda:0 \
            --resolution "$RESOLUTION" \
            --precision "$PRECISION" \
            --confidence-threshold "$CONFIDENCE_THRESHOLD" \
            --mask-threshold "$MASK_THRESHOLD" \
            --duplicate-mask-iou "$DUPLICATE_MASK_IOU" \
            --min-mask-pixels "$MIN_MASK_PIXELS" \
            --max-per-class "$MAX_PER_CLASS" \
            --max-proposals "$MAX_PROPOSALS" \
            --gap "$GAP" \
            --proposal-interval "$PROPOSAL_INTERVAL" \
            --num-shards "$worker_count" \
            --shard-index "$shard" \
            --metadata-path "$METADATA_ROOT/shard${shard}.json" \
            "${optional_args[@]}"
    ) >"$LOG_ROOT/shards/shard${shard}.log" 2>&1 &
    child_pids+=("$!")
done

worker_status=0
for index in "${!child_pids[@]}"; do
    if ! wait "${child_pids[$index]}"; then
        worker_status=1
        echo "[$(date '+%F %T')] ERROR: shard $index failed" >&2
        tail -n 60 "$LOG_ROOT/shards/shard${index}.log" >&2 || true
    else
        echo "[$(date '+%F %T')] shard $index completed"
    fi
done
trap - INT TERM
if [[ "$worker_status" -ne 0 ]]; then
    echo "At least one SAM3 teacher shard failed" >&2
    exit 1
fi

cache_count="$(
    find "$CACHE_ROOT" -maxdepth 1 -type f -name '*.npz' | wc -l
)"
if [[ "$cache_count" -lt 1 ]]; then
    echo "SAM3 workers completed without producing any NPZ cache entries" >&2
    exit 1
fi

echo "[$(date '+%F %T')] Strictly verifying every scheduled cache key"
verify_optional_args=()
if [[ -n "$OBSERVER_LOG_ROOT" ]]; then
    verify_optional_args+=(--observer-log-root "$OBSERVER_LOG_ROOT")
fi
for shard in "${!GPUS[@]}"; do
    verify_shard_args=()
    if [[ -n "$MAX_FRAMES" ]]; then
        verify_shard_args+=(--max-frames "$MAX_FRAMES")
    fi
    (
        cd "$ROOT"
        CONDA_PREFIX="$ENV_ROOT" \
        CONDA_DEFAULT_ENV="sam3" \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONNOUSERSITE=1 \
        PYTHONSAFEPATH=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH="$SAM3_PYTHONPATH" \
        LD_LIBRARY_PATH="$SAM3_LD_LIBRARY_PATH" \
        "$PYTHON" tools/build_scannet_sam3_proposal_cache.py \
            --scene-list "$SCENE_LIST" \
            --frames-root "$FRAMES_ROOT" \
            --output-dir "$CACHE_ROOT" \
            --runtime-rgb-dir "$RUNTIME_RGB_ROOT" \
            --namespace "$NAMESPACE" \
            --checkpoint "$CHECKPOINT" \
            --sam3-root "$SAM3_ROOT" \
            --bpe-path "$BPE_PATH" \
            --gap "$GAP" \
            --proposal-interval "$PROPOSAL_INTERVAL" \
            --num-shards "$worker_count" \
            --shard-index "$shard" \
            --metadata-path "$METADATA_ROOT/verify_shard${shard}.json" \
            "${verify_optional_args[@]}" \
            "${verify_shard_args[@]}" \
            --verify-only
    ) | tee "$LOG_ROOT/verify_shard${shard}.log"
done

echo "[$(date '+%F %T')] SAM3 teacher cache completed: $cache_count entries"
echo "Replay with the exact immutable namespace:"
echo "  BOXFUSION_MASK_GRAPH_PROVIDER=cache_only \\"
echo "  BOXFUSION_MASK_GRAPH_TEACHER_CACHE_DIRECTORY=$CACHE_ROOT \\"
echo "  BOXFUSION_MASK_GRAPH_TEACHER_CACHE_NAMESPACE=$NAMESPACE \\"
echo "  BOXFUSION_MASK_GRAPH_TEACHER_CACHE_MISSING_POLICY=error \\"
echo "  BOXFUSION_MASK_GRAPH_RUN_TAG=maskgraph_sam3_observer_${SCOPE_TAG}_v1 \\"
echo "    bash scripts/run_scannet_missing_mask_graph.sh $GPU_SPEC observer"

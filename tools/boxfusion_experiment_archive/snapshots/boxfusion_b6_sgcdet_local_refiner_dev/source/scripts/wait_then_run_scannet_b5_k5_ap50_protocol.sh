#!/usr/bin/env bash
set -euo pipefail

# Lightweight hand-off watcher. It performs no CUDA work while the isolated
# B3-v2+B6 full100 run is active. After that driver reports successful
# evaluation, it copies the now-stable read-only YOLOE proposal cache and
# invokes the guarded K5 -> fixed10 -> conditional AP50 protocol.

GPU_SPEC="${1:-0,1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
UPSTREAM_ROOT="$WORKSPACE/boxfusion_b3_dev"
UPSTREAM_LOG="${BOXFUSION_B5_WAIT_FOR_LOG:-$UPSTREAM_ROOT/logs/b3v2_oriented_pair_b6_blend040_extent040_full100_clean/driver.log}"
SOURCE_CACHE="${BOXFUSION_B5_SOURCE_CACHE:-$UPSTREAM_ROOT/cache/yoloe_scannet}"
DESTINATION_CACHE="$ROOT/cache/yoloe_scannet"
WATCH_LOG="${BOXFUSION_B5_WATCH_LOG:-$ROOT/logs/b5_k5_ap50_protocol_watcher.log}"
POLL_SECONDS="${BOXFUSION_B5_WATCH_POLL_SECONDS:-30}"

mkdir -p "$(dirname "$WATCH_LOG")"
exec > >(tee -a "$WATCH_LOG") 2>&1

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "BOXFUSION_B5_WATCH_POLL_SECONDS must be a positive integer" >&2
    exit 1
fi
if [[ ! -f "$UPSTREAM_LOG" ]]; then
    echo "Missing upstream driver log: $UPSTREAM_LOG" >&2
    exit 1
fi

echo "Waiting without CUDA work for the current full100 driver:"
echo "  $UPSTREAM_LOG"
while ! grep -q \
    "Online-refinement inference and evaluation completed" \
    "$UPSTREAM_LOG"; do
    if grep -q \
        "At least one worker failed; evaluation was not started" \
        "$UPSTREAM_LOG"; then
        echo "Upstream run failed; refusing to start the B5 protocol." >&2
        exit 1
    fi
    sleep "$POLL_SECONDS"
done

echo "Upstream run completed successfully."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for the no-interference GPU guard" >&2
    exit 1
fi
while true; do
    if ! active_cuda_pids="$(
        nvidia-smi \
            --query-compute-apps=pid \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk 'NF { print $1 }'
    )"; then
        echo "Unable to inspect CUDA processes; refusing to continue." >&2
        exit 1
    fi
    if [[ -z "$active_cuda_pids" ]]; then
        break
    fi
    echo "CUDA is still in use; waiting without starting B5: $active_cuda_pids"
    sleep "$POLL_SECONDS"
done
echo "CUDA process list is empty; handing off to the B5 protocol."

if [[ -d "$SOURCE_CACHE" ]]; then
    mkdir -p "$DESTINATION_CACHE"
    # Never overwrite a destination entry. A fresh isolated destination is
    # expected; -n also makes an accidental watcher restart non-destructive.
    cp -a -n "$SOURCE_CACHE/." "$DESTINATION_CACHE/"
    echo "Copied stable YOLOE cache into the isolated experiment root."
else
    echo "No source YOLOE cache found; the provider will rebuild it."
fi

exec bash "$ROOT/scripts/run_scannet_b5_k5_then_ap50_protocol.sh" \
    "$GPU_SPEC"

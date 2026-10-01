#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

SOURCE_TAG="${BOXFUSION_C3_ONLINE_SOURCE_RUN_TAG:-c3_online_identity_full100_telemetry_v1}"
SHADOW_TAG="${BOXFUSION_C3_ONLINE_SHADOW_RUN_TAG:-c3_online_shadow_full100_telemetry_v1}"
SCENE_LIST="${BOXFUSION_C3_ONLINE_SHADOW_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
LIST_SHA="$(sha256sum "$SCENE_LIST" | awk '{print substr($1,1,12)}')"
SOURCE_NAMESPACE="$SOURCE_TAG/$(basename "$SCENE_LIST" .txt)-$LIST_SHA"
SOURCE_LOG_ROOT="$ROOT/logs/b6_g0_tr3d_terminal/$SOURCE_NAMESPACE"
AUDIT="$SOURCE_LOG_ROOT/c3_online_identity_audit.json"
DRIVER="$SOURCE_LOG_ROOT/driver.log"
POLL_SECONDS="${BOXFUSION_C3_ONLINE_FOLLOWUP_POLL_SECONDS:-60}"
TIMEOUT_SECONDS="${BOXFUSION_C3_ONLINE_FOLLOWUP_TIMEOUT_SECONDS:-28800}"

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid poll seconds" >&2; exit 2; }
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid timeout seconds" >&2; exit 2; }

echo "Waiting for full100 C3 observer audit"
echo "  source: $SOURCE_NAMESPACE"
echo "  audit: $AUDIT"
echo "  follow-up shadow tag: $SHADOW_TAG"

deadline=$((SECONDS + TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if [[ -s "$AUDIT" ]]; then
    /home/admin1/miniconda3/envs/boxfusion2/bin/python -c \
      'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("pass") and p.get("complete") and p.get("scene_count")==100' \
      "$AUDIT"
    echo "Full100 observer audit passed; starting append-only AP shadow"
    BOXFUSION_C3_ONLINE_SOURCE_RUN_TAG="$SOURCE_TAG" \
    BOXFUSION_C3_ONLINE_SHADOW_SCENE_LIST="$SCENE_LIST" \
    TMPDIR=/dev/shm \
      bash scripts/run_tr3d_c3_online_shadow.sh "$SHADOW_TAG"
    exit 0
  fi
  if [[ -s "$DRIVER" ]] && grep -qE \
    'At least one worker failed|evaluation was not started|ERROR:' "$DRIVER"; then
    echo "Full100 observer failed; refusing dependent AP shadow" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

echo "Timed out waiting for full100 observer audit after ${TIMEOUT_SECONDS}s" >&2
exit 1

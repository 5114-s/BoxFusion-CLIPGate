#!/usr/bin/env bash
set -u

REPO="${REPO:-Kevin1804/BoxFusion}"
LOCAL_DIR="${LOCAL_DIR:-/extra/ZhaoX/boxfusion_ca1m}"
HF_HOME="${HF_HOME:-/extra/ZhaoX/hf_cache}"
TMPDIR="${TMPDIR:-/extra/ZhaoX/hf_tmp}"
MAX_WORKERS="${MAX_WORKERS:-1}"
SLEEP_SECONDS="${SLEEP_SECONDS:-900}"

export HF_HOME TMPDIR

mkdir -p "${LOCAL_DIR}" "${HF_HOME}" "${TMPDIR}"

attempt=1
while true; do
  echo "[$(date '+%F %T')] attempt ${attempt}: hf download ${REPO} -> ${LOCAL_DIR} (max-workers=${MAX_WORKERS})"

  hf download "${REPO}" \
    --repo-type dataset \
    --local-dir "${LOCAL_DIR}" \
    --max-workers "${MAX_WORKERS}"

  rc=$?
  if [ "${rc}" -eq 0 ]; then
    echo "[$(date '+%F %T')] download finished successfully"
    exit 0
  fi

  echo "[$(date '+%F %T')] download failed with exit code ${rc}; retrying after ${SLEEP_SECONDS}s"
  sleep "${SLEEP_SECONDS}"
  attempt=$((attempt + 1))
done

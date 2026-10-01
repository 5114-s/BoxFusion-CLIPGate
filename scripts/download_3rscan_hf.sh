#!/usr/bin/env bash
set -uo pipefail

destination="${1:-/extra/ZhaoX/3RScan}"
cache_root="${2:-/extra/ZhaoX/.cache/huggingface}"
max_workers="${3:-2}"
hf_bin="/home/admin1/miniconda3/envs/temp/bin/hf"

mkdir -p "${destination}" "${cache_root}"
log_file="${destination}/download.log"
exit_file="${destination}/download.exit"
rm -f "${exit_file}"

{
  echo "[$(date '+%F %T')] Starting/resuming 3RScan download"
  echo "destination=${destination}"
  echo "cache_root=${cache_root}"
  echo "max_workers=${max_workers}"
} >> "${log_file}"

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date '+%F %T')] Download attempt=${attempt}" >> "${log_file}"
  HF_HOME="${cache_root}" \
  HF_HUB_DOWNLOAD_TIMEOUT=120 \
  HF_HUB_ETAG_TIMEOUT=60 \
    "${hf_bin}" download maartenImjalli143/3RScan \
      --repo-type dataset \
      --local-dir "${destination}" \
      --max-workers "${max_workers}" >> "${log_file}" 2>&1
  status=$?
  if [[ "${status}" -eq 0 ]]; then
    break
  fi
  echo "[$(date '+%F %T')] Attempt=${attempt} failed with status=${status}; retrying in 30 seconds" >> "${log_file}"
  sleep 30
done

echo "[$(date '+%F %T')] Download finished with status=${status}" >> "${log_file}"
printf '%s\n' "${status}" > "${exit_file}"
exit "${status}"

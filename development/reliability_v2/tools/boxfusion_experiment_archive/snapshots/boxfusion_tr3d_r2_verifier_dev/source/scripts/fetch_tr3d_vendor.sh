#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
VENDOR_DIR="${ROOT_DIR}/third_party/mmdetection3d"
REPOSITORY="https://github.com/open-mmlab/mmdetection3d.git"
PIN="fe25f7a51d36e3702f961e198894580d83c4387b"
RELEASE_TAG="v1.4.0"

if [[ -e "${VENDOR_DIR}" && ! -d "${VENDOR_DIR}/.git" ]]; then
  echo "Refusing non-git vendor path: ${VENDOR_DIR}" >&2
  exit 2
fi

if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  mkdir -p "$(dirname -- "${VENDOR_DIR}")"
  git clone --no-checkout --filter=blob:none "${REPOSITORY}" "${VENDOR_DIR}"
fi

remote_url="$(git -C "${VENDOR_DIR}" remote get-url origin)"
if [[ "${remote_url%.git}" != "${REPOSITORY%.git}" ]]; then
  echo "Refusing unrelated origin: ${remote_url}" >&2
  exit 2
fi

unexpected_status="$(
  git -C "${VENDOR_DIR}" status --porcelain --untracked-files=all |
    sed \
      -e '/^?? mmdet3d\/\.mim\//d' \
      -e '/^?? mmdet3d\.egg-info\//d'
)"
if [[ -n "${unexpected_status}" ]]; then
  echo "Refusing to change a dirty TR3D vendor checkout." >&2
  echo "${unexpected_status}" >&2
  exit 2
fi

if ! git -C "${VENDOR_DIR}" cat-file -e "${PIN}^{commit}" 2>/dev/null; then
  git -C "${VENDOR_DIR}" fetch --depth 1 origin "refs/tags/${RELEASE_TAG}"
fi

resolved="$(git -C "${VENDOR_DIR}" rev-parse "${PIN}^{commit}")"
if [[ "${resolved}" != "${PIN}" ]]; then
  echo "Fetched TR3D commit does not match the lock: ${resolved}" >&2
  exit 2
fi

git -C "${VENDOR_DIR}" checkout --detach "${PIN}"
exec "${SCRIPT_DIR}/check_tr3d_vendor.sh"

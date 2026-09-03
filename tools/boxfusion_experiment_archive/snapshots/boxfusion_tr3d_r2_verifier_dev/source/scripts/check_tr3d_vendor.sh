#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
VENDOR_DIR="${ROOT_DIR}/third_party/mmdetection3d"
PIN="fe25f7a51d36e3702f961e198894580d83c4387b"
TR3D_TREE="e7d4f3eaaeb39473babf52ef47af0d81fe72d6c8"

if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  echo "Missing vendor checkout: ${VENDOR_DIR}" >&2
  echo "Run: bash scripts/fetch_tr3d_vendor.sh" >&2
  exit 2
fi

head="$(git -C "${VENDOR_DIR}" rev-parse HEAD)"
tree="$(git -C "${VENDOR_DIR}" rev-parse HEAD:projects/TR3D 2>/dev/null || true)"
version="$(
  sed -n "s/^__version__ = ['\"]\\([^'\"]*\\)['\"]$/\\1/p" \
    "${VENDOR_DIR}/mmdet3d/version.py"
)"

status=0
if [[ "${head}" != "${PIN}" ]]; then
  echo "BAD commit: expected ${PIN}, found ${head}" >&2
  status=1
fi
if [[ "${tree}" != "${TR3D_TREE}" ]]; then
  echo "BAD TR3D tree: expected ${TR3D_TREE}, found ${tree:-missing}" >&2
  status=1
fi
if [[ "${version}" != "1.4.0" ]]; then
  echo "BAD mmdet3d version: expected 1.4.0, found ${version:-missing}" >&2
  status=1
fi
unexpected_status="$(
  git -C "${VENDOR_DIR}" status --porcelain --untracked-files=all |
    sed \
      -e '/^?? mmdet3d\/\.mim\//d' \
      -e '/^?? mmdet3d\.egg-info\//d'
)"
if [[ -n "${unexpected_status}" ]]; then
  echo "BAD vendor state: working tree is dirty" >&2
  echo "${unexpected_status}" >&2
  status=1
fi

required=(
  "projects/TR3D/README.md"
  "projects/TR3D/configs/tr3d.py"
  "projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py"
  "projects/TR3D/tr3d/mink_resnet.py"
  "projects/TR3D/tr3d/tr3d_head.py"
  "projects/TR3D/tr3d/tr3d_neck.py"
)
for relative in "${required[@]}"; do
  if [[ ! -f "${VENDOR_DIR}/${relative}" ]]; then
    echo "BAD vendor state: missing ${relative}" >&2
    status=1
  fi
done

if (( status != 0 )); then
  exit "${status}"
fi

echo "TR3D vendor OK"
echo "  commit: ${head}"
echo "  tree: ${tree}"
echo "  mmdet3d: ${version}"
echo "  config: projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py"

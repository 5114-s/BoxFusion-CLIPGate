#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ENV_REF="${1:-${ROOT_DIR}/.conda/boxfusion-tr3d}"
MODE="${2:-}"
MARKER=".boxfusion_tr3d_isolated_env_v1"
PIN="fe25f7a51d36e3702f961e198894580d83c4387b"
SOURCE_ENV="${BOXFUSION_TR3D_SOURCE_ENV:-openmmlab}"

case "${ENV_REF}" in
  base|temp|boxfusion|boxfusion2|boxfusion-online|openmmlab|openmmlab1)
    echo "Refusing to mutate reserved environment: ${ENV_REF}" >&2
    exit 2
    ;;
esac

if [[ "${ENV_REF}" == */* ]]; then
  if [[ "${ENV_REF}" != /* ]]; then
    ENV_REF="${ROOT_DIR}/${ENV_REF}"
  fi
  CONDA_SELECTOR=(-p "${ENV_REF}")
else
  CONDA_SELECTOR=(-n "${ENV_REF}")
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found." >&2
  exit 2
fi

"${SCRIPT_DIR}/fetch_tr3d_vendor.sh"

env_exists=0
if conda run "${CONDA_SELECTOR[@]}" python -c "import sys" >/dev/null 2>&1; then
  env_exists=1
fi

if (( env_exists == 1 )); then
  prefix="$(
    env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD \
      conda run "${CONDA_SELECTOR[@]}" python -c "import sys; print(sys.prefix)"
  )"
  if [[ "${MODE}" != "--resume" || ! -f "${prefix}/${MARKER}" ]]; then
    echo "Refusing existing environment ${ENV_REF}." >&2
    echo "Only a marked environment may be continued with --resume." >&2
    exit 2
  fi
else
  if conda run -n "${SOURCE_ENV}" python -c "import sys" >/dev/null 2>&1; then
    echo "Cloning proven environment ${SOURCE_ENV} -> ${ENV_REF}"
    conda create -y "${CONDA_SELECTOR[@]}" --clone "${SOURCE_ENV}"
  else
    conda create -y "${CONDA_SELECTOR[@]}" \
      -c conda-forge python=3.8 pip=23.3 ninja=1.11 cmake=3.27 \
      openblas-devel
  fi
  prefix="$(
    env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD \
      conda run "${CONDA_SELECTOR[@]}" python -c "import sys; print(sys.prefix)"
  )"
  {
    echo "schema=boxfusion.tr3d_env.v1"
    echo "vendor_commit=${PIN}"
  } >"${prefix}/${MARKER}"
fi

clean_conda_run=(
  env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD
  conda run --no-capture-output "${CONDA_SELECTOR[@]}"
  env PYTHONNOUSERSITE=1
)

if ! "${clean_conda_run[@]}" python -c \
  "import torch, mmcv, mmdet, mmengine, MinkowskiEngine"; then
  "${clean_conda_run[@]}" python -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.2 torchvision==0.16.2
  "${clean_conda_run[@]}" python -m pip install openmim==0.3.9
  "${clean_conda_run[@]}" mim install "mmengine==0.10.7" "mmcv==2.1.0"
  "${clean_conda_run[@]}" python -m pip install \
    "mmdet==3.2.0" "numpy==1.23.5" "numba==0.56.4" \
    "llvmlite==0.39.1" \
    "networkx>=2.5" plyfile "scikit-image<0.22" "trimesh>=3.9"

  # MinkowskiEngine must be compiled against the torch/CUDA ABI in this new
  # environment. This never touches BoxFusion or an existing OpenMMLab env.
  "${clean_conda_run[@]}" env MAX_JOBS="${MAX_JOBS:-8}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}" \
    python -m pip install --no-deps -v \
    "git+https://github.com/NVIDIA/MinkowskiEngine.git@v0.5.4"
fi

"${clean_conda_run[@]}" python -m pip install --no-deps -v -e \
  "${ROOT_DIR}/third_party/mmdetection3d"

# A conda clone can copy stale numpy-base 1.24 metadata/files over the pip
# NumPy 1.23.5 required by Numba 0.56.4. Remove that exact conda package from
# this marked, isolated environment, then reinstall all three ABI-coupled
# wheels after every other install step. This deliberately happens last.
if compgen -G "${prefix}/conda-meta/numpy-base-*.json" >/dev/null; then
  mkdir -p "${ROOT_DIR}/.conda/pkgs"
  CONDA_PKGS_DIRS="${ROOT_DIR}/.conda/pkgs" \
    conda remove -y "${CONDA_SELECTOR[@]}" numpy-base --force --offline
fi
"${clean_conda_run[@]}" python -m pip install \
  --only-binary=:all: --no-deps --force-reinstall \
  numpy==1.23.5 llvmlite==0.39.1 numba==0.56.4

"${clean_conda_run[@]}" python -c \
  "import numpy, numba, llvmlite, mmdet3d; \
from numba.np.ufunc import _internal; \
assert numpy.__version__ == '1.23.5'; \
assert numba.__version__ == '0.56.4'; \
assert llvmlite.__version__ == '0.39.1'; \
assert mmdet3d.__version__ == '1.4.0'"

exec "${SCRIPT_DIR}/check_tr3d_environment.sh" "${ENV_REF}" --build-model

# Memory-based Adapters local test setup (2026-11-08)

## Scope

This setup evaluates the released Online3D **Memory-based Adapters** detector.
It is an online, reconstruction-free, closed-vocabulary model (`OV = no`).  No
ScanNetV2-100 or CA-1M-107 evaluation scene is used for training.

The test runner collapses the released model's 18 ScanNet categories to one
foreground category, applies class-agnostic AABB NMS at IoU 0.5, and writes the
terminal scene map in the existing BoxFusion evaluator format.

## Environment and source

- Environment: `/data/ZhaoX/BoxFusion/.conda/online3d_py38`
- Official source snapshot:
  `/data/ZhaoX/BoxFusion/external/Online3D_official_20261108`
- Python 3.8.20
- PyTorch 1.11.0+cu113 / torchvision 0.12.0+cu113
- mmcv-full 1.6.2 / mmdet 2.28.2 / mmdet3d 1.0.0rc6
- MinkowskiEngine 0.5.4, locally compiled for RTX 3090 (`sm_86`)
- `sort_vertices`, locally compiled for the same PyTorch/CUDA ABI

The local CUDA build uses the CUDA 11.8 toolkit exposed through
`external/cuda_11_8_compat`.  Runtime library paths are set by the queue script
and should not be inherited from another conda environment.

## Weights

- Detector: `external/Online3D_official_20261108/checkpoints/FCAF3D-SV+Ours.pth`
  - bytes: 1,778,991,907
  - MD5: `beca65ecbe9dea20c7a930c7854c0c76`
  - SHA256: `0853ec5bd4d84eca3be56f9f25248b2945412db55e8f311248f0a4a72f4462a0`
- Image backbone: `external/Online3D_official_20261108/mmdet3d/models/backbones/img_backbone.pth`
  - bytes: 351,749,235
  - MD5: `c3000ac0325ed1666e6579629f01c982`
  - SHA256: `7366bbfe90ad5ea0fd6bfe8e1a7055cff0dbbdc83b21cfe84b99bd9020e52441`

The detector checkpoint was loaded on CPU into the complete 166,026,646
parameter model.  The checkpoint metadata contains all 18 ScanNet classes.

## Data protocol

- ScanNetV2: the frozen 100-scene list at
  `evaluation/data_util/meta_data/scannetv2_val.txt`; raw frames are read from
  `/extra/ZhaoX/scannet_data/scans`; every twentieth raw frame is used, matching
  Online3D's official ScanNet-MV preprocessing.
- CA-1M: all 107 processed sequences under `/extra/ZhaoX/boxfusion_ca1m`; every
  valid RGB-D frame is used.
- Each depth frame is back-projected and deterministically sampled to 5,000
  RGB points.  The scene memory persists across frames.  Only one frame is
  resident on the GPU at a time.

## Commands and outputs

The persistent queue is running in tmux session:

```bash
tmux attach-session -t online3d_memory_adapters_20261108
```

It waits for both GPUs to report at least 22,000 MiB free for three consecutive
polls, runs a two-frame smoke test, then uses one deterministic scene shard per
GPU.

Main entry points:

```bash
bash scripts/queue_online3d_memory_adapters_after_gpu_idle.sh
python tools/run_online3d_memory_adapters_benchmark.py --help
```

Outputs:

- `results/memory_adapters_scannet100`
- `results/memory_adapters_ca1m107`
- `logs/memory_adapters_benchmark/status.log`
- `logs/memory_adapters_benchmark/scannet/gpu{0,1}.log`
- `logs/memory_adapters_benchmark/ca1m/gpu{0,1}.log`
- `logs/memory_adapters_benchmark/eval_scannet100.log`
- `logs/memory_adapters_benchmark/eval_ca1m107.log`

Each completed scene has an atomic prediction pickle and a JSON receipt with
frame count, elapsed time, FPS, pre/post-NMS box counts, and protocol fields.
Interrupted runs reuse only readable, completed prediction files.

#!/usr/bin/env python3
"""No-GT integrated-runtime harness for native T05 plus S3R H10 provider.

This program is a timing harness, not an evaluator.  A coordinator launches
exactly two ``spawn`` workers on the same visible CUDA device: one persistent
native T05 worker and one persistent fresh OWLv2/Boxer/K8/S3R worker.  At most
one current-frame command can be outstanding on any queue.  For every native
frame the provider must acknowledge the current identity before the
coordinator is allowed to yield that identity to native T05; the next frame is
not requested until native has synchronized and acknowledged the current one.

Only timing, bounded-state, identity, resource, and failure counters may be
published.  No box, label, class, embedding, native prediction, annotation,
GT, oracle result, or AP value is serialized.  The real H10 manifest and model
bindings are intentionally fail-closed until their independent final seal.
Focused tests exercise the process/IPC protocol with injectable fake workers;
they do not start CUDA or formal H10 inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import importlib
import importlib.machinery
import io
import json
import math
import multiprocessing as mp
from multiprocessing import spawn as mp_spawn
import os
from pathlib import Path, PurePosixPath
import queue
import random
import resource
import stat
import subprocess
import sys
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))


def _import_shadow_candidate_paths(source_paths: Sequence[Path]) -> tuple[Path, ...]:
    """Enumerate adjacent legacy-bytecode/native-extension loader candidates."""

    candidates: set[Path] = set()
    for source_path in source_paths:
        source = Path(os.path.abspath(os.fspath(source_path)))
        if source.suffix != ".py":
            continue
        candidates.add(source.with_suffix(".pyc"))
        if source.name != "__init__.py":
            # A same-stem package directory is selected ahead of ``foo.py``.
            candidates.add(source.with_suffix(""))
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            candidates.add(source.with_name(f"{source.stem}{suffix}"))
        if source.name == "__init__.py":
            candidates.add(source.parent.parent / f"{source.parent.name}.pyc")
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                candidates.add(
                    source.parent.parent / f"{source.parent.name}{suffix}"
                )
    return tuple(sorted(candidates, key=os.fspath))


def _assert_import_shadow_candidates_absent(
    source_paths: Sequence[Path],
) -> tuple[Path, ...]:
    """Reject loader candidates that could execute ahead of pinned ``.py``."""

    candidates = _import_shadow_candidate_paths(source_paths)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for candidate in candidates:
        if not candidate.is_absolute() or len(candidate.parts) < 2:
            raise RuntimeError(f"import-shadow candidate is not absolute: {candidate}")
        current_descriptor = os.open("/", directory_flags)
        try:
            for component in candidate.parts[1:-1]:
                try:
                    named = os.stat(
                        component,
                        dir_fd=current_descriptor,
                        follow_symlinks=False,
                    )
                    next_descriptor = os.open(
                        component, directory_flags, dir_fd=current_descriptor
                    )
                except OSError as error:
                    raise RuntimeError(
                        f"import-shadow parent must be a non-symlink directory: {candidate}"
                    ) from error
                try:
                    opened = os.fstat(next_descriptor)
                    if (
                        not stat.S_ISDIR(named.st_mode)
                        or not stat.S_ISDIR(opened.st_mode)
                        or (named.st_dev, named.st_ino)
                        != (opened.st_dev, opened.st_ino)
                    ):
                        raise RuntimeError(
                            f"import-shadow parent identity changed: {candidate}"
                        )
                except BaseException:
                    os.close(next_descriptor)
                    raise
                os.close(current_descriptor)
                current_descriptor = next_descriptor
            try:
                os.stat(
                    candidate.parts[-1],
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            except OSError as error:
                raise RuntimeError(
                    f"cannot inspect import-shadow candidate: {candidate}"
                ) from error
            raise RuntimeError(
                f"import-shadow candidate must be absent: {candidate}"
            )
        finally:
            os.close(current_descriptor)
    return candidates


_EARLY_LOCAL_IMPORT_SOURCES = (
    REPOSITORY_ROOT / "boxfusion" / "__init__.py",
    REPOSITORY_ROOT / "boxfusion" / "s3r_h10_provider_core.py",
    REPOSITORY_ROOT / "boxfusion" / "s3r_receipt_tracker.py",
    REPOSITORY_ROOT / "tools" / "__init__.py",
    REPOSITORY_ROOT / "tools" / "build_scannet_s3r_h10_native_full_stream.py",
)
_assert_import_shadow_candidates_absent(_EARLY_LOCAL_IMPORT_SOURCES)

from boxfusion.s3r_h10_provider_core import parse_exact_schedule_bundle  # noqa: E402
from boxfusion.s3r_receipt_tracker import S3RObservation, S3RReceiptTracker  # noqa: E402
from tools import build_scannet_s3r_h10_native_full_stream as native_manifest_builder  # noqa: E402


SCHEMA = "boxfusion.s3r_h10_integrated_runtime.v2"
EXPECTED_NATIVE_MANIFEST_SCHEMA = "boxfusion.s3r_h10_native_full_stream.v1"
DEFAULT_NATIVE_MANIFEST = (
    REPOSITORY_ROOT / "docs" / "data" / "S3R_H10_NATIVE_FULL_STREAM_V1.json"
)
DEFAULT_PROVIDER_SCHEDULE = (
    REPOSITORY_ROOT / "docs" / "data" / "S3R_H10_EXACT_SCHEDULE_V2.json"
)
DEFAULT_SCENE_ROOT = REPOSITORY_ROOT / "upstream_clean" / "scannet_readme_frames"
FORMAL_T05_ROOT = REPOSITORY_ROOT / "results" / "scannet_topk_fusion_score05"
FORMAL_CONTROL_OUTPUT = (
    REPOSITORY_ROOT / "logs" / "scannet_s3r_h10_runtime_control_v2.json"
)
FORMAL_INTEGRATED_OUTPUT = (
    REPOSITORY_ROOT / "logs" / "scannet_s3r_h10_runtime_integrated_v2.json"
)
NATIVE_CONFIG_PATH = REPOSITORY_ROOT / "config" / "scannet_topk_fusion_score05.yaml"
NATIVE_CUTR_CHECKPOINT = REPOSITORY_ROOT / "models" / "cutr_rgbd.pth"
NATIVE_CLIP_CHECKPOINT = REPOSITORY_ROOT / "models" / "open_clip_pytorch_model.bin"
NATIVE_CLASS_FEATURES = REPOSITORY_ROOT / "data" / "class_features.pt"
NATIVE_CLASS_NAMES = REPOSITORY_ROOT / "data" / "panoptic_categories_nomerge.txt"
NATIVE_PST = REPOSITORY_ROOT / "data" / "pst_1024_0.tiff"
NATIVE_DEMO_SOURCE = REPOSITORY_ROOT / "demo.py"
NATIVE_CAPTURE_SOURCE = REPOSITORY_ROOT / "boxfusion" / "capture_stream.py"
NATIVE_MANIFEST_BUILDER_SOURCE = (
    REPOSITORY_ROOT / "tools" / "build_scannet_s3r_h10_native_full_stream.py"
)
NATIVE_MANIFEST_BUILDER_TEST_SOURCE = (
    REPOSITORY_ROOT / "tests" / "test_build_scannet_s3r_h10_native_full_stream.py"
)
INTEGRATED_RUNNER_TEST_SOURCE = (
    REPOSITORY_ROOT / "tests" / "test_benchmark_scannet_s3r_h10_integrated_runtime.py"
)
PROVIDER_RUNNER_SOURCE = (
    REPOSITORY_ROOT / "tools" / "run_scannet_s3r_h10_fresh_boxer_provider.py"
)
PROVIDER_CONTRACT_PATH = REPOSITORY_ROOT / "docs" / "S3R_H10_FRESH_BOXER_PROVIDER_CONTRACT.md"
FORMAL_RAW_SOURCE_JSON = (
    REPOSITORY_ROOT
    / "logs"
    / "scannet_s3r_h10_raw_boxer_source_score05_v1"
    / "S3R_H10_RAW_BOXER_SOURCE.json"
)
FORMAL_RAW_SOURCE_NPZ = FORMAL_RAW_SOURCE_JSON.with_suffix(".npz")
NATIVE_PYTHON_EXECUTABLE = Path(
    "/home/admin1/miniconda3/envs/boxfusion-online/bin/python"
)
PROVIDER_PYTHON_EXECUTABLE = Path("/home/admin1/miniconda3/envs/ovm3d-1/bin/python")
NATIVE_SETUPTOOLS_INIT = Path(
    "/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10/"
    "site-packages/setuptools/__init__.py"
)
NATIVE_SETUPTOOLS_VENDOR = NATIVE_SETUPTOOLS_INIT.parent / "_vendor"
NATIVE_TORCH_REMOTE_INSTANTIATOR = Path(
    "/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10/"
    "site-packages/torch/distributed/nn/jit/instantiator.py"
)
NATIVE_TORCH_REMOTE_TEMPLATE_PARENT = Path("/tmp")
NATIVE_TORCH_REMOTE_TEMPLATE_BASENAME = "_remote_module_non_scriptable.py"
NATIVE_TORCH_REMOTE_TEMPLATE_SIZE = 2_355
NATIVE_TORCH_REMOTE_TEMPLATE_SHA256 = (
    "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
)
PROVIDER_BOXER_ROOT = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer"
)
GIT_EXECUTABLE = Path("/usr/bin/git")
NVIDIA_SMI_EXECUTABLE = Path("/usr/bin/nvidia-smi")
TRUSTED_EXECUTABLE_PATH = "/usr/local/cuda-12.1/bin:/usr/bin:/bin"
EXPECTED_PROVIDER_BOXER_COMMIT = "1f86542dc342a4b1d474c87c97c5d1d6566d9148"
EXPECTED_PROVIDER_BOXER_TREE = "e81129bcd217280fedfde540bf2fe83f53a46476"
NATIVE_PYTHON_SHA256 = "0b713d4abbdf074ab38362c1542060b0e9841695d759df37d706baf1decf9a8b"
PROVIDER_PYTHON_SHA256 = "8d53381a3c7b869a331da9112ea494d0c1f90c17b69ccee9b1f6d4ef12273e5f"
NATIVE_FROZEN_SYS_PATH = (
    os.fspath(REPOSITORY_ROOT),
    "/home/admin1/miniconda3/envs/boxfusion-online/lib/python310.zip",
    "/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10",
    "/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10/lib-dynload",
    "/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10/site-packages",
    "/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10/site-packages/rerun_sdk",
)
PROVIDER_FROZEN_SYS_PATH = (
    os.fspath(REPOSITORY_ROOT),
    "/home/admin1/miniconda3/envs/ovm3d-1/lib/python310.zip",
    "/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10",
    "/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/lib-dynload",
    "/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages",
    "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src",
    "__editable__.detectron2-0.6.finder.__path_hook__",
    "__editable__.mast3r-1.0.0.finder.__path_hook__",
    "__editable__.unidepth-0.1.finder.__path_hook__",
    "/data/ZhaoX/OVM3D-Dett/Fast-SAM3D",
    "/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/black-26.3.1-py3.10.egg",
    "/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/omegaconf-2.3.0-py3.10.egg",
    "/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/iopath-0.1.9-py3.10.egg",
    "/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/tensorboard-2.20.0-py3.10.egg",
    "/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/cloudpickle-3.1.2-py3.10.egg",
    "/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/pytokens-0.4.1-py3.10.egg",
    "/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/pathspec-1.0.4-py3.10.egg",
    "/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/mypy_extensions-1.1.0-py3.10.egg",
)
NATIVE_POST_FACTORY_SYS_PATH = NATIVE_FROZEN_SYS_PATH
PROVIDER_POST_FACTORY_SYS_PATH = (
    os.fspath(PROVIDER_BOXER_ROOT.resolve(strict=True)),
    *PROVIDER_FROZEN_SYS_PATH,
)
FROZEN_PYCACHE_PREFIX = "/dev/null"
PROVIDER_IGNORED_CHECKPOINT_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt": (
            "d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f"
        ),
        "ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth": (
            "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea"
        ),
    }
)
PROVIDER_IGNORED_DINO_RELPATH = (
    "ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
)
PROVIDER_IGNORED_DINO_SYMLINK_TARGET = (
    "/data/ZhaoX/OVM3D-Dett/third_party/boxer/ckpts/"
    "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
)


def _sys_path_sha256(paths: Sequence[str]) -> str:
    payload = json.dumps(
        list(paths), ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


NATIVE_RUNTIME_IDENTITY: Mapping[str, str] = MappingProxyType(
    {
        "python_version": "3.10.13",
        "numpy_version": "1.26.4",
        "torch_version": "2.6.0+cu124",
        "cuda_version": "12.4",
        "numpy_origin": (
            "/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10/"
            "site-packages/numpy/__init__.py"
        ),
        "torch_origin": (
            "/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10/"
            "site-packages/torch/__init__.py"
        ),
        "python_pycache_prefix_environment": FROZEN_PYCACHE_PREFIX,
        "python_pycache_prefix": FROZEN_PYCACHE_PREFIX,
        "spawn_entry_python_sys_path_sha256": _sys_path_sha256(
            NATIVE_FROZEN_SYS_PATH
        ),
        "post_factory_python_sys_path_sha256": _sys_path_sha256(
            NATIVE_POST_FACTORY_SYS_PATH
        ),
    }
)
PROVIDER_RUNTIME_IDENTITY: Mapping[str, str] = MappingProxyType(
    {
        "python_version": "3.10.19",
        "numpy_version": "1.26.4",
        "torch_version": "2.2.0+cu121",
        "cuda_version": "12.1",
        "numpy_origin": (
            "/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/"
            "site-packages/numpy/__init__.py"
        ),
        "torch_origin": (
            "/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/"
            "site-packages/torch/__init__.py"
        ),
        "python_pycache_prefix_environment": FROZEN_PYCACHE_PREFIX,
        "python_pycache_prefix": FROZEN_PYCACHE_PREFIX,
        "spawn_entry_python_sys_path_sha256": _sys_path_sha256(
            PROVIDER_FROZEN_SYS_PATH
        ),
        "post_factory_python_sys_path_sha256": _sys_path_sha256(
            PROVIDER_POST_FACTORY_SYS_PATH
        ),
    }
)


def _validate_parent_runtime_identity() -> dict[str, Any]:
    """Fail before formal input reads/spawn if the coordinator runtime drifts."""

    executable = Path(sys.executable).resolve(strict=True)
    expected_executable = NATIVE_PYTHON_EXECUTABLE.resolve(strict=True)
    identity = {
        "python_executable": os.fspath(executable),
        "python_version": sys.version.split()[0],
        "numpy_version": str(np.__version__),
        "torch_imported": "torch" in sys.modules,
        "python_pycache_prefix_environment": os.environ.get(
            "PYTHONPYCACHEPREFIX"
        ),
        "python_pycache_prefix": sys.pycache_prefix,
    }
    if (
        executable != expected_executable
        or identity["python_version"] != NATIVE_RUNTIME_IDENTITY["python_version"]
        or identity["numpy_version"] != NATIVE_RUNTIME_IDENTITY["numpy_version"]
        or identity["torch_imported"]
        or identity["python_pycache_prefix_environment"]
        != FROZEN_PYCACHE_PREFIX
        or identity["python_pycache_prefix"] != FROZEN_PYCACHE_PREFIX
    ):
        raise IntegratedRuntimeError("formal parent runtime identity differs")
    return identity

EXPECTED_NATIVE_MANIFEST_SHA256 = (
    "449cf3d8e7e765fa53b46226b9c2a342a04dfd246d0baa6ee1ff1116cba0cf5d"
)
EXPECTED_NATIVE_MANIFEST_BUILDER_SHA256 = (
    "0d4a629a8c59c12a0bf150fcbf624e25c6c075ef31def783462988745eff8f9f"
)
EXPECTED_NATIVE_MANIFEST_BUILDER_TEST_SHA256 = (
    "6318024644d393cfdbc1747c5ad5aa75052e409015f9b2347ae9c02b0562c116"
)

EXPECTED_PROVIDER_SCHEDULE_SHA256 = (
    "1ce565a65510b80d69a0402fe7a40ea89920625f6a81147d42f9232f7a7761e9"
)
EXPECTED_PROVIDER_VALID_CALLS = 769
EXPECTED_PROVIDER_ABSTENTIONS = 1
EXPECTED_SCENE_COUNT = 10
EXPECTED_NATIVE_FRAME_COUNT = 19_370
EXPECTED_NATIVE_NONFINITE_POSE_COUNT = 37
EXPECTED_NATIVE_SCHEDULED_KEYFRAME_SLOTS = 780
EXPECTED_PROVIDER_OUTSIDE_COUNT = 18_600
EXPECTED_PROVIDER_RAW_ROWS = 6_338
EXPECTED_PROVIDER_K8_ROWS = 4_557
EXPECTED_PROVIDER_TRACKER_COMMITS = 769
EXPECTED_PROVIDER_RAW_ROWS_PER_SCENE: Mapping[str, int] = MappingProxyType(
    {
        "scene0304_00": 291,
        "scene0412_00": 857,
        "scene0019_00": 134,
        "scene0575_00": 891,
        "scene0426_00": 726,
        "scene0426_03": 383,
        "scene0578_00": 386,
        "scene0665_00": 390,
        "scene0050_01": 1_270,
        "scene0025_00": 1_010,
    }
)
EXPECTED_PROVIDER_K8_ROWS_PER_SCENE: Mapping[str, int] = MappingProxyType(
    {
        "scene0304_00": 230,
        "scene0412_00": 577,
        "scene0019_00": 112,
        "scene0575_00": 757,
        "scene0426_00": 520,
        "scene0426_03": 289,
        "scene0578_00": 361,
        "scene0665_00": 303,
        "scene0050_01": 860,
        "scene0025_00": 548,
    }
)
EXPECTED_PROVIDER_COMMITS_PER_SCENE: Mapping[str, int] = MappingProxyType(
    {
        "scene0304_00": 70,
        "scene0412_00": 93,
        "scene0019_00": 23,
        "scene0575_00": 117,
        "scene0426_00": 92,
        "scene0426_03": 49,
        "scene0578_00": 59,
        "scene0665_00": 41,
        "scene0050_01": 148,
        "scene0025_00": 77,
    }
)

EXPECTED_PROVIDER_CONTRACT_SHA256 = (
    "11cc5ab398809ccfab9fafdcc9645e796321eb2db527e78ef2515e99946883d0"
)
EXPECTED_TRACKER_SHA256 = (
    "277316c36b7a7fcb8005a24e907e0f232e41f6b5874411293eb26b0744df9628"
)
EXPECTED_TRACKER_TEST_SHA256 = (
    "f08fd59ee2888c936e5b783de668fd789ba6b676bc4864e001b000ea287b1e3c"
)
EXPECTED_PROVIDER_RUNNER_SHA256 = (
    "72e42f3a3865ee9f52687d2a5a5a40ecabe189864c4d7d2cce18daf6be056403"
)
EXPECTED_PROVIDER_RUNNER_TEST_SHA256 = (
    "89595cf544e60efdde5637f7315f42ce8d59b3a0088d50d7913c3a442d000a6e"
)
EXPECTED_PROVIDER_CORE_SHA256 = (
    "c70e114dabe1ef1081967027e4b5a15955ac16bab745652984dfe981100f21dd"
)
EXPECTED_PROVIDER_CORE_TEST_SHA256 = (
    "40f75dac98e5774e9b1637a7c51c4ab5676df38a074e3c3b97a0d3a40a305ce2"
)
EXPECTED_RAW_SOURCE_JSON_SHA256 = (
    "ca65214f3e6327cea66ec8cb700ab3501572be9325af4366beaffa2b7cc2859e"
)
EXPECTED_RAW_SOURCE_NPZ_SHA256 = (
    "fdb688cc1372985f2ffaf3d257ed470cd4de28ff42f7a2d04a5f72311a1225f2"
)
EXPECTED_RAW_SOURCE_ARRAY_CONTENT_SHA256 = (
    "a5efdb8d0d2c7b95f63368a3249229659a1052c400539321ce461da32732b862"
)
EXPECTED_RAW_SOURCE_K8_MEMBERSHIP_SHA256 = (
    "a2a94b11461e8c1bdd15d6a4ad99d058f42db6fd73690c69269ff1b89deb6391"
)

NATIVE_ASSET_EXPECTED_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "native_config": "596b42b22828360aa780a95f188244fcef4ef69d4ee0096a37c7b8094daebe4c",
        "native_demo": "b691eee823737fc34e22d4f4a51c8b359bdc0537909cfe2c1112e3570189216a",
        "native_capture": "a2bfadbe1ac1ec6bf54eca9c7fd01ee67c611b0b8d52966d874ae82c9274b25a",
        "native_cutr": "856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217",
        "native_clip": "9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4",
        "native_class_features": "49ab2384fbc01406eb7eb24ce89403bbfa9516bc213e11e8cd2014fa8eeea197",
        "native_class_names": "0d628e3140d491acfce107268fe51233e1df44f84581f582fe253842fc6557c9",
        "native_pst": "867f0546addc35a5000a421e9f81af4577470751b7a8ffc28e859cca97376660",
        "native_instances": "0fc5ff77f9fcbe55cfd79501066e4eb5f1d87abb0a3f6df38d2f5b651202d42b",
        "native_boxes": "510e16d63af8ca7f021ef65647afb39abfd50010ff58e8767cc13adf3224f5a6",
        "native_box_fusion": "76a1be9d2202527e50fc8e0d2c598367309812a45ff6cd0ca6405bfe19bcea23",
        "native_box_manager": "a9212008afa81488b2479c86096e148789cc127d5b91836b835475aa08e2dd49",
        "native_reliable_views": "e5cd196edba19dd92379d3fc865f48dbb656e4a684c4525e93610b9749c7231a",
        "native_cubify": "a965136cc37ebcbc2b7db01d9138254434e145e53541fa698d51ec4e5a5c16e7",
        "native_preprocessor": "9e32373f76d147d57e35823cd669e294c967dd909809f33f731e356f7ac32468",
        "native_measurement": "15a406dcab05c851cadefe3ca8624a7e26077a4fc540b69b1ed996f441ea7472",
        "native_orientation": "415b9d2f538035da6c6a47b928b238cb234e1de801b45ec07b34abd5eed8c9ce",
        "native_sensor": "ded126c543284797a92d8c0d6eb837900b1b74330012de4aa62cb9c9c55baace",
        "native_boxfusion_package_init": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "native_tools_package_init": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "native_tools_utils": "f27d123bc4f470a5e434e1516447d88376626a14113b6ccbdd59a8b0e7838942",
        "native_vit": "827998a562cb47f7d2d8fb0914bcb11de043e45e0d6daed1cce2e2ce564e4758",
        "native_batching": "213080093ed72e87fba94b665c1413905db042b6e4c1daa69291e8a05ffabb5f",
        "native_pos": "ce2ca3837bbf7742028e9c91c6e8c251488054b733c195f350742a1375ea7256",
        "native_transforms": "2790a0118d1fb8246ae040d4df9e79c3bef3bfb0254ec88d011424c0c1adeb2a",
        "native_imagelist": "002ba160272bd3da1d7535b2509f4e9e8850536fc867c559b433ce3a7f5a34b2",
        "native_color": "4db6fc7bdb6d6b856a7a2667863dea3804835571f898b637f5bc0b3e90de3b95",
        "native_proposal_cache": "0903f3472cac50a5a1ad69dadd5bc414989dcce0ddefc46669671b0d785f9265",
        "native_graw_fragments": "bc771d23fb3f69d89e9583f1ab9f76378ded93afe46ee6fd183150f64ca855fc",
        "native_graw_shadow": "7ab57f8f8a9641c55d90735f59a40da64b3e416cfad47f31aaaf45d8afacfeb3",
        "native_gclean_shadow": "4808b151c7d84926cf729b48da3d054f50ec64154b7e9119c976acdfb8279478",
        "native_puf_gclean_shadow": "b1fa9ca597a139d3eca47ec6ff2960e1fbe5ff8e181932b10174f78abf8cfb86",
        "native_observer_track_adapter": "181ec0b425da83ab1fec02c5fc0f0e9d85a71b9153b9f7b3b39e73ace2364798",
        "native_observer_track_registry": "346d207d27211fcde0f66674857f209ad6df43f8749c524271434297ecab60e7",
        "native_smov_fragments": "804430fa8431de2460ae77c51b1446cfc3e16f1f180f4ad15a1dfef954cf1789",
        "native_group3d_lite": "c25ada38a0de304a5dcdfa27da72dfe4c7d2c92857c67a1ffc9b1de0f66bc553",
        "native_group3d_lite_oracle": "d9dd89081f21601ca66153144abfa68241d91087786d643371513213d6abf05f",
        "native_puf_lite": "f6d8ee1f90ff5e9fda5a1eff7f4f22fbaf909d80deb048f3fa2b4cd28e7b132b",
        "provider_external_owlv2_model": "0931b38bb557529767f36f083feedf44d9ba72b1abc80ec7d898adedac3a7208",
        "provider_external_dinov3_wrapper": "7409b44952f6300df24e93d34a6a7aeefbe32e95dbe657a8660daf03344a24f4",
        "provider_external_demo_utils": "61827f436788d5f356439198c24c78f0d5aa6341cc37626663a1a97fa12e6c94",
        "provider_external_gravity": "6c1cfdba6d7b5171eacb1e9718f9bd172bd522bb3eb8ade274e8dd146cd81ac6",
        "provider_external_tw_obb": "557006dce4d02be37994cce52c81b5bf7957c01adf80b81628f75edbbdafc3a1",
        "provider_external_tw_tensor_utils": "e63d19e4ae8e97e7fafb9fbbd981bd35c50e6b67885d3943331e71c981221a10",
        "provider_external_tw_tensor_wrapper": "201232174fa95b18074844bae80fdfdbf891339040e2ccab7f162f3ca91acd9c",
        "provider_external_boxernet_init": "a114d2726afc5a6effdec28ce3ad895cbbe706dc1df04f3f07c7cc3d38c4acce",
        "provider_external_loaders_init": "a114d2726afc5a6effdec28ce3ad895cbbe706dc1df04f3f07c7cc3d38c4acce",
        "provider_external_owl_init": "a114d2726afc5a6effdec28ce3ad895cbbe706dc1df04f3f07c7cc3d38c4acce",
        "provider_external_utils_init": "a114d2726afc5a6effdec28ce3ad895cbbe706dc1df04f3f07c7cc3d38c4acce",
        "provider_external_tw_init": "a114d2726afc5a6effdec28ce3ad895cbbe706dc1df04f3f07c7cc3d38c4acce",
        "tracker": EXPECTED_TRACKER_SHA256,
        "tracker_test": EXPECTED_TRACKER_TEST_SHA256,
        "provider_runner": EXPECTED_PROVIDER_RUNNER_SHA256,
        "provider_runner_test": EXPECTED_PROVIDER_RUNNER_TEST_SHA256,
        "provider_core": EXPECTED_PROVIDER_CORE_SHA256,
        "provider_core_test": EXPECTED_PROVIDER_CORE_TEST_SHA256,
        "provider_contract": EXPECTED_PROVIDER_CONTRACT_SHA256,
        "native_manifest_builder": EXPECTED_NATIVE_MANIFEST_BUILDER_SHA256,
        "native_manifest_builder_test": EXPECTED_NATIVE_MANIFEST_BUILDER_TEST_SHA256,
        "native_python": NATIVE_PYTHON_SHA256,
        "native_setuptools_init": (
            "eddb9a7016889b1ceb51ad0e821233f25560689d5f230efeb8bdafd7abd8fd21"
        ),
        "native_torch_remote_instantiator": (
            "440a619c764e4133564d7956ba060a7223e94664854b94a4a2074d095756db7e"
        ),
        "provider_python": PROVIDER_PYTHON_SHA256,
        "system_git": "c3edb15c9715b79fcfb1fa978256cdfc14a9ad72a4a8d5680a9fc5ebc6a57e0e",
        "system_nvidia_smi": (
            "4b45d6578bea1488ca04c91f0b9252a5bbfc20b9058870755e2b48a755f0644a"
        ),
    }
)

QUEUE_MAXSIZE = 1
WORKER_READY_TIMEOUT_SECONDS = 900.0
FRAME_ACK_TIMEOUT_SECONDS = 120.0
NATIVE_LOCAL_COMPLETION_TIMEOUT_SECONDS = 110.0
WORKER_STOP_TIMEOUT_SECONDS = 30.0
PROVIDER_DEADLINE_SECONDS = 0.83333
TRACKER_P95_LIMIT_NS = 2_000_000
TRACKER_MAX_LIMIT_NS = 10_000_000
NATIVE_MIN_FPS = 10.0
MAX_NATIVE_FRAMES = 1_000_000
MAX_TIMING_JSON_BYTES = 256 * 1024 * 1024
MAX_FRAME_FILE_BYTES = 128 * 1024 * 1024
MAX_MATRIX_FILE_BYTES = 1024 * 1024
MAX_ASSET_BYTES = 4 * 1024 * 1024 * 1024
MAX_GIT_AUDIT_OUTPUT_BYTES = 32 * 1024 * 1024

REQUIRED_ENVIRONMENT: Mapping[str, str] = MappingProxyType(
    {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONHOME": "",
        "PYTHONPATH": "",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": FROZEN_PYCACHE_PREFIX,
        "PATH": TRUSTED_EXECUTABLE_PATH,
        "LD_LIBRARY_PATH": "",
        "LD_PRELOAD": "",
    }
)

PROVIDER_MEMBER = "provider_member"
PROVIDER_ABSTAIN = "provider_abstain_nonfinite_pose"
OUTSIDE_PROVIDER = "outside_provider_gap25"
PROVIDER_STATUSES = frozenset(
    {PROVIDER_MEMBER, PROVIDER_ABSTAIN, OUTSIDE_PROVIDER}
)

_HEX = frozenset("0123456789abcdef")
_COMMAND_KEYS = frozenset({"type", "sequence", "payload"})
_RESPONSE_KEYS = frozenset({"type", "sequence", "role", "payload"})


class IntegratedRuntimeError(RuntimeError):
    """A manifest, causal IPC, resource, or publication invariant failed."""


class RuntimeWorker(Protocol):
    role: str

    def ready(self) -> Mapping[str, Any]: ...

    def start_scene(self, scene: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def process_frame(self, frame: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def end_scene(self, scene_id: str) -> Mapping[str, Any]: ...

    def close(self) -> Mapping[str, Any]: ...


WorkerFactory = Callable[[Mapping[str, Any]], RuntimeWorker]


@dataclass(frozen=True)
class HarnessFactories:
    native: WorkerFactory
    provider: WorkerFactory


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise IntegratedRuntimeError(f"{label} must be lowercase SHA-256 hex")
    return str(value)


def _strict_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = (1 << 63) - 1,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise IntegratedRuntimeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise IntegratedRuntimeError(f"{label} is out of range")
    return result


def _relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise IntegratedRuntimeError(f"{label} must be a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise IntegratedRuntimeError(f"{label} is not normalized relative path")
    if path.as_posix() != value:
        raise IntegratedRuntimeError(f"{label} is not canonical")
    return value


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _read_regular_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = os.lstat(absolute)
    except OSError as error:
        raise IntegratedRuntimeError(f"cannot stat {label}: {absolute}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise IntegratedRuntimeError(f"{label} must be a non-symlink regular file")
    if before.st_size > maximum:
        raise IntegratedRuntimeError(f"{label} exceeds byte cap")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise IntegratedRuntimeError(f"{label} identity changed while opening")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) > maximum
            or len(payload) != opened.st_size
            or (after.st_size, after.st_mtime_ns)
            != (opened.st_size, opened.st_mtime_ns)
        ):
            raise IntegratedRuntimeError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return payload


def _read_held_descriptor(
    descriptor: int, *, maximum: int, label: str
) -> tuple[bytes, os.stat_result]:
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise IntegratedRuntimeError(f"{label} identity or size differs")
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as error:
        raise IntegratedRuntimeError(f"cannot read held {label}") from error
    if (
        len(payload) > maximum
        or len(payload) != before.st_size
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    ):
        raise IntegratedRuntimeError(f"{label} changed while reading held inode")
    return payload, after


class _HeldPinnedRegularFile:
    """Bind one public path, parent chain, inode, and exact bytes for a run."""

    def __init__(
        self,
        path: Path,
        *,
        maximum: int,
        label: str,
        expected_sha256: str,
    ) -> None:
        self.path = Path(os.path.abspath(os.fspath(path)))
        self.maximum = maximum
        self.label = label
        self.expected_sha256 = _require_sha256(
            expected_sha256, f"{label} expected hash"
        )
        self._directory_fds: list[int] = []
        self._directory_bindings: list[tuple[str, int, int]] = []
        self._descriptor = -1
        self._stat: os.stat_result | None = None
        self.payload = b""
        self.sha256 = ""
        try:
            try:
                self._directory_fds, self._directory_bindings = (
                    native_manifest_builder._open_bound_directory_chain(
                        self.path.parent
                    )
                )
            except BaseException as error:
                raise IntegratedRuntimeError(
                    f"cannot bind {label} parent chain"
                ) from error
            parent_fd = self._directory_fds[-1]
            try:
                named = os.stat(
                    self.path.name, dir_fd=parent_fd, follow_symlinks=False
                )
                if stat.S_ISLNK(named.st_mode) or not stat.S_ISREG(named.st_mode):
                    raise IntegratedRuntimeError(
                        f"{label} must be a non-symlink regular file"
                    )
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                self._descriptor = os.open(
                    self.path.name, flags, dir_fd=parent_fd
                )
                opened = os.fstat(self._descriptor)
            except OSError as error:
                raise IntegratedRuntimeError(f"cannot open bound {label}") from error
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            ):
                raise IntegratedRuntimeError(f"{label} identity changed while opening")
            self.payload, self._stat = _read_held_descriptor(
                self._descriptor, maximum=maximum, label=label
            )
            self.sha256 = _hash_bytes(self.payload)
            if self.sha256 != self.expected_sha256:
                raise IntegratedRuntimeError(f"{label} hash differs")
        except BaseException:
            self.close()
            raise

    def verify_after_stream(self) -> None:
        if self._descriptor < 0 or self._stat is None:
            raise IntegratedRuntimeError(f"{self.label} held binding is closed")
        try:
            native_manifest_builder._verify_directory_chain(
                self._directory_fds, self._directory_bindings
            )
            named = os.stat(
                self.path.name,
                dir_fd=self._directory_fds[-1],
                follow_symlinks=False,
            )
            held = os.fstat(self._descriptor)
        except BaseException as error:
            raise IntegratedRuntimeError(
                f"{self.label} public path changed during stream"
            ) from error
        frozen = self._stat
        frozen_identity = (
            frozen.st_dev,
            frozen.st_ino,
            frozen.st_size,
            frozen.st_mtime_ns,
        )
        if (
            stat.S_ISLNK(named.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns)
            != frozen_identity
            or (held.st_dev, held.st_ino, held.st_size, held.st_mtime_ns)
            != frozen_identity
        ):
            raise IntegratedRuntimeError(
                f"{self.label} public path/inode identity changed during stream"
            )
        payload, _ = _read_held_descriptor(
            self._descriptor, maximum=self.maximum, label=self.label
        )
        if payload != self.payload or _hash_bytes(payload) != self.sha256:
            raise IntegratedRuntimeError(f"{self.label} bytes changed during stream")

    def close(self) -> None:
        if self._descriptor >= 0:
            try:
                os.close(self._descriptor)
            finally:
                self._descriptor = -1
        for descriptor in reversed(self._directory_fds):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._directory_fds = []

    def __enter__(self) -> "_HeldPinnedRegularFile":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _hash_file(path: Path, *, maximum: int, label: str) -> str:
    return _hash_bytes(_read_regular_bytes(path, maximum=maximum, label=label))


def _stream_file_identity(
    path: Path, *, maximum: int, label: str
) -> dict[str, Any]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = os.lstat(absolute)
    except OSError as error:
        raise IntegratedRuntimeError(f"cannot stat {label}: {absolute}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise IntegratedRuntimeError(f"{label} must be a non-symlink regular file")
    if before.st_size > maximum:
        raise IntegratedRuntimeError(f"{label} exceeds byte cap")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise IntegratedRuntimeError(f"{label} identity changed while opening")
        digest = hashlib.sha256()
        count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > maximum:
                raise IntegratedRuntimeError(f"{label} exceeds byte cap")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            count != opened.st_size
            or (after.st_size, after.st_mtime_ns)
            != (opened.st_size, opened.st_mtime_ns)
        ):
            raise IntegratedRuntimeError(f"{label} changed while hashing")
        return {
            "sha256": digest.hexdigest(),
            "size": count,
            "device": int(opened.st_dev),
            "inode": int(opened.st_ino),
            "mtime_ns": int(opened.st_mtime_ns),
        }
    finally:
        os.close(descriptor)


def _ledger_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_json_bytes(dict(row)))
    return digest.hexdigest()


def _snapshot_asset_paths(
    paths: Mapping[str, Path], expected: Mapping[str, str]
) -> dict[str, Any]:
    if frozenset(paths) != frozenset(expected):
        raise IntegratedRuntimeError("asset path/expected ledgers differ")
    try:
        shadow_candidates = _assert_import_shadow_candidates_absent(
            tuple(Path(path) for path in paths.values())
        )
    except RuntimeError as error:
        raise IntegratedRuntimeError(
            "frozen asset import-shadow candidate differs"
        ) from error
    rows = []
    for name in sorted(paths):
        identity = _stream_file_identity(
            Path(paths[name]).resolve(strict=True),
            maximum=MAX_ASSET_BYTES,
            label=f"frozen asset {name}",
        )
        required = _require_sha256(expected[name], f"asset {name} expected hash")
        if identity["sha256"] != required:
            raise IntegratedRuntimeError(f"frozen asset {name} hash differs")
        rows.append(
            {
                "name": name,
                "sha256": identity["sha256"],
                "size": identity["size"],
            }
        )
    asset_entries_identity = _ledger_digest(rows)
    shadow_identity = _ledger_digest(
        [
            {"absolute_path": os.fspath(candidate), "absent": True}
            for candidate in shadow_candidates
        ]
    )
    combined_identity = _hash_bytes(
        _canonical_json_bytes(
            {
                "asset_entries_identity_sha256": asset_entries_identity,
                "import_shadow_candidates_identity_sha256": shadow_identity,
                "import_shadow_candidates_absent": True,
            }
        )
    )
    return {
        "entry_count": len(rows),
        "identity_sha256": combined_identity,
        "asset_entries_identity_sha256": asset_entries_identity,
        "entries": rows,
        "import_shadow_candidates_absent": True,
        "import_shadow_candidate_count": len(shadow_candidates),
        "import_shadow_candidates_identity_sha256": shadow_identity,
    }


def _minimal_external_command_environment(*, git: bool = False) -> dict[str, str]:
    """Return a non-inheriting environment for pinned external executables."""

    environment = {
        "PATH": TRUSTED_EXECUTABLE_PATH,
        "LC_ALL": "C",
        "LANG": "C",
        "LD_LIBRARY_PATH": "",
        "LD_PRELOAD": "",
    }
    if git:
        environment.update(
            {
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_SYSTEM": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
    return environment


def _snapshot_external_command_binaries() -> dict[str, Any]:
    """Hash-pin external commands before the first formal subprocess use."""

    return _snapshot_asset_paths(
        {
            "system_git": GIT_EXECUTABLE,
            "system_nvidia_smi": NVIDIA_SMI_EXECUTABLE,
        },
        {
            "system_git": NATIVE_ASSET_EXPECTED_SHA256["system_git"],
            "system_nvidia_smi": NATIVE_ASSET_EXPECTED_SHA256[
                "system_nvidia_smi"
            ],
        },
    )


def _minimal_python_probe_environment() -> dict[str, str]:
    """Return only frozen variables needed by a CPU child-runtime probe."""

    environment = {
        name: value
        for name, value in REQUIRED_ENVIRONMENT.items()
        if name not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment.update({"CUDA_VISIBLE_DEVICES": "", "LC_ALL": "C", "LANG": "C"})
    return environment


def _git_checkout_bytes_at_descriptor(
    descriptor: int, *arguments: str
) -> bytes:
    """Run a binary-safe read-only git query at the held checkout inode."""

    environment = _minimal_external_command_environment(git=True)
    try:
        result = subprocess.run(
            [
                os.fspath(GIT_EXECUTABLE),
                "-C",
                f"/proc/self/fd/{descriptor}",
                *arguments,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            pass_fds=(descriptor,),
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise IntegratedRuntimeError("cannot audit frozen provider checkout") from error
    if len(result.stdout) > MAX_GIT_AUDIT_OUTPUT_BYTES:
        raise IntegratedRuntimeError("provider checkout audit output exceeds cap")
    return result.stdout


def _git_checkout_text_at_descriptor(
    descriptor: int, *arguments: str
) -> str:
    """Run a UTF-8 read-only git query against the held checkout inode."""

    payload = _git_checkout_bytes_at_descriptor(descriptor, *arguments)
    try:
        return payload.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise IntegratedRuntimeError("provider checkout audit output is not UTF-8") from error


def _parse_git_nul_relative_paths(payload: bytes) -> tuple[str, ...]:
    """Parse an exact ``git ... -z`` path list without lossy stripping."""

    if not isinstance(payload, bytes):
        raise IntegratedRuntimeError("provider ignored-file audit is not bytes")
    if not payload:
        return ()
    if not payload.endswith(b"\0"):
        raise IntegratedRuntimeError("provider ignored-file audit lacks NUL terminator")
    raw_paths = payload[:-1].split(b"\0")
    if any(not raw_path for raw_path in raw_paths):
        raise IntegratedRuntimeError("provider ignored-file audit contains empty path")
    paths: list[str] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        try:
            relative = raw_path.decode("utf-8")
        except UnicodeDecodeError as error:
            raise IntegratedRuntimeError(
                "provider ignored-file path is not UTF-8"
            ) from error
        pure = PurePosixPath(relative)
        if (
            not relative
            or relative.startswith("/")
            or pure.as_posix() != relative
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise IntegratedRuntimeError("provider ignored-file path is not canonical")
        if relative in seen:
            raise IntegratedRuntimeError("provider ignored-file audit contains duplicate")
        seen.add(relative)
        paths.append(relative)
    return tuple(paths)


def _stream_checkout_regular_file_identity(
    checkout_descriptor: int,
    relative: str,
    *,
    maximum: int,
    label: str,
) -> dict[str, Any]:
    """Hash a checkout file through a no-symlink held-root component walk."""

    pure = PurePosixPath(relative)
    if (
        not relative
        or relative.startswith("/")
        or pure.as_posix() != relative
        or len(pure.parts) < 1
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise IntegratedRuntimeError(f"{label} path is not canonical")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    current_descriptor = os.dup(checkout_descriptor)
    try:
        for component in pure.parts[:-1]:
            try:
                named = os.stat(
                    component, dir_fd=current_descriptor, follow_symlinks=False
                )
                next_descriptor = os.open(
                    component, directory_flags, dir_fd=current_descriptor
                )
            except OSError as error:
                raise IntegratedRuntimeError(
                    f"{label} parent must be a non-symlink directory"
                ) from error
            try:
                opened = os.fstat(next_descriptor)
                if (
                    not stat.S_ISDIR(named.st_mode)
                    or not stat.S_ISDIR(opened.st_mode)
                    or (named.st_dev, named.st_ino)
                    != (opened.st_dev, opened.st_ino)
                ):
                    raise IntegratedRuntimeError(
                        f"{label} parent identity changed while opening"
                    )
            except BaseException:
                os.close(next_descriptor)
                raise
            os.close(current_descriptor)
            current_descriptor = next_descriptor

        try:
            named_file = os.stat(
                pure.parts[-1],
                dir_fd=current_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise IntegratedRuntimeError(f"cannot stat {label}") from error
        if not stat.S_ISREG(named_file.st_mode) or named_file.st_size > maximum:
            raise IntegratedRuntimeError(
                f"{label} must be a bounded non-symlink regular file"
            )
        try:
            file_descriptor = os.open(
                pure.parts[-1], file_flags, dir_fd=current_descriptor
            )
        except OSError as error:
            raise IntegratedRuntimeError(f"cannot open {label}") from error
        try:
            opened_file = os.fstat(file_descriptor)
            if (
                not stat.S_ISREG(opened_file.st_mode)
                or (opened_file.st_dev, opened_file.st_ino)
                != (named_file.st_dev, named_file.st_ino)
            ):
                raise IntegratedRuntimeError(
                    f"{label} identity changed while opening"
                )
            digest = hashlib.sha256()
            count = 0
            while True:
                chunk = os.read(file_descriptor, 1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > maximum:
                    raise IntegratedRuntimeError(f"{label} exceeds byte cap")
                digest.update(chunk)
            after_file = os.fstat(file_descriptor)
            if (
                count != opened_file.st_size
                or (after_file.st_size, after_file.st_mtime_ns)
                != (opened_file.st_size, opened_file.st_mtime_ns)
            ):
                raise IntegratedRuntimeError(f"{label} changed while hashing")
            return {"sha256": digest.hexdigest(), "size": count}
        finally:
            os.close(file_descriptor)
    finally:
        os.close(current_descriptor)


def _stream_checkout_symlink_target_identity(
    checkout_descriptor: int,
    relative: str,
    *,
    expected_target: str,
    maximum: int,
    label: str,
) -> dict[str, Any]:
    """Bind an exact checkout symlink and its resolved regular-file target."""

    pure = PurePosixPath(relative)
    if (
        not relative
        or relative.startswith("/")
        or pure.as_posix() != relative
        or len(pure.parts) < 1
        or any(part in {"", ".", ".."} for part in pure.parts)
        or not isinstance(expected_target, str)
        or not expected_target.startswith("/")
    ):
        raise IntegratedRuntimeError(f"{label} symlink binding is invalid")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current_descriptor = os.dup(checkout_descriptor)
    try:
        for component in pure.parts[:-1]:
            try:
                named = os.stat(
                    component, dir_fd=current_descriptor, follow_symlinks=False
                )
                next_descriptor = os.open(
                    component, directory_flags, dir_fd=current_descriptor
                )
            except OSError as error:
                raise IntegratedRuntimeError(
                    f"{label} parent must be a non-symlink directory"
                ) from error
            try:
                opened = os.fstat(next_descriptor)
                if (
                    not stat.S_ISDIR(named.st_mode)
                    or not stat.S_ISDIR(opened.st_mode)
                    or (named.st_dev, named.st_ino)
                    != (opened.st_dev, opened.st_ino)
                ):
                    raise IntegratedRuntimeError(
                        f"{label} parent identity changed while opening"
                    )
            except BaseException:
                os.close(next_descriptor)
                raise
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        try:
            before = os.stat(
                pure.parts[-1], dir_fd=current_descriptor, follow_symlinks=False
            )
            observed_target = os.readlink(
                pure.parts[-1], dir_fd=current_descriptor
            )
            after = os.stat(
                pure.parts[-1], dir_fd=current_descriptor, follow_symlinks=False
            )
        except OSError as error:
            raise IntegratedRuntimeError(f"cannot inspect {label} symlink") from error
        if (
            not stat.S_ISLNK(before.st_mode)
            or (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size)
            != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size)
            or observed_target != expected_target
        ):
            raise IntegratedRuntimeError(f"{label} symlink identity differs")
        try:
            resolved_target = Path(expected_target).resolve(strict=True)
        except OSError as error:
            raise IntegratedRuntimeError(f"cannot resolve {label} target") from error
        target_identity = _stream_file_identity(
            resolved_target, maximum=maximum, label=f"{label} target"
        )
        return {
            "symlink_target": observed_target,
            "resolved_target": os.fspath(resolved_target),
            "sha256": target_identity["sha256"],
            "size": target_identity["size"],
        }
    finally:
        os.close(current_descriptor)


def _snapshot_provider_ignored_files(
    checkout_descriptor: int,
    *,
    expected_checkpoints: Mapping[str, str],
    expected_dino_symlink_target: str,
) -> dict[str, Any]:
    """Fail closed over every ignored checkout file that could shadow code."""

    if frozenset(expected_checkpoints) != frozenset(
        PROVIDER_IGNORED_CHECKPOINT_SHA256
    ):
        raise IntegratedRuntimeError("provider ignored checkpoint paths differ")
    required_checkpoints = {
        relative: _require_sha256(digest, f"ignored checkpoint {relative} hash")
        for relative, digest in expected_checkpoints.items()
    }
    payload = _git_checkout_bytes_at_descriptor(
        checkout_descriptor,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    relative_paths = _parse_git_nul_relative_paths(payload)
    extension_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    rows: list[dict[str, Any]] = []
    observed_checkpoints: set[str] = set()
    for relative in relative_paths:
        pure = PurePosixPath(relative)
        if relative.endswith(extension_suffixes) or relative.endswith(".so"):
            raise IntegratedRuntimeError(
                "provider checkout contains an ignored extension module"
            )
        if relative in required_checkpoints:
            category = "hash_bound_checkpoint"
            observed_checkpoints.add(relative)
        elif pure.suffix == ".pyc" and pure.parent.name == "__pycache__":
            category = "redirected_adjacent_pycache"
        else:
            raise IntegratedRuntimeError(
                "provider checkout contains a non-allowlisted ignored file"
            )
        if relative == PROVIDER_IGNORED_DINO_RELPATH:
            identity = _stream_checkout_symlink_target_identity(
                checkout_descriptor,
                relative,
                expected_target=expected_dino_symlink_target,
                maximum=MAX_ASSET_BYTES,
                label=f"ignored provider file {relative}",
            )
            entry_type = "symlink_to_hash_bound_regular_file"
        else:
            identity = _stream_checkout_regular_file_identity(
                checkout_descriptor,
                relative,
                maximum=MAX_ASSET_BYTES,
                label=f"ignored provider file {relative}",
            )
            entry_type = "regular_file"
        if (
            category == "hash_bound_checkpoint"
            and identity["sha256"] != required_checkpoints[relative]
        ):
            raise IntegratedRuntimeError("ignored provider checkpoint hash differs")
        rows.append(
            {
                "relative_path": relative,
                "type": entry_type,
                "category": category,
                "size": identity["size"],
                "sha256": identity["sha256"],
                **(
                    {
                        "symlink_target": identity["symlink_target"],
                        "resolved_target": identity["resolved_target"],
                    }
                    if entry_type == "symlink_to_hash_bound_regular_file"
                    else {}
                ),
            }
        )
    if observed_checkpoints != set(required_checkpoints):
        raise IntegratedRuntimeError("ignored provider checkpoints are incomplete")
    rows.sort(key=lambda row: row["relative_path"])
    return {
        "entry_count": len(rows),
        "checkpoint_count": len(observed_checkpoints),
        "pycache_count": sum(
            row["category"] == "redirected_adjacent_pycache" for row in rows
        ),
        "entries": rows,
        "identity_sha256": _ledger_digest(rows),
    }


def _snapshot_provider_checkout(
    boxer_root: Path,
    *,
    expected_commit: str = EXPECTED_PROVIDER_BOXER_COMMIT,
    expected_tree: str = EXPECTED_PROVIDER_BOXER_TREE,
    expected_ignored_checkpoints: Mapping[str, str] = (
        PROVIDER_IGNORED_CHECKPOINT_SHA256
    ),
    expected_ignored_dino_symlink_target: str = (
        PROVIDER_IGNORED_DINO_SYMLINK_TARGET
    ),
) -> dict[str, Any]:
    """Bind tracked, ordinary-untracked, and allowlisted ignored checkout state."""

    if (
        not isinstance(expected_commit, str)
        or len(expected_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_commit)
        or not isinstance(expected_tree, str)
        or len(expected_tree) != 40
        or any(character not in "0123456789abcdef" for character in expected_tree)
    ):
        raise IntegratedRuntimeError("provider checkout git pins are invalid")
    absolute = Path(os.path.abspath(os.fspath(boxer_root)))
    try:
        named_before = os.lstat(absolute)
    except OSError as error:
        raise IntegratedRuntimeError("cannot inspect provider checkout") from error
    if stat.S_ISLNK(named_before.st_mode) or not stat.S_ISDIR(named_before.st_mode):
        raise IntegratedRuntimeError("provider checkout must be a non-symlink directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (
            named_before.st_dev,
            named_before.st_ino,
        ):
            raise IntegratedRuntimeError("provider checkout identity changed on open")
        top_level = _git_checkout_text_at_descriptor(
            descriptor, "rev-parse", "--show-toplevel"
        )
        commit = _git_checkout_text_at_descriptor(descriptor, "rev-parse", "HEAD")
        tree = _git_checkout_text_at_descriptor(
            descriptor, "rev-parse", "HEAD^{tree}"
        )
        status = _git_checkout_text_at_descriptor(
            descriptor,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        ignored = _snapshot_provider_ignored_files(
            descriptor,
            expected_checkpoints=expected_ignored_checkpoints,
            expected_dino_symlink_target=expected_ignored_dino_symlink_target,
        )
        opened_after = os.fstat(descriptor)
        named_after = os.lstat(absolute)
        if (
            (opened_after.st_dev, opened_after.st_ino)
            != (opened.st_dev, opened.st_ino)
            or (named_after.st_dev, named_after.st_ino)
            != (opened.st_dev, opened.st_ino)
        ):
            raise IntegratedRuntimeError("provider checkout identity changed during audit")
    finally:
        os.close(descriptor)
    if Path(top_level).resolve(strict=True) != absolute.resolve(strict=True):
        raise IntegratedRuntimeError("provider checkout top-level differs")
    if commit != expected_commit or tree != expected_tree:
        raise IntegratedRuntimeError("provider checkout commit/tree differs")
    if status:
        raise IntegratedRuntimeError("provider checkout is not clean including untracked")
    identity = {
        "absolute_path": os.fspath(absolute),
        "directory_device": int(opened.st_dev),
        "directory_inode": int(opened.st_ino),
        "commit": commit,
        "tree": tree,
        "status_porcelain_v1_empty": True,
        "ordinary_untracked_files_absent": True,
        "ignored_files_allowlist_enforced": True,
        "ignored_files": ignored,
    }
    return {**identity, "identity_sha256": _hash_bytes(_canonical_json_bytes(identity))}


def _native_static_asset_paths() -> dict[str, Path]:
    """Return the exact native/provider bridge assets pinned by this runner."""

    boxfusion_root = REPOSITORY_ROOT / "boxfusion"
    return {
        "native_config": NATIVE_CONFIG_PATH,
        "native_demo": NATIVE_DEMO_SOURCE,
        "native_capture": NATIVE_CAPTURE_SOURCE,
        "native_cutr": NATIVE_CUTR_CHECKPOINT,
        "native_clip": NATIVE_CLIP_CHECKPOINT,
        "native_class_features": NATIVE_CLASS_FEATURES,
        "native_class_names": NATIVE_CLASS_NAMES,
        "native_pst": NATIVE_PST,
        "native_instances": boxfusion_root / "instances.py",
        "native_boxes": boxfusion_root / "boxes.py",
        "native_box_fusion": boxfusion_root / "box_fusion.py",
        "native_box_manager": boxfusion_root / "box_manager.py",
        "native_reliable_views": boxfusion_root / "reliable_views.py",
        "native_cubify": boxfusion_root / "cubify_transformer.py",
        "native_preprocessor": boxfusion_root / "preprocessor.py",
        "native_measurement": boxfusion_root / "measurement.py",
        "native_orientation": boxfusion_root / "orientation.py",
        "native_sensor": boxfusion_root / "sensor.py",
        "native_boxfusion_package_init": boxfusion_root / "__init__.py",
        "native_tools_package_init": REPOSITORY_ROOT / "tools" / "__init__.py",
        "native_tools_utils": REPOSITORY_ROOT / "tools" / "utils.py",
        "native_vit": boxfusion_root / "vit.py",
        "native_batching": boxfusion_root / "batching.py",
        "native_pos": boxfusion_root / "pos.py",
        "native_transforms": boxfusion_root / "transforms.py",
        "native_imagelist": boxfusion_root / "imagelist.py",
        "native_color": boxfusion_root / "color.py",
        "native_proposal_cache": boxfusion_root / "proposal_cache.py",
        "native_graw_fragments": boxfusion_root / "graw_fragments.py",
        "native_graw_shadow": boxfusion_root / "graw_shadow.py",
        "native_gclean_shadow": boxfusion_root / "gclean_shadow.py",
        "native_puf_gclean_shadow": boxfusion_root / "puf_gclean_shadow.py",
        "native_observer_track_adapter": (
            boxfusion_root / "observer_track_adapter.py"
        ),
        "native_observer_track_registry": (
            boxfusion_root / "observer_track_registry.py"
        ),
        "native_smov_fragments": boxfusion_root / "smov_fragments.py",
        "native_group3d_lite": boxfusion_root / "group3d_lite.py",
        "native_group3d_lite_oracle": boxfusion_root / "group3d_lite_oracle.py",
        "native_puf_lite": boxfusion_root / "puf_lite.py",
        "provider_external_owlv2_model": PROVIDER_BOXER_ROOT
        / "owl"
        / "owlv2_model.py",
        "provider_external_dinov3_wrapper": PROVIDER_BOXER_ROOT
        / "boxernet"
        / "dinov3_wrapper.py",
        "provider_external_demo_utils": PROVIDER_BOXER_ROOT
        / "utils"
        / "demo_utils.py",
        "provider_external_gravity": PROVIDER_BOXER_ROOT / "utils" / "gravity.py",
        "provider_external_tw_obb": PROVIDER_BOXER_ROOT
        / "utils"
        / "tw"
        / "obb.py",
        "provider_external_tw_tensor_utils": PROVIDER_BOXER_ROOT
        / "utils"
        / "tw"
        / "tensor_utils.py",
        "provider_external_tw_tensor_wrapper": PROVIDER_BOXER_ROOT
        / "utils"
        / "tw"
        / "tensor_wrapper.py",
        "provider_external_boxernet_init": PROVIDER_BOXER_ROOT
        / "boxernet"
        / "__init__.py",
        "provider_external_loaders_init": PROVIDER_BOXER_ROOT
        / "loaders"
        / "__init__.py",
        "provider_external_owl_init": PROVIDER_BOXER_ROOT
        / "owl"
        / "__init__.py",
        "provider_external_utils_init": PROVIDER_BOXER_ROOT
        / "utils"
        / "__init__.py",
        "provider_external_tw_init": PROVIDER_BOXER_ROOT
        / "utils"
        / "tw"
        / "__init__.py",
        "tracker": boxfusion_root / "s3r_receipt_tracker.py",
        "tracker_test": REPOSITORY_ROOT / "tests" / "test_s3r_receipt_tracker.py",
        "provider_runner": PROVIDER_RUNNER_SOURCE,
        "provider_runner_test": (
            REPOSITORY_ROOT
            / "tests"
            / "test_run_scannet_s3r_h10_fresh_boxer_provider.py"
        ),
        "provider_core": boxfusion_root / "s3r_h10_provider_core.py",
        "provider_core_test": (
            REPOSITORY_ROOT / "tests" / "test_s3r_h10_provider_core.py"
        ),
        "provider_contract": PROVIDER_CONTRACT_PATH,
        "native_manifest_builder": NATIVE_MANIFEST_BUILDER_SOURCE,
        "native_manifest_builder_test": NATIVE_MANIFEST_BUILDER_TEST_SOURCE,
        "native_python": NATIVE_PYTHON_EXECUTABLE,
        "native_setuptools_init": NATIVE_SETUPTOOLS_INIT,
        "native_torch_remote_instantiator": NATIVE_TORCH_REMOTE_INSTANTIATOR,
        "provider_python": PROVIDER_PYTHON_EXECUTABLE,
        "system_git": GIT_EXECUTABLE,
        "system_nvidia_smi": NVIDIA_SMI_EXECUTABLE,
    }


def _snapshot_manifest_inputs(
    manifest_view: Mapping[str, Any], *, scene_root: Path
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scene in manifest_view["scenes"]:
        base = scene_root / scene["scene_id"]
        for role, relative, expected, maximum in (
            (
                "intrinsic_color",
                scene["intrinsic_color_relpath"],
                scene["intrinsic_color_sha256"],
                MAX_MATRIX_FILE_BYTES,
            ),
            (
                "intrinsic_depth",
                scene["intrinsic_depth_relpath"],
                scene["intrinsic_depth_sha256"],
                MAX_MATRIX_FILE_BYTES,
            ),
        ):
            identity = _stream_file_identity(
                base / relative,
                maximum=maximum,
                label=f"{scene['scene_id']} {role}",
            )
            if identity["sha256"] != expected:
                raise IntegratedRuntimeError("manifest intrinsic input hash differs")
            rows.append(
                {
                    "scene_id": scene["scene_id"],
                    "frame_id": None,
                    "role": role,
                    "relpath": relative,
                    "sha256": identity["sha256"],
                    "size": identity["size"],
                }
            )
        for frame in scene["frames"]:
            for role, relative, expected, maximum in (
                (
                    "color",
                    frame["color_relpath"],
                    frame["color_sha256"],
                    MAX_FRAME_FILE_BYTES,
                ),
                (
                    "depth",
                    frame["depth_relpath"],
                    frame["depth_sha256"],
                    MAX_FRAME_FILE_BYTES,
                ),
                (
                    "pose_raw",
                    frame["pose_relpath"],
                    frame["pose_sha256"],
                    MAX_MATRIX_FILE_BYTES,
                ),
                (
                    "pose_effective",
                    frame["effective_pose_relpath"],
                    frame["effective_pose_sha256"],
                    MAX_MATRIX_FILE_BYTES,
                ),
            ):
                identity = _stream_file_identity(
                    base / relative,
                    maximum=maximum,
                    label=f"{scene['scene_id']}/{frame['frame_id']} {role}",
                )
                if identity["sha256"] != expected:
                    raise IntegratedRuntimeError("manifest frame input hash differs")
                rows.append(
                    {
                        "scene_id": scene["scene_id"],
                        "frame_id": frame["frame_id"],
                        "role": role,
                        "relpath": relative,
                        "sha256": identity["sha256"],
                        "size": identity["size"],
                    }
                )
    return {
        "logical_entry_count": len(rows),
        "identity_sha256": _ledger_digest(rows),
    }


def _snapshot_t05_opaque(bundle: Any) -> dict[str, Any]:
    rows = []
    for scene in bundle.scenes:
        path = REPOSITORY_ROOT / scene.formal_t05_relpath
        identity = _stream_file_identity(
            path,
            maximum=MAX_ASSET_BYTES,
            label=f"opaque T05 {scene.scene_id}",
        )
        if identity["sha256"] != scene.formal_t05_sha256:
            raise IntegratedRuntimeError(f"opaque T05 {scene.scene_id} hash differs")
        rows.append(
            {
                "scene_id": scene.scene_id,
                "sha256": identity["sha256"],
                "size": identity["size"],
            }
        )
    return {
        "entry_count": len(rows),
        "identity_sha256": _ledger_digest(rows),
        "prediction_deserialization": False,
    }


def _validate_environment(*, require_cuda: bool) -> dict[str, Any]:
    observed = {name: os.environ.get(name, "") for name in REQUIRED_ENVIRONMENT}
    git_environment_names = sorted(
        name for name in os.environ if name.startswith("GIT_")
    )
    unexpected_loader = sorted(
        name
        for name in os.environ
        if name.startswith("LD_") and name not in REQUIRED_ENVIRONMENT
    )
    if observed != dict(REQUIRED_ENVIRONMENT) or git_environment_names or unexpected_loader:
        raise IntegratedRuntimeError(
            "runtime environment differs from frozen values: "
            + json.dumps(
                {
                    "required": observed,
                    "git_environment_names": git_environment_names,
                    "unexpected_loader_names": unexpected_loader,
                },
                sort_keys=True,
            )
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if require_cuda:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if len(tokens) != 1 or tokens[0] in ("-1", "none", "None"):
            raise IntegratedRuntimeError(
                "formal harness requires exactly one CUDA_VISIBLE_DEVICES token"
            )
    return {
        **observed,
        "CUDA_VISIBLE_DEVICES": visible,
        "git_environment_names_absent": True,
    }


def _minimal_manifest_view(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate fields used by IPC without weakening the builder validation."""

    if value.get("schema") != EXPECTED_NATIVE_MANIFEST_SCHEMA:
        raise IntegratedRuntimeError("native stream schema differs")
    scene_order = value.get("scene_order")
    scenes = value.get("scenes")
    if (
        not isinstance(scene_order, list)
        or not isinstance(scenes, list)
        or len(scene_order) != len(scenes)
        or not scene_order
        or len(set(scene_order)) != len(scene_order)
    ):
        raise IntegratedRuntimeError("native scene ledger is invalid")
    if _strict_int(value.get("scene_count"), "native scene count", minimum=1) != len(
        scene_order
    ):
        raise IntegratedRuntimeError("native scene count differs")
    total = _strict_int(
        value.get("native_frame_count"),
        "native_frame_count",
        minimum=1,
        maximum=MAX_NATIVE_FRAMES,
    )
    provider = value.get("provider_schedule")
    if not isinstance(provider, Mapping):
        raise IntegratedRuntimeError("provider schedule ledger is missing")
    _require_sha256(provider.get("sha256"), "provider schedule hash")

    normalized_scenes: list[dict[str, Any]] = []
    observed_total = 0
    provider_calls = 0
    abstentions = 0
    previous_global: tuple[int, int] | None = None
    for scene_index, (scene_id, scene) in enumerate(zip(scene_order, scenes)):
        if not isinstance(scene_id, str) or not scene_id:
            raise IntegratedRuntimeError("scene ID is invalid")
        if not isinstance(scene, Mapping) or scene.get("scene_id") != scene_id:
            raise IntegratedRuntimeError("scene order/record differs")
        frame_records = scene.get("frames")
        frame_ids = scene.get("frame_ids")
        if not isinstance(frame_records, list) or not isinstance(frame_ids, list):
            raise IntegratedRuntimeError(f"{scene_id} frame ledger is invalid")
        if len(frame_records) != len(frame_ids) or not frame_records:
            raise IntegratedRuntimeError(f"{scene_id} frame counts differ")
        if frame_ids != sorted(set(frame_ids)):
            raise IntegratedRuntimeError(f"{scene_id} frames are not strictly ordered")
        if _strict_int(
            scene.get("native_frame_count"), f"{scene_id} native frame count", minimum=1
        ) != len(frame_ids):
            raise IntegratedRuntimeError(f"{scene_id} native frame count differs")
        intrinsic_color_relpath = _relative_path(
            scene.get("intrinsic_color_relpath"), f"{scene_id} color intrinsic"
        )
        intrinsic_depth_relpath = _relative_path(
            scene.get("intrinsic_depth_relpath"), f"{scene_id} depth intrinsic"
        )
        intrinsic_color_sha256 = _require_sha256(
            scene.get("intrinsic_color_sha256"), f"{scene_id} color intrinsic"
        )
        intrinsic_depth_sha256 = _require_sha256(
            scene.get("intrinsic_depth_sha256"), f"{scene_id} depth intrinsic"
        )
        role_mounts = scene.get("role_mounts")
        if (
            not isinstance(role_mounts, Mapping)
            or frozenset(role_mounts) != frozenset({"color", "depth", "pose", "intrinsic"})
        ):
            raise IntegratedRuntimeError(f"{scene_id} role-mount ledger differs")
        normalized_frames: list[dict[str, Any]] = []
        latest_finite_id: int | None = None
        for scene_frame_index, (frame_id, frame) in enumerate(
            zip(frame_ids, frame_records)
        ):
            if not isinstance(frame, Mapping):
                raise IntegratedRuntimeError(f"{scene_id}/{frame_id} record is invalid")
            fid = _strict_int(frame_id, f"{scene_id} frame ID")
            if _strict_int(
                frame.get("frame_id"), f"{scene_id}/{fid} record frame ID"
            ) != fid:
                raise IntegratedRuntimeError(f"{scene_id}/{fid} frame identity differs")
            color_relpath = _relative_path(
                frame.get("color_relpath"), f"{scene_id}/{fid} color"
            )
            if PurePosixPath(color_relpath).suffix.lower() != ".jpg":
                raise IntegratedRuntimeError(
                    f"{scene_id}/{fid} native color producer must be JPG-only"
                )
            depth_relpath = _relative_path(
                frame.get("depth_relpath"), f"{scene_id}/{fid} depth"
            )
            pose_relpath = _relative_path(
                frame.get("pose_relpath"), f"{scene_id}/{fid} pose"
            )
            effective_relpath = _relative_path(
                frame.get("effective_pose_relpath"),
                f"{scene_id}/{fid} effective pose",
            )
            raw_finite = frame.get("raw_pose_finite")
            if not isinstance(raw_finite, bool):
                raise IntegratedRuntimeError(f"{scene_id}/{fid} pose flag is invalid")
            effective_id = _strict_int(
                frame.get("effective_pose_frame_id"),
                f"{scene_id}/{fid} effective pose ID",
            )
            if raw_finite:
                latest_finite_id = fid
                if effective_id != fid or effective_relpath != pose_relpath:
                    raise IntegratedRuntimeError("finite pose effective identity differs")
            elif latest_finite_id is None or effective_id != latest_finite_id:
                raise IntegratedRuntimeError("non-finite pose did not use past-most-recent")
            status = frame.get("provider_status")
            if status not in PROVIDER_STATUSES:
                raise IntegratedRuntimeError(f"{scene_id}/{fid} provider status differs")
            if (
                frame.get("intrinsic_color_relpath") != intrinsic_color_relpath
                or frame.get("intrinsic_color_sha256") != intrinsic_color_sha256
            ):
                raise IntegratedRuntimeError(
                    f"{scene_id}/{fid} color intrinsic identity differs"
                )
            provider_calls += status == PROVIDER_MEMBER
            abstentions += status == PROVIDER_ABSTAIN
            normalized_frames.append(
                {
                    "scene_id": scene_id,
                    "scene_index": scene_index,
                    "scene_frame_index": scene_frame_index,
                    "global_frame_index": observed_total,
                    "frame_id": fid,
                    "color_relpath": color_relpath,
                    "color_sha256": _require_sha256(
                        frame.get("color_sha256"), f"{scene_id}/{fid} color"
                    ),
                    "depth_relpath": depth_relpath,
                    "depth_sha256": _require_sha256(
                        frame.get("depth_sha256"), f"{scene_id}/{fid} depth"
                    ),
                    "pose_relpath": pose_relpath,
                    "pose_sha256": _require_sha256(
                        frame.get("pose_sha256"), f"{scene_id}/{fid} pose"
                    ),
                    "raw_pose_finite": raw_finite,
                    "effective_pose_frame_id": effective_id,
                    "effective_pose_relpath": effective_relpath,
                    "effective_pose_sha256": _require_sha256(
                        frame.get("effective_pose_sha256"),
                        f"{scene_id}/{fid} effective pose",
                    ),
                    "pose_resolution": frame.get("pose_resolution"),
                    "provider_status": status,
                }
            )
            current_global = (scene_index, scene_frame_index)
            if previous_global is not None and current_global <= previous_global:
                raise IntegratedRuntimeError("global frame order is not increasing")
            previous_global = current_global
            observed_total += 1
        normalized_scenes.append(
            {
                "scene_id": scene_id,
                "scene_index": scene_index,
                "native_frame_count": len(normalized_frames),
                "intrinsic_color_relpath": intrinsic_color_relpath,
                "intrinsic_color_sha256": intrinsic_color_sha256,
                "intrinsic_depth_relpath": intrinsic_depth_relpath,
                "intrinsic_depth_sha256": intrinsic_depth_sha256,
                "role_mounts": deepcopy(dict(role_mounts)),
                "provider_member_frame_count": sum(
                    row["provider_status"] == PROVIDER_MEMBER
                    for row in normalized_frames
                ),
                "provider_abstention_frame_count": sum(
                    row["provider_status"] == PROVIDER_ABSTAIN
                    for row in normalized_frames
                ),
                "frames": normalized_frames,
            }
        )
    if observed_total != total:
        raise IntegratedRuntimeError("native frame total differs")
    if _strict_int(
        provider.get("valid_frame_count"), "provider valid-frame count"
    ) != provider_calls:
        raise IntegratedRuntimeError("provider valid-call count differs")
    if _strict_int(
        provider.get("excluded_frame_count"), "provider excluded-frame count"
    ) != abstentions:
        raise IntegratedRuntimeError("provider abstention count differs")
    return {
        "schema": EXPECTED_NATIVE_MANIFEST_SCHEMA,
        "scene_order": list(scene_order),
        "scene_count": len(scene_order),
        "native_frame_count": observed_total,
        "provider_schedule_sha256": provider["sha256"],
        "provider_valid_call_count": provider_calls,
        "provider_abstention_count": abstentions,
        "scenes": normalized_scenes,
    }


def _scene_command_payload(scene: Mapping[str, Any], scene_root: Path) -> dict[str, Any]:
    return {
        "scene_id": scene["scene_id"],
        "scene_index": scene["scene_index"],
        "native_frame_count": scene["native_frame_count"],
        "scene_directory": os.fspath(
            Path(os.path.abspath(os.fspath(scene_root / scene["scene_id"])))
        ),
        "intrinsic_color_relpath": scene["intrinsic_color_relpath"],
        "intrinsic_color_sha256": scene["intrinsic_color_sha256"],
        "intrinsic_depth_relpath": scene["intrinsic_depth_relpath"],
        "intrinsic_depth_sha256": scene["intrinsic_depth_sha256"],
        "role_mounts": deepcopy(scene["role_mounts"]),
    }


def _frame_command_payload(frame: Mapping[str, Any]) -> dict[str, Any]:
    # Copy only the current frame.  No future record or full manifest is sent.
    return dict(frame)


def _frame_input_identity(
    frame: Mapping[str, Any],
    *,
    color_sha256: str,
    depth_sha256: str,
    pose_sha256: str,
    effective_pose_sha256: str,
) -> str:
    identity = {
        "scene_id": frame["scene_id"],
        "frame_id": int(frame["frame_id"]),
        "color_relpath": frame["color_relpath"],
        "color_sha256": _require_sha256(color_sha256, "observed color hash"),
        "depth_relpath": frame["depth_relpath"],
        "depth_sha256": _require_sha256(depth_sha256, "observed depth hash"),
        "pose_relpath": frame["pose_relpath"],
        "pose_sha256": _require_sha256(pose_sha256, "observed pose hash"),
        "effective_pose_relpath": frame["effective_pose_relpath"],
        "effective_pose_sha256": _require_sha256(
            effective_pose_sha256, "observed effective-pose hash"
        ),
    }
    return _hash_bytes(_canonical_json_bytes(identity))


def _expected_frame_input_identity(frame: Mapping[str, Any]) -> str:
    return _frame_input_identity(
        frame,
        color_sha256=frame["color_sha256"],
        depth_sha256=frame["depth_sha256"],
        pose_sha256=frame["pose_sha256"],
        effective_pose_sha256=frame["effective_pose_sha256"],
    )


def _role_filename(relative: str, role: str, label: str) -> str:
    normalized = _relative_path(relative, label)
    path = PurePosixPath(normalized)
    if len(path.parts) != 3 or path.parts[:2] != ("frames", role):
        raise IntegratedRuntimeError(f"{label} is outside the held {role} role")
    name = path.name
    if name in ("", ".", "..") or "/" in name or "\\" in name:
        raise IntegratedRuntimeError(f"{label} filename is invalid")
    return name


def _numeric_matrix(
    payload: bytes,
    label: str,
    *,
    finite: bool,
) -> np.ndarray:
    try:
        matrix = np.loadtxt(io.BytesIO(payload), dtype=np.float64)
    except (OSError, ValueError) as error:
        raise IntegratedRuntimeError(f"{label} is not a numeric matrix") from error
    if matrix.shape != (4, 4) or np.isnan(matrix).any():
        raise IntegratedRuntimeError(f"{label} must be a non-NaN 4x4 matrix")
    if finite and not np.isfinite(matrix).all():
        raise IntegratedRuntimeError(f"{label} must be finite")
    return np.ascontiguousarray(matrix)


@dataclass(frozen=True)
class ObservedCurrentFrame:
    scene_id: str
    frame_id: int
    scene_frame_index: int
    color_bytes: bytes
    depth_bytes: bytes
    pose_raw: np.ndarray
    pose_effective: np.ndarray
    intrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    color_sha256: str
    depth_sha256: str
    pose_sha256: str
    effective_pose_sha256: str
    input_identity_sha256: str


class HeldManifestSceneReader:
    """Open fixed role directories once and read only the commanded frame.

    No directory is enumerated.  Native mode sees every manifest frame and
    resolves an infinite raw pose solely from the already cached most-recent
    finite pose.  Provider mode may skip an outside/abstained command without
    opening any current-frame file.
    """

    def __init__(self, scene: Mapping[str, Any], *, mode: str):
        if mode not in ("native", "provider"):
            raise IntegratedRuntimeError("reader mode differs")
        self.mode = mode
        self.scene_id = str(scene["scene_id"])
        self.scene_directory = Path(str(scene["scene_directory"]))
        self.expected_frame_count = _strict_int(
            scene["native_frame_count"], "scene reader frame count", minimum=1
        )
        mounts = scene.get("role_mounts")
        if not isinstance(mounts, Mapping) or frozenset(mounts) != frozenset(
            {"color", "depth", "pose", "intrinsic"}
        ):
            raise IntegratedRuntimeError("scene reader mount ledger differs")
        self._mounts = deepcopy(dict(mounts))
        self._fds: dict[str, int] = {}
        self._next_scene_index = 0
        self._last_frame_id: int | None = None
        self._last_finite_frame_id: int | None = None
        self._last_finite_pose: np.ndarray | None = None
        self._last_finite_pose_hash: str | None = None
        self._closed = False
        frames_directory = self.scene_directory / "frames"
        try:
            for role in ("color", "depth", "pose", "intrinsic"):
                self._fds[role] = native_manifest_builder._open_role_directory(
                    frames_directory / role,
                    expected_mount=self._mounts[role],
                    scene_id=self.scene_id,
                    role=role,
                )
            self._intrinsic_color_bytes = self._read_role(
                "intrinsic",
                _role_filename(
                    str(scene["intrinsic_color_relpath"]),
                    "intrinsic",
                    "color intrinsic",
                ),
                str(scene["intrinsic_color_sha256"]),
                MAX_MATRIX_FILE_BYTES,
                "color intrinsic",
            )
            self._intrinsic_depth_bytes = self._read_role(
                "intrinsic",
                _role_filename(
                    str(scene["intrinsic_depth_relpath"]),
                    "intrinsic",
                    "depth intrinsic",
                ),
                str(scene["intrinsic_depth_sha256"]),
                MAX_MATRIX_FILE_BYTES,
                "depth intrinsic",
            )
            self.intrinsic_color = _numeric_matrix(
                self._intrinsic_color_bytes,
                f"{self.scene_id} color intrinsic",
                finite=True,
            )
            self.intrinsic_depth = _numeric_matrix(
                self._intrinsic_depth_bytes,
                f"{self.scene_id} depth intrinsic",
                finite=True,
            )
            self.intrinsic_color_sha256 = _hash_bytes(self._intrinsic_color_bytes)
            self.intrinsic_depth_sha256 = _hash_bytes(self._intrinsic_depth_bytes)
        except BaseException:
            self.close(abort=True)
            raise

    def _read_role(
        self,
        role: str,
        name: str,
        expected_sha256: str,
        maximum: int,
        label: str,
    ) -> bytes:
        expected = _require_sha256(expected_sha256, f"{label} expected hash")
        try:
            payload = native_manifest_builder._read_regular_bytes_at(
                self._fds[role],
                name,
                maximum=maximum,
                label=f"{self.scene_id} {label}",
            )
        except BaseException as error:
            raise IntegratedRuntimeError(
                f"cannot read held {self.scene_id} {label}"
            ) from error
        if _hash_bytes(payload) != expected:
            raise IntegratedRuntimeError(f"{self.scene_id} {label} hash differs")
        return payload

    def _accept_command(self, frame: Mapping[str, Any]) -> None:
        if self._closed:
            raise IntegratedRuntimeError("scene reader is closed")
        if frame.get("scene_id") != self.scene_id:
            raise IntegratedRuntimeError("scene reader command scene differs")
        frame_id = _strict_int(frame.get("frame_id"), "reader frame ID")
        scene_index = _strict_int(
            frame.get("scene_frame_index"), "reader scene-frame index"
        )
        if scene_index != self._next_scene_index:
            raise IntegratedRuntimeError("scene reader command order differs")
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise IntegratedRuntimeError("scene reader frame IDs are not increasing")
        self._next_scene_index += 1
        self._last_frame_id = frame_id

    def skip_current(self, frame: Mapping[str, Any]) -> None:
        if self.mode != "provider" or frame.get("provider_status") == PROVIDER_MEMBER:
            raise IntegratedRuntimeError("only a provider no-op frame may be skipped")
        self._accept_command(frame)

    def read_current(self, frame: Mapping[str, Any]) -> ObservedCurrentFrame:
        if self.mode == "provider" and frame.get("provider_status") != PROVIDER_MEMBER:
            raise IntegratedRuntimeError("provider cannot read an excluded current frame")
        self._accept_command(frame)
        frame_id = int(frame["frame_id"])
        color_name = _role_filename(
            str(frame["color_relpath"]), "color", "current color"
        )
        depth_name = _role_filename(
            str(frame["depth_relpath"]), "depth", "current depth"
        )
        pose_name = _role_filename(
            str(frame["pose_relpath"]), "pose", "current raw pose"
        )
        color_bytes = self._read_role(
            "color",
            color_name,
            str(frame["color_sha256"]),
            MAX_FRAME_FILE_BYTES,
            f"frame {frame_id} color",
        )
        depth_bytes = self._read_role(
            "depth",
            depth_name,
            str(frame["depth_sha256"]),
            MAX_FRAME_FILE_BYTES,
            f"frame {frame_id} depth",
        )
        pose_bytes = self._read_role(
            "pose",
            pose_name,
            str(frame["pose_sha256"]),
            MAX_MATRIX_FILE_BYTES,
            f"frame {frame_id} raw pose",
        )
        pose_raw = _numeric_matrix(
            pose_bytes,
            f"{self.scene_id}/{frame_id} raw pose",
            finite=False,
        )
        raw_finite = bool(np.isfinite(pose_raw).all())
        if raw_finite is not frame["raw_pose_finite"]:
            raise IntegratedRuntimeError("observed raw-pose finiteness differs")
        raw_hash = _hash_bytes(pose_bytes)
        if raw_finite:
            if (
                frame["effective_pose_frame_id"] != frame_id
                or frame["effective_pose_relpath"] != frame["pose_relpath"]
                or frame["effective_pose_sha256"] != raw_hash
            ):
                raise IntegratedRuntimeError("finite effective-pose identity differs")
            pose_effective = np.array(pose_raw, copy=True)
            effective_hash = raw_hash
            self._last_finite_frame_id = frame_id
            self._last_finite_pose = np.array(pose_effective, copy=True)
            self._last_finite_pose_hash = effective_hash
        else:
            if self.mode != "native":
                raise IntegratedRuntimeError("provider member pose must be finite")
            if (
                self._last_finite_frame_id is None
                or self._last_finite_pose is None
                or self._last_finite_pose_hash is None
                or frame["effective_pose_frame_id"] != self._last_finite_frame_id
                or frame["effective_pose_sha256"] != self._last_finite_pose_hash
            ):
                raise IntegratedRuntimeError(
                    "non-finite pose is not the cached most-recent past pose"
                )
            pose_effective = np.array(self._last_finite_pose, copy=True)
            effective_hash = self._last_finite_pose_hash
        identity = _frame_input_identity(
            frame,
            color_sha256=_hash_bytes(color_bytes),
            depth_sha256=_hash_bytes(depth_bytes),
            pose_sha256=raw_hash,
            effective_pose_sha256=effective_hash,
        )
        return ObservedCurrentFrame(
            scene_id=self.scene_id,
            frame_id=frame_id,
            scene_frame_index=int(frame["scene_frame_index"]),
            color_bytes=color_bytes,
            depth_bytes=depth_bytes,
            pose_raw=pose_raw,
            pose_effective=pose_effective,
            intrinsic_color=np.array(self.intrinsic_color, copy=True),
            intrinsic_depth=np.array(self.intrinsic_depth, copy=True),
            color_sha256=_hash_bytes(color_bytes),
            depth_sha256=_hash_bytes(depth_bytes),
            pose_sha256=raw_hash,
            effective_pose_sha256=effective_hash,
            input_identity_sha256=identity,
        )

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        if not abort:
            if self._next_scene_index != self.expected_frame_count:
                error = IntegratedRuntimeError("scene reader did not consume every command")
            frames_directory = self.scene_directory / "frames"
            for role, descriptor in self._fds.items():
                try:
                    opened = os.fstat(descriptor)
                    expected = self._mounts[role]
                    if (opened.st_dev, opened.st_ino) != (
                        expected["target_device"],
                        expected["target_inode"],
                    ):
                        raise IntegratedRuntimeError(
                            f"{self.scene_id} held {role} descriptor identity changed"
                        )
                    current = native_manifest_builder._mount_identity(
                        frames_directory / role,
                        role=role,
                        label=f"{self.scene_id} {role} directory",
                    )
                    if current != expected:
                        raise IntegratedRuntimeError(
                            f"{self.scene_id} {role} mount changed during stream"
                        )
                except BaseException as current_error:
                    if error is None:
                        error = current_error
        for descriptor in reversed(tuple(self._fds.values())):
            try:
                os.close(descriptor)
            except OSError as close_error:
                if error is None and not abort:
                    error = close_error
        self._fds.clear()
        if error is not None:
            raise IntegratedRuntimeError("scene reader close verification failed") from error


def _build_native_scannet_sample(frame: ObservedCurrentFrame) -> Mapping[str, Any]:
    """Reproduce ``ScannetDataset.__iter__`` for exactly one supplied frame.

    Imports are deliberately child-lazy.  The ``sensor_info.gt`` member below
    is the released code's sensor-pose/depth-camera namespace; no annotation,
    instance GT, semantic GT, evaluator input, or prediction file is opened.
    """

    cv2 = importlib.import_module("cv2")
    torch = importlib.import_module("torch")
    capture = importlib.import_module("boxfusion.capture_stream")

    color_bgr = cv2.imdecode(
        np.frombuffer(frame.color_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if color_bgr is None or color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise IntegratedRuntimeError("native current color is not 3-channel JPG")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth_raw = cv2.imdecode(
        np.frombuffer(frame.depth_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if depth_raw is None or depth_raw.ndim != 2:
        raise IntegratedRuntimeError("native current depth is not a 2D image")
    depth_m = depth_raw.astype(np.float32) / 1000.0
    height, width = depth_m.shape
    if (height, width) != (480, 640):
        raise IntegratedRuntimeError("native ScanNet depth shape differs from 480x640")
    color_rgb = cv2.resize(color_rgb, (width, height))
    if color_rgb.shape != (480, 640, 3):
        raise IntegratedRuntimeError("native aligned color shape differs")

    intrinsic = np.asarray(frame.intrinsic_depth, dtype=np.float32)[:3, :3]
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise IntegratedRuntimeError("native depth intrinsic differs")
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise IntegratedRuntimeError("native focal length is nonpositive")

    image_info = capture.ImageMeasurementInfo(
        size=(640, 480),
        K=torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )[None],
    )
    depth_info = capture.DepthMeasurementInfo(
        size=(640, 480),
        K=torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )[None],
    )
    result: dict[str, Any] = {"wide": {}}
    wide = capture.PosedSensorInfo()
    wide.image = image_info
    result["wide"]["image"] = torch.tensor(
        np.moveaxis(np.asarray(color_rgb).reshape((480, 640, 3)), -1, 0)
    )[None]
    wide.depth = depth_info
    resized_depth = cv2.resize(depth_m, (640, 480))
    result["wide"]["depth"] = torch.tensor(
        resized_depth.view(dtype=np.float32).reshape((480, 640))
    )[None].float()

    pose = np.asarray(frame.pose_effective, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise IntegratedRuntimeError("native effective pose is invalid")
    wide.RT = torch.from_numpy(pose.astype(np.float32).reshape((4, 4)))[None]
    current_orientation = wide.orientation
    target_orientation = capture.ImageOrientation.UPRIGHT
    gravity = capture.get_camera_to_gravity_transform(
        wide.RT[-1], current_orientation, target=target_orientation
    )
    wide = wide.orient(current_orientation, target_orientation)
    result["wide"]["image"] = capture.rotate_tensor(
        result["wide"]["image"], current_orientation, target=target_orientation
    )
    result["wide"]["depth"] = capture.rotate_tensor(
        result["wide"]["depth"], current_orientation, target=target_orientation
    )
    wide.RT = torch.eye(4)[None]
    wide.T_gravity = gravity[None]

    # This is the native API's camera-to-world carrier, despite its historical
    # attribute name.  It contains only the manifest pose and depth intrinsics.
    pose_carrier = capture.PosedSensorInfo()
    pose_carrier.RT = capture.parse_transform_4x4_np(pose)[None]
    pose_carrier.depth = depth_info
    sensor_info = capture.SensorArrayInfo()
    sensor_info.wide = wide
    sensor_info.gt = pose_carrier
    result["meta"] = {
        "video_id": [frame.scene_id],
        "timestamp": frame.scene_frame_index,
    }
    result["sensor_info"] = sensor_info
    return result


def _cuda_memory(torch_module: Any) -> tuple[int, int, int]:
    """Sample Torch allocator highs and device-wide use at a CUDA-sync boundary."""

    free_bytes, total_bytes = torch_module.cuda.mem_get_info()
    free = int(free_bytes)
    total = int(total_bytes)
    if free < 0 or total <= 0 or free > total:
        raise IntegratedRuntimeError("CUDA device-wide memory sample is invalid")
    return (
        int(torch_module.cuda.max_memory_allocated()),
        int(torch_module.cuda.max_memory_reserved()),
        total - free,
    )


def _cuda_gpu_uuid(torch_module: Any) -> str:
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
        raise IntegratedRuntimeError("worker must see exactly one CUDA device")
    properties = torch_module.cuda.get_device_properties(0)
    direct = getattr(properties, "uuid", None)
    if direct is not None:
        text = str(direct).strip()
        if text:
            return text
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        raise IntegratedRuntimeError("worker CUDA visibility is not one token")
    try:
        result = subprocess.run(
            [
                os.fspath(NVIDIA_SMI_EXECUTABLE),
                f"--id={visible}",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            env=_minimal_external_command_environment(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise IntegratedRuntimeError("cannot resolve CUDA GPU UUID") from error
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].startswith("GPU-"):
        raise IntegratedRuntimeError("nvidia-smi returned ambiguous GPU UUID")
    return lines[0]


def _cuda_device_identity(torch_module: Any) -> tuple[str, str, int, str]:
    uuid = _cuda_gpu_uuid(torch_module)
    properties = torch_module.cuda.get_device_properties(0)
    name = str(getattr(properties, "name", "")).strip()
    total_memory = int(getattr(properties, "total_memory", 0))
    if not name or total_memory <= 0:
        raise IntegratedRuntimeError("CUDA device name/total memory is invalid")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    try:
        result = subprocess.run(
            [
                os.fspath(NVIDIA_SMI_EXECUTABLE),
                f"--id={visible}",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            env=_minimal_external_command_environment(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise IntegratedRuntimeError("cannot resolve CUDA driver version") from error
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0]:
        raise IntegratedRuntimeError("CUDA driver version is ambiguous")
    return uuid, name, total_memory, lines[0]


class _ExpectedDemoExit(BaseException):
    def __init__(self, code: object):
        self.code = code


def _force_native_runtime_paths(cfg: dict[str, Any], pst_path: Path) -> None:
    """Bind BoxFusion's relative PST setting to the already-hashed absolute file."""

    resolved = Path(pst_path).resolve(strict=True)
    if not resolved.is_absolute() or resolved != NATIVE_PST.resolve(strict=True):
        raise IntegratedRuntimeError("native PST path differs from frozen absolute asset")
    box_fusion = cfg.get("box_fusion")
    if not isinstance(box_fusion, dict):
        raise IntegratedRuntimeError("native box_fusion configuration is invalid")
    box_fusion["pst_path"] = os.fspath(resolved)


def _restore_native_cubify_import_sys_path() -> None:
    """Audit Cubify's two frozen dependency side effects, then remove them."""

    original = tuple(sys.path)
    setuptools_module = sys.modules.get("setuptools")
    instantiator_module = sys.modules.get(
        "torch.distributed.nn.jit.instantiator"
    )
    generated_module = sys.modules.get("_remote_module_non_scriptable")

    def require_module_origin(module: object, expected: Path, label: str) -> None:
        module_file = getattr(module, "__file__", None)
        module_spec = getattr(module, "__spec__", None)
        spec_origin = getattr(module_spec, "origin", None)
        expected_text = os.fspath(expected)
        if (
            not isinstance(module_file, str)
            or os.path.abspath(module_file) != expected_text
            or spec_origin != expected_text
        ):
            raise IntegratedRuntimeError(f"{label} origin differs")

    if setuptools_module is None or instantiator_module is None:
        raise IntegratedRuntimeError("native Cubify import side-effect module is missing")
    require_module_origin(
        setuptools_module, NATIVE_SETUPTOOLS_INIT, "native setuptools"
    )
    require_module_origin(
        instantiator_module,
        NATIVE_TORCH_REMOTE_INSTANTIATOR,
        "native Torch remote instantiator",
    )
    remote_value = getattr(
        instantiator_module, "INSTANTIATED_TEMPLATE_DIR_PATH", None
    )
    if not isinstance(remote_value, str) or not remote_value:
        raise IntegratedRuntimeError("native Torch remote template path is invalid")
    remote_path = Path(remote_value)
    if (
        not remote_path.is_absolute()
        or os.path.abspath(remote_value) != remote_value
        or remote_path.parent != NATIVE_TORCH_REMOTE_TEMPLATE_PARENT
        or not remote_path.name.startswith("tmp")
        or len(remote_path.name) <= 3
        or os.path.realpath(remote_value) != remote_value
    ):
        raise IntegratedRuntimeError("native Torch remote template path differs")
    expected_observed = (
        *NATIVE_FROZEN_SYS_PATH,
        os.fspath(NATIVE_SETUPTOOLS_VENDOR),
        remote_value,
    )
    if original != expected_observed or len(set(original)) != len(original):
        raise IntegratedRuntimeError("native Cubify import sys.path delta differs")

    generated_path = remote_path / NATIVE_TORCH_REMOTE_TEMPLATE_BASENAME
    if generated_module is None:
        raise IntegratedRuntimeError("native Torch generated module is missing")
    require_module_origin(
        generated_module, generated_path, "native Torch generated module"
    )
    generated_spec = getattr(generated_module, "__spec__", None)
    if getattr(generated_spec, "name", None) != "_remote_module_non_scriptable":
        raise IntegratedRuntimeError("native Torch generated module name differs")

    directory_chains: list[tuple[list[int], list[tuple[str, int, int]]]] = []
    try:
        for directory in (NATIVE_SETUPTOOLS_VENDOR, remote_path):
            try:
                directory_chains.append(
                    native_manifest_builder._open_bound_directory_chain(directory)
                )
            except BaseException as error:
                raise IntegratedRuntimeError(
                    "native Cubify import directory binding differs"
                ) from error
        for descriptors, bindings in directory_chains:
            try:
                native_manifest_builder._verify_directory_chain(
                    descriptors, bindings
                )
            except BaseException as error:
                raise IntegratedRuntimeError(
                    "native Cubify import directory identity changed"
                ) from error

        remote_descriptor = directory_chains[1][0][-1]
        remote_stat = os.fstat(remote_descriptor)
        if (
            not stat.S_ISDIR(remote_stat.st_mode)
            or stat.S_IMODE(remote_stat.st_mode) != 0o700
            or remote_stat.st_uid != os.geteuid()
            or remote_stat.st_gid != os.getegid()
        ):
            raise IntegratedRuntimeError(
                "native Torch remote template directory identity differs"
            )
        try:
            entries = list(os.scandir(remote_descriptor))
        except OSError as error:
            raise IntegratedRuntimeError(
                "cannot inspect native Torch remote template directory"
            ) from error
        if (
            len(entries) != 1
            or entries[0].name != NATIVE_TORCH_REMOTE_TEMPLATE_BASENAME
            or entries[0].is_symlink()
            or not entries[0].is_file(follow_symlinks=False)
        ):
            raise IntegratedRuntimeError(
                "native Torch remote template directory contents differ"
            )
        generated_stat = os.stat(
            NATIVE_TORCH_REMOTE_TEMPLATE_BASENAME,
            dir_fd=remote_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(generated_stat.st_mode)
            or generated_stat.st_uid != os.geteuid()
            or generated_stat.st_gid != os.getegid()
        ):
            raise IntegratedRuntimeError(
                "native Torch generated module identity differs"
            )
        generated_identity = _stream_checkout_regular_file_identity(
            remote_descriptor,
            NATIVE_TORCH_REMOTE_TEMPLATE_BASENAME,
            maximum=MAX_MATRIX_FILE_BYTES,
            label="native Torch generated module",
        )
        if (
            generated_identity["size"] != NATIVE_TORCH_REMOTE_TEMPLATE_SIZE
            or generated_identity["sha256"]
            != NATIVE_TORCH_REMOTE_TEMPLATE_SHA256
        ):
            raise IntegratedRuntimeError("native Torch generated module bytes differ")

        loaded_from_remote: set[str] = set()
        for module_name, module in tuple(sys.modules.items()):
            candidates: list[str] = []
            module_file = getattr(module, "__file__", None)
            if isinstance(module_file, str):
                candidates.append(module_file)
            module_spec = getattr(module, "__spec__", None)
            spec_origin = getattr(module_spec, "origin", None)
            if isinstance(spec_origin, str):
                candidates.append(spec_origin)
            module_path = getattr(module, "__path__", None)
            if module_path is not None and not isinstance(module_path, str):
                try:
                    candidates.extend(
                        value for value in tuple(module_path) if isinstance(value, str)
                    )
                except TypeError:
                    pass
            for candidate in candidates:
                try:
                    Path(os.path.abspath(candidate)).relative_to(remote_path)
                except ValueError:
                    continue
                loaded_from_remote.add(module_name)
        if loaded_from_remote != {"_remote_module_non_scriptable"}:
            raise IntegratedRuntimeError(
                "native Torch remote template loaded-module closure differs"
            )
        for descriptors, bindings in directory_chains:
            try:
                native_manifest_builder._verify_directory_chain(
                    descriptors, bindings
                )
            except BaseException as error:
                raise IntegratedRuntimeError(
                    "native Cubify import directory identity changed"
                ) from error

        try:
            sys.path[:] = list(NATIVE_FROZEN_SYS_PATH)
            importlib.invalidate_caches()
            if tuple(sys.path) != NATIVE_FROZEN_SYS_PATH:
                raise IntegratedRuntimeError(
                    "native Cubify import sys.path restore failed"
                )
        except BaseException:
            sys.path[:] = list(original)
            raise
    finally:
        for descriptors, _bindings in reversed(directory_chains):
            for descriptor in reversed(descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass


class _FrozenNativeT05Engine:
    """Child-only frozen T05 model stack; construction performs all loading."""

    def __init__(self, config: Mapping[str, Any]):
        torch = importlib.import_module("torch")
        yaml = importlib.import_module("yaml")
        if not torch.cuda.is_available():
            raise IntegratedRuntimeError("native T05 requires CUDA")
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)
        self.torch = torch
        (
            self.gpu_uuid,
            self.gpu_device_name,
            self.gpu_total_memory_bytes,
            self.gpu_driver_version,
        ) = _cuda_device_identity(torch)

        config_path = Path(str(config["native_config_path"]))
        try:
            loaded = yaml.safe_load(
                _read_regular_bytes(
                    config_path, maximum=MAX_MATRIX_FILE_BYTES, label="native config"
                ).decode("utf-8")
            )
        except (UnicodeDecodeError, ValueError) as error:
            raise IntegratedRuntimeError("native config cannot be decoded") from error
        if not isinstance(loaded, Mapping):
            raise IntegratedRuntimeError("native config root must be a mapping")
        self.base_cfg = deepcopy(dict(loaded))
        if (
            self.base_cfg.get("dataset") != "scannet"
            or self.base_cfg.get("detection", {}).get("score_thresh") != 0.5
            or self.base_cfg.get("data", {}).get("gap") != 25
            or not self.base_cfg.get("box_fusion", {})
            .get("reliable_views", {})
            .get("enabled", False)
        ):
            raise IntegratedRuntimeError("native T05 configuration differs")
        self.pst_path = Path(str(config["pst_path"]))
        if (
            not self.pst_path.is_absolute()
            or self.pst_path.resolve(strict=True) != NATIVE_PST.resolve(strict=True)
        ):
            raise IntegratedRuntimeError("native PST path is not the frozen absolute asset")

        if tuple(sys.path) != NATIVE_FROZEN_SYS_PATH:
            raise IntegratedRuntimeError("native pre-Cubify sys.path differs")
        cubify = importlib.import_module("boxfusion.cubify_transformer")
        _restore_native_cubify_import_sys_path()
        preprocessor_module = importlib.import_module("boxfusion.preprocessor")
        if tuple(sys.path) != NATIVE_FROZEN_SYS_PATH:
            raise IntegratedRuntimeError("native post-preprocessor sys.path differs")
        tools_utils = importlib.import_module("tools.utils")
        if tuple(sys.path) != NATIVE_FROZEN_SYS_PATH:
            raise IntegratedRuntimeError("native post-tools sys.path differs")
        checkpoint = torch.load(
            str(config["cutr_checkpoint"]), map_location="cuda"
        )["model"]
        dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
        self.model = cubify.make_cubify_transformer(
            dimension=dimension, depth_model=True
        ).eval()
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to("cuda")
        self.clip_model, self.clip_preprocess = tools_utils.load_clip(
            str(config["clip_checkpoint"])
        )
        if tuple(sys.path) != NATIVE_FROZEN_SYS_PATH:
            raise IntegratedRuntimeError("native post-CLIP sys.path differs")
        self.text_class = np.genfromtxt(
            str(config["class_names"]), delimiter="\n", dtype=str
        )
        self.text_features = torch.load(
            str(config["class_features"]), map_location="cuda"
        ).cuda()
        self.augmentor = preprocessor_module.Augmentor(("wide/image", "wide/depth"))
        self.preprocessor = preprocessor_module.Preprocessor()
        self.demo = importlib.import_module("demo")
        if tuple(sys.path) != NATIVE_FROZEN_SYS_PATH:
            raise IntegratedRuntimeError("native post-demo sys.path differs")
        self._pycuda_driver = importlib.import_module("pycuda.driver")
        pycuda_primary = importlib.import_module("pycuda.autoprimaryctx")
        self._pycuda_primary_context = getattr(pycuda_primary, "context", None)
        if (
            self._pycuda_primary_context is None
            or self._pycuda_driver.Context.get_current() is None
        ):
            raise IntegratedRuntimeError("native primary CUDA context is absent")
        self._install_no_write_guards()
        torch.cuda.synchronize()
        expected_post_path = tuple(config["expected_post_factory_python_sys_path"])
        if (
            expected_post_path != NATIVE_POST_FACTORY_SYS_PATH
            or tuple(sys.path) != expected_post_path
            or len(set(sys.path)) != len(sys.path)
        ):
            raise IntegratedRuntimeError("native post-factory sys.path differs")

    def _install_no_write_guards(self) -> None:
        def forbidden(*_args: Any, **_kwargs: Any) -> None:
            raise IntegratedRuntimeError(
                "native timing harness reached a forbidden write/eval surface"
            )

        for name in (
            "save_box",
            "write_graw_shadow_diagnostics",
            "write_gclean_shadow_diagnostics",
            "write_puf_gclean_shadow_diagnostics",
            "write_smov_shadow_diagnostics",
            "load_data",
        ):
            if hasattr(self.demo, name):
                setattr(self.demo, name, forbidden)
        if hasattr(self.demo, "rerun"):
            self.demo.rerun.spawn = forbidden

    def synchronize(self) -> None:
        self.torch.cuda.synchronize()

    def memory(self) -> tuple[int, int, int]:
        return _cuda_memory(self.torch)

    @contextlib.contextmanager
    def thread_context(self):
        """Make the frozen primary PyCUDA context current in the demo thread."""

        self._pycuda_primary_context.push()
        try:
            if self._pycuda_driver.Context.get_current() is None:
                raise IntegratedRuntimeError(
                    "native demo thread CUDA context is absent"
                )
            yield
        finally:
            self._pycuda_driver.Context.pop()

    def run_scene(
        self,
        dataset: Any,
        *,
        scene_id: str,
        scene_directory: str,
    ) -> None:
        cfg = deepcopy(self.base_cfg)
        _force_native_runtime_paths(cfg, self.pst_path)
        cfg["data"]["datadir"] = os.fspath(Path(scene_directory) / "frames")
        cfg["data"]["start"] = 0
        cfg["data"]["output_dir"] = None
        cfg["eval"] = False
        cfg["vis"]["rerun"] = False
        cfg.setdefault("lifting", {})["proposal_cache"] = {"mode": "disabled"}
        cfg.setdefault("association", {})["group3d_lite"] = {
            "mode": "disabled",
            "diagnostics_root": None,
        }
        if cfg["data"]["output_dir"] is not None or cfg["eval"] or cfg["vis"]["rerun"]:
            raise IntegratedRuntimeError("native no-write configuration was not enforced")

        previous_exit = getattr(self.demo, "exit", None)
        had_exit = hasattr(self.demo, "exit")

        def terminal_exit(code: object = 0) -> None:
            raise _ExpectedDemoExit(code)

        self.demo.exit = terminal_exit
        try:
            self.demo.run(
                cfg,
                self.model,
                dataset,
                self.clip_model,
                self.clip_preprocess,
                self.text_class,
                self.text_features,
                self.augmentor,
                self.preprocessor,
                score_thresh=0.5,
                viz_on_gt_points=False,
                gap=25,
                re_vis=False,
            )
        except _ExpectedDemoExit as terminal:
            if terminal.code not in (0, None):
                raise IntegratedRuntimeError("native demo exited nonzero")
        finally:
            if had_exit:
                self.demo.exit = previous_exit
            else:
                delattr(self.demo, "exit")

class _CausalNativeDataset:
    """One-frame handoff iterator; completion is ACKed before the next get."""

    _STOP = object()

    def __init__(
        self,
        *,
        reader: HeldManifestSceneReader,
        engine: Any,
        frame_count: int,
        sample_builder: Callable[[ObservedCurrentFrame], Mapping[str, Any]],
    ):
        self.reader = reader
        self.engine = engine
        self.frame_count = frame_count
        self.sample_builder = sample_builder
        self.input_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self.completion_queue: queue.Queue[Mapping[str, Any]] = queue.Queue(maxsize=1)
        self.yielded_count = 0
        self._pending: tuple[Mapping[str, Any], ObservedCurrentFrame, int] | None = None
        self._stop_sent = False

    def __len__(self) -> int:
        # Explicit full-stream extension: prevents demo.run's N-25 early exit.
        return self.frame_count + 25

    def __iter__(self) -> "_CausalNativeDataset":
        return self

    def _completion_from_observation(
        self,
        frame: Mapping[str, Any],
        observed: ObservedCurrentFrame,
        started_ns: int,
    ) -> dict[str, Any]:
        self.engine.synchronize()
        synchronized_ns = time.perf_counter_ns()
        return {
            "ok": True,
            "scene_id": observed.scene_id,
            "frame_id": observed.frame_id,
            "current_read_started_ns": started_ns,
            "cuda_sync_finished_ns": synchronized_ns,
            "total_ns": synchronized_ns - started_ns,
            "input_identity_sha256": observed.input_identity_sha256,
            "color_sha256_observed": observed.color_sha256,
            "depth_sha256_observed": observed.depth_sha256,
            "pose_sha256_observed": observed.pose_sha256,
            "effective_pose_sha256_observed": observed.effective_pose_sha256,
            "model_scheduled": int(frame["frame_id"]) % 25 == 0,
        }

    def complete_pending(self) -> None:
        if self._pending is None:
            return
        frame, observed, started_ns = self._pending
        self._pending = None
        try:
            result = self._completion_from_observation(frame, observed, started_ns)
        except BaseException as error:
            result = {"ok": False, **_bounded_error(error)}
        self.completion_queue.put(result)

    def fail_pending(self, error: BaseException) -> None:
        # A demo can fail before it asks the iterator for its first frame (for
        # example while constructing BoxFusion).  Publish one terminal
        # completion even when no frame is pending so the first submit cannot
        # wait until the outer 120-second IPC deadline.
        self._pending = None
        try:
            self.completion_queue.put_nowait(
                {"ok": False, **_bounded_error(error)}
            )
        except queue.Full:
            # A completion already in the one-slot causal queue wins.  The
            # worker also records the thread error and rejects that result.
            pass

    def __next__(self) -> Mapping[str, Any]:
        self.complete_pending()
        command = self.input_queue.get()
        if command is self._STOP:
            raise StopIteration
        if not isinstance(command, Mapping):
            raise IntegratedRuntimeError("native local frame command differs")
        started_ns = time.perf_counter_ns()
        try:
            observed = self.reader.read_current(command)
            sample = self.sample_builder(observed)
        except BaseException as error:
            self.completion_queue.put(
                {"ok": False, **_bounded_error(error)}
            )
            raise
        self._pending = (dict(command), observed, started_ns)
        self.yielded_count += 1
        if self.yielded_count > self.frame_count:
            raise IntegratedRuntimeError("native dataset yielded beyond manifest")
        return sample

    def submit(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        deadline = time.monotonic() + NATIVE_LOCAL_COMPLETION_TIMEOUT_SECONDS
        try:
            self.input_queue.put(
                dict(frame), timeout=NATIVE_LOCAL_COMPLETION_TIMEOUT_SECONDS
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise IntegratedRuntimeError(
                    "native local frame completion timed out"
                )
            return self.completion_queue.get(timeout=remaining)
        except queue.Full as error:
            raise IntegratedRuntimeError(
                "native local frame input queue timed out"
            ) from error
        except queue.Empty as error:
            raise IntegratedRuntimeError(
                "native local frame completion timed out"
            ) from error

    def stop(self) -> None:
        if not self._stop_sent:
            self._stop_sent = True
            self.input_queue.put(self._STOP)


class NativeT05RuntimeWorker:
    role = "native"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        engine: Any | None = None,
        reader_factory: Callable[..., HeldManifestSceneReader] = HeldManifestSceneReader,
        sample_builder: Callable[[ObservedCurrentFrame], Mapping[str, Any]] = _build_native_scannet_sample,
    ):
        self._engine = engine if engine is not None else _FrozenNativeT05Engine(config)
        self._reader_factory = reader_factory
        self._sample_builder = sample_builder
        self._reader: HeldManifestSceneReader | None = None
        self._dataset: _CausalNativeDataset | None = None
        self._thread: threading.Thread | None = None
        self._thread_errors: list[BaseException] = []
        self._scene: dict[str, Any] | None = None
        self._scene_frames = 0
        self._frames_total = 0
        self._closed = False
        self._model_load_count = 1

    def _memory(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self._engine.memory())

    def ready(self) -> Mapping[str, Any]:
        allocated, reserved, device_used = self._memory()
        return {
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_uuid": str(self._engine.gpu_uuid),
            "gpu_device_name": str(self._engine.gpu_device_name),
            "gpu_total_memory_bytes": int(self._engine.gpu_total_memory_bytes),
            "gpu_driver_version": str(self._engine.gpu_driver_version),
            "model_load_count": 1,
            "initialization_complete": True,
            "torch_allocator_max_memory_allocated_bytes": allocated,
            "torch_allocator_max_memory_reserved_bytes": reserved,
            "device_wide_used_at_sync_bytes": device_used,
            "device_wide_memory_sampled_at_sync": True,
            "rss_peak_bytes": _peak_rss_bytes(),
            "tracker_execution_device": None,
            "tracker_gpu_execution": False,
            "tracker_gpu_bytes": 0,
            "owl_constructor_dummy_warmup": False,
            "full_pipeline_warmup": False,
            "first_real_forward_included": True,
            "first_real_owl_call_included": False,
            "first_real_owl_kernels_pre_warmed": False,
            "first_real_boxer_forward_included": False,
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "numpy_version": str(np.__version__),
            "torch_version": str(self._engine.torch.__version__),
            "cuda_version": str(self._engine.torch.version.cuda),
            "numpy_origin": os.path.realpath(str(np.__file__)),
            "torch_origin": os.path.realpath(str(self._engine.torch.__file__)),
        }

    def _thread_main(self) -> None:
        assert self._dataset is not None and self._scene is not None
        try:
            context_factory = getattr(self._engine, "thread_context", None)
            thread_context = (
                context_factory()
                if callable(context_factory)
                else contextlib.nullcontext()
            )
            with thread_context:
                self._engine.run_scene(
                    self._dataset,
                    scene_id=self._scene["scene_id"],
                    scene_directory=self._scene["scene_directory"],
                )
                if self._dataset.yielded_count < self._dataset.frame_count:
                    raise IntegratedRuntimeError(
                        "native demo returned before full stream"
                    )
                self._dataset.complete_pending()
        except BaseException as error:
            self._thread_errors.append(error)
            self._dataset.fail_pending(error)

    def start_scene(self, scene: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._closed or self._scene is not None:
            raise IntegratedRuntimeError("native scene lifecycle differs")
        self._scene = dict(scene)
        self._reader = self._reader_factory(scene, mode="native")
        self._dataset = _CausalNativeDataset(
            reader=self._reader,
            engine=self._engine,
            frame_count=int(scene["native_frame_count"]),
            sample_builder=self._sample_builder,
        )
        self._scene_frames = 0
        self._thread_errors = []
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"native-demo-{scene['scene_id']}",
            daemon=False,
        )
        self._thread.start()
        return {
            "scene_id": scene["scene_id"],
            "intrinsic_color_sha256_observed": self._reader.intrinsic_color_sha256,
            "intrinsic_depth_sha256_observed": self._reader.intrinsic_depth_sha256,
            "held_role_descriptors": True,
        }

    def process_frame(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._dataset is None or self._thread is None or self._scene is None:
            raise IntegratedRuntimeError("native FRAME outside scene")
        if not self._thread.is_alive() and self._thread_errors:
            raise IntegratedRuntimeError("native demo thread failed before FRAME")
        completion = self._dataset.submit(frame)
        if self._thread_errors:
            raise IntegratedRuntimeError("native demo thread failed during FRAME")
        if not completion.get("ok"):
            raise IntegratedRuntimeError("native demo current frame failed")
        if completion["scene_id"] != frame["scene_id"] or _strict_int(
            completion["frame_id"], "native local ACK frame"
        ) != frame["frame_id"]:
            raise IntegratedRuntimeError("native local ACK identity differs")
        self._scene_frames += 1
        self._frames_total += 1
        allocated, reserved, device_used = self._memory()
        return {
            "scene_id": completion["scene_id"],
            "frame_id": completion["frame_id"],
            "current_read_started_ns": completion["current_read_started_ns"],
            "cuda_sync_finished_ns": completion["cuda_sync_finished_ns"],
            "total_ns": completion["total_ns"],
            "torch_allocator_max_memory_allocated_bytes": allocated,
            "torch_allocator_max_memory_reserved_bytes": reserved,
            "device_wide_used_at_sync_bytes": device_used,
            "device_wide_memory_sampled_at_sync": True,
            "rss_peak_bytes": _peak_rss_bytes(),
            "oom_failure_reported": False,
            "cap_violation": False,
            "input_read": True,
            "input_identity_sha256": completion["input_identity_sha256"],
            "color_sha256_observed": completion["color_sha256_observed"],
            "depth_sha256_observed": completion["depth_sha256_observed"],
            "pose_sha256_observed": completion["pose_sha256_observed"],
            "effective_pose_sha256_observed": completion[
                "effective_pose_sha256_observed"
            ],
            "processed": True,
            "cuda_synchronized": True,
            "model_scheduled": completion["model_scheduled"],
        }

    def end_scene(self, scene_id: str) -> Mapping[str, Any]:
        if (
            self._scene is None
            or self._reader is None
            or self._dataset is None
            or self._thread is None
            or scene_id != self._scene["scene_id"]
        ):
            raise IntegratedRuntimeError("native END_SCENE lifecycle differs")
        finalize_started_ns = time.perf_counter_ns()
        self._thread.join(timeout=0.01)
        if self._thread.is_alive():
            self._dataset.stop()
            self._thread.join(timeout=FRAME_ACK_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise IntegratedRuntimeError("native demo thread did not stop")
        if self._thread_errors:
            raise IntegratedRuntimeError("native demo thread failed") from self._thread_errors[0]
        if self._scene_frames != int(self._scene["native_frame_count"]):
            raise IntegratedRuntimeError("native scene frame count differs")
        self._reader.close()
        self._engine.synchronize()
        synchronized_ns = time.perf_counter_ns()
        _, _, device_used = self._memory()
        result = {
            "scene_id": scene_id,
            "frames_processed": self._scene_frames,
            "provider_calls": 0,
            "provider_abstentions": 0,
            "raw_rows": 0,
            "k8_rows": 0,
            "tracker_commits": 0,
            "finalize_started_ns": finalize_started_ns,
            "cuda_sync_finished_ns": synchronized_ns,
            "total_ns": synchronized_ns - finalize_started_ns,
            "cuda_synchronized": True,
            "device_wide_used_at_sync_bytes": device_used,
            "device_wide_memory_sampled_at_sync": True,
        }
        self._reader = None
        self._dataset = None
        self._thread = None
        self._scene = None
        return result

    def close(self) -> Mapping[str, Any]:
        if self._closed:
            raise IntegratedRuntimeError("native worker closed twice")
        if self._scene is not None:
            raise IntegratedRuntimeError("native worker closed during scene")
        self._closed = True
        self._engine.synchronize()
        allocated, reserved, device_used = self._memory()
        return {
            "frames_processed": self._frames_total,
            "provider_calls": 0,
            "provider_abstentions": 0,
            "raw_rows": 0,
            "k8_rows": 0,
            "tracker_commits": 0,
            "model_load_count": self._model_load_count,
            "torch_allocator_max_memory_allocated_bytes": allocated,
            "torch_allocator_max_memory_reserved_bytes": reserved,
            "device_wide_used_at_sync_bytes": device_used,
            "device_wide_memory_sampled_at_sync": True,
            "rss_peak_bytes": _peak_rss_bytes(),
            "oom_failure_reported": False,
            "cap_violation": False,
        }


_OBB_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def _obb_corners_wxyz(
    center: np.ndarray, extent: np.ndarray, quaternion: np.ndarray
) -> np.ndarray:
    center_value = np.asarray(center, dtype=np.float64)
    extent_value = np.asarray(extent, dtype=np.float64)
    q = np.asarray(quaternion, dtype=np.float64)
    if (
        center_value.shape != (3,)
        or extent_value.shape != (3,)
        or q.shape != (4,)
        or not np.isfinite(center_value).all()
        or not np.isfinite(extent_value).all()
        or not np.isfinite(q).all()
        or np.any(extent_value <= 0.0)
    ):
        raise IntegratedRuntimeError("provider OBB row is invalid")
    norm_squared = float(q @ q)
    if not math.isfinite(norm_squared) or norm_squared <= 1e-12:
        raise IntegratedRuntimeError("provider OBB quaternion is invalid")
    w, x, y, z = q
    scale = 2.0 / norm_squared
    rotation = np.asarray(
        [
            [
                1.0 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
            ],
            [
                scale * (x * y + z * w),
                1.0 - scale * (x * x + z * z),
                scale * (y * z - x * w),
            ],
            [
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1.0 - scale * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
    return np.ascontiguousarray(
        _OBB_SIGNS * (extent_value / 2.0) @ rotation.T + center_value
    )


class _FrozenProviderEngine:
    """Child-only OWLv2/Boxer stack with no persistence surface."""

    def __init__(self, config: Mapping[str, Any]):
        self.fresh = importlib.import_module(
            "tools.run_scannet_s3r_h10_fresh_boxer_provider"
        )
        self.provider = self.fresh.FrozenBoxerProvider(
            Path(str(config["boxer_root"])),
            Path(str(config["boxer_checkpoint"])),
            Path(str(config["owl_checkpoint"])),
        )
        self.torch = self.provider.torch
        (
            self.gpu_uuid,
            self.gpu_device_name,
            self.gpu_total_memory_bytes,
            self.gpu_driver_version,
        ) = _cuda_device_identity(self.torch)
        self.provider.synchronize()
        expected_post_path = tuple(config["expected_post_factory_python_sys_path"])
        if (
            expected_post_path != PROVIDER_POST_FACTORY_SYS_PATH
            or tuple(sys.path) != expected_post_path
            or len(set(sys.path)) != len(sys.path)
        ):
            raise IntegratedRuntimeError("provider post-factory sys.path differs")

    def synchronize(self) -> None:
        self.provider.synchronize()

    def memory(self) -> tuple[int, int, int]:
        return _cuda_memory(self.torch)

    def reset_scene(self, scene_id: str) -> None:
        self.provider.reset_scene_seed(scene_id)

    def infer(
        self,
        observed: ObservedCurrentFrame,
        *,
        world_offset: np.ndarray,
    ) -> Any:
        datum = self.fresh._build_boxer_datum(
            boxer_root=self.provider.boxer_root,
            color_bytes=observed.color_bytes,
            depth_bytes=observed.depth_bytes,
            intrinsic=np.array(observed.intrinsic_color, copy=True),
            pose_absolute=np.array(observed.pose_effective, copy=True),
            world_offset_absolute=np.array(world_offset, copy=True),
            resize=int(self.provider.image_hw),
            frame_id=observed.frame_id,
        )
        frame = self.fresh.FrameDatum(
            scene_id=observed.scene_id,
            frame_id=observed.frame_id,
            boxer_datum=datum,
            world_offset_absolute=np.array(world_offset, copy=True),
        )
        return self.fresh._validate_raw_rows(self.provider.infer(frame))


class ProviderS3RRuntimeWorker:
    role = "provider"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        engine: Any | None = None,
        reader_factory: Callable[..., HeldManifestSceneReader] = HeldManifestSceneReader,
        tracker_factory: Callable[[], S3RReceiptTracker] = S3RReceiptTracker,
    ):
        self._engine = engine if engine is not None else _FrozenProviderEngine(config)
        self._reader_factory = reader_factory
        self._tracker_factory = tracker_factory
        self._reader: HeldManifestSceneReader | None = None
        self._tracker: S3RReceiptTracker | None = None
        self._scene: dict[str, Any] | None = None
        self._world_offset: np.ndarray | None = None
        self._scene_frames = 0
        self._scene_calls = 0
        self._scene_abstentions = 0
        self._scene_raw_rows = 0
        self._scene_k8_rows = 0
        self._scene_tracker_commits = 0
        self._frames_total = 0
        self._provider_call_index = 0
        self._raw_rows_total = 0
        self._k8_rows_total = 0
        self._tracker_commits_total = 0
        self._abstentions_total = 0
        self._closed = False
        self._model_load_count = 1
        self._last_memory = (0, 0, 0)

    def _memory(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self._engine.memory())

    def ready(self) -> Mapping[str, Any]:
        allocated, reserved, device_used = self._memory()
        self._last_memory = (allocated, reserved, device_used)
        return {
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_uuid": str(self._engine.gpu_uuid),
            "gpu_device_name": str(self._engine.gpu_device_name),
            "gpu_total_memory_bytes": int(self._engine.gpu_total_memory_bytes),
            "gpu_driver_version": str(self._engine.gpu_driver_version),
            "model_load_count": 1,
            "initialization_complete": True,
            "torch_allocator_max_memory_allocated_bytes": allocated,
            "torch_allocator_max_memory_reserved_bytes": reserved,
            "device_wide_used_at_sync_bytes": device_used,
            "device_wide_memory_sampled_at_sync": True,
            "rss_peak_bytes": _peak_rss_bytes(),
            "tracker_execution_device": "cpu",
            "tracker_gpu_execution": False,
            "tracker_gpu_bytes": 0,
            "owl_constructor_dummy_warmup": True,
            "full_pipeline_warmup": False,
            "first_real_forward_included": True,
            "first_real_owl_call_included": True,
            "first_real_owl_kernels_pre_warmed": True,
            "first_real_boxer_forward_included": True,
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "numpy_version": str(np.__version__),
            "torch_version": str(self._engine.torch.__version__),
            "cuda_version": str(self._engine.torch.version.cuda),
            "numpy_origin": os.path.realpath(str(np.__file__)),
            "torch_origin": os.path.realpath(str(self._engine.torch.__file__)),
        }

    def start_scene(self, scene: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._closed or self._scene is not None:
            raise IntegratedRuntimeError("provider scene lifecycle differs")
        self._scene = dict(scene)
        self._reader = self._reader_factory(scene, mode="provider")
        self._tracker = self._tracker_factory()
        self._engine.reset_scene(str(scene["scene_id"]))
        self._world_offset = None
        self._scene_frames = 0
        self._scene_calls = 0
        self._scene_abstentions = 0
        self._scene_raw_rows = 0
        self._scene_k8_rows = 0
        self._scene_tracker_commits = 0
        return {
            "scene_id": scene["scene_id"],
            "intrinsic_color_sha256_observed": self._reader.intrinsic_color_sha256,
            "intrinsic_depth_sha256_observed": self._reader.intrinsic_depth_sha256,
            "held_role_descriptors": True,
        }

    def _identity_fields(
        self, observed: ObservedCurrentFrame | None
    ) -> dict[str, Any]:
        if observed is None:
            return {
                "input_read": False,
                "input_identity_sha256": None,
                "color_sha256_observed": None,
                "depth_sha256_observed": None,
                "pose_sha256_observed": None,
                "effective_pose_sha256_observed": None,
            }
        return {
            "input_read": True,
            "input_identity_sha256": observed.input_identity_sha256,
            "color_sha256_observed": observed.color_sha256,
            "depth_sha256_observed": observed.depth_sha256,
            "pose_sha256_observed": observed.pose_sha256,
            "effective_pose_sha256_observed": observed.effective_pose_sha256,
        }

    def process_frame(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._scene is None or self._reader is None or self._tracker is None:
            raise IntegratedRuntimeError("provider FRAME outside scene")
        started_ns = time.perf_counter_ns()
        status = frame["provider_status"]
        observed: ObservedCurrentFrame | None = None
        raw_count = 0
        k8_count = 0
        tracker_ns = 0
        # A no-op frame must be invisible to both the model and tracker.  In
        # particular, do not snapshot/query/commit or synchronize CUDA for the
        # 18,600 outside frames or the single causal-pose abstention.
        audit_complete = True
        if status == PROVIDER_MEMBER:
            observed = self._reader.read_current(frame)
            if not np.isfinite(observed.pose_raw).all():
                raise IntegratedRuntimeError("provider member raw pose is non-finite")
            if self._world_offset is None:
                self._world_offset = np.array(
                    observed.pose_effective[:3, 3], dtype=np.float64, copy=True
                )
            rows = self._engine.infer(
                observed, world_offset=np.array(self._world_offset, copy=True)
            )
            self._engine.synchronize()
            center = np.asarray(rows.center, dtype=np.float64)
            extent = np.asarray(rows.extent, dtype=np.float64)
            quaternion = np.asarray(rows.quaternion, dtype=np.float64)
            score = np.asarray(rows.score, dtype=np.float64)
            raw_count = len(score)
            source_rows = np.arange(raw_count, dtype=np.int64)
            order = np.lexsort((source_rows, -score))[:8]
            observations = []
            for row_index in order.tolist():
                source_instance_id = self._provider_call_index * 2048 + row_index
                observations.append(
                    S3RObservation(
                        frame_id=int(frame["frame_id"]),
                        source_row=row_index,
                        sealed_npz_row=source_instance_id,
                        source_instance_id=source_instance_id,
                        score=float(score[row_index]),
                        corners=_obb_corners_wxyz(
                            center[row_index], extent[row_index], quaternion[row_index]
                        ),
                    )
                )
            k8_count = len(observations)
            tracker_started_ns = time.perf_counter_ns()
            query_token = self._tracker.query(int(frame["frame_id"]), observations)
            commit = self._tracker.commit(query_token)
            tracker_finished_ns = time.perf_counter_ns()
            tracker_ns = tracker_finished_ns - tracker_started_ns
            audit_complete = bool(query_token.audit_complete and commit.audit_complete)
            self._provider_call_index += 1
            self._scene_calls += 1
            self._scene_raw_rows += raw_count
            self._scene_k8_rows += k8_count
            self._scene_tracker_commits += 1
            self._raw_rows_total += raw_count
            self._k8_rows_total += k8_count
            self._tracker_commits_total += 1
            self._last_memory = self._memory()
        else:
            if status not in (PROVIDER_ABSTAIN, OUTSIDE_PROVIDER):
                raise IntegratedRuntimeError("provider status differs")
            self._reader.skip_current(frame)
            self._scene_abstentions += int(status == PROVIDER_ABSTAIN)
            self._abstentions_total += int(status == PROVIDER_ABSTAIN)
        synchronized_ns = time.perf_counter_ns()
        self._scene_frames += 1
        self._frames_total += 1
        allocated, reserved, device_used = self._last_memory
        memory_sampled = status == PROVIDER_MEMBER
        return {
            "scene_id": frame["scene_id"],
            "frame_id": frame["frame_id"],
            "current_read_started_ns": started_ns,
            "cuda_sync_finished_ns": synchronized_ns,
            "total_ns": synchronized_ns - started_ns,
            "torch_allocator_max_memory_allocated_bytes": allocated,
            "torch_allocator_max_memory_reserved_bytes": reserved,
            "device_wide_used_at_sync_bytes": (
                device_used if memory_sampled else None
            ),
            "device_wide_memory_sampled_at_sync": memory_sampled,
            "rss_peak_bytes": _peak_rss_bytes(),
            "oom_failure_reported": False,
            "cap_violation": not audit_complete,
            **self._identity_fields(observed),
            "provider_status": status,
            "call_executed": status == PROVIDER_MEMBER,
            "raw_row_count": raw_count,
            "k8_row_count": k8_count,
            "tracker_ns": tracker_ns,
            "tracker_audit_complete": audit_complete,
        }

    def end_scene(self, scene_id: str) -> Mapping[str, Any]:
        if (
            self._scene is None
            or self._reader is None
            or self._tracker is None
            or scene_id != self._scene["scene_id"]
        ):
            raise IntegratedRuntimeError("provider END_SCENE lifecycle differs")
        finalize_started_ns = time.perf_counter_ns()
        snapshot = self._tracker.snapshot()
        if snapshot.pending_frame_id is not None or not snapshot.audit_complete:
            raise IntegratedRuntimeError("provider tracker did not close cleanly")
        self._reader.close()
        self._engine.synchronize()
        synchronized_ns = time.perf_counter_ns()
        self._last_memory = self._memory()
        result = {
            "scene_id": scene_id,
            "frames_processed": self._scene_frames,
            "provider_calls": self._scene_calls,
            "provider_abstentions": self._scene_abstentions,
            "raw_rows": self._scene_raw_rows,
            "k8_rows": self._scene_k8_rows,
            "tracker_commits": self._scene_tracker_commits,
            "finalize_started_ns": finalize_started_ns,
            "cuda_sync_finished_ns": synchronized_ns,
            "total_ns": synchronized_ns - finalize_started_ns,
            "cuda_synchronized": True,
            "device_wide_used_at_sync_bytes": self._last_memory[2],
            "device_wide_memory_sampled_at_sync": True,
        }
        self._reader = None
        self._tracker = None
        self._scene = None
        self._world_offset = None
        return result

    def close(self) -> Mapping[str, Any]:
        if self._closed:
            raise IntegratedRuntimeError("provider worker closed twice")
        if self._scene is not None:
            raise IntegratedRuntimeError("provider worker closed during scene")
        self._closed = True
        self._engine.synchronize()
        allocated, reserved, device_used = self._memory()
        return {
            "frames_processed": self._frames_total,
            "provider_calls": self._provider_call_index,
            "provider_abstentions": self._abstentions_total,
            "raw_rows": self._raw_rows_total,
            "k8_rows": self._k8_rows_total,
            "tracker_commits": self._tracker_commits_total,
            "model_load_count": self._model_load_count,
            "torch_allocator_max_memory_allocated_bytes": allocated,
            "torch_allocator_max_memory_reserved_bytes": reserved,
            "device_wide_used_at_sync_bytes": device_used,
            "device_wide_memory_sampled_at_sync": True,
            "rss_peak_bytes": _peak_rss_bytes(),
            "oom_failure_reported": False,
            "cap_violation": self._provider_call_index > EXPECTED_PROVIDER_VALID_CALLS,
        }


def _real_native_factory(config: Mapping[str, Any]) -> RuntimeWorker:
    return NativeT05RuntimeWorker(config)


def _real_provider_factory(config: Mapping[str, Any]) -> RuntimeWorker:
    return ProviderS3RRuntimeWorker(config)


REAL_FACTORIES = HarnessFactories(
    native=_real_native_factory,
    provider=_real_provider_factory,
)


def _peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.  This harness is frozen to the Linux CUDA
    # host used by BoxFusion; failing on another unit is preferable to silently
    # publishing incomparable memory numbers.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IntegratedRuntimeError("ru_maxrss is not numeric")
    result = int(value) * 1024
    if result < 0:
        raise IntegratedRuntimeError("ru_maxrss is negative")
    return result


def _strict_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise IntegratedRuntimeError(f"{label} must be boolean")
    return value


def _strict_nonnegative_number(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise IntegratedRuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise IntegratedRuntimeError(f"{label} must be finite and nonnegative")
    return result


def _exact_mapping_keys(
    value: object, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegratedRuntimeError(f"{label} must be a mapping")
    actual = frozenset(value)
    if actual != expected:
        raise IntegratedRuntimeError(
            f"{label} keys differ: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    return value


_READY_PAYLOAD_KEYS = frozenset(
    {
        "process_started_ns",
        "ready_ns",
        "prestream_initialization_ns",
        "pid",
        "cuda_visible_devices",
        "gpu_uuid",
        "gpu_device_name",
        "gpu_total_memory_bytes",
        "gpu_driver_version",
        "model_load_count",
        "initialization_complete",
        "torch_allocator_max_memory_allocated_bytes",
        "torch_allocator_max_memory_reserved_bytes",
        "device_wide_used_at_sync_bytes",
        "device_wide_memory_sampled_at_sync",
        "rss_peak_bytes",
        "tracker_execution_device",
        "tracker_gpu_execution",
        "tracker_gpu_bytes",
        "model_lifecycle_fd1_fd2_redirected_to_devnull",
        "owl_constructor_dummy_warmup",
        "full_pipeline_warmup",
        "first_real_forward_included",
        "first_real_owl_call_included",
        "first_real_owl_kernels_pre_warmed",
        "first_real_boxer_forward_included",
        "python_executable",
        "python_version",
        "numpy_version",
        "torch_version",
        "cuda_version",
        "numpy_origin",
        "torch_origin",
        "python_pycache_prefix_environment",
        "python_pycache_prefix",
        "spawn_entry_python_sys_path_sha256",
        "post_factory_python_sys_path_sha256",
    }
)
_FRAME_COMMON_KEYS = frozenset(
    {
        "scene_id",
        "frame_id",
        "current_read_started_ns",
        "cuda_sync_finished_ns",
        "total_ns",
        "torch_allocator_max_memory_allocated_bytes",
        "torch_allocator_max_memory_reserved_bytes",
        "device_wide_used_at_sync_bytes",
        "device_wide_memory_sampled_at_sync",
        "rss_peak_bytes",
        "oom_failure_reported",
        "cap_violation",
        "input_read",
        "input_identity_sha256",
        "color_sha256_observed",
        "depth_sha256_observed",
        "pose_sha256_observed",
        "effective_pose_sha256_observed",
    }
)
_PROVIDER_FRAME_KEYS = _FRAME_COMMON_KEYS | frozenset(
    {
        "provider_status",
        "call_executed",
        "raw_row_count",
        "k8_row_count",
        "tracker_ns",
        "tracker_audit_complete",
    }
)
_NATIVE_FRAME_KEYS = _FRAME_COMMON_KEYS | frozenset(
    {"processed", "cuda_synchronized", "model_scheduled"}
)
_SCENE_PAYLOAD_KEYS = frozenset(
    {
        "scene_id",
        "intrinsic_color_sha256_observed",
        "intrinsic_depth_sha256_observed",
        "held_role_descriptors",
    }
)
_END_SCENE_PAYLOAD_KEYS = frozenset(
    {
        "scene_id",
        "frames_processed",
        "provider_calls",
        "provider_abstentions",
        "raw_rows",
        "k8_rows",
        "tracker_commits",
        "finalize_started_ns",
        "cuda_sync_finished_ns",
        "total_ns",
        "cuda_synchronized",
        "device_wide_used_at_sync_bytes",
        "device_wide_memory_sampled_at_sync",
    }
)
_CLOSE_PAYLOAD_KEYS = frozenset(
    {
        "frames_processed",
        "provider_calls",
        "provider_abstentions",
        "raw_rows",
        "k8_rows",
        "tracker_commits",
        "model_load_count",
        "torch_allocator_max_memory_allocated_bytes",
        "torch_allocator_max_memory_reserved_bytes",
        "device_wide_used_at_sync_bytes",
        "device_wide_memory_sampled_at_sync",
        "rss_peak_bytes",
        "oom_failure_reported",
        "cap_violation",
    }
)


def _validate_ready_payload(
    payload: object,
    *,
    role: str,
    expected_cuda_visible_devices: str,
    expected_python_executable: str | None = None,
    expected_runtime_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    value = _exact_mapping_keys(payload, _READY_PAYLOAD_KEYS, f"{role} READY")
    process_started_ns = _strict_int(
        value["process_started_ns"], f"{role} process start", minimum=1
    )
    ready_ns = _strict_int(value["ready_ns"], f"{role} READY time", minimum=process_started_ns)
    prestream_initialization_ns = _strict_int(
        value["prestream_initialization_ns"], f"{role} prestream initialization"
    )
    if prestream_initialization_ns != ready_ns - process_started_ns:
        raise IntegratedRuntimeError(f"{role} cold-start timing differs")
    pid = _strict_int(value["pid"], f"{role} PID", minimum=1)
    if value["cuda_visible_devices"] != expected_cuda_visible_devices:
        raise IntegratedRuntimeError(f"{role} CUDA_VISIBLE_DEVICES differs")
    gpu_uuid = value["gpu_uuid"]
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise IntegratedRuntimeError(f"{role} GPU UUID is empty")
    gpu_device_name = value["gpu_device_name"]
    gpu_driver_version = value["gpu_driver_version"]
    if (
        not isinstance(gpu_device_name, str)
        or not gpu_device_name
        or not isinstance(gpu_driver_version, str)
        or not gpu_driver_version
    ):
        raise IntegratedRuntimeError(f"{role} GPU device identity is invalid")
    gpu_total_memory_bytes = _strict_int(
        value["gpu_total_memory_bytes"], f"{role} GPU total memory", minimum=1
    )
    if _strict_int(value["model_load_count"], f"{role} model load count") != 1:
        raise IntegratedRuntimeError(f"{role} model was not loaded exactly once")
    if not _strict_bool(
        value["initialization_complete"], f"{role} initialization complete"
    ):
        raise IntegratedRuntimeError(f"{role} did not complete initialization")
    allocated = _strict_int(
        value["torch_allocator_max_memory_allocated_bytes"],
        f"{role} Torch allocator allocated high-water mark",
    )
    reserved = _strict_int(
        value["torch_allocator_max_memory_reserved_bytes"],
        f"{role} Torch allocator reserved high-water mark",
    )
    device_used = _strict_int(
        value["device_wide_used_at_sync_bytes"],
        f"{role} device-wide used memory at synchronization",
        maximum=gpu_total_memory_bytes,
    )
    if not _strict_bool(
        value["device_wide_memory_sampled_at_sync"],
        f"{role} device-wide synchronization sample flag",
    ):
        raise IntegratedRuntimeError(
            f"{role} READY lacks a device-wide synchronization sample"
        )
    rss = _strict_int(value["rss_peak_bytes"], f"{role} RSS")
    tracker_device = value["tracker_execution_device"]
    if tracker_device not in (None, "cpu"):
        raise IntegratedRuntimeError(f"{role} tracker device differs")
    tracker_gpu_execution = _strict_bool(
        value["tracker_gpu_execution"], f"{role} tracker GPU execution"
    )
    tracker_gpu_bytes = _strict_int(
        value["tracker_gpu_bytes"], f"{role} tracker GPU bytes"
    )
    if role == "provider":
        if tracker_device != "cpu" or tracker_gpu_execution or tracker_gpu_bytes:
            raise IntegratedRuntimeError("provider tracker must remain CPU-only")
    elif tracker_device is not None or tracker_gpu_execution or tracker_gpu_bytes:
        raise IntegratedRuntimeError("native READY contains tracker execution")
    if not _strict_bool(
        value["model_lifecycle_fd1_fd2_redirected_to_devnull"],
        f"{role} model-lifecycle stdout/stderr descriptor redirection",
    ):
        raise IntegratedRuntimeError(f"{role} native stdout/stderr descriptors escaped")
    warmup_observed = {
        name: _strict_bool(value[name], f"{role} {name}")
        for name in (
            "owl_constructor_dummy_warmup",
            "full_pipeline_warmup",
            "first_real_forward_included",
            "first_real_owl_call_included",
            "first_real_owl_kernels_pre_warmed",
            "first_real_boxer_forward_included",
        )
    }
    warmup_expected = (
        {
            "owl_constructor_dummy_warmup": True,
            "full_pipeline_warmup": False,
            "first_real_forward_included": True,
            "first_real_owl_call_included": True,
            "first_real_owl_kernels_pre_warmed": True,
            "first_real_boxer_forward_included": True,
        }
        if role == "provider"
        else {
            "owl_constructor_dummy_warmup": False,
            "full_pipeline_warmup": False,
            "first_real_forward_included": True,
            "first_real_owl_call_included": False,
            "first_real_owl_kernels_pre_warmed": False,
            "first_real_boxer_forward_included": False,
        }
    )
    if warmup_observed != warmup_expected:
        raise IntegratedRuntimeError(f"{role} warm-up disclosure differs")
    python_executable = value["python_executable"]
    python_version = value["python_version"]
    numpy_version = value["numpy_version"]
    torch_version = value["torch_version"]
    cuda_version = value["cuda_version"]
    numpy_origin = value["numpy_origin"]
    torch_origin = value["torch_origin"]
    python_pycache_prefix_environment = value[
        "python_pycache_prefix_environment"
    ]
    python_pycache_prefix = value["python_pycache_prefix"]
    spawn_entry_sys_path_sha256 = _require_sha256(
        value["spawn_entry_python_sys_path_sha256"],
        f"{role} spawn-entry Python sys.path digest",
    )
    post_factory_sys_path_sha256 = _require_sha256(
        value["post_factory_python_sys_path_sha256"],
        f"{role} post-factory Python sys.path digest",
    )
    if not all(
        isinstance(item, str) and item
        for item in (
            python_executable,
            python_version,
            numpy_version,
            torch_version,
            cuda_version,
            numpy_origin,
            torch_origin,
        )
    ):
        raise IntegratedRuntimeError(f"{role} runtime version identity is invalid")
    if (
        python_pycache_prefix_environment is not None
        and (
            not isinstance(python_pycache_prefix_environment, str)
            or not python_pycache_prefix_environment
        )
    ) or (
        python_pycache_prefix is not None
        and (
            not isinstance(python_pycache_prefix, str)
            or not python_pycache_prefix
        )
    ):
        raise IntegratedRuntimeError(f"{role} Python pycache prefix is invalid")
    if expected_python_executable is not None and Path(python_executable).resolve() != Path(
        expected_python_executable
    ).resolve():
        raise IntegratedRuntimeError(f"{role} Python executable differs")
    observed_runtime = {
        "python_version": python_version,
        "numpy_version": numpy_version,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "numpy_origin": os.path.realpath(numpy_origin),
        "torch_origin": os.path.realpath(torch_origin),
        "python_pycache_prefix_environment": (
            python_pycache_prefix_environment
        ),
        "python_pycache_prefix": python_pycache_prefix,
        "spawn_entry_python_sys_path_sha256": spawn_entry_sys_path_sha256,
        "post_factory_python_sys_path_sha256": post_factory_sys_path_sha256,
    }
    if expected_runtime_identity is not None and observed_runtime != dict(
        expected_runtime_identity
    ):
        raise IntegratedRuntimeError(f"{role} frozen runtime identity differs")
    return {
        "process_started_ns": process_started_ns,
        "ready_ns": ready_ns,
        "prestream_initialization_ns": prestream_initialization_ns,
        "pid": pid,
        "cuda_visible_devices": expected_cuda_visible_devices,
        "gpu_uuid": gpu_uuid,
        "gpu_device_name": gpu_device_name,
        "gpu_total_memory_bytes": gpu_total_memory_bytes,
        "gpu_driver_version": gpu_driver_version,
        "model_load_count": 1,
        "initialization_complete": True,
        "torch_allocator_max_memory_allocated_bytes": allocated,
        "torch_allocator_max_memory_reserved_bytes": reserved,
        "device_wide_used_at_sync_bytes": device_used,
        "device_wide_memory_sampled_at_sync": True,
        "rss_peak_bytes": rss,
        "tracker_execution_device": tracker_device,
        "tracker_gpu_execution": tracker_gpu_execution,
        "tracker_gpu_bytes": tracker_gpu_bytes,
        "model_lifecycle_fd1_fd2_redirected_to_devnull": True,
        **warmup_observed,
        "python_executable": python_executable,
        "python_version": python_version,
        "numpy_version": numpy_version,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "numpy_origin": os.path.realpath(numpy_origin),
        "torch_origin": os.path.realpath(torch_origin),
        "python_pycache_prefix_environment": (
            python_pycache_prefix_environment
        ),
        "python_pycache_prefix": python_pycache_prefix,
        "spawn_entry_python_sys_path_sha256": spawn_entry_sys_path_sha256,
        "post_factory_python_sys_path_sha256": post_factory_sys_path_sha256,
    }


def _validate_frame_payload(
    payload: object,
    *,
    role: str,
    frame: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _PROVIDER_FRAME_KEYS if role == "provider" else _NATIVE_FRAME_KEYS
    value = _exact_mapping_keys(payload, expected, f"{role} FRAME ACK")
    response_frame_id = _strict_int(
        value["frame_id"], f"{role} acknowledged frame ID"
    )
    if value["scene_id"] != frame["scene_id"] or response_frame_id != frame["frame_id"]:
        raise IntegratedRuntimeError(f"{role} acknowledged a different frame")
    started = _strict_int(
        value["current_read_started_ns"], f"{role} current-read timestamp", minimum=1
    )
    synchronized = _strict_int(
        value["cuda_sync_finished_ns"], f"{role} synchronization timestamp", minimum=started
    )
    total_ns = _strict_int(value["total_ns"], f"{role} frame total")
    if total_ns != synchronized - started:
        raise IntegratedRuntimeError(f"{role} frame timing is not end-to-end")
    normalized = {
        "scene_id": frame["scene_id"],
        "frame_id": response_frame_id,
        "current_read_started_ns": started,
        "cuda_sync_finished_ns": synchronized,
        "total_ns": total_ns,
        "torch_allocator_max_memory_allocated_bytes": _strict_int(
            value["torch_allocator_max_memory_allocated_bytes"],
            f"{role} Torch allocator allocated high-water mark",
        ),
        "torch_allocator_max_memory_reserved_bytes": _strict_int(
            value["torch_allocator_max_memory_reserved_bytes"],
            f"{role} Torch allocator reserved high-water mark",
        ),
        "rss_peak_bytes": _strict_int(value["rss_peak_bytes"], f"{role} RSS"),
        "oom_failure_reported": _strict_bool(
            value["oom_failure_reported"], f"{role} OOM failure"
        ),
        "cap_violation": _strict_bool(
            value["cap_violation"], f"{role} cap violation"
        ),
    }
    if normalized["oom_failure_reported"] or normalized["cap_violation"]:
        raise IntegratedRuntimeError(f"{role} reported an OOM failure/cap violation")
    if role == "provider":
        status = value["provider_status"]
        if status != frame["provider_status"]:
            raise IntegratedRuntimeError("provider status differs from manifest")
        executed = _strict_bool(value["call_executed"], "provider call flag")
        if executed != (status == PROVIDER_MEMBER):
            raise IntegratedRuntimeError("provider call/abstention contract differs")
        raw_count = _strict_int(
            value["raw_row_count"], "provider raw row count", maximum=2048
        )
        k8_count = _strict_int(
            value["k8_row_count"], "provider K8 row count", maximum=8
        )
        tracker_ns = _strict_int(value["tracker_ns"], "provider tracker time")
        audit = _strict_bool(
            value["tracker_audit_complete"], "provider tracker audit"
        )
        if not executed and (raw_count or k8_count or tracker_ns):
            raise IntegratedRuntimeError("abstained provider returned work metrics")
        if k8_count > raw_count:
            raise IntegratedRuntimeError("provider K8 count exceeds raw count")
        normalized.update(
            {
                "provider_status": status,
                "call_executed": executed,
                "raw_row_count": raw_count,
                "k8_row_count": k8_count,
                "tracker_ns": tracker_ns,
                "tracker_audit_complete": audit,
            }
        )
    else:
        if not _strict_bool(value["processed"], "native processed"):
            raise IntegratedRuntimeError("native did not process current frame")
        if not _strict_bool(value["cuda_synchronized"], "native CUDA synchronized"):
            raise IntegratedRuntimeError("native ACK preceded CUDA synchronization")
        model_scheduled = _strict_bool(
            value["model_scheduled"], "native gap-25 scheduled-slot flag"
        )
        if model_scheduled != (int(frame["frame_id"]) % 25 == 0):
            raise IntegratedRuntimeError("native gap-25 model schedule differs")
        normalized.update(
            {
                "processed": True,
                "cuda_synchronized": True,
                "model_scheduled": model_scheduled,
            }
        )

    memory_sampled = _strict_bool(
        value["device_wide_memory_sampled_at_sync"],
        f"{role} device-wide synchronization sample flag",
    )
    expected_memory_sample = role == "native" or normalized.get("call_executed") is True
    if memory_sampled != expected_memory_sample:
        raise IntegratedRuntimeError(
            f"{role} device-wide synchronization sampling contract differs"
        )
    if memory_sampled:
        device_used: int | None = _strict_int(
            value["device_wide_used_at_sync_bytes"],
            f"{role} device-wide used memory at synchronization",
        )
    else:
        if value["device_wide_used_at_sync_bytes"] is not None:
            raise IntegratedRuntimeError(
                "provider no-op returned a device-wide memory sample"
            )
        device_used = None
    normalized.update(
        {
            "device_wide_used_at_sync_bytes": device_used,
            "device_wide_memory_sampled_at_sync": memory_sampled,
        }
    )

    input_read = _strict_bool(value["input_read"], f"{role} input-read flag")
    expected_read = role == "native" or normalized.get("call_executed") is True
    if input_read != expected_read:
        raise IntegratedRuntimeError(f"{role} current-input read contract differs")
    identity_names = (
        "input_identity_sha256",
        "color_sha256_observed",
        "depth_sha256_observed",
        "pose_sha256_observed",
        "effective_pose_sha256_observed",
    )
    if input_read:
        observed = {
            name: _require_sha256(value[name], f"{role} {name}")
            for name in identity_names
        }
        expected_hashes = {
            "color_sha256_observed": frame["color_sha256"],
            "depth_sha256_observed": frame["depth_sha256"],
            "pose_sha256_observed": frame["pose_sha256"],
            "effective_pose_sha256_observed": frame["effective_pose_sha256"],
        }
        if any(observed[name] != expected for name, expected in expected_hashes.items()):
            raise IntegratedRuntimeError(f"{role} observed current-input hash differs")
        if observed["input_identity_sha256"] != _expected_frame_input_identity(frame):
            raise IntegratedRuntimeError(f"{role} current-input identity digest differs")
    else:
        if any(value[name] is not None for name in identity_names):
            raise IntegratedRuntimeError(f"{role} no-op ACK forged input identity")
        observed = {name: None for name in identity_names}
    normalized.update({"input_read": input_read, **observed})
    return normalized


def _response(
    response_type: str,
    sequence: int,
    role: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "type": response_type,
        "sequence": sequence,
        "role": role,
        "payload": dict(payload),
    }


def _bounded_error(_error: BaseException) -> dict[str, Any]:
    """Return one content-free worker failure code for the coordinator IPC."""

    return {"error_code": "worker_failure"}


def _worker_protocol_entry(
    role: str,
    factory: WorkerFactory,
    factory_config: Mapping[str, Any],
    command_queue: Any,
    response_queue: Any,
) -> None:
    """Run after the spawn target redirects FD 1/2 for the model lifecycle."""

    process_started_ns = time.perf_counter_ns()
    worker: RuntimeWorker | None = None
    last_sequence = -1
    try:
        if role not in ("native", "provider"):
            raise IntegratedRuntimeError("worker role differs")
        spawn_entry_sys_path = tuple(sys.path)
        spawn_entry_sys_path_sha256 = _sys_path_sha256(spawn_entry_sys_path)
        spawn_entry_pycache_prefix_environment = os.environ.get(
            "PYTHONPYCACHEPREFIX"
        )
        spawn_entry_pycache_prefix = sys.pycache_prefix
        configured_post_path = factory_config.get(
            "expected_post_factory_python_sys_path", spawn_entry_sys_path
        )
        if (
            not isinstance(configured_post_path, (list, tuple))
            or not configured_post_path
            or any(
                not isinstance(item, str) or not item
                for item in configured_post_path
            )
        ):
            raise IntegratedRuntimeError("worker expected post-factory sys.path is invalid")
        expected_post_path = tuple(configured_post_path)
        formal_runtime_identity = factory_config.get("python_executable") is not None
        if formal_runtime_identity:
            expected_runtime_identity = factory_config.get(
                "expected_runtime_identity"
            )
            if not isinstance(expected_runtime_identity, Mapping):
                raise IntegratedRuntimeError(
                    f"{role} configured runtime identity is invalid"
                )
            expected_entry_digest = _require_sha256(
                expected_runtime_identity.get(
                    "spawn_entry_python_sys_path_sha256"
                ),
                f"{role} configured spawn-entry sys.path digest",
            )
            expected_post_digest = _require_sha256(
                expected_runtime_identity.get(
                    "post_factory_python_sys_path_sha256"
                ),
                f"{role} configured post-factory sys.path digest",
            )
            role_expected_post = (
                NATIVE_POST_FACTORY_SYS_PATH
                if role == "native"
                else PROVIDER_POST_FACTORY_SYS_PATH
            )
            if (
                spawn_entry_sys_path_sha256 != expected_entry_digest
                or expected_post_path != role_expected_post
                or _sys_path_sha256(expected_post_path) != expected_post_digest
                or (
                    role == "native"
                    and expected_post_path != spawn_entry_sys_path
                )
                or (
                    role == "provider"
                    and expected_post_path
                    != (os.fspath(PROVIDER_BOXER_ROOT.resolve(strict=True)), *spawn_entry_sys_path)
                )
                or len(set(expected_post_path)) != len(expected_post_path)
                or expected_runtime_identity.get(
                    "python_pycache_prefix_environment"
                )
                != FROZEN_PYCACHE_PREFIX
                or expected_runtime_identity.get("python_pycache_prefix")
                != FROZEN_PYCACHE_PREFIX
                or spawn_entry_pycache_prefix_environment
                != FROZEN_PYCACHE_PREFIX
                or spawn_entry_pycache_prefix != FROZEN_PYCACHE_PREFIX
            ):
                raise IntegratedRuntimeError(
                    f"{role} configured entry/post-factory sys.path differs"
                )

        def assert_post_factory_runtime_identity() -> None:
            if (
                tuple(sys.path) != expected_post_path
                or os.environ.get("PYTHONPYCACHEPREFIX")
                != spawn_entry_pycache_prefix_environment
                or sys.pycache_prefix != spawn_entry_pycache_prefix
            ):
                raise IntegratedRuntimeError(
                    f"{role} post-factory runtime identity changed during lifecycle"
                )

        worker = factory(dict(factory_config))
        assert_post_factory_runtime_identity()
        if worker.role != role:
            raise IntegratedRuntimeError("worker factory returned a different role")
        assert_post_factory_runtime_identity()
        ready = dict(worker.ready())
        assert_post_factory_runtime_identity()
        ready_ns = time.perf_counter_ns()
        if any(
            name in ready
            for name in (
                "process_started_ns",
                "ready_ns",
                "prestream_initialization_ns",
                "model_lifecycle_fd1_fd2_redirected_to_devnull",
                "python_pycache_prefix_environment",
                "python_pycache_prefix",
                "spawn_entry_python_sys_path_sha256",
                "post_factory_python_sys_path_sha256",
            )
        ):
            raise IntegratedRuntimeError("worker may not forge coordinator cold-start fields")
        ready.update(
            {
                "process_started_ns": process_started_ns,
                "ready_ns": ready_ns,
                "prestream_initialization_ns": ready_ns - process_started_ns,
                "model_lifecycle_fd1_fd2_redirected_to_devnull": True,
                "python_pycache_prefix_environment": (
                    spawn_entry_pycache_prefix_environment
                ),
                "python_pycache_prefix": spawn_entry_pycache_prefix,
                "spawn_entry_python_sys_path_sha256": (
                    spawn_entry_sys_path_sha256
                ),
                "post_factory_python_sys_path_sha256": _sys_path_sha256(
                    expected_post_path
                ),
            }
        )
        response_queue.put(
            _response("READY", 0, role, ready), timeout=FRAME_ACK_TIMEOUT_SECONDS
        )
        while True:
            command = command_queue.get(timeout=FRAME_ACK_TIMEOUT_SECONDS)
            value = _exact_mapping_keys(command, _COMMAND_KEYS, f"{role} command")
            command_type = value["type"]
            sequence = _strict_int(value["sequence"], f"{role} command sequence")
            if sequence <= last_sequence:
                raise IntegratedRuntimeError("worker command sequence is not increasing")
            last_sequence = sequence
            payload = value["payload"]
            if not isinstance(payload, Mapping):
                raise IntegratedRuntimeError("worker command payload must be a mapping")
            assert_post_factory_runtime_identity()
            if command_type == "START_SCENE":
                result = worker.start_scene(payload)
                response_type = "SCENE_READY"
            elif command_type == "FRAME":
                result = worker.process_frame(payload)
                response_type = "FRAME_ACK"
            elif command_type == "END_SCENE":
                scene_id = payload.get("scene_id")
                if not isinstance(scene_id, str) or not scene_id:
                    raise IntegratedRuntimeError("END_SCENE identity is invalid")
                result = dict(worker.end_scene(scene_id))
                response_type = "SCENE_DONE"
            elif command_type == "STOP":
                if payload:
                    raise IntegratedRuntimeError("STOP payload must be empty")
                result = dict(worker.close())
                assert_post_factory_runtime_identity()
                response_queue.put(
                    _response("STOPPED", sequence, role, result),
                    timeout=FRAME_ACK_TIMEOUT_SECONDS,
                )
                return
            else:
                raise IntegratedRuntimeError("unknown worker command")
            assert_post_factory_runtime_identity()
            response_queue.put(
                _response(response_type, sequence, role, result),
                timeout=FRAME_ACK_TIMEOUT_SECONDS,
            )
    except BaseException as error:
        try:
            response_queue.put(
                _response("ERROR", max(last_sequence, 0), role, _bounded_error(error)),
                timeout=1.0,
            )
        except BaseException:
            pass
        if worker is not None:
            try:
                worker.close()
            except BaseException:
                pass


def _worker_entry(
    role: str,
    factory: WorkerFactory,
    factory_config: Mapping[str, Any],
    command_queue: Any,
    response_queue: Any,
) -> None:
    """Redirect FD 1/2 from target entry through factory/model shutdown."""

    try:
        flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
        null_descriptor = os.open(os.devnull, flags)
        try:
            for target in (1, 2):
                if null_descriptor == target:
                    os.set_inheritable(target, False)
                else:
                    os.dup2(null_descriptor, target, inheritable=False)
        finally:
            if null_descriptor not in (1, 2):
                os.close(null_descriptor)
    except BaseException as error:
        try:
            response_queue.put(
                _response("ERROR", 0, role, _bounded_error(error)), timeout=1.0
            )
        except BaseException:
            pass
        return
    _worker_protocol_entry(
        role,
        factory,
        factory_config,
        command_queue,
        response_queue,
    )


@dataclass
class _ProcessEndpoint:
    role: str
    process: Any
    command_queue: Any
    response_queue: Any
    outstanding: int = 0
    max_outstanding: int = 0


_SPAWN_CONFIGURATION_LOCK = threading.Lock()


def _spawn_runtime_endpoint(
    context: Any,
    *,
    role: str,
    factory: WorkerFactory,
    config: Mapping[str, Any],
) -> _ProcessEndpoint:
    frozen_config = dict(config)
    command_queue: Any | None = None
    response_queue: Any | None = None
    process: Any | None = None

    def cleanup_partial_spawn() -> None:
        if process is not None:
            try:
                if process.is_alive():
                    process.terminate()
            except BaseException:
                pass
            try:
                process.join(timeout=WORKER_STOP_TIMEOUT_SECONDS)
            except BaseException:
                pass
            try:
                if process.is_alive():
                    process.kill()
                    process.join(timeout=WORKER_STOP_TIMEOUT_SECONDS)
            except BaseException:
                pass
        for runtime_queue in (command_queue, response_queue):
            if runtime_queue is None:
                continue
            try:
                runtime_queue.close()
            except BaseException:
                pass
            try:
                runtime_queue.join_thread()
            except BaseException:
                pass

    try:
        command_queue = context.Queue(maxsize=QUEUE_MAXSIZE)
        response_queue = context.Queue(maxsize=QUEUE_MAXSIZE)
        process = context.Process(
            target=_worker_entry,
            args=(role, factory, frozen_config, command_queue, response_queue),
            name=f"s3r-h10-{role}",
        )
        child_executable = frozen_config.get("python_executable")
        child_sys_path = frozen_config.get("python_sys_path")
        expected_post_path = frozen_config.get(
            "expected_post_factory_python_sys_path"
        )
        if child_executable is not None:
            if not isinstance(child_executable, str) or not child_executable:
                raise IntegratedRuntimeError(f"{role} child executable is invalid")
            executable_path = Path(child_executable).resolve(strict=True)
            if not executable_path.is_file():
                raise IntegratedRuntimeError(f"{role} child executable is not a file")
            if (
                not isinstance(child_sys_path, (list, tuple))
                or not child_sys_path
                or any(not isinstance(item, str) or not item for item in child_sys_path)
                or child_sys_path[0] != os.fspath(REPOSITORY_ROOT)
                or len(set(child_sys_path)) != len(child_sys_path)
            ):
                raise IntegratedRuntimeError(f"{role} child sys.path is invalid")
            runtime_identity = frozen_config.get("expected_runtime_identity")
            if (
                not isinstance(runtime_identity, Mapping)
                or _require_sha256(
                    runtime_identity.get("spawn_entry_python_sys_path_sha256"),
                    f"{role} expected spawn-entry sys.path digest",
                )
                != _sys_path_sha256(tuple(child_sys_path))
                or not isinstance(expected_post_path, (list, tuple))
                or _require_sha256(
                    runtime_identity.get("post_factory_python_sys_path_sha256"),
                    f"{role} expected post-factory sys.path digest",
                )
                != _sys_path_sha256(tuple(expected_post_path))
                or runtime_identity.get("python_pycache_prefix_environment")
                != FROZEN_PYCACHE_PREFIX
                or runtime_identity.get("python_pycache_prefix")
                != FROZEN_PYCACHE_PREFIX
            ):
                raise IntegratedRuntimeError(f"{role} child sys.path digest differs")
        elif child_sys_path is not None or expected_post_path is not None:
            raise IntegratedRuntimeError(
                f"{role} child entry/post sys.path requires an explicit executable"
            )
        with _SPAWN_CONFIGURATION_LOCK:
            parent_executable = os.fsdecode(mp_spawn.get_executable())
            parent_sys_path = list(sys.path)
            start_error: BaseException | None = None
            restore_error: BaseException | None = None
            try:
                if child_executable is not None:
                    mp.set_executable(os.fspath(executable_path))
                    sys.path[:] = list(child_sys_path)
                process.start()
            except BaseException as error:
                start_error = error
            finally:
                try:
                    sys.path[:] = parent_sys_path
                except BaseException as error:
                    restore_error = error
                try:
                    mp.set_executable(parent_executable)
                except BaseException as error:
                    if restore_error is None:
                        restore_error = error
            if restore_error is not None:
                raise IntegratedRuntimeError(
                    "spawn configuration restoration failed"
                ) from restore_error
            if start_error is not None:
                raise start_error
    except BaseException:
        # A multiprocessing start may fail after partially launching a child.
        # Clean this not-yet-returned endpoint here; the coordinator cannot see
        # it and therefore cannot include it in its outer endpoint cleanup.
        cleanup_partial_spawn()
        raise
    assert process is not None and command_queue is not None and response_queue is not None
    return _ProcessEndpoint(role, process, command_queue, response_queue)


def _send_command(
    endpoint: _ProcessEndpoint,
    command_type: str,
    sequence: int,
    payload: Mapping[str, Any],
    *,
    timeout: float,
) -> int:
    if endpoint.outstanding != 0:
        raise IntegratedRuntimeError(
            f"{endpoint.role} queue backlog would exceed one command"
        )
    sent_ns = time.perf_counter_ns()
    try:
        endpoint.command_queue.put(
            {"type": command_type, "sequence": sequence, "payload": dict(payload)},
            timeout=timeout,
        )
    except queue.Full as error:
        raise IntegratedRuntimeError(f"{endpoint.role} command queue is full") from error
    endpoint.outstanding = 1
    endpoint.max_outstanding = max(endpoint.max_outstanding, endpoint.outstanding)
    return sent_ns


def _receive_response(
    endpoint: _ProcessEndpoint,
    expected_type: str,
    expected_sequence: int,
    *,
    timeout: float,
) -> tuple[Mapping[str, Any], int]:
    if endpoint.outstanding != 1 and expected_type != "READY":
        raise IntegratedRuntimeError(f"{endpoint.role} has no outstanding command")
    try:
        response = endpoint.response_queue.get(timeout=timeout)
    except queue.Empty as error:
        if not endpoint.process.is_alive():
            raise IntegratedRuntimeError(
                f"{endpoint.role} worker exited before {expected_type}"
            ) from error
        raise IntegratedRuntimeError(
            f"{endpoint.role} worker timed out waiting for {expected_type}"
        ) from error
    received_ns = time.perf_counter_ns()
    value = _exact_mapping_keys(response, _RESPONSE_KEYS, f"{endpoint.role} response")
    if value["role"] != endpoint.role:
        raise IntegratedRuntimeError("worker response role differs")
    if value["type"] == "ERROR":
        error_payload = value["payload"]
        if not isinstance(error_payload, Mapping) or dict(error_payload) != {
            "error_code": "worker_failure"
        }:
            raise IntegratedRuntimeError("worker failure payload differs")
        raise IntegratedRuntimeError(f"{endpoint.role} worker failed")
    response_sequence = _strict_int(
        value["sequence"], f"{endpoint.role} response sequence"
    )
    if value["type"] != expected_type or response_sequence != expected_sequence:
        raise IntegratedRuntimeError(
            f"{endpoint.role} response type/sequence differs"
        )
    if expected_type != "READY":
        endpoint.outstanding = 0
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise IntegratedRuntimeError("worker response payload must be a mapping")
    return payload, received_ns


def _terminate_endpoints(endpoints: Sequence[_ProcessEndpoint]) -> None:
    for endpoint in endpoints:
        if endpoint.process.is_alive():
            endpoint.process.terminate()
    for endpoint in endpoints:
        endpoint.process.join(timeout=WORKER_STOP_TIMEOUT_SECONDS)
        if endpoint.process.is_alive():
            endpoint.process.kill()
            endpoint.process.join(timeout=WORKER_STOP_TIMEOUT_SECONDS)
        try:
            endpoint.command_queue.close()
            endpoint.response_queue.close()
        except BaseException:
            pass


def _percentile_summary_ns(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "p50_ns": 0,
            "p95_ns": 0,
            "max_ns": 0,
            "p50_seconds": 0.0,
            "p95_seconds": 0.0,
            "max_seconds": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all() or np.any(array < 0):
        raise IntegratedRuntimeError("runtime samples are malformed")
    p50 = int(math.ceil(float(np.percentile(array, 50))))
    p95 = int(math.ceil(float(np.percentile(array, 95))))
    maximum = int(np.max(array))
    return {
        "count": len(values),
        "p50_ns": p50,
        "p95_ns": p95,
        "max_ns": maximum,
        "p50_seconds": p50 / 1_000_000_000.0,
        "p95_seconds": p95 / 1_000_000_000.0,
        "max_seconds": maximum / 1_000_000_000.0,
    }


def _execute_integrated_runtime(
    *,
    manifest_view: Mapping[str, Any],
    scene_root: Path,
    factories: HarnessFactories,
    native_factory_config: Mapping[str, Any],
    provider_factory_config: Mapping[str, Any],
    cuda_visible_devices: str,
    ready_timeout_seconds: float = WORKER_READY_TIMEOUT_SECONDS,
    frame_timeout_seconds: float = FRAME_ACK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the exact causal IPC stream and return a timing-only receipt.

    This private seam is intentionally injectable for spawn-process tests.  The
    public formal entry point below always supplies the frozen real factories.
    """

    view = _minimal_manifest_view(manifest_view)
    if not isinstance(cuda_visible_devices, str):
        raise IntegratedRuntimeError("CUDA_VISIBLE_DEVICES must be a string")
    for value, label in (
        (ready_timeout_seconds, "READY timeout"),
        (frame_timeout_seconds, "frame timeout"),
    ):
        if _strict_nonnegative_number(value, label) <= 0.0:
            raise IntegratedRuntimeError(f"{label} must be positive")

    context = mp.get_context("spawn")
    endpoints: list[_ProcessEndpoint] = []
    factory_configs: dict[str, dict[str, Any]] = {}
    try:
        for role, factory, config in (
            ("native", factories.native, native_factory_config),
            ("provider", factories.provider, provider_factory_config),
        ):
            frozen_config = dict(config)
            factory_configs[role] = frozen_config
            endpoints.append(
                _spawn_runtime_endpoint(
                    context,
                    role=role,
                    factory=factory,
                    config=frozen_config,
                )
            )
    except BaseException:
        _terminate_endpoints(endpoints)
        raise
    native, provider = endpoints

    sequence = 0
    native_frames: list[dict[str, Any]] = []
    provider_frames: list[dict[str, Any]] = []
    causal_ledger: list[dict[str, Any]] = []
    ready_records: dict[str, dict[str, Any]] = {}
    close_records: dict[str, dict[str, Any]] = {}
    scene_end_records: list[dict[str, Any]] = []
    last_native_sync_ns: int | None = None
    first_current_read_ns: int | None = None
    try:
        for endpoint in endpoints:
            payload, _ = _receive_response(
                endpoint, "READY", 0, timeout=ready_timeout_seconds
            )
            ready_records[endpoint.role] = _validate_ready_payload(
                payload,
                role=endpoint.role,
                expected_cuda_visible_devices=cuda_visible_devices,
                expected_python_executable=factory_configs[endpoint.role].get(
                    "python_executable"
                ),
                expected_runtime_identity=factory_configs[endpoint.role].get(
                    "expected_runtime_identity"
                ),
            )
        if ready_records["native"]["pid"] == ready_records["provider"]["pid"]:
            raise IntegratedRuntimeError("native/provider must be distinct processes")
        if ready_records["native"]["gpu_uuid"] != ready_records["provider"]["gpu_uuid"]:
            raise IntegratedRuntimeError("native/provider GPU UUID differs")
        for name in (
            "gpu_device_name",
            "gpu_total_memory_bytes",
            "gpu_driver_version",
        ):
            if ready_records["native"][name] != ready_records["provider"][name]:
                raise IntegratedRuntimeError(
                    f"native/provider {name} differs on the shared GPU"
                )

        for scene in view["scenes"]:
            scene_payload = _scene_command_payload(scene, scene_root)
            for endpoint in (provider, native):
                sequence += 1
                _send_command(
                    endpoint,
                    "START_SCENE",
                    sequence,
                    scene_payload,
                    timeout=frame_timeout_seconds,
                )
                payload, _ = _receive_response(
                    endpoint,
                    "SCENE_READY",
                    sequence,
                    timeout=frame_timeout_seconds,
                )
                value = _exact_mapping_keys(
                    payload, _SCENE_PAYLOAD_KEYS, f"{endpoint.role} SCENE_READY"
                )
                if value["scene_id"] != scene["scene_id"]:
                    raise IntegratedRuntimeError("scene READY identity differs")
                if (
                    value["intrinsic_color_sha256_observed"]
                    != scene["intrinsic_color_sha256"]
                    or value["intrinsic_depth_sha256_observed"]
                    != scene["intrinsic_depth_sha256"]
                ):
                    raise IntegratedRuntimeError(
                        f"{endpoint.role} observed intrinsic identity differs"
                    )
                if not _strict_bool(
                    value["held_role_descriptors"],
                    f"{endpoint.role} held role descriptors",
                ):
                    raise IntegratedRuntimeError(
                        f"{endpoint.role} did not bind scene role descriptors"
                    )

            for frame in scene["frames"]:
                current = _frame_command_payload(frame)
                sequence += 1
                provider_sent_ns = _send_command(
                    provider,
                    "FRAME",
                    sequence,
                    current,
                    timeout=frame_timeout_seconds,
                )
                provider_payload, provider_received_ns = _receive_response(
                    provider,
                    "FRAME_ACK",
                    sequence,
                    timeout=frame_timeout_seconds,
                )
                provider_frame = _validate_frame_payload(
                    provider_payload, role="provider", frame=frame
                )
                if provider_frame["current_read_started_ns"] < provider_sent_ns:
                    raise IntegratedRuntimeError("provider read predates current request")
                if provider_frame["cuda_sync_finished_ns"] > provider_received_ns:
                    raise IntegratedRuntimeError("provider ACK timestamp is in the future")

                sequence += 1
                native_sent_ns = _send_command(
                    native,
                    "FRAME",
                    sequence,
                    current,
                    timeout=frame_timeout_seconds,
                )
                if native_sent_ns < provider_received_ns:
                    raise IntegratedRuntimeError("native current frame preceded provider ACK")
                native_payload, native_received_ns = _receive_response(
                    native,
                    "FRAME_ACK",
                    sequence,
                    timeout=frame_timeout_seconds,
                )
                native_frame = _validate_frame_payload(
                    native_payload, role="native", frame=frame
                )
                if native_frame["current_read_started_ns"] < native_sent_ns:
                    raise IntegratedRuntimeError("native read predates provider ACK")
                if native_frame["cuda_sync_finished_ns"] > native_received_ns:
                    raise IntegratedRuntimeError("native ACK timestamp is in the future")
                if (
                    last_native_sync_ns is not None
                    and provider_frame["current_read_started_ns"] < last_native_sync_ns
                ):
                    raise IntegratedRuntimeError("future provider frame was prefetched")
                first_frame_reads = [native_frame["current_read_started_ns"]]
                if provider_frame["input_read"]:
                    first_frame_reads.append(provider_frame["current_read_started_ns"])
                first_current_read_ns = (
                    min(first_frame_reads)
                    if first_current_read_ns is None
                    else first_current_read_ns
                )
                last_native_sync_ns = native_frame["cuda_sync_finished_ns"]
                provider_frames.append(provider_frame)
                native_frames.append(native_frame)
                causal_ledger.append(
                    {
                        "scene_id": frame["scene_id"],
                        "frame_id": frame["frame_id"],
                        "global_frame_index": frame["global_frame_index"],
                        "provider_request_ns": provider_sent_ns,
                        "provider_ack_ns": provider_received_ns,
                        "native_request_ns": native_sent_ns,
                        "native_ack_ns": native_received_ns,
                    }
                )

            expected_scene_counts = {
                "native": {
                    "frames_processed": scene["native_frame_count"],
                    "provider_calls": 0,
                    "provider_abstentions": 0,
                    "raw_rows": 0,
                    "k8_rows": 0,
                    "tracker_commits": 0,
                },
                "provider": {
                    "frames_processed": scene["native_frame_count"],
                    "provider_calls": scene["provider_member_frame_count"],
                    "provider_abstentions": scene[
                        "provider_abstention_frame_count"
                    ],
                    "raw_rows": sum(
                        row["raw_row_count"]
                        for row in provider_frames[-scene["native_frame_count"] :]
                    ),
                    "k8_rows": sum(
                        row["k8_row_count"]
                        for row in provider_frames[-scene["native_frame_count"] :]
                    ),
                    "tracker_commits": scene["provider_member_frame_count"],
                },
            }
            for endpoint in (provider, native):
                sequence += 1
                _send_command(
                    endpoint,
                    "END_SCENE",
                    sequence,
                    {"scene_id": scene["scene_id"]},
                    timeout=frame_timeout_seconds,
                )
                payload, _ = _receive_response(
                    endpoint,
                    "SCENE_DONE",
                    sequence,
                    timeout=frame_timeout_seconds,
                )
                value = _exact_mapping_keys(
                    payload,
                    _END_SCENE_PAYLOAD_KEYS,
                    f"{endpoint.role} SCENE_DONE",
                )
                if value["scene_id"] != scene["scene_id"]:
                    raise IntegratedRuntimeError("scene DONE identity differs")
                observed_counts = {
                    "frames_processed": _strict_int(
                        value["frames_processed"], "scene frames processed"
                    ),
                    "provider_calls": _strict_int(
                        value["provider_calls"], "scene provider calls"
                    ),
                    "provider_abstentions": _strict_int(
                        value["provider_abstentions"],
                        "scene provider abstentions",
                    ),
                    "raw_rows": _strict_int(value["raw_rows"], "scene raw rows"),
                    "k8_rows": _strict_int(value["k8_rows"], "scene K8 rows"),
                    "tracker_commits": _strict_int(
                        value["tracker_commits"], "scene tracker commits"
                    ),
                }
                if observed_counts != expected_scene_counts[endpoint.role]:
                    raise IntegratedRuntimeError(
                        f"{endpoint.role} scene completion counts differ"
                    )
                finalize_started_ns = _strict_int(
                    value["finalize_started_ns"], "scene finalize start", minimum=1
                )
                finalize_sync_ns = _strict_int(
                    value["cuda_sync_finished_ns"],
                    "scene finalize synchronization",
                    minimum=finalize_started_ns,
                )
                finalize_total_ns = _strict_int(
                    value["total_ns"], "scene finalize total"
                )
                if finalize_total_ns != finalize_sync_ns - finalize_started_ns:
                    raise IntegratedRuntimeError("scene finalize timing differs")
                if not _strict_bool(
                    value["cuda_synchronized"], "scene CUDA synchronization"
                ):
                    raise IntegratedRuntimeError("scene completion was not synchronized")
                scene_device_used = _strict_int(
                    value["device_wide_used_at_sync_bytes"],
                    f"{endpoint.role} scene-final device-wide memory sample",
                    maximum=ready_records[endpoint.role]["gpu_total_memory_bytes"],
                )
                if not _strict_bool(
                    value["device_wide_memory_sampled_at_sync"],
                    f"{endpoint.role} scene-final memory sample flag",
                ):
                    raise IntegratedRuntimeError(
                        f"{endpoint.role} scene-final memory sample is missing"
                    )
                scene_end_records.append(
                    {
                        "role": endpoint.role,
                        "scene_id": scene["scene_id"],
                        "frames_processed": observed_counts["frames_processed"],
                        "provider_calls": observed_counts["provider_calls"],
                        "provider_abstentions": observed_counts[
                            "provider_abstentions"
                        ],
                        "raw_rows": observed_counts["raw_rows"],
                        "k8_rows": observed_counts["k8_rows"],
                        "tracker_commits": observed_counts["tracker_commits"],
                        "total_ns": finalize_total_ns,
                        "cuda_sync_finished_ns": finalize_sync_ns,
                        "device_wide_used_at_sync_bytes": scene_device_used,
                    }
                )
                if endpoint.role == "native":
                    if last_native_sync_ns is None or finalize_sync_ns < last_native_sync_ns:
                        raise IntegratedRuntimeError(
                            "native scene-final sync predates final frame"
                        )
                    last_native_sync_ns = finalize_sync_ns

        for endpoint in (provider, native):
            sequence += 1
            _send_command(
                endpoint,
                "STOP",
                sequence,
                {},
                timeout=frame_timeout_seconds,
            )
            payload, _ = _receive_response(
                endpoint,
                "STOPPED",
                sequence,
                timeout=WORKER_STOP_TIMEOUT_SECONDS,
            )
            value = _exact_mapping_keys(
                payload, _CLOSE_PAYLOAD_KEYS, f"{endpoint.role} STOPPED"
            )
            close_records[endpoint.role] = {
                "frames_processed": _strict_int(
                    value["frames_processed"], f"{endpoint.role} processed frames"
                ),
                "provider_calls": _strict_int(
                    value["provider_calls"], f"{endpoint.role} provider calls"
                ),
                "provider_abstentions": _strict_int(
                    value["provider_abstentions"],
                    f"{endpoint.role} provider abstentions",
                ),
                "raw_rows": _strict_int(
                    value["raw_rows"], f"{endpoint.role} raw rows"
                ),
                "k8_rows": _strict_int(
                    value["k8_rows"], f"{endpoint.role} K8 rows"
                ),
                "tracker_commits": _strict_int(
                    value["tracker_commits"], f"{endpoint.role} tracker commits"
                ),
                "model_load_count": _strict_int(
                    value["model_load_count"], f"{endpoint.role} model load count"
                ),
                "torch_allocator_max_memory_allocated_bytes": _strict_int(
                    value["torch_allocator_max_memory_allocated_bytes"],
                    f"{endpoint.role} Torch allocator allocated high-water mark",
                ),
                "torch_allocator_max_memory_reserved_bytes": _strict_int(
                    value["torch_allocator_max_memory_reserved_bytes"],
                    f"{endpoint.role} Torch allocator reserved high-water mark",
                ),
                "device_wide_used_at_sync_bytes": _strict_int(
                    value["device_wide_used_at_sync_bytes"],
                    f"{endpoint.role} device-wide used memory at STOP synchronization",
                    maximum=ready_records[endpoint.role]["gpu_total_memory_bytes"],
                ),
                "device_wide_memory_sampled_at_sync": _strict_bool(
                    value["device_wide_memory_sampled_at_sync"],
                    f"{endpoint.role} STOP synchronization sample flag",
                ),
                "rss_peak_bytes": _strict_int(
                    value["rss_peak_bytes"], f"{endpoint.role} RSS"
                ),
                "oom_failure_reported": _strict_bool(
                    value["oom_failure_reported"], f"{endpoint.role} OOM failure"
                ),
                "cap_violation": _strict_bool(
                    value["cap_violation"], f"{endpoint.role} cap violation"
                ),
            }
            if (
                close_records[endpoint.role]["model_load_count"] != 1
                or not close_records[endpoint.role][
                    "device_wide_memory_sampled_at_sync"
                ]
                or close_records[endpoint.role]["oom_failure_reported"]
                or close_records[endpoint.role]["cap_violation"]
            ):
                raise IntegratedRuntimeError(f"{endpoint.role} final state differs")
            if close_records[endpoint.role]["frames_processed"] != view["native_frame_count"]:
                raise IntegratedRuntimeError(
                    f"{endpoint.role} global processed-frame count differs"
                )
            expected_final_counts = (
                {
                    "provider_calls": view["provider_valid_call_count"],
                    "provider_abstentions": view["provider_abstention_count"],
                    "raw_rows": sum(row["raw_row_count"] for row in provider_frames),
                    "k8_rows": sum(row["k8_row_count"] for row in provider_frames),
                    "tracker_commits": view["provider_valid_call_count"],
                }
                if endpoint.role == "provider"
                else {
                    "provider_calls": 0,
                    "provider_abstentions": 0,
                    "raw_rows": 0,
                    "k8_rows": 0,
                    "tracker_commits": 0,
                }
            )
            if any(
                close_records[endpoint.role][name] != expected
                for name, expected in expected_final_counts.items()
            ):
                raise IntegratedRuntimeError(
                    f"{endpoint.role} final provider-work counts differ"
                )

        for endpoint in endpoints:
            endpoint.process.join(timeout=WORKER_STOP_TIMEOUT_SECONDS)
            if endpoint.process.is_alive() or endpoint.process.exitcode != 0:
                raise IntegratedRuntimeError(f"{endpoint.role} worker did not exit cleanly")
    except BaseException:
        _terminate_endpoints(endpoints)
        raise
    finally:
        # The successful path has already joined; close queue resources without
        # changing worker semantics.  The failure path is idempotently safe.
        for endpoint in endpoints:
            try:
                endpoint.command_queue.close()
                endpoint.response_queue.close()
            except BaseException:
                pass

    if (
        first_current_read_ns is None
        or last_native_sync_ns is None
        or len(native_frames) != view["native_frame_count"]
        or len(provider_frames) != view["native_frame_count"]
    ):
        raise IntegratedRuntimeError("integrated stream did not cover every frame")
    stream_ns = last_native_sync_ns - first_current_read_ns
    if stream_ns <= 0:
        raise IntegratedRuntimeError("integrated stream duration is nonpositive")

    provider_call_times = [
        row["total_ns"] for row in provider_frames if row["call_executed"]
    ]
    tracker_times = [
        row["tracker_ns"] for row in provider_frames if row["call_executed"]
    ]
    provider_summary = _percentile_summary_ns(provider_call_times)
    tracker_summary = _percentile_summary_ns(tracker_times)
    native_summary = _percentile_summary_ns([row["total_ns"] for row in native_frames])
    native_fps = view["native_frame_count"] * 1_000_000_000.0 / stream_ns
    native_keyframe_slots = sum(row["model_scheduled"] for row in native_frames)
    provider_deadline_met = (
        provider_summary["p50_seconds"] <= PROVIDER_DEADLINE_SECONDS
        and provider_summary["p95_seconds"] <= PROVIDER_DEADLINE_SECONDS
        and provider_summary["max_seconds"] <= PROVIDER_DEADLINE_SECONDS
    )
    tracker_deadline_met = (
        tracker_summary["p95_ns"] <= TRACKER_P95_LIMIT_NS
        and tracker_summary["max_ns"] <= TRACKER_MAX_LIMIT_NS
    )
    native_fps_met = native_fps >= NATIVE_MIN_FPS
    provider_call_count = sum(row["call_executed"] for row in provider_frames)
    provider_abstention_count = sum(
        row["provider_status"] == PROVIDER_ABSTAIN for row in provider_frames
    )
    provider_raw_rows = sum(row["raw_row_count"] for row in provider_frames)
    provider_k8_rows = sum(row["k8_row_count"] for row in provider_frames)
    provider_tracker_commits = sum(
        row["call_executed"] for row in provider_frames
    )
    if provider_call_count != view["provider_valid_call_count"]:
        raise IntegratedRuntimeError("integrated provider call count differs")
    if provider_abstention_count != view["provider_abstention_count"]:
        raise IntegratedRuntimeError("integrated provider abstention count differs")
    if any(not row["tracker_audit_complete"] for row in provider_frames):
        raise IntegratedRuntimeError("tracker audit/cap was not complete")
    if any(endpoint.max_outstanding > QUEUE_MAXSIZE for endpoint in endpoints):
        raise IntegratedRuntimeError("queue hard cap was exceeded")

    role_resources: dict[str, dict[str, int]] = {}
    for role, frames in (("native", native_frames), ("provider", provider_frames)):
        device_samples = [
            ready_records[role]["device_wide_used_at_sync_bytes"],
            close_records[role]["device_wide_used_at_sync_bytes"],
            *(
                row["device_wide_used_at_sync_bytes"]
                for row in scene_end_records
                if row["role"] == role
            ),
            *(
                row["device_wide_used_at_sync_bytes"]
                for row in frames
                if row["device_wide_memory_sampled_at_sync"]
            ),
        ]
        role_resources[role] = {
            "torch_allocator_max_memory_allocated_bytes": max(
                ready_records[role]["torch_allocator_max_memory_allocated_bytes"],
                close_records[role]["torch_allocator_max_memory_allocated_bytes"],
                *(row["torch_allocator_max_memory_allocated_bytes"] for row in frames),
            ),
            "torch_allocator_max_memory_reserved_bytes": max(
                ready_records[role]["torch_allocator_max_memory_reserved_bytes"],
                close_records[role]["torch_allocator_max_memory_reserved_bytes"],
                *(row["torch_allocator_max_memory_reserved_bytes"] for row in frames),
            ),
            "device_wide_used_at_sync_max_bytes": max(device_samples),
            "process_rss_peak_bytes": max(
                ready_records[role]["rss_peak_bytes"],
                close_records[role]["rss_peak_bytes"],
                *(row["rss_peak_bytes"] for row in frames),
            ),
        }
    torch_allocated_upper_sum = sum(
        row["torch_allocator_max_memory_allocated_bytes"]
        for row in role_resources.values()
    )
    torch_reserved_upper_sum = sum(
        row["torch_allocator_max_memory_reserved_bytes"]
        for row in role_resources.values()
    )
    device_wide_sample_max = max(
        row["device_wide_used_at_sync_max_bytes"] for row in role_resources.values()
    )
    if device_wide_sample_max > ready_records["native"]["gpu_total_memory_bytes"]:
        raise IntegratedRuntimeError("device-wide memory sample exceeds GPU capacity")
    rss_peak = sum(row["process_rss_peak_bytes"] for row in role_resources.values())
    return {
        "schema": SCHEMA,
        "mode": "no_gt_integrated_runtime_only",
        "arm": "integrated",
        "spawn_start_method": "spawn",
        "spawn_worker_count": 2,
        "persistent_model_processes": True,
        "model_load_count_per_worker": 1,
        "same_cuda_visible_devices": True,
        "same_gpu_uuid": True,
        "gpu_uuid": ready_records["native"]["gpu_uuid"],
        "opaque_t05_identity_hashing": True,
        "coordinator_native_prediction_semantic_access": False,
        "native_prediction_deserialization": False,
        "native_prediction_geometry_access": False,
        "native_prediction_serialized": False,
        "native_prediction_mutation": False,
        "native_prediction_write": False,
        "gt_access": False,
        "annotation_access": False,
        "evaluation": False,
        "ap_computation": False,
        "birth": False,
        "labels_serialized": False,
        "geometry_serialized": False,
        "coordinator_preflight_opaque_input_hashing": True,
        "online_worker_prefetch": False,
        "online_worker_future_frame_semantic_access": False,
        "queue_maxsize": QUEUE_MAXSIZE,
        "queue_max_observed": max(endpoint.max_outstanding for endpoint in endpoints),
        "backlog_events": 0,
        "native_frame_count": view["native_frame_count"],
        "provider_call_count": provider_call_count,
        "provider_abstention_count": provider_abstention_count,
        "provider_raw_row_count": provider_raw_rows,
        "provider_k8_row_count": provider_k8_rows,
        "provider_tracker_commit_count": provider_tracker_commits,
        "provider_outside_schedule_count": sum(
            row["provider_status"] == OUTSIDE_PROVIDER for row in provider_frames
        ),
        "full_stream_extension": True,
        "upstream_early_terminal_byte_equivalent": False,
        "all_native_frames_preprocessed": True,
        "native_gap25_scheduled_keyframe_slot_count": native_keyframe_slots,
        "stream_clock": {
            "definition": (
                "first_current_frame_read_to_last_native_end_scene_cuda_sync"
            ),
            "first_current_read_ns": first_current_read_ns,
            "last_native_cuda_sync_ns": last_native_sync_ns,
            "duration_ns": stream_ns,
            "duration_seconds": stream_ns / 1_000_000_000.0,
            "prestream_initialization_excluded": True,
            "component_constructor_warmup_disclosed_in_workers": True,
            "full_pipeline_warmup": False,
            "first_real_forward_included": True,
        },
        "native": {
            "frame_count": len(native_frames),
            "gap25_scheduled_keyframe_slot_count": native_keyframe_slots,
            "fps": native_fps,
            "frame_runtime": native_summary,
            "minimum_fps": NATIVE_MIN_FPS,
            "minimum_fps_met": native_fps_met,
        },
        "provider": {
            "raw_row_count": provider_raw_rows,
            "k8_row_count": provider_k8_rows,
            "tracker_commit_count": provider_tracker_commits,
            "call_runtime": provider_summary,
            "deadline_seconds": PROVIDER_DEADLINE_SECONDS,
            "p50_p95_max_deadline_met": provider_deadline_met,
            "tracker_runtime": tracker_summary,
            "tracker_p95_limit_ns": TRACKER_P95_LIMIT_NS,
            "tracker_max_limit_ns": TRACKER_MAX_LIMIT_NS,
            "tracker_deadline_met": tracker_deadline_met,
            "tracker_execution_device": "cpu",
            "tracker_gpu_execution": False,
            "tracker_gpu_bytes": 0,
        },
        "performance_gates": {
            "native_absolute_10fps_met": native_fps_met,
            "provider_deadline_met": provider_deadline_met,
            "tracker_deadline_met": tracker_deadline_met,
            "all_met": native_fps_met and provider_deadline_met and tracker_deadline_met,
        },
        "resources": {
            "per_role": role_resources,
            "torch_allocator_role_peak_upper_sum_allocated_bytes": (
                torch_allocated_upper_sum
            ),
            "torch_allocator_role_peak_upper_sum_reserved_bytes": (
                torch_reserved_upper_sum
            ),
            "device_wide_used_at_sync_max_bytes": device_wide_sample_max,
            "device_wide_sampling_scope": (
                "cuda_synchronization_boundaries_only_not_continuous_peak"
            ),
            "device_wide_samples_include_non_torch_allocations": True,
            "device_wide_samples_cover_both_same_gpu_workers": True,
            "continuous_device_memory_peak_measured": False,
            "numerical_vram_cap_preregistered": False,
            "same_gpu_models_simultaneously_resident_and_full_stream_completed": True,
            "process_rss_role_peak_upper_sum_bytes": rss_peak,
            "oom_failure_reported": False,
            "full_stream_completed_without_oom_failure": True,
            "cap_violation": False,
        },
        "terminal_output": {
            "model_lifecycle_fd1_fd2_redirected_to_devnull": True,
            "suppression_begins_at_spawn_target_before_factory": True,
            "spawn_bootstrap_stdio_suppression_not_claimed": True,
            "fd_redirection_scope": (
                "spawn_target_before_factory_through_worker_close"
            ),
            "stdio_content_retained": False,
            "stdio_character_counts_retained": False,
            "prediction_derived_text_reaches_coordinator_terminal": False,
        },
        "workers": ready_records,
        "causal_frame_ledger": causal_ledger,
        "frame_timing": {
            "native_total_ns": [row["total_ns"] for row in native_frames],
            "provider_total_ns": [row["total_ns"] for row in provider_frames],
            "tracker_ns": [row["tracker_ns"] for row in provider_frames],
            "provider_status": [row["provider_status"] for row in provider_frames],
            "scene_finalize": scene_end_records,
        },
    }


def _execute_control_runtime(
    *,
    manifest_view: Mapping[str, Any],
    scene_root: Path,
    native_factory: WorkerFactory,
    native_factory_config: Mapping[str, Any],
    cuda_visible_devices: str,
    ready_timeout_seconds: float = WORKER_READY_TIMEOUT_SECONDS,
    frame_timeout_seconds: float = FRAME_ACK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Native-only full-stream control arm; it creates no provider ACKs."""

    view = _minimal_manifest_view(manifest_view)
    context = mp.get_context("spawn")
    native = _spawn_runtime_endpoint(
        context,
        role="native",
        factory=native_factory,
        config=native_factory_config,
    )
    frames: list[dict[str, Any]] = []
    causal_ledger: list[dict[str, Any]] = []
    scene_finalize: list[dict[str, Any]] = []
    sequence = 0
    first_read_ns: int | None = None
    last_sync_ns: int | None = None
    ready_record: dict[str, Any] | None = None
    close_record: dict[str, Any] | None = None
    try:
        ready_payload, _ = _receive_response(
            native, "READY", 0, timeout=ready_timeout_seconds
        )
        ready_record = _validate_ready_payload(
            ready_payload,
            role="native",
            expected_cuda_visible_devices=cuda_visible_devices,
            expected_python_executable=native_factory_config.get("python_executable"),
            expected_runtime_identity=native_factory_config.get(
                "expected_runtime_identity"
            ),
        )
        for scene in view["scenes"]:
            sequence += 1
            _send_command(
                native,
                "START_SCENE",
                sequence,
                _scene_command_payload(scene, scene_root),
                timeout=frame_timeout_seconds,
            )
            payload, _ = _receive_response(
                native, "SCENE_READY", sequence, timeout=frame_timeout_seconds
            )
            scene_ready = _exact_mapping_keys(
                payload, _SCENE_PAYLOAD_KEYS, "native SCENE_READY"
            )
            if (
                scene_ready["scene_id"] != scene["scene_id"]
                or scene_ready["intrinsic_color_sha256_observed"]
                != scene["intrinsic_color_sha256"]
                or scene_ready["intrinsic_depth_sha256_observed"]
                != scene["intrinsic_depth_sha256"]
                or not _strict_bool(
                    scene_ready["held_role_descriptors"],
                    "native held role descriptors",
                )
            ):
                raise IntegratedRuntimeError("control scene READY identity differs")
            scene_count = 0
            for frame in scene["frames"]:
                sequence += 1
                sent_ns = _send_command(
                    native,
                    "FRAME",
                    sequence,
                    _frame_command_payload(frame),
                    timeout=frame_timeout_seconds,
                )
                payload, received_ns = _receive_response(
                    native, "FRAME_ACK", sequence, timeout=frame_timeout_seconds
                )
                current = _validate_frame_payload(
                    payload, role="native", frame=frame
                )
                if current["current_read_started_ns"] < sent_ns:
                    raise IntegratedRuntimeError("control native read predates request")
                if current["cuda_sync_finished_ns"] > received_ns:
                    raise IntegratedRuntimeError("control native sync predates ACK")
                if last_sync_ns is not None and current["current_read_started_ns"] < last_sync_ns:
                    raise IntegratedRuntimeError("control native future frame was prefetched")
                if first_read_ns is None:
                    first_read_ns = current["current_read_started_ns"]
                last_sync_ns = current["cuda_sync_finished_ns"]
                frames.append(current)
                scene_count += 1
                causal_ledger.append(
                    {
                        "scene_id": frame["scene_id"],
                        "frame_id": frame["frame_id"],
                        "global_frame_index": frame["global_frame_index"],
                        "native_request_ns": sent_ns,
                        "native_ack_ns": received_ns,
                    }
                )
            sequence += 1
            _send_command(
                native,
                "END_SCENE",
                sequence,
                {"scene_id": scene["scene_id"]},
                timeout=frame_timeout_seconds,
            )
            payload, _ = _receive_response(
                native, "SCENE_DONE", sequence, timeout=frame_timeout_seconds
            )
            done = _exact_mapping_keys(
                payload, _END_SCENE_PAYLOAD_KEYS, "native SCENE_DONE"
            )
            counts = {
                "frames_processed": _strict_int(
                    done["frames_processed"], "control scene frames"
                ),
                "provider_calls": _strict_int(
                    done["provider_calls"], "control provider calls"
                ),
                "provider_abstentions": _strict_int(
                    done["provider_abstentions"], "control provider abstentions"
                ),
                "raw_rows": _strict_int(done["raw_rows"], "control raw rows"),
                "k8_rows": _strict_int(done["k8_rows"], "control K8 rows"),
                "tracker_commits": _strict_int(
                    done["tracker_commits"], "control tracker commits"
                ),
            }
            if (
                done["scene_id"] != scene["scene_id"]
                or counts
                != {
                    "frames_processed": scene_count,
                    "provider_calls": 0,
                    "provider_abstentions": 0,
                    "raw_rows": 0,
                    "k8_rows": 0,
                    "tracker_commits": 0,
                }
            ):
                raise IntegratedRuntimeError("control scene completion counts differ")
            finalize_start = _strict_int(
                done["finalize_started_ns"], "control finalize start", minimum=1
            )
            finalize_sync = _strict_int(
                done["cuda_sync_finished_ns"],
                "control finalize sync",
                minimum=finalize_start,
            )
            finalize_total = _strict_int(
                done["total_ns"], "control finalize total"
            )
            if (
                finalize_total != finalize_sync - finalize_start
                or not _strict_bool(
                    done["cuda_synchronized"], "control finalize synchronized"
                )
                or last_sync_ns is None
                or finalize_sync < last_sync_ns
            ):
                raise IntegratedRuntimeError("control scene final synchronization differs")
            scene_device_used = _strict_int(
                done["device_wide_used_at_sync_bytes"],
                "control scene-final device-wide memory sample",
                maximum=ready_record["gpu_total_memory_bytes"],
            )
            if not _strict_bool(
                done["device_wide_memory_sampled_at_sync"],
                "control scene-final memory sample flag",
            ):
                raise IntegratedRuntimeError(
                    "control scene-final memory sample is missing"
                )
            last_sync_ns = finalize_sync
            scene_finalize.append(
                {
                    "scene_id": scene["scene_id"],
                    "total_ns": finalize_total,
                    "cuda_sync_finished_ns": finalize_sync,
                    "device_wide_used_at_sync_bytes": scene_device_used,
                }
            )

        sequence += 1
        _send_command(
            native, "STOP", sequence, {}, timeout=frame_timeout_seconds
        )
        payload, _ = _receive_response(
            native, "STOPPED", sequence, timeout=WORKER_STOP_TIMEOUT_SECONDS
        )
        stopped = _exact_mapping_keys(payload, _CLOSE_PAYLOAD_KEYS, "native STOPPED")
        close_record = {
            "frames_processed": _strict_int(
                stopped["frames_processed"], "control processed frames"
            ),
            "provider_calls": _strict_int(
                stopped["provider_calls"], "control provider calls"
            ),
            "provider_abstentions": _strict_int(
                stopped["provider_abstentions"], "control provider abstentions"
            ),
            "raw_rows": _strict_int(stopped["raw_rows"], "control raw rows"),
            "k8_rows": _strict_int(stopped["k8_rows"], "control K8 rows"),
            "tracker_commits": _strict_int(
                stopped["tracker_commits"], "control tracker commits"
            ),
            "model_load_count": _strict_int(
                stopped["model_load_count"], "control model loads"
            ),
            "torch_allocator_max_memory_allocated_bytes": _strict_int(
                stopped["torch_allocator_max_memory_allocated_bytes"],
                "control Torch allocator allocated high-water mark",
            ),
            "torch_allocator_max_memory_reserved_bytes": _strict_int(
                stopped["torch_allocator_max_memory_reserved_bytes"],
                "control Torch allocator reserved high-water mark",
            ),
            "device_wide_used_at_sync_bytes": _strict_int(
                stopped["device_wide_used_at_sync_bytes"],
                "control device-wide used memory at STOP synchronization",
                maximum=ready_record["gpu_total_memory_bytes"],
            ),
            "device_wide_memory_sampled_at_sync": _strict_bool(
                stopped["device_wide_memory_sampled_at_sync"],
                "control STOP synchronization sample flag",
            ),
            "rss_peak_bytes": _strict_int(stopped["rss_peak_bytes"], "control RSS"),
            "oom_failure_reported": _strict_bool(
                stopped["oom_failure_reported"], "control OOM failure"
            ),
            "cap_violation": _strict_bool(
                stopped["cap_violation"], "control cap"
            ),
        }
        if (
            close_record["frames_processed"] != view["native_frame_count"]
            or close_record["provider_calls"] != 0
            or close_record["provider_abstentions"] != 0
            or close_record["raw_rows"] != 0
            or close_record["k8_rows"] != 0
            or close_record["tracker_commits"] != 0
            or close_record["model_load_count"] != 1
            or not close_record["device_wide_memory_sampled_at_sync"]
            or close_record["oom_failure_reported"]
            or close_record["cap_violation"]
        ):
            raise IntegratedRuntimeError("control final worker state differs")
        native.process.join(timeout=WORKER_STOP_TIMEOUT_SECONDS)
        if native.process.is_alive() or native.process.exitcode != 0:
            raise IntegratedRuntimeError("control worker did not exit cleanly")
    except BaseException:
        _terminate_endpoints([native])
        raise
    finally:
        try:
            native.command_queue.close()
            native.response_queue.close()
        except BaseException:
            pass

    if (
        ready_record is None
        or close_record is None
        or first_read_ns is None
        or last_sync_ns is None
        or len(frames) != view["native_frame_count"]
    ):
        raise IntegratedRuntimeError("control stream is incomplete")
    duration_ns = last_sync_ns - first_read_ns
    if duration_ns <= 0:
        raise IntegratedRuntimeError("control stream duration is nonpositive")
    fps = len(frames) * 1_000_000_000.0 / duration_ns
    keyframe_slots = sum(row["model_scheduled"] for row in frames)
    frame_runtime = _percentile_summary_ns([row["total_ns"] for row in frames])
    torch_allocated = max(
        ready_record["torch_allocator_max_memory_allocated_bytes"],
        close_record["torch_allocator_max_memory_allocated_bytes"],
        *(row["torch_allocator_max_memory_allocated_bytes"] for row in frames),
    )
    torch_reserved = max(
        ready_record["torch_allocator_max_memory_reserved_bytes"],
        close_record["torch_allocator_max_memory_reserved_bytes"],
        *(row["torch_allocator_max_memory_reserved_bytes"] for row in frames),
    )
    rss = max(
        ready_record["rss_peak_bytes"],
        close_record["rss_peak_bytes"],
        *(row["rss_peak_bytes"] for row in frames),
    )
    device_wide_sample_max = max(
        ready_record["device_wide_used_at_sync_bytes"],
        close_record["device_wide_used_at_sync_bytes"],
        *(row["device_wide_used_at_sync_bytes"] for row in scene_finalize),
        *(
            row["device_wide_used_at_sync_bytes"]
            for row in frames
            if row["device_wide_memory_sampled_at_sync"]
        ),
    )
    if device_wide_sample_max > ready_record["gpu_total_memory_bytes"]:
        raise IntegratedRuntimeError("control device-wide sample exceeds GPU capacity")
    return {
        "schema": SCHEMA,
        "mode": "no_gt_control_runtime_only",
        "arm": "control",
        "spawn_start_method": "spawn",
        "spawn_worker_count": 1,
        "provider_process_present": False,
        "provider_ack_count": 0,
        "provider_call_count": 0,
        "provider_abstention_count": 0,
        "full_stream_extension": True,
        "upstream_early_terminal_byte_equivalent": False,
        "all_native_frames_preprocessed": True,
        "native_gap25_scheduled_keyframe_slot_count": keyframe_slots,
        "opaque_t05_identity_hashing": True,
        "coordinator_native_prediction_semantic_access": False,
        "native_prediction_deserialization": False,
        "native_prediction_geometry_access": False,
        "native_prediction_serialized": False,
        "native_prediction_mutation": False,
        "native_prediction_write": False,
        "gt_access": False,
        "annotation_access": False,
        "evaluation": False,
        "ap_computation": False,
        "birth": False,
        "labels_serialized": False,
        "geometry_serialized": False,
        "coordinator_preflight_opaque_input_hashing": True,
        "online_worker_prefetch": False,
        "online_worker_future_frame_semantic_access": False,
        "queue_maxsize": QUEUE_MAXSIZE,
        "queue_max_observed": native.max_outstanding,
        "backlog_events": 0,
        "native_frame_count": len(frames),
        "stream_clock": {
            "definition": "first_current_frame_read_to_last_native_end_scene_cuda_sync",
            "first_current_read_ns": first_read_ns,
            "last_native_cuda_sync_ns": last_sync_ns,
            "duration_ns": duration_ns,
            "duration_seconds": duration_ns / 1_000_000_000.0,
            "prestream_initialization_excluded": True,
            "component_constructor_warmup_disclosed_in_workers": True,
            "full_pipeline_warmup": False,
            "first_real_forward_included": True,
        },
        "native": {
            "frame_count": len(frames),
            "fps": fps,
            "frame_runtime": frame_runtime,
            "gap25_scheduled_keyframe_slot_count": keyframe_slots,
            "minimum_fps": NATIVE_MIN_FPS,
            "minimum_fps_met": fps >= NATIVE_MIN_FPS,
        },
        "performance_gates": {
            "control_absolute_10fps_met": fps >= NATIVE_MIN_FPS,
            "integrated_primary_gate_applicable": False,
        },
        "resources": {
            "per_role": {
                "native": {
                    "torch_allocator_max_memory_allocated_bytes": torch_allocated,
                    "torch_allocator_max_memory_reserved_bytes": torch_reserved,
                    "device_wide_used_at_sync_max_bytes": device_wide_sample_max,
                    "process_rss_peak_bytes": rss,
                }
            },
            "torch_allocator_role_peak_upper_sum_allocated_bytes": torch_allocated,
            "torch_allocator_role_peak_upper_sum_reserved_bytes": torch_reserved,
            "device_wide_used_at_sync_max_bytes": device_wide_sample_max,
            "device_wide_sampling_scope": (
                "cuda_synchronization_boundaries_only_not_continuous_peak"
            ),
            "device_wide_samples_include_non_torch_allocations": True,
            "device_wide_samples_cover_both_same_gpu_workers": False,
            "continuous_device_memory_peak_measured": False,
            "numerical_vram_cap_preregistered": False,
            "same_gpu_models_simultaneously_resident_and_full_stream_completed": False,
            "process_rss_role_peak_upper_sum_bytes": rss,
            "oom_failure_reported": False,
            "full_stream_completed_without_oom_failure": True,
            "cap_violation": False,
        },
        "terminal_output": {
            "model_lifecycle_fd1_fd2_redirected_to_devnull": True,
            "suppression_begins_at_spawn_target_before_factory": True,
            "spawn_bootstrap_stdio_suppression_not_claimed": True,
            "fd_redirection_scope": (
                "spawn_target_before_factory_through_worker_close"
            ),
            "stdio_content_retained": False,
            "stdio_character_counts_retained": False,
            "prediction_derived_text_reaches_coordinator_terminal": False,
        },
        "workers": {"native": ready_record},
        "causal_frame_ledger": causal_ledger,
        "frame_timing": {
            "native_total_ns": [row["total_ns"] for row in frames],
            "scene_finalize": scene_finalize,
        },
    }


class _SyntheticRuntimeWorker:
    """Pickle-safe fake used only by focused spawn/causality tests."""

    def __init__(self, role: str, config: Mapping[str, Any]):
        self.role = role
        self._config = dict(config)
        if self._config.get("apply_real_provider_sys_path_transform"):
            if role != "provider":
                raise IntegratedRuntimeError(
                    "real provider sys.path transform requested for native"
                )
            fresh = importlib.import_module(
                "tools.run_scannet_s3r_h10_fresh_boxer_provider"
            )
            fresh._import_external_module(PROVIDER_BOXER_ROOT, "utils")
        self._emit_child_text("constructor")
        self._closed = False
        self._scene_id: str | None = None
        self._frames_total = 0
        self._scene_frames = 0
        self._scene_calls = 0
        self._scene_abstentions = 0
        self._scene_raw_rows = 0
        self._scene_k8_rows = 0
        self._scene_tracker_commits = 0
        self._calls_total = 0
        self._abstentions_total = 0
        self._raw_rows_total = 0
        self._k8_rows_total = 0
        self._tracker_commits_total = 0
        self._model_load_count = 1

    def _emit_child_text(self, stage: str) -> None:
        if self._config.get("mutate_sys_path_stage") == stage:
            sys.path.insert(0, f"/forbidden-sys-path-mutation-{self.role}-{stage}")
        if self._config.get("mutate_pycache_prefix_stage") == stage:
            sys.pycache_prefix = (
                f"/forbidden-pycache-prefix-{self.role}-{stage}"
            )
        if self._config.get("mutate_pycache_environment_stage") == stage:
            os.environ["PYTHONPYCACHEPREFIX"] = (
                f"/forbidden-pycache-environment-{self.role}-{stage}"
            )
        if self._config.get("emit_child_text"):
            print(f"suppressed-{self.role}-{stage}-stdout")
            print(
                f"suppressed-{self.role}-{stage}-stderr",
                file=sys.stderr,
            )
        if self._config.get("emit_native_fd_bytes"):
            os.write(
                1,
                f"native-fd-secret-{self.role}-{stage}-stdout\n".encode("ascii"),
            )
            os.write(
                2,
                f"native-fd-secret-{self.role}-{stage}-stderr\n".encode("ascii"),
            )

    def _gpu_memory(self) -> tuple[int, int, int]:
        return (
            int(self._config.get("gpu_allocated_bytes", 1024)),
            int(self._config.get("gpu_reserved_bytes", 2048)),
            int(self._config.get("device_wide_used_at_sync_bytes", 4096)),
        )

    def ready(self) -> Mapping[str, Any]:
        self._emit_child_text("ready")
        allocated, reserved, device_used = self._gpu_memory()
        if self._config.get("probe_real_runtime_identity"):
            torch_module = importlib.import_module("torch")
            torch_version = str(torch_module.__version__)
            cuda_version = str(torch_module.version.cuda)
            torch_origin = os.path.realpath(str(torch_module.__file__))
        else:
            torch_version = str(self._config.get("torch_version", "synthetic"))
            cuda_version = str(self._config.get("cuda_version", "synthetic"))
            torch_origin = str(
                self._config.get("torch_origin", "synthetic-no-torch-import")
            )
        result = {
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_uuid": str(self._config.get("gpu_uuid", "GPU-SYNTHETIC-0")),
            "gpu_device_name": str(
                self._config.get("gpu_device_name", "synthetic-gpu")
            ),
            "gpu_total_memory_bytes": int(
                self._config.get("gpu_total_memory_bytes", 16 * 1024**3)
            ),
            "gpu_driver_version": str(
                self._config.get("gpu_driver_version", "synthetic-driver")
            ),
            "model_load_count": self._model_load_count,
            "initialization_complete": True,
            "torch_allocator_max_memory_allocated_bytes": allocated,
            "torch_allocator_max_memory_reserved_bytes": reserved,
            "device_wide_used_at_sync_bytes": device_used,
            "device_wide_memory_sampled_at_sync": True,
            "rss_peak_bytes": _peak_rss_bytes(),
            "tracker_execution_device": "cpu" if self.role == "provider" else None,
            "tracker_gpu_execution": False,
            "tracker_gpu_bytes": 0,
            "owl_constructor_dummy_warmup": self.role == "provider",
            "full_pipeline_warmup": False,
            "first_real_forward_included": True,
            "first_real_owl_call_included": self.role == "provider",
            "first_real_owl_kernels_pre_warmed": self.role == "provider",
            "first_real_boxer_forward_included": self.role == "provider",
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "numpy_version": str(np.__version__),
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "numpy_origin": os.path.realpath(str(np.__file__)),
            "torch_origin": torch_origin,
        }
        if self._config.get("bool_smuggle_stage") == "ready":
            result["model_load_count"] = True
        return result

    def start_scene(self, scene: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._closed or self._scene_id is not None:
            raise IntegratedRuntimeError("synthetic scene lifecycle differs")
        scene_id = scene.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise IntegratedRuntimeError("synthetic scene ID is invalid")
        self._scene_id = scene_id
        self._scene_frames = 0
        self._scene_calls = 0
        self._scene_abstentions = 0
        self._scene_raw_rows = 0
        self._scene_k8_rows = 0
        self._scene_tracker_commits = 0
        return {
            "scene_id": scene_id,
            "intrinsic_color_sha256_observed": scene["intrinsic_color_sha256"],
            "intrinsic_depth_sha256_observed": scene["intrinsic_depth_sha256"],
            "held_role_descriptors": True,
        }

    def process_frame(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        self._emit_child_text("frame")
        if self._closed or frame.get("scene_id") != self._scene_id:
            raise IntegratedRuntimeError("synthetic frame/scene identity differs")
        frame_id = int(frame["frame_id"])
        if self._config.get("fail_frame_id") == frame_id:
            raise RuntimeError(
                str(
                    self._config.get(
                        "failure_secret", "injected synthetic worker failure"
                    )
                )
            )
        started_ns = time.perf_counter_ns()
        delay = float(self._config.get("frame_delay_seconds", 0.0001))
        if delay > 0.0:
            time.sleep(delay)
        synchronized_ns = time.perf_counter_ns()
        self._frames_total += 1
        self._scene_frames += 1
        allocated, reserved, device_used = self._gpu_memory()
        common: dict[str, Any] = {
            "scene_id": self._scene_id,
            "frame_id": frame_id,
            "current_read_started_ns": started_ns,
            "cuda_sync_finished_ns": synchronized_ns,
            "total_ns": synchronized_ns - started_ns,
            "torch_allocator_max_memory_allocated_bytes": allocated,
            "torch_allocator_max_memory_reserved_bytes": reserved,
            "device_wide_used_at_sync_bytes": (
                device_used
                if self.role == "native"
                or frame["provider_status"] == PROVIDER_MEMBER
                else None
            ),
            "device_wide_memory_sampled_at_sync": (
                self.role == "native"
                or frame["provider_status"] == PROVIDER_MEMBER
            ),
            "rss_peak_bytes": _peak_rss_bytes(),
            "oom_failure_reported": bool(self._config.get("oom", False)),
            "cap_violation": bool(self._config.get("cap_violation", False)),
        }
        if self.role == "provider":
            status = frame["provider_status"]
            executed = status == PROVIDER_MEMBER
            raw_count = int(self._config.get("raw_row_count", 5)) if executed else 0
            k8_count = min(raw_count, 8) if executed else 0
            tracker_ns = int(self._config.get("tracker_ns", 1000)) if executed else 0
            self._scene_calls += int(executed)
            self._scene_abstentions += int(status == PROVIDER_ABSTAIN)
            self._scene_raw_rows += raw_count
            self._scene_k8_rows += k8_count
            self._scene_tracker_commits += int(executed)
            self._calls_total += int(executed)
            self._abstentions_total += int(status == PROVIDER_ABSTAIN)
            self._raw_rows_total += raw_count
            self._k8_rows_total += k8_count
            self._tracker_commits_total += int(executed)
            common.update(
                {
                    "provider_status": status,
                    "call_executed": executed,
                    "raw_row_count": raw_count,
                    "k8_row_count": k8_count,
                    "tracker_ns": tracker_ns,
                    "tracker_audit_complete": not bool(
                        self._config.get("tracker_audit_failure", False)
                    ),
                }
            )
        else:
            common.update(
                {
                    "processed": True,
                    "cuda_synchronized": True,
                    "model_scheduled": frame_id % 25 == 0,
                }
            )
        input_read = self.role == "native" or (
            self.role == "provider" and frame["provider_status"] == PROVIDER_MEMBER
        )
        if input_read:
            common.update(
                {
                    "input_read": True,
                    "input_identity_sha256": _expected_frame_input_identity(frame),
                    "color_sha256_observed": frame["color_sha256"],
                    "depth_sha256_observed": frame["depth_sha256"],
                    "pose_sha256_observed": frame["pose_sha256"],
                    "effective_pose_sha256_observed": frame[
                        "effective_pose_sha256"
                    ],
                }
            )
        else:
            common.update(
                {
                    "input_read": False,
                    "input_identity_sha256": None,
                    "color_sha256_observed": None,
                    "depth_sha256_observed": None,
                    "pose_sha256_observed": None,
                    "effective_pose_sha256_observed": None,
                }
            )
        if self._config.get("inject_forbidden_extra_key"):
            common["boxes"] = [[0.0] * 3]
        if self._config.get("bool_smuggle_stage") == "frame":
            common["frame_id"] = True
        return common

    def end_scene(self, scene_id: str) -> Mapping[str, Any]:
        self._emit_child_text("end-scene")
        if self._closed or scene_id != self._scene_id:
            raise IntegratedRuntimeError("synthetic END_SCENE differs")
        finalize_started_ns = time.perf_counter_ns()
        finalize_sync_ns = time.perf_counter_ns()
        result = {
            "scene_id": scene_id,
            "frames_processed": self._scene_frames,
            "provider_calls": self._scene_calls if self.role == "provider" else 0,
            "provider_abstentions": (
                self._scene_abstentions if self.role == "provider" else 0
            ),
            "raw_rows": self._scene_raw_rows if self.role == "provider" else 0,
            "k8_rows": self._scene_k8_rows if self.role == "provider" else 0,
            "tracker_commits": (
                self._scene_tracker_commits if self.role == "provider" else 0
            ),
            "finalize_started_ns": finalize_started_ns,
            "cuda_sync_finished_ns": finalize_sync_ns,
            "total_ns": finalize_sync_ns - finalize_started_ns,
            "cuda_synchronized": True,
            "device_wide_used_at_sync_bytes": self._gpu_memory()[2],
            "device_wide_memory_sampled_at_sync": True,
        }
        if self._config.get("bool_smuggle_stage") == "end":
            result["frames_processed"] = True
        self._scene_id = None
        return result

    def close(self) -> Mapping[str, Any]:
        self._emit_child_text("close")
        if self._closed:
            raise IntegratedRuntimeError("synthetic worker closed twice")
        if self._scene_id is not None:
            raise IntegratedRuntimeError("synthetic worker closed during scene")
        self._closed = True
        allocated, reserved, device_used = self._gpu_memory()
        result = {
            "frames_processed": self._frames_total,
            "provider_calls": self._calls_total if self.role == "provider" else 0,
            "provider_abstentions": (
                self._abstentions_total if self.role == "provider" else 0
            ),
            "raw_rows": self._raw_rows_total if self.role == "provider" else 0,
            "k8_rows": self._k8_rows_total if self.role == "provider" else 0,
            "tracker_commits": (
                self._tracker_commits_total if self.role == "provider" else 0
            ),
            "model_load_count": self._model_load_count,
            "torch_allocator_max_memory_allocated_bytes": allocated,
            "torch_allocator_max_memory_reserved_bytes": reserved,
            "device_wide_used_at_sync_bytes": device_used,
            "device_wide_memory_sampled_at_sync": True,
            "rss_peak_bytes": _peak_rss_bytes(),
            "oom_failure_reported": bool(self._config.get("oom", False)),
            "cap_violation": bool(self._config.get("cap_violation", False)),
        }
        if self._config.get("bool_smuggle_stage") == "stop":
            result["frames_processed"] = True
        return result


def _synthetic_native_factory(config: Mapping[str, Any]) -> RuntimeWorker:
    return _SyntheticRuntimeWorker("native", config)


def _synthetic_provider_factory(config: Mapping[str, Any]) -> RuntimeWorker:
    return _SyntheticRuntimeWorker("provider", config)


SYNTHETIC_FACTORIES = HarnessFactories(
    native=_synthetic_native_factory,
    provider=_synthetic_provider_factory,
)


def _publish_timing_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(dict(receipt))
    if len(payload) > MAX_TIMING_JSON_BYTES:
        raise IntegratedRuntimeError("timing-only receipt exceeds byte cap")
    try:
        native_manifest_builder._publish_create_only(path, payload)
    except BaseException as error:
        raise IntegratedRuntimeError("timing-only create-only publication failed") from error


def _run_injected_harness(
    *,
    native_manifest: Mapping[str, Any],
    scene_root: Path,
    output: Path,
    factories: HarnessFactories,
    native_factory_config: Mapping[str, Any],
    provider_factory_config: Mapping[str, Any],
    cuda_visible_devices: str,
    ready_timeout_seconds: float = 10.0,
    frame_timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Private test seam; no production CLI exposes factory injection."""

    result = _execute_integrated_runtime(
        manifest_view=native_manifest,
        scene_root=scene_root,
        factories=factories,
        native_factory_config=native_factory_config,
        provider_factory_config=provider_factory_config,
        cuda_visible_devices=cuda_visible_devices,
        ready_timeout_seconds=ready_timeout_seconds,
        frame_timeout_seconds=frame_timeout_seconds,
    )
    result["formal_h10"] = False
    result["synthetic_worker_injection"] = True
    _publish_timing_receipt(output, result)
    return result


def _run_injected_control(
    *,
    native_manifest: Mapping[str, Any],
    scene_root: Path,
    output: Path,
    native_factory: WorkerFactory,
    native_factory_config: Mapping[str, Any],
    cuda_visible_devices: str,
    ready_timeout_seconds: float = 10.0,
    frame_timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    result = _execute_control_runtime(
        manifest_view=native_manifest,
        scene_root=scene_root,
        native_factory=native_factory,
        native_factory_config=native_factory_config,
        cuda_visible_devices=cuda_visible_devices,
        ready_timeout_seconds=ready_timeout_seconds,
        frame_timeout_seconds=frame_timeout_seconds,
    )
    result["formal_h10"] = False
    result["synthetic_worker_injection"] = True
    _publish_timing_receipt(output, result)
    return result


def _strict_json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise IntegratedRuntimeError(f"{label} payload must be bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=native_manifest_builder._duplicate_guard,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegratedRuntimeError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise IntegratedRuntimeError(f"{label} root must be an object")
    return value


def _strict_json_object(path: Path, *, maximum: int, label: str) -> dict[str, Any]:
    return _strict_json_object_bytes(
        _read_regular_bytes(path, maximum=maximum, label=label), label=label
    )


def _probe_frozen_child_runtime(
    *,
    executable: Path,
    expected_spawn_entry_sys_path: Sequence[str],
    expected_post_factory_sys_path: Sequence[str],
    expected_identity: Mapping[str, str],
    boxer_root: Path | None = None,
) -> dict[str, Any]:
    """Probe frozen entry/post-factory paths and module origins without CUDA."""

    resolved_executable = executable.resolve(strict=True)
    script = (
        "import json,os,sys;"
        "sys.path[:]=[os.getcwd() if item=='' else item for item in sys.path];"
        "entry=list(sys.path);"
        "root=(sys.argv[1] if len(sys.argv)==2 else None);"
        "exec(\"from pathlib import Path\\n"
        "from tools import run_scannet_s3r_h10_fresh_boxer_provider as fresh\\n"
        "fresh._import_external_module(Path(root), 'utils')\" if root else \"\");"
        "post=list(sys.path);"
        "import numpy,torch;"
        "print(json.dumps({"
        "'spawn_entry_sys_path':entry,"
        "'post_factory_sys_path':post,"
        "'python_version':sys.version.split()[0],"
        "'numpy_version':numpy.__version__,"
        "'torch_version':torch.__version__,"
        "'cuda_version':torch.version.cuda,"
        "'numpy_origin':os.path.realpath(numpy.__file__),"
        "'torch_origin':os.path.realpath(torch.__file__),"
        "'python_pycache_prefix_environment':os.environ.get('PYTHONPYCACHEPREFIX'),"
        "'python_pycache_prefix':sys.pycache_prefix"
        "},sort_keys=True,separators=(',',':')))"
    )
    probe_environment = _minimal_python_probe_environment()
    try:
        arguments = [os.fspath(resolved_executable), "-c", script]
        if boxer_root is not None:
            arguments.append(os.fspath(boxer_root.resolve(strict=True)))
        completed = subprocess.run(
            arguments,
            cwd=REPOSITORY_ROOT,
            env=probe_environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise IntegratedRuntimeError("frozen child runtime probe failed") from error
    value = _exact_mapping_keys(
        _strict_json_object_bytes(
            bytes(completed.stdout), label="frozen child runtime probe"
        ),
        frozenset(
            {
                "spawn_entry_sys_path",
                "post_factory_sys_path",
                "python_version",
                "numpy_version",
                "torch_version",
                "cuda_version",
                "numpy_origin",
                "torch_origin",
                "python_pycache_prefix_environment",
                "python_pycache_prefix",
            }
        ),
        "frozen child runtime probe",
    )
    normalized_paths: dict[str, tuple[str, ...]] = {}
    for name in ("spawn_entry_sys_path", "post_factory_sys_path"):
        raw_sys_path = value[name]
        if (
            not isinstance(raw_sys_path, list)
            or not raw_sys_path
            or any(not isinstance(item, str) for item in raw_sys_path)
        ):
            raise IntegratedRuntimeError(f"frozen child {name} probe is invalid")
        normalized_paths[name] = tuple(
            os.fspath(REPOSITORY_ROOT) if item == "" else item
            for item in raw_sys_path
        )
    frozen_entry = tuple(expected_spawn_entry_sys_path)
    frozen_post = tuple(expected_post_factory_sys_path)
    if (
        any(
            not isinstance(item, str) or not item
            for item in (*frozen_entry, *frozen_post)
        )
        or normalized_paths["spawn_entry_sys_path"] != frozen_entry
        or normalized_paths["post_factory_sys_path"] != frozen_post
        or frozen_entry[0] != os.fspath(REPOSITORY_ROOT)
        or len(set(frozen_entry)) != len(frozen_entry)
        or len(set(frozen_post)) != len(frozen_post)
    ):
        raise IntegratedRuntimeError("frozen child entry/post-factory sys.path differs")
    observed_identity = {
        "python_version": value["python_version"],
        "numpy_version": value["numpy_version"],
        "torch_version": value["torch_version"],
        "cuda_version": value["cuda_version"],
        "numpy_origin": os.fspath(
            Path(str(value["numpy_origin"])).resolve(strict=True)
        ),
        "torch_origin": os.fspath(
            Path(str(value["torch_origin"])).resolve(strict=True)
        ),
        "python_pycache_prefix_environment": value[
            "python_pycache_prefix_environment"
        ],
        "python_pycache_prefix": value["python_pycache_prefix"],
        "spawn_entry_python_sys_path_sha256": _sys_path_sha256(frozen_entry),
        "post_factory_python_sys_path_sha256": _sys_path_sha256(frozen_post),
    }
    if observed_identity != dict(expected_identity):
        raise IntegratedRuntimeError("frozen child module identity differs")
    return {
        "python_executable": os.fspath(resolved_executable),
        "python_sys_path": list(frozen_entry),
        "expected_post_factory_python_sys_path": list(frozen_post),
        "runtime_identity": observed_identity,
    }


def _child_runtime_identity_sha256(
    probes: Mapping[str, Mapping[str, Any]],
) -> str:
    """Bind both executables and both entry/post module-search identities."""

    expected = {
        "native": (
            NATIVE_PYTHON_EXECUTABLE,
            NATIVE_PYTHON_SHA256,
            NATIVE_FROZEN_SYS_PATH,
            NATIVE_POST_FACTORY_SYS_PATH,
            NATIVE_RUNTIME_IDENTITY,
        ),
        "provider": (
            PROVIDER_PYTHON_EXECUTABLE,
            PROVIDER_PYTHON_SHA256,
            PROVIDER_FROZEN_SYS_PATH,
            PROVIDER_POST_FACTORY_SYS_PATH,
            PROVIDER_RUNTIME_IDENTITY,
        ),
    }
    if set(probes) != set(expected):
        raise IntegratedRuntimeError("child runtime probe roles differ")
    rows: list[dict[str, Any]] = []
    for role in ("native", "provider"):
        executable, executable_sha256, entry, post, identity = expected[role]
        probe = probes[role]
        if (
            probe.get("python_executable")
            != os.fspath(executable.resolve(strict=True))
            or tuple(probe.get("python_sys_path", ())) != tuple(entry)
            or tuple(probe.get("expected_post_factory_python_sys_path", ()))
            != tuple(post)
            or probe.get("runtime_identity") != dict(identity)
        ):
            raise IntegratedRuntimeError("child runtime probe identity differs")
        rows.append(
            {
                "role": role,
                "python_executable": probe["python_executable"],
                "python_executable_sha256": executable_sha256,
                **dict(identity),
            }
        )
    return _ledger_digest(rows)


def _parse_held_provider_schedule(held: _HeldPinnedRegularFile) -> Any:
    value = _strict_json_object_bytes(held.payload, label=held.label)
    try:
        parsed = parse_exact_schedule_bundle(value)
    except BaseException as error:
        raise IntegratedRuntimeError("held provider schedule validation failed") from error
    # Mapping validation canonicalizes JSON internally.  Replace only the
    # reported digest with the exact held source bytes that were actually
    # parsed; all structural fields are those returned by the frozen parser.
    return replace(parsed, sha256=held.sha256)


def _parse_held_native_manifest(
    held: _HeldPinnedRegularFile,
    *,
    provider_bundle: Any,
    scene_root: Path,
) -> dict[str, Any]:
    value = _strict_json_object_bytes(held.payload, label=held.label)
    try:
        validated = native_manifest_builder.validate_native_manifest(
            value,
            provider_bundle=provider_bundle,
            require_frozen_provider=True,
        )
        native_manifest_builder.verify_manifest_files(
            validated, scene_root=scene_root
        )
    except BaseException as error:
        raise IntegratedRuntimeError("held native manifest validation failed") from error
    return validated


def _paths_overlap(first: Path, second: Path) -> bool:
    left = Path(os.path.abspath(os.fspath(first))).resolve(strict=False)
    right = Path(os.path.abspath(os.fspath(second))).resolve(strict=False)
    return left == right or left in right.parents or right in left.parents


def _require_formal_paths(
    *, arm: str, output: Path, control_receipt: Path | None
) -> tuple[Path, Path | None]:
    if arm not in ("control", "integrated"):
        raise IntegratedRuntimeError("formal arm must be control or integrated")
    output_supplied = Path(output)
    if not output_supplied.is_absolute():
        raise IntegratedRuntimeError("formal output must be the frozen absolute path")
    output_absolute = Path(os.path.abspath(os.fspath(output_supplied)))
    expected_output = (
        FORMAL_CONTROL_OUTPUT if arm == "control" else FORMAL_INTEGRATED_OUTPUT
    )
    if output_absolute != Path(os.path.abspath(os.fspath(expected_output))):
        raise IntegratedRuntimeError(
            f"formal {arm} output path differs from frozen contract"
        )
    if arm == "control":
        if control_receipt is not None:
            raise IntegratedRuntimeError("control arm cannot consume a control receipt")
        return output_absolute, None
    if control_receipt is None:
        raise IntegratedRuntimeError("integrated arm requires the frozen control receipt")
    control_supplied = Path(control_receipt)
    if not control_supplied.is_absolute():
        raise IntegratedRuntimeError(
            "integrated control receipt must be the frozen absolute path"
        )
    control_absolute = Path(os.path.abspath(os.fspath(control_supplied)))
    if control_absolute != Path(os.path.abspath(os.fspath(FORMAL_CONTROL_OUTPUT))):
        raise IntegratedRuntimeError("integrated control receipt path differs from contract")
    return output_absolute, control_absolute


def _formal_output_preflight(
    output: Path,
    *,
    runtime_contract: Path,
    control_receipt: Path | None,
) -> Path:
    absolute = Path(os.path.abspath(os.fspath(output)))
    if absolute.name in ("", ".", ".."):
        raise IntegratedRuntimeError("formal output must name one JSON file")
    try:
        os.lstat(absolute)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise IntegratedRuntimeError("cannot inspect formal output") from error
    else:
        raise IntegratedRuntimeError("formal output already exists")
    protected = [
        DEFAULT_SCENE_ROOT,
        FORMAL_T05_ROOT,
        DEFAULT_NATIVE_MANIFEST,
        DEFAULT_PROVIDER_SCHEDULE,
        FORMAL_RAW_SOURCE_JSON,
        FORMAL_RAW_SOURCE_NPZ,
        runtime_contract,
        Path(__file__).resolve(),
        INTEGRATED_RUNNER_TEST_SOURCE,
        REPOSITORY_ROOT / "boxfusion",
        REPOSITORY_ROOT / "tools",
        REPOSITORY_ROOT / "tests",
        REPOSITORY_ROOT / "docs",
        REPOSITORY_ROOT / "models",
        REPOSITORY_ROOT / "data",
        REPOSITORY_ROOT / "config",
        Path("/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev"),
    ]
    if control_receipt is not None:
        protected.append(control_receipt)
    if any(_paths_overlap(absolute, path) for path in protected):
        raise IntegratedRuntimeError("formal output overlaps an immutable input")
    return absolute


def _validate_formal_self_and_contract(
    *,
    runtime_contract: Path,
    expected_runtime_contract_sha256: str,
    expected_runner_sha256: str,
    expected_runner_test_sha256: str,
) -> dict[str, str]:
    expected = {
        "runtime_contract": _require_sha256(
            expected_runtime_contract_sha256, "runtime contract expected hash"
        ),
        "runner": _require_sha256(expected_runner_sha256, "runner expected hash"),
        "runner_test": _require_sha256(
            expected_runner_test_sha256, "runner test expected hash"
        ),
    }
    observed = {
        "runtime_contract": _hash_file(
            runtime_contract,
            maximum=MAX_ASSET_BYTES,
            label="integrated runtime contract",
        ),
        "runner": _hash_file(
            Path(__file__).resolve(),
            maximum=MAX_ASSET_BYTES,
            label="integrated runtime runner",
        ),
        "runner_test": _hash_file(
            INTEGRATED_RUNNER_TEST_SOURCE,
            maximum=MAX_ASSET_BYTES,
            label="integrated runtime runner test",
        ),
    }
    if observed != expected:
        raise IntegratedRuntimeError("runtime contract/self/test byte pin differs")
    return observed


def _assert_formal_manifest_view(view: Mapping[str, Any], bundle: Any) -> None:
    nonfinite = sum(
        not frame["raw_pose_finite"]
        for scene in view["scenes"]
        for frame in scene["frames"]
    )
    outside = sum(
        frame["provider_status"] == OUTSIDE_PROVIDER
        for scene in view["scenes"]
        for frame in scene["frames"]
    )
    if (
        view["scene_order"] != list(bundle.scene_order)
        or view["scene_count"] != EXPECTED_SCENE_COUNT
        or view["native_frame_count"] != EXPECTED_NATIVE_FRAME_COUNT
        or nonfinite != EXPECTED_NATIVE_NONFINITE_POSE_COUNT
        or view["provider_valid_call_count"] != EXPECTED_PROVIDER_VALID_CALLS
        or view["provider_abstention_count"] != EXPECTED_PROVIDER_ABSTENTIONS
        or outside != EXPECTED_PROVIDER_OUTSIDE_COUNT
    ):
        raise IntegratedRuntimeError("formal full-stream count/identity closure differs")


def _formal_static_snapshot(
    *,
    runtime_contract: Path,
    self_pins: Mapping[str, str],
) -> dict[str, Any]:
    paths = _native_static_asset_paths()
    expected = dict(NATIVE_ASSET_EXPECTED_SHA256)
    paths.update(
        {
            "runtime_contract": runtime_contract,
            "integrated_runner": Path(__file__).resolve(),
            "integrated_runner_test": INTEGRATED_RUNNER_TEST_SOURCE,
            "native_manifest": DEFAULT_NATIVE_MANIFEST,
            "provider_schedule": DEFAULT_PROVIDER_SCHEDULE,
            "raw_source_json": FORMAL_RAW_SOURCE_JSON,
            "raw_source_npz": FORMAL_RAW_SOURCE_NPZ,
        }
    )
    expected.update(
        {
            "runtime_contract": self_pins["runtime_contract"],
            "integrated_runner": self_pins["runner"],
            "integrated_runner_test": self_pins["runner_test"],
            "native_manifest": EXPECTED_NATIVE_MANIFEST_SHA256,
            "provider_schedule": EXPECTED_PROVIDER_SCHEDULE_SHA256,
            "raw_source_json": EXPECTED_RAW_SOURCE_JSON_SHA256,
            "raw_source_npz": EXPECTED_RAW_SOURCE_NPZ_SHA256,
        }
    )
    return _snapshot_asset_paths(paths, expected)


_FORMAL_CONTROL_TOP_KEYS = frozenset(
    {
        "schema",
        "mode",
        "arm",
        "spawn_start_method",
        "spawn_worker_count",
        "provider_process_present",
        "provider_ack_count",
        "provider_call_count",
        "provider_abstention_count",
        "full_stream_extension",
        "upstream_early_terminal_byte_equivalent",
        "all_native_frames_preprocessed",
        "native_gap25_scheduled_keyframe_slot_count",
        "opaque_t05_identity_hashing",
        "coordinator_native_prediction_semantic_access",
        "native_prediction_deserialization",
        "native_prediction_geometry_access",
        "native_prediction_serialized",
        "native_prediction_mutation",
        "native_prediction_write",
        "gt_access",
        "annotation_access",
        "evaluation",
        "ap_computation",
        "birth",
        "labels_serialized",
        "geometry_serialized",
        "coordinator_preflight_opaque_input_hashing",
        "online_worker_prefetch",
        "online_worker_future_frame_semantic_access",
        "queue_maxsize",
        "queue_max_observed",
        "backlog_events",
        "native_frame_count",
        "stream_clock",
        "native",
        "performance_gates",
        "resources",
        "terminal_output",
        "workers",
        "causal_frame_ledger",
        "frame_timing",
        "formal_h10",
        "synthetic_worker_injection",
        "timing_only",
        "runtime_only",
        "original_terminal_exact",
        "full100_not_authorized",
        "h10_gt_oracle_authorized",
        "gt_access_authorized",
        "integrated_provider_runtime_qualified",
        "integrated_realtime_qualified",
        "native_fps_protocol_equivalent",
        "bindings",
        "immutable_before_after_verified",
        "environment",
        "parent_runtime_identity",
    }
)


def _validate_control_receipt(
    value: Mapping[str, Any],
    observed_sha256: str,
    *,
    expected_bindings: Mapping[str, str],
    expected_cuda_visible_devices: str,
    expected_manifest_view: Mapping[str, Any],
) -> dict[str, Any]:
    observed = _require_sha256(
        observed_sha256, "control receipt observed hash"
    )
    control = _exact_mapping_keys(
        value, _FORMAL_CONTROL_TOP_KEYS, "formal control receipt"
    )
    parent_runtime = _exact_mapping_keys(
        control["parent_runtime_identity"],
        frozenset(
            {
                "python_executable",
                "python_version",
                "numpy_version",
                "torch_imported",
                "python_pycache_prefix_environment",
                "python_pycache_prefix",
            }
        ),
        "control parent runtime identity",
    )
    parent_executable = parent_runtime["python_executable"]
    if (
        not isinstance(parent_executable, str)
        or not parent_executable
        or Path(parent_executable).resolve()
        != NATIVE_PYTHON_EXECUTABLE.resolve(strict=True)
        or parent_runtime["python_version"]
        != NATIVE_RUNTIME_IDENTITY["python_version"]
        or parent_runtime["numpy_version"]
        != NATIVE_RUNTIME_IDENTITY["numpy_version"]
        or _strict_bool(
            parent_runtime["torch_imported"], "control parent torch-import flag"
        )
        or parent_runtime["python_pycache_prefix_environment"]
        != FROZEN_PYCACHE_PREFIX
        or parent_runtime["python_pycache_prefix"] != FROZEN_PYCACHE_PREFIX
    ):
        raise IntegratedRuntimeError("control parent runtime identity differs")
    bindings = _exact_mapping_keys(
        control["bindings"], frozenset(expected_bindings), "control bindings"
    )
    if dict(bindings) != dict(expected_bindings):
        raise IntegratedRuntimeError("control receipt binding values differ")
    workers = _exact_mapping_keys(
        control["workers"], frozenset({"native"}), "control workers"
    )
    native_worker = _validate_ready_payload(
        workers["native"],
        role="native",
        expected_cuda_visible_devices=expected_cuda_visible_devices,
        expected_python_executable=os.fspath(
            NATIVE_PYTHON_EXECUTABLE.resolve(strict=True)
        ),
        expected_runtime_identity=NATIVE_RUNTIME_IDENTITY,
    )
    false_fields = (
        "provider_process_present",
        "coordinator_native_prediction_semantic_access",
        "native_prediction_deserialization",
        "native_prediction_geometry_access",
        "native_prediction_serialized",
        "native_prediction_mutation",
        "native_prediction_write",
        "gt_access",
        "annotation_access",
        "evaluation",
        "ap_computation",
        "birth",
        "labels_serialized",
        "geometry_serialized",
        "online_worker_prefetch",
        "online_worker_future_frame_semantic_access",
        "upstream_early_terminal_byte_equivalent",
        "synthetic_worker_injection",
        "original_terminal_exact",
        "h10_gt_oracle_authorized",
        "gt_access_authorized",
        "integrated_provider_runtime_qualified",
        "integrated_realtime_qualified",
        "native_fps_protocol_equivalent",
    )
    true_fields = (
        "full_stream_extension",
        "all_native_frames_preprocessed",
        "opaque_t05_identity_hashing",
        "coordinator_preflight_opaque_input_hashing",
        "formal_h10",
        "timing_only",
        "runtime_only",
        "full100_not_authorized",
        "immutable_before_after_verified",
    )
    if any(_strict_bool(control[name], f"control {name}") for name in false_fields):
        raise IntegratedRuntimeError("control receipt false stopping field differs")
    if any(
        not _strict_bool(control[name], f"control {name}") for name in true_fields
    ):
        raise IntegratedRuntimeError("control receipt true stopping field differs")
    if (
        control["schema"] != SCHEMA
        or control["mode"] != "no_gt_control_runtime_only"
        or control["arm"] != "control"
        or control["spawn_start_method"] != "spawn"
        or _strict_int(control["spawn_worker_count"], "control spawn count") != 1
        or _strict_int(control["provider_ack_count"], "control provider ACKs") != 0
        or _strict_int(control["provider_call_count"], "control provider calls") != 0
        or _strict_int(
            control["provider_abstention_count"], "control provider abstentions"
        )
        != 0
        or _strict_int(control["queue_maxsize"], "control queue cap")
        != QUEUE_MAXSIZE
        or _strict_int(control["queue_max_observed"], "control queue observed")
        != QUEUE_MAXSIZE
        or _strict_int(control["backlog_events"], "control backlog") != 0
        or _strict_int(control["native_frame_count"], "control native frames")
        != EXPECTED_NATIVE_FRAME_COUNT
        or _strict_int(
            control["native_gap25_scheduled_keyframe_slot_count"],
            "control native scheduled keyframe slots",
        )
        != EXPECTED_NATIVE_SCHEDULED_KEYFRAME_SLOTS
    ):
        raise IntegratedRuntimeError("control receipt protocol differs")

    stream = _exact_mapping_keys(
        control["stream_clock"],
        frozenset(
            {
                "definition",
                "first_current_read_ns",
                "last_native_cuda_sync_ns",
                "duration_ns",
                "duration_seconds",
                "prestream_initialization_excluded",
                "component_constructor_warmup_disclosed_in_workers",
                "full_pipeline_warmup",
                "first_real_forward_included",
            }
        ),
        "control stream clock",
    )
    first_read = _strict_int(
        stream["first_current_read_ns"], "control first current read", minimum=1
    )
    last_sync = _strict_int(
        stream["last_native_cuda_sync_ns"],
        "control last CUDA sync",
        minimum=first_read,
    )
    duration = _strict_int(stream["duration_ns"], "control duration", minimum=1)
    duration_seconds = _strict_nonnegative_number(
        stream["duration_seconds"], "control duration seconds"
    )
    if (
        stream["definition"]
        != "first_current_frame_read_to_last_native_end_scene_cuda_sync"
        or duration != last_sync - first_read
        or not math.isclose(duration_seconds, duration / 1_000_000_000.0)
        or not _strict_bool(
            stream["prestream_initialization_excluded"],
            "control prestream initialization exclusion",
        )
        or not _strict_bool(
            stream["component_constructor_warmup_disclosed_in_workers"],
            "control component warm-up disclosure",
        )
        or _strict_bool(stream["full_pipeline_warmup"], "control full warm-up")
        or not _strict_bool(
            stream["first_real_forward_included"],
            "control first real forward inclusion",
        )
    ):
        raise IntegratedRuntimeError("control stream clock differs")

    native = _exact_mapping_keys(
        control["native"],
        frozenset(
            {
                "frame_count",
                "fps",
                "frame_runtime",
                "gap25_scheduled_keyframe_slot_count",
                "minimum_fps",
                "minimum_fps_met",
            }
        ),
        "control native timing",
    )
    fps = _strict_nonnegative_number(native["fps"], "control FPS")
    if (
        fps <= 0.0
        or not math.isclose(
            fps, EXPECTED_NATIVE_FRAME_COUNT * 1_000_000_000.0 / duration
        )
        or _strict_int(native["frame_count"], "control native timing frames")
        != EXPECTED_NATIVE_FRAME_COUNT
        or _strict_int(
            native["gap25_scheduled_keyframe_slot_count"],
            "control native timing scheduled keyframe slots",
        )
        != EXPECTED_NATIVE_SCHEDULED_KEYFRAME_SLOTS
        or _strict_nonnegative_number(native["minimum_fps"], "control minimum FPS")
        != NATIVE_MIN_FPS
        or _strict_bool(native["minimum_fps_met"], "control native FPS result")
        != (fps >= NATIVE_MIN_FPS)
    ):
        raise IntegratedRuntimeError("control native timing differs")
    frame_runtime = _exact_mapping_keys(
        native["frame_runtime"],
        frozenset(
            {
                "count",
                "p50_ns",
                "p95_ns",
                "max_ns",
                "p50_seconds",
                "p95_seconds",
                "max_seconds",
            }
        ),
        "control frame runtime",
    )
    if _strict_int(frame_runtime["count"], "control runtime sample count") != (
        EXPECTED_NATIVE_FRAME_COUNT
    ):
        raise IntegratedRuntimeError("control runtime sample count differs")
    for name in ("p50_ns", "p95_ns", "max_ns"):
        _strict_int(frame_runtime[name], f"control {name}")
    for name in ("p50_seconds", "p95_seconds", "max_seconds"):
        _strict_nonnegative_number(frame_runtime[name], f"control {name}")

    gates = _exact_mapping_keys(
        control["performance_gates"],
        frozenset(
            {"control_absolute_10fps_met", "integrated_primary_gate_applicable"}
        ),
        "control performance gates",
    )
    if (
        _strict_bool(
            gates["control_absolute_10fps_met"], "control absolute FPS gate"
        )
        != (fps >= NATIVE_MIN_FPS)
        or _strict_bool(
            gates["integrated_primary_gate_applicable"],
            "control integrated primary applicability",
        )
    ):
        raise IntegratedRuntimeError("control performance gates differ")

    resources = _exact_mapping_keys(
        control["resources"],
        frozenset(
            {
                "per_role",
                "torch_allocator_role_peak_upper_sum_allocated_bytes",
                "torch_allocator_role_peak_upper_sum_reserved_bytes",
                "device_wide_used_at_sync_max_bytes",
                "device_wide_sampling_scope",
                "device_wide_samples_include_non_torch_allocations",
                "device_wide_samples_cover_both_same_gpu_workers",
                "continuous_device_memory_peak_measured",
                "numerical_vram_cap_preregistered",
                "same_gpu_models_simultaneously_resident_and_full_stream_completed",
                "process_rss_role_peak_upper_sum_bytes",
                "oom_failure_reported",
                "full_stream_completed_without_oom_failure",
                "cap_violation",
            }
        ),
        "control resources",
    )
    per_role = _exact_mapping_keys(
        resources["per_role"], frozenset({"native"}), "control role resources"
    )
    native_resources = _exact_mapping_keys(
        per_role["native"],
        frozenset(
            {
                "torch_allocator_max_memory_allocated_bytes",
                "torch_allocator_max_memory_reserved_bytes",
                "device_wide_used_at_sync_max_bytes",
                "process_rss_peak_bytes",
            }
        ),
        "control native resources",
    )
    resource_pairs = (
        (
            "torch_allocator_max_memory_allocated_bytes",
            "torch_allocator_role_peak_upper_sum_allocated_bytes",
        ),
        (
            "torch_allocator_max_memory_reserved_bytes",
            "torch_allocator_role_peak_upper_sum_reserved_bytes",
        ),
        ("process_rss_peak_bytes", "process_rss_role_peak_upper_sum_bytes"),
    )
    for role_name, combined_name in resource_pairs:
        if _strict_int(
            native_resources[role_name], f"control native {role_name}"
        ) != _strict_int(resources[combined_name], f"control {combined_name}"):
            raise IntegratedRuntimeError("control combined resource accounting differs")
    native_device_sample_max = _strict_int(
        native_resources["device_wide_used_at_sync_max_bytes"],
        "control native sampled device-wide used memory maximum",
        maximum=native_worker["gpu_total_memory_bytes"],
    )
    if native_device_sample_max != _strict_int(
        resources["device_wide_used_at_sync_max_bytes"],
        "control sampled device-wide used memory maximum",
        maximum=native_worker["gpu_total_memory_bytes"],
    ):
        raise IntegratedRuntimeError("control device-wide resource accounting differs")
    if (
        resources["device_wide_sampling_scope"]
        != "cuda_synchronization_boundaries_only_not_continuous_peak"
        or not _strict_bool(
            resources["device_wide_samples_include_non_torch_allocations"],
            "control non-Torch device-wide sampling disclosure",
        )
        or _strict_bool(
            resources["device_wide_samples_cover_both_same_gpu_workers"],
            "control two-worker device-wide sampling disclosure",
        )
        or _strict_bool(
            resources["continuous_device_memory_peak_measured"],
            "control continuous device-memory measurement disclosure",
        )
        or _strict_bool(
            resources["numerical_vram_cap_preregistered"],
            "control numerical VRAM cap disclosure",
        )
        or _strict_bool(
            resources[
                "same_gpu_models_simultaneously_resident_and_full_stream_completed"
            ],
            "control simultaneous model-residency disclosure",
        )
        or _strict_bool(resources["oom_failure_reported"], "control OOM failure")
        or not _strict_bool(
            resources["full_stream_completed_without_oom_failure"],
            "control full-stream OOM-failure completion",
        )
        or _strict_bool(resources["cap_violation"], "control cap violation")
    ):
        raise IntegratedRuntimeError("control resource failure differs")

    terminal = _exact_mapping_keys(
        control["terminal_output"],
        frozenset(
            {
                "model_lifecycle_fd1_fd2_redirected_to_devnull",
                "suppression_begins_at_spawn_target_before_factory",
                "spawn_bootstrap_stdio_suppression_not_claimed",
                "fd_redirection_scope",
                "stdio_content_retained",
                "stdio_character_counts_retained",
                "prediction_derived_text_reaches_coordinator_terminal",
            }
        ),
        "control terminal output",
    )
    if (
        not _strict_bool(
            terminal["model_lifecycle_fd1_fd2_redirected_to_devnull"],
            "control model-lifecycle descriptor redirection",
        )
        or not _strict_bool(
            terminal["suppression_begins_at_spawn_target_before_factory"],
            "control spawn-target redirection start",
        )
        or not _strict_bool(
            terminal["spawn_bootstrap_stdio_suppression_not_claimed"],
            "control spawn-bootstrap suppression limitation",
        )
        or terminal["fd_redirection_scope"]
        != "spawn_target_before_factory_through_worker_close"
        or _strict_bool(
            terminal["stdio_content_retained"],
            "control stdio content retention",
        )
        or _strict_bool(
            terminal["stdio_character_counts_retained"],
            "control stdio character-count retention",
        )
        or _strict_bool(
            terminal["prediction_derived_text_reaches_coordinator_terminal"],
            "control prediction-derived terminal output",
        )
    ):
        raise IntegratedRuntimeError("control terminal-output safety differs")

    causal = control["causal_frame_ledger"]
    timing = _exact_mapping_keys(
        control["frame_timing"],
        frozenset({"native_total_ns", "scene_finalize"}),
        "control frame timing",
    )
    totals = timing["native_total_ns"]
    finalizes = timing["scene_finalize"]
    if (
        not isinstance(causal, list)
        or not isinstance(totals, list)
        or not isinstance(finalizes, list)
        or len(causal) != EXPECTED_NATIVE_FRAME_COUNT
        or len(totals) != EXPECTED_NATIVE_FRAME_COUNT
        or len(finalizes) != EXPECTED_SCENE_COUNT
    ):
        raise IntegratedRuntimeError("control causal/timing ledger counts differ")
    previous_ack = 0
    validated_totals: list[int] = []
    expected_frames = [
        frame
        for scene in expected_manifest_view["scenes"]
        for frame in scene["frames"]
    ]
    if len(expected_frames) != len(causal):
        raise IntegratedRuntimeError("control manifest/causal count differs")
    for index, (row, total_ns) in enumerate(zip(causal, totals)):
        item = _exact_mapping_keys(
            row,
            frozenset(
                {
                    "scene_id",
                    "frame_id",
                    "global_frame_index",
                    "native_request_ns",
                    "native_ack_ns",
                }
            ),
            f"control causal row {index}",
        )
        if not isinstance(item["scene_id"], str) or not item["scene_id"]:
            raise IntegratedRuntimeError("control causal scene ID is invalid")
        frame_id = _strict_int(item["frame_id"], f"control causal frame {index}")
        if (
            item["scene_id"] != expected_frames[index]["scene_id"]
            or frame_id != expected_frames[index]["frame_id"]
        ):
            raise IntegratedRuntimeError("control causal manifest identity differs")
        if _strict_int(
            item["global_frame_index"], f"control global frame {index}"
        ) != index:
            raise IntegratedRuntimeError("control causal global order differs")
        request_ns = _strict_int(
            item["native_request_ns"], f"control request {index}", minimum=1
        )
        ack_ns = _strict_int(
            item["native_ack_ns"], f"control ACK {index}", minimum=request_ns
        )
        if request_ns < previous_ack:
            raise IntegratedRuntimeError("control causal ledger overlaps frames")
        previous_ack = ack_ns
        validated_totals.append(
            _strict_int(total_ns, f"control frame total {index}")
        )
    recomputed_runtime = _percentile_summary_ns(validated_totals)
    for name in ("count", "p50_ns", "p95_ns", "max_ns"):
        if _strict_int(frame_runtime[name], f"control recomputed {name}") != (
            recomputed_runtime[name]
        ):
            raise IntegratedRuntimeError("control frame runtime summary differs")
    for name in ("p50_seconds", "p95_seconds", "max_seconds"):
        if not math.isclose(
            _strict_nonnegative_number(frame_runtime[name], f"control recomputed {name}"),
            recomputed_runtime[name],
        ):
            raise IntegratedRuntimeError("control frame runtime summary differs")
    expected_scene_order = tuple(
        scene["scene_id"] for scene in expected_manifest_view["scenes"]
    )
    for index, row in enumerate(finalizes):
        item = _exact_mapping_keys(
            row,
            frozenset(
                {
                    "scene_id",
                    "total_ns",
                    "cuda_sync_finished_ns",
                    "device_wide_used_at_sync_bytes",
                }
            ),
            f"control scene finalize {index}",
        )
        if item["scene_id"] != expected_scene_order[index]:
            raise IntegratedRuntimeError("control finalize scene order differs")
        _strict_int(item["total_ns"], "control scene finalize total")
        _strict_int(
            item["cuda_sync_finished_ns"], "control scene finalize CUDA sync", minimum=1
        )
        _strict_int(
            item["device_wide_used_at_sync_bytes"],
            "control scene-final device-wide memory sample",
            maximum=native_worker["gpu_total_memory_bytes"],
        )

    environment = _exact_mapping_keys(
        control["environment"],
        frozenset(
            (
                *REQUIRED_ENVIRONMENT.keys(),
                "CUDA_VISIBLE_DEVICES",
                "git_environment_names_absent",
            )
        ),
        "control environment",
    )
    if dict(environment) != {
        **dict(REQUIRED_ENVIRONMENT),
        "CUDA_VISIBLE_DEVICES": expected_cuda_visible_devices,
        "git_environment_names_absent": True,
    }:
        raise IntegratedRuntimeError("control environment differs")
    worker_identity_names = (
        "gpu_uuid",
        "gpu_device_name",
        "gpu_total_memory_bytes",
        "gpu_driver_version",
        "cuda_visible_devices",
        "python_executable",
        "python_version",
        "numpy_version",
        "torch_version",
        "cuda_version",
        "numpy_origin",
        "torch_origin",
        "python_pycache_prefix_environment",
        "python_pycache_prefix",
        "spawn_entry_python_sys_path_sha256",
        "post_factory_python_sys_path_sha256",
    )
    worker_identity = {name: native_worker[name] for name in worker_identity_names}
    if (
        not isinstance(worker_identity["gpu_device_name"], str)
        or not worker_identity["gpu_device_name"]
        or not isinstance(worker_identity["gpu_driver_version"], str)
        or not worker_identity["gpu_driver_version"]
        or _strict_int(
            worker_identity["gpu_total_memory_bytes"],
            "control GPU total memory",
            minimum=1,
        )
        != worker_identity["gpu_total_memory_bytes"]
        or any(
            not isinstance(worker_identity[name], str) or not worker_identity[name]
            for name in (
                "python_executable",
                "python_version",
                "numpy_version",
                "torch_version",
                "cuda_version",
            )
        )
    ):
        raise IntegratedRuntimeError("control receipt worker identity is invalid")
    return {
        "sha256": observed,
        "fps": fps,
        "worker_identity": worker_identity,
        "bindings": {name: bindings[name] for name in expected_bindings},
    }


def _assert_formal_runtime_result(result: Mapping[str, Any], *, arm: str) -> None:
    if (
        result.get("native_frame_count") != EXPECTED_NATIVE_FRAME_COUNT
        or result.get("native_gap25_scheduled_keyframe_slot_count")
        != EXPECTED_NATIVE_SCHEDULED_KEYFRAME_SLOTS
        or result.get("full_stream_extension") is not True
        or result.get("upstream_early_terminal_byte_equivalent") is not False
        or result.get("labels_serialized") is not False
        or result.get("geometry_serialized") is not False
        or result.get("opaque_t05_identity_hashing") is not True
        or result.get("coordinator_preflight_opaque_input_hashing") is not True
        or result.get("coordinator_native_prediction_semantic_access") is not False
        or result.get("native_prediction_deserialization") is not False
        or result.get("native_prediction_geometry_access") is not False
        or result.get("native_prediction_serialized") is not False
        or result.get("native_prediction_mutation") is not False
        or result.get("native_prediction_write") is not False
        or result.get("online_worker_prefetch") is not False
        or result.get("online_worker_future_frame_semantic_access") is not False
        or result.get("gt_access") is not False
        or result.get("annotation_access") is not False
        or result.get("evaluation") is not False
        or result.get("ap_computation") is not False
        or result.get("birth") is not False
    ):
        raise IntegratedRuntimeError("formal native runtime closure differs")
    resources = result.get("resources")
    if not isinstance(resources, Mapping):
        raise IntegratedRuntimeError("formal resource receipt is absent")
    expected_two_worker_disclosure = arm == "integrated"
    device_sample_max = _strict_int(
        resources.get("device_wide_used_at_sync_max_bytes"),
        "formal sampled device-wide used memory maximum",
    )
    workers = result.get("workers")
    if not isinstance(workers, Mapping) or "native" not in workers:
        raise IntegratedRuntimeError("formal native worker receipt is absent")
    native_worker = workers["native"]
    if not isinstance(native_worker, Mapping) or device_sample_max > _strict_int(
        native_worker.get("gpu_total_memory_bytes"), "formal GPU total memory", minimum=1
    ):
        raise IntegratedRuntimeError("formal device-wide memory sample is invalid")
    if (
        resources.get("device_wide_sampling_scope")
        != "cuda_synchronization_boundaries_only_not_continuous_peak"
        or not _strict_bool(
            resources.get("device_wide_samples_include_non_torch_allocations"),
            "formal non-Torch device-memory sampling disclosure",
        )
        or _strict_bool(
            resources.get("device_wide_samples_cover_both_same_gpu_workers"),
            "formal two-worker device-memory sampling disclosure",
        )
        != expected_two_worker_disclosure
        or _strict_bool(
            resources.get("continuous_device_memory_peak_measured"),
            "formal continuous device-memory measurement disclosure",
        )
        or _strict_bool(
            resources.get("numerical_vram_cap_preregistered"),
            "formal numerical VRAM cap disclosure",
        )
        or _strict_bool(
            resources.get(
                "same_gpu_models_simultaneously_resident_and_full_stream_completed"
            ),
            "formal same-GPU simultaneous residency disclosure",
        )
        != expected_two_worker_disclosure
        or _strict_bool(
            resources.get("oom_failure_reported"), "formal OOM failure report"
        )
        or not _strict_bool(
            resources.get("full_stream_completed_without_oom_failure"),
            "formal full-stream OOM-failure completion",
        )
        or _strict_bool(resources.get("cap_violation"), "formal cap violation")
    ):
        raise IntegratedRuntimeError("formal resource disclosure differs")
    if arm == "control":
        if (
            result.get("spawn_worker_count") != 1
            or result.get("provider_process_present") is not False
            or result.get("provider_ack_count") != 0
        ):
            raise IntegratedRuntimeError("formal control arm closure differs")
        return
    if (
        result.get("spawn_worker_count") != 2
        or result.get("provider_call_count") != EXPECTED_PROVIDER_VALID_CALLS
        or result.get("provider_abstention_count") != EXPECTED_PROVIDER_ABSTENTIONS
        or result.get("provider_outside_schedule_count")
        != EXPECTED_PROVIDER_OUTSIDE_COUNT
        or result.get("provider_raw_row_count") != EXPECTED_PROVIDER_RAW_ROWS
        or result.get("provider_k8_row_count") != EXPECTED_PROVIDER_K8_ROWS
        or result.get("provider_tracker_commit_count")
        != EXPECTED_PROVIDER_TRACKER_COMMITS
    ):
        raise IntegratedRuntimeError("formal integrated provider closure differs")
    scene_rows = {
        row["scene_id"]: (row["raw_rows"], row["k8_rows"], row["tracker_commits"])
        for row in result["frame_timing"]["scene_finalize"]
        if row["role"] == "provider"
    }
    expected_scenes = {
        scene_id: (
            EXPECTED_PROVIDER_RAW_ROWS_PER_SCENE[scene_id],
            EXPECTED_PROVIDER_K8_ROWS_PER_SCENE[scene_id],
            EXPECTED_PROVIDER_COMMITS_PER_SCENE[scene_id],
        )
        for scene_id in EXPECTED_PROVIDER_RAW_ROWS_PER_SCENE
    }
    if set(scene_rows) != set(expected_scenes):
        raise IntegratedRuntimeError("formal provider per-scene ledger differs")
    for scene_id, (raw_rows, k8_rows, commits) in scene_rows.items():
        if (
            raw_rows != expected_scenes[scene_id][0]
            or k8_rows != expected_scenes[scene_id][1]
            or commits != expected_scenes[scene_id][2]
        ):
            raise IntegratedRuntimeError("formal provider per-scene work differs")
    if sum(row[2] for row in scene_rows.values()) != EXPECTED_PROVIDER_TRACKER_COMMITS:
        raise IntegratedRuntimeError("formal provider per-scene commit total differs")


def _assert_cross_arm_worker_identity(
    control_identity: Mapping[str, Any], integrated_identity: Mapping[str, Any]
) -> None:
    if dict(control_identity) != dict(integrated_identity):
        raise IntegratedRuntimeError("control/integrated GPU/runtime identity differs")


def _run_formal_h10_runtime_arm_with_held_inputs(
    *,
    arm: str,
    output: Path,
    runtime_contract: Path,
    expected_runtime_contract_sha256: str,
    expected_runner_sha256: str,
    expected_runner_test_sha256: str,
    self_pins: Mapping[str, str],
    environment: Mapping[str, Any],
    parent_runtime_identity: Mapping[str, Any],
    child_runtime_probes: Mapping[str, Mapping[str, Any]],
    provider_checkout_preprobe: Mapping[str, Any],
    held_schedule: _HeldPinnedRegularFile,
    held_manifest: _HeldPinnedRegularFile,
    held_control: _HeldPinnedRegularFile | None,
) -> dict[str, Any]:
    """Execute after public preflight has held every control-plane input."""

    schedule_hash = held_schedule.sha256
    manifest_hash = held_manifest.sha256
    bundle = _parse_held_provider_schedule(held_schedule)
    if bundle.sha256 != schedule_hash:
        raise IntegratedRuntimeError("parsed held provider schedule hash differs")
    manifest = _parse_held_native_manifest(
        held_manifest,
        provider_bundle=bundle,
        scene_root=DEFAULT_SCENE_ROOT,
    )
    view = _minimal_manifest_view(manifest)
    _assert_formal_manifest_view(view, bundle)

    provider_checkout_before = _snapshot_provider_checkout(PROVIDER_BOXER_ROOT)
    if provider_checkout_before != dict(provider_checkout_preprobe):
        raise IntegratedRuntimeError(
            "provider checkout changed across child runtime probes"
        )
    static_before = _formal_static_snapshot(
        runtime_contract=runtime_contract, self_pins=self_pins
    )
    inputs_before = _snapshot_manifest_inputs(view, scene_root=DEFAULT_SCENE_ROOT)
    t05_before = _snapshot_t05_opaque(bundle)
    fresh = importlib.import_module("tools.run_scannet_s3r_h10_fresh_boxer_provider")
    if fresh.BOXER_ROOT.resolve(strict=True) != PROVIDER_BOXER_ROOT.resolve(strict=True):
        raise IntegratedRuntimeError("fresh provider checkout root differs")
    try:
        provider_assets_before, provider_asset_paths = fresh._validate_frozen_assets(
            DEFAULT_PROVIDER_SCHEDULE,
            fresh.BOXER_ROOT,
            PROVIDER_CONTRACT_PATH,
            EXPECTED_PROVIDER_CONTRACT_SHA256,
        )
    except BaseException as error:
        raise IntegratedRuntimeError("fresh provider frozen assets differ") from error
    if "torch" in sys.modules:
        raise IntegratedRuntimeError("formal preflight imported torch in the parent")

    provider_assets_identity_sha256 = _ledger_digest(
        [
            {
                "name": name,
                "sha256": provider_assets_before[name]["sha256_before"],
            }
            for name in sorted(provider_assets_before)
        ]
    )
    cross_arm_bindings = {
        "native_manifest_sha256": manifest_hash,
        "provider_schedule_sha256": schedule_hash,
        "runtime_contract_sha256": self_pins["runtime_contract"],
        "runner_sha256": self_pins["runner"],
        "runner_test_sha256": self_pins["runner_test"],
        "raw_source_json_sha256": EXPECTED_RAW_SOURCE_JSON_SHA256,
        "raw_source_npz_sha256": EXPECTED_RAW_SOURCE_NPZ_SHA256,
        "raw_source_array_content_sha256": EXPECTED_RAW_SOURCE_ARRAY_CONTENT_SHA256,
        "raw_source_k8_membership_sha256": (
            EXPECTED_RAW_SOURCE_K8_MEMBERSHIP_SHA256
        ),
        "static_assets_identity_sha256": static_before["identity_sha256"],
        "manifest_inputs_identity_sha256": inputs_before["identity_sha256"],
        "t05_opaque_identity_sha256": t05_before["identity_sha256"],
        "provider_assets_identity_sha256": provider_assets_identity_sha256,
        "provider_checkout_identity_sha256": provider_checkout_before[
            "identity_sha256"
        ],
        "provider_checkout_commit": provider_checkout_before["commit"],
        "provider_checkout_tree": provider_checkout_before["tree"],
        "child_runtime_identity_sha256": _child_runtime_identity_sha256(
            child_runtime_probes
        ),
    }
    control_binding = None
    if held_control is not None:
        control_value = _strict_json_object_bytes(
            held_control.payload, label="control timing receipt"
        )
        control_binding = _validate_control_receipt(
            control_value,
            held_control.sha256,
            expected_bindings=cross_arm_bindings,
            expected_cuda_visible_devices=environment["CUDA_VISIBLE_DEVICES"],
            expected_manifest_view=view,
        )
    native_config = {
        "python_executable": child_runtime_probes["native"]["python_executable"],
        "python_sys_path": list(
            child_runtime_probes["native"]["python_sys_path"]
        ),
        "expected_post_factory_python_sys_path": list(
            child_runtime_probes["native"][
                "expected_post_factory_python_sys_path"
            ]
        ),
        "expected_runtime_identity": dict(NATIVE_RUNTIME_IDENTITY),
        "native_config_path": os.fspath(NATIVE_CONFIG_PATH),
        "cutr_checkpoint": os.fspath(NATIVE_CUTR_CHECKPOINT),
        "clip_checkpoint": os.fspath(NATIVE_CLIP_CHECKPOINT),
        "class_features": os.fspath(NATIVE_CLASS_FEATURES),
        "class_names": os.fspath(NATIVE_CLASS_NAMES),
        "pst_path": os.fspath(NATIVE_PST.resolve(strict=True)),
    }
    provider_config = {
        "python_executable": child_runtime_probes["provider"][
            "python_executable"
        ],
        "python_sys_path": list(
            child_runtime_probes["provider"]["python_sys_path"]
        ),
        "expected_post_factory_python_sys_path": list(
            child_runtime_probes["provider"][
                "expected_post_factory_python_sys_path"
            ]
        ),
        "expected_runtime_identity": dict(PROVIDER_RUNTIME_IDENTITY),
        "boxer_root": os.fspath(fresh.BOXER_ROOT),
        "boxer_checkpoint": os.fspath(
            fresh.BOXER_ROOT / fresh.BOXER_CHECKPOINT_RELPATH
        ),
        "owl_checkpoint": os.fspath(fresh.OWL_CHECKPOINT),
    }
    cuda_visible_devices = environment["CUDA_VISIBLE_DEVICES"]
    if arm == "control":
        result = _execute_control_runtime(
            manifest_view=view,
            scene_root=DEFAULT_SCENE_ROOT,
            native_factory=REAL_FACTORIES.native,
            native_factory_config=native_config,
            cuda_visible_devices=cuda_visible_devices,
        )
    else:
        result = _execute_integrated_runtime(
            manifest_view=view,
            scene_root=DEFAULT_SCENE_ROOT,
            factories=REAL_FACTORIES,
            native_factory_config=native_config,
            provider_factory_config=provider_config,
            cuda_visible_devices=cuda_visible_devices,
        )
    _assert_formal_runtime_result(result, arm=arm)
    if control_binding is not None:
        integrated_worker_identity = {
            name: result["workers"]["native"][name]
            for name in control_binding["worker_identity"]
        }
        _assert_cross_arm_worker_identity(
            control_binding["worker_identity"], integrated_worker_identity
        )

    # Complete both post-stream interpreter probes before the final immutable
    # barrier.  The provider probe imports the frozen local bridge and external
    # ``utils`` package, so putting it after a checkout/static snapshot would
    # leave one last executable import outside the before/after closure.
    child_runtime_probes_after = {
        "native": _probe_frozen_child_runtime(
            executable=NATIVE_PYTHON_EXECUTABLE,
            expected_spawn_entry_sys_path=NATIVE_FROZEN_SYS_PATH,
            expected_post_factory_sys_path=NATIVE_POST_FACTORY_SYS_PATH,
            expected_identity=NATIVE_RUNTIME_IDENTITY,
        ),
        "provider": _probe_frozen_child_runtime(
            executable=PROVIDER_PYTHON_EXECUTABLE,
            expected_spawn_entry_sys_path=PROVIDER_FROZEN_SYS_PATH,
            expected_post_factory_sys_path=PROVIDER_POST_FACTORY_SYS_PATH,
            expected_identity=PROVIDER_RUNTIME_IDENTITY,
            boxer_root=PROVIDER_BOXER_ROOT,
        ),
    }

    # Re-open every manifest-named input and every opaque asset after the
    # stream and post probes.  Nothing is trusted merely because a worker or
    # probe reported success.
    try:
        native_manifest_builder.verify_manifest_files(
            manifest, scene_root=DEFAULT_SCENE_ROOT
        )
    except BaseException as error:
        raise IntegratedRuntimeError("formal manifest inputs changed after stream") from error
    inputs_after = _snapshot_manifest_inputs(view, scene_root=DEFAULT_SCENE_ROOT)
    t05_after = _snapshot_t05_opaque(bundle)
    try:
        provider_assets_after = fresh._rehash_assets(
            provider_assets_before, provider_asset_paths
        )
    except BaseException as error:
        raise IntegratedRuntimeError("provider asset changed during stream") from error
    held_schedule.verify_after_stream()
    held_manifest.verify_after_stream()
    control_binding_after = None
    if held_control is not None:
        held_control.verify_after_stream()
        control_binding_after = _validate_control_receipt(
            _strict_json_object_bytes(
                held_control.payload, label="control timing receipt after stream"
            ),
            held_control.sha256,
            expected_bindings=cross_arm_bindings,
            expected_cuda_visible_devices=environment["CUDA_VISIBLE_DEVICES"],
            expected_manifest_view=view,
        )
    self_pins_after = _validate_formal_self_and_contract(
        runtime_contract=runtime_contract,
        expected_runtime_contract_sha256=expected_runtime_contract_sha256,
        expected_runner_sha256=expected_runner_sha256,
        expected_runner_test_sha256=expected_runner_test_sha256,
    )
    parent_runtime_identity_after = _validate_parent_runtime_identity()

    # Final executable-file barrier.  Every post-stream child probe and every
    # local/provider asset verifier above has completed.  After these two
    # snapshots, only comparisons, receipt assembly, and create-only
    # publication run; no provider/local executable is imported again.
    static_after = _formal_static_snapshot(
        runtime_contract=runtime_contract, self_pins=self_pins
    )
    provider_checkout_after = _snapshot_provider_checkout(PROVIDER_BOXER_ROOT)
    if (
        inputs_after != inputs_before
        or static_after != static_before
        or provider_checkout_after != provider_checkout_before
        or t05_after != t05_before
        or child_runtime_probes_after != dict(child_runtime_probes)
        or any(
            provider_assets_after[name] != row["sha256_before"]
            for name, row in provider_assets_before.items()
        )
        or self_pins_after != self_pins
        or parent_runtime_identity_after != dict(parent_runtime_identity)
        or control_binding_after != control_binding
    ):
        raise IntegratedRuntimeError("formal immutable before/after snapshot differs")

    result.update(
        {
            "formal_h10": True,
            "synthetic_worker_injection": False,
            "timing_only": True,
            "runtime_only": True,
            "original_terminal_exact": False,
            "full100_not_authorized": True,
            "h10_gt_oracle_authorized": False,
            "gt_access_authorized": False,
            "integrated_provider_runtime_qualified": (
                bool(result["performance_gates"]["all_met"])
                if arm == "integrated"
                else False
            ),
            "integrated_realtime_qualified": (
                bool(result["performance_gates"]["all_met"])
                if arm == "integrated"
                else False
            ),
            "native_fps_protocol_equivalent": False,
            "bindings": cross_arm_bindings,
            "immutable_before_after_verified": True,
            "environment": environment,
            "parent_runtime_identity": dict(parent_runtime_identity),
        }
    )
    if control_binding is not None:
        integrated_fps = float(result["native"]["fps"])
        result["control_comparison"] = {
            "control_receipt_sha256": control_binding["sha256"],
            "control_fps": control_binding["fps"],
            "integrated_fps": integrated_fps,
            "integrated_over_control_fps_ratio": (
                integrated_fps / control_binding["fps"]
            ),
            "diagnostic_only": True,
            "primary_gate": "integrated_absolute_fps_gte_10",
        }
    _publish_timing_receipt(output, result)
    return result


def run_formal_h10_runtime_arm(
    *,
    arm: str,
    output: Path,
    runtime_contract: Path,
    expected_runtime_contract_sha256: str,
    expected_runner_sha256: str,
    expected_runner_test_sha256: str,
    control_receipt: Path | None = None,
    expected_control_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one frozen formal timing arm; this function has no injection seam."""

    parent_runtime_identity = _validate_parent_runtime_identity()
    if (control_receipt is None) != (expected_control_receipt_sha256 is None):
        raise IntegratedRuntimeError("control receipt path/hash must be supplied together")
    output, control_receipt = _require_formal_paths(
        arm=arm, output=output, control_receipt=control_receipt
    )
    self_pins = _validate_formal_self_and_contract(
        runtime_contract=runtime_contract,
        expected_runtime_contract_sha256=expected_runtime_contract_sha256,
        expected_runner_sha256=expected_runner_sha256,
        expected_runner_test_sha256=expected_runner_test_sha256,
    )
    output = _formal_output_preflight(
        output,
        runtime_contract=runtime_contract,
        control_receipt=control_receipt,
    )
    environment = _validate_environment(require_cuda=True)
    if "torch" in sys.modules:
        raise IntegratedRuntimeError("parent imported torch before worker spawn")
    _snapshot_external_command_binaries()
    try:
        _assert_import_shadow_candidates_absent(
            tuple(_native_static_asset_paths().values())
        )
    except RuntimeError as error:
        raise IntegratedRuntimeError(
            "pre-probe frozen import-shadow candidate differs"
        ) from error
    # This ignored-file guard must precede the provider interpreter probe: that
    # probe imports ``utils`` from the external checkout and an ignored native
    # extension would otherwise execute before its origin could be inspected.
    provider_checkout_preprobe = _snapshot_provider_checkout(PROVIDER_BOXER_ROOT)
    child_runtime_probes = {
        "native": _probe_frozen_child_runtime(
            executable=NATIVE_PYTHON_EXECUTABLE,
            expected_spawn_entry_sys_path=NATIVE_FROZEN_SYS_PATH,
            expected_post_factory_sys_path=NATIVE_POST_FACTORY_SYS_PATH,
            expected_identity=NATIVE_RUNTIME_IDENTITY,
        ),
        "provider": _probe_frozen_child_runtime(
            executable=PROVIDER_PYTHON_EXECUTABLE,
            expected_spawn_entry_sys_path=PROVIDER_FROZEN_SYS_PATH,
            expected_post_factory_sys_path=PROVIDER_POST_FACTORY_SYS_PATH,
            expected_identity=PROVIDER_RUNTIME_IDENTITY,
            boxer_root=PROVIDER_BOXER_ROOT,
        ),
    }
    if "torch" in sys.modules:
        raise IntegratedRuntimeError("child probes imported torch in the parent")

    with contextlib.ExitStack() as held_stack:
        held_schedule = held_stack.enter_context(
            _HeldPinnedRegularFile(
                DEFAULT_PROVIDER_SCHEDULE,
                maximum=MAX_ASSET_BYTES,
                label="provider schedule",
                expected_sha256=EXPECTED_PROVIDER_SCHEDULE_SHA256,
            )
        )
        held_manifest = held_stack.enter_context(
            _HeldPinnedRegularFile(
                DEFAULT_NATIVE_MANIFEST,
                maximum=MAX_ASSET_BYTES,
                label="native full-stream manifest",
                expected_sha256=EXPECTED_NATIVE_MANIFEST_SHA256,
            )
        )
        held_control = None
        if control_receipt is not None:
            held_control = held_stack.enter_context(
                _HeldPinnedRegularFile(
                    control_receipt,
                    maximum=MAX_TIMING_JSON_BYTES,
                    label="control timing receipt",
                    expected_sha256=str(expected_control_receipt_sha256),
                )
            )
        return _run_formal_h10_runtime_arm_with_held_inputs(
            arm=arm,
            output=output,
            runtime_contract=runtime_contract,
            expected_runtime_contract_sha256=expected_runtime_contract_sha256,
            expected_runner_sha256=expected_runner_sha256,
            expected_runner_test_sha256=expected_runner_test_sha256,
            self_pins=self_pins,
            environment=environment,
            parent_runtime_identity=parent_runtime_identity,
            child_runtime_probes=child_runtime_probes,
            provider_checkout_preprobe=provider_checkout_preprobe,
            held_schedule=held_schedule,
            held_manifest=held_manifest,
            held_control=held_control,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create one no-GT, timing-only H10 runtime receipt. Run control "
            "first, then integrated with the exact control receipt hash."
        )
    )
    parser.add_argument("--arm", required=True, choices=("control", "integrated"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-contract", required=True, type=Path)
    parser.add_argument("--expected-runtime-contract-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-runner-test-sha256", required=True)
    parser.add_argument("--control-receipt", type=Path)
    parser.add_argument("--expected-control-receipt-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_formal_h10_runtime_arm(
            arm=args.arm,
            output=args.output,
            runtime_contract=args.runtime_contract,
            expected_runtime_contract_sha256=args.expected_runtime_contract_sha256,
            expected_runner_sha256=args.expected_runner_sha256,
            expected_runner_test_sha256=args.expected_runner_test_sha256,
            control_receipt=args.control_receipt,
            expected_control_receipt_sha256=args.expected_control_receipt_sha256,
        )
    except IntegratedRuntimeError:
        print("ERROR: integrated runtime failed closed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

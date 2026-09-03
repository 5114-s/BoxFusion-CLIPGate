"""Strict BoxerNet lifting adapter for the BoxFusion ablation.

The adapter deliberately keeps CuTR's 2D proposals, proposal order, detector
scores, class logits, and descriptors.  BoxerNet is only allowed to produce
the camera-frame 3D geometry associated with those rows.  Its aleatoric
confidence is recorded for analysis but never used for filtering or scoring.

The official Boxer repository is loaded lazily so the CuTR baseline path does
not import Boxer or perturb its runtime/RNG state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from boxfusion.boxes import GeneralInstance3DBoxes


OFFICIAL_BOXER_COMMIT = "1f86542dc342a4b1d474c87c97c5d1d6566d9148"
OFFICIAL_BOXER_CHECKPOINT = "boxernet_hw960in2x6d768-c88128f8.ckpt"
OFFICIAL_DINOV3_CHECKPOINT = (
    "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
)
OFFICIAL_DINOV3_SHA256 = (
    "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea"
)


class LiftingContractError(RuntimeError):
    """Raised when a strict one-input-row/one-output-row contract is broken."""


@contextmanager
def _preserve_rng_state(include_cuda: bool):
    """Prevent observer/model construction from perturbing BoxFusion RNG."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if include_cuda and torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _as_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return np.ascontiguousarray(result)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_array(value: Any) -> str:
    array = _as_numpy(value)
    header = (
        str(array.dtype).encode("utf-8")
        + b"\0"
        + repr(tuple(array.shape)).encode("utf-8")
        + b"\0"
    )
    return _sha256_bytes(header + array.tobytes(order="C"))


def _sha256_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _field_hash(instances: Any, name: str) -> Optional[str]:
    if not instances.has(name):
        return None
    value = instances.get(name)
    if isinstance(value, torch.Tensor):
        return _sha256_array(value)
    if hasattr(value, "tensor"):
        pieces = [_sha256_array(value.tensor)]
        if hasattr(value, "R"):
            pieces.append(_sha256_array(value.R))
        return _sha256_bytes("|".join(pieces).encode("ascii"))
    if isinstance(value, np.ndarray):
        return _sha256_array(value)
    return _sha256_bytes(repr(value).encode("utf-8"))


def protected_proposal_hashes(instances: Any) -> Dict[str, Optional[str]]:
    """Hash every CuTR field that Boxer is forbidden to modify."""

    names = (
        "pred_boxes",
        "scores",
        "pred_classes",
        "pred_logits",
        "object_desc",
    )
    return {name: _field_hash(instances, name) for name in names}


def geometry_hash(instances: Any) -> str:
    boxes = instances.pred_boxes_3d
    return _sha256_bytes(
        (
            _sha256_array(boxes.tensor)
            + "|"
            + _sha256_array(boxes.R)
        ).encode("ascii")
    )


def _stable_frame_seed(base_seed: int, scene_id: str, frame_id: int) -> int:
    payload = f"{int(base_seed)}:{scene_id}:{int(frame_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def deterministic_sdp_from_depth(
    depth: np.ndarray,
    K: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    num_samples: int,
    seed: int,
) -> torch.Tensor:
    """Match Boxer's ScanNet depth sampling without touching global NumPy RNG."""

    depth = _as_numpy(depth, np.float32)
    K = _as_numpy(K, np.float32)
    camera_to_world = _as_numpy(camera_to_world, np.float32)
    if depth.ndim != 2:
        raise LiftingContractError(
            f"Expected a 2D depth map, received shape {depth.shape}"
        )
    if K.shape != (3, 3):
        raise LiftingContractError(
            f"Expected 3x3 depth intrinsics, received shape {K.shape}"
        )
    if camera_to_world.shape != (4, 4):
        raise LiftingContractError(
            "Expected a 4x4 camera-to-world pose, received "
            f"shape {camera_to_world.shape}"
        )

    finite = np.isfinite(depth)
    valid_depth = np.where(finite & (depth > 0.0), depth, 0.0)
    height, width = valid_depth.shape
    step = max(
        1,
        int(np.sqrt(height * width / max(int(num_samples) * 2, 1))),
    )
    yy, xx = np.mgrid[0:height:step, 0:width:step]
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    zz = valid_depth[yy, xx]
    valid = zz > 0.0
    yy = yy[valid]
    xx = xx[valid]
    zz = zz[valid]

    if zz.shape[0] > int(num_samples):
        rng = np.random.default_rng(int(seed))
        keep = rng.choice(
            zz.shape[0],
            size=int(num_samples),
            replace=False,
        )
        # Sorting preserves image scan order after deterministic sampling.
        keep.sort()
        yy = yy[keep]
        xx = xx[keep]
        zz = zz[keep]

    if zz.shape[0] == 0:
        return torch.zeros((0, 3), dtype=torch.float32)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if min(abs(fx), abs(fy)) < 1e-6:
        raise LiftingContractError("Depth intrinsics contain a zero focal length")

    x = (xx.astype(np.float32) - cx) * zz / fx
    y = (yy.astype(np.float32) - cy) * zz / fy
    points_camera = np.stack((x, y, zz), axis=-1).astype(np.float32)
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    points_world = points_camera @ rotation.T + translation

    if points_world.shape[0] < int(num_samples):
        padding = np.full(
            (int(num_samples) - points_world.shape[0], 3),
            np.nan,
            dtype=np.float32,
        )
        points_world = np.concatenate((points_world, padding), axis=0)
    return torch.from_numpy(points_world.astype(np.float32, copy=False))


def boxer_world_to_boxfusion_camera(
    center_world: torch.Tensor,
    dims_local_xyz: torch.Tensor,
    rotation_world_object: torch.Tensor,
    camera_to_world: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Boxer world OBBs to BoxFusion optical-camera OBBs."""

    center_world = center_world.float()
    dims_local_xyz = dims_local_xyz.float()
    rotation_world_object = rotation_world_object.float()
    camera_to_world = camera_to_world.float()

    rotation_world_camera = camera_to_world[:3, :3]
    translation_world_camera = camera_to_world[:3, 3]
    rotation_camera_world = rotation_world_camera.transpose(0, 1)
    center_camera = torch.einsum(
        "ij,nj->ni",
        rotation_camera_world,
        center_world - translation_world_camera.unsqueeze(0),
    )
    rotation_camera_object = torch.einsum(
        "ij,njk->nik",
        rotation_camera_world,
        rotation_world_object,
    )
    xyz_dims = torch.cat((center_camera, dims_local_xyz), dim=-1)
    return xyz_dims, rotation_camera_object


def project_rotations_to_so3(rotations: torch.Tensor) -> torch.Tensor:
    """Remove mixed-precision drift while preserving the nearest rotation."""

    rotations = rotations.float()
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise LiftingContractError(
            f"Expected Nx3x3 rotations, received {tuple(rotations.shape)}"
        )
    if rotations.shape[0] == 0:
        return rotations
    u, _, vh = torch.linalg.svd(rotations)
    candidate = torch.matmul(u, vh)
    signs = torch.ones(
        (rotations.shape[0], 3),
        dtype=rotations.dtype,
        device=rotations.device,
    )
    signs[:, -1] = torch.where(
        torch.linalg.det(candidate) < 0.0,
        -1.0,
        1.0,
    )
    return torch.matmul(torch.matmul(u, torch.diag_embed(signs)), vh)


def project_centers(
    xyz_camera: torch.Tensor,
    image_K: torch.Tensor,
) -> torch.Tensor:
    """Project camera-frame centers using the same optical pinhole convention."""

    homogeneous = torch.einsum("ij,nj->ni", image_K.float(), xyz_camera.float())
    z = homogeneous[:, 2:3]
    if torch.any(~torch.isfinite(z)) or torch.any(torch.abs(z) < 1e-6):
        raise LiftingContractError(
            "Boxer produced a center that cannot be projected"
        )
    return homogeneous[:, :2] / z


@dataclass(frozen=True)
class BoxerLiftingConfig:
    mode: str
    apply_stage: str
    official_root: str
    checkpoint: str
    expected_commit: str
    checkpoint_sha256: str
    dinov3_sha256: str
    precision: str
    use_sdp: bool
    sdp_samples: int
    seed: int
    diagnostics_dir: str
    cache_image_features: bool = False

    @classmethod
    def from_mapping(
        cls,
        mapping: Dict[str, Any],
        *,
        code_root: str,
    ) -> "BoxerLiftingConfig":
        mode = str(mapping.get("mode", "observer")).lower()
        if mode not in ("observer", "active"):
            raise ValueError("lifting.boxer.mode must be observer or active")
        apply_stage = str(mapping.get("apply_stage", "post_filter")).lower()
        if apply_stage not in ("post_filter", "pre_filter"):
            raise ValueError(
                "lifting.boxer.apply_stage must be post_filter or pre_filter"
            )

        official_root = os.path.abspath(
            os.path.expanduser(
                str(
                    mapping.get(
                        "official_root",
                        os.path.join(code_root, "third_party", "boxer"),
                    )
                )
            )
        )
        checkpoint = os.path.abspath(
            os.path.expanduser(
                str(
                    mapping.get(
                        "checkpoint",
                        os.path.join(
                            official_root,
                            "ckpts",
                            OFFICIAL_BOXER_CHECKPOINT,
                        ),
                    )
                )
            )
        )
        precision = str(mapping.get("precision", "bfloat16")).lower()
        if precision not in ("float32", "bfloat16"):
            raise ValueError(
                "lifting.boxer.precision must be float32 or bfloat16"
            )

        diagnostics_dir = os.path.abspath(
            os.path.expanduser(
                str(
                    mapping.get(
                        "diagnostics_dir",
                        os.path.join(code_root, "diagnostics", "boxer_lifting"),
                    )
                )
            )
        )
        return cls(
            mode=mode,
            apply_stage=apply_stage,
            official_root=official_root,
            checkpoint=checkpoint,
            expected_commit=str(
                mapping.get("expected_commit", OFFICIAL_BOXER_COMMIT)
            ),
            checkpoint_sha256=str(
                mapping.get("checkpoint_sha256", "")
            ).lower(),
            dinov3_sha256=str(
                mapping.get(
                    "dinov3_sha256",
                    OFFICIAL_DINOV3_SHA256,
                )
            ).lower(),
            precision=precision,
            use_sdp=bool(mapping.get("use_sdp", True)),
            sdp_samples=int(mapping.get("sdp_samples", 10000)),
            seed=int(mapping.get("seed", 0)),
            diagnostics_dir=diagnostics_dir,
            cache_image_features=bool(mapping.get("cache_image_features", False)),
        )


class BoxerLiftingAdapter:
    """Lazy, fail-closed BoxerNet adapter."""

    def __init__(self, config: BoxerLiftingConfig, device: str):
        self.config = config
        self.device = "cuda" if str(device).startswith("cuda") else str(device)
        if self.device not in ("cuda", "cpu", "mps"):
            raise ValueError(f"Unsupported Boxer device: {device}")
        self.model = None
        self._BaseLoader = None
        self._PoseTW = None
        self._diag_initialized = set()
        self._stats = {
            "calls": 0,
            "proposals": 0,
            "applied": 0,
            "runtime_ms": [],
            "observer_calls": 0,
            "feature_cache_hits": 0,
            "feature_cache_misses": 0,
        }
        self._checkpoint_sha256 = None
        # Exactly one keyframe encoder state is retained.  Entries are keyed
        # by scene/frame and guarded by a digest of every encoder input; 2D
        # proposal boxes are intentionally excluded because they feed query().
        self._feature_cache = None

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def apply_stage(self) -> str:
        return self.config.apply_stage

    @property
    def mutation_enabled(self) -> bool:
        return self.config.mode == "active"

    def _verify_assets(self) -> None:
        root = Path(self.config.official_root)
        if not (root / "boxernet" / "boxernet.py").is_file():
            raise FileNotFoundError(
                f"Official Boxer source is absent: {root}"
            )
        if not Path(self.config.checkpoint).is_file():
            raise FileNotFoundError(
                f"Official BoxerNet checkpoint is absent: "
                f"{self.config.checkpoint}"
            )
        dino_path = (
            root / "ckpts" / OFFICIAL_DINOV3_CHECKPOINT
        )
        if not dino_path.is_file():
            raise FileNotFoundError(
                f"Official DINOv3 checkpoint is absent: {dino_path}"
            )

        try:
            commit = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise LiftingContractError(
                f"Could not verify official Boxer source commit: {error}"
            ) from error
        if commit != self.config.expected_commit:
            raise LiftingContractError(
                "Official Boxer commit mismatch: "
                f"expected={self.config.expected_commit}, actual={commit}"
            )

        self._checkpoint_sha256 = _sha256_file(self.config.checkpoint)
        if (
            self.config.checkpoint_sha256
            and self._checkpoint_sha256 != self.config.checkpoint_sha256
        ):
            raise LiftingContractError(
                "BoxerNet checkpoint SHA256 mismatch: "
                f"expected={self.config.checkpoint_sha256}, "
                f"actual={self._checkpoint_sha256}"
            )
        dino_sha = _sha256_file(str(dino_path))
        if self.config.dinov3_sha256 and dino_sha != self.config.dinov3_sha256:
            raise LiftingContractError(
                "DINOv3 checkpoint SHA256 mismatch: "
                f"expected={self.config.dinov3_sha256}, actual={dino_sha}"
            )

    def _load_model(self) -> None:
        if self.model is not None:
            return
        self._verify_assets()

        root = self.config.official_root
        if root not in sys.path:
            sys.path.insert(0, root)
        loaded_utils = sys.modules.get("utils")
        if loaded_utils is not None:
            module_file = os.path.abspath(
                str(getattr(loaded_utils, "__file__", ""))
            )
            if module_file and not module_file.startswith(root + os.sep):
                raise LiftingContractError(
                    "A conflicting top-level 'utils' package was imported "
                    f"before Boxer: {module_file}"
                )

        if (
            self.device == "cuda"
            and self.config.precision == "bfloat16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise LiftingContractError(
                "Configured Boxer bfloat16 precision is unsupported by "
                "the active CUDA device; use an explicitly fingerprinted "
                "float32 config instead"
            )

        with _preserve_rng_state(include_cuda=self.device == "cuda"):
            from boxernet.boxernet import BoxerNet
            from loaders.base_loader import BaseLoader
            from utils.tw.pose import PoseTW

            self.model = BoxerNet.load_from_checkpoint(
                self.config.checkpoint,
                device=self.device,
            )
        self.model.eval()
        self._BaseLoader = BaseLoader
        self._PoseTW = PoseTW

    @staticmethod
    def _normalize_image(image: Any) -> np.ndarray:
        image_np = _as_numpy(image)
        if image_np.ndim != 3 or image_np.shape[-1] != 3:
            raise LiftingContractError(
                f"Expected HxWx3 RGB image, received {image_np.shape}"
            )
        if np.issubdtype(image_np.dtype, np.floating):
            if not np.all(np.isfinite(image_np)):
                raise LiftingContractError("RGB image contains NaN or Inf")
            maximum = float(image_np.max()) if image_np.size else 0.0
            if maximum <= 1.0 + 1e-5:
                image_np = np.clip(image_np * 255.0, 0.0, 255.0)
            image_np = np.rint(image_np).astype(np.uint8)
        else:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image_np)

    @staticmethod
    def _normalize_depth(depth: Any) -> np.ndarray:
        depth_np = _as_numpy(depth)
        if depth_np.ndim != 2:
            raise LiftingContractError(
                f"Expected HxW depth, received {depth_np.shape}"
            )
        if np.issubdtype(depth_np.dtype, np.integer):
            depth_np = depth_np.astype(np.float32) / 1000.0
        else:
            depth_np = depth_np.astype(np.float32)
        depth_np[~np.isfinite(depth_np)] = 0.0
        depth_np[depth_np < 0.0] = 0.0
        return np.ascontiguousarray(depth_np)

    def _make_datum(
        self,
        *,
        image: Any,
        depth: Any,
        boxes_xyxy: torch.Tensor,
        image_K: Any,
        depth_K: Any,
        camera_to_world: Any,
        scene_id: str,
        frame_id: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import cv2

        self._load_model()
        image_np = self._normalize_image(image)
        depth_np = self._normalize_depth(depth)
        image_K_np = _as_numpy(image_K, np.float32)
        depth_K_np = _as_numpy(depth_K, np.float32)
        pose_np = _as_numpy(camera_to_world, np.float32)
        if image_K_np.shape != (3, 3):
            raise LiftingContractError(
                f"Expected 3x3 RGB intrinsics, received {image_K_np.shape}"
            )
        if pose_np.shape != (4, 4):
            raise LiftingContractError(
                f"Expected 4x4 camera pose, received {pose_np.shape}"
            )

        original_h, original_w = image_np.shape[:2]
        boxer_hw = int(self.model.hw)
        scale_x = boxer_hw / float(original_w)
        scale_y = boxer_hw / float(original_h)
        resized = cv2.resize(
            image_np,
            (boxer_hw, boxer_hw),
            interpolation=cv2.INTER_LINEAR,
        )
        image_tensor = self._BaseLoader.img_to_tensor(resized).float()

        scaled_K = image_K_np.copy()
        scaled_K[0, :] *= scale_x
        scaled_K[1, :] *= scale_y
        camera = self._BaseLoader.pinhole_from_K(
            boxer_hw,
            boxer_hw,
            float(scaled_K[0, 0]),
            float(scaled_K[1, 1]),
            float(scaled_K[0, 2]),
            float(scaled_K[1, 2]),
            valid_radius=(boxer_hw, boxer_hw),
        ).float()

        boxes = boxes_xyxy.detach().float().cpu()
        boxer_boxes = torch.stack(
            (
                boxes[:, 0] * scale_x,
                boxes[:, 2] * scale_x,
                boxes[:, 1] * scale_y,
                boxes[:, 3] * scale_y,
            ),
            dim=-1,
        )
        pose_data = torch.from_numpy(
            np.concatenate(
                (
                    pose_np[:3, :3].reshape(-1),
                    pose_np[:3, 3],
                )
            ).astype(np.float32)
        )

        if self.config.use_sdp:
            seed = _stable_frame_seed(
                self.config.seed,
                scene_id,
                frame_id,
            )
            sdp_world = deterministic_sdp_from_depth(
                depth_np,
                depth_K_np,
                pose_np,
                num_samples=self.config.sdp_samples,
                seed=seed,
            )
        else:
            seed = None
            sdp_world = torch.zeros((0, 3), dtype=torch.float32)

        datum = {
            "img0": image_tensor,
            "cam0": camera,
            "T_world_rig0": self._PoseTW(pose_data),
            "sdp_w": sdp_world,
            "time_ns0": int(frame_id),
            "rotated0": torch.tensor(False).reshape(1),
            "bb2d": boxer_boxes,
        }
        metadata = {
            "image_np": image_np,
            "depth_np": depth_np,
            "image_K_np": image_K_np,
            "depth_K_np": depth_K_np,
            "pose_np": pose_np,
            "scaled_K": scaled_K,
            "boxer_boxes": boxer_boxes,
            "sdp_seed": seed,
        }
        metadata["encoder_input_sha256"] = _sha256_bytes(
            "|".join(
                (
                    _sha256_array(image_np),
                    _sha256_array(depth_np),
                    _sha256_array(image_K_np),
                    _sha256_array(depth_K_np),
                    _sha256_array(pose_np),
                    _sha256_array(sdp_world),
                )
            ).encode("ascii")
        )
        return datum, metadata

    def _forward(
        self,
        datum: Dict[str, Any],
        camera_to_world: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        with _preserve_rng_state(include_cuda=self.device == "cuda"):
            if self.device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            if self.device == "mps":
                context = nullcontext()
            elif self.config.precision == "bfloat16":
                context = torch.autocast(
                    device_type=self.device,
                    dtype=torch.bfloat16,
                )
            else:
                context = nullcontext()
            with torch.no_grad(), context:
                output = self.model.forward(datum)
            if self.device == "cuda":
                torch.cuda.synchronize()
            runtime_ms = (time.perf_counter() - started) * 1000.0

        obbs_world = output["obbs_pr_w"][0].cpu()
        centers_world = obbs_world.bb3_center_world.float()
        dims = obbs_world.bb3_diagonal.float()
        rotations_world_object = obbs_world.T_world_object.R.float()
        pose_tensor = torch.from_numpy(camera_to_world).float()
        xyz_dims, rotations_camera_object = boxer_world_to_boxfusion_camera(
            centers_world,
            dims,
            rotations_world_object,
            pose_tensor,
        )
        rotations_so3 = project_rotations_to_so3(rotations_camera_object)
        rotation_correction_max_abs = (
            float(
                torch.max(
                    torch.abs(rotations_so3 - rotations_camera_object)
                ).item()
            )
            if rotations_camera_object.shape[0]
            else 0.0
        )
        return {
            "xyz_dims": xyz_dims,
            "rotations": rotations_so3,
            "rotation_correction_max_abs": torch.tensor(
                rotation_correction_max_abs,
                dtype=torch.float64,
            ),
            "confidence": obbs_world.prob.squeeze(-1).float(),
            "logvar": output["obbs_pr_logvar"][0].detach().float().cpu(),
            "raw_params": output["obbs_pr_params"][0].detach().float().cpu(),
            "runtime_ms": torch.tensor(runtime_ms, dtype=torch.float64),
        }

    def forward_raw_with_feature_cache(
        self,
        datum: Dict[str, Any],
        *,
        scene_id: str,
        frame_id: int,
        encoder_input_sha256: str,
    ) -> Tuple[Dict[str, Any], float, bool]:
        """Run Boxer query with an exact, one-keyframe encoder cache.

        The cache is enabled explicitly by configuration.  A scene/frame hit
        is accepted only when the digest of RGB, metric depth, both camera
        matrices, pose and sampled depth points is identical.  Otherwise the
        ordinary prepare/encode/query path is executed and replaces the
        bounded cache entry.
        """

        if not self.config.cache_image_features:
            raise LiftingContractError("Boxer image-feature cache is disabled")
        cache_key = (str(scene_id), int(frame_id))
        fingerprint = str(encoder_input_sha256)
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise LiftingContractError("invalid Boxer encoder-input fingerprint")

        with _preserve_rng_state(include_cuda=self.device == "cuda"):
            if self.device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            if self.device == "mps":
                context = nullcontext()
            elif self.config.precision == "bfloat16":
                context = torch.autocast(
                    device_type=self.device,
                    dtype=torch.bfloat16,
                )
            else:
                context = nullcontext()
            with torch.no_grad(), context:
                inputs = self.model.prepare_inputs(datum)
                cache = self._feature_cache
                cache_hit = bool(
                    cache is not None
                    and cache["key"] == cache_key
                    and cache["fingerprint"] == fingerprint
                )
                if cache_hit:
                    inputs["T_world_voxel0"] = cache["T_world_voxel0"]
                    encoded = dict(cache["encoded"])
                    self._stats["feature_cache_hits"] += 1
                else:
                    encoded = self.model.encode(inputs)
                    if "T_world_voxel0" not in inputs or "input_enc" not in encoded:
                        raise LiftingContractError(
                            "Boxer encode did not expose reusable image features"
                        )
                    self._feature_cache = {
                        "key": cache_key,
                        "fingerprint": fingerprint,
                        "T_world_voxel0": inputs["T_world_voxel0"],
                        "encoded": dict(encoded),
                    }
                    self._stats["feature_cache_misses"] += 1
                output = self.model.query(inputs, encoded)
            if self.device == "cuda":
                torch.cuda.synchronize()
            runtime_ms = (time.perf_counter() - started) * 1000.0
        return output, runtime_ms, cache_hit

    def _forward_with_feature_cache(
        self,
        datum: Dict[str, Any],
        camera_to_world: np.ndarray,
        *,
        scene_id: str,
        frame_id: int,
        encoder_input_sha256: str,
    ) -> Dict[str, torch.Tensor]:
        output, runtime_ms, cache_hit = self.forward_raw_with_feature_cache(
            datum,
            scene_id=scene_id,
            frame_id=frame_id,
            encoder_input_sha256=encoder_input_sha256,
        )
        obbs_world = output["obbs_pr_w"][0].cpu()
        centers_world = obbs_world.bb3_center_world.float()
        dims = obbs_world.bb3_diagonal.float()
        rotations_world_object = obbs_world.T_world_object.R.float()
        pose_tensor = torch.from_numpy(camera_to_world).float()
        xyz_dims, rotations_camera_object = boxer_world_to_boxfusion_camera(
            centers_world,
            dims,
            rotations_world_object,
            pose_tensor,
        )
        rotations_so3 = project_rotations_to_so3(rotations_camera_object)
        rotation_correction_max_abs = (
            float(
                torch.max(
                    torch.abs(rotations_so3 - rotations_camera_object)
                ).item()
            )
            if rotations_camera_object.shape[0]
            else 0.0
        )
        return {
            "xyz_dims": xyz_dims,
            "rotations": rotations_so3,
            "rotation_correction_max_abs": torch.tensor(
                rotation_correction_max_abs,
                dtype=torch.float64,
            ),
            "confidence": obbs_world.prob.squeeze(-1).float(),
            "logvar": output["obbs_pr_logvar"][0].detach().float().cpu(),
            "raw_params": output["obbs_pr_params"][0].detach().float().cpu(),
            "runtime_ms": torch.tensor(runtime_ms, dtype=torch.float64),
            "feature_cache_hit": torch.tensor(cache_hit, dtype=torch.bool),
        }

    @staticmethod
    def _validate_geometry(
        xyz_dims: torch.Tensor,
        rotations: torch.Tensor,
        expected_count: int,
    ) -> None:
        if xyz_dims.shape != (expected_count, 6):
            raise LiftingContractError(
                "Boxer row-count/shape mismatch: "
                f"expected={(expected_count, 6)}, actual={tuple(xyz_dims.shape)}"
            )
        if rotations.shape != (expected_count, 3, 3):
            raise LiftingContractError(
                "Boxer rotation shape mismatch: "
                f"expected={(expected_count, 3, 3)}, "
                f"actual={tuple(rotations.shape)}"
            )
        if not torch.all(torch.isfinite(xyz_dims)):
            raise LiftingContractError("Boxer produced NaN/Inf geometry")
        if not torch.all(torch.isfinite(rotations)):
            raise LiftingContractError("Boxer produced NaN/Inf rotations")
        if torch.any(xyz_dims[:, 3:] <= 0.0):
            raise LiftingContractError("Boxer produced a non-positive dimension")
        if torch.any(xyz_dims[:, 2] <= 1e-4):
            raise LiftingContractError(
                "Boxer produced a box center behind the optical camera"
            )

        identity = torch.eye(3).expand(expected_count, -1, -1)
        orthogonality = torch.matmul(
            rotations.transpose(-1, -2),
            rotations,
        )
        max_orthogonality_error = (
            (orthogonality - identity).abs().max().item()
            if expected_count
            else 0.0
        )
        determinants = torch.linalg.det(rotations)
        if (
            max_orthogonality_error > 5e-3
            or torch.any(torch.abs(determinants - 1.0) > 5e-3)
        ):
            raise LiftingContractError(
                "Boxer produced an invalid rotation: "
                f"max_orthogonality_error={max_orthogonality_error:.6f}, "
                f"det_range="
                f"{determinants.min().item():.6f}/"
                f"{determinants.max().item():.6f}"
            )

    def _diagnostic_path(self, scene_id: str) -> str:
        safe_scene = scene_id.replace(os.sep, "_")
        return os.path.join(
            self.config.diagnostics_dir,
            f"{safe_scene}_boxer_lifting.jsonl",
        )

    def _write_diagnostic(self, scene_id: str, row: Dict[str, Any]) -> None:
        os.makedirs(self.config.diagnostics_dir, exist_ok=True)
        path = self._diagnostic_path(scene_id)
        mode = "a"
        if scene_id not in self._diag_initialized:
            mode = "w"
            self._diag_initialized.add(scene_id)
        with open(path, mode, encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )

    def apply(
        self,
        pred_instances: Any,
        *,
        image: Any,
        depth: Any,
        image_K: Any,
        depth_K: Any,
        camera_to_world: Any,
        scene_id: str,
        frame_id: int,
        attempt_id: str = "primary",
    ) -> Any:
        """Observe or replace one keyframe's CuTR 3D lifting."""

        attempt_id = str(attempt_id)
        if attempt_id not in ("primary", "retry", "mvpr", "gsa"):
            raise ValueError(
                "Boxer attempt_id must be primary, retry, mvpr, or gsa, received "
                f"{attempt_id!r}"
            )
        expected_count = len(pred_instances)
        protected_before = protected_proposal_hashes(pred_instances)
        cutr_geometry_hash = geometry_hash(pred_instances)
        cutr_xyz_dims = (
            pred_instances.pred_boxes_3d.tensor.detach().float().cpu().clone()
        )
        cutr_rotations = (
            pred_instances.pred_boxes_3d.R.detach().float().cpu().clone()
        )
        input_projected_center_hash = _field_hash(
            pred_instances,
            "pred_proj_xy",
        )
        boxes_xyxy = pred_instances.pred_boxes.detach().float().cpu()
        detector_scores = pred_instances.scores.detach().float().cpu()
        scores_hash = _field_hash(pred_instances, "scores")

        if expected_count == 0:
            self._write_diagnostic(
                scene_id,
                {
                    "schema": "boxfusion.boxer_lifting.frame.v1",
                    "scene_id": scene_id,
                    "frame_id": int(frame_id),
                    "attempt_id": attempt_id,
                    "mode": self.mode,
                    "apply_stage": self.apply_stage,
                    "mutation_enabled": self.mutation_enabled,
                    "count": 0,
                    "applied_count": 0,
                    "protected_hashes": protected_before,
                    "cutr_geometry_sha256": cutr_geometry_hash,
                },
            )
            self._stats["calls"] += 1
            return pred_instances

        datum, metadata = self._make_datum(
            image=image,
            depth=depth,
            boxes_xyxy=boxes_xyxy,
            image_K=image_K,
            depth_K=depth_K,
            camera_to_world=camera_to_world,
            scene_id=scene_id,
            frame_id=frame_id,
        )
        try:
            if self.config.cache_image_features:
                prediction = self._forward_with_feature_cache(
                    datum,
                    metadata["pose_np"],
                    scene_id=scene_id,
                    frame_id=frame_id,
                    encoder_input_sha256=metadata["encoder_input_sha256"],
                )
            else:
                prediction = self._forward(
                    datum,
                    metadata["pose_np"],
                )
        except Exception as error:
            self._write_diagnostic(
                scene_id,
                {
                    "schema": "boxfusion.boxer_lifting.frame.v1",
                    "scene_id": scene_id,
                    "frame_id": int(frame_id),
                    "attempt_id": attempt_id,
                    "mode": self.mode,
                    "apply_stage": self.apply_stage,
                    "mutation_enabled": self.mutation_enabled,
                    "count": expected_count,
                    "applied_count": 0,
                    "protected_hashes": protected_before,
                    "input_pred_proj_xy_sha256": (
                        input_projected_center_hash
                    ),
                    "scores_sha256": scores_hash,
                    "boxes_2d_sha256": _sha256_array(boxes_xyxy),
                    "image_sha256": _sha256_array(metadata["image_np"]),
                    "depth_sha256": _sha256_array(metadata["depth_np"]),
                    "camera_to_world_sha256": _sha256_array(
                        metadata["pose_np"]
                    ),
                    "cutr_geometry_sha256": cutr_geometry_hash,
                    "boxer_checkpoint_sha256": self._checkpoint_sha256,
                    "boxer_commit": self.config.expected_commit,
                    "failure_stage": "forward",
                    "failure": f"{type(error).__name__}: {error}",
                },
            )
            raise
        xyz_dims = prediction["xyz_dims"]
        rotations = prediction["rotations"]
        try:
            self._validate_geometry(xyz_dims, rotations, expected_count)
        except Exception as error:
            self._write_diagnostic(
                scene_id,
                {
                    "schema": "boxfusion.boxer_lifting.frame.v1",
                    "scene_id": scene_id,
                    "frame_id": int(frame_id),
                    "attempt_id": attempt_id,
                    "mode": self.mode,
                    "apply_stage": self.apply_stage,
                    "mutation_enabled": self.mutation_enabled,
                    "count": expected_count,
                    "applied_count": 0,
                    "protected_hashes": protected_before,
                    "input_pred_proj_xy_sha256": (
                        input_projected_center_hash
                    ),
                    "scores_sha256": scores_hash,
                    "boxes_2d_sha256": _sha256_array(boxes_xyxy),
                    "image_sha256": _sha256_array(metadata["image_np"]),
                    "depth_sha256": _sha256_array(metadata["depth_np"]),
                    "camera_to_world_sha256": _sha256_array(
                        metadata["pose_np"]
                    ),
                    "cutr_geometry_sha256": cutr_geometry_hash,
                    "failed_boxer_geometry_sha256": _sha256_bytes(
                        (
                            _sha256_array(xyz_dims)
                            + "|"
                            + _sha256_array(rotations)
                        ).encode("ascii")
                    ),
                    "boxer_checkpoint_sha256": self._checkpoint_sha256,
                    "boxer_commit": self.config.expected_commit,
                    "failure_stage": "geometry_validation",
                    "failure": f"{type(error).__name__}: {error}",
                },
            )
            raise

        image_K_tensor = torch.from_numpy(metadata["image_K_np"]).float()
        projected_centers = project_centers(
            xyz_dims[:, :3],
            image_K_tensor,
        )
        boxer_geometry_sha256 = _sha256_bytes(
            (
                _sha256_array(xyz_dims)
                + "|"
                + _sha256_array(rotations)
            ).encode("ascii")
        )

        applied_count = 0
        if self.mutation_enabled:
            output_device = pred_instances.pred_boxes_3d.tensor.device
            pred_instances.pred_boxes_3d = GeneralInstance3DBoxes(
                xyz_dims.to(output_device),
                rotations.to(output_device),
            )
            # In the controlled post-filter ablation, the exact CuTR 2D
            # proposal state is frozen and only the 3D OBB changes.  The
            # complete pre-filter replacement also replaces the projected
            # center because the unchanged UV rule must consume Boxer.
            if self.apply_stage == "pre_filter":
                pred_instances.pred_proj_xy = projected_centers.to(
                    output_device
                )
            applied_count = expected_count

        protected_after = protected_proposal_hashes(pred_instances)
        if protected_before != protected_after:
            raise LiftingContractError(
                "Boxer changed a protected CuTR proposal field: "
                f"before={protected_before}, after={protected_after}"
            )
        if len(pred_instances) != expected_count:
            raise LiftingContractError(
                "Boxer changed the proposal count: "
                f"before={expected_count}, after={len(pred_instances)}"
            )
        if (
            self.apply_stage == "post_filter"
            and _field_hash(pred_instances, "pred_proj_xy")
            != input_projected_center_hash
        ):
            raise LiftingContractError(
                "Controlled post-filter Boxer changed pred_proj_xy"
            )

        runtime_ms = float(prediction["runtime_ms"].item())
        self._stats["calls"] += 1
        self._stats["proposals"] += expected_count
        self._stats["applied"] += applied_count
        self._stats["runtime_ms"].append(runtime_ms)
        if not self.mutation_enabled:
            self._stats["observer_calls"] += 1

        self._write_diagnostic(
            scene_id,
            {
                "schema": "boxfusion.boxer_lifting.frame.v1",
                "scene_id": scene_id,
                "frame_id": int(frame_id),
                "attempt_id": attempt_id,
                "mode": self.mode,
                "apply_stage": self.apply_stage,
                "mutation_enabled": self.mutation_enabled,
                "use_sdp": self.config.use_sdp,
                "sdp_seed": metadata["sdp_seed"],
                "count": expected_count,
                "applied_count": applied_count,
                "protected_hashes": protected_before,
                "input_pred_proj_xy_sha256": input_projected_center_hash,
                "scores_sha256": scores_hash,
                "boxes_2d_sha256": _sha256_array(boxes_xyxy),
                "image_sha256": _sha256_array(metadata["image_np"]),
                "depth_sha256": _sha256_array(metadata["depth_np"]),
                "image_intrinsics_sha256": _sha256_array(
                    metadata["image_K_np"]
                ),
                "depth_intrinsics_sha256": _sha256_array(
                    metadata["depth_K_np"]
                ),
                "camera_to_world_sha256": _sha256_array(
                    metadata["pose_np"]
                ),
                "camera_to_world": metadata["pose_np"].tolist(),
                "cutr_geometry_sha256": cutr_geometry_hash,
                "boxer_geometry_sha256": boxer_geometry_sha256,
                "boxer_checkpoint_sha256": self._checkpoint_sha256,
                "boxer_commit": self.config.expected_commit,
                "projected_center_replaced": bool(
                    self.mutation_enabled
                    and self.apply_stage == "pre_filter"
                ),
                "runtime_ms": runtime_ms,
                "feature_cache_enabled": self.config.cache_image_features,
                "feature_cache_hit": bool(
                    prediction.get(
                        "feature_cache_hit", torch.tensor(False)
                    ).item()
                ),
                "rotation_correction_max_abs": float(
                    prediction["rotation_correction_max_abs"].item()
                ),
                "confidence": prediction["confidence"].tolist(),
                "logvar": prediction["logvar"].reshape(-1).tolist(),
                "input_boxes_xyxy": boxes_xyxy.tolist(),
                "detector_scores": detector_scores.tolist(),
                "boxer_boxes_xxyy": metadata["boxer_boxes"].tolist(),
                "cutr_xyz_dims_camera": cutr_xyz_dims.tolist(),
                "cutr_rotation_camera_object": cutr_rotations.tolist(),
                "output_xyz_dims_camera": xyz_dims.tolist(),
                "output_rotation_camera_object": rotations.tolist(),
                "raw_params_voxel": prediction["raw_params"].tolist(),
            },
        )
        return pred_instances

    def summary(self) -> str:
        runtimes = np.asarray(self._stats["runtime_ms"], dtype=np.float64)
        if runtimes.size:
            median_ms = float(np.median(runtimes))
            p95_ms = float(np.quantile(runtimes, 0.95))
        else:
            median_ms = math.nan
            p95_ms = math.nan
        return (
            "Boxer lifting summary | "
            f"mode={self.mode}, stage={self.apply_stage}, "
            f"calls={self._stats['calls']}, "
            f"proposals={self._stats['proposals']}, "
            f"applied={self._stats['applied']}, "
            f"observer_calls={self._stats['observer_calls']}, "
            f"feature_cache_hits/misses="
            f"{self._stats['feature_cache_hits']}/"
            f"{self._stats['feature_cache_misses']}, "
            f"runtime_median/p95_ms={median_ms:.3f}/{p95_ms:.3f}"
        )


def build_lifting_adapter(
    cfg: Dict[str, Any],
    *,
    device: str,
    code_root: str,
) -> Optional[BoxerLiftingAdapter]:
    lifting_cfg = cfg.get("lifting", {})
    backend = str(lifting_cfg.get("backend", "cutr")).lower()
    if backend == "cutr":
        return None
    if backend != "boxer":
        raise ValueError(
            f"Unsupported lifting.backend={backend}; expected cutr or boxer"
        )
    boxer_cfg = BoxerLiftingConfig.from_mapping(
        lifting_cfg.get("boxer", {}),
        code_root=code_root,
    )
    return BoxerLiftingAdapter(boxer_cfg, device=device)

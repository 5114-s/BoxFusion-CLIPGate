"""Fail-closed dense DINOv3 encoder matching Selective Boxer preprocessing."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch


BOXER_DINO_MODEL = "dinov3_vits16plus"
BOXER_DINO_FILENAME = (
    "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
)
BOXER_DINO_SHA256 = (
    "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea"
)
BOXER_OFFICIAL_COMMIT = "1f86542dc342a4b1d474c87c97c5d1d6566d9148"


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _preserve_rng(include_cuda: bool):
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


@dataclass(frozen=True)
class BoxerDINOv3Config:
    official_root: str
    expected_commit: str = BOXER_OFFICIAL_COMMIT
    checkpoint_sha256: str = BOXER_DINO_SHA256
    input_height: int = 960
    input_width: int = 960
    precision: str = "bfloat16"
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        if self.device not in {"cuda", "cpu", "mps"}:
            raise ValueError("device must be cuda, cpu, or mps")
        if min(self.input_height, self.input_width) < 16:
            raise ValueError("DINO input dimensions must be at least 16")
        if self.input_height % 16 or self.input_width % 16:
            raise ValueError("DINO input dimensions must be multiples of 16")
        for name, value in (
            ("expected_commit", self.expected_commit),
            ("checkpoint_sha256", self.checkpoint_sha256),
        ):
            if len(value) != (40 if name == "expected_commit" else 64):
                raise ValueError(f"invalid {name}")


class BoxerDINOv3DenseEncoder:
    """Lazy feature-only DINO encoder with exact G0 resize semantics.

    The online version should reuse ``BoxerNet.encode(...)[\"dino0\"]``.
    This standalone observer intentionally performs the same stretched
    bilinear resize and tensor conversion so the resulting sidecar is a fair
    predictor of that zero-extra-backbone integration.
    """

    def __init__(self, config: BoxerDINOv3Config):
        self.config = config
        self.model: Any = None
        self._verified_commit: str | None = None
        self._checkpoint_sha256: str | None = None

    @property
    def checkpoint_path(self) -> Path:
        return (
            Path(self.config.official_root)
            / "ckpts"
            / BOXER_DINO_FILENAME
        )

    @property
    def verified_commit(self) -> str:
        self._verify_assets()
        assert self._verified_commit is not None
        return self._verified_commit

    @property
    def verified_checkpoint_sha256(self) -> str:
        self._verify_assets()
        assert self._checkpoint_sha256 is not None
        return self._checkpoint_sha256

    def _verify_assets(self) -> None:
        if self._verified_commit is not None:
            return
        root = Path(self.config.official_root).resolve()
        if not (root / "boxernet" / "dinov3_wrapper.py").is_file():
            raise FileNotFoundError("official Boxer DINO wrapper is absent")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError("could not verify official Boxer commit") from error
        if commit != self.config.expected_commit:
            raise RuntimeError(
                "official Boxer commit mismatch: "
                f"expected={self.config.expected_commit}, actual={commit}"
            )
        checkpoint_sha = sha256_file(self.checkpoint_path)
        if checkpoint_sha != self.config.checkpoint_sha256:
            raise RuntimeError(
                "DINO checkpoint SHA256 mismatch: "
                f"expected={self.config.checkpoint_sha256}, actual={checkpoint_sha}"
            )
        self._verified_commit = commit
        self._checkpoint_sha256 = checkpoint_sha

    def _load(self) -> None:
        if self.model is not None:
            return
        self._verify_assets()
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if (
            self.config.device == "cuda"
            and self.config.precision == "bfloat16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("bfloat16 is unsupported on this CUDA device")
        root = str(Path(self.config.official_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        loaded_utils = sys.modules.get("utils")
        if loaded_utils is not None:
            origin = os.path.abspath(str(getattr(loaded_utils, "__file__", "")))
            if origin and not origin.startswith(root + os.sep):
                raise RuntimeError(
                    "conflicting top-level utils package was imported before Boxer: "
                    + origin
                )
        with _preserve_rng(include_cuda=self.config.device == "cuda"):
            from boxernet.dinov3_wrapper import DinoV3Wrapper

            self.model = DinoV3Wrapper(BOXER_DINO_MODEL)
            self.model = self.model.to(self.config.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _uint8_rgb(image: object) -> np.ndarray:
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[2] < 3:
            raise ValueError("RGB image must have shape [H,W,>=3]")
        value = value[..., :3]
        if np.issubdtype(value.dtype, np.floating):
            if not np.isfinite(value).all():
                raise ValueError("RGB image contains NaN or Inf")
            scale = 255.0 if (float(value.max()) if value.size else 0.0) <= 1.0 + 1e-5 else 1.0
            value = np.rint(np.clip(value * scale, 0.0, 255.0))
        else:
            value = np.clip(value, 0, 255)
        return np.ascontiguousarray(value, dtype=np.uint8)

    def __call__(self, image: object) -> np.ndarray:
        import cv2

        self._load()
        rgb = self._uint8_rgb(image)
        resized = cv2.resize(
            rgb,
            (self.config.input_width, self.config.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = (
            torch.from_numpy(np.ascontiguousarray(resized))
            .permute(2, 0, 1)
            .float()[None]
            .div_(255.0)
            .to(self.config.device)
        )
        if self.config.device == "mps" or self.config.precision == "float32":
            context = nullcontext()
        else:
            context = torch.autocast(
                device_type=self.config.device, dtype=torch.bfloat16
            )
        with torch.inference_mode(), context:
            features = self.model(tensor)
        if features.ndim != 4 or features.shape[0] != 1:
            raise RuntimeError("DINO wrapper returned an unexpected feature shape")
        result = features[0].float().cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("DINO produced non-finite features")
        return np.ascontiguousarray(result, dtype=np.float32)


__all__ = [
    "BOXER_DINO_FILENAME",
    "BOXER_DINO_MODEL",
    "BOXER_DINO_SHA256",
    "BOXER_OFFICIAL_COMMIT",
    "BoxerDINOv3Config",
    "BoxerDINOv3DenseEncoder",
    "sha256_file",
]

"""Compatibility adapter for the official SPGroup3D grouping backbone.

Only ``BiResNet`` and ``SSG`` are loaded.  The ScanNet-18 ``SPHead`` is never
constructed, so this observer cannot alter BoxFusion's CLIP semantics.  The
adapter keeps the official source files read-only and supplies small API
shims for legacy MMDetection3D and torch-scatter interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np


OFFICIAL_COMMIT = "181283547323d3bd54d0e9f58baf0cd413ccc107"
OFFICIAL_CHECKPOINT_SHA256 = "cabd9f88da3bf41dcb8aa46696d47aaa7c94913a3086f9404374a0b149714edf"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class _Registry:
    def register_module(self, module=None, **_kwargs):
        def decorate(value):
            return value
        return decorate(module) if module is not None else decorate


def _scatter_mean(src, index, dim=0):
    import torch

    if dim != 0 or index.ndim != 1 or src.shape[0] != index.shape[0]:
        raise NotImplementedError("SPGroup adapter supports scatter_mean(dim=0) only")
    groups = int(index.max().item()) + 1 if index.numel() else 0
    result = torch.zeros((groups, *src.shape[1:]), dtype=src.dtype, device=src.device)
    result.index_add_(0, index.long(), src)
    count = torch.bincount(index.long(), minlength=groups).to(src.dtype)
    count = count.reshape((groups,) + (1,) * (src.ndim - 1)).clamp_min_(1)
    return result / count


def _scatter_max(src, index, dim=0):
    import torch

    if dim != 0 or index.ndim != 1 or src.shape[0] != index.shape[0]:
        raise NotImplementedError("SPGroup adapter supports scatter_max(dim=0) only")
    groups = int(index.max().item()) + 1 if index.numel() else 0
    shape = (groups, *src.shape[1:])
    result = torch.full(shape, -torch.inf, dtype=src.dtype, device=src.device)
    expanded = index.long().reshape((-1,) + (1,) * (src.ndim - 1)).expand_as(src)
    result.scatter_reduce_(0, expanded, src, reduce="amax", include_self=True)
    # The official SSG discards argmax; retain the legacy tuple contract.
    argmax = torch.full(shape, -1, dtype=torch.long, device=src.device)
    return result, argmax


def _knn(k, xyz, center_xyz=None, transposed=False):
    """Legacy mmdet3d ``knn`` contract backed by deterministic cKDTree."""
    import torch
    from scipy.spatial import cKDTree

    if transposed:
        xyz = xyz.transpose(1, 2)
        if center_xyz is not None:
            center_xyz = center_xyz.transpose(1, 2)
    queries = xyz if center_xyz is None else center_xyz
    if xyz.ndim != 3 or queries.ndim != 3 or xyz.shape[0] != queries.shape[0]:
        raise ValueError("knn expects [B,N,3] support and [B,M,3] query")
    parts = []
    for support, query in zip(xyz.detach().cpu().numpy(), queries.detach().cpu().numpy()):
        if support.shape[0] == 0:
            raise ValueError("knn support is empty")
        effective = min(int(k), int(support.shape[0]))
        _, indices = cKDTree(support).query(query, k=effective, workers=1)
        indices = np.asarray(indices, dtype=np.int64)
        if effective == 1:
            indices = indices[:, None]
        if effective < int(k):
            indices = np.pad(indices, ((0, 0), (0, int(k) - effective)), mode="edge")
        parts.append(indices.T)
    return torch.as_tensor(np.stack(parts), dtype=torch.long, device=xyz.device)


def _install_legacy_import_shims() -> dict[str, types.ModuleType | None]:
    """Install only names imported by the two official source files."""
    import torch.nn as nn

    names: dict[str, types.ModuleType] = {}
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.BACKBONES = _Registry()
    names["mmdet.models"] = mmdet_models

    mmcv_runner = types.ModuleType("mmcv.runner")
    mmcv_runner.BaseModule = nn.Module
    names["mmcv.runner"] = mmcv_runner

    builder = types.ModuleType("mmdet3d.models.builder")
    builder.VOXEL_ENCODERS = _Registry()
    names["mmdet3d.models.builder"] = builder

    scatter = types.ModuleType("torch_scatter")
    scatter.scatter_mean = _scatter_mean
    scatter.scatter_max = _scatter_max
    names["torch_scatter"] = scatter

    ops = types.ModuleType("mmdet3d.ops")
    ops.knn = _knn
    names["mmdet3d.ops"] = ops

    previous: dict[str, types.ModuleType | None] = {}
    for name, module in names.items():
        previous[name] = sys.modules.get(name)
        sys.modules[name] = module
    return previous


def _restore_imports(previous: dict[str, types.ModuleType | None]) -> None:
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official SPGroup3D source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_official_classes(official_root: Path):
    source = official_root.resolve() / "projects" / "spgroup"
    previous = _install_legacy_import_shims()
    try:
        backbone_module = _load_module(
            "_boxfusion_official_spgroup_biresnet", source / "biresnet.py"
        )
        encoder_module = _load_module(
            "_boxfusion_official_spgroup_encoder", source / "Superpoint_encoder.py"
        )
    finally:
        _restore_imports(previous)
    return backbone_module.BiResNet, encoder_module.SSG


@dataclass(frozen=True)
class SPGroupFeatures:
    superpoint_ids: np.ndarray
    centers_aligned: np.ndarray
    embeddings: np.ndarray
    vote_offsets: np.ndarray
    vote_offset_std: np.ndarray
    voxel_counts: np.ndarray


class OfficialSPGroupEncoder:
    """Inference-only wrapper for official pretrained grouping layers."""

    def __init__(
        self,
        official_root: str | Path,
        checkpoint: str | Path,
        *,
        expected_checkpoint_sha256: str = OFFICIAL_CHECKPOINT_SHA256,
        device: str = "cuda",
    ) -> None:
        import torch

        self.official_root = Path(official_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)
        actual_sha = _sha256_file(self.checkpoint)
        if actual_sha != expected_checkpoint_sha256:
            raise ValueError(f"SPGroup3D checkpoint SHA256 mismatch: {actual_sha}")
        self.checkpoint_sha256 = actual_sha
        BiResNet, SSG = load_official_classes(self.official_root)
        self.backbone = BiResNet(in_channels=3, out_channels=64)
        self.encoder = SSG(
            in_channels=64,
            local_k=8,
            voxel_size=0.02,
            latter_voxel_size=0.04,
            feat_channels=(64, 128, 128),
            with_xyz=True,
            with_distance=False,
            with_cluster_center=False,
            with_superpoint_center=True,
            mode="max",
        )
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
            raise ValueError("SPGroup3D checkpoint has no state_dict")
        state = payload["state_dict"]
        backbone_state = {
            key[len("backbone."):]: value
            for key, value in state.items() if key.startswith("backbone.")
        }
        encoder_state = {
            key[len("voxel_encoder."):]: value
            for key, value in state.items() if key.startswith("voxel_encoder.")
        }
        self.backbone.load_state_dict(backbone_state, strict=True)
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.backbone.eval().to(self.device)
        self.encoder.eval().to(self.device)

    def encode(
        self,
        vertices_aligned: np.ndarray,
        colors: np.ndarray,
        superpoint_ids: np.ndarray,
    ) -> SPGroupFeatures:
        import MinkowskiEngine as ME
        import torch
        from scipy.spatial import cKDTree

        xyz = np.asarray(vertices_aligned, dtype=np.float32)
        rgb = np.asarray(colors, dtype=np.float32)
        ids = np.asarray(superpoint_ids, dtype=np.int64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or ids.shape != (len(xyz),):
            raise ValueError("invalid SPGroup3D scene arrays")
        if not np.isfinite(xyz).all() or not np.isfinite(rgb).all() or len(xyz) == 0:
            raise ValueError("SPGroup3D scene arrays are empty or non-finite")
        points = torch.from_numpy(np.concatenate((xyz, rgb), axis=1)).to(self.device)
        superpoints = torch.from_numpy(ids).to(self.device)
        coordinates, features = ME.utils.batch_sparse_collate(
            [(points[:, :3] / 0.02, points[:, 3:])], device=self.device
        )
        sparse = ME.SparseTensor(coordinates=coordinates, features=features)
        with torch.inference_mode():
            backbone = self.backbone(sparse)
            grouped = self.encoder(backbone, [points], [superpoints], [{}])
        embeddings = grouped["voxel_feats"].detach().float().cpu().numpy()
        centers = grouped["voxel_coods"][:, 1:].detach().float().cpu().numpy()
        original_map = grouped["orgin_superpoints"][0].detach().long()
        raw_vote_offsets = grouped["vote_offsets"][0].detach().float()
        raw_vote_points = grouped["vote_voxel_points"][0].detach().float().cpu().numpy()
        nearest = cKDTree(xyz).query(raw_vote_points, k=1, workers=1)[1]
        present_ids = np.unique(ids[np.asarray(nearest, dtype=np.int64)])
        if len(present_ids) != len(embeddings):
            raise RuntimeError(
                "official compact superpoint order could not be reconstructed: "
                f"{len(present_ids)} != {len(embeddings)}"
            )
        groups = len(present_ids)
        offset_mean = _scatter_mean(raw_vote_offsets, original_map, dim=0)
        centered = raw_vote_offsets - offset_mean[original_map]
        offset_var = _scatter_mean(centered.square(), original_map, dim=0)
        counts = torch.bincount(original_map, minlength=groups)
        result = SPGroupFeatures(
            superpoint_ids=present_ids.astype(np.int32),
            centers_aligned=np.asarray(centers, dtype=np.float32),
            embeddings=np.asarray(embeddings, dtype=np.float32),
            vote_offsets=offset_mean.cpu().numpy().astype(np.float32),
            vote_offset_std=offset_var.sqrt().cpu().numpy().astype(np.float32),
            voxel_counts=counts.cpu().numpy().astype(np.int32),
        )
        if result.embeddings.shape != (groups, 390):
            raise RuntimeError(f"unexpected official grouping feature shape: {result.embeddings.shape}")
        return result

#!/usr/bin/env python3
"""Validate paired G0-distribution active/identity sparse checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise TypeError(f"invalid sparse checkpoint: {path}")
    return payload["metadata"], sha256_file(path)


def audit(active_path: Path, identity_path: Path, manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "boxfusion.g0_sgcdet_train_collection_manifest.v1":
        raise ValueError("collection manifest schema mismatch")
    active, active_sha = load_checkpoint(active_path)
    identity, identity_sha = load_checkpoint(identity_path)
    for name, metadata, expected_identity in (
        ("active", active, False),
        ("identity", identity, True),
    ):
        if bool(metadata.get("identity_only")) != expected_identity:
            raise ValueError(f"{name} identity_only mismatch")
        if bool(metadata.get("scene_leakage", True)):
            raise ValueError(f"{name} reports scene leakage")
        if metadata.get("collection_manifest_schema") != manifest["schema"]:
            raise ValueError(f"{name} collection manifest schema mismatch")
        if metadata.get("collection_manifest_sha256") != manifest_sha:
            raise ValueError(f"{name} collection manifest SHA mismatch")
    for key in (
        "training_dataset_sha256",
        "source_joint_dataset_sha256",
        "source_b5_dataset_sha256",
        "training_scene_sha256",
        "validation_scene_sha256",
        "forbidden_scene_sha256",
    ):
        if active.get(key) != identity.get(key):
            raise ValueError(f"paired checkpoint provenance differs: {key}")

    if active.get("selection_metric") != "ap50_proxy":
        raise ValueError("active checkpoint was not selected AP50-first")
    proxy = float(active.get("best_validation_local_net_tp50_proxy", 0.0))
    validation = active.get("validation_metrics")
    if not isinstance(validation, dict):
        raise TypeError("active validation_metrics are absent")
    cross_success = float(validation.get("cross50_success_count", 0.0))
    eligible = float(validation.get("eligible_matched_count", 0.0))
    drop = float(validation.get("drop50_count", 0.0))
    drop_rate = drop / eligible if eligible > 0.0 else 1.0
    if proxy <= 0.0 or cross_success < 1.0:
        raise ValueError("active checkpoint has no held-out-train AP50 crossing")
    if drop_rate > 0.01 + 1e-12:
        raise ValueError(f"active held-out drop50 rate is unsafe: {drop_rate}")

    return {
        "schema": "boxfusion.g0_sgcdet_retrained_checkpoint_audit.v1",
        "ok": True,
        "active_checkpoint": str(active_path.resolve()),
        "active_checkpoint_sha256": active_sha,
        "identity_checkpoint": str(identity_path.resolve()),
        "identity_checkpoint_sha256": identity_sha,
        "collection_manifest_sha256": manifest_sha,
        "training_dataset_sha256": active["training_dataset_sha256"],
        "best_epoch": int(active["best_epoch"]),
        "validation_tp50_proxy": proxy,
        "validation_cross50_success_count": cross_success,
        "validation_drop50_rate": drop_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.active, args.identity, args.manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

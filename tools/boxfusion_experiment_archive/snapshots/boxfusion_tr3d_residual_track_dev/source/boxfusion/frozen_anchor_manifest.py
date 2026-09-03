"""Generic content-addressed prediction anchor manifests.

The original genuine-TR3D route froze only the B6 prediction tree.  This
module keeps that legacy manifest readable while supporting stronger anchors
whose provenance may contain several checkpoints, configs, and launchers.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from boxfusion.frozen_b6_manifest import (
    FROZEN_B6_MANIFEST_SCHEMA,
    read_scene_list,
    sha256_file,
    verify_frozen_b6_manifest,
)


FROZEN_ANCHOR_MANIFEST_SCHEMA = "boxfusion.frozen_anchor_manifest.v1"
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _tree_hash(rows: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, value in sorted(rows.items()):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _prediction_files(root: Path, scenes: tuple[str, ...]) -> dict[str, str]:
    paths = {
        path.name[: -len("_boxes.pkl")]: path
        for path in root.glob("*_boxes.pkl")
        if path.is_file()
    }
    if set(paths) != set(scenes):
        raise ValueError(
            "frozen anchor prediction set disagrees with the scene list; "
            f"missing={sorted(set(scenes)-set(paths))[:8]}, "
            f"extra={sorted(set(paths)-set(scenes))[:8]}"
        )
    return {
        f"{scene}_boxes.pkl": sha256_file(paths[scene])
        for scene in sorted(scenes)
    }


def _metrics(values: Mapping[str, Any]) -> dict[str, float]:
    expected = {"AP15", "AP25", "AP50"}
    if set(values) != expected:
        raise ValueError("anchor metrics must contain AP15/AP25/AP50 exactly")
    result = {key: float(values[key]) for key in sorted(expected)}
    if not all(math.isfinite(value) and 0 <= value <= 100 for value in result.values()):
        raise ValueError("anchor metrics must be finite percentages in [0,100]")
    return result


def _metadata(values: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = {} if values is None else dict(values)
    # Reject non-JSON values and normalize tuples/numpy-free scalar variants.
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("anchor metadata must be a JSON mapping")
    return decoded


def build_frozen_anchor_manifest(
    *,
    anchor_name: str,
    reference_root: str | Path,
    scene_list: str | Path,
    artifacts: Mapping[str, str | Path],
    anchor_metrics_percent: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    required_scene_count: int | None = 100,
) -> dict[str, Any]:
    if _SAFE_NAME_RE.fullmatch(anchor_name) is None:
        raise ValueError(f"invalid anchor name: {anchor_name!r}")
    root = Path(reference_root).resolve()
    scene_list_path = Path(scene_list).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not scene_list_path.is_file():
        raise FileNotFoundError(scene_list_path)
    scenes = read_scene_list(scene_list_path)
    if required_scene_count is not None and len(scenes) != required_scene_count:
        raise ValueError(
            f"expected {required_scene_count} scenes, found {len(scenes)}"
        )
    if not artifacts:
        raise ValueError("at least one anchor provenance artifact is required")
    artifact_rows: dict[str, dict[str, str]] = {}
    for name, raw_path in sorted(artifacts.items()):
        if _SAFE_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"invalid artifact name: {name!r}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact_rows[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    predictions = _prediction_files(root, scenes)
    artifact_hashes = {
        name: row["sha256"] for name, row in artifact_rows.items()
    }
    return {
        "schema": FROZEN_ANCHOR_MANIFEST_SCHEMA,
        "anchor_name": anchor_name,
        "reference_result_root": str(root),
        "scene_list_path": str(scene_list_path),
        "scene_list_sha256": sha256_file(scene_list_path),
        "scene_count": len(scenes),
        "scene_ids": list(scenes),
        "prediction_files": predictions,
        "prediction_tree_sha256": _tree_hash(predictions),
        "artifacts": artifact_rows,
        "artifact_tree_sha256": _tree_hash(artifact_hashes),
        "anchor_metrics_percent": _metrics(anchor_metrics_percent),
        "metadata": _metadata(metadata),
    }


def write_frozen_anchor_manifest(
    path: str | Path, payload: Mapping[str, Any]
) -> str:
    target = Path(path)
    encoded = json.dumps(
        dict(payload), indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"immutable anchor manifest disagrees: {target}")
        return "verified"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(encoded, encoding="utf-8")
    return "created"


def _normalize_legacy_b6(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact_hashes = {"quality_checkpoint": payload["checkpoint_sha256"]}
    return {
        **dict(payload),
        "anchor_name": "B6",
        "artifacts": {
            "quality_checkpoint": {
                "path": payload["checkpoint_path"],
                "sha256": payload["checkpoint_sha256"],
            }
        },
        "artifact_tree_sha256": _tree_hash(artifact_hashes),
        "metadata": {"legacy_schema": FROZEN_B6_MANIFEST_SCHEMA},
    }


def verify_frozen_anchor_manifest(
    path: str | Path,
    *,
    required_scene_count: int | None = None,
) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("frozen anchor manifest must contain a mapping")
    if payload.get("schema") == FROZEN_B6_MANIFEST_SCHEMA:
        return _normalize_legacy_b6(
            verify_frozen_b6_manifest(
                manifest_path, required_scene_count=required_scene_count
            )
        )
    if payload.get("schema") != FROZEN_ANCHOR_MANIFEST_SCHEMA:
        raise ValueError("unsupported frozen anchor manifest schema")
    artifact_paths = {
        name: row["path"] for name, row in payload["artifacts"].items()
    }
    expected = build_frozen_anchor_manifest(
        anchor_name=payload["anchor_name"],
        reference_root=payload["reference_result_root"],
        scene_list=payload["scene_list_path"],
        artifacts=artifact_paths,
        anchor_metrics_percent=payload["anchor_metrics_percent"],
        metadata=payload.get("metadata", {}),
        required_scene_count=(
            int(payload["scene_count"])
            if required_scene_count is None
            else required_scene_count
        ),
    )
    if dict(payload) != expected:
        raise ValueError("frozen anchor files or manifest metadata changed")
    for row in payload["artifacts"].values():
        if _SHA_RE.fullmatch(str(row["sha256"])) is None:
            raise ValueError("invalid anchor artifact SHA256")
    return expected

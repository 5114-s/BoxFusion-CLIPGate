"""Content-addressed manifest for the frozen 100-scene B6 anchor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


FROZEN_B6_MANIFEST_SCHEMA = "boxfusion.frozen_b6_manifest.v1"
_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_scene_list(path: str | Path) -> tuple[str, ...]:
    rows = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list must be non-empty and unique")
    invalid = [row for row in rows if _SCENE_RE.fullmatch(row) is None]
    if invalid:
        raise ValueError(f"invalid scene id: {invalid[0]!r}")
    return rows


def build_frozen_b6_manifest(
    *,
    reference_root: str | Path,
    checkpoint: str | Path,
    scene_list: str | Path,
    required_scene_count: int | None = 100,
) -> dict[str, Any]:
    root = Path(reference_root).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    scene_list_path = Path(scene_list).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not scene_list_path.is_file():
        raise FileNotFoundError(scene_list_path)
    scenes = read_scene_list(scene_list_path)
    if required_scene_count is not None and len(scenes) != required_scene_count:
        raise ValueError(
            f"expected {required_scene_count} scenes, found {len(scenes)}"
        )
    paths = {
        path.name[: -len("_boxes.pkl")]: path
        for path in root.glob("*_boxes.pkl")
        if path.is_file()
    }
    if set(paths) != set(scenes):
        raise ValueError(
            "frozen B6 prediction set disagrees with the scene list; "
            f"missing={sorted(set(scenes)-set(paths))[:8]}, "
            f"extra={sorted(set(paths)-set(scenes))[:8]}"
        )
    predictions = {
        f"{scene}_boxes.pkl": sha256_file(paths[scene])
        for scene in sorted(scenes)
    }
    return {
        "schema": FROZEN_B6_MANIFEST_SCHEMA,
        "reference_result_root": str(root),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "scene_list_path": str(scene_list_path),
        "scene_list_sha256": sha256_file(scene_list_path),
        "scene_count": len(scenes),
        "scene_ids": list(scenes),
        "prediction_files": predictions,
        "prediction_tree_sha256": _prediction_tree_hash(predictions),
        "anchor_metrics_percent": {
            "AP15": 40.0434,
            "AP25": 33.5492,
            "AP50": 12.1613,
        },
    }


def _prediction_tree_hash(predictions: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, value in sorted(predictions.items()):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def write_frozen_b6_manifest(
    path: str | Path, payload: Mapping[str, Any]
) -> str:
    target = Path(path)
    encoded = json.dumps(
        dict(payload), indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(
                f"immutable frozen-B6 manifest disagrees: {target}"
            )
        return "verified"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(encoded, encoding="utf-8")
    return "created"


def verify_frozen_b6_manifest(
    path: str | Path,
    *,
    required_scene_count: int | None = None,
) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("frozen B6 manifest must contain a mapping")
    if payload.get("schema") != FROZEN_B6_MANIFEST_SCHEMA:
        raise ValueError("unsupported frozen B6 manifest schema")
    expected = build_frozen_b6_manifest(
        reference_root=payload["reference_result_root"],
        checkpoint=payload["checkpoint_path"],
        scene_list=payload["scene_list_path"],
        required_scene_count=(
            int(payload["scene_count"])
            if required_scene_count is None
            else required_scene_count
        ),
    )
    if dict(payload) != expected:
        raise ValueError("frozen B6 files or manifest metadata changed")
    if _SHA_RE.fullmatch(str(payload["checkpoint_sha256"])) is None:
        raise ValueError("invalid frozen B6 checkpoint SHA256")
    return expected

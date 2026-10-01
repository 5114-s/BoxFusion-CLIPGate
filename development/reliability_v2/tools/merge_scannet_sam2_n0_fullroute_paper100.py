#!/usr/bin/env python3
"""Seal the two completed N0 full-route paper100 shards into one receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCENE_SCHEMA = "boxfusion.scannet_sam2_n0_fullroute_paper100.scene.v1"
SHARD_SCHEMA = "boxfusion.scannet_sam2_n0_fullroute_paper100.shard.v1"
SCHEMA = "boxfusion.scannet_sam2_n0_fullroute_paper100.merge.v1"
PROTOCOL_ID = "F0-F3-N0-SAM2-TSDF-MV3DIS-DUAL-OBB-SHADOW-PAPER100-V1"
DEFAULT_ROOT = ROOT / "logs/scannet_sam2_n0_fullroute_paper100_score05"


class MergeError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MergeError(f"missing regular input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MergeError(f"input is not a JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise MergeError(f"refusing to overwrite merge receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--scene-list", type=Path, default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.root / "final/N0_FULLROUTE_PAPER100.json"
    scenes = [line.strip() for line in args.scene_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise MergeError("paper100 scene list must contain exactly 100 unique scenes")
    shard_paths = [args.root / "shards/shard_00_of_02.json", args.root / "shards/shard_01_of_02.json"]
    shards = [_json(path) for path in shard_paths]
    if any(
        row.get("schema") != SHARD_SCHEMA
        or row.get("protocol_id") != PROTOCOL_ID
        or row.get("complete") is not True
        or row.get("shard_count") != 2
        or row.get("shard_index") != index
        for index, row in enumerate(shards)
    ):
        raise MergeError("shard receipt contract differs")
    rows = []
    totals = {
        "confirmed_track_count": 0,
        "observation_count": 0,
        "valid_geometry_track_count": 0,
        "invalid_geometry_track_count": 0,
        "invalid_lift_observation_count": 0,
    }
    for scene_index, scene in enumerate(scenes):
        path = args.root / "scenes" / f"{scene}.json"
        payload = _json(path)
        contracts = payload.get("contracts")
        counts = payload.get("counts")
        arrays = payload.get("arrays")
        if (
            payload.get("schema") != SCENE_SCHEMA
            or payload.get("protocol_id") != PROTOCOL_ID
            or payload.get("scene_id") != scene
            or payload.get("complete") is not True
            or not isinstance(contracts, dict)
            or not isinstance(counts, dict)
            or not isinstance(arrays, dict)
            or contracts.get("ground_truth_access") is not False
            or contracts.get("annotation_access") is not False
            or contracts.get("evaluator_access") is not False
            or contracts.get("native_output_mutation") is not False
            or contracts.get("training") is not False
            or contracts.get("query_before_commit") is not True
            or contracts.get("maximum_lookahead_observations") != 0
        ):
            raise MergeError(f"scene contract differs: {scene}")
        arrays_path = Path(str(arrays.get("path", "")))
        if not arrays_path.is_file() or _sha(arrays_path) != arrays.get("sha256"):
            raise MergeError(f"scene arrays identity differs: {scene}")
        for key in totals:
            value = counts.get(key)
            if type(value) is not int or value < 0:
                raise MergeError(f"scene count differs: {scene}:{key}")
            totals[key] += value
        rows.append(
            {
                "scene_id": scene,
                "scene_index": scene_index,
                "sidecar": {"path": os.fspath(path.resolve()), "sha256": _sha(path)},
                "arrays": {"path": os.fspath(arrays_path.resolve()), "sha256": arrays["sha256"]},
                "counts": counts,
            }
        )
    receipt = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "overall_pass": True,
        "scene_count": 100,
        "scene_order": scenes,
        "totals": totals,
        "shards": [
            {"path": os.fspath(path.resolve()), "sha256": _sha(path)} for path in shard_paths
        ],
        "scenes": rows,
        "contracts": {
            "shadow_only": True,
            "birth_enabled": False,
            "native_output_mutation": False,
            "ground_truth_access": False,
            "training": False,
            "current_and_past_only": True,
            "query_before_commit": True,
        },
        "conclusion_guardrail": "Merged shadow evidence has no AP; oracle must run only after this seal.",
    }
    _write(output, receipt)
    print(json.dumps({"out": os.fspath(output), "totals": totals}, sort_keys=True))


if __name__ == "__main__":
    main()

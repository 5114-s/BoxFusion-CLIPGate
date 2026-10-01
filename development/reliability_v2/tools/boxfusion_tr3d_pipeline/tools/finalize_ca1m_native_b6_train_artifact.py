#!/usr/bin/env python3
"""Validate and seal one train-only CA-1M collection scene or collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import stat
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "boxfusion.ca1m_native_b6_train_scene_completion.v1"
COLLECTION_SCHEMA = "boxfusion.ca1m_native_b6_train_collection.v1"
DIAGNOSTIC_SCHEMA = "boxfusion.ca1m_native_b6_observer.v1"
CACHE_SCHEMA = "boxfusion.cutr_postfilter_cache.v2"


def regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(mode) or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_rows(path: Path) -> int:
    regular(path, "prediction")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
        if handle.read(1):
            raise ValueError(f"trailing prediction bytes: {path}")
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], list):
        raise ValueError(f"invalid prediction batch: {path}")
    for index, row in enumerate(payload[0]):
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(f"invalid prediction row {index}: {path}")
        label, corners, score = row
        corners = np.asarray(corners)
        if (
            int(label) != 0
            or corners.shape != (8, 3)
            or not np.isfinite(corners).all()
            or not np.isfinite(float(score))
        ):
            raise ValueError(f"invalid prediction values at row {index}: {path}")
    return len(payload[0])


def create_or_verify(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() or path.is_symlink():
        regular(path, "completion artifact")
        if path.read_bytes() != data:
            raise ValueError(f"existing completion artifact drifted: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def record(args: argparse.Namespace) -> dict[str, Any]:
    rows = prediction_rows(args.prediction)
    regular(args.log, "record log")
    text = args.log.read_text(errors="replace")
    if "Finalized immutable CuTR proposal cache:" not in text:
        raise ValueError("record log lacks cache-finalization marker")
    if "eval mAP:" in text:
        raise ValueError("evaluation marker found in train-only collection log")
    manifest_path = args.cache_scene_root / "manifest.json"
    regular(manifest_path, "proposal cache manifest")
    cache = json.loads(manifest_path.read_text())
    if (
        cache.get("schema") != CACHE_SCHEMA
        or cache.get("scene_id") != args.scene
        or cache.get("namespace") != args.cache_namespace
        or cache.get("producer_fingerprint") != args.proposal_fingerprint
        or cache.get("prediction_file") != args.prediction.name
        or cache.get("prediction_sha256") != sha256(args.prediction)
    ):
        raise ValueError("proposal cache manifest contract mismatch")
    frame_hashes: list[dict[str, Any]] = []
    for item in cache.get("records", []):
        frame = args.cache_scene_root / f"frame_{int(item['frame_id']):06d}.pt"
        regular(frame, "cached proposal frame")
        digest = sha256(frame)
        if digest != item.get("sha256"):
            raise ValueError(f"cached proposal frame hash mismatch: {frame}")
        frame_hashes.append({"path": frame.name, "sha256": digest})
    if not frame_hashes or len(frame_hashes) != int(cache.get("record_count", -1)):
        raise ValueError("empty/incomplete proposal cache records")
    return {
        "schema": SCHEMA,
        "phase": "cutr_record",
        "scene_id": args.scene,
        "complete": True,
        "train_only": True,
        "evaluation_invoked": False,
        "validation_ground_truth_access": False,
        "prediction_rows": rows,
        "proposal_fingerprint": args.proposal_fingerprint,
        "artifacts": {
            "prediction": {"path": str(args.prediction.resolve()), "sha256": sha256(args.prediction)},
            "cache_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
            "cache_frames": frame_hashes,
            "log": {"path": str(args.log.resolve()), "sha256": sha256(args.log)},
        },
    }


def observer(args: argparse.Namespace) -> dict[str, Any]:
    rows = prediction_rows(args.prediction)
    anchor_rows = prediction_rows(args.anchor)
    if rows != anchor_rows or sha256(args.prediction) != sha256(args.anchor):
        raise ValueError("observer prediction is not byte-identical to same-run anchor")
    regular(args.log, "observer log")
    text = args.log.read_text(errors="replace")
    if "CA-1M native B6 observer summary" not in text:
        raise ValueError("observer log lacks completion marker")
    if "eval mAP:" in text:
        raise ValueError("evaluation marker found in train-only collection log")
    regular(args.diagnostic, "native-B6 diagnostic")
    with np.load(args.diagnostic, allow_pickle=False) as payload:
        scalar = lambda name: np.asarray(payload[name]).item()
        if (
            scalar("schema") != DIAGNOSTIC_SCHEMA
            or str(scalar("scene_id")) != args.scene
            or bool(scalar("complete")) is not True
            or bool(scalar("observer_only")) is not True
            or bool(scalar("mutation_enabled")) is not False
            or int(scalar("applied_count")) != 0
            or bool(scalar("ground_truth_access")) is not False
            or len(payload["result_indices"]) != rows
            or not np.array_equal(payload["result_indices"], np.arange(rows))
        ):
            raise ValueError("native-B6 diagnostic safety/mapping contract mismatch")
    regular(args.boxer, "Selective Boxer diagnostic")
    boxer_lines = [json.loads(line) for line in args.boxer.read_text().splitlines() if line.strip()]
    if not boxer_lines or any(str(row.get("scene_id")) != args.scene for row in boxer_lines):
        raise ValueError("Selective Boxer diagnostic is empty or scene-mismatched")
    return {
        "schema": SCHEMA,
        "phase": "g0_native_b6_observer",
        "scene_id": args.scene,
        "complete": True,
        "train_only": True,
        "evaluation_invoked": False,
        "validation_ground_truth_access": False,
        "output_mutation_authorized": False,
        "prediction_rows": rows,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in {
                "prediction": args.prediction,
                "same_run_anchor": args.anchor,
                "native_b6_diagnostic": args.diagnostic,
                "boxer_diagnostic": args.boxer,
                "log": args.log,
            }.items()
        },
    }


def collection(args: argparse.Namespace) -> dict[str, Any]:
    regular(args.subset_manifest, "frozen subset manifest")
    subset = json.loads(args.subset_manifest.read_text())
    scenes = [str(row["scene_id"]) for row in subset["entries"]]
    rows = []
    digest = hashlib.sha256()
    for scene in scenes:
        record_path = args.record_completion_root / f"{scene}.json"
        observer_path = args.observer_completion_root / f"{scene}.json"
        regular(record_path, "record completion")
        regular(observer_path, "observer completion")
        record_value = json.loads(record_path.read_text())
        observer_value = json.loads(observer_path.read_text())
        if (
            record_value.get("scene_id") != scene
            or record_value.get("phase") != "cutr_record"
            or observer_value.get("scene_id") != scene
            or observer_value.get("phase") != "g0_native_b6_observer"
            or record_value.get("evaluation_invoked") is not False
            or observer_value.get("evaluation_invoked") is not False
        ):
            raise ValueError(f"invalid scene completion contract: {scene}")
        record_sha, observer_sha = sha256(record_path), sha256(observer_path)
        digest.update(f"{scene}\t{record_sha}\t{observer_sha}\n".encode())
        rows.append({"scene_id": scene, "record_completion_sha256": record_sha,
                     "observer_completion_sha256": observer_sha})
    actual_record = {path.stem for path in args.record_completion_root.glob("*.json")}
    actual_observer = {path.stem for path in args.observer_completion_root.glob("*.json")}
    if actual_record != set(scenes) or actual_observer != set(scenes):
        raise ValueError("completion roots contain missing or extra scene artifacts")
    return {
        "schema": COLLECTION_SCHEMA,
        "complete": True,
        "train_only": True,
        "evaluation_invoked": False,
        "validation_ground_truth_access": False,
        "scene_count": len(rows),
        "scene_ids_sha256": subset["selection"]["scene_ids_sha256"],
        "subset_manifest_sha256": sha256(args.subset_manifest),
        "completion_collection_sha256": digest.hexdigest(),
        "scenes": rows,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    one = sub.add_parser("record")
    one.add_argument("--scene", required=True)
    one.add_argument("--prediction", type=Path, required=True)
    one.add_argument("--cache-scene-root", type=Path, required=True)
    one.add_argument("--cache-namespace", required=True)
    one.add_argument("--proposal-fingerprint", required=True)
    one.add_argument("--log", type=Path, required=True)
    one.add_argument("--output", type=Path, required=True)
    two = sub.add_parser("observer")
    two.add_argument("--scene", required=True)
    two.add_argument("--prediction", type=Path, required=True)
    two.add_argument("--anchor", type=Path, required=True)
    two.add_argument("--diagnostic", type=Path, required=True)
    two.add_argument("--boxer", type=Path, required=True)
    two.add_argument("--log", type=Path, required=True)
    two.add_argument("--output", type=Path, required=True)
    three = sub.add_parser("collection")
    three.add_argument("--subset-manifest", type=Path, required=True)
    three.add_argument("--record-completion-root", type=Path, required=True)
    three.add_argument("--observer-completion-root", type=Path, required=True)
    three.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "record":
        value = record(args)
    elif args.command == "observer":
        value = observer(args)
    else:
        value = collection(args)
    create_or_verify(args.output, value)
    print(json.dumps({
        "schema": value["schema"], "complete": value["complete"],
        "scene_id": value.get("scene_id"), "scene_count": value.get("scene_count"),
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

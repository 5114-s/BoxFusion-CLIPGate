#!/usr/bin/env python3
"""Seal GT-free canonical-103 native-B6 artifacts without evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from finalize_ca1m_native_b6_train_artifact import (
    create_or_verify,
    observer as validate_observer,
    record as validate_record,
    regular,
    sha256,
)


SCENE_SCHEMA = "boxfusion.ca1m_native_b6_canonical103_scene_completion.v1"
COLLECTION_SCHEMA = "boxfusion.ca1m_native_b6_canonical103_collection.v1"


def canonicalize(value: dict) -> dict:
    value = dict(value)
    value["schema"] = SCENE_SCHEMA
    value.pop("train_only", None)
    value.pop("validation_ground_truth_access", None)
    value.update({
        "dataset_split": "official_validation_canonical103",
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "training_authorized": False,
    })
    return value


def read_scenes(path: Path) -> list[str]:
    regular(path, "frozen canonical103 scene list")
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 103 or len(rows) != len(set(rows)) or any(not row.isdigit() for row in rows):
        raise ValueError("canonical scene list must contain 103 unique numeric IDs")
    return rows


def collection(args: argparse.Namespace) -> dict:
    scenes = read_scenes(args.scene_list)
    digest = hashlib.sha256()
    rows = []
    for scene in scenes:
        record_path = args.record_completion_root / f"{scene}.json"
        observer_path = args.observer_completion_root / f"{scene}.json"
        regular(record_path, "record completion")
        regular(observer_path, "observer completion")
        record_value = json.loads(record_path.read_text())
        observer_value = json.loads(observer_path.read_text())
        for value, phase in ((record_value, "cutr_record"), (observer_value, "g0_native_b6_observer")):
            if (
                value.get("schema") != SCENE_SCHEMA
                or value.get("scene_id") != scene
                or value.get("phase") != phase
                or value.get("ground_truth_access") is not False
                or value.get("evaluation_invoked") is not False
                or value.get("training_authorized") is not False
            ):
                raise ValueError(f"invalid canonical scene completion: {scene}/{phase}")
        record_sha, observer_sha = sha256(record_path), sha256(observer_path)
        digest.update(f"{scene}\t{record_sha}\t{observer_sha}\n".encode())
        rows.append({
            "scene_id": scene,
            "record_completion_sha256": record_sha,
            "observer_completion_sha256": observer_sha,
        })
    expected = set(scenes)
    if (
        {path.stem for path in args.record_completion_root.glob("*.json")} != expected
        or {path.stem for path in args.observer_completion_root.glob("*.json")} != expected
    ):
        raise ValueError("completion roots contain missing or extra scenes")
    return {
        "schema": COLLECTION_SCHEMA,
        "complete": True,
        "dataset_split": "official_validation_canonical103",
        "scene_count": 103,
        "scene_list_sha256": sha256(args.scene_list),
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "training_authorized": False,
        "same_run_anchor_byte_identity_required": True,
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
    three.add_argument("--scene-list", type=Path, required=True)
    three.add_argument("--record-completion-root", type=Path, required=True)
    three.add_argument("--observer-completion-root", type=Path, required=True)
    three.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "record":
        value = canonicalize(validate_record(args))
    elif args.command == "observer":
        value = canonicalize(validate_observer(args))
    else:
        value = collection(args)
    create_or_verify(args.output, value)
    print(json.dumps({
        "schema": value["schema"],
        "complete": value["complete"],
        "scene_id": value.get("scene_id"),
        "scene_count": value.get("scene_count"),
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

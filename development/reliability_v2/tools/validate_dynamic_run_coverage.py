#!/usr/bin/env python3
"""Fail-closed scene coverage validation for dynamic BoxFusion runs.

This tool deliberately checks *coverage only*.  It must run before any dynamic
metric code so that a missing prediction cannot be interpreted as an empty
prediction set.  Two artifact layouts are supported:

* evaluator pickles: ``<scene>_boxes.pkl`` in paired persistent/current roots;
* event ledgers: ``<scene>_events.jsonl`` (also ``<scene>.events.jsonl`` or
  ``<scene>.jsonl``) in one or more event roots.

Unrelated files, including ``*.dual_state.json`` sidecars, are ignored.  Every
recognized artifact must be a non-empty regular file, and every supplied root
must contain exactly one artifact for every manifest scene and no artifacts for
other scenes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


PREDICTION_SUFFIXES = ("_boxes.pkl",)
EVENT_SUFFIXES = ("_events.jsonl", ".events.jsonl", ".jsonl")
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
REPORT_SCHEMA = "boxfusion.dynamic_run_coverage.v1"
DYNAMIC_EVENT_SCHEMA = "boxfusion.causal_dynamic_branch.v1"


class DynamicCoverageError(ValueError):
    """Raised when a manifest or artifact root violates exact coverage."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    counts = Counter(key for key, _ in pairs)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise DynamicCoverageError(
            "manifest contains duplicate JSON keys: " + ",".join(duplicates)
        )
    return dict(pairs)


def _validate_scene_ids(scene_ids: Iterable[Any]) -> tuple[str, ...]:
    scenes: list[str] = []
    for index, value in enumerate(scene_ids):
        if not isinstance(value, str):
            raise DynamicCoverageError(
                f"manifest scene {index} is not a string: {value!r}"
            )
        if not value or value != value.strip():
            raise DynamicCoverageError(
                f"manifest scene {index} is empty or has surrounding whitespace"
            )
        if value in {".", ".."} or not SCENE_ID_RE.fullmatch(value):
            raise DynamicCoverageError(f"unsafe manifest scene ID: {value!r}")
        scenes.append(value)
    if not scenes:
        raise DynamicCoverageError("manifest contains no scenes")
    counts = Counter(scenes)
    duplicates = sorted(scene for scene, count in counts.items() if count > 1)
    if duplicates:
        raise DynamicCoverageError(
            "manifest contains duplicate scene IDs: " + ",".join(duplicates)
        )
    return tuple(scenes)


def read_manifest_scenes(path: str | Path) -> tuple[str, ...]:
    """Read scene IDs from either the existing map or a v2 ``scenes`` field."""

    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise DynamicCoverageError(
            f"manifest is not a regular file: {manifest_path}"
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    except DynamicCoverageError:
        raise
    except Exception as exc:
        raise DynamicCoverageError(
            f"could not decode manifest {manifest_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise DynamicCoverageError("manifest root must be a JSON object")

    if "scenes" not in payload:
        # Current data_dyn manifests are direct {scene_id: event-spec} maps.
        return _validate_scene_ids(payload.keys())

    scene_payload = payload["scenes"]
    if isinstance(scene_payload, dict):
        return _validate_scene_ids(scene_payload.keys())
    if not isinstance(scene_payload, list):
        raise DynamicCoverageError("manifest 'scenes' must be a list or object")

    scene_ids: list[Any] = []
    for index, entry in enumerate(scene_payload):
        if isinstance(entry, str):
            scene_ids.append(entry)
        elif isinstance(entry, dict) and "scene_id" in entry:
            scene_ids.append(entry["scene_id"])
        else:
            raise DynamicCoverageError(
                f"manifest scenes[{index}] must be a string or contain scene_id"
            )
    return _validate_scene_ids(scene_ids)


def _scene_from_artifact_name(name: str, suffixes: Sequence[str]) -> str | None:
    # Longest-first makes ``scene_events.jsonl`` resolve to ``scene`` rather
    # than ``scene_events`` through the generic ``.jsonl`` suffix.
    for suffix in sorted(set(suffixes), key=len, reverse=True):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _complete_dynamic_event_ledger(path: Path, scene_id: str) -> bool:
    final_record = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    final_record = json.loads(line)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(final_record, dict)
        and final_record.get("schema") == DYNAMIC_EVENT_SCHEMA
        and final_record.get("type") == "summary"
        and final_record.get("scene_id") == scene_id
    )


def _validate_artifact_root(
    root: str | Path,
    expected_scenes: Sequence[str],
    *,
    kind: str,
    suffixes: Sequence[str],
) -> dict[str, Any]:
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise DynamicCoverageError(f"{kind} root is not a directory: {resolved}")

    paths_by_scene: dict[str, list[Path]] = {}
    invalid_paths: list[str] = []
    for path in resolved.iterdir():
        scene = _scene_from_artifact_name(path.name, suffixes)
        if scene is None:
            continue
        if not scene or scene in {".", ".."} or not SCENE_ID_RE.fullmatch(scene):
            invalid_paths.append(path.name)
            continue
        if not path.is_file() or path.stat().st_size <= 0:
            invalid_paths.append(path.name)
            continue
        paths_by_scene.setdefault(scene, []).append(path)

    duplicates = sorted(
        scene for scene, paths in paths_by_scene.items() if len(paths) != 1
    )
    incomplete_events = []
    if kind.startswith("events"):
        incomplete_events = sorted(
            scene
            for scene, paths in paths_by_scene.items()
            if len(paths) == 1
            and not _complete_dynamic_event_ledger(paths[0], scene)
        )
    discovered = set(paths_by_scene)
    expected = set(expected_scenes)
    missing = sorted(expected - discovered)
    extra = sorted(discovered - expected)
    if invalid_paths or duplicates or missing or extra or incomplete_events:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        if duplicates:
            duplicate_details = [
                scene
                + "["
                + ",".join(sorted(path.name for path in paths_by_scene[scene]))
                + "]"
                for scene in duplicates
            ]
            details.append("duplicate=" + ",".join(duplicate_details))
        if invalid_paths:
            details.append("invalid_or_empty=" + ",".join(sorted(invalid_paths)))
        if incomplete_events:
            details.append(
                "incomplete_event=" + ",".join(incomplete_events)
            )
        raise DynamicCoverageError(
            f"{kind} scene coverage mismatch for {resolved}: " + "; ".join(details)
        )

    files = {
        scene: paths_by_scene[scene][0]
        for scene in expected_scenes
    }
    return {
        "kind": kind,
        "root": str(resolved),
        "scene_count": len(files),
        "suffixes": list(suffixes),
        "files": {
            scene: {
                "name": files[scene].name,
                "size_bytes": files[scene].stat().st_size,
                "sha256": _sha256(files[scene]),
            }
            for scene in expected_scenes
        },
    }


def validate_dynamic_run_coverage(
    manifest: str | Path,
    *,
    persistent_root: str | Path | None = None,
    current_root: str | Path | None = None,
    event_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Validate exact scene coverage and return a sealed coverage receipt."""

    if (persistent_root is None) != (current_root is None):
        raise DynamicCoverageError(
            "persistent-root and current-root must be supplied together"
        )
    if persistent_root is None and not event_roots:
        raise DynamicCoverageError(
            "supply paired persistent/current roots or at least one event root"
        )

    manifest_path = Path(manifest).resolve()
    scenes = read_manifest_scenes(manifest_path)
    roots: list[dict[str, Any]] = []
    resolved_roots: list[Path] = []
    if persistent_root is not None:
        resolved_roots.extend(
            (Path(persistent_root).resolve(), Path(current_root).resolve())
        )
        roots.append(
            _validate_artifact_root(
                persistent_root,
                scenes,
                kind="persistent",
                suffixes=PREDICTION_SUFFIXES,
            )
        )
        roots.append(
            _validate_artifact_root(
                current_root,
                scenes,
                kind="current",
                suffixes=PREDICTION_SUFFIXES,
            )
        )
    for index, root in enumerate(event_roots):
        resolved_roots.append(Path(root).resolve())
        roots.append(
            _validate_artifact_root(
                root,
                scenes,
                kind=f"events[{index}]",
                suffixes=EVENT_SUFFIXES,
            )
        )

    counts = Counter(resolved_roots)
    repeated_roots = sorted(str(root) for root, count in counts.items() if count > 1)
    if repeated_roots:
        raise DynamicCoverageError(
            "artifact roots must resolve to distinct directories: "
            + ",".join(repeated_roots)
        )

    return {
        "schema": REPORT_SCHEMA,
        "passed": True,
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "scene_count": len(scenes),
            "scenes": list(scenes),
        },
        "artifact_roots": roots,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--persistent-root")
    parser.add_argument("--current-root")
    parser.add_argument(
        "--event-root",
        action="append",
        default=[],
        help="event-ledger root; may be supplied more than once",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = validate_dynamic_run_coverage(
            args.manifest,
            persistent_root=args.persistent_root,
            current_root=args.current_root,
            event_roots=args.event_root,
        )
    except DynamicCoverageError as exc:
        json.dump(
            {"schema": REPORT_SCHEMA, "passed": False, "error": str(exc)},
            sys.stderr,
            ensure_ascii=False,
            sort_keys=True,
        )
        sys.stderr.write("\n")
        return 2
    json.dump(report, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

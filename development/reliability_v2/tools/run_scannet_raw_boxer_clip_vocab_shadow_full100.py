#!/usr/bin/env python3
"""Frozen CLIP vocabulary shadow for Raw-Boxer Past3 receipts.

This program is deliberately output-inert: it reads the already sealed Raw
OWLv2/Boxer CSVs and the birth-v2 receipt audit, then writes a JSON sidecar.
It never reads ScanNet annotations, evaluator state, or prediction pickles.

The Raw Boxer ``instance`` column is the original per-frame OWL detection
index (BoxerNet assigns ``arange(M)`` before its 3D-confidence filter).  That
makes ``(time_ns, instance)`` an exact key into ``owl_2dbbs.csv`` and avoids
heuristic label matching when several detections share a name.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import cv2
import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "boxfusion.scannet_raw_boxer_clip_vocab_shadow_full100.v1"
EXPECTED_RECEIPT_SCHEMA = (
    "boxfusion.scannet_raw_boxer_past3_birth_full100.v2_m50"
)
EXPECTED_VOCABULARY_SIZE = 473
EXPECTED_FEATURE_DIMENSION = 1024

# Fixed before any GT/evaluator access.  These receipts have passed every
# birth-v2-M50 gate before terminal self-NMS and the per-scene cap.  Including
# the self-NMS/cap rejects lets the active integrator insert CLIP immediately
# before those two terminal operations without changing any earlier gate.
PREFILTER_ELIGIBLE_V2_DECISIONS = ("accepted", "scene_cap", "self_nms")
PREFILTER = {
    "eligible_v2_decisions": list(PREFILTER_ELIGIBLE_V2_DECISIONS),
    "insertion_point": "after_v2_geometry_native_gates_before_self_nms_and_scene_cap",
}

# Existing native BoxFusion 473-way vocabulary entries that are compatible
# with the ScanNet18 object vocabulary.  No new text encoder or prompt is used;
# indices resolve directly against data/panoptic_categories_nomerge.txt and the
# already frozen data/class_features.pt matrix.
TARGET_GROUP_ALIASES = {
    "cabinet_or_bookshelf": ("Cabinet/shelf",),
    "bed": ("bed",),
    "chair": ("chair",),
    "sofa": ("couch",),
    "table": ("Coffee Table", "dining-table"),
    "door": ("door",),
    "window": ("window", "window-other"),
    "picture": ("Picture/Frame",),
    "counter": ("counter",),
    "desk": ("Desk",),
    "curtain": ("curtain",),
    "refrigerator": ("refrigerator",),
    "toilet": ("toilet",),
    "sink": ("sink",),
    "bathtub": ("Bathtub",),
    "garbage_bin": ("Trash bin Can",),
}

# Raw OWLv2 exports WordNet/LVIS-style names, while the cached native CLIP
# vocabulary above uses display strings.  Keep these namespaces separate:
# this table only collapses the original OWL name to a ScanNet-compatible
# group and never selects a CLIP feature row.
OWL_TARGET_GROUP_ALIASES = {
    "cabinet_or_bookshelf": (
        "cabinet",
        "cupboard",
        "file cabinet",
        "bookshelf",
        "bookcase",
        "shelf",
    ),
    "bed": ("bed",),
    "chair": ("chair", "armchair"),
    "sofa": ("sofa", "couch"),
    "table": ("table", "dining table", "coffee table"),
    "door": ("door",),
    "window": ("window",),
    "picture": ("picture", "painting", "poster"),
    "counter": ("counter",),
    "desk": ("desk",),
    "curtain": ("curtain", "shower curtain"),
    "refrigerator": ("refrigerator", "fridge"),
    "toilet": ("toilet", "urinal"),
    "sink": ("sink",),
    "bathtub": ("bathtub", "bath tub"),
    "garbage_bin": (
        "trash can",
        "garbage bin",
        "waste bin",
        "wastebasket",
        "recycling bin",
    ),
}
EXPECTED_TARGET_INDICES = [
    55,
    69,
    150,
    156,
    183,
    244,
    285,
    302,
    311,
    343,
    346,
    356,
    366,
    402,
    406,
    461,
    469,
    470,
]

GATE_POLICY = {
    "name": "clip_vocab_gate_v1",
    "owl_exact_alias_all_three_same_group": True,
    "clip_all_vocab_top1_target_votes_gte": 2,
    "clip_top1_same_group_as_owl_votes_gte": 2,
    "median_target_best_cosine_gte": 0.20,
    "median_target_non_target_margin_gte": -0.01,
}

RAW_REQUIRED_COLUMNS = {
    "time_ns",
    "name",
    "instance",
    "sem_id",
    "prob",
}
OWL_REQUIRED_COLUMNS = {
    "time_ns",
    "frame_id",
    "img_width",
    "img_height",
    "x1",
    "y1",
    "x2",
    "y2",
    "name",
    "sem_id",
    "prob",
}


class ClipVocabShadowError(RuntimeError):
    """Raised when a sealed input or shadow invariant is violated."""


def _normalize_owl_name(value: object) -> str:
    """Normalize separators/parentheticals without fuzzy semantic matching."""

    name = str(value).strip().lower()
    name = re.sub(r"[_\-/()\[\]{}]+", " ", name)
    return " ".join(name.split())


def _owl_alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for group, aliases in OWL_TARGET_GROUP_ALIASES.items():
        for alias in aliases:
            normalized = _normalize_owl_name(alias)
            previous = lookup.get(normalized)
            if previous is not None and previous != group:
                raise ClipVocabShadowError(
                    f"OWL alias belongs to multiple groups: {normalized!r}"
                )
            lookup[normalized] = group
    return lookup


def _resolve_owl_target_group(value: object) -> str | None:
    return _owl_alias_lookup().get(_normalize_owl_name(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ClipVocabShadowError(f"cannot read JSON: {path}") from error


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ClipVocabShadowError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ClipVocabShadowError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ClipVocabShadowError(f"{label} must be finite")
    return result


def _csv_integer(value: object, label: str, *, minimum: int = 0) -> int:
    number = _finite_number(value, label)
    if not number.is_integer() or number < minimum:
        raise ClipVocabShadowError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return int(number)


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ClipVocabShadowError(f"CSV must be a regular non-symlink file: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = sorted(required - columns)
            if missing:
                raise ClipVocabShadowError(
                    f"CSV is missing columns {missing}: {path}"
                )
            return [dict(row) for row in reader]
    except (OSError, csv.Error) as error:
        raise ClipVocabShadowError(f"cannot read CSV: {path}") from error


def _prefilter_reason(receipt: Mapping[str, Any]) -> str | None:
    """Return the frozen prior-v2 rejection, or ``None`` at the CLIP insertion."""

    decision = receipt.get("decision")
    if not isinstance(decision, str) or not decision:
        raise ClipVocabShadowError("receipt decision must be a non-empty string")
    if decision in PREFILTER_ELIGIBLE_V2_DECISIONS:
        return None
    return f"prior_v2_{decision}"


def _resolve_target_indices(
    vocabulary: Sequence[str],
) -> tuple[dict[str, list[int]], list[int], list[int]]:
    if len(vocabulary) != EXPECTED_VOCABULARY_SIZE:
        raise ClipVocabShadowError(
            f"expected {EXPECTED_VOCABULARY_SIZE} vocabulary rows, "
            f"found {len(vocabulary)}"
        )
    name_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(vocabulary):
        if not name:
            raise ClipVocabShadowError(f"empty vocabulary row at index {index}")
        name_to_indices[name].append(index)
    resolved: dict[str, list[int]] = {}
    for group, aliases in TARGET_GROUP_ALIASES.items():
        indices: list[int] = []
        for alias in aliases:
            matches = name_to_indices.get(alias, [])
            if len(matches) != 1:
                raise ClipVocabShadowError(
                    f"target alias must resolve exactly once: {alias!r}: {matches}"
                )
            indices.extend(matches)
        resolved[group] = sorted(set(indices))
    target = sorted({index for rows in resolved.values() for index in rows})
    target_set = set(target)
    non_target = [
        index for index in range(len(vocabulary)) if index not in target_set
    ]
    if not target or not non_target:
        raise ClipVocabShadowError("target and non-target vocabularies must be non-empty")
    if target != EXPECTED_TARGET_INDICES:
        raise ClipVocabShadowError(
            f"audited target index set changed: {target} != {EXPECTED_TARGET_INDICES}"
        )
    return resolved, target, non_target


def _index_owl_rows(
    rows: Sequence[Mapping[str, str]],
) -> dict[int, list[Mapping[str, str]]]:
    by_frame: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        frame_id = _csv_integer(row.get("time_ns"), f"OWL row {row_index} time_ns")
        by_frame[frame_id].append(row)
    return dict(by_frame)


def _resolve_evidence(
    *,
    raw_rows: Sequence[Mapping[str, str]],
    owl_by_frame: Mapping[int, Sequence[Mapping[str, str]]],
    raw_source_row: int,
    expected_frame_id: int,
) -> dict[str, Any]:
    """Resolve one receipt row to its exact cached OWL crop provenance."""

    if raw_source_row < 0 or raw_source_row >= len(raw_rows):
        raise ClipVocabShadowError(
            f"Raw source row is outside CSV: {raw_source_row}/{len(raw_rows)}"
        )
    raw = raw_rows[raw_source_row]
    frame_id = _csv_integer(raw.get("time_ns"), "Raw time_ns")
    if frame_id != expected_frame_id:
        raise ClipVocabShadowError(
            f"receipt/raw frame mismatch at row {raw_source_row}: "
            f"{expected_frame_id} != {frame_id}"
        )
    boxer_instance = _csv_integer(raw.get("instance"), "Raw Boxer instance")
    frame_rows = owl_by_frame.get(frame_id)
    if frame_rows is None or boxer_instance >= len(frame_rows):
        raise ClipVocabShadowError(
            f"missing exact OWL key ({frame_id}, {boxer_instance})"
        )
    owl = frame_rows[boxer_instance]
    raw_name = str(raw.get("name", ""))
    owl_name = str(owl.get("name", ""))
    raw_semantic_id = _csv_integer(raw.get("sem_id"), "Raw semantic ID")
    owl_semantic_id = _csv_integer(owl.get("sem_id"), "OWL semantic ID")
    if raw_name != owl_name or raw_semantic_id != owl_semantic_id:
        raise ClipVocabShadowError(
            f"Raw/OWL exact-key provenance mismatch at row {raw_source_row}: "
            f"{(raw_name, raw_semantic_id)} != {(owl_name, owl_semantic_id)}"
        )
    image_width = _csv_integer(owl.get("img_width"), "OWL image width", minimum=1)
    image_height = _csv_integer(
        owl.get("img_height"), "OWL image height", minimum=1
    )
    bbox = [
        _finite_number(owl.get(key), f"OWL {key}")
        for key in ("x1", "y1", "x2", "y2")
    ]
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ClipVocabShadowError(f"degenerate OWL bbox at key {frame_id, boxer_instance}")
    return {
        "raw_source_row": raw_source_row,
        "frame_id": frame_id,
        "boxer_instance": boxer_instance,
        "raw_name": raw_name,
        "raw_semantic_id": raw_semantic_id,
        "raw_score": _finite_number(raw.get("prob"), "Raw score"),
        "owl_name": owl_name,
        "owl_semantic_id": owl_semantic_id,
        "owl_score": _finite_number(owl.get("prob"), "OWL score"),
        "owl_bbox_xyxy": bbox,
        "owl_image_width": image_width,
        "owl_image_height": image_height,
    }


def _find_color_path(scene_root: Path, scene: str, frame_id: int) -> Path:
    base = scene_root / scene / "frames" / "color" / str(frame_id)
    found = [base.with_suffix(suffix) for suffix in (".jpg", ".png", ".jpeg")]
    found = [path for path in found if path.is_file()]
    if len(found) != 1:
        raise ClipVocabShadowError(
            f"expected one color image for {scene}/{frame_id}, found {found}"
        )
    return found[0]


def _load_resized_rgb(path: Path, width: int, height: int) -> np.ndarray:
    image_bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ClipVocabShadowError(f"cannot decode RGB image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # This reproduces ScanNetLoader.load() in the sealed Boxer provider.
    return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)


def _crop_rgb(image_rgb: np.ndarray, bbox: Sequence[float]) -> Image.Image:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ClipVocabShadowError("RGB image must have shape [H,W,3]")
    height, width = image_rgb.shape[:2]
    x1 = min(max(int(math.floor(float(bbox[0]))), 0), width - 1)
    y1 = min(max(int(math.floor(float(bbox[1]))), 0), height - 1)
    x2 = min(max(int(math.ceil(float(bbox[2]))), x1 + 1), width)
    y2 = min(max(int(math.ceil(float(bbox[3]))), y1 + 1), height)
    crop = np.ascontiguousarray(image_rgb[y1:y2, x1:x2])
    if crop.size == 0:
        raise ClipVocabShadowError("OWL crop is empty after clamping")
    return Image.fromarray(crop)


def _score_summary(
    similarities: np.ndarray,
    vocabulary: Sequence[str],
    target_indices: Sequence[int],
    non_target_indices: Sequence[int],
    target_index_groups: Mapping[int, Sequence[str]] | None = None,
) -> dict[str, Any]:
    values = np.asarray(similarities, dtype=np.float64)
    if values.shape != (len(vocabulary),) or not np.isfinite(values).all():
        raise ClipVocabShadowError("CLIP similarity row has an invalid shape/value")
    all_index = int(np.argmax(values))
    target_array = np.asarray(target_indices, dtype=np.int64)
    non_target_array = np.asarray(non_target_indices, dtype=np.int64)
    target_index = int(target_array[int(np.argmax(values[target_array]))])
    non_target_index = int(
        non_target_array[int(np.argmax(values[non_target_array]))]
    )
    target_score = float(values[target_index])
    non_target_score = float(values[non_target_index])
    groups = target_index_groups or {}
    return {
        "all_vocab_top1_index": all_index,
        "all_vocab_top1_name": vocabulary[all_index],
        "all_vocab_top1_cosine": float(values[all_index]),
        "all_vocab_top1_is_target": all_index in set(target_indices),
        "all_vocab_top1_target_alias_groups": list(groups.get(all_index, ())),
        "target_best_index": target_index,
        "target_best_name": vocabulary[target_index],
        "target_best_alias_groups": list(groups.get(target_index, ())),
        "target_best_cosine": target_score,
        "non_target_best_index": non_target_index,
        "non_target_best_name": vocabulary[non_target_index],
        "non_target_best_cosine": non_target_score,
        "target_non_target_margin": target_score - non_target_score,
    }


def _track_summary(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    margins = np.asarray(
        [row["target_non_target_margin"] for row in evidence], dtype=np.float64
    )
    target_votes = sum(bool(row["all_vocab_top1_is_target"]) for row in evidence)
    target_best_names = Counter(str(row["target_best_name"]) for row in evidence)
    voted_target = sorted(
        target_best_names.items(), key=lambda item: (-item[1], item[0])
    )[0]
    owl_groups = [tuple(row["owl_exact_target_alias_groups"]) for row in evidence]
    owl_exact_same = (
        len(owl_groups) == 3
        and all(len(groups) == 1 for groups in owl_groups)
        and len({groups[0] for groups in owl_groups}) == 1
    )
    owl_group = owl_groups[0][0] if owl_exact_same else None
    clip_same_group_votes = (
        sum(
            owl_group in row["all_vocab_top1_target_alias_groups"]
            for row in evidence
        )
        if owl_group is not None
        else 0
    )
    target_cosines = np.asarray(
        [row["target_best_cosine"] for row in evidence], dtype=np.float64
    )
    median_target_cosine = float(np.median(target_cosines))
    median_margin = float(np.median(margins))
    checks = {
        "owl_exact_alias_all_three_same_group": owl_exact_same,
        "clip_all_vocab_top1_target_votes": target_votes
        >= GATE_POLICY["clip_all_vocab_top1_target_votes_gte"],
        "clip_top1_same_group_as_owl_votes": clip_same_group_votes
        >= GATE_POLICY["clip_top1_same_group_as_owl_votes_gte"],
        "median_target_best_cosine": median_target_cosine
        >= GATE_POLICY["median_target_best_cosine_gte"],
        "median_target_non_target_margin": median_margin
        >= GATE_POLICY["median_target_non_target_margin_gte"],
    }
    return {
        "evidence_count": len(evidence),
        "all_vocab_top1_target_votes": target_votes,
        "all_vocab_top1_target_unanimous": target_votes == len(evidence),
        "all_vocab_top1_target_majority": target_votes * 2 >= len(evidence) + 1,
        "target_margin_min": float(np.min(margins)),
        "target_margin_median": float(np.median(margins)),
        "target_margin_mean": float(np.mean(margins)),
        "target_margin_max": float(np.max(margins)),
        "target_best_vote_name": voted_target[0],
        "target_best_vote_count": voted_target[1],
        "owl_collapsed_target_group": owl_group,
        "clip_top1_same_group_as_owl_votes": clip_same_group_votes,
        "median_target_best_cosine": median_target_cosine,
        "gate_checks": checks,
        "gate_rejection_reasons": [
            name for name, passed in checks.items() if not passed
        ],
        "gate_pass": all(checks.values()),
    }


def _load_clip_runtime(checkpoint: Path, device: str) -> tuple[Any, Any]:
    try:
        import open_clip
        import torch
    except ImportError as error:
        raise ClipVocabShadowError("open_clip and torch are required") from error
    if not checkpoint.is_file():
        raise ClipVocabShadowError(f"missing CLIP checkpoint: {checkpoint}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ClipVocabShadowError(f"CUDA device is unavailable: {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained=os.fspath(checkpoint)
    )
    model = model.to(device).eval()
    return model, preprocess


def _load_text_features(path: Path, device: str) -> Any:
    import torch

    try:
        features = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        features = torch.load(path, map_location="cpu")
    if not torch.is_tensor(features) or tuple(features.shape) != (
        EXPECTED_VOCABULARY_SIZE,
        EXPECTED_FEATURE_DIMENSION,
    ):
        raise ClipVocabShadowError(
            f"class features must have shape "
            f"[{EXPECTED_VOCABULARY_SIZE},{EXPECTED_FEATURE_DIMENSION}]"
        )
    if not torch.isfinite(features).all():
        raise ClipVocabShadowError("class features contain non-finite values")
    features = features.float()
    norms = features.norm(dim=1, keepdim=True)
    if torch.any(norms <= 1e-8):
        raise ClipVocabShadowError("class features contain zero-norm rows")
    return (features / norms).to(device)


def _flush_batch(
    *,
    tensors: list[Any],
    destinations: list[MutableMapping[str, Any]],
    model: Any,
    text_features: Any,
    vocabulary: Sequence[str],
    target_indices: Sequence[int],
    non_target_indices: Sequence[int],
    target_index_groups: Mapping[int, Sequence[str]],
    device: str,
) -> None:
    if not tensors:
        return
    import torch

    batch = torch.stack(tensors).to(device, non_blocking=True)
    with torch.inference_mode():
        image_features = model.encode_image(batch).float()
        if image_features.ndim != 2 or image_features.shape[1] != text_features.shape[1]:
            raise ClipVocabShadowError("CLIP image feature shape is incompatible")
        norms = image_features.norm(dim=1, keepdim=True)
        if torch.any(norms <= 1e-8) or not torch.isfinite(image_features).all():
            raise ClipVocabShadowError("CLIP image features are invalid")
        similarities = (image_features / norms) @ text_features.T
    rows = similarities.detach().cpu().numpy()
    if len(rows) != len(destinations):
        raise ClipVocabShadowError("CLIP batch cardinality changed")
    for values, destination in zip(rows, destinations):
        destination.update(
            _score_summary(
                values,
                vocabulary,
                target_indices,
                non_target_indices,
                target_index_groups,
            )
        )
    tensors.clear()
    destinations.clear()


def _selected_receipts(
    manifest: Mapping[str, Any], scene_order: Sequence[str]
) -> tuple[dict[str, list[Mapping[str, Any]]], Counter[str]]:
    scenes = manifest.get("scenes")
    if not isinstance(scenes, dict) or set(scenes) != set(scene_order):
        raise ClipVocabShadowError("receipt manifest scene set differs from scene list")
    selected: dict[str, list[Mapping[str, Any]]] = {}
    counts: Counter[str] = Counter()
    for scene in scene_order:
        rows = scenes[scene].get("receipt_decisions")
        if not isinstance(rows, list):
            raise ClipVocabShadowError(f"missing receipt decisions for {scene}")
        kept: list[Mapping[str, Any]] = []
        seen_tracks: set[int] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ClipVocabShadowError(f"invalid receipt decision for {scene}")
            track_id = _csv_integer(row.get("track_id"), "track ID")
            if track_id in seen_tracks:
                raise ClipVocabShadowError(f"duplicate track {scene}/{track_id}")
            seen_tracks.add(track_id)
            reason = _prefilter_reason(row)
            counts["accepted" if reason is None else reason] += 1
            if reason is None:
                evidence_rows = row.get("evidence_source_rows")
                evidence_frames = row.get("evidence_frame_ids")
                if (
                    not isinstance(evidence_rows, list)
                    or not isinstance(evidence_frames, list)
                    or len(evidence_rows) != 3
                    or len(evidence_frames) != 3
                ):
                    raise ClipVocabShadowError(
                        f"receipt must contain exactly three evidence rows: {scene}/{track_id}"
                    )
                kept.append(row)
        selected[scene] = kept
    return selected, counts


def run_shadow(
    *,
    receipt_manifest_path: Path,
    raw_log_root: Path,
    scene_root: Path,
    scene_list_path: Path,
    clip_checkpoint: Path,
    class_features_path: Path,
    vocabulary_path: Path,
    output_path: Path,
    device: str,
    batch_size: int,
    expected_scene_count: int = 100,
    plan_only: bool = False,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ClipVocabShadowError("batch size must be positive")
    if output_path.exists() and not plan_only:
        raise ClipVocabShadowError(f"refusing to overwrite output: {output_path}")
    scene_order = [
        line.strip()
        for line in scene_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(scene_order) != expected_scene_count or len(set(scene_order)) != len(scene_order):
        raise ClipVocabShadowError(
            f"expected {expected_scene_count} unique scenes, found {len(scene_order)}"
        )
    manifest = _read_json(receipt_manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != EXPECTED_RECEIPT_SCHEMA
        or manifest.get("selection_policy") != "v2_m50"
        or manifest.get("gt_access") is not False
        or manifest.get("evaluator_access") is not False
    ):
        raise ClipVocabShadowError("receipt manifest contract mismatch")
    selected, prefilter_counts = _selected_receipts(manifest, scene_order)

    vocabulary = vocabulary_path.read_text(encoding="utf-8").splitlines()
    target_groups, target_indices, non_target_indices = _resolve_target_indices(
        vocabulary
    )
    target_index_groups: dict[int, list[str]] = defaultdict(list)
    for group, indices in target_groups.items():
        for index in indices:
            target_index_groups[index].append(group)
    target_index_groups = {
        index: sorted(groups) for index, groups in target_index_groups.items()
    }
    selected_count = sum(len(rows) for rows in selected.values())
    plan = {
        "scene_count": len(scene_order),
        "input_receipt_count": sum(prefilter_counts.values()),
        "prefilter_selected_receipt_count": selected_count,
        "crop_count": selected_count * 3,
        "prefilter_decision_counts": dict(sorted(prefilter_counts.items())),
    }
    if plan_only:
        print(json.dumps(plan, sort_keys=True), flush=True)
        return plan

    model, preprocess = _load_clip_runtime(clip_checkpoint, device)
    text_features = _load_text_features(class_features_path, device)
    output_scenes: dict[str, Any] = {}
    total_evidence = 0
    for scene_index, scene in enumerate(scene_order):
        raw_scene = raw_log_root / "boxer_raw" / scene
        raw_path = raw_scene / "boxer_3dbbs.csv"
        owl_path = raw_scene / "owl_2dbbs.csv"
        raw_rows = _read_csv(raw_path, RAW_REQUIRED_COLUMNS)
        owl_rows = _read_csv(owl_path, OWL_REQUIRED_COLUMNS)
        owl_by_frame = _index_owl_rows(owl_rows)
        tracks: dict[str, Any] = {}
        tasks: list[tuple[int, int, MutableMapping[str, Any]]] = []
        for receipt in selected[scene]:
            track_id = _csv_integer(receipt.get("track_id"), "track ID")
            source_rows = [
                _csv_integer(value, "evidence source row")
                for value in receipt["evidence_source_rows"]
            ]
            frame_ids = [
                _csv_integer(value, "evidence frame ID")
                for value in receipt["evidence_frame_ids"]
            ]
            evidence = [
                _resolve_evidence(
                    raw_rows=raw_rows,
                    owl_by_frame=owl_by_frame,
                    raw_source_row=source_row,
                    expected_frame_id=frame_id,
                )
                for source_row, frame_id in zip(source_rows, frame_ids)
            ]
            for row in evidence:
                row["owl_normalized_name"] = _normalize_owl_name(row["owl_name"])
                owl_group = _resolve_owl_target_group(row["owl_name"])
                row["owl_exact_target_alias_groups"] = (
                    [owl_group] if owl_group is not None else []
                )
            track = {
                "track_id": track_id,
                "confirmation_frame_id": _csv_integer(
                    receipt.get("confirmation_frame_id"), "confirmation frame ID"
                ),
                "evidence_source_rows": source_rows,
                "evidence_frame_ids": frame_ids,
                "geometry": {
                    key: receipt[key]
                    for key in (
                        "min_pairwise_aabb_iou",
                        "max_pairwise_center_distance_m",
                        "first_last_frame_span",
                        "min_medoid_aabb_extent_m",
                        "max_camera_baseline_m",
                        "max_view_ray_span_deg",
                        "max_native_aabb_iou",
                        "max_candidate_in_native_containment",
                        "max_native_in_candidate_containment",
                    )
                },
                "evidence": evidence,
            }
            tracks[str(track_id)] = track
            for evidence_index, destination in enumerate(evidence):
                tasks.append((destination["frame_id"], evidence_index, destination))

        tensors: list[Any] = []
        destinations: list[MutableMapping[str, Any]] = []
        cached_key: tuple[int, int, int] | None = None
        cached_image: np.ndarray | None = None
        for frame_id, _, destination in sorted(
            tasks,
            key=lambda row: (
                row[0],
                row[2]["raw_source_row"],
                row[1],
            ),
        ):
            width = int(destination["owl_image_width"])
            height = int(destination["owl_image_height"])
            image_key = (frame_id, width, height)
            if image_key != cached_key:
                image_path = _find_color_path(scene_root, scene, frame_id)
                cached_image = _load_resized_rgb(image_path, width, height)
                cached_key = image_key
            assert cached_image is not None
            crop = _crop_rgb(cached_image, destination["owl_bbox_xyxy"])
            tensor = preprocess(crop)
            if not hasattr(tensor, "shape") or len(tensor.shape) != 3:
                raise ClipVocabShadowError("CLIP preprocess returned an invalid tensor")
            tensors.append(tensor)
            destinations.append(destination)
            if len(tensors) >= batch_size:
                _flush_batch(
                    tensors=tensors,
                    destinations=destinations,
                    model=model,
                    text_features=text_features,
                    vocabulary=vocabulary,
                    target_indices=target_indices,
                    non_target_indices=non_target_indices,
                    target_index_groups=target_index_groups,
                    device=device,
                )
        _flush_batch(
            tensors=tensors,
            destinations=destinations,
            model=model,
            text_features=text_features,
            vocabulary=vocabulary,
            target_indices=target_indices,
            non_target_indices=non_target_indices,
            target_index_groups=target_index_groups,
            device=device,
        )
        for track in tracks.values():
            track["clip_summary"] = _track_summary(track["evidence"])
            track["gate_pass"] = bool(track["clip_summary"]["gate_pass"])
        total_evidence += len(tasks)
        output_scenes[scene] = {
            "prefilter_selected_receipt_count": len(tracks),
            "tracks": tracks,
        }
        print(
            f"[{scene_index + 1}/{len(scene_order)}] {scene}: "
            f"{len(tracks)} receipts, {len(tasks)} crops",
            flush=True,
        )

    output = {
        "schema": SCHEMA,
        "mode": "shadow_output_inert",
        "gt_access": False,
        "annotation_access": False,
        "evaluator_access": False,
        "scene_count": len(scene_order),
        "input_receipt_count": sum(prefilter_counts.values()),
        "prefilter_selected_receipt_count": selected_count,
        "clip_evidence_count": total_evidence,
        "prefilter": PREFILTER,
        "prefilter_decision_counts": dict(sorted(prefilter_counts.items())),
        "target_group_aliases": {
            group: list(aliases) for group, aliases in TARGET_GROUP_ALIASES.items()
        },
        "owl_target_group_aliases": {
            group: list(aliases)
            for group, aliases in OWL_TARGET_GROUP_ALIASES.items()
        },
        "target_group_indices": target_groups,
        "target_indices": target_indices,
        "non_target_index_count": len(non_target_indices),
        "vocabulary_size": len(vocabulary),
        "clip_model": "ViT-H-14",
        "clip_similarity": "normalized_image_dot_normalized_cached_text_cosine",
        "gate_policy": GATE_POLICY,
        "image_replay": "sealed_provider_RGB_BGR_to_RGB_then_cv2_resize_960x960",
        "crop_rule": "floor_xy1_ceil_xy2_clamp_then_unmasked_RGB_crop",
        "raw_to_owl_key": "(raw.time_ns, raw.instance_as_per_frame_OWL_index)",
        "inputs": {
            "receipt_manifest": os.fspath(receipt_manifest_path.resolve()),
            "receipt_manifest_sha256": _sha256(receipt_manifest_path),
            "raw_log_root": os.fspath(raw_log_root.resolve()),
            "scene_root": os.fspath(scene_root.resolve()),
            "scene_list": os.fspath(scene_list_path.resolve()),
            "scene_list_sha256": _sha256(scene_list_path),
            "clip_checkpoint": os.fspath(clip_checkpoint.resolve()),
            "clip_checkpoint_sha256": _sha256(clip_checkpoint),
            "class_features": os.fspath(class_features_path.resolve()),
            "class_features_sha256": _sha256(class_features_path),
            "vocabulary": os.fspath(vocabulary_path.resolve()),
            "vocabulary_sha256": _sha256(vocabulary_path),
        },
        "contracts": {
            "gt_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "prediction_pickle_access": False,
            "prediction_mutation": False,
            "training": False,
            "online_learning": False,
            "external_pretraining_frozen": True,
            "past_only_receipts": True,
            "native_clip_checkpoint_and_text_features_reused": True,
        },
        "scenes": output_scenes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    with output_path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"Saved: {output_path}", flush=True)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a frozen no-GT CLIP vocabulary shadow on Raw Boxer receipts"
    )
    parser.add_argument(
        "--receipt-manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "results/scannet_cbest_raw_boxer_past3_birth_v2_m50_score05/"
        "RAW_BOXER_PAST3_BIRTH_FULL100.json",
    )
    parser.add_argument(
        "--raw-log-root",
        type=Path,
        default=REPOSITORY_ROOT / "logs/scannet_raw_boxer_full100_score05_v1",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=REPOSITORY_ROOT / "upstream_clean/scannet_readme_frames",
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "models/open_clip_pytorch_model.bin",
    )
    parser.add_argument(
        "--class-features",
        type=Path,
        default=REPOSITORY_ROOT / "data/class_features.pt",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=REPOSITORY_ROOT / "data/panoptic_categories_nomerge.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "logs/scannet_cbest_raw_boxer_clip_vocab_shadow_score05/"
        "CLIP_VOCAB_SHADOW_FULL100.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate manifests and print crop counts without loading CLIP/RGB",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_shadow(
        receipt_manifest_path=args.receipt_manifest,
        raw_log_root=args.raw_log_root,
        scene_root=args.scene_root,
        scene_list_path=args.scene_list,
        clip_checkpoint=args.clip_checkpoint,
        class_features_path=args.class_features,
        vocabulary_path=args.vocabulary,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        expected_scene_count=args.expected_scene_count,
        plan_only=args.plan_only,
    )


if __name__ == "__main__":
    main()

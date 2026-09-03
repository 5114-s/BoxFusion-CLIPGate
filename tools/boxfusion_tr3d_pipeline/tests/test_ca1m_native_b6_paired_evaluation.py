from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_ca1m_native_b6_paired import (  # noqa: E402
    COLLECTION_SCHEMA,
    IDENTITY_SCHEMA,
    SCENE_COMPLETION_SCHEMA,
    SCENE_LIST_SHA256,
    compare_score_only,
    independently_recompute_scores,
    metric_delta,
    parse_metrics,
    validate_collection_chain,
    validate_eval_batches,
    validate_short_tmp_root,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction(path: Path, value: int, score: float) -> None:
    corners = np.full((8, 3), value, dtype=np.float32)
    path.write_bytes(pickle.dumps([[(0, corners, score)]], protocol=pickle.HIGHEST_PROTOCOL))


def _sealed_collection(tmp_path: Path) -> tuple[argparse.Namespace, tuple[str, ...]]:
    scenes = tuple(f"{42_000_000 + index:08d}" for index in range(103))
    roots = {
        name: tmp_path / name
        for name in ("anchor", "observer", "diagnostic", "record", "completion")
    }
    for root in roots.values():
        root.mkdir()
    collection_rows = []
    identity_scenes = {}
    for index, scene in enumerate(scenes):
        anchor = roots["anchor"] / f"{scene}_boxes.pkl"
        observer = roots["observer"] / f"{scene}_boxes.pkl"
        diagnostic = roots["diagnostic"] / f"{scene}_ca1m_native_b6.npz"
        score = 0.2 + 0.001 * index
        _prediction(anchor, index, score)
        observer.write_bytes(anchor.read_bytes())
        diagnostic.write_bytes(f"diagnostic-{scene}".encode())
        record = {
            "schema": SCENE_COMPLETION_SCHEMA,
            "phase": "cutr_record",
            "scene_id": scene,
            "complete": True,
            "dataset_split": "official_validation_canonical103",
            "ground_truth_access": False,
            "evaluation_invoked": False,
            "training_authorized": False,
        }
        completion = {
            "schema": SCENE_COMPLETION_SCHEMA,
            "phase": "g0_native_b6_observer",
            "scene_id": scene,
            "complete": True,
            "dataset_split": "official_validation_canonical103",
            "ground_truth_access": False,
            "evaluation_invoked": False,
            "training_authorized": False,
            "output_mutation_authorized": False,
            "prediction_rows": 1,
            "artifacts": {
                "prediction": {"path": str(observer.resolve()), "sha256": _sha(observer)},
                "same_run_anchor": {"path": str(anchor.resolve()), "sha256": _sha(anchor)},
                "native_b6_diagnostic": {
                    "path": str(diagnostic.resolve()), "sha256": _sha(diagnostic)
                },
            },
        }
        record_path = roots["record"] / f"{scene}.json"
        completion_path = roots["completion"] / f"{scene}.json"
        record_path.write_text(json.dumps(record, sort_keys=True))
        completion_path.write_text(json.dumps(completion, sort_keys=True))
        collection_rows.append({
            "scene_id": scene,
            "record_completion_sha256": _sha(record_path),
            "observer_completion_sha256": _sha(completion_path),
        })
        identity_scenes[scene] = {
            "identity": {
                "rows": 1, "byte_identity": True, "semantic_identity": True,
                "prediction_sha256": _sha(anchor),
            },
            "diagnostic": {
                "mapping_coverage": 1.0, "diagnostic_sha256": _sha(diagnostic)
            },
        }
    collection_path = tmp_path / "collection.json"
    collection_path.write_text(json.dumps({
        "schema": COLLECTION_SCHEMA,
        "complete": True,
        "dataset_split": "official_validation_canonical103",
        "scene_count": 103,
        "scene_list_sha256": SCENE_LIST_SHA256,
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "training_authorized": False,
        "same_run_anchor_byte_identity_required": True,
        "scenes": collection_rows,
    }))
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps({
        "schema": IDENTITY_SCHEMA,
        "ok": True,
        "dataset_split": "official_validation_canonical103",
        "scenes": 103,
        "scene_list_sha256": SCENE_LIST_SHA256,
        "observer_only": True,
        "mutation_enabled": False,
        "same_run_byte_identity_scenes": 103,
        "prediction_rows": 103,
        "mapping_rows": 103,
        "mapping_coverage": 1.0,
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "training_authorized": False,
        "per_scene": identity_scenes,
    }))
    args = argparse.Namespace(
        anchor_root=roots["anchor"], observer_root=roots["observer"],
        diagnostics_root=roots["diagnostic"], record_completion_root=roots["record"],
        observer_completion_root=roots["completion"],
        collection_manifest=collection_path, identity_audit=identity_path,
    )
    return args, scenes


def test_sealed_collection_is_exact103_and_rejects_extra_files(tmp_path: Path) -> None:
    args, scenes = _sealed_collection(tmp_path)
    result = validate_collection_chain(args, scenes)
    assert result["rows"] == 103
    assert result["real_score_min"] < result["real_score_max"]
    (args.anchor_root / "extra_boxes.pkl").write_bytes(b"extra")
    with pytest.raises(ValueError, match="artifact set differs"):
        validate_collection_chain(args, scenes)


def test_score_only_comparison_checks_raw_float32_corner_bytes() -> None:
    corners = np.arange(24, dtype=np.float32).reshape(8, 3)
    anchor = [(0, corners.copy(), 0.4)]
    active = [(0, corners.copy(), 0.6)]
    assert compare_score_only(anchor, active) == 1
    active[0][1][0, 0] = np.float32(-0.0)
    anchor[0][1][0, 0] = np.float32(0.0)
    with pytest.raises(ValueError, match="geometry/order changed"):
        compare_score_only(anchor, active)


def test_independent_formula_is_frozen_04_06(tmp_path: Path) -> None:
    scene = "42000000"
    corners = np.arange(24, dtype=np.float32).reshape(1, 8, 3)
    detector = np.asarray((0.25,), dtype=np.float32)
    diagnostic = tmp_path / "diagnostic.npz"
    np.savez_compressed(
        diagnostic,
        result_indices=np.arange(1, dtype=np.int64), corners=corners,
        scores=detector, features=np.zeros((1, 14), dtype=np.float32),
    )
    checkpoint = {
        "num_layers": np.asarray(1, dtype=np.int64),
        "detector_blend": np.asarray(0.4, dtype=np.float32),
        "feature_mean": np.zeros(14, dtype=np.float32),
        "feature_scale": np.ones(14, dtype=np.float32),
        "weight_0": np.zeros((14, 4), dtype=np.float32),
        "bias_0": np.zeros(4, dtype=np.float32),
        "ranking_weights": np.asarray((0.1, 0.2, 0.3, 0.4), dtype=np.float32),
    }
    rows = [(0, corners[0].copy(), float(detector[0]))]
    observed = independently_recompute_scores(diagnostic, scene, rows, checkpoint)
    np.testing.assert_array_equal(observed, np.asarray((0.4,), dtype=np.float32))


def test_official_metric_parser_batches_and_delta() -> None:
    scenes = ("42000001", "42000000")
    text = "\n".join([
        "Eval batch: 0 scan_idx 42000000", "Eval batch: 1 scan_idx 42000001",
        "eval mAP: 0.4", "eval APrec: 0.5", "eval ARecall: 0.6",
        "eval mAP: 0.3", "eval APrec: 0.4", "eval ARecall: 0.5",
        "eval mAP: 0.2", "eval APrec: 0.3", "eval ARecall: 0.4",
    ])
    validate_eval_batches(text, scenes)
    baseline = parse_metrics(text)
    active = {threshold: {key: value + 0.01 for key, value in row.items()}
              for threshold, row in baseline.items()}
    delta, points = metric_delta(baseline, active)
    assert delta["0.50"]["mAP"] == pytest.approx(0.01)
    assert points["0.15"]["mAP"] == pytest.approx(1.0)


def test_shell_runner_gates_before_active_and_freezes_official_protocol() -> None:
    source = (ROOT / "scripts/run_ca1m_native_b6_paired_canonical103.sh").read_text()
    tool = (ROOT / "tools/evaluate_ca1m_native_b6_paired.py").read_text()
    assert source.index('"$PYTHON" "$TOOL" preflight') < source.index('bash "$SCORE_RUNNER" --active')
    for value in (
        "b2e0219a7284249bad4a4a8925066839fe2fa33b",
        "3c9260cd57da342fd25b664a0091c4345a44bba499c2cfbf3c8ecff4eaa4c788",
        "c2b08890cf6b6497165d7d7af0bf16f9205a65698c197639db70adf702f27d6f",
        "6ef54c395e46716e364547115090bae96643bf346b3e8eb1b859719781a557dd",
        "44aadf0088c0ccd5e9f51a1cded22fb1080d59aa50d0fb914fe6e83896aaa107",
        "50d4e03db6f1fa9e540fd7f9c6ceab85d180ed61a14251d0e1971c717e741f8d",
    ):
        assert value in tool or value in source
    assert '"min_delta_AP50": 0.005' in tool
    assert "world enclosing-AABB IoU" in tool and "not true OBB IoU" in tool
    assert "/tmp/bfc103b6-$TAG_SHA" in source


def test_evaluation_tmp_root_reserves_af_unix_path_margin(tmp_path: Path) -> None:
    assert validate_short_tmp_root(Path("/tmp/bfc103b6-unit")) == Path(
        "/tmp/bfc103b6-unit"
    )
    with pytest.raises(ValueError, match="AF_UNIX"):
        validate_short_tmp_root(tmp_path / ("nested" * 20))

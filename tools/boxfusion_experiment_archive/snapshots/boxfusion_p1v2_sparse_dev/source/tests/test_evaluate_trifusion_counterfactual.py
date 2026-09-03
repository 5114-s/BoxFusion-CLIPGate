from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from tools.evaluate_trifusion_counterfactual import (
    COMBINED_METHOD,
    DIAGNOSTIC_SCHEMA,
    FROZEN_METHOD,
    GATE_DIAGNOSTIC_SCHEMA,
    GATE_FEATURE_NAMES,
    M3_FEATURE_NAMES,
    M3_METHOD,
    MISSING_DIAGNOSTIC_SCHEMA,
    REPORT_SCHEMA,
    SUPPLEMENTAL_METHOD,
    apply_m3_counterfactual,
    evaluate,
    load_frozen_predictions,
    load_m3_diagnostics,
)
from tools.report_trifusion_oracles import (
    CORNER_FRAME,
    SUPPLEMENTAL_CANDIDATE_SCHEMA,
)


SCENE = "scene0000_00"


def _corners(center: tuple[float, float, float]) -> np.ndarray:
    cx, cy, cz = center
    x0, x1 = cx - 0.5, cx + 0.5
    y0, y1 = cy - 0.5, cy + 0.5
    z0, z1 = cz - 0.5, cz + 0.5
    return np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _fixture(
    root: Path,
    *,
    with_supplemental: bool,
) -> tuple[dict[str, Path], dict[str, np.ndarray]]:
    paths = {
        "pred": root / "pred",
        "diagnostics": root / "diagnostics",
        "gt": root / "gt",
        "scan": root / "scan",
        "supplemental": root / "supplemental",
        "scene_list": root / "fixed-scenes.txt",
    }
    for name in ("pred", "diagnostics", "gt", "scan"):
        paths[name].mkdir(parents=True)
    paths["scene_list"].write_text(f"{SCENE}\n", encoding="utf-8")

    frozen_corners = np.stack(
        (_corners((20.0, 0.0, 0.0)), _corners((4.0, 0.0, 0.0)))
    )
    frozen_scores = np.asarray([0.9, 0.8], dtype=np.float32)
    with (paths["pred"] / f"{SCENE}_boxes.pkl").open("wb") as handle:
        pickle.dump(
            [
                [
                    (0, frozen_corners[index], float(frozen_scores[index]))
                    for index in range(2)
                ]
            ],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    config = {
        "enabled": True,
        "mutate": False,
        "collect_diagnostics": True,
        "safety_gate": {
            "enabled": True,
            "mutate": False,
            "collect_diagnostics": True,
        },
    }
    candidate_corners = np.stack(
        (_corners((0.0, 0.0, 0.0)), frozen_corners[1])
    )
    nan_tail = np.asarray([0.1, np.nan], dtype=np.float32)
    diagnostic = {
        "scene_id": np.asarray(SCENE),
        "result_indices": np.asarray([0, 1], dtype=np.int64),
        "track_ids": np.asarray([101, 102], dtype=np.int64),
        "refit_candidate_corners": frozen_corners.copy(),
        "scores": frozen_scores.copy(),
        "trifusion_diagnostics_schema": np.asarray(DIAGNOSTIC_SCHEMA),
        "trifusion_enabled": np.asarray(True, dtype=np.bool_),
        "trifusion_mutation_enabled": np.asarray(False, dtype=np.bool_),
        "trifusion_config_json": np.asarray(
            json.dumps(config, sort_keys=True)
        ),
        "trifusion_result_indices": np.asarray([0, 1], dtype=np.int64),
        "trifusion_stable_ids": np.asarray([101, 102], dtype=np.int64),
        "trifusion_feature_names": np.asarray(M3_FEATURE_NAMES),
        "trifusion_features": np.zeros(
            (2, len(M3_FEATURE_NAMES)), dtype=np.float32
        ),
        "trifusion_original_corners": frozen_corners.copy(),
        "trifusion_candidate_corners": candidate_corners,
        "trifusion_gate_diagnostics_schema": np.asarray(
            GATE_DIAGNOSTIC_SCHEMA
        ),
        "trifusion_gate_enabled": np.asarray(True, dtype=np.bool_),
        "trifusion_gate_mutation_enabled": np.asarray(
            False, dtype=np.bool_
        ),
        "trifusion_gate_feature_names": np.asarray(GATE_FEATURE_NAMES),
        "trifusion_gate_features": np.zeros(
            (2, len(GATE_FEATURE_NAMES)), dtype=np.float32
        ),
        "trifusion_gate_evaluated": np.asarray(
            [True, False], dtype=np.bool_
        ),
        "trifusion_gate_accepted": np.asarray(
            [True, False], dtype=np.bool_
        ),
        "trifusion_gate_reason": np.asarray(
            ["accepted", "pending_geometry"]
        ),
        "trifusion_gate_lower_confidence_delta": nan_tail.copy(),
        "trifusion_gate_delta_mean": nan_tail.copy(),
        "trifusion_gate_delta_std": nan_tail.copy(),
        "trifusion_gate_improvement_probability": np.asarray(
            [0.9, np.nan], dtype=np.float32
        ),
        "trifusion_gate_harm_probability": np.asarray(
            [0.1, np.nan], dtype=np.float32
        ),
        "trifusion_gate_original_iou": np.asarray(
            [0.1, np.nan], dtype=np.float32
        ),
        "trifusion_gate_candidate_iou": np.asarray(
            [0.9, np.nan], dtype=np.float32
        ),
        "trifusion_gate_cross_iou25_probability": np.asarray(
            [0.9, np.nan], dtype=np.float32
        ),
        "trifusion_gate_cross_iou50_probability": np.asarray(
            [0.9, np.nan], dtype=np.float32
        ),
        "trifusion_candidate_valid": np.asarray(
            [True, False], dtype=np.bool_
        ),
        "trifusion_is_candidate": np.asarray(
            [True, False], dtype=np.bool_
        ),
        "trifusion_candidate_verified": np.asarray(
            [True, False], dtype=np.bool_
        ),
        "trifusion_applied": np.zeros(2, dtype=np.bool_),
        "trifusion_reason": np.asarray(
            ["verified_observer", "proposal_not_run"]
        ),
        "trifusion_source": np.asarray(
            ["occupancy_msr", "occupancy_msr"]
        ),
    }

    supplemental_corner = _corners((8.0, 0.0, 0.0))[None, ...]
    if with_supplemental:
        diagnostic.update(
            {
                "trifusion_missing_diagnostics_schema": np.asarray(
                    MISSING_DIAGNOSTIC_SCHEMA
                ),
                "trifusion_missing_enabled": np.asarray(
                    True, dtype=np.bool_
                ),
                "trifusion_missing_mutation_enabled": np.asarray(
                    False, dtype=np.bool_
                ),
                "trifusion_missing_candidate_ids": np.asarray(
                    [501], dtype=np.int64
                ),
                "trifusion_missing_sources": np.asarray(
                    ["missing_graph"]
                ),
                "trifusion_missing_corners": supplemental_corner,
                "trifusion_missing_valid": np.asarray(
                    [True], dtype=np.bool_
                ),
                "trifusion_missing_verified": np.asarray(
                    [True], dtype=np.bool_
                ),
                "trifusion_missing_confirmed": np.asarray(
                    [True], dtype=np.bool_
                ),
                "trifusion_missing_applied": np.asarray(
                    [False], dtype=np.bool_
                ),
            }
        )
        paths["supplemental"].mkdir(parents=True)
        _write_npz(
            paths["supplemental"]
            / f"{SCENE}_supplemental_candidates.npz",
            {
                "schema": np.asarray(SUPPLEMENTAL_CANDIDATE_SCHEMA),
                "format_version": np.asarray(1, dtype=np.int64),
                "scene_id": np.asarray(SCENE),
                "corner_frame": np.asarray(CORNER_FRAME),
                "candidate_corners": supplemental_corner,
                "candidate_ids": np.asarray(
                    [f"{SCENE}:missing_graph:track:501"]
                ),
                "candidate_sources": np.asarray(["missing_graph"]),
                "candidate_valid": np.asarray([True], dtype=np.bool_),
                "candidate_verified": np.asarray([True], dtype=np.bool_),
                "candidate_confirmed": np.asarray(
                    [True], dtype=np.bool_
                ),
                # These diagnostics must be ignored in favor of the explicit
                # ablation score.
                "candidate_scores": np.asarray(
                    [0.02], dtype=np.float32
                ),
                "observer_only": np.asarray(True, dtype=np.bool_),
                "uses_ground_truth": np.asarray(False, dtype=np.bool_),
            },
        )
    _write_npz(
        paths["diagnostics"] / f"{SCENE}_tracks.npz", diagnostic
    )

    np.save(
        paths["gt"] / f"{SCENE}_bbox.npy",
        np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [4.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [8.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    # Its malformed contents prove evaluation does not discover extra GT.
    (paths["gt"] / "scene9999_99_bbox.npy").write_bytes(b"not-a-npy")
    scene_scan = paths["scan"] / SCENE
    scene_scan.mkdir(parents=True)
    scene_scan.joinpath(f"{SCENE}.txt").write_text(
        "axisAlignment = "
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n",
        encoding="utf-8",
    )
    return paths, diagnostic


def _hash_inputs(paths: dict[str, Path]) -> dict[str, str]:
    roots = [
        paths["pred"],
        paths["diagnostics"],
        paths["gt"],
        paths["scan"],
    ]
    if paths["supplemental"].is_dir():
        roots.append(paths["supplemental"])
    result: dict[str, str] = {}
    for root in roots:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_gate_counterfactual_and_optional_supplemental_are_read_only(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path, with_supplemental=True)
    before = _hash_inputs(paths)

    report = evaluate(
        pred_root=paths["pred"],
        diagnostics_root=paths["diagnostics"],
        scene_list=paths["scene_list"],
        gt_root=paths["gt"],
        scan_root=paths["scan"],
        supplemental_candidates_root=paths["supplemental"],
        supplemental_fixed_score=0.7,
    )

    assert _hash_inputs(paths) == before
    assert report["schema"] == REPORT_SCHEMA
    assert report["evaluation_only"] is True
    assert report["gate_checkpoint_loaded"] is False
    assert report["gate_training_inputs_read"] is False
    assert report["prediction_artifacts_mutated"] is False
    assert report["diagnostic_artifacts_mutated"] is False
    assert report["ground_truth_scope"]["files_read"] == [
        f"{SCENE}_bbox.npy"
    ]
    assert "Do not use" in report["heldout_warning"]
    assert report["inventory"]["m3_replacements"] == 1
    assert report["inventory"]["supplemental_eligible"] == 1
    assert (
        report["inventory"][
            "supplemental_artifact_score_rows_ignored"
        ]
        == 1
    )
    assert report["scenes"][0]["m3_replaced_result_indices"] == [0]
    assert report["scenes"][0]["m3_replaced_stable_ids"] == [101]
    for key in ("AP15", "AP25", "AP50"):
        frozen = report["metrics"][FROZEN_METHOD][key]["average_precision"]
        m3 = report["metrics"][M3_METHOD][key]["average_precision"]
        supplemental = report["metrics"][SUPPLEMENTAL_METHOD][key][
            "average_precision"
        ]
        combined = report["metrics"][COMBINED_METHOD][key][
            "average_precision"
        ]
        assert combined > m3 > frozen
        assert combined > supplemental > frozen
    assert json.loads(json.dumps(report, allow_nan=False)) == report

    frozen = load_frozen_predictions(
        paths["pred"] / f"{SCENE}_boxes.pkl"
    )
    diagnostic = load_m3_diagnostics(
        paths["diagnostics"] / f"{SCENE}_tracks.npz",
        scene_id=SCENE,
        frozen=frozen,
        require_missing_observer=True,
    )
    counterfactual = apply_m3_counterfactual(frozen, diagnostic)
    np.testing.assert_array_equal(counterfactual.scores, frozen.scores)
    np.testing.assert_array_equal(
        counterfactual.corners[0], _corners((0.0, 0.0, 0.0))
    )
    np.testing.assert_array_equal(
        counterfactual.corners[1], frozen.corners[1]
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "trifusion_diagnostics_schema",
            np.asarray("wrong.schema"),
            "unsupported TriFusion schema",
        ),
        (
            "trifusion_stable_ids",
            np.asarray([999, 102], dtype=np.int64),
            "stable IDs do not align",
        ),
        (
            "trifusion_applied",
            np.asarray([True, False], dtype=np.bool_),
            "must be all false",
        ),
        (
            "trifusion_gate_evaluated",
            np.asarray([False, False], dtype=np.bool_),
            "must exactly align",
        ),
    ],
)
def test_strict_m3_contract_rejects_corruption(
    tmp_path: Path,
    field: str,
    value: np.ndarray,
    match: str,
) -> None:
    paths, payload = _fixture(tmp_path, with_supplemental=False)
    payload[field] = value
    _write_npz(
        paths["diagnostics"] / f"{SCENE}_tracks.npz", payload
    )
    with pytest.raises(ValueError, match=match):
        evaluate(
            pred_root=paths["pred"],
            diagnostics_root=paths["diagnostics"],
            scene_list=paths["scene_list"],
            gt_root=paths["gt"],
            scan_root=paths["scan"],
        )


def test_supplemental_requires_an_explicit_fixed_score(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path, with_supplemental=True)
    with pytest.raises(ValueError, match="must be provided together"):
        evaluate(
            pred_root=paths["pred"],
            diagnostics_root=paths["diagnostics"],
            scene_list=paths["scene_list"],
            gt_root=paths["gt"],
            scan_root=paths["scan"],
            supplemental_candidates_root=paths["supplemental"],
        )

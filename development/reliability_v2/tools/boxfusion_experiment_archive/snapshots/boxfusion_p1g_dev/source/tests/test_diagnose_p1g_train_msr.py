"""Causal classification tests for paired train-only P1G replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.diagnose_p1g_train_msr import (
    CLASSIFICATIONS,
    INPUT_SCHEMA,
    REPORT_SCHEMA,
    diagnose,
    main,
)


SCENES = ("scene0001_00", "scene0002_00")
SHA = {
    "scene": "1" * 64,
    "forbidden": "2" * 64,
    "p1s": "3" * 64,
    "b6": "4" * 64,
}


def _scene(
    scene_id: str,
    *,
    candidates: int,
    changed: int,
    actual_cross: int,
    feasible_018: int,
    feasible_050: int,
    runtime: float,
) -> dict:
    improved = max(actual_cross, 1)
    actual = {
        "matched_residual_count": candidates,
        "improved_count": improved,
        "degraded_count": 1,
        "cross_iou50": actual_cross,
        "fall_iou50": 1,
        "median_delta_iou": 0.01,
    }
    feasibility = {
        "0.18": {
            "sample_count": candidates,
            "cross_iou50": feasible_018,
        },
        "0.25": {
            "sample_count": candidates,
            "cross_iou50": feasible_018,
        },
        "0.50": {
            "sample_count": candidates,
            "cross_iou50": feasible_050,
        },
        "0.75": {
            "sample_count": candidates,
            "cross_iou50": feasible_050,
        },
    }
    return {
        "scene_id": scene_id,
        "candidate_count": candidates,
        "p1g_row_count": candidates,
        "p1g_changed_count": changed,
        "p1g_failure_count": 0,
        "p1g_runtime_seconds": runtime,
        "actual": actual,
        "feasibility": feasibility,
    }


def _summary(
    profile: str,
    *,
    actual_cross: int,
    feasible_018: int,
    feasible_050: int,
    changed: int = 4,
    runtime: float = 0.4,
) -> dict:
    per_cross = (actual_cross // 2, actual_cross - actual_cross // 2)
    per_018 = (
        feasible_018 // 2,
        feasible_018 - feasible_018 // 2,
    )
    per_050 = (
        feasible_050 // 2,
        feasible_050 - feasible_050 // 2,
    )
    per_changed = (changed // 2, changed - changed // 2)
    rows = [
        _scene(
            scene,
            candidates=10,
            changed=per_changed[index],
            actual_cross=per_cross[index],
            feasible_018=per_018[index],
            feasible_050=per_050[index],
            runtime=runtime / 2.0,
        )
        for index, scene in enumerate(SCENES)
    ]
    actual = {
        key: sum(row["actual"][key] for row in rows)
        for key in (
            "matched_residual_count",
            "improved_count",
            "degraded_count",
            "cross_iou50",
            "fall_iou50",
        )
    }
    feasibility = {
        limit: {
            key: sum(
                row["feasibility"][limit][key] for row in rows
            )
            for key in ("sample_count", "cross_iou50")
        }
        for limit in ("0.18", "0.25", "0.50", "0.75")
    }
    proposal = {
        "maximum_face_shift_ratio": 0.18,
        "min_total_points": 64 if profile == "permissive" else 128,
    }
    return {
        "schema": INPUT_SCHEMA,
        "scene_count": len(SCENES),
        "scene_ids": list(SCENES),
        "candidate_count": 20,
        "p1g_changed_count": changed,
        "p1g_failure_count": 0,
        "p1g_runtime_seconds": runtime,
        "actual": actual,
        "feasibility": feasibility,
        "provenance": {
            "scene_list": "/train/scenes.txt",
            "scene_list_sha256": SHA["scene"],
            "forbidden_scene_list": "/meta/val.txt",
            "forbidden_scene_list_sha256": SHA["forbidden"],
            "forbidden_overlap": [],
            "p1s_checkpoint": "/models/p1s.pt",
            "p1s_checkpoint_sha256": SHA["p1s"],
            "b6_checkpoint": "/models/b6.pt",
            "b6_checkpoint_sha256": SHA["b6"],
            "face_limits": [0.18, 0.25, 0.50, 0.75],
            "covered_iou": 0.15,
            "msr_evidence_profile": profile,
            "p1g_config": {
                "enabled": True,
                "observer_only": True,
                "mutate": False,
                "collect_diagnostics": True,
                "proposal": proposal,
            },
        },
        "scenes": rows,
    }


def _pair(
    tmp_path: Path,
    *,
    conservative_cross: int,
    permissive_cross: int,
    feasible_018: int,
    feasible_050: int,
) -> tuple[Path, Path]:
    conservative = tmp_path / "conservative.json"
    permissive = tmp_path / "permissive.json"
    conservative.write_text(
        json.dumps(
            _summary(
                "conservative",
                actual_cross=conservative_cross,
                feasible_018=feasible_018,
                feasible_050=feasible_050,
                changed=4,
                runtime=0.4,
            )
        ),
        encoding="utf-8",
    )
    permissive.write_text(
        json.dumps(
            _summary(
                "permissive",
                actual_cross=permissive_cross,
                feasible_018=feasible_018,
                feasible_050=feasible_050,
                changed=8,
                runtime=0.8,
            )
        ),
        encoding="utf-8",
    )
    return conservative, permissive


@pytest.mark.parametrize(
    (
        "conservative_cross",
        "permissive_cross",
        "feasible_018",
        "feasible_050",
        "expected",
    ),
    [
        (5, 5, 5, 8, "effective"),
        (1, 2, 4, 8, "clamp_parameter_problem"),
        (1, 5, 7, 8, "internal_gate_parameter_problem"),
        (
            1,
            2,
            7,
            8,
            "association_or_boundary_estimation_method_problem",
        ),
        (1, 2, 3, 4, "parent_proposal_method_limit"),
    ],
)
def test_five_way_diagnosis(
    tmp_path,
    conservative_cross,
    permissive_cross,
    feasible_018,
    feasible_050,
    expected,
):
    conservative, permissive = _pair(
        tmp_path,
        conservative_cross=conservative_cross,
        permissive_cross=permissive_cross,
        feasible_018=feasible_018,
        feasible_050=feasible_050,
    )
    report = diagnose(
        conservative_summary=conservative,
        permissive_summary=permissive,
    )

    assert report["schema"] == REPORT_SCHEMA
    assert report["decision"]["classification"] == expected
    assert expected in CLASSIFICATIONS
    assert report["comparison"]["changed_count"] == {
        "conservative": 4,
        "permissive": 8,
        "delta": 4,
        "permissive_over_conservative": 2.0,
    }
    assert report["comparison"]["runtime"][
        "conservative_seconds_per_scene"
    ] == pytest.approx(0.2)
    assert report["comparison"]["runtime"][
        "permissive_seconds_per_scene"
    ] == pytest.approx(0.4)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(
                scene_ids=list(reversed(SCENES)),
                scenes=list(reversed(payload["scenes"])),
            ),
            "scene_ids mismatch",
        ),
        (
            lambda payload: payload.update(candidate_count=21),
            "aggregate candidate_count disagrees",
        ),
        (
            lambda payload: payload["provenance"].update(
                p1s_checkpoint_sha256="9" * 64
            ),
            "p1s_checkpoint_sha256 mismatch",
        ),
        (
            lambda payload: payload["provenance"].update(
                scene_list_sha256="8" * 64
            ),
            "scene_list_sha256 mismatch",
        ),
    ],
)
def test_pair_identity_mismatch_is_rejected(tmp_path, mutation, message):
    conservative, permissive = _pair(
        tmp_path,
        conservative_cross=1,
        permissive_cross=2,
        feasible_018=7,
        feasible_050=8,
    )
    payload = json.loads(permissive.read_text(encoding="utf-8"))
    mutation(payload)
    permissive.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        diagnose(
            conservative_summary=conservative,
            permissive_summary=permissive,
        )


def test_candidate_count_must_match_per_scene_not_only_aggregate(tmp_path):
    conservative, permissive = _pair(
        tmp_path,
        conservative_cross=1,
        permissive_cross=2,
        feasible_018=7,
        feasible_050=8,
    )
    payload = json.loads(permissive.read_text(encoding="utf-8"))
    payload["scenes"][0]["candidate_count"] = 9
    payload["scenes"][0]["p1g_row_count"] = 9
    payload["scenes"][1]["candidate_count"] = 11
    payload["scenes"][1]["p1g_row_count"] = 11
    payload["scenes"][0]["actual"]["matched_residual_count"] = 9
    payload["scenes"][1]["actual"]["matched_residual_count"] = 11
    permissive.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="per_scene_candidate_counts mismatch"):
        diagnose(
            conservative_summary=conservative,
            permissive_summary=permissive,
        )


def test_cli_writes_strict_json(tmp_path, capsys):
    conservative, permissive = _pair(
        tmp_path,
        conservative_cross=5,
        permissive_cross=5,
        feasible_018=5,
        feasible_050=8,
    )
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "--conservative-summary",
                str(conservative),
                "--permissive-summary",
                str(permissive),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    rendered = json.loads(output.read_text(encoding="utf-8"))
    assert rendered["decision"]["classification"] == "effective"
    assert json.loads(capsys.readouterr().out) == rendered

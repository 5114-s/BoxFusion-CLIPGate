from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from boxfusion import ca1m_e961_locked_fold1_deploy_v2 as route


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/ca1m_e961_locked_fold1_deploy_v2_pending.json"
PREFLIGHT = ROOT / "tools/preflight_ca1m_e961_locked_fold1_deploy_v2.py"
SPLITS = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/data/tr3d_ca1m_e961_v1/splits"
)
B6_MANIFEST = ROOT / "models/ca1m_native_b6_final_base_oof_row_scores_v2.manifest.json"


def _rows(name: str) -> tuple[str, ...]:
    return tuple((SPLITS / name).read_text(encoding="ascii").splitlines())


def _sha(rows: tuple[str, ...]) -> str:
    payload = "".join(f"{row}\n" for row in rows).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _mutated_config(tmp_path: Path, mutation) -> Path:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    mutation(value)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _set(path: tuple[str, ...], value):
    def mutate(config: dict) -> None:
        node = config
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return mutate


def _extra(path: tuple[str, ...], key: str, value):
    def mutate(config: dict) -> None:
        node = config
        for part in path:
            node = node[part]
        node[key] = value

    return mutate


def test_static_design_passes_but_l6_null_blocker_prevents_seal():
    report = route.validate_pending_config()
    assert report["static_design_pass"] is True
    assert report["static_protocol_sealable"] is False
    assert report["operational_authority"] is False
    assert report["static_protocol_seal_blockers"] == [
        "final_incremental_l6_static_protocol_path_and_sha256",
        "exact_l6_locked_gate_subtree_and_sha256",
        "final_l6_pass_stop_receipt_schemas_and_statuses",
    ]
    assert report["fold1_source_path_resolved"] is False
    assert report["fold1_canonical_scene_list_opened"] is False
    assert report["fold1_ground_truth_or_prediction_opened"] is False
    assert report["official_validation_opened"] is False
    assert report["native_b6_manifest_or_sidecar_opened"] is False
    assert report["gpu_started"] is False
    assert report["runtime_output_created"] is False
    assert report["metadata_audit"]["official_ca_ap_implementation_sha256"] == (
        route.EXPECTED_OFFICIAL_CA_AP_SHA256
    )


def test_route_receipts_allow_r4_stop_l6_pass_but_forbid_terminal_policy():
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    table = value["pre_fold1_result_freeze_barrier"]["deterministic_route_table"]
    assert table["r4_scientific_stop_l6_pass"] == "baseline_plus_l6"
    receipt = value["pre_fold1_route_lock"]["route_receipt_requirements"][
        "baseline_plus_l6"
    ]
    assert receipt["r4_result"] == {
        "kind": "scientific_stop",
        "schema": "boxfusion.ca1m_tr3d_terminal_gate_stop.v5.final.r4",
        "status": "STOP_FOLD234_OOF_GATE_FAIL",
        "terminal_inactive": True,
    }
    assert receipt["l6_result"]["kind"] == "pass_on_identity_anchors"
    assert receipt["terminal_policy"] == "forbidden"
    assert receipt["terminal_threshold_receipt_application"] == "forbidden"
    assert receipt["terminal_materialization"] == "forbidden_identity_anchors_only"
    gate = value["fold1_metric_and_route_gates"]["route_gates"]["baseline_plus_l6"]
    assert gate["terminal_policy"] == gate["terminal_materialization"] == "forbidden"
    assert not any(
        "terminal_all100_refit" in step
        for step in value["canonical_deploy_after_fold1_pass"]
        ["deployment_order_by_route"]["baseline_plus_l6"]
    )


def test_threshold_semantics_forbid_search_selection_retuning_but_allow_application():
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    barrier = value["pre_fold1_result_freeze_barrier"]
    one_time = value["locked_fold1_one_time_check"]
    deploy = value["canonical_deploy_after_fold1_pass"]
    assert barrier["fold0_threshold_search_selection_retuning"] is False
    assert barrier["fold1_threshold_search_selection_retuning"] is False
    assert one_time["fold0_threshold_search_selection_retuning"] is False
    assert one_time["fold1_threshold_search_selection_retuning"] is False
    for name in ("terminal_refit", "incremental_refit"):
        refit = deploy[name]
        assert refit["fold0_or_fold1_threshold_search_selection_retuning"] is False
        assert refit["frozen_threshold_application_allowed"] is True


def test_incremental_rows_are_causal_stage6_confirmed_tracks():
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    check = value["locked_fold1_one_time_check"]
    assert check["incremental_observer_source"] == (
        "causal_lightweight_stage6_observer_from_locked_holdout1_iter_11268_only"
    )
    assert check["incremental_candidate_source"] == (
        "causal_lightweight_stage6_confirmed_tracks_from_locked_holdout1_iter_11268_only"
    )
    rows = value["canonical_deploy_after_fold1_pass"]["incremental_refit"][
        "training_rows"
    ]
    assert rows == (
        "five_fold_causal_lightweight_stage6_confirmed_track_oof_collection_after_"
        "route_specific_five_fold_terminal_or_identity_oof_state"
    )
    assert "raw_detector_oof_candidate_universe" not in rows


def test_static_audit_opens_only_known_txt_and_never_b6_f1_or_val(monkeypatch):
    opened: list[Path] = []
    original = route._stable_bytes

    def recording(path: Path, name: str) -> bytes:
        opened.append(Path(path))
        return original(path, name)

    monkeypatch.setattr(route, "_stable_bytes", recording)
    route.validate_pending_config()
    opened_txt = {path.name for path in opened if path.suffix == ".txt"}
    assert opened_txt == {
        "e961_rank100_1060.txt", "e941_outer_rank100_1040.txt",
        "fold0_heldout.txt", "fold2.txt", "fold3.txt", "fold4.txt",
        "outer_dev_train1001.txt", "inner_holdout2_train1001.txt",
        "inner_holdout3_train1001.txt", "inner_holdout4_train1001.txt",
    }
    assert B6_MANIFEST not in opened
    assert not any(path.suffix == ".npz" for path in opened)
    assert not any(path.name == "locked_internal_check_scenes.txt" for path in opened)
    assert not any(path.name == "ca1m_val_full107.txt" for path in opened)


def test_pending_has_zero_fold1_val_or_l6_gate_source_path_loader():
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    for name in ("fold1", "official_validation"):
        commitment = value["opaque_heldout_commitments"][name]
        assert commitment["path"] is None
        assert commitment["loader"] is None
        assert commitment["opened"] is False
    l6 = value["static_bindings"]["incremental_l6_static_protocol"]
    assert l6["path"] is None and l6["sha256"] is None
    locked_gate = value["locked_fold1_one_time_check"]["l6_locked_gate_binding"]
    assert locked_gate["source_protocol_path"] is None
    assert locked_gate["source_protocol_sha256"] is None
    assert locked_gate["exact_json_subtree"] is None
    assert locked_gate["exact_json_subtree_sha256"] is None
    serialized = json.dumps(value, sort_keys=True)
    assert "locked_internal_check_scenes.txt" not in serialized
    assert "ca1m_val_full107.txt" not in serialized


def test_e921_f0234_locked_detector_order_and_sha_exactly():
    e961 = _rows("e961_rank100_1060.txt")
    f0, f2, f3, f4 = (
        _rows("fold0_heldout.txt"), _rows("fold2.txt"),
        _rows("fold3.txt"), _rows("fold4.txt"),
    )
    result = route.compose_locked_fold1_train(e961, f0, f2, f3, f4)
    assert len(result) == len(set(result)) == 1001
    assert result[:921] == e961[:921]
    assert result[921:941] == f0
    assert result[941:961] == f2
    assert result[961:981] == f3
    assert result[981:1001] == f4
    assert _sha(result) == route.EXPECTED_F1_TRAIN_SHA256


def test_e901_all100_future_order_without_accessing_real_fold1():
    e961 = _rows("e961_rank100_1060.txt")
    f0, f2, f3, f4 = (
        _rows("fold0_heldout.txt"), _rows("fold2.txt"),
        _rows("fold3.txt"), _rows("fold4.txt"),
    )
    used = set(e961) | set(f0) | set(f2) | set(f3) | set(f4)
    synthetic_f1 = tuple(
        value for value in (f"{90_000_000 + index:08d}" for index in range(1000))
        if value not in used
    )[:20]
    result = route.compose_canonical_deploy_train(
        e961, f0, synthetic_f1, f2, f3, f4,
    )
    assert len(result) == len(set(result)) == 1001
    assert result[:901] == e961[:901]
    assert result[901:921] == f0
    assert result[921:941] == synthetic_f1
    assert result[941:961] == f2
    assert result[961:981] == f3
    assert result[981:1001] == f4


# Independent fail-closed mutations spanning every required key/subtree class.
BAD_MUTATIONS = [
    pytest.param(
        _extra(("authorizations",), "debug_authorization", False),
        id="01_authorization_extra_key",
    ),
    pytest.param(
        _set(("access_at_static_stage", "native_b6_manifest_opened"), True),
        id="02_access_escape",
    ),
    pytest.param(
        _set(("locked_fold1_detector", "seed"), 1),
        id="03_locked_detector_recipe",
    ),
    pytest.param(
        _set(("pre_fold1_result_freeze_barrier", "deterministic_route_table",
              "r4_scientific_stop_l6_pass"), "no_fold1"),
        id="04_freeze_route_table",
    ),
    pytest.param(
        _set(("pre_fold1_result_freeze_barrier",
              "fold1_threshold_search_selection_retuning"), True),
        id="05_freeze_threshold_retuning",
    ),
    pytest.param(
        _set(("pre_fold1_route_lock", "route_receipt_requirements", "terminal_only",
              "r4_result", "status"), "PASS"),
        id="06_terminal_only_receipt",
    ),
    pytest.param(
        _set(("pre_fold1_route_lock", "route_receipt_requirements",
              "baseline_plus_l6", "terminal_policy"), "required"),
        id="07_baseline_l6_terminal_forbidden",
    ),
    pytest.param(
        _set(("locked_fold1_one_time_check", "claim",
              "created_before_fold1_source_path_resolution"), False),
        id="08_one_time_claim_order",
    ),
    pytest.param(
        _set(("locked_fold1_one_time_check", "terminal_locked_gate",
              "min_delta_ap50"), 0.0),
        id="09_terminal_locked_gate",
    ),
    pytest.param(
        _set(("fold1_metric_and_route_gates", "route_gates", "terminal_only",
              "l6_policy"), "optional"),
        id="10_terminal_only_gate",
    ),
    pytest.param(
        _set(("fold1_metric_and_route_gates", "route_gates", "terminal_plus_l6",
              "end_to_end_min_delta_ap50"), 0.0),
        id="11_terminal_plus_l6_gate",
    ),
    pytest.param(
        _set(("fold1_metric_and_route_gates", "route_gates", "baseline_plus_l6",
              "terminal_materialization"), "allowed"),
        id="12_baseline_plus_l6_gate",
    ),
    pytest.param(
        _set(("fold1_metric_and_route_gates", "official_evaluator",
              "implementation_sha256"), "0" * 64),
        id="13_official_evaluator_sha",
    ),
    pytest.param(
        _set(("canonical_deploy_after_fold1_pass", "detector", "optimizer_updates"),
             11267),
        id="14_canonical_detector",
    ),
    pytest.param(
        _set(("canonical_deploy_after_fold1_pass", "terminal_refit",
              "fold0_or_fold1_threshold_search_selection_retuning"), True),
        id="15_terminal_refit_retuning",
    ),
    pytest.param(
        _set(("canonical_deploy_after_fold1_pass", "incremental_refit",
              "training_rows"), "raw_detector_oof_candidate_universe"),
        id="16_incremental_refit_rows",
    ),
    pytest.param(
        _set(("canonical_deploy_after_fold1_pass", "deployment_order_by_route",
              "baseline_plus_l6"), ["terminal_all100_refit"]),
        id="17_deployment_order",
    ),
    pytest.param(
        _set(("future_artifacts", "static_protocol"), "/tmp/escape.json"),
        id="18_future_artifact_namespace",
    ),
]


@pytest.mark.parametrize("mutation", BAD_MUTATIONS)
def test_bad_mutations_fail_closed(tmp_path: Path, mutation):
    with pytest.raises((ValueError, PermissionError), match="exact"):
        route.validate_pending_config(_mutated_config(tmp_path, mutation))


def test_l6_protocol_and_gate_must_remain_null_until_separate_revision(tmp_path: Path):
    def mutate(value: dict) -> None:
        value["static_bindings"]["incremental_l6_static_protocol"].update({
            "state": "bound",
            "path": str(ROOT / "config/ca1m_e961_incremental_l6_v2_pending.json"),
            "sha256": "0" * 64,
        })

    with pytest.raises(ValueError, match="exact"):
        route.validate_pending_config(_mutated_config(tmp_path, mutate))


def test_preflight_default_pass_and_require_sealable_exit3():
    default = subprocess.run(
        [sys.executable, str(PREFLIGHT)], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert default.returncode == 0, default.stderr
    assert json.loads(default.stdout)["static_protocol_sealable"] is False
    pending = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--require-sealable"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert pending.returncode == 3, pending.stderr
    help_result = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--help"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert help_result.returncode == 0
    for forbidden in ("--run", "--train", "--fold1-path", "--gt", "--device"):
        assert forbidden not in help_result.stdout
    for forbidden_name in (
        "run", "train_detector", "load_ground_truth", "open_fold1",
        "authorize", "seal_preregistration",
    ):
        assert not hasattr(route, forbidden_name)

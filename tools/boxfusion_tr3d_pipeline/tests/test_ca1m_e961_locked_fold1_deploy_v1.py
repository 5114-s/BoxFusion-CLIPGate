from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from boxfusion import ca1m_e961_locked_fold1_deploy_v1 as route


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/ca1m_e961_locked_fold1_deploy_v1_pending.json"
PREFLIGHT = ROOT / "tools/preflight_ca1m_e961_locked_fold1_deploy_v1.py"
SPLITS = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/data/tr3d_ca1m_e961_v1/splits"
)


def _rows(name: str) -> tuple[str, ...]:
    return tuple((SPLITS / name).read_text(encoding="ascii").splitlines())


def _sha(rows: tuple[str, ...]) -> str:
    return hashlib.sha256("".join(f"{row}\n" for row in rows).encode("ascii")).hexdigest()


def _config(tmp_path: Path, mutate) -> Path:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    mutate(value)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_static_design_passes_but_is_not_sealable_before_final_l6_protocol():
    report = route.validate_pending_config()
    assert report["static_design_pass"] is True
    assert report["static_protocol_sealable"] is False
    assert report["static_protocol_seal_blocker"] == (
        "final_incremental_l6_static_protocol_sha256"
    )
    assert report["operational_authority"] is False
    assert report["fold1_canonical_scene_list_opened"] is False
    assert report["fold1_ground_truth_or_prediction_opened"] is False
    assert report["official_validation_opened"] is False
    assert report["gpu_started"] is False
    assert report["output_created"] is False
    assert report["metadata_audit"]["b6_fold1_model_sha256"] == (
        route.EXPECTED_B6_F1_MODEL_SHA256
    )
    assert report["metadata_audit"]["b6_fold1_train_heldout_overlap"] == 0
    assert report["metadata_audit"]["b6_sidecar_opened"] is False
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    check = value["locked_fold1_one_time_check"]
    assert check["incremental_candidate_source"] == (
        "causal_lightweight_stage6_confirmed_tracks_from_locked_holdout1_iter_11268_only"
    )
    assert value["canonical_deploy_after_fold1_pass"]["incremental_refit"][
        "training_rows"
    ].startswith("five_fold_causal_lightweight_stage6_confirmed_track_oof_collection")


def test_static_audit_opens_only_allowlisted_known_txt_splits(monkeypatch):
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
    assert not any(path.suffix == ".npz" for path in opened)
    assert not any(path.name == "locked_internal_check_scenes.txt" for path in opened)
    assert not any(path.name == "ca1m_val_full107.txt" for path in opened)


def test_pending_config_has_zero_fold1_or_val_source_path_and_loader():
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
    assert locked_gate["exact_json_subtree"] is None
    serialized = json.dumps(value, sort_keys=True)
    assert "locked_internal_check_scenes.txt" not in serialized
    assert "ca1m_val_full107.txt" not in serialized


def test_e921_f0234_locked_detector_order_and_sha_recompute_exactly():
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


def test_e901_all100_future_deploy_order_recomputes_without_real_fold1_access():
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


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda value: value["locked_fold1_detector"].__setitem__(
                "optimizer_updates", 11267
            ),
            "locked-F1 detector science",
        ),
        (
            lambda value: value["pre_fold1_result_freeze_barrier"][
                "deterministic_route_table"
            ].__setitem__("r4_scientific_stop_l6_pass", "no_fold1"),
            "result-freeze state machine",
        ),
        (
            lambda value: value["opaque_heldout_commitments"]["fold1"].__setitem__(
                "path", "/tmp/forbidden-fold1.txt"
            ),
            "opaque commitment",
        ),
        (
            lambda value: value["locked_fold1_one_time_check"].__setitem__(
                "post_fold1_branch_switch_or_fallback", True
            ),
            "one-time locked-F1",
        ),
        (
            lambda value: value["fold1_metric_and_tie_contract"].__setitem__(
                "ranking", "stable_argsort"
            ),
            "metric/tie",
        ),
    ],
)
def test_science_or_isolation_mutations_fail_closed(
    tmp_path: Path, mutation, message: str,
):
    with pytest.raises((ValueError, PermissionError), match=message):
        route.validate_pending_config(_config(tmp_path, mutation))


def test_l6_pending_config_cannot_be_substituted_for_final_static_protocol(tmp_path: Path):
    def mutate(value: dict) -> None:
        row = value["static_bindings"]["incremental_l6_static_protocol"]
        row.update({
            "state": "bound",
            "path": str(ROOT / "config/ca1m_e961_incremental_l6_v2_pending.json"),
            "sha256": "bc5b824b4f4271b49766d65f422527665917e4b12f8e48531061c1eb1537bfc3",
        })

    with pytest.raises(ValueError, match="unbound blocker"):
        route.validate_pending_config(_config(tmp_path, mutate))


def test_preflight_has_no_runtime_or_fold1_loader_interface():
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
    assert pending.returncode == 3
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

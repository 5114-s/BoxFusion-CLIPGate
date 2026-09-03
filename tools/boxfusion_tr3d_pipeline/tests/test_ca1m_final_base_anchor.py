from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

from boxfusion.tr3d_terminal_active import (
    link_prediction_create_only,
    save_prediction_create_only,
)


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit_ca1m_final_base_anchor.py"
CONTROL = ROOT / "config" / "ca1m_c4_final_base_g0_control_fixed10_v1.yaml"
ACTIVE = ROOT / "config" / "ca1m_c4_final_base_g0_clip_topk3_fixed10_v1.yaml"
TRAIN = ROOT / "config" / "ca1m_native_final_base_train100_v1.yaml"


def load_tool():
    spec = importlib.util.spec_from_file_location("ca1m_final_base_audit_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_is_ca_native_frozen_and_requires_downstream_retraining():
    report = load_tool().audit_contract(CONTROL, ACTIVE, TRAIN)
    assert report["ok"] is True
    assert report["ground_truth_access"] is False
    assert report["training_invoked"] is False
    assert report["learned_dataset_specific_assets"] == []
    assert report["scannet_learned_b6_or_gate_reused"] is False
    assert report["ca_geometry_contract"]["axis_alignment_required"] is False
    assert report["modules"]["clip_appearance_gate"]["training_required"] is False
    assert report["modules"]["reliable_view_topk3"]["top_k"] == 3
    assert report["downstream_contract"] == {
        "native_b6_recollection_required": True,
        "native_b6_retraining_required": True,
        "old_ca_b6_activation_authorized": False,
    }


def test_fixed10_control_and_active_differ_only_by_the_two_frozen_modules():
    control = yaml.safe_load(CONTROL.read_text())
    active = yaml.safe_load(ACTIVE.read_text())
    assert "appearance_gate" not in control["association"]
    assert "reliable_views" not in control["box_fusion"]
    assert active["association"]["appearance_gate"]["enabled"] is True
    reliable = active["box_fusion"]["reliable_views"]
    assert reliable["enabled"] is True
    assert reliable["top_k"] == reliable["min_views"] == 3
    assert active["association"]["small_threshold"] == 0.2
    assert active["box_fusion"]["small_size"] == 0.5
    assert active["online_refinement"] == {"enabled": False}
    assert active["ca1m_native_b6_observer"] == {"enabled": False}


def test_train100_is_the_same_algorithm_with_isolated_outputs_and_ca_cache():
    train = yaml.safe_load(TRAIN.read_text())
    assert "ca1m_native_final_base_train100_v1" in train["data"]["output_dir"]
    cache = train["lifting"]["proposal_cache"]
    assert cache["mode"] == "replay"
    assert cache["namespace"] == "ca1m-native-b6-train100-score04-gap20-cutr-v1"
    assert train["ca1m_native_b6_observer"] == {"enabled": False}
    checkpoint_keys = []
    def walk(value, path=()):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "checkpoint":
                    checkpoint_keys.append(".".join(path + (key,)))
                walk(child, path + (key,))
    walk(train)
    assert checkpoint_keys == ["lifting.boxer.checkpoint"]


def _box(center=(0.0, 0.0, 2.0), extent=(1.0, 2.0, 1.0)):
    signs = np.asarray(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=np.float32,
    )
    return np.asarray(center, np.float32) + signs * (np.asarray(extent, np.float32) / 2)


def test_identity_audit_proves_same_finalizer_bytes_without_forcing_control_identity(tmp_path):
    tool = load_tool()
    scene = "42446540"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(f"{scene}\n")
    control_root = tmp_path / "control"
    active_root = tmp_path / "active"
    identity_root = tmp_path / "identity"
    boxer_root = tmp_path / "boxer"
    log_root = tmp_path / "logs"
    control_boxer_root = tmp_path / "control_boxer"
    control_log_root = tmp_path / "control_logs"
    for root in (
        control_root,
        active_root,
        identity_root,
        boxer_root,
        log_root,
        control_boxer_root,
        control_log_root,
    ):
        root.mkdir()
    save_prediction_create_only(
        np.stack((_box(extent=(2.0, 1.0, 1.0)),)),
        np.asarray([0.60], np.float32),
        control_root / f"{scene}_boxes.pkl",
    )
    active_path = save_prediction_create_only(
        np.stack((_box(extent=(1.0, 2.0, 1.0)),)),
        np.asarray([0.60], np.float32),
        active_root / f"{scene}_boxes.pkl",
    )
    link_prediction_create_only(active_path, identity_root / f"{scene}_boxes.pkl")
    boxer_row = json.dumps(
        {
            "scene_id": scene,
            "mode": "active",
            "selective_gate_enabled": True,
            "boxer_checkpoint_sha256": tool.BOXER_SHA,
            "count": 1,
            "eligible_count": 1,
            "applied_count": 1,
            "fallback_count": 0,
        }
    ) + "\n"
    (boxer_root / f"{scene}_boxer_lifting.jsonl").write_text(boxer_row)
    (control_boxer_root / f"{scene}_boxer_lifting.jsonl").write_text(boxer_row)
    (log_root / f"{scene}.log").write_text(
        "Appearance gate summary | spatial\n"
        "Reliable-view fusion summary | updates=1\n"
        "Prediction same-run byte-identity anchor saved to x\n"
    )
    (control_log_root / f"{scene}.log").write_text("Selective Boxer control\n")
    report = tool.audit_identity(
        Namespace(
            scene_list=scene_list,
            expected_scenes=1,
            split="unit",
            control_root=control_root,
            control_boxer_root=control_boxer_root,
            control_log_root=control_log_root,
            active_root=active_root,
            identity_root=identity_root,
            boxer_root=boxer_root,
            log_root=log_root,
        )
    )
    assert report["same_run"]["hard_link_identity_scenes"] == 1
    assert report["paired_g0_control"]["identity_expected"] is False
    assert report["paired_g0_control"]["scenes_with_any_difference"] == 1
    assert report["evaluation_invoked"] is False


def test_boxer_audit_accepts_zero_proposal_rows_without_checkpoint_metadata(tmp_path):
    tool = load_tool()
    scene = "42446540"
    diagnostic = tmp_path / f"{scene}_boxer_lifting.jsonl"
    rows = [
        {
            "scene_id": scene,
            "mode": "active",
            "selective_gate_enabled": True,
            "count": 0,
            "eligible_count": 0,
            "applied_count": 0,
            "fallback_count": 0,
            "boxer_checkpoint_sha256": None,
            "boxer_commit": None,
        },
        {
            "scene_id": scene,
            "mode": "active",
            "selective_gate_enabled": True,
            "count": 1,
            "eligible_count": 1,
            "applied_count": 1,
            "fallback_count": 0,
            "boxer_checkpoint_sha256": tool.BOXER_SHA,
        },
    ]
    diagnostic.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert tool.audit_boxer(diagnostic, scene) == {
        "calls": 2,
        "eligible": 1,
        "applied": 1,
        "fallback": 0,
    }


def test_runner_orders_no_gt_identity_before_paired_evaluation():
    runner = (ROOT / "scripts" / "run_ca1m_c4_final_base_fixed10.sh").read_text()
    identity_call = runner.index("identity_and_paired_audit.json")
    eval_view = runner.index('mkdir "$EVAL_VIEW"')
    evaluate_call = runner.index('evaluate control "$CONTROL_ROOT"')
    assert identity_call < eval_view < evaluate_call
    assert "--prediction-same-run-anchor-root" in runner
    assert "--control-root" in runner


def test_train100_runner_has_no_evaluator_and_requires_explicit_run():
    runner = (
        ROOT / "scripts" / "collect_ca1m_native_final_base_train100.sh"
    ).read_text()
    assert 'MODE="preflight"' in runner
    assert "eval_ca1m.py" not in runner
    assert "--prediction-same-run-anchor-root" in runner
    assert "downstream native B6: recollection and CA-only retraining required" in runner


def test_demo_exposes_plain_finalizer_identity_without_enabling_observers():
    source = (ROOT / "demo.py").read_text()
    assert "--prediction-same-run-anchor-root" in source
    assert "generic prediction identity requires the plain finalizer" in source
    assert "link_prediction_create_only(output_path, identity_path)" in source

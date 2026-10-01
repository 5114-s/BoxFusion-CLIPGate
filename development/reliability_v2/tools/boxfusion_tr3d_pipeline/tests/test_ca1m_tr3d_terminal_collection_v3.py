from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest

from boxfusion import ca1m_tr3d_checkpoint_binding as binding
from tools import run_ca1m_tr3d_terminal_observer_v3 as observer_v3
from tools.preflight_ca1m_tr3d_terminal_train100_v3 import validate_config
from tools.seal_ca1m_tr3d_checkpoint_binding import write_json_create_only


ROOT = Path(__file__).resolve().parents[1]
BINDING_PATH = (
    ROOT
    / "manifests/ca1m_tr3d_terminal_ca_native_train100_v3/checkpoint_binding.json"
)
DEV_V4_REPORT = Path(
    "/extra/ZhaoX/tr3d_ca1m_dev_dumps/"
    "ca1m_fg_scratch_seed0_fp32_gb16_v1_fold0_dev_v4/"
    "ca1m_fold0_dev_ca_ap.json"
)


def _effective_config(work_root: Path, *, batch: int = 8) -> str:
    return f"""
load_from = None
resume = False
model = dict(
    type='MinkSingleStage3DDetector',
    bbox_head=dict(type='TR3DClassAgnosticHead', num_reg_outs=6),
)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0001),
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=12)
train_dataloader = dict(
    batch_size={batch},
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='RepeatDataset',
        times=15,
        dataset=dict(
            type='TR3DForegroundCA1MDataset',
            ann_file='annotations/ca1m_infos_weights_train_foreground.pkl',
        ),
    ),
)
test_cfg = None
test_dataloader = None
test_evaluator = None
randomness = dict(deterministic=True, seed=0)
work_dir = {str(work_root)!r}
"""


def _driver_log(*, exit_marker: bool = True) -> str:
    rows = [
        "Genuine CA-1M-native TR3D foreground scratch training",
        "  GPUs: 0,1 (2 processes)",
        "  initialization: random scratch (no ScanNet checkpoint/module)",
        "  resume: 0; AMP: 0",
        "  per-GPU/global batch: 8/16; workers: 4",
        "  precision/CuBLAS: FP32/:4096:8",
        "  protocol: fixed 12 epochs; only epoch_12 checkpoint; dev AP diagnostic",
        "Fixed CA-1M epoch-12 checkpoint completed: /tmp/epoch_12.pth",
    ]
    if exit_marker:
        rows.append("TRAIN_EXIT=0")
    return "\n".join(rows) + "\n"


def _fake_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    work_root = tmp_path / "work"
    work_root.mkdir()
    source_config = tmp_path / "tr3d_ca1m_foreground.py"
    source_config.write_text("load_from = None\n", encoding="utf-8")
    effective = work_root / binding.EXPECTED_EFFECTIVE_CONFIG_NAME
    effective.write_text(_effective_config(work_root), encoding="utf-8")
    (work_root / binding.EXPECTED_CHECKPOINT_NAME).write_bytes(b"ca-native-epoch12")
    training_log = tmp_path / "driver.log"
    training_log.write_text(_driver_log(), encoding="utf-8")
    monkeypatch.setattr(binding, "EXPECTED_WORK_ROOT", work_root)
    monkeypatch.setattr(binding, "EXPECTED_SOURCE_CONFIG", source_config)
    monkeypatch.setattr(
        binding,
        "EXPECTED_SOURCE_CONFIG_SHA256",
        binding.sha256_file(source_config),
    )
    return source_config, training_log


def test_checkpoint_binding_round_trip_is_create_only_and_ca_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_config, training_log = _fake_run(tmp_path, monkeypatch)
    payload = binding.build_binding_payload(
        work_root=binding.EXPECTED_WORK_ROOT,
        source_config=source_config,
        training_log=training_log,
    )
    assert payload["initialization"] == {
        "kind": "random_scratch",
        "load_from": None,
        "pretrained_or_init_cfg": False,
        "scannet_trained_module_access": False,
    }
    assert payload["training"]["global_batch"] == 16
    assert payload["training"]["precision"] == "fp32"
    assert payload["metric_protocol"]["training_dev_evaluator"] == "mmdet3d.IndoorMetric"
    assert "box3d_iou_v2" in payload["metric_protocol"]["ca_official_evaluator"]
    assert payload["metric_protocol"]["metrics_are_not_interchangeable"] is True
    target = tmp_path / "binding.json"
    write_json_create_only(target, payload, "test binding")
    assert target.stat().st_mode & 0o222 == 0
    loaded = binding.load_checkpoint_binding(target)
    assert loaded.checkpoint_sha256 == payload["checkpoint"]["sha256"]
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_json_create_only(target, payload, "test binding")
    assert target.read_bytes() == original


def test_checkpoint_binding_requires_real_success_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_config, training_log = _fake_run(tmp_path, monkeypatch)
    training_log.write_text(_driver_log(exit_marker=False), encoding="utf-8")
    with pytest.raises(ValueError, match="TRAIN_EXIT=0"):
        binding.build_binding_payload(
            work_root=binding.EXPECTED_WORK_ROOT,
            source_config=source_config,
            training_log=training_log,
        )


def test_checkpoint_binding_rejects_forbidden_scannet_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_config, training_log = _fake_run(tmp_path, monkeypatch)
    checkpoint = binding.EXPECTED_WORK_ROOT / binding.EXPECTED_CHECKPOINT_NAME
    monkeypatch.setattr(
        binding, "FORBIDDEN_SCANNET_SHA256", frozenset({binding.sha256_file(checkpoint)})
    )
    with pytest.raises(ValueError, match="forbidden ScanNet"):
        binding.build_binding_payload(
            work_root=binding.EXPECTED_WORK_ROOT,
            source_config=source_config,
            training_log=training_log,
        )


def test_real_v3_static_preflight_is_exact100_and_gpu_unready():
    report = validate_config(
        ROOT / "config/ca1m_tr3d_terminal_train100_v3.json", None
    )
    assert report["scene_count"] == 100
    assert report["static_preflight"] is True
    assert report["ready_for_gpu"] is False
    assert report["ground_truth_access"] is False
    assert report["validation_ground_truth_access"] is False
    assert report["forbidden_v1_v2_reuse"] is True


def test_formal_binding_is_valid_but_final_route_keeps_gpu_blocked():
    report = validate_config(
        ROOT / "config/ca1m_tr3d_terminal_train100_v3.json", BINDING_PATH
    )
    assert report["scene_count"] == 100
    assert report["checkpoint_binding_valid"] is True
    assert report["ready_for_gpu"] is False
    assert report["run_authorized"] is False
    assert "g0_clip_reliable_topk3_anchor_manifest_pending" in report["blocked_reasons"]
    assert report["checkpoint_binding"]["checkpoint_sha256"] == (
        "d3ba6cc22f0a1a11ab47e55ccdd21c2ef4a84efaf3c6359b7e8231a6c8d3b4a7"
    )
    assert BINDING_PATH.stat().st_mode & 0o222 == 0


def test_v4_ca_metric_receipt_is_diagnostic_only_and_internally_recomputed(
    tmp_path: Path,
):
    payload = binding.build_dev_diagnostic_receipt(
        binding_path=BINDING_PATH, dev_report_path=DEV_V4_REPORT
    )
    assert payload["preferred_revision"] == "v4_cpu_safe"
    assert payload["prediction_count"] == 5061
    assert payload["ap"] == {
        "ap15": 0.14674587321548369,
        "ap25": 0.08002573214889205,
        "ap50": 0.006813445463668614,
    }
    assert payload["authorization"] == {
        "diagnostic_only": True,
        "activation_authorized": False,
        "checkpoint_selection_authorized": False,
        "terminal_collection_authorized": False,
    }
    target = tmp_path / "dev_receipt.json"
    write_json_create_only(target, payload, "dev receipt")
    assert binding.load_dev_diagnostic_receipt(target) == payload


def test_v3_config_has_isolated_outputs_and_no_terminal_cache_input():
    config = json.loads(
        (ROOT / "config/ca1m_tr3d_terminal_train100_v3.json").read_text()
    )
    assert config["schema"].endswith(".v3")
    assert config["tr3d"]["raw_checkpoint_argument_allowed"] is False
    assert config["tr3d"]["raw_config_argument_allowed"] is False
    assert config["tr3d"]["scannet_checkpoint_or_config_allowed"] is False
    assert "terminal_cache_root" not in config["inputs"]
    assert "anchor_root" not in config["inputs"]
    assert "native_b6_diagnostics_root" not in config["inputs"]
    assert config["run_authorized"] is False
    assert config["proposal_stage"]["anchor_access"] is False
    assert config["proposal_stage"]["b6_access"] is False
    assert config["proposal_stage"]["frame_lineage_manifest"] is None
    assert config["anchor_overlay_stage"]["final_anchor_root"] is None
    assert config["anchor_overlay_stage"]["retrained_native_b6_checkpoint"] is None
    assert all(
        config["namespace"] in path
        for key, path in config["outputs"].items()
        if key.endswith("_root")
    )


def test_v3_observer_parser_requires_binding_and_hides_raw_model_inputs():
    actions = {action.dest: action for action in observer_v3.parser()._actions}
    assert actions["tr3d_binding_manifest"].required is True
    assert actions["tr3d_config"].required is False
    assert actions["tr3d_checkpoint"].required is False
    assert actions["tr3d_config"].help == argparse.SUPPRESS
    sources = (ROOT / "tools/run_ca1m_tr3d_terminal_observer_v3.py").read_text()
    assert "checkpoint_binding" in sources
    assert "collection_config" in sources
    assert "a484fd79093aa" not in sources
    assert "tr3d_scannet_foreground.py" not in sources


def test_v3_observer_runtime_rejects_raw_checkpoint_before_any_model_access():
    args = argparse.Namespace(
        tr3d_config=Path("/tmp/forbidden_config.py"),
        tr3d_checkpoint=None,
    )
    with pytest.raises(ValueError, match="forbids raw"):
        observer_v3.run(args)


def test_v3_collection_launcher_is_static_first_and_valid_bash():
    path = ROOT / "scripts/collect_ca1m_tr3d_terminal_train100_v3.sh"
    result = subprocess.run(
        ["bash", "-n", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    source = path.read_text(encoding="utf-8")
    assert 'MODE="${1:---static-preflight}"' in source
    assert "BOXFUSION_CA1M_TR3D_V3_CHECKPOINT_BINDING" in source
    assert "raw BOXFUSION_TR3D_CHECKPOINT is forbidden" in source
    assert "--checkpoint-binding" in source
    assert "--tr3d-checkpoint" not in source
    assert "--tr3d-config" not in source
    assert "v3 run is not authorized" in source
    assert "ca1m_tr3d_benefit_train100_v1/terminal" not in source
    assert "ca1m_tr3d_benefit_train100_v2/terminal" not in source

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MATERIALIZE = ROOT / "tools" / "materialize_ca1m_native_b6_canonical_config.py"
INPUT_AUDIT = ROOT / "tools" / "audit_ca1m_native_b6_canonical_inputs.py"
ASSET_AUDIT = ROOT / "tools" / "audit_ca1m_native_b6_canonical_assets.py"


def test_canonical_templates_freeze_real_score_g0_and_independent_cache() -> None:
    record = yaml.safe_load(
        (ROOT / "config" / "ca1m_native_b6_canonical103_cutr_record.yaml").read_text()
    )
    observer = yaml.safe_load(
        (ROOT / "config" / "ca1m_native_b6_canonical103_g0_observer.yaml").read_text()
    )
    namespace = "ca1m-native-b6-canonical103-score04-gap20-cutr-v1"
    assert record["detection"]["score_thresh"] == 0.4
    assert observer["detection"]["score_thresh"] == 0.4
    assert record["data"]["gap"] == observer["data"]["gap"] == 20
    assert record["lifting"]["backend"] == "cutr"
    assert record["lifting"]["proposal_cache"] == {
        "mode": "record",
        "namespace": namespace,
        "root": str(ROOT / "cache" / "ca1m_native_b6_canonical103_v1"),
    }
    assert observer["lifting"]["proposal_cache"]["mode"] == "replay"
    assert observer["lifting"]["proposal_cache"]["namespace"] == namespace
    assert observer["lifting"]["boxer"]["selective_gate"] == {
        "enabled": True,
        "max_center_shift_m": 0.1,
        "min_volume_ratio": 0.5,
        "max_volume_ratio": 2.0,
    }
    assert observer["ca1m_native_b6_observer"]["observer_only"] is True


def test_materializer_is_create_only_and_phase_separated(tmp_path: Path) -> None:
    output = tmp_path / "observer.yaml"
    command = [
        sys.executable,
        str(MATERIALIZE),
        "--template",
        str(ROOT / "config" / "ca1m_native_b6_canonical103_g0_observer.yaml"),
        "--phase",
        "observer",
        "--data-root",
        str(tmp_path / "data"),
        "--output-root",
        str(tmp_path / "prediction"),
        "--cache-root",
        str(tmp_path / "cache"),
        "--baseline-root",
        str(tmp_path / "record"),
        "--native-diagnostics-root",
        str(tmp_path / "native"),
        "--boxer-diagnostics-root",
        str(tmp_path / "boxer"),
        "--output",
        str(output),
    ]
    assert subprocess.run(command, capture_output=True, text=True).returncode == 0
    cfg = yaml.safe_load(output.read_text())
    assert cfg["lifting"]["proposal_cache"]["baseline_prediction_root"] == str(
        tmp_path / "record"
    )
    assert subprocess.run(command, capture_output=True, text=True).returncode != 0


def test_gt_free_input_preflight_proves_official_103_plus_4(tmp_path: Path) -> None:
    canonical = [f"{42_000_000 + index:08d}" for index in range(103)]
    excluded = [f"{43_000_000 + index:08d}" for index in range(4)]
    scene_list = tmp_path / "canonical.txt"
    excluded_list = tmp_path / "excluded.txt"
    official = tmp_path / "val.txt"
    scene_list.write_text("\n".join(canonical) + "\n")
    excluded_list.write_text("\n".join(excluded) + "\n")
    official.write_text("\n".join(
        f"https://ml-site.cdn-apple.com/datasets/ca1m/val/ca1m-val-{scene}.tar"
        for scene in canonical + excluded
    ))
    data = tmp_path / "data"
    for scene in canonical:
        (data / scene).mkdir(parents=True)
    output = tmp_path / "audit.json"
    result = subprocess.run([
        sys.executable,
        str(INPUT_AUDIT),
        "--scene-list",
        str(scene_list),
        "--excluded-scene-list",
        str(excluded_list),
        "--official-url-list",
        str(official),
        "--data-root",
        str(data),
        "--preflight",
        "--output",
        str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["ground_truth_access"] is False
    assert payload["evaluation_invoked"] is False
    assert payload["scene_directories_present"] == 103
    assert payload["forbidden_inputs_opened"] == []


def test_existing_asset_audit_marks_legacy_cache_non_reusable(tmp_path: Path) -> None:
    scenes = [f"{42_000_000 + index:08d}" for index in range(103)]
    scene_list = tmp_path / "canonical.txt"
    scene_list.write_text("\n".join(scenes) + "\n")
    c0 = tmp_path / "c0"
    c0.mkdir()
    for index, scene in enumerate(scenes):
        corners = __import__("numpy").zeros((8, 3), dtype="float32")
        score = 0.4 + 0.001 * index
        (c0 / f"{scene}_boxes.pkl").write_bytes(
            pickle.dumps([[(0, corners, score)]], protocol=pickle.HIGHEST_PROTOCOL)
        )
    cache = tmp_path / "ca1m-score04-gap20-c0-v2"
    for scene in scenes[:10]:
        root = cache / scene
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps({
            "scene_id": scene,
            "namespace": cache.name,
            "producer_fingerprint": "fixed10",
        }))
    output = tmp_path / "assets.json"
    result = subprocess.run([
        sys.executable,
        str(ASSET_AUDIT),
        "--scene-list",
        str(scene_list),
        "--c0-root",
        str(c0),
        "--legacy-cache-root",
        str(cache),
        "--output",
        str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["historical_c0"]["scenes"] == 103
    assert payload["historical_c0"]["real_score_variation"] is True
    assert payload["legacy_cache"]["canonical103_coverage"] == 10
    assert payload["legacy_cache"]["reusable_for_new_collection"] is False


def test_runner_has_no_evaluator_or_gt_input_path() -> None:
    source = (ROOT / "scripts" / "collect_ca1m_native_b6_canonical103.sh").read_text()
    input_source = INPUT_AUDIT.read_text()
    assert "eval_ca1m.py" not in source
    assert "after_filter_boxes" not in source
    assert "after_filter_boxes" not in input_source
    assert "--run" in source and "--preflight" in source
    assert "BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT" in source
    assert "BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT" in source

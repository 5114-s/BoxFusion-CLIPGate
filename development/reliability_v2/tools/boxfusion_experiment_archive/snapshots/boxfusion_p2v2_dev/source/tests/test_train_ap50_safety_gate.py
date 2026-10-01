import numpy as np
import pytest

from boxfusion.ap50_safety_gate import load_ap50_safety_gate
from tools.train_ap50_safety_gate import (
    GateTrainingData,
    TRAINING_FORMAT_VERSION,
    TRAINING_SCHEMA,
    load_training_data,
    scene_disjoint_split,
    train_gate,
    validate_forbidden_scenes,
)


def _write_training_archive(path, *, scene_prefix="scene"):
    count = 24
    features = np.stack(
        (
            np.linspace(0.0, 1.0, count),
            np.linspace(1.0, 0.0, count),
            np.tile(np.asarray([0.2, 0.8]), count // 2),
        ),
        axis=1,
    ).astype(np.float32)
    original = np.linspace(0.15, 0.65, count).astype(np.float32)
    delta = (0.08 * (features[:, 0] - 0.45)).astype(np.float32)
    candidate = np.clip(original + delta, 0.0, 1.0)
    scenes = np.asarray(
        [f"{scene_prefix}{index % 4:04d}_00" for index in range(count)]
    )
    np.savez(
        path,
        schema=np.asarray(TRAINING_SCHEMA),
        format_version=np.asarray(TRAINING_FORMAT_VERSION, dtype=np.int64),
        feature_names=np.asarray(("support", "projection", "occupancy")),
        gate_features=features,
        original_iou=original,
        candidate_iou=candidate,
        scene_ids=scenes,
    )


def test_load_and_scene_disjoint_split(tmp_path):
    path = tmp_path / "training.npz"
    _write_training_archive(path)
    data = load_training_data((path,))
    assert data.sample_count == 24
    training, validation, train_scenes, val_scenes = scene_disjoint_split(
        data.scene_ids, validation_fraction=0.25, seed=7
    )
    assert training.any() and validation.any()
    assert not (training & validation).any()
    assert set(train_scenes).isdisjoint(val_scenes)
    assert set(data.scene_ids[training]) == set(train_scenes)
    assert set(data.scene_ids[validation]) == set(val_scenes)


def test_forbidden_validation_overlap_is_rejected(tmp_path):
    path = tmp_path / "training.npz"
    _write_training_archive(path)
    data = load_training_data((path,))
    with pytest.raises(ValueError, match="forbidden validation"):
        validate_forbidden_scenes(data, {"scene0001_00"})
    validate_forbidden_scenes(data, {"scene9999_00"})


def test_training_archive_schema_is_strict(tmp_path):
    path = tmp_path / "bad.npz"
    _write_training_archive(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["unexpected"] = np.asarray(1)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="keys mismatch"):
        load_training_data((path,))


def test_short_cpu_training_exports_runtime_checkpoint(tmp_path):
    pytest.importorskip("torch")
    source = tmp_path / "training.npz"
    output = tmp_path / "gate.npz"
    _write_training_archive(source)
    data = load_training_data((source,))
    summary = train_gate(
        data,
        output,
        validation_fraction=0.25,
        hidden_dims=(8,),
        epochs=4,
        learning_rate=5e-3,
        seed=11,
    )
    assert output.is_file()
    assert summary["training_scene_count"] == 3
    assert summary["validation_scene_count"] == 1
    gate = load_ap50_safety_gate(output)
    predictions = gate.predict(data.features[:3])
    assert len(predictions) == 3
    assert all(np.isfinite(item.delta_mean) for item in predictions)
    assert all(0.0 <= item.candidate_iou <= 1.0 for item in predictions)


def test_archive_feature_schema_must_match_when_concatenating(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_training_archive(first, scene_prefix="train")
    _write_training_archive(second, scene_prefix="other")
    with np.load(second, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["feature_names"] = np.asarray(("a", "b", "c"))
    np.savez(second, **arrays)
    with pytest.raises(ValueError, match="different feature schemas"):
        load_training_data((first, second))

import json
import hashlib

import numpy as np
import pytest

from boxfusion.ap50_safety_gate import (
    AP50_SAFETY_GATE_FORMAT_VERSION,
    AP50_SAFETY_GATE_OUTPUT_NAMES,
    AP50_SAFETY_GATE_SCHEMA,
)
from boxfusion.yidu_local_observer import (
    YIDU_GATE_FEATURE_DIM,
    YIDU_GATE_FEATURE_NAMES,
)
from tools.train_ap50_safety_gate import (
    TRAINING_FORMAT_VERSION,
    TRAINING_SCHEMA,
)
from tools.validate_yidu_gate_provenance import (
    main,
    validate_yidu_gate_provenance,
)


SCENES = (
    "scene0000_00",
    "scene0001_00",
    "scene0002_00",
    "scene0003_00",
)


def _write_scene_list(path, scenes):
    path.write_text("\n".join(scenes) + "\n", encoding="utf-8")


def _write_archive(path, *, feature_names=YIDU_GATE_FEATURE_NAMES):
    scene_ids = np.asarray(
        [
            SCENES[0],
            SCENES[0],
            SCENES[1],
            SCENES[1],
            SCENES[2],
            SCENES[2],
            SCENES[3],
            SCENES[3],
        ]
    )
    feature_names = tuple(feature_names)
    np.savez(
        path,
        schema=np.asarray(TRAINING_SCHEMA),
        format_version=np.asarray(
            TRAINING_FORMAT_VERSION, dtype=np.int64
        ),
        feature_names=np.asarray(feature_names),
        gate_features=np.zeros(
            (len(scene_ids), len(feature_names)), dtype=np.float32
        ),
        original_iou=np.linspace(
            0.1, 0.4, len(scene_ids), dtype=np.float32
        ),
        candidate_iou=np.linspace(
            0.2, 0.5, len(scene_ids), dtype=np.float32
        ),
        scene_ids=scene_ids,
    )


def _metadata():
    return {
        "training_schema": TRAINING_SCHEMA,
        "training_samples": 6,
        "validation_samples": 2,
        "training_scene_count": 3,
        "validation_scene_count": 1,
        "training_scenes": list(SCENES[:3]),
        "validation_scenes": [SCENES[3]],
    }


def _write_checkpoint(
    path,
    *,
    training_archive,
    feature_names=YIDU_GATE_FEATURE_NAMES,
    metadata=None,
):
    names = tuple(feature_names)
    metadata = _metadata() if metadata is None else metadata
    digest = hashlib.sha256(training_archive.read_bytes()).hexdigest()
    metadata = dict(metadata)
    metadata["training_archives"] = [
        {
            "path": str(training_archive.resolve()),
            "sha256": digest,
        }
    ]
    np.savez(
        path,
        schema=np.asarray(AP50_SAFETY_GATE_SCHEMA),
        format_version=np.asarray(
            AP50_SAFETY_GATE_FORMAT_VERSION, dtype=np.int64
        ),
        feature_names=np.asarray(names),
        output_names=np.asarray(AP50_SAFETY_GATE_OUTPUT_NAMES),
        feature_mean=np.zeros(len(names), dtype=np.float32),
        feature_scale=np.ones(len(names), dtype=np.float32),
        maximum_absolute_delta=np.asarray(1.0, dtype=np.float32),
        num_layers=np.asarray(1, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        weight_0=np.zeros(
            (len(names), len(AP50_SAFETY_GATE_OUTPUT_NAMES)),
            dtype=np.float32,
        ),
        bias_0=np.zeros(
            len(AP50_SAFETY_GATE_OUTPUT_NAMES), dtype=np.float32
        ),
    )


def _valid_inputs(tmp_path):
    archive = tmp_path / "training.npz"
    checkpoint = tmp_path / "gate.npz"
    train_list = tmp_path / "train.txt"
    forbidden_list = tmp_path / "forbidden.txt"
    _write_archive(archive)
    _write_checkpoint(checkpoint, training_archive=archive)
    _write_scene_list(train_list, (*SCENES, "scene0100_00"))
    _write_scene_list(forbidden_list, ("scene0900_00",))
    return checkpoint, archive, train_list, forbidden_list


def test_valid_provenance_report_and_cli(tmp_path, capsys):
    checkpoint, archive, train_list, forbidden = _valid_inputs(
        tmp_path
    )
    report = validate_yidu_gate_provenance(
        checkpoint=checkpoint,
        training_archive=archive,
        train_scene_list=train_list,
        forbidden_scene_list=forbidden,
    )
    assert report["valid"] is True
    assert report["feature_dim"] == YIDU_GATE_FEATURE_DIM == 91
    assert report["archive_samples"] == 8
    assert report["archive_scene_count"] == 4

    assert main(
        [
            "--checkpoint",
            str(checkpoint),
            "--training-archive",
            str(archive),
            "--train-scene-list",
            str(train_list),
            "--forbidden-scene-list",
            str(forbidden),
        ]
    ) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema"] == (
        "boxfusion.yidu.gate_provenance_report.v1"
    )
    assert printed["valid"] is True


def test_archive_must_use_exact_ordered_91d_schema(tmp_path):
    checkpoint, archive, train_list, forbidden = _valid_inputs(
        tmp_path
    )
    swapped = list(YIDU_GATE_FEATURE_NAMES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    _write_archive(archive, feature_names=swapped)
    with pytest.raises(ValueError, match="training archive feature schema"):
        validate_yidu_gate_provenance(
            checkpoint=checkpoint,
            training_archive=archive,
            train_scene_list=train_list,
            forbidden_scene_list=forbidden,
        )


def test_checkpoint_must_use_exact_ordered_91d_schema(tmp_path):
    checkpoint, archive, train_list, forbidden = _valid_inputs(
        tmp_path
    )
    swapped = list(YIDU_GATE_FEATURE_NAMES)
    swapped[-1], swapped[-2] = swapped[-2], swapped[-1]
    _write_checkpoint(
        checkpoint,
        training_archive=archive,
        feature_names=swapped,
    )
    with pytest.raises(ValueError, match="checkpoint feature schema"):
        validate_yidu_gate_provenance(
            checkpoint=checkpoint,
            training_archive=archive,
            train_scene_list=train_list,
            forbidden_scene_list=forbidden,
        )


def test_archive_scene_must_be_declared_train_only(tmp_path):
    checkpoint, archive, train_list, forbidden = _valid_inputs(
        tmp_path
    )
    _write_scene_list(train_list, SCENES[:3])
    with pytest.raises(ValueError, match="outside the declared training"):
        validate_yidu_gate_provenance(
            checkpoint=checkpoint,
            training_archive=archive,
            train_scene_list=train_list,
            forbidden_scene_list=forbidden,
        )


def test_forbidden_scene_overlap_fails_closed(tmp_path):
    checkpoint, archive, train_list, forbidden = _valid_inputs(
        tmp_path
    )
    _write_scene_list(forbidden, (SCENES[2],))
    with pytest.raises(ValueError, match="scene lists overlap"):
        validate_yidu_gate_provenance(
            checkpoint=checkpoint,
            training_archive=archive,
            train_scene_list=train_list,
            forbidden_scene_list=forbidden,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value.pop("training_scenes"),
            "missing provenance keys",
        ),
        (
            lambda value: value.update(validation_scenes=[]),
            "non-empty scene list",
        ),
        (
            lambda value: value.update(
                validation_scenes=[SCENES[0]],
                validation_scene_count=1,
                validation_samples=2,
            ),
            "internal training/validation scenes overlap",
        ),
        (
            lambda value: value.update(
                validation_scenes=["scene0200_00"]
            ),
            "scene union disagrees",
        ),
        (
            lambda value: value.update(training_scene_count=2),
            "training_scene_count disagrees",
        ),
        (
            lambda value: value.update(training_samples=5),
            "training_samples disagrees",
        ),
    ],
)
def test_metadata_inconsistency_fails_closed(
    tmp_path, mutation, match
):
    checkpoint, archive, train_list, forbidden = _valid_inputs(
        tmp_path
    )
    metadata = _metadata()
    mutation(metadata)
    _write_checkpoint(
        checkpoint,
        training_archive=archive,
        metadata=metadata,
    )
    with pytest.raises(ValueError, match=match):
        validate_yidu_gate_provenance(
            checkpoint=checkpoint,
            training_archive=archive,
            train_scene_list=train_list,
            forbidden_scene_list=forbidden,
        )


def test_training_archive_schema_remains_strict(tmp_path):
    checkpoint, archive, train_list, forbidden = _valid_inputs(
        tmp_path
    )
    with np.load(archive, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["unexpected"] = np.asarray(1)
    np.savez(archive, **arrays)
    with pytest.raises(ValueError, match="schema keys mismatch"):
        validate_yidu_gate_provenance(
            checkpoint=checkpoint,
            training_archive=archive,
            train_scene_list=train_list,
            forbidden_scene_list=forbidden,
        )


def test_checkpoint_is_cryptographically_bound_to_training_archive(
    tmp_path,
):
    checkpoint, archive, train_list, forbidden = _valid_inputs(
        tmp_path
    )
    with np.load(archive, allow_pickle=False) as payload:
        arrays = {
            name: np.array(payload[name], copy=True)
            for name in payload.files
        }
    arrays["candidate_iou"][0] += np.float32(0.001)
    np.savez(archive, **arrays)
    with pytest.raises(ValueError, match="archive SHA256 mismatch"):
        validate_yidu_gate_provenance(
            checkpoint=checkpoint,
            training_archive=archive,
            train_scene_list=train_list,
            forbidden_scene_list=forbidden,
        )

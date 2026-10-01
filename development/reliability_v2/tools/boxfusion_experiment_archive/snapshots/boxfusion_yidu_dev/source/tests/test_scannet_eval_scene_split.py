from __future__ import annotations

import pytest

from evaluation.eval_scannet import (
    dataset_split_for_scene_list,
    read_requested_scenes,
    validate_requested_scenes,
)


def test_default_evaluation_keeps_val_split() -> None:
    assert dataset_split_for_scene_list(None) == "val"


def test_explicit_scene_list_uses_all_prepared_gt_scans(tmp_path) -> None:
    scene_list = tmp_path / "train_scenes.txt"
    scene_list.write_text(
        "# train-only smoke set\n"
        "scene0191_00\n"
        "\n"
        "scene0119_00\n",
        encoding="utf-8",
    )

    requested = read_requested_scenes(scene_list)

    assert dataset_split_for_scene_list(scene_list) == "all"
    assert validate_requested_scenes(
        requested,
        ["scene0000_00", "scene0119_00", "scene0191_00"],
    ) == ["scene0191_00", "scene0119_00"]


def test_scene_list_duplicate_validation_is_unchanged(tmp_path) -> None:
    scene_list = tmp_path / "duplicate.txt"
    scene_list.write_text(
        "scene0191_00\nscene0191_00\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="--scene_list contains duplicate scene IDs",
    ):
        read_requested_scenes(scene_list)


def test_scene_list_missing_validation_is_unchanged() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "--scene_list contains unavailable scenes: "
            "scene0191_00, scene0119_00"
        ),
    ):
        validate_requested_scenes(
            ["scene0191_00", "scene0119_00"],
            ["scene0000_00"],
        )

"""OpenMMLab integration checks; skipped in the lightweight BoxFusion env."""

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "mmdetection3d"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VENDOR))

pytest.importorskip("mmengine")
pytest.importorskip("MinkowskiEngine")

from mmengine.config import Config  # noqa: E402
from mmdet3d.registry import DATASETS, MODELS  # noqa: E402
from mmdet3d.utils import register_all_modules  # noqa: E402


def test_foreground_config_builds_real_dataset_sample_and_model():
    register_all_modules(init_default_scope=True)
    config = Config.fromfile(
        str(ROOT / "config" / "tr3d" / "tr3d_scannet_foreground.py"))
    dataset = DATASETS.build(config.train_dataloader.dataset)
    sample = dataset[0]
    assert len(dataset) == 1001 * 15
    assert dataset.dataset.metainfo["classes"] == ("foreground",)
    assert sample["inputs"]["points"].shape[1] == 6
    assert sample[
        "data_samples"].gt_instances_3d.labels_3d.unique().tolist() == [0]

    model = MODELS.build(config.model)
    assert type(model.bbox_head).__name__ == "TR3DClassAgnosticHead"
    assert sum(parameter.numel() for parameter in model.parameters()) == 14659271


def test_final_config_uses_official_val_only_at_test_time():
    register_all_modules(init_default_scope=True)
    config = Config.fromfile(
        str(
            ROOT / "config" / "tr3d"
            / "tr3d_scannet_foreground_official_val.py"))
    assert (
        config.train_dataloader.dataset.dataset.ann_file
        == "annotations/scannet_infos_train_foreground.pkl")
    assert (
        config.test_dataloader.dataset.ann_file
        == "annotations/scannet_infos_official_val_foreground.pkl")

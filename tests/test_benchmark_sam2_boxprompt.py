from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import benchmark_sam2_boxprompt as benchmark


def _write_f0(tmp_path: Path, *, boxes: list[list[int]]) -> Path:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image_path = tmp_path / "rgb.png"
    assert cv2.imwrite(str(image_path), image)
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    candidates = [
        {
            "rank": rank,
            "raw_index": 10 + rank,
            "mask_sha256": f"{rank + 1:064x}",
            "tight_box_xyxy": box,
        }
        for rank, box in enumerate(boxes)
    ]
    payload = {
        "schema": "boxfusion.scannet_fastsam_f0_full200.scene.v1",
        "protocol_id": "F0-frozen-FastSAM-x-residual-automatic-mask-shadow-full200",
        "complete": True,
        "scene_id": "scene0000_00",
        "scene_index": 0,
        "frames": [
            {
                "frame_id": 0,
                "frame_ordinal": 0,
                "successful": True,
                "inputs": {"rgb_path": str(image_path), "rgb_sha256": image_sha},
                "funnel": {"candidates": candidates},
            }
        ],
    }
    sidecar = tmp_path / "f0.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar


def test_load_f0_prompts_preserves_rank_and_seal(tmp_path: Path) -> None:
    sidecar = _write_f0(tmp_path, boxes=[[1, 2, 20, 21], [2, 3, 22, 23]])
    image, image_sha, boxes, metadata = benchmark._load_f0_prompts(
        sidecar, frame_ordinal=0, prompt_count=2
    )
    assert image.name == "rgb.png"
    assert image_sha == hashlib.sha256(image.read_bytes()).hexdigest()
    assert boxes.shape == (2, 4)
    assert metadata["f0_sources"][1]["raw_index"] == 11


def test_load_f0_prompts_fails_on_rank_or_image_change(tmp_path: Path) -> None:
    sidecar = _write_f0(tmp_path, boxes=[[1, 2, 20, 21]])
    payload = json.loads(sidecar.read_text())
    changed = copy.deepcopy(payload)
    changed["frames"][0]["funnel"]["candidates"][0]["rank"] = 1
    sidecar.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(benchmark.SAM2BenchmarkError, match="rank/order"):
        benchmark._load_f0_prompts(sidecar, frame_ordinal=0, prompt_count=1)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    Path(payload["frames"][0]["inputs"]["rgb_path"]).write_bytes(b"changed")
    with pytest.raises(benchmark.SAM2BenchmarkError, match="hash differs"):
        benchmark._load_f0_prompts(sidecar, frame_ordinal=0, prompt_count=1)


def test_select_best_masks_is_per_prompt_and_tie_stable() -> None:
    masks = np.zeros((2, 3, 480, 640), dtype=bool)
    masks[0, 1, 2, 3] = True
    masks[1, 0, 4, 5] = True
    scores = np.asarray([[0.2, 0.8, 0.8], [0.9, 0.1, 0.2]])
    selected, selected_scores, best = benchmark._select_best_masks(
        masks, scores, prompt_count=2
    )
    assert best.tolist() == [1, 0]
    assert selected[:, 2, 3].tolist() == [True, False]
    assert selected[:, 4, 5].tolist() == [False, True]
    np.testing.assert_allclose(selected_scores, [0.8, 0.9])


def test_select_best_masks_rejects_wrong_geometry() -> None:
    with pytest.raises(benchmark.SAM2BenchmarkError, match="output shape"):
        benchmark._select_best_masks(
            np.zeros((2, 3, 10, 10), dtype=bool),
            np.zeros((2, 3), dtype=float),
            prompt_count=2,
        )

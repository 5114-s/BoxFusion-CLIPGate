from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from boxfusion.live_sam3_client import LiveSAM3Client, LiveSAM3Config
from boxfusion.sam3_diverse_maskdepth_birth import SAM3MemoryTeacherView
from boxfusion.stream3dv2_live import (
    NativeTargetMaskFrame,
    Stream3Dv2LiveConfig,
    Stream3Dv2LiveError,
    Stream3Dv2LiveRoute,
    _SemanticAvailability,
    build_stream3dv2_live_route,
    tm_fpf_c1_view_abstention_reason,
)
from boxfusion.tm_fpf_c1 import (
    TMFPFC1ContractError,
    match_fastsam_target_masks,
    resolve_tm_fpf_c1_config,
)


_CORNER_SIGNS = np.asarray(
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, +1.0),
        (-1.0, +1.0, -1.0),
        (-1.0, +1.0, +1.0),
        (+1.0, -1.0, -1.0),
        (+1.0, -1.0, +1.0),
        (+1.0, +1.0, -1.0),
        (+1.0, +1.0, +1.0),
    ],
    dtype=np.float64,
)


def _box(
    center: tuple[float, float, float],
    extent: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> np.ndarray:
    return np.asarray(center)[None] + _CORNER_SIGNS * np.asarray(extent)[None] * 0.5


def _config() -> Stream3Dv2LiveConfig:
    sam3 = LiveSAM3Config(enabled=False, device="cpu")
    return Stream3Dv2LiveConfig(
        enabled=True,
        fastsam_checkpoint="unused-in-tests.pt",
        native_score_lower_bound=0.125,
        keyframe_deadline_ms=833.333333,
        max_semantic_views=8,
        diagnostics_root=None,
        sam3_enabled=False,
        sam3_interval_keyframes=999,
        sam3_drain_timeout_seconds=0.001,
        sam3_config=sam3,
    )


class _BombFastSAM:
    def infer_bgr(self, image: np.ndarray) -> object:  # pragma: no cover - assertion path
        raise AssertionError("FastSAM must not run for an abstained/invalid frame")


class _BombF4:
    def infer_batch(self, *args: object, **kwargs: object) -> object:  # pragma: no cover
        raise AssertionError("F4 must not run for an abstained/invalid frame")


def _route(
    *, fastsam_provider: object | None = None, f4_provider: object | None = None
) -> Stream3Dv2LiveRoute:
    sam3 = LiveSAM3Client(LiveSAM3Config(enabled=False, device="cpu"))
    return Stream3Dv2LiveRoute(
        _config(),
        lifting_adapter=None,
        device="cpu",
        fastsam_provider=_BombFastSAM() if fastsam_provider is None else fastsam_provider,
        sam3_client=sam3,
        f4_provider=_BombF4() if f4_provider is None else f4_provider,
    )


def _frame_inputs() -> dict[str, object]:
    y, x = np.indices((480, 640))
    depth = 1.0 + (x + y).astype(np.float32) * 0.001
    return {
        "scene_id": "scene0000_00",
        "frame_id": 0,
        "rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "depth_m": depth,
        "intrinsics": np.asarray(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        "camera_to_world": np.eye(4),
        "native_boxes_xyxy": np.empty((0, 4), dtype=np.float64),
    }


def test_disabled_builder_returns_none_without_constructing_models():
    route = build_stream3dv2_live_route(
        {"online_stream3dv2": {"enabled": False}},
        lifting_adapter=None,
        device="cuda:does-not-exist",
    )
    assert route is None


def test_tm_fpf_view_abstention_classification_is_exact_and_fail_closed():
    assert tm_fpf_c1_view_abstention_reason(
        TMFPFC1ContractError("target mask has too few pixels")
    ) == "too_few_mask_pixels"
    assert tm_fpf_c1_view_abstention_reason(
        TMFPFC1ContractError("target mask has too few valid depth pixels")
    ) == "too_few_valid_depth_pixels"
    assert tm_fpf_c1_view_abstention_reason(
        TMFPFC1ContractError("intrinsics focal lengths must be non-zero")
    ) is None
    assert tm_fpf_c1_view_abstention_reason(
        ValueError("target mask has too few pixels")
    ) is None


def test_non_upright_640_by_480_frame_performs_a_causal_empty_commit():
    route = _route()
    inputs = _frame_inputs()
    inputs["rgb"] = np.zeros((640, 480, 3), dtype=np.uint8)
    inputs["depth_m"] = np.ones((640, 480), dtype=np.float32)

    mask_frame = route.process_keyframe(**inputs)
    result = route.finalize(
        native_boxes_3d=np.stack((_box((10.0, 10.0, 10.0)),)),
        native_scores=np.asarray([0.8]),
        final_frame_id=0,
    )

    assert result.birth_count == 0
    assert result.overlay_count == 0
    assert result.diagnostics["counts"]["keyframes"] == 1
    assert result.diagnostics["counts"]["abstain_non_upright_producer_frame"] == 1
    assert result.diagnostics["state"]["committed_frame_count"] == 1
    assert result.diagnostics["state"]["committed_view_count"] == 0
    assert result.diagnostics["state"]["last_committed_frame_ordinal"] == 0
    assert result.diagnostics["future_access_count"] == 0
    assert mask_frame is None


class _OneMaskFastSAM:
    def __init__(self) -> None:
        self.calls = 0

    def infer_bgr(self, image: np.ndarray) -> object:
        self.calls += 1
        assert image.shape == (480, 640, 3)
        mask = np.zeros((1, 480, 640), dtype=np.bool_)
        mask[0, 100:150, 100:150] = True
        return SimpleNamespace(
            masks=mask,
            confidences=np.asarray([0.9], dtype=np.float32),
            boxes_xyxy=np.asarray([[100.0, 100.0, 149.0, 149.0]], dtype=np.float32),
            count=1,
        )


class _OneValidF4:
    def __init__(self) -> None:
        self.calls = 0

    def infer_batch(
        self,
        scene_id: str,
        frame_id: int,
        image: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        pose: np.ndarray,
        boxes: np.ndarray,
        source_ids: tuple[str, ...],
    ) -> object:
        self.calls += 1
        assert scene_id == "scene0000_00" and frame_id == 0
        assert boxes.shape == (1, 4) and len(source_ids) == 1
        corners = _box((-0.39, -0.23, 1.08), (0.16, 0.16, 0.16))
        row = SimpleNamespace(
            source_id=source_ids[0],
            valid=True,
            world_corners=corners,
            confidence=0.8,
        )
        return SimpleNamespace(
            rows=(row,),
            diagnostics=SimpleNamespace(valid_count=1, invalid_count=0),
        )


def test_terminal_native_order_and_scores_are_preserved_and_birth_is_below_0125():
    fastsam = _OneMaskFastSAM()
    f4 = _OneValidF4()
    route = _route(fastsam_provider=fastsam, f4_provider=f4)
    route.process_keyframe(**_frame_inputs())

    native_boxes = np.stack(
        (_box((10.0, 10.0, 10.0)), _box((20.0, 20.0, 20.0)))
    )
    native_scores = np.asarray([0.6, 0.9], dtype=np.float64)
    result = route.finalize(
        native_boxes_3d=native_boxes,
        native_scores=native_scores,
        final_frame_id=0,
    )

    assert fastsam.calls == 1 and f4.calls == 1
    assert result.birth_count == 1
    assert result.overlay_count == 0
    np.testing.assert_array_equal(result.boxes_3d[:2], native_boxes.astype(np.float32))
    np.testing.assert_array_equal(result.scores[:2], native_scores.astype(np.float32))
    assert result.scores.shape == (3,)
    assert 0.05 < float(result.scores[2]) < 0.125
    assert float(result.scores[2]) < float(native_scores.min())
    assert result.diagnostics["native_scores_preserved"] is True
    assert result.diagnostics["future_access_count"] == 0


def test_lightweight_route_keeps_single_view_v2_birth_without_v3_gate():
    fastsam = _OneMaskFastSAM()
    f4 = _OneValidF4()
    sam3 = LiveSAM3Client(LiveSAM3Config(enabled=False, device="cpu"))
    config = replace(
        _config(),
        lightweight_enabled=True,
        depth_trigger_enabled=False,
        fastsam_box_shortlist=1,
        fastsam_top_k=1,
        conditional_f2=True,
        f4_top_m_tracks=1,
    )
    route = Stream3Dv2LiveRoute(
        config,
        lifting_adapter=None,
        device="cpu",
        fastsam_provider=fastsam,
        sam3_client=sam3,
        f4_provider=f4,
    )
    mask_frame = route.process_keyframe(**_frame_inputs())
    result = route.finalize(
        native_boxes_3d=np.stack((_box((10.0, 10.0, 10.0)),)),
        native_scores=np.asarray([0.8]),
        final_frame_id=0,
    )

    assert result.birth_count == 1
    assert result.diagnostics["lightweight"]["enabled"] is True
    assert result.diagnostics["counts"]["f4_track_shortlist"] == 1
    assert result.diagnostics["counts"]["f2_candidates"] == 1
    assert isinstance(mask_frame, NativeTargetMaskFrame)
    assert mask_frame.frame_id == 0
    assert mask_frame.masks.shape == (1, 480, 640)
    np.testing.assert_array_equal(
        mask_frame.automatic_boxes_xyxy,
        np.asarray([[100.0, 100.0, 149.0, 149.0]], dtype=np.float32),
    )


def test_returned_mask_frame_reuses_the_same_fastsam_result_for_native_matching():
    fastsam = _OneMaskFastSAM()
    sam3 = LiveSAM3Client(LiveSAM3Config(enabled=False, device="cpu"))
    config = replace(
        _config(),
        lightweight_enabled=True,
        depth_trigger_enabled=False,
        fastsam_box_shortlist=1,
        fastsam_top_k=1,
        conditional_f2=True,
        f4_top_m_tracks=1,
    )
    route = Stream3Dv2LiveRoute(
        config,
        lifting_adapter=None,
        device="cpu",
        fastsam_provider=fastsam,
        sam3_client=sam3,
        f4_provider=_BombF4(),
    )
    inputs = _frame_inputs()
    inputs["native_boxes_xyxy"] = np.asarray(
        [[100.0, 100.0, 150.0, 150.0]], dtype=np.float32
    )
    mask_frame = route.process_keyframe(**inputs)

    assert isinstance(mask_frame, NativeTargetMaskFrame)
    matched = match_fastsam_target_masks(
        native_boxes_xyxy=mask_frame.native_boxes_xyxy,
        automatic_masks=mask_frame.masks,
        automatic_boxes_xyxy=mask_frame.automatic_boxes_xyxy,
        automatic_confidences=mask_frame.automatic_confidences,
        config=resolve_tm_fpf_c1_config({"tm_fpf_c1": {"enabled": True}}),
    )
    assert matched == (0,)
    assert fastsam.calls == 1
    route.close()


def test_lightweight_depth_trigger_skips_fastsam_during_cooldown():
    fastsam = _OneMaskFastSAM()
    f4 = _OneValidF4()
    sam3 = LiveSAM3Client(LiveSAM3Config(enabled=False, device="cpu"))
    config = replace(
        _config(),
        lightweight_enabled=True,
        depth_trigger_enabled=True,
        depth_trigger_config=replace(
            _config().depth_trigger_config,
            burst_keyframes=0,
            cooldown_keyframes=4,
        ),
        fastsam_box_shortlist=1,
        fastsam_top_k=1,
        conditional_f2=True,
        f4_top_m_tracks=1,
    )
    route = Stream3Dv2LiveRoute(
        config,
        lifting_adapter=None,
        device="cpu",
        fastsam_provider=fastsam,
        sam3_client=sam3,
        f4_provider=f4,
    )
    first_mask_frame = route.process_keyframe(**_frame_inputs())
    second = _frame_inputs()
    second["frame_id"] = 25
    second_mask_frame = route.process_keyframe(**second)
    result = route.finalize(
        native_boxes_3d=np.stack((_box((10.0, 10.0, 10.0)),)),
        native_scores=np.asarray([0.8]),
        final_frame_id=25,
    )

    assert fastsam.calls == 1
    assert f4.calls == 1
    assert result.diagnostics["counts"]["keyframes"] == 2
    assert result.diagnostics["counts"]["abstain_depth_trigger_cooldown"] == 1
    assert result.diagnostics["depth_trigger"]["runs"] == 1
    assert result.diagnostics["depth_trigger"]["skips"] == 1
    assert isinstance(first_mask_frame, NativeTargetMaskFrame)
    assert second_mask_frame is None


def _semantic_view(frame_id: int) -> SAM3MemoryTeacherView:
    height, width = 2, 4
    return SAM3MemoryTeacherView(
        frame_id=frame_id,
        intrinsics=np.eye(3),
        camera_to_world=np.eye(4),
        depth_m=np.ones((height, width), dtype=np.float32),
        masks_packbits=np.empty((0, (height * width + 7) // 8), dtype=np.uint8),
        scores=np.empty((0,), dtype=np.float64),
        labels=np.empty((0,), dtype=str),
        image_shape=(height, width),
    )


def test_semantic_view_must_be_ready_no_later_than_the_candidate_decision():
    route = _route()
    ready_late = _semantic_view(10)
    source_in_future = _semantic_view(31)
    route._semantic_views = [
        _SemanticAvailability(view=ready_late, ready_frame_id=30),
        _SemanticAvailability(view=source_in_future, ready_frame_id=20),
    ]

    assert route._eligible_semantic_views(29) == []
    assert route._eligible_semantic_views(30) == [ready_late]
    assert route._eligible_semantic_views(31) == [ready_late, source_in_future]
    route.close()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("rgb", np.zeros((479, 640, 3), dtype=np.uint8)),
        ("depth_m", np.ones((479, 640), dtype=np.float32)),
        ("intrinsics", np.eye(4)),
        ("camera_to_world", np.eye(3)),
        ("native_boxes_xyxy", np.zeros((1, 5))),
    ],
    ids=("rgb", "depth", "intrinsics", "pose", "native-boxes"),
)
def test_bad_live_input_shapes_fail_closed_before_model_inference(
    field: str, bad_value: np.ndarray
):
    route = _route()
    inputs = _frame_inputs()
    inputs[field] = bad_value
    try:
        with pytest.raises(Stream3Dv2LiveError):
            route.process_keyframe(**inputs)
        assert route._state.statistics.committed_frame_count == 0
        assert route._state.statistics.committed_view_count == 0
    finally:
        route.close()

import pickle

import numpy as np

from boxfusion.online_candidate_map import EvidenceFrame
from boxfusion.online_candidate_runtime import OnlineCandidateRuntime


SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=np.float64,
)


class FakeProvider:
    def __init__(self):
        self.calls = []

    def process(self, **kwargs):
        self.calls.append(kwargs["frame_id"])
        box = SIGNS * 0.5 + np.asarray([0.0, 0.0, 4.0])
        return EvidenceFrame(
            proposal_ids=np.asarray([kwargs["frame_id"]]),
            proposal_boxes_2d=np.asarray([[110.0, 110.0, 146.0, 146.0]]),
            proposal_corners=np.asarray([box]),
            proposal_scores=np.asarray([0.8]),
        )

    def diagnostics(self):
        return {"fake": True, "calls": len(self.calls)}


def test_runtime_writes_each_causal_prefix_and_close_does_no_inference(tmp_path):
    output = tmp_path / "predictions"
    diagnostics = tmp_path / "diagnostics"
    provider = FakeProvider()
    runtime = OnlineCandidateRuntime(
        provider=provider,
        output_root=str(output),
        diagnostics_root=str(diagnostics),
        write_every_keyframe=True,
    )
    pose = np.eye(4)
    K = np.asarray(
        [[100.0, 0.0, 128.0], [0.0, 100.0, 128.0], [0.0, 0.0, 1.0]]
    )
    empty_corners = np.empty((0, 8, 3))
    for frame in range(3):
        snapshot = runtime.process_keyframe(
            scene_id="scene-test",
            frame_id=frame,
            rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            depth_m=np.ones((256, 256), dtype=np.float32),
            image_intrinsic=K,
            depth_intrinsic=K,
            camera_to_world=pose,
            native_ids=np.empty(0, dtype=np.int64),
            native_corners=empty_corners,
            native_scores=np.empty(0),
        )
        with (output / "scene-test_boxes.pkl").open("rb") as handle:
            rows = pickle.load(handle)[0]
        assert len(rows) == len(snapshot.boxes)
    assert provider.calls == [0, 1, 2]
    assert len(snapshot.boxes) == 1
    before_close = len(provider.calls)
    report = runtime.close()
    assert len(provider.calls) == before_close
    assert report["scene_end_inference"] is False
    assert report["state"]["frames_seen"] == 3
    assert (diagnostics / "scene-test.json").is_file()

    terminal, report = runtime.materialize_terminal(
        frame_id=2,
        native_ids=np.asarray([9]),
        native_corners=np.asarray([SIGNS * 0.5 + [3.0, 0.0, 4.0]]),
        native_scores=np.asarray([0.2]),
    )
    assert terminal.sources.count("native") == 1
    assert report["scene_end_inference"] is False
    assert report["scene_end_assembly"] is True
    assert report["state"]["terminal_native_map_readout"] is True


def test_async_runtime_commits_in_order_and_close_only_drains_queued_frames(tmp_path):
    provider = FakeProvider()
    runtime = OnlineCandidateRuntime(
        provider=provider,
        output_root=str(tmp_path / "predictions"),
        asynchronous=True,
        queue_capacity=2,
    )
    pose = np.eye(4)
    K = np.asarray(
        [[100.0, 0.0, 128.0], [0.0, 100.0, 128.0], [0.0, 0.0, 1.0]]
    )
    for frame in range(4):
        runtime.process_keyframe(
            scene_id="scene-test",
            frame_id=frame,
            rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            depth_m=np.ones((256, 256), dtype=np.float32),
            image_intrinsic=K,
            depth_intrinsic=K,
            camera_to_world=pose,
            native_ids=np.empty(0, dtype=np.int64),
            native_corners=np.empty((0, 8, 3)),
            native_scores=np.empty(0),
        )
    report = runtime.close()
    assert provider.calls == [0, 1, 2, 3]
    assert report["asynchronous"] is True
    assert report["scene_end_inference"] is False
    assert report["state"]["frames_seen"] == 4


def test_runtime_forwards_current_native_nms_children_to_m1p(tmp_path):
    provider = FakeProvider()
    runtime = OnlineCandidateRuntime(provider=provider)
    pose = np.eye(4)
    K = np.asarray(
        [[100.0, 0.0, 128.0], [0.0, 100.0, 128.0], [0.0, 0.0, 1.0]]
    )
    child = SIGNS * 0.25 + np.asarray([5.0, 0.0, 4.0])
    for frame in range(2):
        runtime.begin_keyframe(
            scene_id="scene-test",
            frame_id=frame,
            rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            depth_m=np.ones((256, 256), dtype=np.float32),
            image_intrinsic=K,
            depth_intrinsic=K,
            camera_to_world=pose,
        )
        runtime.finish_keyframe(
            frame_id=frame,
            native_ids=np.empty(0, dtype=np.int64),
            native_corners=np.empty((0, 8, 3)),
            native_scores=np.empty(0),
            child_ids=np.asarray([100 + frame]),
            child_corners=np.asarray([child]),
            child_scores=np.asarray([0.4]),
        )
    assert runtime.last_snapshot.sources.count("m1p") == 1
    birth = next(iter(runtime.state.m1p.births.values()))
    assert birth.evidence_sources == ("child", "child")

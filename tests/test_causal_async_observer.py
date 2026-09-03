from __future__ import annotations

import numpy as np
import pytest

from boxfusion.causal_async_observer import (
    BoundedCausalAsyncObserver,
    CausalAsyncObserverConfig,
    CausalObserverTask,
)
from boxfusion.s3r_receipt_tracker import S3RObservation, S3RReceiptTracker


def _corners() -> np.ndarray:
    return np.asarray(
        [
            [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5], [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5], [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )


def _receipt():
    tracker = S3RReceiptTracker()
    for index, frame_id in enumerate((0, 25, 50)):
        observation = S3RObservation(
            frame_id=frame_id,
            source_row=index,
            sealed_npz_row=index,
            source_instance_id=index,
            score=0.8 - 0.1 * index,
            corners=_corners(),
        )
        query = tracker.query(frame_id, (observation,))
        commit = tracker.commit(query)
    assert len(commit.newly_frozen_receipts) == 1
    return commit.newly_frozen_receipts[0]


def test_async_observer_completes_immutable_causal_receipt() -> None:
    observer = BoundedCausalAsyncObserver(
        "scene0000_00",
        CausalAsyncObserverConfig(
            max_workers=1,
            max_pending_tasks=2,
            max_results=2,
            max_result_lag_keyframes=4,
        ),
    )
    receipt = _receipt()
    assert observer.submit(receipt, keyframe_step=2, memory_version=3)
    observer.close(2)
    assert observer.summary()["completed_results"] == 1
    assert observer.summary()["dropped_tasks"] == 0
    result = observer.result_rows()[0]
    assert result["candidate_id"] == receipt.track_id
    assert result["evidence_frame_ids"] == (0, 25, 50)
    assert result["enqueue_frame_id"] == 50


def test_async_task_rejects_future_evidence() -> None:
    with pytest.raises(ValueError, match="causal"):
        CausalObserverTask(
            serial=1,
            scene_id="scene0000_00",
            candidate_id=0,
            enqueue_frame_id=50,
            enqueue_keyframe_step=2,
            memory_version=3,
            evidence_frame_ids=(0, 25, 75),
            evidence_source_rows=(0, 1, 2),
            evidence_scores=(0.8, 0.7, 0.6),
            evidence_corners=np.stack((_corners(),) * 3),
        )

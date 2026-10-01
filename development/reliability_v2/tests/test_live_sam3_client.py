from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from boxfusion.live_sam3_client import LiveSAM3Client, LiveSAM3Config


WORKER = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "run_stream3dv2_live_sam3_worker.py"
)


def _fake_config(**updates):
    values = dict(
        enabled=True,
        python_executable=sys.executable,
        worker_path=str(WORKER),
        worker_backend="fake",
        startup_timeout_s=5.0,
        close_timeout_s=2.0,
        late_after_s=2.0,
    )
    values.update(updates)
    return LiveSAM3Config(**values)


def test_disabled_client_starts_no_process_and_records_drops():
    client = LiveSAM3Client(LiveSAM3Config(enabled=False))
    assert client.start() is False
    assert client.worker_pid is None
    assert client.submit(np.zeros((8, 8, 3), dtype=np.uint8)) is None
    stats = client.snapshot()
    assert stats.queue_depth == 0
    assert stats.submitted == 0
    assert stats.dropped_disabled == 1
    assert stats.drop_count == 1
    client.close()
    assert client.snapshot().closed is True


def test_fake_worker_roundtrip_is_bounded_and_packbits_are_exact():
    client = LiveSAM3Client(_fake_config())
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    request_id = client.submit(
        image,
        context={"scene_id": "scene0000_00", "frame_id": 125},
    )
    assert request_id == 1
    assert client.pending is True

    # The single-slot contract drops work instead of growing an async queue.
    assert client.submit(image, context={"frame_id": 126}) is None
    assert client.snapshot().queue_depth == 1
    assert client.snapshot().max_queue_depth == 1

    result = client.drain(timeout_s=5.0)
    assert result is not None
    assert result.request_id == request_id
    assert result.context == {"scene_id": "scene0000_00", "frame_id": 125}
    assert result.image_shape == (24, 32)
    assert result.count == 2
    assert result.labels == ("chair", "cabinet")
    np.testing.assert_allclose(result.scores, [0.91, 0.77], rtol=0.0, atol=1e-6)
    assert result.boxes.shape == (2, 4)
    assert result.masks_packbits.shape == (2, (24 * 32 + 7) // 8)
    assert not result.masks_packbits.flags.writeable
    assert not result.scores.flags.writeable
    assert not result.boxes.flags.writeable
    masks = result.unpack_masks()
    assert masks.shape == (2, 24, 32)
    assert masks.dtype == np.bool_
    assert np.all(masks[0, :, :16]) and not np.any(masks[0, :, 16:])
    assert not np.any(masks[1, :, :16]) and np.all(masks[1, :, 16:])
    assert result.gpu_runtime_ms == 0.0
    assert result.worker_runtime_ms >= 0.0
    assert result.end_to_end_runtime_ms >= result.worker_runtime_ms
    assert result.late is False

    stats = client.snapshot()
    assert stats.queue_depth == 0
    assert stats.submitted == 1
    assert stats.completed == 1
    assert stats.delivered == 1
    assert stats.dropped_pending == 1
    assert stats.drop_count == 1
    assert stats.worker_error_count == 0
    assert stats.last_gpu_runtime_ms == 0.0
    assert stats.last_worker_runtime_ms is not None
    assert client.diagnostics()["queue_depth"] == 0
    client.close()
    assert client.snapshot().closed is True


def test_late_result_is_measured_and_fail_closed_when_configured():
    client = LiveSAM3Client(
        _fake_config(
            fake_delay_ms=30.0,
            late_after_s=0.001,
            drop_late_results=True,
        )
    )
    assert client.submit(np.zeros((24, 32, 3), dtype=np.uint8)) == 1
    assert client.drain(timeout_s=5.0) is None
    stats = client.snapshot()
    assert stats.completed == 1
    assert stats.delivered == 0
    assert stats.late_count == 1
    assert stats.dropped_late == 1
    assert stats.drop_count == 1
    assert stats.last_end_to_end_runtime_ms is not None
    assert stats.last_end_to_end_runtime_ms >= 20.0
    client.close()


def test_late_result_can_be_delivered_with_explicit_late_bit():
    client = LiveSAM3Client(
        _fake_config(
            fake_delay_ms=20.0,
            late_after_s=0.001,
            drop_late_results=False,
        )
    )
    assert client.submit(np.zeros((24, 32, 3), dtype=np.uint8)) == 1
    result = client.drain(timeout_s=5.0)
    assert result is not None and result.late is True
    stats = client.snapshot()
    assert stats.late_count == 1
    assert stats.dropped_late == 0
    assert stats.delivered == 1
    client.close()


def test_empty_worker_result_preserves_strict_array_shapes():
    client = LiveSAM3Client(_fake_config())
    assert client.submit(np.full((24, 32, 3), 255, dtype=np.uint8)) == 1
    result = client.drain(timeout_s=5.0)
    assert result is not None and result.count == 0
    assert result.masks_packbits.shape == (0, (24 * 32 + 7) // 8)
    assert result.scores.shape == (0,)
    assert result.boxes.shape == (0, 4)
    assert result.labels == ()
    assert result.unpack_masks().shape == (0, 24, 32)
    client.close()


def test_invalid_input_and_config_fail_before_crossing_process_boundary():
    with pytest.raises(ValueError, match="max_proposals"):
        LiveSAM3Client(_fake_config(max_proposals=65))
    client = LiveSAM3Client(_fake_config())
    with pytest.raises(ValueError, match="dtype uint8"):
        client.submit(np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        client.submit(np.zeros((8, 8), dtype=np.uint8))
    assert client.worker_pid is None
    client.close()


def test_close_without_drain_terminates_pending_worker():
    client = LiveSAM3Client(_fake_config(fake_delay_ms=500.0))
    client.submit(np.zeros((24, 32, 3), dtype=np.uint8))
    pid = client.worker_pid
    assert pid is not None
    client.close(drain=False)
    assert client.pending is False
    assert client.snapshot().closed is True

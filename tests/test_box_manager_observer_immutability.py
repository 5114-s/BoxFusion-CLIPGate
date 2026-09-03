import numpy as np
import pytest
import torch

from boxfusion.box_manager import BoxManager, _copy_observer_index
from boxfusion.observer_track_registry import ObserverTrackRegistry


def _config():
    return {
        "association": {"rotation_gap": 30, "translation_gap": 0.8},
        "box_fusion": {"small_size": 0.35},
    }


class _CountingIterator:
    def __init__(self, values):
        self._values = iter(values)
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return self

    def __next__(self):
        return next(self._values)


class _InspectingRegistry:
    def __init__(self, expected_rows):
        self.expected_rows = expected_rows
        self.received = None

    def assert_native_row_count(self, token, count):
        assert count == self.expected_rows

    def record_association(self, token, winner, losers, *, stage):
        self.received = losers
        try:
            losers[0] = 99
        except TypeError:
            pass

    def abort_keyframe(self, token):
        raise AssertionError("observer should not abort")


class _TensorMutatingRegistry(_InspectingRegistry):
    def record_association(self, token, winner, losers, *, stage):
        self.received = losers
        for index in losers:
            if hasattr(index, "fill_"):
                index.fill_(99)
        assert all(type(index) is int for index in losers)


def test_enabled_observer_receives_immutable_association_indices():
    manager = BoxManager(_config())
    manager.init_new_predictions(2, 0)
    registry = _InspectingRegistry(expected_rows=2)
    token = object()
    assert manager.attach_observer_track_registry(registry, token)

    losers = [0]
    keep = manager.record(
        1,
        losers,
        [0, 1],
        torch.eye(4).repeat(2, 1, 1),
        np.ones((2, 3), dtype=np.float32),
        [1],
        np.zeros((2, 3), dtype=np.float32),
    )

    assert registry.received == (0,)
    assert isinstance(registry.received, tuple)
    assert losers == [0]
    assert keep == [1]


def test_enabled_observer_rejects_iterator_without_preconsuming_native_input():
    manager = BoxManager(_config())
    manager.init_new_predictions(2, 0)
    registry = ObserverTrackRegistry()
    token = registry.begin_keyframe(0, (0, 1))
    assert token is not None
    assert manager.attach_observer_track_registry(registry, token)
    keep = _CountingIterator((1,))

    manager.update(keep)

    assert keep.iter_calls == 1
    assert manager.fusion_list == [[1]]
    assert manager.observer_track_registry is None
    assert "bounded indexable" in manager.observer_track_error


def test_tensor_indices_are_detached_as_python_ints():
    manager = BoxManager(_config())
    manager.init_new_predictions(2, 0)
    registry = _TensorMutatingRegistry(expected_rows=2)
    token = object()
    assert manager.attach_observer_track_registry(registry, token)

    losers = torch.tensor([0], dtype=torch.int64)
    keep = manager.record(
        1,
        losers,
        [0, 1],
        torch.eye(4).repeat(2, 1, 1),
        np.ones((2, 3), dtype=np.float32),
        [1],
        np.zeros((2, 3), dtype=np.float32),
    )

    assert registry.received == (0,)
    assert losers.tolist() == [0]
    assert keep == [1]


def test_boolean_indices_fail_open_before_reaching_observer():
    manager = BoxManager(_config())
    manager.init_new_predictions(2, 0)
    registry = ObserverTrackRegistry()
    token = registry.begin_keyframe(0, (0, 1))
    assert token is not None
    assert manager.attach_observer_track_registry(registry, token)

    manager.update(np.asarray([True], dtype=np.bool_))

    assert manager.observer_track_registry is None
    assert "not booleans" in manager.observer_track_error


@pytest.mark.parametrize(
    "value",
    [1, np.int64(1), np.uint32(1), torch.tensor(1, dtype=torch.int64)],
)
def test_observer_index_copy_returns_strict_python_int(value):
    copied = _copy_observer_index(value)
    assert type(copied) is int
    assert copied == 1


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        torch.tensor(True),
        1.0,
        np.float32(1.0),
        torch.tensor(1.0),
        -1,
        np.asarray([1], dtype=np.int64),
        torch.tensor([1], dtype=torch.int64),
    ],
)
def test_observer_index_copy_rejects_ambiguous_or_unsafe_values(value):
    with pytest.raises(ValueError):
        _copy_observer_index(value)

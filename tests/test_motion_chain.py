"""Unit tests for the causal motion-chain module (synthetic walker)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boxfusion.motion_chain import MotionChain


class FakeBoxes:
    def __init__(self, xyzlwh, R):
        xyzlwh = np.asarray(xyzlwh, dtype=float)
        self.tensor = torch.tensor(xyzlwh, dtype=torch.float64)
        self.R = torch.tensor(R, dtype=torch.float64)
        n = len(xyzlwh)
        self.corners = torch.zeros(n, 8, 3, dtype=torch.float64)
        half = xyzlwh[:, 3:] / 2
        for i in range(n):
            local = np.array([[sx * half[i, 0], sy * half[i, 1], sz * half[i, 2]]
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            self.corners[i] = torch.tensor(
                np.asarray(xyzlwh[i, :3])[:, None] + R[i] @ local.T).T

    def __len__(self):
        return len(self.tensor)


class FakeIns:
    def __init__(self, xyzlwh, frames, init_ids):
        n = len(xyzlwh)
        self.pred_boxes_3d = FakeBoxes(xyzlwh, np.tile(np.eye(3), (n, 1, 1)))
        self.scores = torch.ones(n)
        self.frame_id = torch.tensor(frames)
        self.init_id = torch.tensor(init_ids)

    def __len__(self):
        return len(self.pred_boxes_3d)


def make_cfg():
    return {'motion_chain': {'enabled': True}}


def test_trail_chain_drops_superseded_rows():
    mc = MotionChain(make_cfg())
    size = [0.6, 0.7, 0.7]
    # kf0: row A at x=0; kf1: new row B at x=0.5 (A dead); kf2: new row C at
    # x=1.0; kf3: C re-observed (confirmed tip, n_obs=2)
    ins = FakeIns(
        [[0.0, 0.5, 0.5, *size], [0.5, 0.5, 0.5, *size],
         [1.0, 0.5, 0.5, *size], [1.0, 0.5, 0.5, *size]],
        frames=[0, 1, 2, 3], init_ids=[0, 1, 2, 3])
    mc.process_keyframe(ins, [[0]], kf=0)
    mc.process_keyframe(ins, [[0], [1]], kf=1)
    mc.process_keyframe(ins, [[0], [1], [2]], kf=2)
    assert mc.superseded_by.get(1) == 2
    mc.process_keyframe(ins, [[0], [1], [2, 3]], kf=3)
    # full chain 0->1->2 is direction-consistent (2 links): the walker's
    # start and middle positions are both stale trail members
    drop = mc._chain_drop_set()
    assert 0 in drop and 1 in drop
    all_pred = FakeIns([[0.0, 0.5, 0.5, *size], [0.5, 0.5, 0.5, *size],
                        [1.0, 0.5, 0.5, *size]], frames=[3, 3, 3], init_ids=[0, 1, 2])
    out = mc.materialize(all_pred, [[0], [1], [2, 3]])
    assert out['mask'].tolist() == [False, False, True]


def test_velocity_gate_rejects_inconsistent_link():
    mc = MotionChain(make_cfg())
    size = [0.6, 0.7, 0.7]
    # A(0) -> B(0.5) established; C born at x=2.2 violates velocity gate
    ins = FakeIns(
        [[0.0, 0.5, 0.5, *size], [0.5, 0.5, 0.5, *size], [2.2, 0.5, 0.5, *size]],
        frames=[0, 1, 2], init_ids=[0, 1, 2])
    mc.process_keyframe(ins, [[0]], kf=0)
    mc.process_keyframe(ins, [[0], [1]], kf=1)
    mc.process_keyframe(ins, [[0], [1], [2]], kf=2)
    assert mc.superseded_by.get(1) is None


def test_size_gate_blocks_different_objects():
    mc = MotionChain(make_cfg())
    # A: a big box; B: tiny box 0.4 m away
    ins = FakeIns(
        [[0.0, 0.5, 0.5, 0.9, 0.9, 0.9], [0.4, 0.5, 0.5, 0.2, 0.2, 0.2]],
        frames=[0, 1], init_ids=[0, 1])
    mc.process_keyframe(ins, [[0]], kf=0)
    mc.process_keyframe(ins, [[0], [1]], kf=1)
    assert mc.superseded_by == {}


def test_reactivation_cancels_edge():
    mc = MotionChain(make_cfg())
    size = [0.6, 0.7, 0.7]
    ins = FakeIns(
        [[0.0, 0.5, 0.5, *size], [0.5, 0.5, 0.5, *size], [0.02, 0.5, 0.5, *size]],
        frames=[0, 1, 2], init_ids=[0, 1, 2])
    mc.process_keyframe(ins, [[0]], kf=0)
    mc.process_keyframe(ins, [[0], [1]], kf=1)
    assert mc.superseded_by.get(0) == 1
    # A observed again at kf2 cancels the edge
    mc.process_keyframe(ins, [[0, 2], [1]], kf=2)
    assert 0 not in mc.superseded_by


def test_drift_row_gets_latest_observation_geometry():
    mc = MotionChain(make_cfg())
    size = [0.6, 0.7, 0.7]
    # single row whose observations walk from x=0 to x=0.5 (drift 0.5 > 0.35)
    ins = FakeIns(
        [[0.0, 0.5, 0.5, *size], [0.25, 0.5, 0.5, *size], [0.5, 0.5, 0.5, *size]],
        frames=[0, 1, 2], init_ids=[0, 1, 2])
    mc.process_keyframe(ins, [[0]], kf=0)
    mc.process_keyframe(ins, [[0, 1]], kf=1)
    mc.process_keyframe(ins, [[0, 1, 2]], kf=2)
    # all_pred_box is row-level: native fused geometry sits at the mean x=0.25
    all_pred = FakeIns([[0.25, 0.5, 0.5, *size]], frames=[2], init_ids=[0])
    out = mc.materialize(all_pred, [[0, 1, 2]])
    assert out is not None
    assert out['mask'].tolist() == [True]
    center = out['corners'][0].mean(0)
    assert abs(center[0] - 0.5) < 1e-6


def test_static_rows_untouched():
    mc = MotionChain(make_cfg())
    size = [0.6, 0.7, 0.7]
    # two static rows, no births after kf1
    ins = FakeIns(
        [[0.0, 0.5, 0.5, *size], [2.5, 0.5, 0.5, *size], [0.05, 0.5, 0.5, *size],
         [2.45, 0.5, 0.5, *size]],
        frames=[0, 0, 1, 1], init_ids=[0, 1, 2, 3])
    mc.process_keyframe(ins, [[0], [1]], kf=0)
    mc.process_keyframe(ins, [[0, 2], [1, 3]], kf=1)
    assert mc.superseded_by == {}
    all_pred = FakeIns([[0.02, 0.5, 0.5, *size], [2.47, 0.5, 0.5, *size]],
                       frames=[1, 1], init_ids=[0, 1])
    out = mc.materialize(all_pred, [[0, 2], [1, 3]])
    assert out['mask'].all()

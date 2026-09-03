from copy import deepcopy

import numpy as np

from tools.audit_b6_selective_boxer import EXPECTED_GATE, _validate_row


def _row(role):
    identity = np.eye(3, dtype=np.float32).tolist()
    cutr_xyz = [
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 2.0, 1.0, 1.0, 1.0],
    ]
    boxer_xyz = [
        [0.05, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.20, 0.0, 2.0, 1.0, 1.0, 1.0],
    ]
    selective_xyz = [boxer_xyz[0], cutr_xyz[1]]
    observer = role == "observer"
    return {
        "mode": role,
        "apply_stage": "post_filter",
        "selective_gate_enabled": True,
        "mutation_enabled": not observer,
        "projected_center_replaced": False,
        "count": 2,
        "eligible_count": 1,
        "applied_count": 0 if observer else 1,
        "fallback_count": 1,
        "selective_gate": deepcopy(EXPECTED_GATE),
        "gate_accepted": [True, False],
        "gate_reasons": [[], ["center_shift"]],
        "gate_rejection_counts": {
            "nonfinite": 0,
            "cutr_invalid": 0,
            "boxer_invalid": 0,
            "center_shift": 1,
            "volume_low": 0,
            "volume_high": 0,
        },
        "cutr_xyz_dims_camera": cutr_xyz,
        "cutr_rotation_camera_object": [identity, identity],
        "boxer_xyz_dims_camera": boxer_xyz,
        "boxer_rotation_camera_object": [identity, identity],
        "selective_xyz_dims_camera": selective_xyz,
        "selective_rotation_camera_object": [identity, identity],
        "actual_xyz_dims_camera": cutr_xyz if observer else selective_xyz,
        "actual_rotation_camera_object": [identity, identity],
        "cutr_geometry_sha256": "cutr",
        "actual_geometry_sha256": "cutr" if observer else "selective",
    }


def test_valid_observer_and_active_rows_follow_selective_contract():
    for role in ("observer", "active"):
        issues = []
        _validate_row(
            _row(role),
            role=role,
            context={"scene": "scene0000_00", "frame_id": 0},
            issues=issues,
        )
        assert issues == []


def test_audit_rejects_active_row_that_applies_rejected_boxer_geometry():
    row = _row("active")
    row["actual_xyz_dims_camera"] = row["boxer_xyz_dims_camera"]
    issues = []
    _validate_row(
        row,
        role="active",
        context={"scene": "scene0000_00", "frame_id": 0},
        issues=issues,
    )
    assert any(issue["kind"] == "actual_xyz_contract_broken" for issue in issues)

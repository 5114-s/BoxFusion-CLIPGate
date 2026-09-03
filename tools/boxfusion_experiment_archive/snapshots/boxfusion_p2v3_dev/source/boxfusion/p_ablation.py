"""Strict P0/P1/P2/P2V2/P2V3 residual-proposal research profiles.

All stages share the exact frozen B6 output path.  P1 adds a read-only
residual RGB-D proposal stream; P2 adds only foreground-occupancy Top-K
selection on the frozen P1 stream; P2V2 adds detached local connected
components fitted from already-lifted unmatched Mask-RGBD proposals; P2V3
adds only read-only reliability-weighted parent/component fusion.  This helper
aggressively disables every older experimental mutation so stale YAML values
cannot turn a P observer run into an undocumented module combination.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping


P_STAGE_TO_PROFILE = {
    "P0": "p0_frozen_b6",
    "P1": "p1_residual_proposal_observer",
    "P2": "p2_occupancy_topk_observer",
    "P2V2": "p2v2_local_component_mask_rgbd_observer",
    "P2V3": "p2v3_reliability_geometry_fusion_observer",
}
P_PROFILE_TO_STAGE = {
    profile: stage for stage, profile in P_STAGE_TO_PROFILE.items()
}
P_STAGE_ADDED_MODULE = {
    "P0": None,
    "P1": "residual_proposal",
    "P2": "occupancy_topk",
    "P2V2": "p2_local_mask_geometry",
    "P2V3": "p2_reliability_fusion",
}
P_STAGE_MODULE_MATRIX = {
    "P0": {
        "residual_proposal": False,
        "occupancy_topk": False,
        "p2_local_mask_geometry": False,
        "p2_reliability_fusion": False,
    },
    "P1": {
        "residual_proposal": True,
        "occupancy_topk": False,
        "p2_local_mask_geometry": False,
        "p2_reliability_fusion": False,
    },
    "P2": {
        "residual_proposal": True,
        "occupancy_topk": True,
        "p2_local_mask_geometry": False,
        "p2_reliability_fusion": False,
    },
    "P2V2": {
        "residual_proposal": True,
        "occupancy_topk": True,
        "p2_local_mask_geometry": True,
        "p2_reliability_fusion": False,
    },
    "P2V3": {
        "residual_proposal": True,
        "occupancy_topk": True,
        "p2_local_mask_geometry": True,
        "p2_reliability_fusion": True,
    },
}


def _mapping(parent: MutableMapping[str, Any], key: str) -> MutableMapping[str, Any]:
    value = parent.get(key)
    if value is None:
        value = {}
        parent[key] = value
    elif not isinstance(value, MutableMapping):
        if not isinstance(value, Mapping):
            raise TypeError(f"online_refinement.{key} must be a mapping")
        value = dict(value)
        parent[key] = value
    return value


def apply_p_ablation(
    config: Mapping[str, Any],
    stage: str,
) -> Dict[str, Any]:
    """Return a deep-copied, output-safe P-stage configuration."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(stage, str):
        raise TypeError("stage must be a string")
    canonical = stage.strip().upper()
    if canonical not in P_STAGE_TO_PROFILE:
        raise ValueError("P stage must be P0, P1, P2, P2V2, or P2V3")
    if "online_refinement" not in config:
        raise ValueError("config is missing online_refinement")
    if not isinstance(config["online_refinement"], Mapping):
        raise TypeError("online_refinement must be a mapping")

    result: Dict[str, Any] = deepcopy(dict(config))
    online = result["online_refinement"]
    if not isinstance(online, MutableMapping):
        online = dict(online)
        result["online_refinement"] = online
    online["enabled"] = True
    # B6's established resolver/profile identity remains ``quality_only``.
    # P-stage provenance is recorded separately below.
    online["ablation_profile"] = "quality_only"
    online["p_ablation_stage"] = canonical
    online["p_ablation_profile"] = P_STAGE_TO_PROFILE[canonical]
    online["p_added_module"] = P_STAGE_ADDED_MODULE[canonical]

    quality = _mapping(online, "quality")
    quality["enabled"] = True
    soft_nms = _mapping(quality, "soft_nms")
    soft_nms["enabled"] = False
    quality["apply_to_unobserved"] = False

    # Only learned B6 score calibration may change formal output rows.
    for name in (
        "refit",
        "box_refiner",
        "supplemental_output",
        "missing_track_identity",
        "mask_graph",
        "fragment_stitch",
    ):
        _mapping(online, name)["enabled"] = False
    for name in (
        "generic_local_geometry_refiner",
        "supplemental_geometry_refiner",
        "trifusion_observer",
        "yidu_ablation",
    ):
        section = _mapping(online, name)
        section["enabled"] = False
        section["collect_diagnostics"] = False
        section["mutate"] = False
    joint = _mapping(online, "joint_local_head")
    joint["enabled"] = False
    joint["mutate_geometry"] = False
    joint["mutate_scores"] = False
    joint["collect_diagnostics"] = False

    residual = _mapping(online, "residual_proposal")
    p1_stages = {"P1", "P2", "P2V2", "P2V3"}
    p2_stages = {"P2", "P2V2", "P2V3"}
    p2v2_stages = {"P2V2", "P2V3"}
    residual["enabled"] = canonical in p1_stages
    residual["observer_only"] = True
    residual["mutate"] = False
    residual["collect_diagnostics"] = canonical in p1_stages
    # Keep observer MLP allocation and execution off the BoxFusion GPU.
    # This reduces, but cannot eliminate, the upstream CUDA fusion
    # nondeterminism measured by the P0-repeat control.
    residual["device"] = "cpu"
    if canonical in p2_stages:
        residual["mode"] = "infer"
    elif canonical == "P1":
        residual["mode"] = str(residual.get("mode", "infer")).strip().lower()
    occupancy = _mapping(online, "occupancy_topk")
    occupancy["enabled"] = canonical in p2_stages
    occupancy["observer_only"] = True
    occupancy["mutate"] = False
    occupancy["collect_diagnostics"] = canonical in p2_stages
    occupancy["device"] = "cpu"
    local_geometry = _mapping(online, "p2_local_mask_geometry")
    local_geometry["enabled"] = canonical in p2v2_stages
    local_geometry["observer_only"] = True
    local_geometry["mutate"] = False
    local_geometry["collect_diagnostics"] = canonical in p2v2_stages
    # P2 selected coordinates are expressed in the frozen P1 voxel grid.
    # Bind the anchor size explicitly so custom P1 configs cannot silently
    # desynchronise the P2V2 component-to-anchor test.
    local_geometry["occupancy_voxel_size"] = residual.get(
        "voxel_size", 0.08
    )
    reliability_fusion = _mapping(online, "p2_reliability_fusion")
    reliability_fusion["enabled"] = canonical == "P2V3"
    reliability_fusion["observer_only"] = True
    reliability_fusion["mutate"] = False
    reliability_fusion["collect_diagnostics"] = canonical == "P2V3"
    return result


__all__ = [
    "P_PROFILE_TO_STAGE",
    "P_STAGE_ADDED_MODULE",
    "P_STAGE_MODULE_MATRIX",
    "P_STAGE_TO_PROFILE",
    "apply_p_ablation",
]

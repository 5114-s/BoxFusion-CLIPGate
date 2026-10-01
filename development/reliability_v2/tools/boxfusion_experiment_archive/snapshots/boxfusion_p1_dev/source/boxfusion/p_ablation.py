"""Strict P0/P1 profiles for the residual-proposal research route.

P0 and P1 share the exact frozen B6 output path.  Their only difference is a
read-only residual RGB-D proposal stream in P1.  This helper aggressively
disables every older experimental mutation so stale YAML values cannot turn a
P1 observer run into an undocumented module combination.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping


P_STAGE_TO_PROFILE = {
    "P0": "p0_frozen_b6",
    "P1": "p1_residual_proposal_observer",
}
P_PROFILE_TO_STAGE = {
    profile: stage for stage, profile in P_STAGE_TO_PROFILE.items()
}
P_STAGE_ADDED_MODULE = {
    "P0": None,
    "P1": "residual_proposal",
}
P_STAGE_MODULE_MATRIX = {
    "P0": {"residual_proposal": False},
    "P1": {"residual_proposal": True},
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
    """Return a deep-copied, output-safe P0 or P1 configuration."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(stage, str):
        raise TypeError("stage must be a string")
    canonical = stage.strip().upper()
    if canonical not in P_STAGE_TO_PROFILE:
        raise ValueError("P stage must be P0 or P1")
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
    residual["enabled"] = canonical == "P1"
    residual["observer_only"] = True
    residual["mutate"] = False
    residual["collect_diagnostics"] = canonical == "P1"
    if canonical == "P1":
        residual["mode"] = str(residual.get("mode", "infer")).strip().lower()
    return result


__all__ = [
    "P_PROFILE_TO_STAGE",
    "P_STAGE_ADDED_MODULE",
    "P_STAGE_MODULE_MATRIX",
    "P_STAGE_TO_PROFILE",
    "apply_p_ablation",
]

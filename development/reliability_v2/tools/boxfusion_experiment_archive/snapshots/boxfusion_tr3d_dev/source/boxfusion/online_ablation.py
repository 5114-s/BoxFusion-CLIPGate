"""Pure configuration profiles for online-refinement ablations.

The profiles isolate final-output mutations while leaving the configured
supplemental proposal provider, RGB-D memory parameters, matching parameters,
and diagnostics untouched.  This module deliberately has no model or runtime
imports, so profiles can be prepared before any optional dependency is
loaded.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping


ONLINE_ABLATION_PROFILES = (
    "observer",
    "quality_observer",
    "refit_only",
    "supplemental_only",
    "supplemental_conservative",
    "quality_only",
    "full",
)

_REQUIRED_SECTIONS = (
    "appearance_memory",
    "supplemental_proposals",
    "object_memory",
    "refit",
    "box_refiner",
    "quality",
    "supplemental_output",
    "output_filter",
)


def _require_mapping(
    parent: Mapping[str, Any],
    key: str,
    *,
    qualified_parent: str,
) -> Mapping[str, Any]:
    if key not in parent:
        raise ValueError(f"Missing {qualified_parent}.{key} section")
    value = parent[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{qualified_parent}.{key} must be a mapping")
    return value


def _validate_enabled(section: Mapping[str, Any], qualified_name: str) -> None:
    if "enabled" not in section:
        raise ValueError(f"Missing {qualified_name}.enabled")
    if not isinstance(section["enabled"], bool):
        raise TypeError(f"{qualified_name}.enabled must be Boolean")


def _validate_config(config: Mapping[str, Any]) -> None:
    if "online_refinement" not in config:
        raise ValueError("Missing online_refinement section")
    online = config["online_refinement"]
    if not isinstance(online, Mapping):
        raise TypeError("online_refinement must be a mapping")
    if "enabled" not in online:
        raise ValueError("Missing online_refinement.enabled")
    if not isinstance(online["enabled"], bool):
        raise TypeError("online_refinement.enabled must be Boolean")

    sections = {
        name: _require_mapping(
            online, name, qualified_parent="online_refinement"
        )
        for name in _REQUIRED_SECTIONS
    }
    for name in (
        "appearance_memory",
        "supplemental_proposals",
        "object_memory",
        "refit",
        "box_refiner",
        "quality",
        "supplemental_output",
    ):
        _validate_enabled(
            sections[name], f"online_refinement.{name}"
        )
    soft_nms = _require_mapping(
        sections["quality"],
        "soft_nms",
        qualified_parent="online_refinement.quality",
    )
    _validate_enabled(soft_nms, "online_refinement.quality.soft_nms")
    minimum_extent = sections["output_filter"].get("minimum_extent")
    if minimum_extent is None:
        raise ValueError(
            "Missing online_refinement.output_filter.minimum_extent"
        )
    if isinstance(minimum_extent, bool) or not isinstance(
        minimum_extent, (int, float)
    ):
        raise TypeError(
            "online_refinement.output_filter.minimum_extent "
            "must be numeric"
        )


def _mutable_section(
    online: MutableMapping[str, Any], name: str
) -> MutableMapping[str, Any]:
    section = online[name]
    if not isinstance(section, MutableMapping):
        # ``_validate_config`` accepts general mappings as input.  Convert a
        # copied read-only/custom mapping before applying profile overrides.
        section = dict(section)
        online[name] = section
    return section


def apply_online_ablation_profile(
    config: Mapping[str, Any],
    profile: str,
) -> Dict[str, Any]:
    """Return an independent configuration with one ablation profile applied.

    ``full`` is a deep-copy-only profile: no value is changed.  The other
    profiles keep proposal and object-memory configuration unchanged:

    ``observer``
        Collect proposal/memory evidence and diagnostics without changing
        exported geometry, scores, or detection count.
    ``quality_observer``
        Same no-op output contract as ``observer``, but retain masked CLIP
        appearance extraction so B6 training diagnostics match inference.
    ``refit_only``
        Enable only the gated robust geometry refit.
    ``supplemental_only``
        Enable only confirmed supplemental-track output.
    ``supplemental_conservative``
        Enable confirmed supplemental-track output with the fixed B1 quality
        gates used by the ScanNet experiment: score/projection/global-IoU
        thresholds of 0.25/0.30/0.30 and a 0.30-m output extent filter.
    ``quality_only``
        Change only observed detections' scores with the configured learned
        quality model.  Geometry, detection count, and Soft-NMS stay disabled
        so this is a pure B6 ablation rather than a B6+B7 mixture.

    Args:
        config: Full BoxFusion configuration containing an
            ``online_refinement`` mapping.
        profile: One of :data:`ONLINE_ABLATION_PROFILES`.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(profile, str):
        raise TypeError("profile must be a string")
    if profile not in ONLINE_ABLATION_PROFILES:
        choices = ", ".join(ONLINE_ABLATION_PROFILES)
        raise ValueError(
            f"Unknown online ablation profile {profile!r}; choose {choices}"
        )
    _validate_config(config)

    result: Dict[str, Any] = deepcopy(dict(config))
    if profile == "full":
        return result

    online = result["online_refinement"]
    if not isinstance(online, MutableMapping):
        online = dict(online)
        result["online_refinement"] = online
    online["enabled"] = True

    appearance = _mutable_section(online, "appearance_memory")
    refit = _mutable_section(online, "refit")
    box_refiner = _mutable_section(online, "box_refiner")
    quality = _mutable_section(online, "quality")
    supplemental = _mutable_section(online, "supplemental_output")
    output_filter = _mutable_section(online, "output_filter")
    soft_nms = quality["soft_nms"]
    if not isinstance(soft_nms, MutableMapping):
        soft_nms = dict(soft_nms)
        quality["soft_nms"] = soft_nms

    appearance["enabled"] = profile in {
        "quality_observer",
        "quality_only",
    }
    box_refiner["enabled"] = False
    quality["enabled"] = profile == "quality_only"
    soft_nms["enabled"] = False
    if profile != "supplemental_conservative":
        output_filter["minimum_extent"] = 0.0

    refit["enabled"] = profile == "refit_only"
    supplemental["enabled"] = profile in {
        "supplemental_only",
        "supplemental_conservative",
    }
    if profile == "quality_only":
        quality["apply_to_unobserved"] = False
    if profile == "supplemental_conservative":
        supplemental["min_score"] = 0.25
        supplemental["min_projection_iou"] = 0.30
        supplemental["drop_if_global_iou"] = 0.30
        output_filter["minimum_extent"] = 0.30
    return result


__all__ = [
    "ONLINE_ABLATION_PROFILES",
    "apply_online_ablation_profile",
]

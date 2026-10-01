"""Strict observer-only ablations distilled from the YiDu paper survey.

The stages in this module are deliberately cumulative.  ``B0`` is the frozen
B6 output, and every following stage adds exactly one diagnostics-only module:

``B0 -> A1 -> A2 -> A3 -> A4 -> A5 -> A6``.

This module prepares configuration only.  In particular, it exposes no output
mutation switch and forces both the stage and every module to
``observer_only=True`` and ``mutate=False``.  An active implementation must use
a separately named, separately reviewed profile after held-out gates have been
frozen.
"""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple


YIDU_SCHEMA = "boxfusion.yidu.incremental_observer.v1"

YIDU_STAGES: Tuple[str, ...] = (
    "B0",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
)

YIDU_MODULES: Tuple[str, ...] = (
    "adaptive_erosion",
    "dfu_filter",
    "voxel_components",
    "occupancy_msr",
    "raw_fused_query",
    "quality_gate",
)

YIDU_STAGE_ADDED_MODULE = MappingProxyType(
    {
        "B0": None,
        "A1": "adaptive_erosion",
        "A2": "dfu_filter",
        "A3": "voxel_components",
        "A4": "occupancy_msr",
        "A5": "raw_fused_query",
        "A6": "quality_gate",
    }
)

YIDU_STAGE_TO_PROFILE = MappingProxyType(
    {
        "B0": "yidu_b0_frozen_b6",
        "A1": "yidu_a1_adaptive_erosion_observer",
        "A2": "yidu_a2_dfu_filter_observer",
        "A3": "yidu_a3_voxel_components_observer",
        "A4": "yidu_a4_occupancy_msr_observer",
        "A5": "yidu_a5_raw_fused_query_observer",
        "A6": "yidu_a6_quality_gate_observer",
    }
)
YIDU_PROFILE_TO_STAGE = MappingProxyType(
    {profile: stage for stage, profile in YIDU_STAGE_TO_PROFILE.items()}
)


def _build_module_matrix() -> Dict[str, Mapping[str, bool]]:
    active = set()
    matrix: Dict[str, Mapping[str, bool]] = {}
    for stage in YIDU_STAGES:
        added = YIDU_STAGE_ADDED_MODULE[stage]
        if added is not None:
            active.add(added)
        matrix[stage] = MappingProxyType(
            {module: module in active for module in YIDU_MODULES}
        )
    return matrix


YIDU_STAGE_MODULE_MATRIX = MappingProxyType(_build_module_matrix())

# Short aliases kept intentionally explicit for scripts which describe the
# objects as a profile map or a module matrix.
YIDU_STAGE_PROFILE_MAP = YIDU_STAGE_TO_PROFILE
YIDU_PROFILE_STAGE_MAP = YIDU_PROFILE_TO_STAGE
YIDU_MODULE_MATRIX = YIDU_STAGE_MODULE_MATRIX


def resolve_yidu_stage(stage_or_profile: str) -> str:
    """Return the canonical stage name for a stage or profile string."""

    if not isinstance(stage_or_profile, str):
        raise TypeError("YiDu stage/profile must be a string")
    value = stage_or_profile.strip()
    if not value:
        raise ValueError("YiDu stage/profile cannot be empty")
    stage = value.upper()
    if stage in YIDU_STAGES:
        return stage
    if value in YIDU_PROFILE_TO_STAGE:
        return YIDU_PROFILE_TO_STAGE[value]
    choices = ", ".join((*YIDU_STAGES, *YIDU_PROFILE_TO_STAGE))
    raise ValueError(
        f"Unknown YiDu stage/profile {stage_or_profile!r}; choose {choices}"
    )


def profile_for_yidu_stage(stage_or_profile: str) -> str:
    """Return the immutable observer profile name for one YiDu stage."""

    return YIDU_STAGE_TO_PROFILE[resolve_yidu_stage(stage_or_profile)]


def validate_yidu_stage_matrix(
    matrix: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> None:
    """Validate the cumulative, exactly-one-module transition contract.

    A valid matrix has the exact public stage/module axes, an all-false ``B0``,
    never disables an earlier module, and enables exactly the declared module
    at every adjacent transition.
    """

    candidate = YIDU_STAGE_MODULE_MATRIX if matrix is None else matrix
    if not isinstance(candidate, Mapping):
        raise TypeError("YiDu module matrix must be a mapping")
    if tuple(candidate.keys()) != YIDU_STAGES:
        raise ValueError(
            "YiDu module matrix stages must exactly match "
            + ", ".join(YIDU_STAGES)
        )

    previous = {module: False for module in YIDU_MODULES}
    for index, stage in enumerate(YIDU_STAGES):
        row = candidate[stage]
        if not isinstance(row, Mapping):
            raise TypeError(f"YiDu matrix row {stage} must be a mapping")
        if tuple(row.keys()) != YIDU_MODULES:
            raise ValueError(
                f"YiDu matrix row {stage} modules must exactly match "
                + ", ".join(YIDU_MODULES)
            )
        if any(not isinstance(value, bool) for value in row.values()):
            raise TypeError(
                f"YiDu matrix row {stage} values must be Boolean"
            )

        current = dict(row)
        removed = [
            module
            for module in YIDU_MODULES
            if previous[module] and not current[module]
        ]
        added = [
            module
            for module in YIDU_MODULES
            if current[module] and not previous[module]
        ]
        if removed:
            raise ValueError(
                f"YiDu stage {stage} disables earlier module(s): "
                + ", ".join(removed)
            )
        expected = YIDU_STAGE_ADDED_MODULE[stage]
        if index == 0:
            if added:
                raise ValueError("YiDu B0 must keep every module disabled")
        elif added != [expected]:
            raise ValueError(
                f"YiDu transition {YIDU_STAGES[index - 1]}->{stage} "
                f"must add only {expected}; got {added}"
            )
        previous = current


def adjacent_yidu_added_module(
    previous_stage_or_profile: str,
    next_stage_or_profile: str,
) -> str:
    """Return the sole module added by one valid adjacent transition."""

    previous = resolve_yidu_stage(previous_stage_or_profile)
    current = resolve_yidu_stage(next_stage_or_profile)
    previous_index = YIDU_STAGES.index(previous)
    current_index = YIDU_STAGES.index(current)
    if current_index != previous_index + 1:
        raise ValueError(
            "YiDu transitions must advance exactly one adjacent stage; "
            f"got {previous}->{current}"
        )
    added = [
        module
        for module in YIDU_MODULES
        if YIDU_STAGE_MODULE_MATRIX[current][module]
        and not YIDU_STAGE_MODULE_MATRIX[previous][module]
    ]
    expected = YIDU_STAGE_ADDED_MODULE[current]
    if added != [expected]:
        raise RuntimeError(
            f"Invalid built-in YiDu transition {previous}->{current}: {added}"
        )
    return added[0]


def _mutable_mapping_copy(value: Any, *, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return deepcopy(dict(value))


def apply_yidu_ablation(
    config: Mapping[str, Any],
    stage_or_profile: str,
) -> Dict[str, Any]:
    """Deep-copy ``config`` and apply one canonical YiDu observer stage.

    Existing per-module hyperparameters under
    ``online_refinement.yidu_ablation.<module>`` are retained, while all
    lifecycle flags are overwritten by the canonical stage.  This prevents a
    stale YAML ``enabled`` or ``mutate`` value from activating a module.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if "online_refinement" not in config:
        raise ValueError("Missing online_refinement section")
    if not isinstance(config["online_refinement"], Mapping):
        raise TypeError("online_refinement must be a mapping")

    stage = resolve_yidu_stage(stage_or_profile)
    profile = YIDU_STAGE_TO_PROFILE[stage]
    module_flags = dict(YIDU_STAGE_MODULE_MATRIX[stage])

    result: Dict[str, Any] = deepcopy(dict(config))
    online = _mutable_mapping_copy(
        result["online_refinement"], name="online_refinement"
    )
    result["online_refinement"] = online
    previous_section = _mutable_mapping_copy(
        online.get("yidu_ablation"),
        name="online_refinement.yidu_ablation",
    )

    module_sections: Dict[str, Dict[str, Any]] = {}
    for module, enabled in module_flags.items():
        module_config = _mutable_mapping_copy(
            previous_section.get(module),
            name=f"online_refinement.yidu_ablation.{module}",
        )
        module_config.update(
            {
                "enabled": bool(enabled),
                "observer_only": True,
                "mutate": False,
            }
        )
        module_sections[module] = module_config

    # Replace every lifecycle field with the canonical immutable contract.
    # B0 does not execute a YiDu observer; A1-A6 collect diagnostics only.
    section = {
        key: value
        for key, value in previous_section.items()
        if key not in YIDU_MODULES
        and key
        not in {
            "schema",
            "stage",
            "profile",
            "enabled",
            "observer_only",
            "mutate",
            "collect_diagnostics",
            "frozen_b6",
            "added_module",
            "modules",
        }
    }
    section.update(
        {
            "schema": YIDU_SCHEMA,
            "stage": stage,
            "profile": profile,
            "enabled": stage != "B0",
            "observer_only": True,
            "mutate": False,
            "collect_diagnostics": stage != "B0",
            "frozen_b6": True,
            "added_module": YIDU_STAGE_ADDED_MODULE[stage],
            "modules": module_flags,
            **module_sections,
        }
    )
    online["yidu_ablation"] = section
    return result


def apply_yidu_ablation_stage(
    config: Mapping[str, Any],
    stage_or_profile: str,
) -> Dict[str, Any]:
    """Alias emphasizing stage-based use."""

    return apply_yidu_ablation(config, stage_or_profile)


def apply_yidu_ablation_profile(
    config: Mapping[str, Any],
    stage_or_profile: str,
) -> Dict[str, Any]:
    """Alias emphasizing profile-based use."""

    return apply_yidu_ablation(config, stage_or_profile)


# Fail at import time if a future edit accidentally turns the public ablation
# into a non-incremental or multi-module transition.
validate_yidu_stage_matrix()


__all__ = [
    "YIDU_MODULES",
    "YIDU_MODULE_MATRIX",
    "YIDU_PROFILE_STAGE_MAP",
    "YIDU_PROFILE_TO_STAGE",
    "YIDU_SCHEMA",
    "YIDU_STAGES",
    "YIDU_STAGE_ADDED_MODULE",
    "YIDU_STAGE_MODULE_MATRIX",
    "YIDU_STAGE_PROFILE_MAP",
    "YIDU_STAGE_TO_PROFILE",
    "adjacent_yidu_added_module",
    "apply_yidu_ablation",
    "apply_yidu_ablation_profile",
    "apply_yidu_ablation_stage",
    "profile_for_yidu_stage",
    "resolve_yidu_stage",
    "validate_yidu_stage_matrix",
]

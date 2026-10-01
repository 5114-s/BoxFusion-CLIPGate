"""Strict, dependency-free CLI overrides for the residual-track observer."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, MutableMapping, Optional


RESIDUAL_TRACK_OBSERVER_PROFILE = "residual_track_observer"


def configure_residual_min_semantic_score(
    config: MutableMapping[str, Any],
    value: Optional[Real],
    *,
    cli_profile: Optional[str] = None,
) -> MutableMapping[str, Any]:
    """Override one graph gate after selecting the exact observer profile.

    This helper intentionally exposes no geometry, overlap, source-mode, or
    output-mutation setting.  It runs before model construction so an invalid
    value/profile fails closed.
    """

    if not isinstance(config, MutableMapping):
        raise TypeError("config must be a mutable mapping")
    if value is None:
        return config
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            "residual minimum semantic score must be a real number"
        )
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(
            "residual minimum semantic score must lie in [0, 1]"
        )

    online = config.setdefault("online_refinement", {})
    if not isinstance(online, MutableMapping):
        raise TypeError("online_refinement must be a mutable mapping")
    configured_profile = online.get("ablation_profile")
    selected_profile = (
        cli_profile if cli_profile is not None else configured_profile
    )
    if selected_profile != RESIDUAL_TRACK_OBSERVER_PROFILE:
        raise ValueError(
            "--residual-track-min-semantic-score is allowed only with "
            f"{RESIDUAL_TRACK_OBSERVER_PROFILE}"
        )
    if online.get("enabled") is not True:
        raise ValueError(
            "--residual-track-min-semantic-score requires enabled online "
            "refinement"
        )
    if (
        cli_profile is not None
        and configured_profile is not None
        and configured_profile != cli_profile
    ):
        raise ValueError(
            "CLI and resolved residual-track ablation profiles disagree"
        )

    observer = online.setdefault("residual_track_observer", {})
    if not isinstance(observer, MutableMapping):
        raise TypeError(
            "online_refinement.residual_track_observer must be a mutable "
            "mapping"
        )
    graph = observer.setdefault("missing_instance_graph", {})
    if not isinstance(graph, MutableMapping):
        raise TypeError(
            "residual_track_observer.missing_instance_graph must be a "
            "mutable mapping"
        )
    graph["minimum_semantic_score"] = score
    return config


__all__ = [
    "RESIDUAL_TRACK_OBSERVER_PROFILE",
    "configure_residual_min_semantic_score",
]

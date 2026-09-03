"""Strict, dependency-free CLI configuration for TriFusion observers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping, Optional


TRIFUSION_OBSERVER_PROFILE = "trifusion_plus10_observer"


def configure_trifusion_ap50_gate(
    config: MutableMapping[str, Any],
    checkpoint: Optional[str],
    *,
    cli_profile: Optional[str] = None,
) -> MutableMapping[str, Any]:
    """Attach one AP50 gate checkpoint to the exact observer profile.

    The helper mutates the already loaded command-line configuration.  It is
    deliberately independent of the runtime controller so invalid profile,
    path, or mapping state fails before model/GPU construction.
    """

    if not isinstance(config, MutableMapping):
        raise TypeError("config must be a mutable mapping")
    if checkpoint is None:
        return config
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError(
            "trifusion AP50 gate checkpoint must be a non-empty path"
        )

    online = config.setdefault("online_refinement", {})
    if not isinstance(online, MutableMapping):
        raise TypeError("online_refinement must be a mutable mapping")
    configured_profile = online.get("ablation_profile")
    selected_profile = (
        cli_profile if cli_profile is not None else configured_profile
    )
    if selected_profile != TRIFUSION_OBSERVER_PROFILE:
        raise ValueError(
            "--trifusion-ap50-gate-checkpoint is allowed only with "
            f"{TRIFUSION_OBSERVER_PROFILE}"
        )
    if online.get("enabled") is not True:
        raise ValueError(
            "--trifusion-ap50-gate-checkpoint requires enabled online "
            "refinement"
        )
    if (
        cli_profile is not None
        and configured_profile is not None
        and configured_profile != cli_profile
    ):
        raise ValueError(
            "CLI and resolved online ablation profiles disagree"
        )

    path = Path(checkpoint.strip())
    if path.suffix.casefold() != ".npz":
        raise ValueError(
            "trifusion AP50 gate checkpoint must use the .npz format"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"TriFusion AP50 gate checkpoint not found: {path}"
        )

    observer = online.setdefault("trifusion_observer", {})
    if not isinstance(observer, MutableMapping):
        raise TypeError(
            "online_refinement.trifusion_observer must be a mutable mapping"
        )
    safety_gate = observer.setdefault("safety_gate", {})
    if not isinstance(safety_gate, MutableMapping):
        raise TypeError(
            "online_refinement.trifusion_observer.safety_gate must be a "
            "mutable mapping"
        )
    safety_gate.update(
        {
            "enabled": True,
            "checkpoint": str(path),
            "collect_diagnostics": True,
            "mutate": False,
        }
    )
    return config


__all__ = [
    "TRIFUSION_OBSERVER_PROFILE",
    "configure_trifusion_ap50_gate",
]

"""Strict dependency-free CLI configuration for YiDu A6."""

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping, Optional

from boxfusion.yidu_ablation import YIDU_STAGE_TO_PROFILE


YIDU_GATE_PROFILE = YIDU_STAGE_TO_PROFILE["A6"]


def configure_yidu_ap50_gate(
    config: MutableMapping[str, Any],
    checkpoint: Optional[str],
    *,
    cli_profile: Optional[str] = None,
) -> MutableMapping[str, Any]:
    """Attach a train-only gate checkpoint to the exact A6 observer."""

    if not isinstance(config, MutableMapping):
        raise TypeError("config must be a mutable mapping")
    if checkpoint is None:
        return config
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("YiDu AP50 gate checkpoint must be a non-empty path")
    online = config.setdefault("online_refinement", {})
    if not isinstance(online, MutableMapping):
        raise TypeError("online_refinement must be a mutable mapping")
    selected = (
        cli_profile
        if cli_profile is not None
        else online.get("ablation_profile")
    )
    if selected != YIDU_GATE_PROFILE:
        raise ValueError(
            "--yidu-ap50-gate-checkpoint is allowed only with "
            f"{YIDU_GATE_PROFILE}"
        )
    if online.get("enabled") is not True:
        raise ValueError(
            "--yidu-ap50-gate-checkpoint requires enabled online refinement"
        )
    if (
        cli_profile is not None
        and online.get("ablation_profile") != cli_profile
    ):
        raise ValueError("CLI and resolved YiDu profiles disagree")
    path = Path(checkpoint.strip())
    if path.suffix.casefold() != ".npz":
        raise ValueError("YiDu AP50 gate checkpoint must use .npz")
    if not path.is_file():
        raise FileNotFoundError(
            f"YiDu AP50 gate checkpoint not found: {path}"
        )
    yidu = online.setdefault("yidu_ablation", {})
    if not isinstance(yidu, MutableMapping):
        raise TypeError("online_refinement.yidu_ablation must be a mapping")
    gate = yidu.setdefault("quality_gate", {})
    if not isinstance(gate, MutableMapping):
        raise TypeError("yidu_ablation.quality_gate must be a mapping")
    gate.update(
        {
            "enabled": True,
            "observer_only": True,
            "mutate": False,
            "checkpoint": str(path),
        }
    )
    return config


__all__ = ["YIDU_GATE_PROFILE", "configure_yidu_ap50_gate"]

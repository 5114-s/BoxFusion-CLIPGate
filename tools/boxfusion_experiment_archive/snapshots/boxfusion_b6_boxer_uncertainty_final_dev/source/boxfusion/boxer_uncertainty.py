"""Boxer scalar aleatoric-confidence weighting for reliable views.

Boxer's AleHead predicts one scalar ``log(sigma^2)`` per proposal and exposes
``q = 1 / (1 + sigma^2)`` as its confidence.  This module deliberately treats
that value as a scalar reliability term; it is not a 7-DoF covariance.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


VALID_MODES = ("disabled", "observer", "active")

DEFAULT_BOXER_UNCERTAINTY_CONFIG = {
    "mode": "disabled",
    "confidence_power": 1.0,
    "minimum_confidence": 0.05,
    "diagnostics_dir": "",
}

DEFAULT_FINAL_BOXER_UNCERTAINTY_CONFIG = {
    "mode": "disabled",
    "confidence_power": 1.0,
    "minimum_confidence": 0.05,
    "diagnostics_dir": "",
}


def resolve_boxer_uncertainty_config(
    reliable_view_cfg: Mapping,
) -> Dict[str, object]:
    """Resolve and validate the opt-in uncertainty-fusion configuration."""

    raw = reliable_view_cfg.get("boxer_uncertainty", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            "reliable_views.boxer_uncertainty must be a mapping"
        )
    cfg: Dict[str, object] = dict(DEFAULT_BOXER_UNCERTAINTY_CONFIG)
    cfg.update(raw)
    cfg["mode"] = str(cfg["mode"]).lower()
    cfg["confidence_power"] = float(cfg["confidence_power"])
    cfg["minimum_confidence"] = float(cfg["minimum_confidence"])
    cfg["diagnostics_dir"] = str(cfg.get("diagnostics_dir", ""))

    if cfg["mode"] not in VALID_MODES:
        choices = ", ".join(VALID_MODES)
        raise ValueError(
            f"boxer_uncertainty.mode must be one of {choices}"
        )
    if (
        not np.isfinite(cfg["confidence_power"])
        or cfg["confidence_power"] < 0.0
    ):
        raise ValueError(
            "boxer_uncertainty.confidence_power must be finite and "
            "non-negative"
        )
    if (
        not np.isfinite(cfg["minimum_confidence"])
        or not 0.0 < cfg["minimum_confidence"] <= 1.0
    ):
        raise ValueError(
            "boxer_uncertainty.minimum_confidence must lie in (0, 1]"
        )
    return cfg


def resolve_final_boxer_uncertainty_config(
    reliable_view_cfg: Mapping,
) -> Dict[str, object]:
    """Resolve the post-B6, geometry-only uncertainty configuration.

    This configuration is intentionally separate from
    :func:`resolve_boxer_uncertainty_config`.  The latter changes geometry
    during online fusion, while this route is permitted to change only the
    final exported geometry after association and B6 score calibration have
    finished.
    """

    raw = reliable_view_cfg.get("final_boxer_uncertainty", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            "reliable_views.final_boxer_uncertainty must be a mapping"
        )
    cfg: Dict[str, object] = dict(
        DEFAULT_FINAL_BOXER_UNCERTAINTY_CONFIG
    )
    cfg.update(raw)
    cfg["mode"] = str(cfg["mode"]).lower()
    cfg["confidence_power"] = float(cfg["confidence_power"])
    cfg["minimum_confidence"] = float(cfg["minimum_confidence"])
    cfg["diagnostics_dir"] = str(cfg.get("diagnostics_dir", ""))

    if cfg["mode"] not in VALID_MODES:
        choices = ", ".join(VALID_MODES)
        raise ValueError(
            f"final_boxer_uncertainty.mode must be one of {choices}"
        )
    if (
        not np.isfinite(cfg["confidence_power"])
        or cfg["confidence_power"] < 0.0
    ):
        raise ValueError(
            "final_boxer_uncertainty.confidence_power must be finite and "
            "non-negative"
        )
    if (
        not np.isfinite(cfg["minimum_confidence"])
        or not 0.0 < cfg["minimum_confidence"] <= 1.0
    ):
        raise ValueError(
            "final_boxer_uncertainty.minimum_confidence must lie in (0, 1]"
        )
    if cfg["mode"] != "disabled" and not cfg["diagnostics_dir"]:
        raise ValueError(
            "final_boxer_uncertainty.diagnostics_dir is required in "
            "observer/active mode"
        )
    return cfg


def fixed_topk_uncertainty_reweighting(
    base_selection: Mapping[str, np.ndarray],
    boxer_confidence: np.ndarray,
    boxer_geometry_applied: np.ndarray,
    cfg: Mapping,
) -> Dict[str, np.ndarray]:
    """Reweight the *existing* G0 Top-K members without selecting again.

    The selected row ids and their order are copied byte-for-byte from the
    base selection.  Only selected rows whose geometry actually came from
    Boxer receive ``q**power``; CuTR fallback rows and invalid confidence
    values remain neutral.  This is the causal primitive used by the
    final-only route and deliberately contains no ranking operation.
    """

    base_weights = np.asarray(
        base_selection["weights"], dtype=np.float64
    ).reshape(-1)
    base_selected = np.asarray(
        base_selection["selected_indices"], dtype=np.int64
    ).reshape(-1)
    boxer_confidence = np.asarray(
        boxer_confidence, dtype=np.float64
    ).reshape(-1)
    boxer_geometry_applied = np.asarray(
        boxer_geometry_applied, dtype=bool
    ).reshape(-1)

    count = base_weights.shape[0]
    if boxer_confidence.shape[0] != count:
        raise ValueError(
            "Boxer confidence must have one value per reliable-view "
            "candidate"
        )
    if boxer_geometry_applied.shape[0] != count:
        raise ValueError(
            "Boxer provenance must have one value per reliable-view "
            "candidate"
        )
    if base_selected.size == 0 or base_selected.size > count:
        raise ValueError("base selection must contain between 1 and N rows")
    if (
        np.any(base_selected < 0)
        or np.any(base_selected >= count)
        or np.unique(base_selected).size != base_selected.size
    ):
        raise ValueError("base selected indices must be unique and in range")
    if not np.isfinite(base_weights).all() or np.any(base_weights <= 0.0):
        raise ValueError("base reliable-view weights must be finite and positive")

    selected_mask = np.zeros(count, dtype=bool)
    selected_mask[base_selected] = True
    valid_confidence = (
        np.isfinite(boxer_confidence)
        & (boxer_confidence >= 0.0)
        & (boxer_confidence <= 1.0)
    )
    weighted_rows = (
        selected_mask & boxer_geometry_applied & valid_confidence
    )
    factors = np.ones(count, dtype=np.float64)
    safe_confidence = np.clip(
        boxer_confidence[weighted_rows],
        float(cfg["minimum_confidence"]),
        1.0,
    )
    factors[weighted_rows] = np.power(
        safe_confidence,
        float(cfg["confidence_power"]),
    )

    adjusted_weights = base_weights.copy()
    adjusted_weights[base_selected] *= factors[base_selected]
    if not np.isfinite(adjusted_weights).all() or np.any(
        adjusted_weights <= 0.0
    ):
        raise ValueError(
            "uncertainty-adjusted weights must be finite and positive"
        )

    base_effective_weights = np.zeros(count, dtype=np.float64)
    selected_base_weights = base_weights[base_selected]
    base_effective_weights[base_selected] = selected_base_weights / max(
        float(selected_base_weights.mean()), 1e-12
    )
    uncertainty_effective_weights = np.zeros(count, dtype=np.float64)
    selected_adjusted_weights = adjusted_weights[base_selected]
    uncertainty_effective_weights[base_selected] = (
        selected_adjusted_weights
        / max(float(selected_adjusted_weights.mean()), 1e-12)
    )
    effective_weights_changed = not np.allclose(
        uncertainty_effective_weights,
        base_effective_weights,
        rtol=1e-7,
        atol=1e-7,
    )

    result = {
        key: np.asarray(value).copy()
        if isinstance(value, np.ndarray)
        else value
        for key, value in base_selection.items()
    }
    result.update(
        {
            "base_weights": base_weights.astype(np.float32),
            "base_selected_indices": base_selected.copy(),
            "boxer_confidence": boxer_confidence.astype(np.float32),
            "boxer_geometry_applied": boxer_geometry_applied.copy(),
            "boxer_confidence_valid": valid_confidence,
            "uncertainty_weighted_rows": weighted_rows,
            "uncertainty_factors": factors.astype(np.float32),
            "uncertainty_weights": adjusted_weights.astype(np.float32),
            "weights": adjusted_weights.astype(np.float32),
            # These four fields make the fixed-membership contract explicit.
            "ranked_indices": base_selected.copy(),
            "selected_indices": base_selected.copy(),
            "selected_mask": selected_mask,
            "selected_weights": adjusted_weights[base_selected].astype(
                np.float32
            ),
            "base_effective_weights": base_effective_weights.astype(
                np.float32
            ),
            "uncertainty_effective_weights": (
                uncertainty_effective_weights.astype(np.float32)
            ),
            "candidate_weights_changed": np.asarray(
                bool(np.any(np.abs(factors[base_selected] - 1.0) > 1e-7)),
                dtype=bool,
            ),
            "effective_weights_changed": np.asarray(
                effective_weights_changed, dtype=bool
            ),
            "selection_changed": np.asarray(False, dtype=bool),
            "ranking_changed": np.asarray(False, dtype=bool),
        }
    )
    return result


def uncertainty_adjusted_selection(
    base_selection: Mapping[str, np.ndarray],
    boxer_confidence: np.ndarray,
    boxer_geometry_applied: np.ndarray,
    cfg: Mapping,
) -> Dict[str, np.ndarray]:
    """Return a stable counterfactual Top-K selection using Boxer confidence.

    Only rows whose actual geometry came from Boxer receive ``q**power``.
    CuTR fallback rows and invalid Boxer confidence values receive a neutral
    factor of one.  The input selection is never mutated.
    """

    base_weights = np.asarray(
        base_selection["weights"], dtype=np.float64
    ).reshape(-1)
    detector_confidence = np.asarray(
        base_selection["confidence"], dtype=np.float64
    ).reshape(-1)
    base_selected = np.asarray(
        base_selection["selected_indices"], dtype=np.int64
    ).reshape(-1)
    boxer_confidence = np.asarray(
        boxer_confidence, dtype=np.float64
    ).reshape(-1)
    boxer_geometry_applied = np.asarray(
        boxer_geometry_applied, dtype=bool
    ).reshape(-1)

    count = base_weights.shape[0]
    for name, values in (
        ("detector confidence", detector_confidence),
        ("Boxer confidence", boxer_confidence),
        ("Boxer provenance", boxer_geometry_applied),
    ):
        if values.shape[0] != count:
            raise ValueError(
                f"{name} must have one value per reliable-view candidate"
            )
    if base_selected.size == 0 or base_selected.size > count:
        raise ValueError("base selection must contain between 1 and N rows")
    if not np.isfinite(base_weights).all() or np.any(base_weights <= 0.0):
        raise ValueError("base reliable-view weights must be finite and positive")

    factors = np.ones(count, dtype=np.float64)
    valid_confidence = (
        np.isfinite(boxer_confidence)
        & (boxer_confidence >= 0.0)
        & (boxer_confidence <= 1.0)
    )
    weighted_rows = boxer_geometry_applied & valid_confidence
    safe_confidence = np.clip(
        boxer_confidence[weighted_rows],
        float(cfg["minimum_confidence"]),
        1.0,
    )
    factors[weighted_rows] = np.power(
        safe_confidence,
        float(cfg["confidence_power"]),
    )
    adjusted_weights = base_weights * factors
    if not np.isfinite(adjusted_weights).all() or np.any(
        adjusted_weights <= 0.0
    ):
        raise ValueError(
            "uncertainty-adjusted weights must be finite and positive"
        )

    ranked = np.lexsort(
        (
            np.arange(count, dtype=np.int64),
            -detector_confidence,
            -adjusted_weights,
        )
    ).astype(np.int64)
    selected = ranked[: base_selected.size]
    selected_mask = np.zeros(count, dtype=bool)
    selected_mask[selected] = True

    # Compare the actual normalized contribution of each source row.  This
    # avoids claiming an effective fusion change when only an unselected row
    # was down-weighted, or when every selected row received the same scalar
    # and mean normalization cancelled it exactly.
    base_effective_weights = np.zeros(count, dtype=np.float64)
    base_selected_weights = base_weights[base_selected]
    base_effective_weights[base_selected] = base_selected_weights / max(
        float(base_selected_weights.mean()), 1e-12
    )
    uncertainty_effective_weights = np.zeros(count, dtype=np.float64)
    adjusted_selected_weights = adjusted_weights[selected]
    uncertainty_effective_weights[selected] = adjusted_selected_weights / max(
        float(adjusted_selected_weights.mean()), 1e-12
    )
    selection_changed = not np.array_equal(
        np.sort(selected), np.sort(base_selected)
    )
    ranking_changed = not np.array_equal(selected, base_selected)
    candidate_weights_changed = bool(np.any(np.abs(factors - 1.0) > 1e-7))
    effective_weights_changed = not np.allclose(
        uncertainty_effective_weights,
        base_effective_weights,
        rtol=1e-7,
        atol=1e-7,
    )

    result = {
        key: np.asarray(value).copy()
        if isinstance(value, np.ndarray)
        else value
        for key, value in base_selection.items()
    }
    result.update(
        {
            "base_weights": base_weights.astype(np.float32),
            "base_selected_indices": base_selected.copy(),
            "boxer_confidence": boxer_confidence.astype(np.float32),
            "boxer_geometry_applied": boxer_geometry_applied.copy(),
            "boxer_confidence_valid": valid_confidence,
            "uncertainty_weighted_rows": weighted_rows,
            "uncertainty_factors": factors.astype(np.float32),
            "uncertainty_weights": adjusted_weights.astype(np.float32),
            "weights": adjusted_weights.astype(np.float32),
            "ranked_indices": ranked,
            "selected_indices": selected,
            "selected_mask": selected_mask,
            "selected_weights": adjusted_weights[selected].astype(
                np.float32
            ),
            "base_effective_weights": base_effective_weights.astype(
                np.float32
            ),
            "uncertainty_effective_weights": (
                uncertainty_effective_weights.astype(np.float32)
            ),
            "candidate_weights_changed": np.asarray(
                candidate_weights_changed, dtype=bool
            ),
            "effective_weights_changed": np.asarray(
                effective_weights_changed, dtype=bool
            ),
            "selection_changed": np.asarray(
                selection_changed, dtype=bool
            ),
            "ranking_changed": np.asarray(
                ranking_changed, dtype=bool
            ),
        }
    )
    return result


__all__ = [
    "DEFAULT_BOXER_UNCERTAINTY_CONFIG",
    "DEFAULT_FINAL_BOXER_UNCERTAINTY_CONFIG",
    "VALID_MODES",
    "fixed_topk_uncertainty_reweighting",
    "resolve_boxer_uncertainty_config",
    "resolve_final_boxer_uncertainty_config",
    "uncertainty_adjusted_selection",
]

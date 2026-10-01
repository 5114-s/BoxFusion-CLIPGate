"""Zero-write integration boundary for frozen-B6/TR3D experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tr3d_residual_cache import (
    TR3DResidualCache,
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)


TR3D_RESIDUAL_OBSERVER_SCHEMA = "boxfusion.tr3d_residual_observer.v1"


@dataclass(frozen=True)
class TR3DResidualObservation:
    """Observer result whose prediction member is the original object."""

    predictions: Any
    residual: TR3DResidualCache
    schema: str = TR3D_RESIDUAL_OBSERVER_SCHEMA
    observer_only: bool = True
    mutation_enabled: bool = False
    applied_count: int = 0

    def diagnostic_summary(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "scene_id": self.residual.scene_id,
            "prefix_id": self.residual.prefix_id,
            "observer_only": self.observer_only,
            "mutation_enabled": self.mutation_enabled,
            "applied_count": self.applied_count,
            "proposal_count": self.residual.proposal_count,
            "runtime_s": self.residual.runtime_s,
            "checkpoint_sha256": self.residual.checkpoint_sha256,
            "config_sha256": self.residual.config_sha256,
        }


class TR3DResidualObserver:
    """Load immutable proposals without reading or changing B6 predictions."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        checkpoint_sha256: str,
        config_sha256: str,
        prefix_id: str = "full",
    ) -> None:
        self.cache_root = Path(cache_root)
        self.checkpoint_sha256 = checkpoint_sha256
        self.config_sha256 = config_sha256
        self.prefix_id = prefix_id

    def observe(self, scene_id: str, predictions: Any) -> TR3DResidualObservation:
        """Return the exact prediction object and read-only residual arrays."""

        residual = load_tr3d_residual_cache(
            tr3d_residual_cache_path(
                self.cache_root, scene_id, self.prefix_id
            ),
            expected_scene_id=scene_id,
            expected_prefix_id=self.prefix_id,
            expected_checkpoint_sha256=self.checkpoint_sha256,
            expected_config_sha256=self.config_sha256,
        )
        result = TR3DResidualObservation(
            predictions=predictions,
            residual=residual,
        )
        # This assertion is intentionally local and unconditional: future
        # refactors cannot silently replace the frozen prediction container.
        assert result.predictions is predictions
        return result

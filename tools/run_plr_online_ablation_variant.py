#!/usr/bin/env python3
"""Run demo.py with a process-local PLR ablation.

This launcher does not modify the production online implementation.  It
patches only the PLR class selected by ``build_online_candidate_map`` before
``demo.py`` is executed in the same process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def argument_value(flag: str) -> str:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"missing required demo argument: {flag}") from error


def main() -> None:
    mode = os.environ.get("PLR_ABLATION_MODE", "").strip()
    if mode not in {"no_native_dedup", "direct_matched_budget"}:
        raise RuntimeError(f"unsupported PLR_ABLATION_MODE: {mode!r}")

    import boxfusion.online_candidate_map as online

    original = online.OnlineProposalRecovery
    scene = argument_value("--seq")

    if mode == "no_native_dedup":
        original_diagnostics = original.diagnostics

        def never_overlaps_native(self, corners, native_corners):
            return False

        def no_dedup_diagnostics(self):
            row = original_diagnostics(self)
            row.update({
                "ablation_mode": mode,
                "native_dedup_disabled": True,
            })
            return row

        original._overlaps_native = never_overlaps_native
        original.diagnostics = no_dedup_diagnostics
    else:
        manifest_path = Path(os.environ["PLR_REFERENCE_MANIFEST"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        quota = int(manifest["per_scene"][scene]["m1p"])

        class DirectMatchedBudget(original):
            """Causal direct proposal admission with a matched output cap."""

            def __init__(self, **kwargs):
                # Retain a strong pool so the output budget, rather than the
                # internal storage cap, controls the comparison.
                kwargs["max_births"] = max(4096, quota)
                kwargs["max_tracks"] = max(4096, int(kwargs.get("max_tracks", 0)))
                super().__init__(**kwargs)

            def _required_views(self, track):
                # Remove temporal confirmation: every valid post-NMS proposal
                # is immediately eligible for the output pool.
                return 1

            @staticmethod
            def _rank_and_cap(rows):
                ordered = sorted(
                    rows,
                    key=lambda row: (
                        -float(row["raw_mean_score"]),
                        int(row["confirmation_frame_id"]),
                        int(row["track_id"]),
                    ),
                )
                return ordered[:quota]

            def rows(self):
                return self._rank_and_cap(super().rows())

            def terminal_rows(self, native_corners):
                return self._rank_and_cap(
                    super().terminal_rows(native_corners)
                )

            def diagnostics(self):
                row = super().diagnostics()
                row.update({
                    "ablation_mode": mode,
                    "matched_output_quota": quota,
                    "temporal_confirmation_disabled": True,
                    "selection": "top raw proposal score within causal state",
                })
                return row

        online.OnlineProposalRecovery = DirectMatchedBudget

        cache_root = os.environ.get("PLR_EVIDENCE_CACHE_ROOT", "").strip()
        if cache_root:
            import boxfusion.online_candidate_runtime as runtime

            provider_class = runtime.FrozenWeDetectBoxerProvider
            original_process = provider_class.process
            original_provider_diagnostics = provider_class.diagnostics

            def cached_process(self, **kwargs):
                evidence = original_process(self, **kwargs)
                if not hasattr(self, "_plr_anchor_cache"):
                    self._plr_anchor_cache = []
                    self._plr_anchor_cache_scene = str(kwargs["scene_id"])
                self._plr_anchor_cache.append((
                    int(kwargs["frame_id"]),
                    np.asarray(evidence.anchor_ids, dtype=np.int64).copy(),
                    np.asarray(evidence.anchor_corners, dtype=np.float32).copy(),
                    np.asarray(evidence.anchor_scores, dtype=np.float32).copy(),
                    np.asarray(evidence.proposal_ids, dtype=np.int64).copy(),
                    np.asarray(evidence.proposal_corners, dtype=np.float32).copy(),
                    np.asarray(evidence.proposal_scores, dtype=np.float32).copy(),
                ))
                return evidence

            def cached_diagnostics(self):
                row = original_provider_diagnostics(self)
                records = getattr(self, "_plr_anchor_cache", [])
                if records:
                    destination = Path(cache_root)
                    destination.mkdir(parents=True, exist_ok=True)
                    scene_id = self._plr_anchor_cache_scene
                    frame_ids = np.concatenate([
                        np.full(len(ids), frame, dtype=np.int64)
                        for frame, ids, _, _, _, _, _ in records
                    ])
                    anchor_ids = np.concatenate([
                        ids for _, ids, _, _, _, _, _ in records
                    ])
                    corners = np.concatenate([
                        value for _, _, value, _, _, _, _ in records
                    ])
                    scores = np.concatenate([
                        value for _, _, _, value, _, _, _ in records
                    ])
                    proposal_frame_ids = np.concatenate([
                        np.full(len(ids), frame, dtype=np.int64)
                        for frame, _, _, _, ids, _, _ in records
                    ])
                    proposal_ids = np.concatenate([
                        ids for _, _, _, _, ids, _, _ in records
                    ])
                    proposal_corners = np.concatenate([
                        value for _, _, _, _, _, value, _ in records
                    ])
                    proposal_scores = np.concatenate([
                        value for _, _, _, _, _, _, value in records
                    ])
                    path = destination / f"{scene_id}.npz"
                    temporary = path.with_name(path.name + ".tmp")
                    with temporary.open("wb") as handle:
                        np.savez(
                            handle,
                            frame_ids=frame_ids,
                            anchor_ids=anchor_ids,
                            corners_raw=corners,
                            scores=scores,
                            proposal_frame_ids=proposal_frame_ids,
                            proposal_ids=proposal_ids,
                            proposal_corners_raw=proposal_corners,
                            proposal_scores=proposal_scores,
                        )
                    os.replace(temporary, path)
                    row["anchor_evidence_cache"] = str(path)
                    row["anchor_evidence_rows"] = int(len(scores))
                    row["proposal_evidence_rows"] = int(len(proposal_scores))
                return row

            provider_class.process = cached_process
            provider_class.diagnostics = cached_diagnostics

    sys.argv[0] = str(ROOT / "demo.py")
    runpy.run_path(str(ROOT / "demo.py"), run_name="__main__")


if __name__ == "__main__":
    main()

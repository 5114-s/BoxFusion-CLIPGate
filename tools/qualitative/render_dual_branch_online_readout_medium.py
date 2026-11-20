#!/usr/bin/env python3
"""Re-render the audited scene0599_01 readout with method-figure line weights.

This wrapper only changes the prediction and audit stroke widths.  It uses the
same cached predictions, mesh, camera, selected records, colors, and renderer
as ``render_dual_branch_online_readout.py``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/qualitative/render_dual_branch_online_readout.py"


def load_source():
    spec = importlib.util.spec_from_file_location("dual_branch_readout_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    source = load_source()
    source.OUTPUT = (
        ROOT
        / "figure_assets/online_detection_readout_real_dual_branch_medium_20261107"
    )
    source.PREDICTION_LINEWIDTH = 2.10
    source.AUDIT_LINEWIDTH = 1.10
    source.main()


if __name__ == "__main__":
    main()

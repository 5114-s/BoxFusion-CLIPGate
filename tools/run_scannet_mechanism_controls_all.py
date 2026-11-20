#!/usr/bin/env python3
"""Run/resume the four prespecified CPU-only control groups in dependency order."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"reports/scannet_mechanism_protocol_controls_20261029"


def main():
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MPLCONFIGDIR="/tmp/recar3d_mechanism_mpl")
    commands = [
        ["tools/evaluate_scannet_mechanism_controls.py", "--phase", "cached"],
        ["tools/replay_scannet_unified_history_controls.py"],
        ["tools/evaluate_scannet_mechanism_controls.py", "--phase", "sensitivity"],
        ["tools/validate_scannet_mechanism_controls.py"],
        ["tools/evaluate_scannet_mechanism_controls.py", "--phase", "report"],
        ["tools/build_supplement_docx.py", str(OUT/"实验结果与论文补充内容.md"),
         str(OUT/"实验结果与论文补充内容.docx"), "--footer",
         "ReCaR-3D Mechanism and Protocol Controls — 2026-10-29"],
    ]
    for index, arguments in enumerate(commands, 1):
        print(f"phase {index}/{len(commands)}: {arguments[0]}", flush=True)
        subprocess.run([sys.executable, *arguments], cwd=ROOT, env=environment, check=True)
    print(f"All four experiment groups complete: {OUT}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create/verify the R2-bound, GT-free E961 outer protocol preregistration V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_e961_outer_eval_v1 import (  # noqa: E402
    CONFIG_PATH,
    seal_protocol_preregistration,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    path, value = seal_protocol_preregistration(args.config)
    print(json.dumps({
        "complete": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": value["schema"],
        "outer_train_r2_independent_review_pass": True,
        "expanded_training_receipt_access": False,
        "expanded_checkpoint_access": False,
        "anchor_array_access": False,
        "fold0_gt_access": False,
        "fold1_access": False,
        "official_validation_access": False,
        "gpu_started": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Static safety contract for the P2-v3 audit entry point."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_scannet_p2v3.sh"


def test_p2v3_audit_script_is_valid_and_uses_frozen_gate_report():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "validate_p2v3_run_artifacts.py" in source
    assert "report_p2v3_reliability_fusion_recall.py" in source
    assert "minimum-delta-r25-pp" in source
    assert "minimum-delta-r50-pp" in source
    assert "BOXFUSION_P2V3_REQUIRE_GO" in source
    assert "report_p2v2_local_geometry_recall.py" not in source

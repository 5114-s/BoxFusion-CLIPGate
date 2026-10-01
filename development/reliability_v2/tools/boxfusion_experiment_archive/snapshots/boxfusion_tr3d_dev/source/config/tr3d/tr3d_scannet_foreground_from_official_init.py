"""One-class TR3D training initialized from the deterministic ScanNet18 fold.

This is deliberately ``load_from`` rather than ``resume``: the converted
checkpoint preserves the source optimizer only for payload auditability, but
its classifier slots have the obsolete 18-class shape.
"""

_base_ = ["./tr3d_scannet_foreground.py"]

load_from = "models/tr3d_1xb16_scannet-3d-foreground-init.pth"
resume = False

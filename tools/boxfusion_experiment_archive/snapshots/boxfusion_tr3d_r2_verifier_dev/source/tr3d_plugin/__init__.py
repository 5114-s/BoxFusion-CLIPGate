"""BoxFusion plugins for the genuine TR3D supervised-hybrid branch."""

from .class_agnostic_head import TR3DClassAgnosticHead
from .foreground_scannet_dataset import TR3DForegroundScanNetDataset

__all__ = [
    "TR3DClassAgnosticHead",
    "TR3DForegroundScanNetDataset",
]

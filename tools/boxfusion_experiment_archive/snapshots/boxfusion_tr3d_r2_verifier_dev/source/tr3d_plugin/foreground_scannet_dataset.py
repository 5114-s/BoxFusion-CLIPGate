"""One-class ScanNet dataset adapter for genuine class-agnostic TR3D."""

from copy import deepcopy

from mmdet3d.datasets import ScanNetDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class TR3DForegroundScanNetDataset(ScanNetDataset):
    """ScanNet detection dataset whose only semantic class is foreground.

    MMDetection3D's base :class:`ScanNetDataset` interprets a custom
    ``metainfo.classes`` tuple as a named subset of its fixed 18 categories.
    Consequently ``classes=('foreground',)`` fails before the already
    collapsed annotation labels can be loaded. This adapter changes only the
    dataset class metadata; point loading, axis alignment, box structures and
    evaluation behavior remain upstream ScanNet implementations.
    """

    METAINFO = deepcopy(ScanNetDataset.METAINFO)
    METAINFO["classes"] = ("foreground", )

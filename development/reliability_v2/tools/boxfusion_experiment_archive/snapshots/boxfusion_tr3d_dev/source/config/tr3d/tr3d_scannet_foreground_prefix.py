"""Trajectory-prefix fine-tuning config for class-agnostic TR3D.

The annotation exporter filters each aligned GT by observed-prefix point
support. Prefix point files remain in ``world_unaligned``; the inherited
``GlobalAlignment`` transform aligns them once at model input.
"""

_base_ = ['./tr3d_scannet_foreground.py']

data_root = 'data/tr3d_scannet/'

train_dataloader = dict(
    dataset=dict(
        # Prefixes already provide four deterministic trajectory states per
        # scene; avoid multiplying the set by the official RepeatDataset x15.
        _delete_=True,
        type='TR3DForegroundScanNetDataset',
        data_root=data_root,
        ann_file='annotations/scannet_infos_prefix_train_foreground.pkl',
        pipeline={{_base_.train_pipeline}},
        filter_empty_gt=True,
        metainfo=dict(classes=('foreground',)),
        box_type_3d='Depth',
    ))

work_dir = 'work_dirs/tr3d_scannet_foreground_prefix'

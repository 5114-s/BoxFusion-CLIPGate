"""Genuine one-class TR3D on leak-free ScanNet-train partitions.

This inherits the official OpenMMLab TR3D common model and official ScanNet
dataset base by source-relative paths at stable v1.4.0 commit
fe25f7a51d36e3702f961e198894580d83c4387b (TR3D tree
e7d4f3eaaeb39473babf52ef47af0d81fe72d6c8). The ScanNet-specific pipeline
below is copied without semantic changes from
``projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py``. The only model
change is the class-agnostic assignment head: all 18 detection categories are
collapsed to foreground label 0, while every GT is assigned independently on
every feature level.

Run ``tools/prepare_tr3d_scannet.py`` before loading this config.
"""

_base_ = [
    '../../third_party/mmdetection3d/projects/TR3D/configs/tr3d.py',
    '../../third_party/mmdetection3d/configs/_base_/datasets/scannet-3d.py',
]

custom_imports = dict(
    imports=['projects.TR3D.tr3d', 'tr3d_plugin'],
    allow_failed_imports=False,
)

data_root = 'data/tr3d_scannet/'
metainfo = dict(classes=('foreground',))

# Official TR3D ScanNet pipeline (PR #2274), retained here so loading this
# source checkout never depends on the installed-package ``mmdet3d::`` scope.
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(type='TR3DPointSample', num_points=0.33),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.02, 0.02],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='NormalizePointsColor', color_mean=None),
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

model = dict(
    bbox_head=dict(
        _delete_=True,
        type='TR3DClassAgnosticHead',
        in_channels=128,
        voxel_size=0.01,
        pts_center_threshold=6,
        num_reg_outs=6,
    ),
    # Keep low-score proposals in the observer cache. Deployment filtering is
    # performed only after frozen-union recall and precision are audited.
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0.01),
)

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RepeatDataset',
        times=15,
        dataset=dict(
            type='TR3DForegroundScanNetDataset',
            data_root=data_root,
            ann_file='annotations/scannet_infos_train_foreground.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            metainfo=metainfo,
            box_type_3d='Depth',
        )))

# Calibration and audit are disjoint subsets of official ScanNet train.
# Official validation is intentionally not used for checkpoint selection.
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='TR3DForegroundScanNetDataset',
        data_root=data_root,
        ann_file='annotations/scannet_infos_calibration_foreground.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Depth',
    ))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='TR3DForegroundScanNetDataset',
        data_root=data_root,
        ann_file='annotations/scannet_infos_audit_foreground.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Depth',
    ))

val_evaluator = dict(type='IndoorMetric')
test_evaluator = dict(type='IndoorMetric')

work_dir = 'work_dirs/tr3d_scannet_foreground'

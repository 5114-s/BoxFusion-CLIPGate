backend_args = None
ca1m_dataset = dict(
    backend_args=None,
    box_type_3d='Depth',
    data_prefix=dict(pts='points'),
    data_root='/extra/ZhaoX/tr3d_ca1m_train100_v1/',
    filter_empty_gt=False,
    metainfo=dict(categories=dict(foreground=0), classes=('foreground', )),
    type='TR3DForegroundCA1MDataset')
custom_hooks = [
    dict(after_iter=True, type='EmptyCacheHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.TR3D.tr3d',
        'tr3d_plugin',
    ])
data_root = '/extra/ZhaoX/tr3d_ca1m_train100_v1/'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=11268,
        max_keep_ckpts=1,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
metainfo = dict(categories=dict(foreground=0), classes=('foreground', ))
model = dict(
    backbone=dict(
        depth=34,
        in_channels=3,
        norm='batch',
        num_planes=(
            64,
            128,
            128,
            128,
        ),
        type='TR3DMinkResNet'),
    bbox_head=dict(
        in_channels=128,
        num_reg_outs=6,
        pts_center_threshold=6,
        type='TR3DClassAgnosticHead',
        voxel_size=0.01),
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    neck=dict(
        in_channels=(
            64,
            128,
            128,
            128,
        ), out_channels=128, type='TR3DNeck'),
    test_cfg=dict(iou_thr=0.5, nms_pre=1000, score_thr=0.01),
    train_cfg=dict(),
    type='MinkSingleStage3DDetector')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=11268,
        gamma=0.1,
        milestones=[
            7512,
            10329,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=True, seed=0)
resume = False
source_point_root = '/extra/ZhaoX/tr3d_ca1m_train100_v1/points'
test_cfg = None
test_dataloader = None
test_evaluator = None
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='DEPTH',
        load_dim=6,
        shift_height=False,
        type='LoadPointsFromFile',
        use_color=True,
        use_dim=[
            0,
            1,
            2,
            3,
            4,
            5,
        ]),
    dict(rotation_axis=2, type='GlobalAlignment'),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(color_mean=None, type='NormalizePointsColor'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_iters=11268, type='IterBasedTrainLoop')
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        dataset=dict(
            ann_file=
            'annotations/ca1m_infos_train_folds234_visible_foreground_xfit_v2_formal.pkl',
            backend_args=None,
            box_type_3d='Depth',
            data_prefix=dict(pts='/extra/ZhaoX/tr3d_ca1m_train100_v1/points'),
            data_root=
            '/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/data/tr3d_ca1m_visible_xfit_v2_formal',
            filter_empty_gt=False,
            metainfo=dict(
                categories=dict(foreground=0), classes=('foreground', )),
            pipeline=[
                dict(
                    backend_args=None,
                    coord_type='DEPTH',
                    load_dim=6,
                    shift_height=False,
                    type='LoadPointsFromFile',
                    use_color=True,
                    use_dim=[
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                    ]),
                dict(type='LoadAnnotations3D'),
                dict(rotation_axis=2, type='GlobalAlignment'),
                dict(num_points=0.33, type='TR3DPointSample'),
                dict(
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    sync_2d=False,
                    type='RandomFlip3D'),
                dict(
                    rot_range=[
                        -0.02,
                        0.02,
                    ],
                    scale_ratio_range=[
                        0.9,
                        1.1,
                    ],
                    shift_height=False,
                    translation_std=[
                        0.1,
                        0.1,
                        0.1,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(color_mean=None, type='NormalizePointsColor'),
                dict(
                    keys=[
                        'points',
                        'gt_bboxes_3d',
                        'gt_labels_3d',
                    ],
                    type='Pack3DDetInputs'),
            ],
            type='TR3DForegroundCA1MDataset'),
        times=1,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='DEPTH',
        load_dim=6,
        shift_height=False,
        type='LoadPointsFromFile',
        use_color=True,
        use_dim=[
            0,
            1,
            2,
            3,
            4,
            5,
        ]),
    dict(type='LoadAnnotations3D'),
    dict(rotation_axis=2, type='GlobalAlignment'),
    dict(num_points=0.33, type='TR3DPointSample'),
    dict(
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        sync_2d=False,
        type='RandomFlip3D'),
    dict(
        rot_range=[
            -0.02,
            0.02,
        ],
        scale_ratio_range=[
            0.9,
            1.1,
        ],
        shift_height=False,
        translation_std=[
            0.1,
            0.1,
            0.1,
        ],
        type='GlobalRotScaleTrans'),
    dict(color_mean=None, type='NormalizePointsColor'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = None
val_dataloader = None
val_evaluator = None
work_dir = '/extra/ZhaoX/tr3d_ca1m_work_dirs/ca1m_xfit_v2_formal_r2/ca1m_xfit_v2_formal_outer_dev_seed0_r2'
xfit_heldout_fold = 0
xfit_role = 'outer_dev'
xfit_root = '/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/data/tr3d_ca1m_visible_xfit_v2_formal'
xfit_train_folds = [
    2,
    3,
    4,
]

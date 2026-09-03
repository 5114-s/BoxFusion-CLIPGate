"""Final-only official ScanNet-val evaluation for the frozen TR3D checkpoint."""

_base_ = ['./tr3d_scannet_foreground.py']

data_root = 'data/tr3d_scannet/'

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/scannet_infos_official_val_foreground.pkl',
        metainfo=dict(classes=('foreground',)),
    ))

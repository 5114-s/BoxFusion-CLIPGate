"""One real optimizer-step smoke derived from the foreground-init config.

This config is deliberately unsuitable for an accuracy experiment. It exists
only to exercise one genuine data -> forward -> loss -> backward -> optimizer
path without validation or a long-running job.
"""

_base_ = ["./tr3d_scannet_foreground_from_official_init.py"]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(times=1),
)
train_cfg = dict(
    _delete_=True,
    type="IterBasedTrainLoop",
    max_iters=1,
    val_interval=2,
)
val_cfg = None
val_dataloader = None
val_evaluator = None
param_scheduler = []
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        by_epoch=False,
        max_keep_ckpts=1,
    ),
)
log_processor = dict(by_epoch=False)
work_dir = "work_dirs/tr3d_smoke/never_use_config_default"

# Trained TR3D epoch-12 dual-anchor observer

This worktree is isolated from every active BoxFusion experiment:

```text
/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_epoch12_observer_dev
```

It reads the trained one-class TR3D checkpoint and frozen prediction roots,
but writes caches, logs, reports, and temporary files only below this worktree.
It never appends, rescales, reorders, or overwrites a BoxFusion prediction.

## Frozen inputs

- trained TR3D: `a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448`
- B6 anchor: `40.0434 / 33.5492 / 12.1613`
- G0 Selective Boxer anchor: `40.2787 / 35.4508 / 15.2181`
- G0 prediction tree: `fe10ee44a56bc5160a606cc8f6d68c90ed08775874130c5e7840e7e184b74e17`

The generic anchor manifest hashes all 100 prediction files and the G0
quality/YOLOE checkpoints, config, launcher, and relevant implementation
files. Both anchors are verified before and after each observer audit.

## Safe launch

The experiment launcher refuses GPUs with active compute processes. It also
places `TMPDIR`, XDG, Torch, Matplotlib, and CUDA caches under `/data` because
the host root filesystem has very little free space.

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_epoch12_observer_dev

bash scripts/run_epoch12_observer_experiment.sh \
  0,1 tr3d_t1_epoch12_fp32_observer10_v1
```

The command first runs one immutable-cache smoke scene, then the frozen ten
scenes, and finally audits the same TR3D cache against B6 and G0 separately.

To wait for another protected job and require three consecutive idle GPU
polls before launch:

```bash
BOXFUSION_TR3D_WAIT_PID=<protected-driver-pid> \
bash scripts/wait_for_idle_gpus_then_run_epoch12.sh \
  0,1 tr3d_t1_epoch12_fp32_observer10_v1
```

## Interpretation contract

This first experiment uses `prefix_id=full`, hence it measures a full-scene
proposal ceiling, not online trajectory-prefix performance. TR3D's observer
threshold `0.01` retains candidates for analysis and does not change the
BoxFusion/G0 score threshold `0.4`.

The original preregistered continuation gate remains unchanged:

- delta union-oracle Recall@0.25 at least 0.08;
- delta union-oracle Recall@0.50 at least 0.05;
- at least five novel IoU-0.50 TPs in at least three scenes;
- raw-stream novel IoU-0.50 precision upper bound at least 0.15.

The report additionally includes a fixed TR3D score grid. That grid diagnoses
whether low raw precision is caused by score calibration, but no threshold may
be selected on the ten validation scenes. Any deployment threshold must be
frozen on the disjoint ScanNet-train calibration split.

# TR3D ScanNet18 → foreground initialization

This artifact is only an initialization for genuine one-class TR3D training.
It is not a trained one-class checkpoint and makes no AP or speed claim.

Official TR3D uses 18 independent sigmoid focal-loss outputs, rather than a
softmax. For source logits `z_c(x) = b_c + w_c^T x`, the converter defines:

```text
p_fg(x) = 1 - product_c(1 - sigmoid(z_c(x)))
```

One affine foreground logit cannot represent this nonlinear union exactly.
The converter takes its first-order Taylor approximation at the zero input
feature:

```text
b_fg = logit(1 - product_c(1 - sigmoid(b_c)))
w_fg = sum_c [sigmoid(b_c) / p_fg(0)] * w_c
```

This exactly matches the union prior at the expansion point and its local
feature gradient. A plain mean would retain approximately a single-class
background prior and discard the semantics of “any of 18 classes”.

Only `head.conv_cls.kernel` (`[128,18] → [128,1]`) and
`head.conv_cls.bias` (`[1,18] → [1,1]`) change. All 260 backbone, neck,
normalization, and regression tensors are byte-exact. Metadata and optimizer
payloads are retained for auditing; never resume the old optimizer.

Generate once (the converter refuses overwrite):

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev
/home/admin1/miniconda3/envs/boxfusion2/bin/python \
  tools/convert_tr3d_foreground_checkpoint.py
```

Re-audit source/output/provenance:

```bash
/home/admin1/miniconda3/envs/boxfusion2/bin/python \
  tools/verify_tr3d_foreground_checkpoint.py
```

Strict-load the actual one-class config on CPU or GPU:

```bash
BOXFUSION_TR3D_ENV=openmmlab \
  bash scripts/smoke_load_tr3d_foreground_init.sh cpu
BOXFUSION_TR3D_ENV=openmmlab \
  bash scripts/smoke_load_tr3d_foreground_init.sh cuda
```

For training use
`config/tr3d/tr3d_scannet_foreground_from_official_init.py`. It sets
`load_from` and explicitly disables `resume`.

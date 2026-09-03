# CA-1M C0 corrected reproduction protocol

This repository treats the released `1.0` score written by the upstream
`demo.py` as an export bug.  The formal C0 baseline is therefore:

- the exact 107 CA-1M validation scenes from upstream `data/val.txt`;
- upstream BoxFusion geometry, association, and multi-view fusion unchanged;
- `detection.score_thresh = 0.4` and `gap = 20`;
- the actual detector confidence from `all_pred_box.scores` in every saved
  prediction tuple;
- no CLIP appearance gate, Top-K memory, B6, Boxer, TR3D, C3, or other added
  module.

The upstream first-frame empty-detection fallback (`score_thresh / 4`) remains
unchanged.  Removing it would alter the upstream proposal/geometry path, so a
strict per-frame `>= 0.4` variant must be reported as a separate ablation.

The paper's `31.19 / 25.51 / 8.82` is retained only as a published reference.
Because its released exporter wrote constant scores, the corrected real-score
result is the baseline used for all subsequent paired ablations and is not
required to be byte- or metric-identical to that published reference.

Formal entry point:

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
bash scripts/run_ca1m_c0_score04_real_score_full107.sh 0,1
```

The runner fails closed unless data, evaluation view, and predictions each
contain the exact same 107-scene set.  It also validates every resumed pickle
before skipping it and rejects constant-score outputs.

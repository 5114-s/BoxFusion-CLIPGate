# CA-1M-native B6 train-only subset protocol

This stage prepares data provenance only. It does not train a model, access
validation ground truth, change predictions, or touch the canonical103/derived107
evaluation roots.

The default protocol reads Apple's official BoxFusion copies of `data/train.txt`
and `data/val.txt`, verifies that the source splits have zero scene-ID overlap, and
selects 100 train scenes by ascending
`SHA256("boxfusion.ca1m-native-b6.train100.v1" + NUL + scene_id)`. The scene list,
URLs, per-URL hashes, selection keys, source-list hashes, and artifact hashes are
frozen under `manifests/ca1m_native_b6_train100_v1`. Re-running with different
inputs or selection settings fails closed instead of rewriting that manifest.

Preflight (safe default; downloads nothing):

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
bash scripts/prepare_ca1m_native_b6_train_subset.sh
```

Explicit resumable download, only after reviewing `subset_manifest.json` and
`readiness.json`:

```bash
bash scripts/prepare_ca1m_native_b6_train_subset.sh --download
```

The download mode uses `wget --continue`, fully lists every tar before promotion,
and finally records each local tar's SHA256 in `downloaded_sha256.tsv`. It does not
extract or preprocess the archives. The default destination is
`/extra/ZhaoX/ca1m_apple_train_tars`; override it with
`BOXFUSION_CA1M_NATIVE_B6_TRAIN_TAR_ROOT`.

This route is train-only but not training-free once a quality model is eventually
fit. Any later model must disclose CA-1M train supervision and must never use the
fixed10, canonical103, or derived107 validation labels for fitting or threshold
selection.

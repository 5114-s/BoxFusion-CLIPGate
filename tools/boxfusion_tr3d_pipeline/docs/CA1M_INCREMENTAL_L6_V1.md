# CA-1M incremental/L6 v1 protocol

This route ports the **method**, not a ScanNet-trained artifact.  Its current
state is a static, fail-closed contract.  It does not start a GPU, open GT,
load an incremental policy, or write predictions.

## Module mapping

| ScanNet best-route component | CA-1M replacement |
|---|---|
| Frozen `B6 + G0 + terminal-R3` anchor | Sealed final-base + CA-native B6 v2 + authorized terminal-benefit-v2 active anchor |
| ScanNet incremental TR3D worker | New CA processed-train100 causal observer using only the sealed CA-scratch TR3D binding |
| ScanNet novelty gate | New CA-only dual scene-grouped `novel25` / `quality50` logistic policy |
| L6 source-aware ranking | Same deterministic visibility/support/free-space/fused-geometry rank, under a CA-only schema |
| Low-score materialization | Append-only rows globally ranked at distinct positive float32 scores below every sealed anchor |

The accepted TR3D model identity is fixed by
`checkpoint_binding.json` (`19b8c3...5043`) and its CA-scratch checkpoint
(`d3ba6c...b4a7`).  The binding may live in the older v3 *manifest namespace*
because it is only a model identity.  No v3 terminal proposal/overlay cache is
accepted.

## Scientific split

The 100 CA training scenes retain the preregistered 60/20/20 roles:

- folds 2/3/4 (60 scenes): model fitting and normalization only;
- fold 0 (20 scenes): thresholds and candidate capacity only;
- fold 1 (20 scenes): one-time locked activation audit only.

The terminal-benefit anchor used to construct L6 labels must be cross-fitted:
every train100 scene must be produced by an upstream model that did not fit on
that scene.  This prevents stacking leakage from making the incremental gate
look stronger than it is.  Official validation is represented only by a
hashed 107-scene identity exclusion list; no validation GT or prediction is a
legal training input.

Training stops without relaxing thresholds if any minimum is missing:

- fit60: at least 120 candidates, 20 novel25 positives, 20 novel25 negatives,
  and 10 quality50 positives;
- dev20: at least 20 candidates and positive discoveries in at least 4 scenes;
- locked20: at least 20 candidates and positive discoveries in at least 4
  scenes.

## Required new upstream chain

All bindings are all-or-none and currently `null`:

1. the sealed 100-scene final-base identity manifest and anchor root;
2. native-B6 v2 collection plus its newly trained authorized checkpoint;
3. terminal-v4 proposal/overlay seal using the CA-scratch TR3D model;
4. terminal-benefit policy schema v2, its active anchor, and an exact100
   cross-fit receipt.

This deliberately rejects the ScanNet novelty/source policies, ScanNet TR3D
hashes, CA terminal v1/v2/v3 caches, and the old CA native-B6 checkpoints or
diagnostics.  The processed CA train100 RGB-D directory is data, not an old B6
model/cache, and is the only explicitly permitted `...train100_v1` exception.

## Current safe commands

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
bash scripts/run_ca1m_incremental_l6_train100_v1.sh --static-preflight
```

The expected report has `static_contract_ready=true`,
`run_authorized=false`, and reports the pending new upstream chain.  The run
entry point must fail with exit code 2:

```bash
bash scripts/run_ca1m_incremental_l6_train100_v1.sh --run
```

Do not replace the `null` bindings or implement collection/training merely by
pointing at an older cache.  Once all new upstream artifacts are sealed, add a
separate create-only execution driver and tests that prove observer GT-free
collection, train-only label construction, fold isolation, append identity,
and scores below the anchor floor before changing either authorization flag.

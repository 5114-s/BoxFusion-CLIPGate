# CA-1M E961 TR3D terminal inputs v5 R2

Status: **static PASS; operationally blocked by design**.  This revision does
not authorize a GPU run and does not contain formal receipt paths.  It has not
read ground truth, fold 1, or official validation, and it has not created the
formal output namespace.

This is a new revision and namespace.  It does not modify or consume the
previous reviewed six-file v5 revision, and it never consumes ScanNet weights,
old CA terminal-v1-v4 candidates, overlays, evidence, policies, or gates.

## Frozen producer boundary

The outer role accepts only
`boxfusion.tr3d.ca1m_e961_outer_train_run.r2` and must pass the producer's
`verify_success_receipt` in
`tr3d_ca1m_e961_outer_train_r2.py` (SHA256
`36f2f02cbc1201aa55adb1104cb198f1cb4fda478e466101dcafc7660790a70f`).

All three inner roles accept only
`boxfusion.tr3d.ca1m_e961_inner_train_run.r2`.  Every canonical receipt must
pass `tr3d_ca1m_e961_inner_queue_r2.py::verify_success_receipt` (SHA256
`d6d7a6c30f15d6f11a8c7e84b9e27d08665f7d4af1ebe0ae7ca4aee68aec9f03`).
The legacy 60-scene schema
`boxfusion.tr3d.ca1m_xfit_r2_inner_run_receipt.v1` is explicitly rejected.
The inner verifier itself enforces the canonical
`.../runs/<queue>/roles/<order>_<role>/RUN_RECEIPT.json` location and rechecks
the checkpoint, effective config, log, exact1001 train list, optimizer history,
outer/evaluation lineage, and CA-only access claims.

Operational preflight validates the passing outer continuation, then invokes
all four producer deep verifiers, then validates a create-only run
authorization binding those exact receipt/checkpoint hashes.  Only after that
may it inspect output parents or create anything.  The checked-in pending
config stops before verifier import, receipt/checkpoint access, device use,
mkdir, or worker construction.

## Executable P/O/E/M route

- **P**: for each of `outer_dev`, `inner_holdout2`, `inner_holdout3`, and
  `inner_holdout4`, reconstruct the frozen CA train RGB-D points and run the
  role checkpoint on its excluded exact20 fold.  Caches contain detector
  proposals only: no anchor, B6, or GT.
- **O**: on CPU, load geometry and row order only from sealed final-base
  train100, load its sealed CA-native B6-v2 observer rows, and replace the
  stacked anchor score with `deployment_blend_oof_scores` from the all-fold OOF
  sidecar.  Scene, fold, row index, detector score, and exclusion identities
  are rechecked.  Deployment/in-sample B6 scores are not used.
- **E**: replay proposal frame lineage through a fresh CA-native B6 observer
  over processed CA train RGB-D, then assemble the frozen 40-D terminal-v5
  features from anchor-native, candidate-native, and relation features.
- **M**: seal four exact20 role collections and the terminal-gate-v5-compatible
  exact80 collection: fit60 from folds 2/3/4 plus reused-dev20 from fold 0.
  A separate R2 wrapper binds the new namespace and authorization.

The existing science/runtime code is reused by frozen API and SHA rather than
copied: point builder `db2c4c...1523`, worker client `aad340...db6b`, worker CLI
`e01c8b...70d0`, 40-D feature builder `5ee4c2...b0ce`, candidate scene reader
`aaaf4c...545c`, native observer `e22965...4280`, diagnostic loader
`6daea1...750`, association `b39e1c...aa2`, and generic v5 evidence/manifest
runtime `a1128f...e6a`.  Full hashes and CLI/API names are sealed in the config.
Code reuse is not artifact reuse; all P/O/E outputs live under the new
namespace and are freshly produced.

## Filesystem and resume contract

Every input is read with a no-follow descriptor and stable
device/inode/size/mtime checks.  Every output parent is checked component by
component for symlinks.  Publication uses host-writable, FUSE-safe
`O_CREAT|O_EXCL|O_NOFOLLOW`, complete in-memory bytes, `fsync`, and read-only
mode; it does not require hard links or rename-overwrite.  A crash residue is
retained and rejected.  Resume accepts only a fully reparsed artifact with
exact upstream hashes; unexpected/partial per-stage inventory fails closed.
`NAMESPACE_OWNER.json` prevents a different authorization from sharing the
namespace.

## Commands

Static preflight (the only currently passing mode):

```bash
python tools/preflight_ca1m_tr3d_e961_terminal_inputs_v5_r2.py --static
```

Operational preflight currently exits 3 before output/GPU access:

```bash
python tools/preflight_ca1m_tr3d_e961_terminal_inputs_v5_r2.py --operational
```

After a separately reviewed ready config binds all formal receipts and the run
authorization, the runner exposes:

```bash
python tools/run_ca1m_tr3d_e961_terminal_inputs_v5_r2.py --config READY.json --run-p inner_holdout2 --device cuda:0
python tools/run_ca1m_tr3d_e961_terminal_inputs_v5_r2.py --config READY.json --run-o inner_holdout2
python tools/run_ca1m_tr3d_e961_terminal_inputs_v5_r2.py --config READY.json --run-e inner_holdout2
python tools/run_ca1m_tr3d_e961_terminal_inputs_v5_r2.py --config READY.json --seal-m
```

`--run-all` performs P/O/E role-by-role and then M.  No command in this
revision joins GT or trains/activates the terminal gate.

## Current blockers

1. The canonical outer success receipt is not yet bound.
2. The three canonical E961 inner R2 success receipts are not yet bound.
3. The passing continuation and terminal-input run authorization are not yet
   bound.
4. Therefore all operational booleans remain false and no ready config has
   been sealed.

Synthetic tests cover pending-before-mkdir/GPU behavior, schema rejection,
FUSE-safe create-only publication, finite P/O round trips, OOF-vs-deployment
score separation, fresh-candidate 40-D evidence assembly, and a full synthetic
80-scene M seal accepted by the generic terminal-v5 collection loader.

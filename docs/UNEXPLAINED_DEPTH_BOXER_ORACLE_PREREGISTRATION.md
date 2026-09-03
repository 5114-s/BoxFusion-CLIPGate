# Unexplained-depth + frozen Boxer shadow/oracle preregistration

Date frozen: 2026-08-23 (before inspecting any Boxer/GT overlap result)

Protocol correction, also made before inspecting any Boxer/GT overlap: the
first API smoke and exploratory run used a shared development checkout whose
BoxerNet source was locally modified.  Those CSVs are excluded from every
oracle result.  The formal shadow below instead uses the clean inference-only
checkout at commit `1f86542dc342a4b1d474c87c97c5d1d6566d9148`.

A second pre-oracle correction freezes NumPy/Torch scene seeds to zero because
the released ScanNet loader subsamples depth with NumPy, and obtains the exact
per-scene keyframe count from the sealed T05 proposal-cache manifest.  Earlier
clean smoke CSVs without both controls are excluded.  No candidate/GT overlap
was inspected before either correction.

Schedule audit then found one invalid-pose T05 keyframe
(`scene0606_01/1325`).  The released Boxer loader removes that frame before
applying its frame-count cap and would otherwise substitute the future frame
2825.  The final v4 run therefore caps by the number of finite-pose IDs in the
sealed schedule (66/112/30), treats frame 1325 as an explicit supplemental
abstention, and never substitutes a later frame.  This correction was made
after a preliminary internal overlap check; no model, score threshold, depth
threshold, or gate was changed, and all earlier candidate metrics are excluded.

A strict input-access audit after v4 found that released `run_boxer.py` passes
`full_annotations.json` to `ScanNetLoader` even when `gt2d=False`.  The v4
proposal path did not consume those boxes (the three logs reported zero), but
opening an annotation file is incompatible with the stronger no-GT runtime
contract.  The final v5 run therefore wraps the unmodified clean loader in the
process only, forces `annotation_path=None`, and installs an access guard that
would abort on any attempt to open `full_annotations.json`.  It additionally
seals the DINOv3 weight and OWLv2 LVIS+ text-embedding cache hashes.  This
correction was made after v4 metrics were visible; no model, prompt, schedule,
seed, score threshold, depth threshold, or gate was changed, and all v4
candidate/oracle metrics are excluded from the final result.

## Purpose

Measure whether a frozen, general proposal source contains enough geometry that
is complementary to T05 to make a **+10 absolute AP-point** target possible.
This stage is diagnostic only.  It must not add, delete, reorder, rescore, fuse,
or relabel a native BoxFusion prediction.

## Frozen native arm

- T05 protocol: ScanNet, `score_thresh=0.5`, appearance gate disabled,
  Reliable-View Top-K with `top_k=3`, frozen CuTR and frozen CLIP.
- Formal evaluation replaces every prediction score with exactly `1.0`.
- Preflight scenes, in fixed order:
  `scene0568_00`, `scene0606_01`, `scene0377_02`.
- Baseline prediction root:
  `results/scannet_graw_e2_replay1_score05`.

## Frozen universal proposal shadow

- Source: clean inference-only Boxer checkout
  `/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer`, commit
  `1f86542dc342a4b1d474c87c97c5d1d6566d9148`.
- 2D detector: frozen OWLv2 base patch16 ensemble.
- Text prompts: released `lvisplus` vocabulary (1,220 prompts); GT labels are
  never used as prompts.
- 2D threshold: `0.25` (the checked-in `run_boxer.py` default).
- 2D per-class NMS IoU: `0.50` (released default).
- 3D lifter: frozen released BoxerNet
  `boxernet_hw960in2x6d768-c88128f8.ckpt`, SHA-256
  `d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f`.
- DINOv3 prior SHA-256:
  `4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea`.
- OWLv2 checkpoint is read-only from the shared weight store, SHA-256
  `14aa78ffe7b13e5b3ebf55845bc9a07e339a095cfd88f4c4e8f726b38ce1ebbf`;
  no shared source file is imported.
- 3D threshold: `0.50` (released default).
- Detector/lifter resolution: `960 x 960`.
- Input cadence: the exact sealed T05 keyframe schedule (frame 0 and every
  25th frame, `record_count` 66/113/30).  Frozen Boxer runs on its finite-pose
  subset 66/112/30; invalid-pose frame 1325 is recorded as an abstention.
- Reproducibility: `PYTHONHASHSEED=0`, NumPy seed 0, Torch seed 0, CUDA seed 0,
  and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, reset once at each scene start.
- Temporal branch: released causal online tracker only (`--track`); no offline
  fusion.  Raw per-keyframe proposals and terminal online tracks are reported
  separately.
- All weights remain frozen; no optimizer, fitting, fine-tuning, train split,
  or GT access is permitted.

## Frozen unexplained-depth test

The test is applied after proposal generation in a separate inference-only
sidecar.  It does not choose thresholds from GT.

- RGB-D sampling stride: 4 pixels.
- Valid metric depth: `[0.10 m, 8.00 m]`.
- Native explanation geometry: final T05 boxes for the loose analytical
  oracle; expand each AABB by `0.05 m`.
- Candidate support: points inside the Boxer OBB.
- Voxel size: `0.05 m`, signed floor quantization.
- Minimum candidate support: 16 unique voxels.
- Minimum unexplained support: 16 unique voxels.
- Unexplained ratio gate: at least `0.50` of candidate-supported voxels lie
  outside every expanded T05 AABB.

Using final T05 boxes makes this particular depth test future-aware.  It is
therefore an **offline oracle filter**, not a deployable online gate.  A later
active implementation must replace it with query-before-commit past-only
native memory without changing these geometry constants.

## Oracle outputs and promotion rule

For IoU `0.15`, `0.25`, and `0.50`, report all of the following:

1. T05 exact official constant-score AP, greedy TP, and unmatched GT count.
2. Raw Boxer per-view proposal coverage and maximum matching.
3. Unexplained-depth-gated Boxer coverage and maximum matching.
4. Terminal online-track coverage and maximum matching.
5. T05-unmatched GT recovered by each candidate pool.
6. A GT-selected, fixed-native-prefix candidate suffix AP upper bound, marked
   `oracle_only=true`, `deployable=false`, and `gt_used=true`.

The proposal branch has enough measured headroom to support the requested goal
only if the unexplained-depth candidate-pool oracle improves official AP by at
least **+10.0 points at all three IoU thresholds**.  Three scenes can reject a
weak mechanism but cannot promote it; a passing preflight must be repeated on
a larger sealed scene set before any birth path is enabled.

## Isolation and non-interference

- Boxer writes only its own shadow CSV/log directory.
- The oracle is a separate process that may read GT but cannot write native
  predictions or an active-result directory.
- The shadow process has no GT argument and no import from ScanNet evaluation
  code.
- Native T05 pickle SHA-256 values are recorded before and after the run and
  must be identical.
- No Boxer category, embedding, or score is allowed to modify the original
  CLIP category/embedding path.

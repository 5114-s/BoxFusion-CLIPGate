# Boxer-Past3 S1 H10 frozen-proposal contract

Date: 2026-08-23

## Scope

This contract covers only frozen OWLv2 + Boxer proposal inference for the S1
one-shot holdout.  It does not define a birth rule and does not authorize any
prediction mutation.  The proposal job must not open ScanNet annotations or
use ground truth, CLIP features, or future frames.

## Fixed holdout

The ten scenes in
`evaluation/data_util/meta_data/scannetv2_boxer_past3_s1_holdout10.txt` were
fixed before any per-scene H10 ground-truth inspection.  They are the first ten
official validation scenes after removing the three S0 development scenes.
The list SHA-256 is
`8965d0534ed3028f85d8b0ea7227d348a6faa1387b858ddf42c3183bd9ebdf90`.

The S0 development scenes `scene0568_00`, `scene0606_01`, and
`scene0377_02` are forbidden in H10.  H10 is a one-shot promotion gate.  If it
is opened with GT, it becomes a development/gating split; a later unchanged
run on the remaining 87 untouched official scenes is required for a
confirmatory result.

## Frozen proposal profile

- OWLv2 Base Patch16 Ensemble checkpoint and frozen 1,220-prompt LVIS+
  taxonomy;
- frozen BoxerNet `boxernet_hw960in2x6d768-c88128f8.ckpt` and DINOv3
  backbone;
- Boxer repository commit
  `1f86542dc342a4b1d474c87c97c5d1d6566d9148` with a clean worktree;
- `thresh2d=0.25`, `thresh3d=0.5`, detector height/width profile `960`,
  bfloat16 inference, tracking enabled;
- online keyframes in the sealed T05 gap-25 schedule; invalid poses are
  omitted without substituting future frames;
- `annotation_path=None`, with any attempted access to
  `full_annotations.json` rejected at runtime.

The schedule manifests are read only from
`/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals/scannet-score05-gap25-postfilter-v2`.
They provide frame IDs only; their cached CuTR proposal tensors are not used as
S1 detections.

## Native invariants

The native prefix is the completed T05 result in
`results/scannet_topk_fusion_score05`: `score_thresh=0.5`, appearance gate
disabled, Reliable-View Top-K3.  Every H10 native prediction file is hashed
before and after frozen proposal inference and must remain byte-identical.

The proposal output is a separate shadow namespace.  No native row is added,
removed, reordered, rescored, or geometrically changed.  Formal evaluation,
if later authorized, uses constant score `1.0` for both the native prefix and
any fixed suffix.


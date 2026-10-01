# N0a frozen SAM2 image-mask lifting shadow — extra100 protocol freeze (v2 pre-AP amendment)

Originally frozen: 2026-08-29 (Asia/Shanghai), before any N0a production
execution and before any N0a access to ScanNet ground truth or evaluator
output.  Warning-evidence v2 was amended later on the same date, after the
first no-GT extra100 producer completed but still before any N0a ground-truth,
evaluator or AP access.

Amendment status: the original pre-production amendments clarified the
float64 voxel centroid, deterministic `source_id`, RGB resize, binary-mask
materialization, timing accounting, the sole CUDA `cumsum_cuda_kernel`
exception and audit-sample byte serialization.  A subsequent source audit
found that runner SHA-256
`ab49afebeb2bc92159ca2834615641d1a1c691b4af757818ffa293adb02f8769`
ignored every captured warning whose message did not contain `determin` and
did not require the two expected cumsum warnings to be present.  Although its
completed run happened to report 11,478 allowed-warning rows, that fail-open
logic is a contract violation.  Therefore every artifact below
`logs/scannet_sam2_n0a_extra100_score05`, including all scene receipts,
evidence arrays and its shard manifest, is permanently invalid for merge,
replay, capacity, AP or any admission decision.  It may be retained only as a
labelled debugging artifact and must never be resumed.

Warning-evidence v2 changes only fail-closed warning authentication and its
receipt schema.  It does not change the cohort, source set, model, prompt,
mask choice, geometry, capacity threshold, runtime threshold or replay sample.
The v2 producer must start from an empty, new output root; the frozen default
is `logs/scannet_sam2_n0a_extra100_score05_v2_strictwarn`.  Its warning policy
ID is `N0A-WARN-V2-EXACT-2XCUMSUM-POSENC-143-144`.

The permanently invalid v1 root
`logs/scannet_sam2_n0a_extra100_score05`, every descendant of that root and
every path resolving to either are forbidden v2 output roots.  A production
invocation with `--no-resume` must observe an absent or empty output root before
it authenticates or creates any output; any pre-existing scene, shard, final or
other artifact is fatal.  A completed v2 resume is permitted only after the
runner authenticates the exact assigned scene count, order and sealed F0
census; opens every scene receipt and evidence array at its exact path beneath
the current output root; verifies their hashes, schemas, run signatures, scene
identities, content hashes and per-frame warning distribution; and recomputes
the shard totals from those authenticated scene receipts.  An empty scene list,
a cross-root reference or an aggregate that does not exactly recompute is
fatal.  A shard manifest's self-recorded content hash alone never authorizes a
resume.

A second pre-AP amendment was required when the first two warning-v2 launch
attempts stopped at the strict warning gate.  The affected create-only roots
are `logs/scannet_sam2_n0a_extra100_score05_v2_strictwarn` and
`logs/scannet_sam2_n0a_extra100_score05_v2_strictwarn_failed_countdiag`.
Both contain only an empty `.evidence-spool` directory: neither contains a
scene receipt, evidence NPZ, shard/final receipt or prediction.  Neither run
opened ground truth, an evaluator or AP output.  They are failed diagnostics,
are invalid for resume/merge/AP and do not count as an N0a execution.

The failure exposed seven identical PyTorch deprecation `FutureWarning`s in
every warm prompt decoder forward.  They originate from SAM2's call to the
deprecated `torch.backends.cuda.sdp_kernel`, not from a nondeterministic
operation.  They are deliberately **not** admitted to the authenticated warning
policy.  Instead, the frozen BoxFusion provider installs compatibility policy
`N0A-TORCH251-SDPA-KERNEL-COMPAT-OLD-FLAGS-EXACT-V1` after every production
predictor build.  It leaves the third-party SAM2 tree and checkpoint unchanged
and replaces only the imported transformer's SDP context factory with
`torch.nn.attention.sdpa_kernel`, preserving the old flags exactly:
`FLASH_ATTENTION` iff `USE_FLASH_ATTN`, `EFFICIENT_ATTENTION` iff `OLD_GPU`,
`MATH` iff `(OLD_GPU and dropout_p>0) or MATH_KERNEL_ON`, and always
`CUDNN_ATTENTION` because the deprecated API's fourth argument defaults true.
`ALLOW_ALL_KERNELS=true` still returns a null context.  Missing APIs, changed
bool flags, another installed patch, a different torch build/source or a failed
post-install identity check are fatal.

The unchanged SAM2 transformer SHA-256 is
`17aac13abc8f73023f6be4b78af708df9f9f254964729421b1ba60e72a9011c1`;
the pinned torch-2.5.1 `torch/nn/attention/__init__.py` SHA-256 is
`32f2d016ba9292c182ef4e3ffa1c8b4143d16e99f9d01ffc4999366a5a342374`;
and the amended provider SHA-256 is
`f90a9791943bdceabf22d00198c13f10dd2121e6a37bd5ff455aaaee49e1227c`.
The no-GT RTX-3090 A/B receipt is
`diagnostics/n0a_sdpa_compat_gpu_ab_receipt.json`, SHA-256
`d1069d8673d970b2d74ad5aeab9bf5b03d647d51c5b88c900036f4bc2c07b420`.
Three patched forwards each emitted only the original ordered cumsum pair and
reproduced the deprecated-API mask packbits, selected-index and float32-IoU
bytes exactly.  Thus the warning contract remains exactly two records per
non-empty forward and the full-producer target remains 11,478 records.

Protocol ID:
`N0A-FROZEN-SAM2-IMAGE-BOXPROMPT-MASKLIFT-EXTRA100-SHADOW`.

## Question and experimental status

N0a asks whether a frozen, generally pretrained SAM2 mask prior can turn the
exact FastSAM locations sealed by F0 into a materially different and usable
current-view RGB-D geometry hypothesis.  It is a source-preserving geometry
shadow, not a detector, tracker, selector or birth branch.

The local `SAM2VideoPredictor` is deliberately excluded from N0a.  In the
inspected source it rejects a new object ID after tracking starts, while its
streaming frame and output dictionaries grow with the sequence.  Using it
unchanged would therefore fail the required dynamic-object and bounded-state
contracts.  N0a instead uses `SAM2ImagePredictor` independently on each
current frame.  It must not be described as a past-only video result.

For every sealed F0 source, N0a emits either one new `HS` geometry hypothesis
or an explicit abstention.  It cannot add or remove a source, change `H0`,
write a prediction pickle, create a birth, read or mutate native BoxFusion
predictions, or change CLIP, class, embedding, rank, source order or score.
Passing N0a authorizes only a separately frozen N0b past-only shadow and, if
the gates below permit, one sealed geometry-capacity evaluation.  It never
authorizes an active output.

Frozen schemas to be implemented are:

- `boxfusion.sam2_image_masklift_n0a.v1`;
- `boxfusion.scannet_sam2_image_masklift_n0a_extra100.scene.v2`;
- `boxfusion.scannet_sam2_image_masklift_n0a_extra100.shard.v2`;
- `boxfusion.scannet_sam2_image_masklift_n0a_extra100.merge.v2`.

## Exact extra100 cohort and sealed F0 identity

N0a uses records 101--200, zero-based scene indices 100--199, of
`evaluation/data_util/meta_data/scannetv2_val_f0_full200.txt`.  Equivalently,
the cohort is the last 100 newline-delimited records of that file, in their
released order.  No scene may be substituted, reordered or discovered by
directory enumeration.

- F0 full200 scene-list SHA-256:
  `0e7e722d3e93ec4b721f12293a3f1e98ca62d475b42cc8b9d491878a897e9bd1`;
- paper100 prefix SHA-256:
  `4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5`;
- derived extra100 list SHA-256, calculated over the 100 UTF-8 scene IDs
  joined by LF with a final LF:
  `f28e6997b2f50799020cf827edfe6a1520b4afe8e17de7c5564004208b8a2287`;
- sealed F0 full200 merge receipt SHA-256:
  `07249ead31ad150cb43d7a35f4c922ac70a8a2f95bcf0fcd24f61f944c1e58a1`;
- sealed F0 FastSAM checkpoint SHA-256:
  `c0be4e7ddbe4c15333d15a859c676d053c486d0a746a3be6a7a9790d52a9b6d7`.

The exact extra100 execution census, obtained by taking only scene indices
100--199 from the sealed receipt, is:

- 100 scenes;
- 6,124 scheduled keyframes;
- 5,984 successful F0 frames;
- 46,090 F0 source identities.

The canonical frame ledger is a compact JSON array, with no whitespace or
terminal newline, whose rows are
`[scene_index,scene_id,frame_ordinal,frame_id]`.  Its SHA-256 is
`f4fa82ce8a1513262fe10278eed54a33874df00c1cea0964c8afb3945b137818`.

The canonical source ledger uses the same JSON encoding and rows
`[scene_index,scene_id,frame_ordinal,frame_id,rank,raw_index,mask_sha256,
points_and_voxel_keys_sha256,tight_box_xyxy]`.  Its SHA-256 is
`1f03cc600de29930d3b314588326f35a7f0fcd995ab2700341a2469d8bbbcb00`.

Every result derives its otherwise absent F0 `source_id` by this exact ASCII
formula, with no alternative escaping or numbering:
`f"{scene_id}/frame_{frame_id:06d}/raw_{raw_index:03d}"`.  The runner must
also verify that `scene_id` and the three integer fields come from the sealed
ledger row, that `rank` is the source's zero-based order within that frame,
and that all derived IDs are unique.  This formula is part of the source
lineage and result hash.

The 100-sidecar ledger uses rows
`[scene_index,scene_id,sidecar_basename,sidecar_sha256]` in scene order and
the same compact JSON encoding.  Its SHA-256 is
`0471aa066706ed6ccd17da58bf986fb3d7434d65833c5d01d23dcac976957834`.
The runner must also verify every individual sidecar SHA recorded by the F0
merge.  A missing, extra, reordered or altered row is fatal.

## Allowed and forbidden inputs

The runner may read only:

1. the frozen protocol and the sealed F0 receipt/sidecars above;
2. each sidecar-recorded current RGB, registered depth, current rigid-valid
   camera-to-world pose and the sealed 640x480 depth-camera intrinsic;
3. the exact current source identity and its F0 `tight_box_xyxy`, `H0`, mask
   hash and point/voxel hash; and
4. the frozen SAM2 assets listed below.

F0 source confidence and residual statistics may be copied into diagnostics,
but cannot filter, reorder or choose an N0a mask or geometry.  N0a does not
need to replay FastSAM and may not reconstruct a different F0 source set.
F0 already performed mask-IoU/containment deduplication and the Top-16 cap;
N0a performs no second inter-source NMS or deduplication.

Forbidden inputs are ground truth, annotations, evaluator code or output,
all F1--F6 oracle/match ledgers, native or terminal prediction pickles,
terminal BoxFusion boxes, scene axis alignment, class labels, semantics,
CLIP features, future frames, nearest-frame or directory searches,
ScanNet-specific training/fitting/calibration, optimizer state, mutable model
parameters and online learning.  An attempted forbidden read is fatal.

Input RGB/depth/pose files must be opened only through the exact paths bound
by the F0 sidecar.  Their hashes are checked before and after N0a.  An invalid
or abstained F0 frame remains an abstention; a later valid frame is never used
as a substitute.

## Frozen SAM2 assets and execution profile

The only learned component added by N0a is SAM2.1 Hiera-L:

- checkpoint:
  `/data/ZhaoX/OVM3D-Dett/third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt`;
- checkpoint bytes: `898083611`;
- checkpoint SHA-256:
  `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`;
- config `sam2/configs/sam2.1/sam2.1_hiera_l.yaml` SHA-256:
  `545e4325aa5c19a1615d43c946b07276ed4c57214eacf1437e38fa3d9374f636`;
- `sam2/sam2_image_predictor.py` SHA-256:
  `f13e5f9d94e5c8d9d2c3622dab20c8f334c089ef2ee5ea8e199da7d332b029ba`;
- `sam2/build_sam.py` SHA-256:
  `bc49ac8e9ebf871790fa2e5f0e70bd5734e010966eff66af0c350ceaf14f3f1e`;
- `sam2/modeling/sam2_base.py` SHA-256:
  `6d81450e897d0735f9be369771f2b3fb6eadb90dd3f3ac16b3f7a8c8eb1a052a`;
- `sam2/utils/transforms.py` SHA-256:
  `ba3a64f4600c62f209206a6df3b40e3fcf133edae32fad658831bb0c2a6d1146`;
- sorted manifest SHA-256 of all 23 `sam2/**/*.py` files, where each manifest
  line is the ordinary `sha256sum` output using its relative path:
  `cc5a594bab1508ab69cbedfbb83ba8e226f848dd142a3deba8c195ee1e2469cf`.

The execution environment is `gsam2_env`: Python `3.10.19`, PyTorch
`2.5.1+cu121`, torchvision `0.20.1+cu121`, NumPy `2.2.6`, OpenCV `4.13.0`,
Hydra `1.3.2`, OmegaConf `2.3.0` and Pillow `12.0.0`.  Production requires
an RTX 3090 with compute capability 8.6; every shard records and verifies its
GPU UUID and a complete environment receipt.

All model parameters are immutable, `eval()` and inference-only.  Inference
uses CUDA BF16 autocast.  Seeds are fixed, cuDNN benchmarking and TF32 are
disabled.  PyTorch deterministic algorithms are requested with
`warn_only=True` solely because the pinned PyTorch 2.5.1 CUDA implementation
reports that SAM2's position-encoding `cumsum_cuda_kernel` has no registered
deterministic implementation.  This exception was admitted only after strict
mode stopped before the first prompt decoder produced any mask; it does not
authorize a precision/kernel fallback or any other ignored warning.  Exact
same-device mask/result-hash replay remains mandatory and is the operative
determinism gate.  SAM2 is generally pretrained outside this target experiment
and receives no ScanNet fine-tuning or calibration.

Warning-evidence v2 authenticates this exception by the complete warning
record, not a substring.  A read-only real CUDA forward with all frozen assets
and flags captured exactly two warnings with this tuple:

- category class and message instance type: exactly `builtins.UserWarning`;
- source file: exactly
  `/data/ZhaoX/OVM3D-Dett/third_party/Grounded-SAM-2/sam2/modeling/position_encoding.py`;
- source-file SHA-256:
  `14ae89d7ae68f61e2ffcba09eb171d8df9a7298332d4da99036d703294f89ec1`;
- ordered source lines: exactly `[143,144]`, once each; and
- complete UTF-8 message, including punctuation and the internal-source
  suffix:
  `cumsum_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at ../aten/src/ATen/Context.cpp:91.)`;
- complete-message UTF-8 SHA-256:
  `ed71c50715686ffdf28200dc9deb5f46c8d1f641a112c5050777b9401be90fd8`.

Every non-empty provider forward must capture exactly two warning records in
the ordered line sequence `[143,144]`.  Zero, one, three or more records;
duplicate or reversed lines; another warning category or message type; a
different absolute file even with the same basename; a different line or
message; and every additional warning are fatal.  There is no ignored-warning
branch.  A frame receipt stores only the warning-policy ID, ordered line tuple,
count, source SHA and message SHA; repeating the 333-byte message in every
frame is unnecessary.  Empty or abstained frames invoke no provider and record
zero warning rows.

## Exact current-frame mask prediction

For every successful F0 frame with at least one source:

1. Decode the exact sidecar-bound uint8 BGR JPEG at its stored resolution and
   verify its file hash.  Convert BGR to RGB, resize RGB exactly once to
   640x480 with OpenCV `INTER_LINEAR`, and make it contiguous uint8.  (The
   channel conversion and channel-wise resize commute, but this order matches
   the sealed runtime benchmark.)  Call `SAM2ImagePredictor.set_image` exactly
   once.  N0a authenticates the raw JPEG hash but deliberately does not claim
   to reproduce F0's OpenCV-4.6 decoded CuTR image-array signature: its frozen
   OpenCV-4.13 decode/resize path is the one used by the N0a runtime benchmark.
2. Form one `float32` `B x 4` array, `1 <= B <= 16`, in ascending F0 rank.
   Each row is the exact inclusive-pixel F0
   `tight_box_xyxy=[x0,y0,x1,y1]`, passed numerically unchanged.  There is no
   `+1`, padding, expansion, crop, box sweep or provider-box alternative.
3. Make one batched call with `point_coords=None`, `point_labels=None`,
   `mask_input=None`, `box=boxes`, `multimask_output=True`,
   `return_logits=False` and `normalize_coords=True`.
4. The call must return exactly three 480x640 binary-valued masks and three
   finite predicted-IoU values for every source row.  The pinned predictor
   materializes its thresholded masks as float32 NumPy arrays, so every value
   must be exactly `0` or `1` before the provider converts it to boolean;
   arbitrary logits, fractional values and negative values are fatal.  Select
   the mask with maximum predicted IoU; exact ties choose the lowest hypothesis
   index `0,1,2`.  A malformed row or non-finite score is fatal.  If the
   selected mask later fails a frozen geometry check, the source abstains;
   N0a never falls back to the second-best SAM2 mask.
5. Predicted IoU is used only for this fixed intra-source mask choice and is
   recorded as a diagnostic.  It is not a detection score, source filter,
   cross-source rank or learned target-data calibrator.

Each source remains present even when prompts overlap.  N0a may not suppress
one source because its selected mask overlaps another source's mask.

## Frozen selected-mask checks and RGB-D lifting

The selected SAM2 mask is first checked with the unchanged F0 sanity bounds:

- mask pixels in `[200,122880]`;
- tight-box minimum side at least 16 pixels;
- tight-box aspect ratio at most 6;
- finite metric-depth ratio at least 0.50, where valid depth is in
  `[0.10,6.00] m`.

F0 residual membership is not recomputed.  In particular, N0a never lifts
only an unexplained fragment and never rejects an otherwise valid selected
mask because it overlaps a current CuTR box.  The F0 source itself already
establishes frozen residual membership; `HS` uses the complete SAM2 mask.

Lifting exactly transfers the deterministic F0 construction:

1. A 3x3 binary erosion with constant-zero border removes one mask pixel.
2. Mark both endpoints of every valid horizontal or vertical depth pair whose
   absolute jump is strictly greater than `0.15 m`, and remove all marked
   pixels.
3. Back-project remaining valid pixels with the current intrinsic and pose,
   using float64 throughout the geometry path.
4. Signed-floor voxelize world points at `0.02 m`.  Unique voxel keys are
   ordered lexicographically.  For each unique key, sum all of its contributing
   world points in original row order using float64 and divide by its integer
   point count; this float64 per-voxel centroid is the voxel point.
5. Fewer than 16 unique voxels is an `HS` abstention.  Otherwise compute
   per-axis NumPy linear q02/q98 over all per-voxel centroids and enforce a
   minimum AABB extent of `0.02 m` about the unchanged centre.
6. If more than 2,048 centroids exist, retain indices
   `floor(j*(N-1)/2047), j=0,...,2047` from the ordered sequence.  Geometry
   quantiles use all centroids; the cap affects only sealed evidence.

The bounded points and voxel keys are hashed as little-endian float64 and
int64 bytes in that order.  Invalid pose, intrinsic, depth, point range or
source lineage is fatal rather than a recoverable mask abstention.

## Shadow output and bounded-state contract

Every source row records at least:

- complete F0 source identity and lineage;
- exact prompt box and selected hypothesis index;
- all three SAM2 predicted-IoU values;
- selected mask pixel count, tight box, packbits with little-endian bit order,
  and mask SHA-256;
- valid-depth, cleaned-support and unique-voxel counts;
- bounded points/voxel keys and their joint SHA-256;
- `HS` q02, q98, centre and extent, or one explicit abstention reason;
- copied `H0` geometry and a source-result SHA-256; and
- its frame ordinal and source-local geometry receipt.  Every non-empty frame
  records image-encoder, combined prompt-decoder/host-mask, batched lift and
  complete latency; its source rows reference that frame rather than claiming
  a fictitious per-source share of batched latency.

Scene, shard and merge receipts must set and validate:

- `shadow_only=true`;
- `birth_enabled=false` and `active_authorized=false`;
- `native_prediction_access=false` and `native_output_mutation=false`;
- `gt_access=false`, `annotation_access=false` and
  `evaluator_access=false`;
- `future_frame_access=false`, `history_or_tracking=false`;
- `class_clip_or_semantic_use=false`;
- `training=false` and `online_learning=false`.

Online working state is only the current image embedding, at most 16x3
decoder masks, current RGB-D inputs and at most 16 lifted point buffers.  It
is released before the next frame.  There is no cross-frame object state.
Compressed masks and points in create-only sidecars are offline evidence and
are excluded from the online-state count and timing, but their byte sizes are
reported.  Cold model weights are persistent and reported separately.

## Integrity and deterministic replay gates

All of the following must pass before any GT-assisted process is allowed:

- exactly 100 scenes, 6,124 scheduled keyframes, 5,984 successful frames and
  46,090 source/result rows;
- exactly 5,739 non-empty provider forwards and 245 successful empty frames;
  every non-empty forward must satisfy warning policy
  `N0A-WARN-V2-EXACT-2XCUMSUM-POSENC-143-144`, hence the full producer must
  contain exactly 11,478 authenticated warning records.  Each scene and shard
  must independently satisfy `warning_count == 2 * provider_forward_count`;
  aggregate equality cannot hide a zero-warning frame compensated by a
  four-warning frame;
- exact scene, frame, source and sidecar ledgers above, with zero duplicate or
  missing source IDs and exactly one `HS-or-abstention` row per source;
- all input files and receipts byte-identical before and after execution;
- zero forbidden-read, native-output-write, birth, class/semantic, future,
  training, learning or state-bound violations;
- a half-prefix replay reproduces every corresponding source-result hash;
- a fresh same-GPU replay of a prespecified 1% source sample, selected by the
  first two bytes (big-endian) of `SHA256(source_id.encode("ascii"))` being less
  than `0x0290`, reproduces exact selected indices, mask hashes, abstentions and
  geometry hashes; and
- every fresh replay frame batch independently emits the same exact two-record
  warning tuple; replay warning count must equal twice its executed frame-batch
  count.  The mirrored future-file fixture executes baseline, add, alter and
  delete forwards and therefore must authenticate exactly two warnings per
  case and eight in total; and
- adding, removing or altering files belonging only to future frame IDs leaves
  every earlier result hash unchanged.

Failure is `discard_n0a_contract_or_determinism_failure`.  No precision,
kernel, prompt or geometry rule may be changed and called the same protocol.

## No-GT geometry-capacity gates

These gates establish only that the shadow produced enough distinct valid
geometry to justify the next experiment.  They do not measure correctness or
AP:

1. valid `HS` lifts must be at least 80% of the 46,090 sources, i.e. at least
   36,872, and must cover at least 90 scenes;
2. at least 1,440 valid sources across at least 50 scenes must be
   non-trivially different from `H0`, defined before execution as
   `IoU3D(H0,HS) < 0.90` or maximum absolute displacement among the six
   corresponding q02/q98 faces at least `0.05 m`; and
3. every count and histogram is computed from all rows, including explicit
   abstentions; model-predicted IoU cannot remove a row from a denominator.

Passing yields `retain_n0a_for_n0b_and_one_sealed_capacity_evaluation_only`.
Failure yields `stop_n0a_insufficient_valid_or_distinct_geometry`.  A pass is
not evidence that AP improved and does not authorize a prediction suffix.

## Runtime gates and the 15 FPS requirement

Cold model load and the first three non-empty forwards per shard are reported
but excluded from warm distributions.  Timed N0a online work includes
RGB/depth decode, BGR-to-RGB conversion and resize, one image encoder, the
batched prompt decoder, mask transfer/selection and complete RGB-D lifting.
For conservative accounting, the core's in-memory canonical mask/depth/K/pose
and result hashes remain inside the measured lift interval.  Rehashing input
files, sidecar compression, JSON serialization and deterministic audit replays
are reported separately and excluded.

The frozen provider loads its authenticated assets and model lazily inside its
first non-empty `predict` call.  Separating that load from the first inference
would change the frozen execution interval, so the runner reports (a) provider
object initialization and (b) the combined lazy model/provider load plus first
forward as two explicit cold metrics; the latter is the measured outer provider
interval for call zero and is not claimed to be a load-only measurement.  It is
inside the all-forward diagnostic but outside every warm distribution.  Each
scene also reports pre-input rehash, intrinsic decode, end-input rehash, NPZ
compression/write and scene-JSON serialization/write milliseconds.  The shard
aggregates those fields and separately reports full-universe pre-authentication
and global end-rehash milliseconds.  Every value is finite and non-negative and
is explicitly marked excluded from online/warm distributions.  A shard
manifest cannot contain the duration of its own serialization/write without
self-reference, so that one duration is measured around the create-only write
and returned out of band by the process; it is non-authorizing and not part of
the sealed runtime gates.

On the same RTX 3090 class used by the frozen online experiments, all frozen
replay gates must pass:

- N0a incremental warm p95 `<=250 ms/non-empty keyframe`;
- replay-composed F0+N0a warm p95 `<=500 ms/keyframe`;
- replay-composed warm maximum `<833.33 ms/keyframe`;
- replay-composed warm mean divided by 25 `<=20 ms/raw frame`;
- zero warm gap-25 deadline misses; and
- total CUDA peak allocation `<=4 GiB`.

Replay composition is only an engineering envelope.  It is not proof of
integrated realtime performance.  Before any N0-derived active output, a
separately frozen same-GPU live run must include native CuTR, BoxFusion and
the complete N0 branch for at least 6,000 consecutive raw frames and show:

- throughput at least `15.0 FPS`;
- bounded asynchronous queue depth at most one;
- no sustained backlog or future-frame substitution;
- every result becomes visible only after its computation completes; and
- complete latency and CUDA memory remain within the device budget.

An accuracy pass without this live pass remains shadow-only.

Failure of any frozen replay runtime gate yields
`stop_n0a_realtime_gate_failed`.  This is distinct from
`stop_n0a_insufficient_valid_or_distinct_geometry` and from the integrity or
determinism decision `discard_n0a_contract_or_determinism_failure`.

## Admission to N0b

N0b is allowed only if every N0a integrity, determinism, geometry-capacity and
runtime gate passes.  N0b requires a new protocol frozen before execution and
must remain output-inert.  It may associate current-frame `H0/HS` evidence
against only a bounded past snapshot, but it may not silently enable the
currently unbounded/dynamic-object-incompatible SAM2 video state.

The intended minimum N0b contract is geometry-only mutual-best association
over at most the previous three successful frames, at most 16 sources per
frame, TTL three and at most 8,192 stored points per track.  A fixed
visibility/free-space diagnostic may use a 16x16 cell-centre grid over the
projected box hull: `0.05 m` is the depth-support tolerance and `0.10 m` the
free-space contradiction tolerance.  At least two causal views, each with at
least 64 valid probes, support at least 0.60 and contradiction at most 0.15,
are required for a fused shadow.  These values may be changed only by freezing
a differently named protocol before N0b sees any GT or evaluator output.

N0b still cannot create a birth.  Its first task is to demonstrate a bounded,
causal, GT-free selector or fused hypothesis and the same live 15 FPS target.

## Admission to a GT-assisted AP/capacity stage

The extra100 N0a merge must first be complete and sealed without GT.  The
N0a code, source manifest, protocol, thresholds and result schemas then become
immutable.  The same code may next produce a separately sealed paper100
shadow.  Because aggregate paper100 F1--F6 outcomes were already known when
N0a was designed, paper100 is a development comparison, not an untouched
confirmation set.

Only a new read-only evaluator, frozen after both no-GT receipts and before it
opens GT, may compare source-grouped hypotheses.  It must treat one F0 source
as one graph node even when `H0/HL/HLG/HB/HS` all cross a threshold.  The
primary comparison is the sealed F4 group
`G4={H0,HL,HLG,HB}` against `GN0={H0,HL,HLG,HB,HS}`, always unioned with the
unchanged native prefix.  GT-selected suffixes are threshold-specific,
nondeployable oracle diagnostics and cannot be reported as actual N0a AP.

At strict IoU thresholds `>0.15`, `>0.25` and `>0.50`, progression requires:

1. `GN0` adds at least `ceil(0.10*N_gt)` maximum native-union matches and its
   constructive oracle suffix improves native AP by at least 10.0 points at
   every threshold;
2. at IoU 0.50, `GN0` adds at least `ceil(0.15*N_gt)` native-union matches and
   its oracle AP delta is at least 15.0 points, providing margin for a later
   GT-free selector; and
3. `GN0-G4` contributes at least 15 additional grouped native-union matches
   at IoU 0.50, so the margin is attributable to `HS` rather than merely the
   already known F4 pool.

For paper100, where `N_gt=1433`, the corresponding 10% and 15% match gates are
144 and 215 respectively.  Passing authorizes only the preregistered N0b
shadow.  It does not authorize active birth.

If the development comparison passes, the unchanged protocol must be tested
once on a scene list frozen by physical ScanNet scene prefix before opening
its GT, excluding all implementation-smoke prefixes.  No threshold, prompt,
mask choice, geometry or selector may change between the development and
held-out runs.  Failure at any threshold discards the N0 route for the
requested +10-point claim; it cannot be tuned and rerun on the same cohort.

Because N0a is shadow-only, its actual prediction list and actual AP always
remain exactly those of the unchanged native baseline.

# S3R H10 integrated-runtime contract v2

Date: 2026-08-24

Status: **implementation-frozen, pre-runtime, no-GT timing-only candidate.**
After an independent reviewer records a control-only GO for the runner, test,
and this opaque contract hash, these exact bytes may authorize one fresh
create-only `control` timing run. They do not pre-authorize `integrated`.
The integrated arm remains forbidden until the control exits zero, its
create-only receipt is durably published and externally hashed, and a second
independent review records an integrated-only GO for those exact bytes. No
review or receipt in this contract authorizes ScanNet annotations or GT, an
oracle, AP, birth, native prediction export or mutation, C87, or full100.

## Quarantined v1 attempt and v2 repair

The v1 control invocation on 2026-08-24 used contract SHA-256
`060fb12f23be63dcfe5ec8c91d7af0d755dd7d516fc92a6e5d3d67da4562f44c`,
runner SHA-256
`0ef05e13f6ed730e6661ad0ac0b11a779255696eecfc388e0492bf0006e03753`,
and test SHA-256
`d9b567a5da331a56392469c0383f1d3c4dafe8c4bf3047e74422836d66d5d7bd`.
It exited nonzero with code 2 and the single content-free terminal line
`ERROR: integrated runtime failed closed`. Neither v1 formal output exists;
no receipt was published or externally accepted, and the integrated arm did
not run. The v1 output names remain quarantined and can never be supplied to
this v2 contract.

Independent failure review and no-model probes confirmed two coupled defects.
The main factory thread had a current PyCUDA primary context, while the new
`native-demo-*` thread had none; explicitly pushing the same primary context
in that thread made it current. Separately, a demo failure before its first
dataset yield produced no local completion, leaving the first FRAME to the
outer 120-second timeout. V2 scopes a primary-context push/finally-pop around
the complete per-scene demo-thread body, publishes one bounded failure
completion even before the first yield, and uses one monotonic 110-second
local submit deadline rather than separate input/completion budgets. These
repairs change only runtime correctness and failure propagation. They do not
authorize GT, AP, birth, prediction export, or an accuracy claim.

## Isolated question and non-claim

This experiment asks only whether the frozen online T05 pipeline and the
fresh frozen OWLv2/Boxer/K8/S3R side branch meet the preregistered real-time
limits while resident on the same physical GPU. It is a runtime experiment,
not an accuracy experiment.

Both arms use a deliberately conservative full-stream extension. They read
and preprocess all 19,370 real manifest-bound native frames and schedule 780
gap-25 keyframe slots. The value 780 is a scheduler-slot count, not a
separately instrumented proof that 780 model forwards completed. No synthetic
frame read is allowed in either formal arm. This is not byte-for-byte or
terminal-for-terminal equivalent to the released demo's early terminal
behavior. Therefore every formal receipt must state:

```text
runtime_only=true
full_stream_extension=true
original_terminal_exact=false
upstream_early_terminal_byte_equivalent=false
native_fps_protocol_equivalent=false
```

Neither receipt may be used to compute AP, replace a formal T05 prediction,
or claim that the original early-terminal program itself ran at the measured
FPS.

## Exact frozen implementation

| file | SHA-256 |
|---|---|
| `tools/benchmark_scannet_s3r_h10_integrated_runtime.py` | `52ac883176d592d5b131e2de6a1493759bc86d1659ae1c42b588e7320e6655d1` |
| `tests/test_benchmark_scannet_s3r_h10_integrated_runtime.py` | `0f743cc2da11e15d3480e29b10b354f40b2601f9e970b0fda509df683f1c8b20` |

The focused suite passed 134/134 independently in both frozen Python
environments under a clean `env -i` environment with `HOME=/tmp`, with
third-party pytest plugin auto-loading disabled in both runs:

```bash
cd /data/ZhaoX/BoxFusion
/usr/bin/env -i HOME=/tmp USER=admin1 LOGNAME=admin1 TMPDIR=/tmp \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PATH=/usr/local/cuda-12.1/bin:/usr/bin:/bin \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONHOME= PYTHONPATH= \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/dev/null \
  LD_LIBRARY_PATH= LD_PRELOAD= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/admin1/miniconda3/envs/boxfusion-online/bin/python -m pytest -q \
  tests/test_benchmark_scannet_s3r_h10_integrated_runtime.py
/usr/bin/env -i HOME=/tmp USER=admin1 LOGNAME=admin1 TMPDIR=/tmp \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PATH=/usr/local/cuda-12.1/bin:/usr/bin:/bin \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONHOME= PYTHONPATH= \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/dev/null \
  LD_LIBRARY_PATH= LD_PRELOAD= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/admin1/miniconda3/envs/ovm3d-1/bin/python -m pytest -q \
  tests/test_benchmark_scannet_s3r_h10_integrated_runtime.py
```

Both interpreters also independently passed `python -m py_compile` for the
frozen runner and focused test in separate clean syntax-check environments
whose cache prefixes were isolated under `/tmp`; these were not either formal
`/dev/null`-prefix invocation. The suite includes a real no-GPU heterogeneous-spawn smoke:
one boxfusion-online parent sequentially starts a native child under Python
3.10.13 and a provider child under Python 3.10.19, both children really import
their own NumPy and Torch, and they exchange bounded multiprocessing Queue
commands and acknowledgements. It also covers the real provider-only
`sys.path` transformation, path drift at every worker lifecycle boundary,
concurrent spawn configuration, and partial-start cleanup. This is
implementation evidence, not a formal runtime result.

The six focused v2 regressions additionally cover normal and exceptional
primary-context push/pop, a push that still yields no current context,
pre-yield failure wake-up, context scope around a pre-yield engine failure,
and the single end-to-end local submit deadline. Both complete 134-test suites
passed after the final v2 schema and output-path change.

The suite also validates exact environment scrubbing, absolute/hash-pinned
external commands, import-shadow rejection, the provider ignored-file
allowlist, and final-guard ordering. It constructs a valid adjacent
timestamp-based `.pyc` whose value differs from its source: an ordinary
subprocess loads the cached value, whereas a subprocess with
`PYTHONPYCACHEPREFIX=/dev/null` loads the source and leaves that adjacent
cache unchanged. Parent validation, both child-runtime probes, spawn entry,
post-factory identity, READY, and every worker lifecycle boundary require the
environment value and `sys.pycache_prefix` to remain exactly `/dev/null`.
This is evidence about standard CPython source-cache lookup, not a blanket
claim about every generated artifact.

The public formal API has no model, reader, or tracker factory-injection
parameter. Test injection remains private. The CLI requires the externally
declared hashes of this runner, its focused test, and this contract before it
may inspect formal inputs or spawn a worker. This contract is opaque to the
runner; its externally computed SHA-256 is supplied on the command line and
rechecked after the complete stream. This avoids a self-hash cycle.

## Frozen full-stream input

The only native stream manifest is
`docs/data/S3R_H10_NATIVE_FULL_STREAM_V1.json`, SHA-256
`449cf3d8e7e765fa53b46226b9c2a342a04dfd246d0baa6ee1ff1116cba0cf5d`.

| implementation | SHA-256 |
|---|---|
| `tools/build_scannet_s3r_h10_native_full_stream.py` | `0d4a629a8c59c12a0bf150fcbf624e25c6c075ef31def783462988745eff8f9f` |
| `tests/test_build_scannet_s3r_h10_native_full_stream.py` | `6318024644d393cfdbc1747c5ad5aa75052e409015f9b2347ae9c02b0562c116` |

The manifest freezes:

- ten scenes in the exact provider-schedule order;
- 19,370 JPG/depth/pose current frames;
- 19,333 finite raw poses and 37 positive/negative-infinity poses resolved
  only to the most recent strictly past finite native pose;
- native input identity
  `a1237771d9e71ebd37551c2a35ea77c93f4da4c684166f689fda5f5efd873037`;
- provider-subset identity
  `74a0a0c79dd60ff4d0d690040d48a8327479c909dbd8be74e30ae99666c50f55`;
- role-mount identity
  `e7cb2c2f313c19e3241ca9732aa6e3513c280b79ce3fe27973e1fb88a01d1f7b`;
- one scene-level `intrinsic_color` and one `intrinsic_depth` identity per
  scene;
- exact current color, depth, raw pose, and effective-past-pose paths and
  hashes for every frame.

The exact provider schedule is
`docs/data/S3R_H10_EXACT_SCHEDULE_V2.json`, SHA-256
`1ce565a65510b80d69a0402fe7a40ea89920625f6a81147d42f9232f7a7761e9`.
Within the 19,370-frame native stream it fixes exactly:

| provider status | count | permitted provider work |
|---|---:|---|
| member | 769 | current-frame read, OWLv2 inference, conditional Boxer inference when accepted 2D candidates exist, K8, tracker query/commit, CUDA sync |
| causal-pose abstention (`scene0412_00/2325`) | 1 | `skip_current` and timing ACK only |
| outside schedule | 18,600 | `skip_current` and timing ACK only |

The abstention and all outside frames must perform zero provider input reads,
zero provider inference, zero tracker snapshot/query/commit, and zero provider
CUDA synchronization or per-frame CUDA-memory query. Their ACK identity and
work fields remain null/zero. Only a provider-member `FRAME` command may run
provider inference or tracker query/commit. `START_SCENE` may construct a
fresh CPU tracker and reset the frozen provider's scene seed; `END_SCENE` may
take one read-only tracker snapshot solely to validate pending/audit closure.
Provider `END_SCENE` then closes the reader, synchronizes provider CUDA, and
takes its memory sample; native `END_SCENE` completes native finalization,
closes the reader, synchronizes native CUDA, and then takes its memory sample.
These lifecycle operations are not attributed to an outside/abstention frame
and produce no proposal rows.

The full stream intentionally creates ten additional tail-region native
gap-25 scheduler slots relative to the 770-frame raw provider schedule. They
are native scheduled work but provider-outside no-ops. The exact global
closure is 780 native scheduled slots, 769 provider-member operations, one
provider abstention, and 18,600 provider-outside frames.

## Current-only and past-only data access

Neither real worker may instantiate `ScannetDataset`, call its directory glob,
or preload all poses. Each scene opens and holds the four manifest-bound role
directory descriptors. The native reader processes one requested JPG, depth,
and raw pose at a time and obtains an infinite-pose fallback only from its
already cached most-recent finite pose. The provider reader opens current
frame bytes only for a provider member.

The coordinator byte-hashes every manifest-named JPG, depth, pose, and
intrinsic input before and after the stream. In separate pre/post validation
it also numerically parses the pose and intrinsic text as 4-by-4 matrices,
checks shape and NaN/Inf/finiteness constraints, and checks every raw pose
against its manifest-declared finite/non-finite class. It does not decode JPG
or depth payloads, run model inference on these inputs, retain future numeric
matrices for online use, or give future numeric contents to either online
worker. `coordinator_preflight_opaque_input_hashing=true` describes only the
independent byte-hash ledger; it does not claim that all coordinator preflight
validation is opaque. The other precise claims are
`online_worker_future_frame_semantic_access=false` and
`online_worker_prefetch=false`; the contract does not make the broader and
false claim that no process ever hashes or validates a future file.

Every native-frame acknowledgement and every provider-member acknowledgement
is bound to hashes observed from bytes read via the held directory
descriptors:

```text
color_sha256_observed
depth_sha256_observed
pose_sha256_observed
effective_pose_sha256_observed
input_identity_sha256
```

The coordinator compares those values with the exact manifest record. It
does not accept a worker that merely echoes a different current identity.
Provider outside-schedule and causal-pose-abstention acknowledgements perform
no input read and must keep all five identity fields null, as specified
above. Frame IDs, sequence numbers, scene/global counts, and every numeric
protocol field reject booleans as integers.

## Frozen native arm

The native worker is the frozen score-0.5 Reliable-View Top-K T05 pipeline.
Its configuration is `config/scannet_topk_fusion_score05.yaml`, SHA-256
`596b42b22828360aa780a95f188244fcef4ef69d4ee0096a37c7b8094daebe4c`.
It preserves `score_thresh=0.5`, ScanNet, gap 25, and enabled reliable views.

| native asset | SHA-256 |
|---|---|
| CuTR checkpoint | `856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217` |
| OpenCLIP checkpoint | `9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4` |
| class features | `49ab2384fbc01406eb7eb24ce89403bbfa9516bc213e11e8cd2014fa8eeea197` |
| class names | `0d628e3140d491acfce107268fe51233e1df44f84581f582fe253842fc6557c9` |
| PST image | `867f0546addc35a5000a421e9f81af4577470751b7a8ffc28e859cca97376660` |
| `demo.py` | `b691eee823737fc34e22d4f4a51c8b359bdc0537909cfe2c1112e3570189216a` |
| `boxfusion/capture_stream.py` | `a2bfadbe1ac1ec6bf54eca9c7fd01ee67c611b0b8d52966d874ae82c9274b25a` |

The runner pins and re-hashes an explicitly enumerated and reviewed local
application/bridge source ledger:
the `boxfusion` and `tools` package initializers; instances, boxes, fusion,
manager, reliable views, Cubify, preprocessing, measurement, orientation,
sensor, `tools/utils.py`, ViT, batching, positional encoding, transforms,
image-list and color helpers; and the proposal-cache, Graw/Gclean/PUF,
observer, SMOV, Group3D-lite, and PUF-lite modules imported by `demo.py`.
Their hashes are hard-coded in the frozen runner and form part of each
receipt's static-asset identity. The child forces PST to the absolute,
hash-bound path rather than a configuration-relative working-directory path.

For every enumerated `.py` source, the runner also requires the enumerated
adjacent legacy `.pyc`, native-extension suffix, and applicable same-stem
package/module shadow paths to be absent through non-symlink parent
descriptors. The frozen inventory checks are: 65 base ledger entries and 294
base shadow candidates; 72 formal static entries and 304 formal shadow
candidates after the additional formal/self/system assets are included. The
same-stem package-directory subset is 46 candidates in the base inventory and
48 candidates in the formal inventory.

These numbers are reproducibility checks, not a claim of complete import
closure. Unknown top-level shadows outside these enumerated candidate paths,
third-party site-packages/shared libraries, generated code, and caches are not
comprehensively byte-ledgered.

The native child reconstructs the released per-frame RGB/depth/intrinsic/
orientation/pose datum directly. It supplies an in-memory configuration with
`output_dir=None`, evaluation and rerun disabled, proposal cache disabled,
and Group3D diagnostics disabled. Save, diagnostic, rerun, and native data
loader surfaces are guarded. Native predictions may exist transiently inside
the native algorithm because they are needed for its online state, but the
coordinator cannot access them and no prediction is serialized, deserialized,
or written.

All 19,370 current frames are real manifest-bound reads and are preprocessed.
The fixed scheduler selects 780 gap-25 keyframe slots; this field is not
presented as an independently instrumented forward counter. Final per-scene
computation and CUDA synchronization are completed before the scene ACK and
are included in the stream clock.

## Frozen provider and tracker arm

The provider is the exact frozen OWLv2/Boxer implementation bound by
`docs/S3R_H10_FRESH_BOXER_PROVIDER_CONTRACT.md`, SHA-256
`11cc5ab398809ccfab9fafdcc9645e796321eb2db527e78ef2515e99946883d0`.

| implementation | SHA-256 |
|---|---|
| fresh provider runner | `72e42f3a3865ee9f52687d2a5a5a40ecabe189864c4d7d2cce18daf6be056403` |
| fresh provider test | `89595cf544e60efdde5637f7315f42ce8d59b3a0088d50d7913c3a442d000a6e` |
| provider core | `c70e114dabe1ef1081967027e4b5a15955ac16bab745652984dfe981100f21dd` |
| provider-core test | `40f75dac98e5774e9b1637a7c51c4ab5676df38a074e3c3b97a0d3a40a305ce2` |
| S3R tracker | `277316c36b7a7fcb8005a24e907e0f232e41f6b5874411293eb26b0744df9628` |
| S3R tracker test | `f08fd59ee2888c936e5b783de668fd789ba6b676bc4864e001b000ea287b1e3c` |

The external Boxer checkout must remain at commit
`1f86542dc342a4b1d474c87c97c5d1d6566d9148`. Its tree, directory inode, and
`git status --porcelain=v1 --untracked-files=all` are checked at the frozen
barriers; this requires no tracked change and no ordinary non-ignored
untracked file. Ignored paths are audited separately under a category policy.
The runner hard-codes the two checkpoint paths and hashes: Boxer must be a
regular file, while DINO must be the exact frozen symlink to its hash-bound
regular target
`/data/ZhaoX/OVM3D-Dett/third_party/boxer/ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`.
Every other permitted ignored entry must be a regular
`**/__pycache__/*.pyc` file. At review time the observed ledger has 21 entries:
19 such pyc files and the two checkpoints. The runner does not hard-code the
19/21 counts or individual pyc paths. Adding, removing, or changing an
otherwise allowed pyc changes the ignored-file ledger identity, which must
remain identical across the pre-probe guard, pre/post barriers, and the
control/integrated binding.

An ignored entry outside those categories, an ignored native-extension or
`.so`, a non-regular entry other than the exact DINO symlink, a symlinked
parent component, or checkpoint path/hash/type/target drift is rejected. The
allowed adjacent pyc files may exist, but the exact
`PYTHONPYCACHEPREFIX=/dev/null` environment and `sys.pycache_prefix` redirect
standard CPython source-cache lookup away from them.

The fresh-provider asset ledger is supplemented by an explicitly enumerated
and reviewed Boxer source ledger, including `owlv2_model.py`,
`dinov3_wrapper.py`, `demo_utils.py`, `gravity.py`, the OBB/tensor utility
wrappers, and their package initializers. This is not a complete byte ledger
of all transitive third-party code. All listed frozen model assets are also
verified, including:

| provider asset | SHA-256 |
|---|---|
| OWLv2 checkpoint | `14aa78ffe7b13e5b3ebf55845bc9a07e339a095cfd88f4c4e8f726b38ce1ebbf` |
| frozen OWLv2 text cache | `59193fc014d381b2200edf1c1e6dc86324edb55a067189d3e84226a184185283` |
| BoxerNet checkpoint | `d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f` |
| DINOv3 checkpoint | `4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea` |

The provider uses the frozen 1,220-prompt LVIS+ profile, 960-pixel input,
OWL threshold 0.25, Boxer 3D threshold 0.50, bfloat16, and fixed scene seed.
For every member it runs fresh OWLv2. It runs Boxer only when OWLv2 leaves at
least one accepted 2D candidate. It then orders any returned 3D rows by
`(-score, source_row)`, takes at most eight, converts them to numeric corners,
and performs one past-only S3R `query` followed by the exact matching
`commit`. The tracker is CPU-only, performs no GPU execution, and allocates
zero tracker GPU bytes. The 769 member operations/commits are not a claim that
769 Boxer forwards occur.

The live integrated run must close exactly on the already sealed no-GT source
totals: 6,338 fresh raw rows, 4,557 K8 rows, and 769 tracker commits. It also
checks the frozen per-scene raw/K8/commit counts. These counts do not expose
geometry or semantics.

The opaque reference source is bound before and after without NPZ
deserialization:

| source identity | SHA-256 |
|---|---|
| `S3R_H10_RAW_BOXER_SOURCE.json` | `ca65214f3e6327cea66ec8cb700ab3501572be9325af4366beaffa2b7cc2859e` |
| `S3R_H10_RAW_BOXER_SOURCE.npz` | `fdb688cc1372985f2ffaf3d257ed470cd4de28ff42f7a2d04a5f72311a1225f2` |
| numeric array content | `a5efdb8d0d2c7b95f63368a3249229659a1052c400539321ce461da32732b862` |
| K8 membership | `a2a94b11461e8c1bdd15d6a4ad99d058f42db6fd73690c69269ff1b89deb6391` |

No per-proposal raw row, K8 geometry, S3R assignment row/receipt payload,
semantic label/class/prompt text, or embedding is sent to the coordinator or
written by this timing harness. Counts, bounded aggregate audit state,
timing, resource use, input identities, and the one aggregate timing receipt
are the permitted protocol outputs.

## Heterogeneous same-GPU execution

The formal parent is exactly
`/home/admin1/miniconda3/envs/boxfusion-online/bin/python`, Python 3.10.13,
NumPy 1.26.4. The runner verifies this identity before formal input reads or
worker spawn and requires that Torch is absent from `sys.modules`; the parent
must not import or initialize Torch/CUDA. It launches children with the
multiprocessing `spawn` start method and lazy child imports.

| role | interpreter | Python | NumPy | Torch / CUDA build | binary SHA-256 |
|---|---|---:|---:|---|---|
| native | `/home/admin1/miniconda3/envs/boxfusion-online/bin/python` | 3.10.13 | 1.26.4 | 2.6.0+cu124 / 12.4 | `0b713d4abbdf074ab38362c1542060b0e9841695d759df37d706baf1decf9a8b` |
| provider | `/home/admin1/miniconda3/envs/ovm3d-1/bin/python` | 3.10.19 | 1.26.4 | 2.2.0+cu121 / 12.1 | `8d53381a3c7b869a331da9112ea494d0c1f90c17b69ccee9b1f6d4ef12273e5f` |

The native NumPy/Torch origins are exactly:

```text
/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10/site-packages/numpy/__init__.py
/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10/site-packages/torch/__init__.py
```

The provider NumPy/Torch origins are exactly:

```text
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/numpy/__init__.py
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/torch/__init__.py
```

The native spawn-entry `sys.path` is the following complete ordered list:

```text
/data/ZhaoX/BoxFusion
/home/admin1/miniconda3/envs/boxfusion-online/lib/python310.zip
/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10
/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10/lib-dynload
/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10/site-packages
/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10/site-packages/rerun_sdk
```

Its post-factory list is exactly the same complete list. Both digests are
`f2b923d772901d38f9b84790464baafaaaf660d91b2df00148c75f0f15d92b3b`.

Importing the frozen Cubify module has two known, transient process-global
side effects in this native environment. In order, the imports append

```text
/home/admin1/miniconda3/envs/boxfusion-online/lib/python3.10/site-packages/setuptools/_vendor
<torch.distributed.nn.jit.instantiator.INSTANTIATED_TEMPLATE_DIR_PATH>
```

to the otherwise exact native spawn-entry list. The second entry is the
absolute temporary directory created by Torch's remote-module instantiator
for that child, so its concrete name is intentionally not frozen across
processes. The runner hash-pins and rechecks the source files that authorize
these two side effects:

| native import-side-effect source | SHA-256 |
|---|---|
| `setuptools/__init__.py` | `eddb9a7016889b1ceb51ad0e821233f25560689d5f230efeb8bdafd7abd8fd21` |
| `torch/distributed/nn/jit/instantiator.py` | `440a619c764e4133564d7956ba060a7223e94664854b94a4a2074d095756db7e` |

Immediately after the Cubify import returns, and before the runner's next
native factory import or any checkpoint/model construction, the native-only
restore helper validates the complete observed list, the two loaded-module
origins, and equality between the temporary tail entry and the
instantiator's live `INSTANTIATED_TEMPLATE_DIR_PATH`. It accepts exactly the
frozen native list followed by those two entries, each once and in that
order. Both `__file__` and `__spec__.origin` must identify the exact frozen
source for each authorizing module.

The instantiator directory is also treated as an untrusted generated-code
surface, not accepted merely because its name begins with `/tmp/tmp`. Its path
must be canonical and absolute, its direct parent must be exactly `/tmp`, and
an `lstat` must show a non-symlink directory owned by the child's effective
UID and effective GID with mode `0700`. Its complete directory listing must contain only
`_remote_module_non_scriptable.py`. That entry must itself be an owned,
regular, non-symlink file; its size must be exactly 2,355 bytes and its
SHA-256 must be
`8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`.
The already-loaded `_remote_module_non_scriptable` module must be present
under that exact `sys.modules` key, and both its `__file__` and
`__spec__.origin` must equal that exact generated file. Its spec name must be
exactly `_remote_module_non_scriptable`, and no other loaded module's file,
spec origin, or package path may point into the temporary directory.

Validation is atomic: any missing, reordered, duplicated, substituted, or
additional path entry, missing module, origin drift, noncanonical or symlinked
temporary path, owner/mode drift, extra generated entry, or generated-module
identity/content drift raises before `sys.path` is changed. Only after every
check passes does the helper restore the original list in place and invalidate
import caches. An exception during restoration or cache invalidation restores
the complete observed pre-call list before it propagates. Thus this
narrowly documented import cleanup does not widen
`NATIVE_POST_FACTORY_SYS_PATH`; all subsequent post-factory and lifecycle
checks still require the exact frozen digest above.

The provider spawn-entry `sys.path` is the following complete ordered list:

```text
/data/ZhaoX/BoxFusion
/home/admin1/miniconda3/envs/ovm3d-1/lib/python310.zip
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/lib-dynload
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages
/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src
__editable__.detectron2-0.6.finder.__path_hook__
__editable__.mast3r-1.0.0.finder.__path_hook__
__editable__.unidepth-0.1.finder.__path_hook__
/data/ZhaoX/OVM3D-Dett/Fast-SAM3D
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/black-26.3.1-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/omegaconf-2.3.0-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/iopath-0.1.9-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/tensorboard-2.20.0-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/cloudpickle-3.1.2-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/pytokens-0.4.1-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/pathspec-1.0.4-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/mypy_extensions-1.1.0-py3.10.egg
```

Its spawn-entry digest is
`ddbe40f516d32818e41e45efa4095c401b4ab8ba7369f25520affc5fdfbf9531`.
The provider post-factory `sys.path` is the following complete ordered list:

```text
/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer
/data/ZhaoX/BoxFusion
/home/admin1/miniconda3/envs/ovm3d-1/lib/python310.zip
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/lib-dynload
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages
/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src
__editable__.detectron2-0.6.finder.__path_hook__
__editable__.mast3r-1.0.0.finder.__path_hook__
__editable__.unidepth-0.1.finder.__path_hook__
/data/ZhaoX/OVM3D-Dett/Fast-SAM3D
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/black-26.3.1-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/omegaconf-2.3.0-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/iopath-0.1.9-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/tensorboard-2.20.0-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/cloudpickle-3.1.2-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/pytokens-0.4.1-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/pathspec-1.0.4-py3.10.egg
/home/admin1/miniconda3/envs/ovm3d-1/lib/python3.10/site-packages/mypy_extensions-1.1.0-py3.10.egg
```

Its post-factory digest is
`3d59f388c8bb4234e4924946c42b5341c5f9a824d3e2b7c372344a17e1e35ae2`.
The only persistent role-specific path transformation is the frozen
fresh-provider helper prepending the resolved Boxer root once. No duplicate
or other drift is allowed. The native Cubify cleanup described above is a
one-time restoration to its unchanged entry identity, not a distinct
post-factory identity. The spawn target itself captures the entry path before
factory construction, derives and checks the exact role-specific post-factory
path, and checks it after factory construction, READY, every
scene/frame/end-scene operation, STOP, and close. READY entry/post digests are
injected by the protocol wrapper rather than trusted from a model worker.

The canonical cross-arm child-runtime identity, which binds both interpreter
binaries, Python/NumPy/Torch/CUDA identities and origins, and both roles'
entry/post digests plus exact environment/interpreter pycache-prefix values,
is
`69f34717eae0d1cd8521e78d4d53c0bce74d12e272ea7af99f52dc85a5f955ac`.
It is probed before and after the stream and recorded in the cross-arm
bindings.

Changing the global multiprocessing executable and parent `sys.path` for a
heterogeneous child start occurs in one process-wide lock. Inside that lock,
the runner captures both parent values, switches to the role-specific values,
calls `Process.start()`, and restores both values in one `finally`. The outer
failure path performs only bounded partial-process and Queue cleanup and does
not touch either global setting.

Both real children must see the same single `CUDA_VISIBLE_DEVICES` token and
report the same physical GPU UUID, device name, total memory, and driver
version. The integrated arm rejects a difference between its two children.
It also rejects any difference in the native child's GPU/CVD/runtime identity
between the bound control receipt and the integrated run.

Each persistent child's engine/model factory is executed once before READY
and before the stream clock. This excluded factory work includes the frozen
OWL constructor dummy call. READY means `initialization_complete=true`; it
does **not** claim a full-pipeline warm-up. The truthful provider disclosure
is:

```text
owl_constructor_dummy_warmup=true
full_pipeline_warmup=false
first_real_forward_included=true
first_real_owl_call_included=true
first_real_owl_kernels_pre_warmed=true
first_real_boxer_forward_included=true
```

The legacy-named `first_real_owl_kernels_pre_warmed=true` field means only
that the constructor dummy traversed the OWL `vision_detector` core. It does
not cover the complete OWL call: input clone, interpolation, normalization,
dtype conversion, postprocessing, GPU-to-CPU transfer, box conversion,
filtering, and NMS are outside that dummy path and may retain cold first-use
costs. The first complete actual OWL call, including those stages, is timed.
If a Boxer invocation occurs, its first actual forward is also not excluded.
The `first_real_boxer_forward_included=true` field means that an actual first
Boxer invocation is not discarded; it does not assert that Boxer runs on the
first provider member or on all 769 members. Native has no constructor dummy
OWL warm-up and its first real scheduled model work is included. The first
provider member remains in provider p50/p95/max and cannot be discarded as a
cold outlier.

## Two-arm order and causal IPC

The two arms are separate fresh processes and separate create-only receipts.
They must run in this order:

1. `control`: one persistent native T05 model worker and no persistent or
   measured provider model worker, provider frame request/ACK, inference, or
   fabricated provider timing. Formal preflight and immutable-after checking
   still execute a short-lived CPU-only provider interpreter/Torch identity
   probe subprocess; it does not construct the provider model, acquire a
   provider frame, or contribute provider timing;
2. `integrated`: one fresh native T05 child plus one fresh provider child.
   Its CLI requires the exact control receipt path and externally computed
   SHA-256.

The integrated preflight verifies that the control receipt binds the same
contract, runner/test, full-stream manifest, schedule, static assets,
manifest-input identity, formal T05 opaque identity, provider assets, and raw
source identities. The integrated native child does not exist at that point.
After the integrated stream completes, but before immutable-after checks and
publication, the runner compares the control GPU UUID, CVD, device name,
total memory, driver, native interpreter, Python, NumPy, Torch, CUDA build,
and path identities with the integrated native child. A mismatch therefore
fails closed before a receipt or qualification is published, but this
cross-child comparison is not described as preflight. The control/integrated
FPS ratio is diagnostic only.

All command and response queues have hard `maxsize=1`. For every native frame
the integrated causal order is:

```text
current provider request
-> provider ACK for that exact current identity
-> current native request
-> native CUDA-synchronized ACK for that exact current identity
-> next-frame provider request
```

There is no online-worker future-frame request, prefetch, backlog, or more
than one outstanding command. The separate coordinator-only byte-hash and
numeric validation passes remain as disclosed above. Scene-end and STOP
ledgers are compared with exact per-scene and global counts. Provider
`END_SCENE` includes its read-only tracker snapshot and CUDA synchronization;
native `END_SCENE` includes native finalization and CUDA synchronization. The
stream clock boundary itself is defined precisely below.

The native thread-local handoff has one monotonic 110-second budget covering
both its one-slot input put and completion get. Time consumed by the put is
deducted from the get; the two waits cannot accumulate to 220 seconds. Queue
Full, Queue Empty, exhausted local budget, a missing current PyCUDA context,
and any pre-yield demo-thread exception all fail closed before the outer
120-second FRAME ACK deadline under the preregistered logic.

## Child terminal-output boundary

At the beginning of each multiprocessing spawn target, before the factory or
model constructor, file descriptors 1 and 2 are redirected to `/dev/null`
with `dup2`. That descriptor-level suppression remains in force through
READY, all frame/scene work, STOP, and worker close. It covers Python and
native-library writes during the model lifecycle, but deliberately does not
claim suppression of Python's multiprocessing bootstrap before the spawn
target begins. No child stdout/stderr content or character count is retained
or sent through IPC. Worker failures cross IPC only as a fixed,
content-free error code; no exception string, representation, traceback,
prediction-derived text, or model diagnostic reaches the coordinator
terminal or receipt. The receipt must state:

```text
model_lifecycle_fd1_fd2_redirected_to_devnull=true
suppression_begins_at_spawn_target_before_factory=true
spawn_bootstrap_stdio_suppression_not_claimed=true
stdio_content_retained=false
stdio_character_counts_retained=false
prediction_derived_text_reaches_coordinator_terminal=false
```

## Timing definitions and acceptance gates

Persistent child engine/model-factory construction, including the disclosed
OWL constructor dummy core call, is excluded. The stream clock begins at the
earliest worker timestamp taken
immediately before the first frame's current-reader operation. Although the
receipt calls this `first_current_read_ns`, it is the start of the worker
operation that calls the current reader, not an instrumented operating-system
read boundary. The prefix from the first coordinator request/Queue transport
to that worker-operation timestamp is excluded.

The first scene's `START_SCENE`/`SCENE_READY` exchange is also before the
clock. This includes intrinsic reads and numeric decoding, provider tracker
construction/reset, and native reader/dataset creation and thread launch.
The native ACK proves only that the reader/dataset exist and the background
thread has started; it does not prove that the demo iterator is ready.

That native background thread can subsequently initialize `BoxManager`,
`BoxFusion`, PyCUDA `SourceModule`, and other pre-iterator state before its
first native current-reader operation. In `control`, the clock begins only at
that first native operation, so this first-scene pre-iterator work is excluded.
In `integrated`, the provider reads first; scheduling can therefore place
part or all of the native pre-iterator work after the provider starts the
clock, where it may overlap provider work. The integrated absolute-FPS gate
can consequently be more conservative than the control timing, and their
ratio remains diagnostic only. For each of the remaining nine scenes, the
entire `START_SCENE` exchange and subsequent background initialization occur
after the global clock has started and are included.

Before this per-scene background body begins, v2 pushes the already-created
PyCUDA primary context in the `native-demo-*` thread. The push covers
`demo.run`, every iterator-driven synchronization, the full-count check, and
the final pending-frame synchronization; a `finally` pop runs on normal
return, body failure, and missing-current failure. Context construction and
model loading remain in the persistent factory before READY as described
above. This thread-local correction does not add a warm-up or exclude any
first real model work.

The clock ends at the timestamp immediately after the final native
`END_SCENE` engine CUDA synchronization. The subsequent final-native memory
sample, `SCENE_DONE` construction, Queue put/transport/coordinator receipt,
STOP, worker close, and process/Queue cleanup are excluded. Within those exact
endpoints, native decode and preprocessing, all 780 scheduled native gap-25
slots, provider member work, causal request/ACK work between frames, scene
transitions, provider scene-final snapshot/synchronization, native scene-final
computation, and actual first model calls are included. This contract claims
IPC inclusion only inside the stated boundary, not end-to-end inclusion of
the excluded prefix or suffix.

The primary acceptance gate is the conjunction below for the `integrated`
arm:

- absolute integrated native throughput is at least `10.0 FPS`;
- provider member p50, p95, and maximum are each at most `0.83333 s`;
- tracker p95 is at most `2,000,000 ns` and tracker maximum is at most
  `10,000,000 ns`;
- no queue backlog, cap violation, reported OOM failure, worker error,
  identity mismatch, or incomplete tracker audit occurs;
- all exact frame, call, row, K8, commit, scene, and close counts match.

The provider deadline sample is the worker-reported interval from the start
of its current-frame operation through current-frame held-descriptor reads,
fresh provider inference, CUDA synchronization, K8 preparation, tracker
query/commit, and the synchronized memory sample. It ends before response
construction, `response_queue.put`, Queue transport, and coordinator receipt,
so it is not called end-to-end member-ACK latency. Those coordinator/IPC
costs remain inside the causal full-stream clock and therefore affect the
global integrated FPS. Outside/abstention no-op timings are not misreported
as provider-call samples.

The control arm records its own absolute FPS, but the control/integrated ratio
is not an acceptance gate. Only the integrated absolute-FPS and provider/
tracker conjunction may set:

```text
integrated_realtime_qualified=true
integrated_provider_runtime_qualified=true
```

The control receipt must keep both values false. A below-threshold timing run
may publish a valid measurement receipt with the relevant gates false; that
receipt is a negative runtime result and cannot authorize GT/oracle access.
An OOM failure, cap, protocol, pin, causal, worker, or publication failure
cannot produce a valid or authorized receipt. Because publication links the
public inode before its final readback/directory-fsync checks, a nonzero exit
may leave a JSON artifact despite best-effort unlink. Any artifact from a
nonzero arm is quarantined and invalid: it must not be externally hashed or
accepted as a receipt, supplied to the integrated arm, or bypassed or retried
before independent review.

Resource fields are diagnostic and deliberately scoped. Each role reports
Torch caching-allocator allocated/reserved high-water marks, plus process RSS
high-water. The sum of the two roles' Torch allocator high-water marks is an
upper-sum diagnostic only; it is not total device VRAM and does not cover
PyCUDA allocations. At CUDA synchronization boundaries the runner also
samples device-wide used memory with `torch.cuda.mem_get_info()`. Those
samples include same-device non-Torch/PyCUDA use visible at that instant, but
their maximum is a boundary-sampled maximum, not a continuous absolute peak.
No numeric VRAM cap is preregistered, and qualification does not depend on a
fabricated total-VRAM peak. Successful qualification requires both models to
be simultaneously resident on the same GPU for the completed full stream and
`oom_failure_reported=false`/
`full_stream_completed_without_oom_failure=true`; this does not claim that
every internally recovered allocation event was continuously observed.
Tracker GPU execution and tracker GPU bytes remain false/zero.

## Immutable before/after barrier

Before any persistent model-worker spawn, and again after the entire stream,
the runner verifies the same immutable bytes and identities:

- this contract, runner, focused test, native manifest, schedule, manifest
  builder/test, both interpreter binaries, and the absolute/hash-pinned
  `/usr/bin/git` and `/usr/bin/nvidia-smi` binaries;
- both child runtime probes, including exact NumPy/Torch origins and the
  spawn-entry/post-factory `sys.path` identities bound by
  `69f34717eae0d1cd8521e78d4d53c0bce74d12e272ea7af99f52dc85a5f955ac`
  and the exact `/dev/null` pycache-prefix environment/interpreter values;
- all exact manifest-named JPG, depth, raw-pose, effective-past-pose, color
  intrinsic, and depth intrinsic inputs plus role-mount identities;
- all listed native model weights, data assets, configuration, and the
  explicitly enumerated/reviewed local application/bridge source and
  import-shadow ledgers used by the adapter and T05 algorithm;
- the fresh-provider contract, runner/test, core/test, external Boxer commit
  and explicitly enumerated/reviewed Boxer source and ignored-file ledgers,
  plus the listed OWLv2/text-cache/Boxer/DINO weights;
- the opaque raw-source JSON/NPZ reference;
- all ten formal T05 prediction files through the hashes frozen in the exact
  schedule.

The coordinator accesses the ten T05 files only to compute opaque identity
hashes. It never unpickles or otherwise deserializes them and has no semantic
or geometric access to their predictions. A manifest, input, mount, source,
weight, code, contract, test, interpreter, runtime-probe, or T05 change fails
closed. The final receipt records the static-asset, frame-input, T05,
provider-asset, and child-runtime identities and
`immutable_before_after_verified=true`.

In preflight, the runner first hash-pins its external commands and checks the
enumerated local import-shadow candidates. The provider checkout/ignored-file
guard then completes before the provider interpreter probe can import from
that checkout. A repeated checkout and formal static snapshot after the
preflight probes ensure that those probes did not change the guarded state
before model-worker execution.

The post-stream native and provider interpreter probes complete first. The
runner then performs its final local static/import-shadow snapshot followed
by its final provider checkout/ignored-file snapshot; only comparisons,
receipt assembly, and create-only publication follow those executable-file
guards. The before/after barriers are point-in-time checks, not continuous
filesystem monitoring. A trusted, exclusive, stable workspace with no
concurrent mutation is therefore a formal precondition.

These ledgers do not comprehensively bind unknown top-level import shadows
outside the enumerated candidates, every third-party site-package or shared
library, PyCUDA/NVCC/TorchInductor-generated code, or framework/model/Python
caches. `/tmp` caches may be read, written, or reused. None of these narrower
non-claims weakens the exact hashes and absence checks explicitly listed
above.

## Environment and exact invocations

Each authorized invocation starts with `/usr/bin/env -i` and explicitly
reconstructs its environment before Python starts. `PYTHONHOME` and
`PYTHONPATH` are supplied as present and empty. The runner's value validator
would treat either missing value like empty, but the exact command text below
does not omit them. The formal required values are:

```text
HOME=/tmp
USER=admin1
LOGNAME=admin1
TMPDIR=/tmp
LANG=C.UTF-8
LC_ALL=C.UTF-8
PATH=/usr/local/cuda-12.1/bin:/usr/bin:/bin
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
PYTHONHASHSEED=0
PYTHONHOME=
PYTHONPATH=
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
PYTHONPYCACHEPREFIX=/dev/null
LD_LIBRARY_PATH=
LD_PRELOAD=
CUDA_VISIBLE_DEVICES=<exactly one non-disabled token>
```

All inherited `GIT_*` variables must be absent, all unexpected `LD_*`
variables are rejected, and the receipt records
`git_environment_names_absent=true` together with the runner-required
`PATH`, empty loader variables, Python controls, thread controls, and CUDA
visibility. `PYTHONPYCACHEPREFIX` must be present and exactly `/dev/null`, and
the parent and children must also report `sys.pycache_prefix=/dev/null`.
`HOME`, `USER`, `LOGNAME`, `TMPDIR`, `LANG`, and `LC_ALL` are exact invocation
preconditions but are not receipt-bound fields.

Runner-owned Git and GPU-identity calls use the absolute, hash-pinned
`/usr/bin/git` and `/usr/bin/nvidia-smi` binaries and minimal non-inheriting
subprocess environments:

| runner-owned external executable | SHA-256 |
|---|---|
| `/usr/bin/git` | `c3edb15c9715b79fcfb1fa978256cdfc14a9ad72a4a8d5680a9fc5ebc6a57e0e` |
| `/usr/bin/nvidia-smi` | `4b45d6578bea1488ca04c91f0b9252a5bbfc20b9058870755e2b48a755f0644a` |

The runner-owned Git subprocess adds its own fixed, minimal Git controls; no
top-level `GIT_*` value is inherited. The legacy fresh-provider validator is
the one exception that invokes bare `git`; under the exact launch it resolves
through the frozen `PATH`, inherits no `GIT_*`, and uses `HOME=/tmp`.
Independent host review must establish that `/tmp/.gitconfig`,
`/tmp/.config/git/config`, and `/etc/gitconfig` are absent for the formal run.
That config absence and the `HOME` value are host/invocation preconditions,
not receipt-bound byte identities. Checkout-local `.git/config`,
`.git/info/exclude`, fsmonitor state, and the Git index are likewise not
individually byte-bound in the receipt, although their observable effects are
constrained by the frozen commit/tree/status and ignored-ledger checks. The
current host was reviewed
with ordinary checkout-local state; the trusted, exclusive, stable-workspace
precondition remains necessary. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` belongs
only to the focused test commands above and must not be added to either formal
command.

Let `CONTRACT_SHA256` be the externally computed SHA-256 of this exact file,
and let `CUDA_TOKEN` be the independently reviewed single non-disabled CUDA
visibility token. The exact same `CUDA_TOKEN` must be used for both arms.
The only authorized formal timing-receipt output paths are:

```text
/data/ZhaoX/BoxFusion/logs/scannet_s3r_h10_runtime_control_v2.json
/data/ZhaoX/BoxFusion/logs/scannet_s3r_h10_runtime_integrated_v2.json
```

The formal CLI rejects any other output path. The integrated arm also rejects
any control-receipt path other than the exact control path above. Neither
output may exist before its arm, and the integrated arm cannot run until the
control bytes have been durably published and externally hashed. Subject to
independent reviewer GO, the exact control invocation is:

```bash
/usr/bin/env -i HOME=/tmp USER=admin1 LOGNAME=admin1 TMPDIR=/tmp \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PATH=/usr/local/cuda-12.1/bin:/usr/bin:/bin \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONHOME= PYTHONPATH= \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/dev/null \
  LD_LIBRARY_PATH= LD_PRELOAD= CUDA_VISIBLE_DEVICES="${CUDA_TOKEN}" \
/home/admin1/miniconda3/envs/boxfusion-online/bin/python \
  /data/ZhaoX/BoxFusion/tools/benchmark_scannet_s3r_h10_integrated_runtime.py \
  --arm control \
  --output /data/ZhaoX/BoxFusion/logs/scannet_s3r_h10_runtime_control_v2.json \
  --runtime-contract /data/ZhaoX/BoxFusion/docs/S3R_H10_INTEGRATED_RUNTIME_CONTRACT.md \
  --expected-runtime-contract-sha256 "${CONTRACT_SHA256}" \
  --expected-runner-sha256 52ac883176d592d5b131e2de6a1493759bc86d1659ae1c42b588e7320e6655d1 \
  --expected-runner-test-sha256 0f743cc2da11e15d3480e29b10b354f40b2601f9e970b0fda509df683f1c8b20
```

After the control receipt is durably published, let
`CONTROL_RECEIPT_SHA256` be its externally computed exact SHA-256. The second
exact integrated invocation is:

```bash
/usr/bin/env -i HOME=/tmp USER=admin1 LOGNAME=admin1 TMPDIR=/tmp \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PATH=/usr/local/cuda-12.1/bin:/usr/bin:/bin \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONHOME= PYTHONPATH= \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/dev/null \
  LD_LIBRARY_PATH= LD_PRELOAD= CUDA_VISIBLE_DEVICES="${CUDA_TOKEN}" \
/home/admin1/miniconda3/envs/boxfusion-online/bin/python \
  /data/ZhaoX/BoxFusion/tools/benchmark_scannet_s3r_h10_integrated_runtime.py \
  --arm integrated \
  --output /data/ZhaoX/BoxFusion/logs/scannet_s3r_h10_runtime_integrated_v2.json \
  --runtime-contract /data/ZhaoX/BoxFusion/docs/S3R_H10_INTEGRATED_RUNTIME_CONTRACT.md \
  --expected-runtime-contract-sha256 "${CONTRACT_SHA256}" \
  --expected-runner-sha256 52ac883176d592d5b131e2de6a1493759bc86d1659ae1c42b588e7320e6655d1 \
  --expected-runner-test-sha256 0f743cc2da11e15d3480e29b10b354f40b2601f9e970b0fda509df683f1c8b20 \
  --control-receipt /data/ZhaoX/BoxFusion/logs/scannet_s3r_h10_runtime_control_v2.json \
  --expected-control-receipt-sha256 "${CONTROL_RECEIPT_SHA256}"
```

No alternative manifest, schedule, scene root, model path, interpreter,
threshold, deadline, queue size, output mode, or factory is exposed by the
formal CLI.

## Create-only timing receipt

After its complete stream, post-stream probes, and final immutable guards
succeed, each arm attempts to publish one canonical JSON timing receipt. Only
a zero-exit arm with successful publication has a valid receipt. The output
parent is opened and inode-bound component by component. Publication uses an
anonymous `O_TMPFILE`, file and directory `fsync`, exact byte readback, and a no-replace
link relative to the held parent descriptor. An existing output, symlink,
parent-name swap, output/input overlap, zero/failed write, incomplete final
length, readback difference, or fsync failure fails closed. A positive partial
write is retried until the exact canonical payload length is reached. Because
the no-replace link precedes some final publication checks, a later failure can
leave an invalid artifact; the nonzero-exit quarantine rule above applies.

Workers do not publish formal timing receipts, predictions, diagnostics,
proposal rows, or S3R assignment artifacts. This harness does not provide a
global write sandbox and does not claim that framework, compiler, model,
TorchInductor, or Python cache writes are absent. `/tmp` cache reuse and such
framework writes are not formal outputs.

The receipt contains timing arrays, causal request/ACK timestamps, counts,
scoped resource diagnostics, runtime identities, gates, and opaque hashes. It
also necessarily contains administrative strings such as schema/mode/arm,
scene identifiers, status, GPU/environment/path, and version strings. It
contains no box, corner, quaternion, proposal row, assignment geometry,
semantic object/class/label/prompt text or text embedding, native prediction,
or GT/annotation/oracle/AP data or result.

## Mandatory stopping rule

Both receipts must preserve at least:

```text
timing_only=true
runtime_only=true
formal_h10=true
synthetic_worker_injection=false
full_stream_extension=true
original_terminal_exact=false
upstream_early_terminal_byte_equivalent=false
native_fps_protocol_equivalent=false
opaque_t05_identity_hashing=true
coordinator_native_prediction_semantic_access=false
native_prediction_deserialization=false
native_prediction_geometry_access=false
native_prediction_serialized=false
native_prediction_mutation=false
native_prediction_write=false
coordinator_preflight_opaque_input_hashing=true
online_worker_prefetch=false
online_worker_future_frame_semantic_access=false
gt_access=false
annotation_access=false
evaluation=false
ap_computation=false
birth=false
labels_serialized=false
geometry_serialized=false
full100_not_authorized=true
h10_gt_oracle_authorized=false
gt_access_authorized=false
immutable_before_after_verified=true
environment.git_environment_names_absent=true
```

Even `integrated_realtime_qualified=true` authorizes no GT, oracle, AP,
birth, C87, full100, or active modification. It satisfies only the independent
runtime prerequisite. Promotion to any H10 matching-only oracle requires a
separate create-only runtime-review receipt and a new explicitly frozen
contract. A failed or unreviewed integrated runtime leaves every later H10 GT
stage forbidden.

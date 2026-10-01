#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
ROOT=/data/ZhaoX/BoxFusion
BOXER_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer
BOXER_COMMIT=1f86542dc342a4b1d474c87c97c5d1d6566d9148
OWL_CKPT=/data/ZhaoX/OVM3D-Dett/third_party/boxer/ckpts/owlv2-base-patch16-ensemble.pt
OWL_TEXT_CACHE=/data/ZhaoX/OVM3D-Dett/third_party/boxer/ckpts/owlv2-base-patch16-ensemble_textemb_878186d327b0.pt
BOXER_CKPT="$BOXER_ROOT/ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt"
DINO_CKPT="$BOXER_ROOT/ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
PYTHON=/home/admin1/miniconda3/envs/ovm3d-1/bin/python
SCENE_ROOT="$ROOT/upstream_clean/scannet_readme_frames"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_graw_e2_preflight3.txt"
BASELINE_ROOT="$ROOT/results/scannet_graw_e2_replay1_score05"
CACHE_ROOT="$ROOT/cache/cutr_postfilter_v3/scannet-graw-e2-score05-preflight3-v3-r1"
LOG_ROOT="$ROOT/logs/scannet_boxer_unexplained_shadow_clean_in2_v5_score05"
OUTPUT_ROOT="$LOG_ROOT/boxer_raw"

if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "GPU must be a non-negative integer" >&2
  exit 2
fi
for required in \
  "$BOXER_ROOT/run_boxer.py" \
  "$OWL_CKPT" \
  "$OWL_TEXT_CACHE" \
  "$BOXER_CKPT" \
  "$DINO_CKPT" \
  "$SCENE_LIST"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing frozen shadow input: $required" >&2
    exit 1
  fi
done
if [[ "$(git -C "$BOXER_ROOT" rev-parse HEAD)" != "$BOXER_COMMIT" ]]; then
  echo "Unexpected clean Boxer commit" >&2
  exit 1
fi
if [[ -n "$(git -C "$BOXER_ROOT" status --porcelain)" ]]; then
  echo "Clean Boxer checkout has local modifications" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT/scenes" "$LOG_ROOT/mplconfig" "$OUTPUT_ROOT"
exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
  echo "Another Boxer shadow preflight holds $LOG_ROOT/run.lock" >&2
  exit 1
fi

printf 'scene\tmanifest_keyframes\tvalid_keyframes\tinvalid_pose_keyframes\traw_candidate_frames\traw_candidates\n' \
  >"$LOG_ROOT/schedule_audit.tsv"

sha256sum \
  "$BOXER_ROOT/run_boxer.py" \
  "$BOXER_ROOT/owl/owl_wrapper.py" \
  "$BOXER_ROOT/boxernet/boxernet.py" \
  "$OWL_CKPT" \
  "$OWL_TEXT_CACHE" \
  "$BOXER_CKPT" \
  "$DINO_CKPT" \
  "$ROOT/docs/UNEXPLAINED_DEPTH_BOXER_ORACLE_PREREGISTRATION.md" \
  >"$LOG_ROOT/frozen_inputs_sha256.txt"
while IFS= read -r scene || [[ -n "$scene" ]]; do
  sha256sum "$CACHE_ROOT/$scene/manifest.json" \
    >>"$LOG_ROOT/frozen_inputs_sha256.txt"
done <"$SCENE_LIST"

: >"$LOG_ROOT/native_before_sha256.txt"
while IFS= read -r scene || [[ -n "$scene" ]]; do
  sha256sum "$BASELINE_ROOT/${scene}_boxes.pkl" \
    >>"$LOG_ROOT/native_before_sha256.txt"
done <"$SCENE_LIST"

echo "[$(date '+%F %T')] Starting frozen OWLv2+Boxer shadow on GPU $GPU"
while IFS= read -r scene || [[ -n "$scene" ]]; do
  scene_dir="$SCENE_ROOT/$scene"
  cache_manifest="$CACHE_ROOT/$scene/manifest.json"
  scene_out="$OUTPUT_ROOT/$scene"
  scene_log="$LOG_ROOT/scenes/$scene.log"
  if [[ ! -f "$cache_manifest" ]]; then
    echo "Missing sealed T05 schedule: $cache_manifest" >&2
    exit 1
  fi
  expected_keyframes=$("$PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(int(d["record_count"]))' \
    "$cache_manifest")
  valid_keyframes=$("$PYTHON" -c \
    'import json,sys,numpy as np; from pathlib import Path; m=json.load(open(sys.argv[1])); root=Path(sys.argv[2]); expected=[str(int(x)) for x in m["recorded_frame_ids"]]; good=lambda x: (lambda p: p.exists() and np.isfinite(np.loadtxt(p)).all())(root/"frames"/"pose"/f"{x}.txt"); expected_valid=[x for x in expected if good(x)]; colors=sorted((p.stem for p in (root/"frames"/"color").iterdir() if p.suffix.lower() in (".png",".jpg")),key=int); loader_valid=[x for x in colors[0::25] if good(x)][:len(expected_valid)]; assert loader_valid == expected_valid, f"Boxer loader schedule differs from sealed T05 schedule: {loader_valid[-3:]} != {expected_valid[-3:]}"; print(len(expected_valid))' \
    "$cache_manifest" "$scene_dir")
  invalid_keyframes=$((expected_keyframes - valid_keyframes))
  if [[ -e "$scene_out/boxer_3dbbs.csv" || -e "$scene_log" ]]; then
    echo "Fresh Boxer shadow artifact already exists for $scene" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] Running $scene"
  (
    cd "$BOXER_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$BOXER_ROOT" \
    BOXFUSION_OWL_CKPT="$OWL_CKPT" \
    MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
    "$PYTHON" -c \
      'import builtins,os,numpy as np,torch; np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0); import owl.owl_wrapper as ow; import owl.clip_tokenizer as ct; ow._CKPT_PATH=os.environ["BOXFUSION_OWL_CKPT"]; ct._CKPT_PATH=os.environ["BOXFUSION_OWL_CKPT"]; import run_boxer; _loader=run_boxer.ScanNetLoader; run_boxer.ScanNetLoader=lambda *a,**kw: _loader(*a,**(kw|{"annotation_path":None})); _open=builtins.open; builtins.open=lambda f,*a,**kw: (_ for _ in ()).throw(RuntimeError("GT annotation access forbidden in shadow")) if isinstance(f,(str,bytes,os.PathLike)) and os.path.basename(os.fspath(f))=="full_annotations.json" else _open(f,*a,**kw); print("BOXFUSION_SHADOW_GT_ACCESS=forbidden annotation_path=None",flush=True); run_boxer.main()' \
      --input "$scene_dir" \
      --start_n 1 \
      --skip_n 25 \
      --max_n "$valid_keyframes" \
      --thresh2d 0.25 \
      --thresh3d 0.5 \
      --labels=lvisplus \
      --detector_hw 960 \
      --track \
      --skip_viz \
      --force_precision bfloat16 \
      --ckpt "$BOXER_CKPT" \
      --output_dir "$OUTPUT_ROOT"
  ) >"$scene_log" 2>&1
  if grep -qE 'Traceback|Exception in thread' "$scene_log"; then
    echo "Boxer shadow raised an exception for $scene; see $scene_log" >&2
    exit 1
  fi
  if ! grep -q "${valid_keyframes}/${valid_keyframes}" "$scene_log"; then
    echo "Boxer shadow did not execute the sealed keyframe count for $scene" >&2
    exit 1
  fi
  for required in \
    "$scene_out/owl_2dbbs.csv" \
    "$scene_out/boxer_3dbbs.csv"; do
    if [[ ! -s "$required" ]]; then
      echo "Missing Boxer shadow artifact: $required; see $scene_log" >&2
      exit 1
    fi
  done
  read -r raw_candidate_frames raw_candidates < <("$PYTHON" -c \
    'import csv,json,sys; m=json.load(open(sys.argv[1])); expected={int(x) for x in m["recorded_frame_ids"]}; rows=list(csv.DictReader(open(sys.argv[2],newline=""))); observed={int(r["time_ns"]) for r in rows}; extra=sorted(observed-expected); assert not extra, f"future/off-schedule Boxer timestamps: {extra}"; print(len(observed),len(rows))' \
    "$cache_manifest" "$scene_out/boxer_3dbbs.csv")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$scene" "$expected_keyframes" "$valid_keyframes" "$invalid_keyframes" \
    "$raw_candidate_frames" "$raw_candidates" >>"$LOG_ROOT/schedule_audit.tsv"
  echo "[$(date '+%F %T')] Completed $scene"
done <"$SCENE_LIST"

: >"$LOG_ROOT/native_after_sha256.txt"
while IFS= read -r scene || [[ -n "$scene" ]]; do
  sha256sum "$BASELINE_ROOT/${scene}_boxes.pkl" \
    >>"$LOG_ROOT/native_after_sha256.txt"
done <"$SCENE_LIST"
cmp "$LOG_ROOT/native_before_sha256.txt" "$LOG_ROOT/native_after_sha256.txt"
echo "[$(date '+%F %T')] Completed output-inert frozen Boxer shadow"

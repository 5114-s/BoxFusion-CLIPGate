#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 GPU SHARD_INDEX SHARD_COUNT}"
SHARD_INDEX="${2:?usage: $0 GPU SHARD_INDEX SHARD_COUNT}"
SHARD_COUNT="${3:?usage: $0 GPU SHARD_INDEX SHARD_COUNT}"

ROOT=/data/ZhaoX/BoxFusion
BOXER_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer
BOXER_COMMIT=1f86542dc342a4b1d474c87c97c5d1d6566d9148
OWL_CKPT=/data/ZhaoX/OVM3D-Dett/third_party/boxer/ckpts/owlv2-base-patch16-ensemble.pt
OWL_TEXT_CACHE=/data/ZhaoX/OVM3D-Dett/third_party/boxer/ckpts/owlv2-base-patch16-ensemble_textemb_878186d327b0.pt
BOXER_CKPT="$BOXER_ROOT/ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt"
DINO_CKPT="$BOXER_ROOT/ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
PYTHON=/home/admin1/miniconda3/envs/ovm3d-1/bin/python
SCENE_ROOT="$ROOT/upstream_clean/scannet_readme_frames"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
BASELINE_ROOT="$ROOT/results/scannet_topk_fusion_score05"
CACHE_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals/scannet-score05-gap25-postfilter-v2
LOG_ROOT="$ROOT/logs/scannet_raw_boxer_full100_score05_v1"
OUTPUT_ROOT="$LOG_ROOT/boxer_raw"
EXPECTED_LIST_SHA=4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5

if [[ ! "$GPU" =~ ^[0-9]+$ || ! "$SHARD_INDEX" =~ ^[0-9]+$ || ! "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU and shard arguments must be non-negative integers; SHARD_COUNT must be positive" >&2
  exit 2
fi
if (( SHARD_INDEX >= SHARD_COUNT )); then
  echo "SHARD_INDEX must be smaller than SHARD_COUNT" >&2
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
    echo "Missing frozen Raw Boxer input: $required" >&2
    exit 1
  fi
done
if [[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" != "$EXPECTED_LIST_SHA" ]]; then
  echo "Official full100 scene-list hash mismatch" >&2
  exit 1
fi
scene_count=$(grep -cvE '^[[:space:]]*(#|$)' "$SCENE_LIST")
if [[ "$scene_count" -ne 100 ]]; then
  echo "Official scene list must contain exactly 100 scenes; found $scene_count" >&2
  exit 1
fi
if [[ "$(git -C "$BOXER_ROOT" rev-parse HEAD)" != "$BOXER_COMMIT" ]]; then
  echo "Unexpected clean Boxer commit" >&2
  exit 1
fi
if [[ -n "$(git -C "$BOXER_ROOT" status --porcelain)" ]]; then
  echo "Clean Boxer checkout has local modifications" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT/scenes" "$LOG_ROOT/mplconfig_worker${SHARD_INDEX}" "$OUTPUT_ROOT"
WORKER_LOG="$LOG_ROOT/worker${SHARD_INDEX}_of_${SHARD_COUNT}.log"
AUDIT_TSV="$LOG_ROOT/schedule_audit_worker${SHARD_INDEX}_of_${SHARD_COUNT}.tsv"
LOCK_PATH="$LOG_ROOT/worker${SHARD_INDEX}_of_${SHARD_COUNT}.lock"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "Another Raw Boxer worker holds $LOCK_PATH" >&2
  exit 1
fi
exec > >(tee -a "$WORKER_LOG") 2>&1

if [[ ! -s "$AUDIT_TSV" ]]; then
  printf 'scene\tmanifest_keyframes\tvalid_keyframes\tinvalid_pose_keyframes\traw_candidate_frames\traw_candidates\n' >"$AUDIT_TSV"
fi

echo "[$(date '+%F %T')] Starting frozen no-GT Raw Boxer full100 worker $SHARD_INDEX/$SHARD_COUNT on GPU $GPU"
scene_index=0
completed=0
while IFS= read -r scene || [[ -n "$scene" ]]; do
  [[ -z "$scene" || "$scene" == \#* ]] && continue
  current_index=$scene_index
  scene_index=$((scene_index + 1))
  if (( current_index % SHARD_COUNT != SHARD_INDEX )); then
    continue
  fi

  scene_dir="$SCENE_ROOT/$scene"
  cache_manifest="$CACHE_ROOT/$scene/manifest.json"
  native_prediction="$BASELINE_ROOT/${scene}_boxes.pkl"
  scene_out="$OUTPUT_ROOT/$scene"
  scene_log="$LOG_ROOT/scenes/$scene.log"
  for required in "$scene_dir/frames" "$cache_manifest" "$native_prediction"; do
    if [[ ! -e "$required" ]]; then
      echo "Missing full100 scene input: $required" >&2
      exit 1
    fi
  done

  expected_keyframes=$("$PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(int(d["record_count"]))' \
    "$cache_manifest")
  valid_keyframes=$("$PYTHON" -c \
    'import json,sys,numpy as np; from pathlib import Path; m=json.load(open(sys.argv[1])); root=Path(sys.argv[2]); expected=[str(int(x)) for x in m["recorded_frame_ids"]]; good=lambda x: (lambda p: p.exists() and np.isfinite(np.loadtxt(p)).all())(root/"frames"/"pose"/f"{x}.txt"); expected_valid=[x for x in expected if good(x)]; colors=sorted((p.stem for p in (root/"frames"/"color").iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg")),key=int); loader_valid=[x for x in colors[0::25] if good(x)][:len(expected_valid)]; assert loader_valid == expected_valid, f"Boxer loader schedule differs from sealed T05 schedule: {loader_valid[-3:]} != {expected_valid[-3:]}"; print(len(expected_valid))' \
    "$cache_manifest" "$scene_dir")
  invalid_keyframes=$((expected_keyframes - valid_keyframes))

  if [[ -s "$scene_out/owl_2dbbs.csv" && -s "$scene_out/boxer_3dbbs.csv" && -s "$scene_log" ]]; then
    if grep -q "${valid_keyframes}/${valid_keyframes}" "$scene_log" && ! grep -qE 'Traceback|Exception in thread' "$scene_log"; then
      echo "[$(date '+%F %T')] Reusing completed $scene"
      completed=$((completed + 1))
      continue
    fi
    echo "Incomplete or invalid existing Raw Boxer scene artifact: $scene" >&2
    exit 1
  fi
  if [[ -e "$scene_out" || -e "$scene_log" ]]; then
    echo "Partial Raw Boxer artifact exists for $scene; refusing to overwrite" >&2
    exit 1
  fi

  echo "[$(date '+%F %T')] Running $scene ($valid_keyframes valid keyframes)"
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
    MPLCONFIGDIR="$LOG_ROOT/mplconfig_worker${SHARD_INDEX}" \
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
    echo "Raw Boxer raised an exception for $scene; see $scene_log" >&2
    exit 1
  fi
  if ! grep -q "${valid_keyframes}/${valid_keyframes}" "$scene_log"; then
    echo "Raw Boxer did not execute the sealed keyframe count for $scene" >&2
    exit 1
  fi
  for required in "$scene_out/owl_2dbbs.csv" "$scene_out/boxer_3dbbs.csv"; do
    if [[ ! -s "$required" ]]; then
      echo "Missing Raw Boxer output: $required" >&2
      exit 1
    fi
  done
  read -r raw_candidate_frames raw_candidates < <("$PYTHON" -c \
    'import csv,json,sys; m=json.load(open(sys.argv[1])); expected={int(x) for x in m["recorded_frame_ids"]}; rows=list(csv.DictReader(open(sys.argv[2],newline=""))); observed={int(r["time_ns"]) for r in rows}; extra=sorted(observed-expected); assert not extra, f"future/off-schedule Boxer timestamps: {extra}"; print(len(observed),len(rows))' \
    "$cache_manifest" "$scene_out/boxer_3dbbs.csv")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$scene" "$expected_keyframes" "$valid_keyframes" "$invalid_keyframes" \
    "$raw_candidate_frames" "$raw_candidates" >>"$AUDIT_TSV"
  completed=$((completed + 1))
  echo "[$(date '+%F %T')] Completed $scene"
done <"$SCENE_LIST"

echo "[$(date '+%F %T')] Raw Boxer worker $SHARD_INDEX/$SHARD_COUNT completed $completed scenes"

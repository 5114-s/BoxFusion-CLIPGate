#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
CODE="$MAIN/development/final_runtime"
PIPE="$MAIN/development/final_pipeline"
PYTHON=/home/admin1/miniconda3/envs/boxfusion-online/bin/python
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
SCENES="$MAIN/evaluation/data_util/meta_data/scannetv2_val.txt"
COMPONENTS="$MAIN/development/final_controls/results/scannet_recar3d_key_control_components"
FACTORIAL="$PIPE/results/scannet_recar3d_final_factorial100"
REPORT="$PIPE/reports/qualitative_recar3d_final"
RERUN="$REPORT/rerun"
CONFIG="$REPORT/final_qualitative.yaml"
RAW=/extra/ZhaoX/scannet_data/scans
GT="$MAIN/evaluation/data_util/scannet_train_detection_data"

mkdir -p "$REPORT" "$RERUN/native" "$RERUN/full" "$RERUN/diagnostics" \
  "$RERUN/evidence" "$RERUN/provider" "$RERUN/logs" "$RERUN/mplconfig"
exec > >(tee -a "$REPORT/driver.log") 2>&1
[[ -s "$FACTORIAL/manifest.json" ]] || { echo "Missing final ScanNet factorial" >&2; exit 1; }

"$PYTHON" "$PIPE/prepare_final_qualitative.py" --scene-list "$SCENES" \
  --components "$COMPONENTS" --factorial "$FACTORIAL" --output "$REPORT"

"$PYTHON" - "$CODE/config/scannet_recar3d_final_runtime.yaml" "$CONFIG" "$RERUN" <<'PY'
import pathlib, sys, yaml
source, target, root=map(pathlib.Path, sys.argv[1:])
cfg=yaml.safe_load(source.read_text())
cfg['data']['output_dir']=str(root/'native')
section=cfg['online_candidate_map']
section['output_root']=str(root/'full')
section['diagnostics_root']=str(root/'diagnostics')
section['audit_output_root']=None
section['provider']['diagnostics_root']=str(root/'provider')
section['provider']['evidence_cache_root']=str(root/'evidence')
section['state']['selected_calr_v1']=True
section['state']['export_qualitative_trace']=True
section['state']['audit_controls']=False
section['state']['audit_plr_controls']=False
section['state']['audit_direct_plr']=False
section['state']['audit_raw_iou_max']=False
target.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

while read -r scene; do
  [[ -n "$scene" ]] || continue
  [[ ! -e "$RERUN/native/${scene}_boxes.pkl" && ! -e "$RERUN/full/${scene}_boxes.pkl" ]] || {
    echo "Qualitative rerun output already exists for $scene" >&2; exit 1;
  }
  echo "[$(date '+%F %T')] qualitative trace $scene"
  (
    cd "$CODE"
    CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MPLCONFIGDIR="$RERUN/mplconfig" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$CODE:$MAIN/third_party/WeDetect" \
    "$PYTHON" demo.py ScanNet --model-path "$MAIN/models/cutr_rgbd.pth" \
      --clip_path "$MAIN/models/open_clip_pytorch_model.bin" \
      --class_txt "$MAIN/data/panoptic_categories_nomerge.txt" \
      --config "$CONFIG" --device cuda:0 --seq "$scene"
  ) > "$RERUN/logs/${scene}.log" 2>&1
  grep -q 'Online runtime receipt |' "$RERUN/logs/${scene}.log"
  [[ -s "$RERUN/diagnostics/${scene}.json" && -s "$RERUN/evidence/${scene}.npz" ]]
done < "$REPORT/selected_scenes.txt"

"$PYTHON" "$PIPE/validate_qualitative_parity.py" --scenes "$REPORT/selected_scenes.txt" \
  --selection "$REPORT/selected_cases.json" --diagnostics "$RERUN/diagnostics" \
  --native "$RERUN/native" --full "$RERUN/full" --components "$COMPONENTS" \
  --factorial "$FACTORIAL" --output "$REPORT/terminal_parity.json"
"$PYTHON" "$PIPE/build_final_qualitative_evidence.py" \
  --selection "$REPORT/selected_cases.json" --diagnostics "$RERUN/diagnostics" \
  --evidence "$RERUN/evidence" --raw-root "$RAW" --parity "$REPORT/terminal_parity.json" \
  --output "$REPORT"
MPLCONFIGDIR="$RERUN/mplconfig" "$PYTHON" "$PIPE/render_final_qualitative.py" \
  --selection "$REPORT/selected_cases.json" --evidence-dir "$REPORT" \
  --components "$COMPONENTS" --gt-root "$GT" --raw-root "$RAW" --output "$REPORT"

sha256sum "$CONFIG" "$REPORT/selected_cases.json" "$REPORT/terminal_parity.json" \
  "$REPORT/qualitative_comparison.pdf" "$REPORT/qualitative_comparison.svg" \
  "$REPORT/qualitative_comparison.png" "$REPORT/qualitative_comparison.tiff" \
  > "$REPORT/artifact_sha256.txt"
{
  echo '# Final ReCaR-3D qualitative evidence'
  echo
  echo 'This figure uses the final strict-causal online, no-child configuration.'
  echo 'Selected scenes were rerun only to export process traces. Rendering is gated by'
  echo 'strict selected-box/source/score/causal-trace parity against locked official100 cases.'
  echo 'Whole-scene replay differences are retained in terminal_parity.json as an audit.'
  echo
  echo 'Reproduction:'
  echo
  echo "    bash $PIPE/run_final_qualitative.sh"
} > "$REPORT/README.md"
echo "[$(date '+%F %T')] final qualitative evidence complete"

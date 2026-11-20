#!/usr/bin/env python3
"""Materialize a terminal count-matched causal streaming-memory CALR control.

For each scene, the frozen causal streaming memory is truncated to the number
of records emitted by n_A=3 CALR in that same scene.  This is the terminal
readout of a causal control whose time-t capacity is the contemporaneous n_A=3
output count: it borrows only the output count, never CALR's selected records.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_calr_causal_controls import read_rows, write_rows, same_rows, prediction, quality, THRESHOLDS
from tools.audit_final_ledger import scannet_inputs
from tools.true_fusion_audit_core import class_agnostic_ap

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--controls-root',type=Path,default=ROOT/'reports/scannet_calr_causal_controls_20261010')
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists() and any(a.output.iterdir()): raise RuntimeError(f'refusing overwrite: {a.output}')
 a.output.mkdir(parents=True,exist_ok=True)
 prefix_root=ROOT/'development/final_controls/experiments/mvsr_design_20261005/factorial_no_visibility/plr_mvsr'
 gts,alignments,transform=scannet_inputs()
 pred={}; total={'added':0,'predictions':0}; q={str(t):{'tp':0,'fp':0,'duplicate':0} for t in THRESHOLDS}; per={}
 for scene in gts:
  prefix=read_rows(prefix_root/f'{scene}_boxes.pkl')
  stream=read_rows(a.controls_root/'predictions/streaming_topk'/f'{scene}_boxes.pkl')
  n3=read_rows(a.controls_root/'predictions/nA3'/f'{scene}_boxes.pkl')
  assert same_rows(prefix,stream[:len(prefix)]) and same_rows(prefix,n3[:len(prefix)])
  count=len(n3)-len(prefix)
  tail=sorted(stream[len(prefix):],key=lambda r:-r[2])[:count]
  rows=prefix+tail
  assert len(rows)==len(n3)
  write_rows(a.output/'predictions'/f'{scene}_boxes.pkl',rows)
  boxes,scores=prediction(rows,transform,alignments[scene]); pred[scene]=(boxes,scores)
  total['added']+=len(tail); total['predictions']+=len(rows); per[scene]={'matched_added':len(tail),'nA3_added':count}
  for t in THRESHOLDS:
   _,added=quality(boxes,scores,len(prefix),gts[scene],t)
   for k in q[str(t)]: q[str(t)][k]+=added[k]
 metrics={str(t):class_agnostic_ap(pred,gts,t)['ap'] for t in THRESHOLDS}
 result={'schema':'boxfusion.scannet100.calr_streaming_count_matched.v1','strict_causal_state':True,'selection_uses_calr_records':False,'budget':'per-scene terminal output count exactly matched to n_A=3; realizable online with contemporaneous n_A=3 count','counts':total,'metrics':metrics,'added_quality':q,'per_scene':per}
 (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# CALR causal streaming-memory count-matched control','',
 'The streaming memory uses the same ordered anchors and causal state as the previously audited control. For each scene, only its terminal output count is matched to contemporaneous `n_A=3`; no CALR-selected record is reused.','',
 '| Added records | AP15 | AP25 | AP50 | Added TP/FP/dup @0.15 |','|---:|---:|---:|---:|---:|',
 f"| {total['added']:,} | {metrics['0.15']:.4f} | {metrics['0.25']:.4f} | {metrics['0.5']:.4f} | {q['0.15']['tp']:,}/{q['0.15']['fp']:,}/{q['0.15']['duplicate']:,} |"]
 (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines))
if __name__=='__main__': main()

# L3B-all active stress test (not deployable L3B active)

This run is retained only as a negative-control stress test. It must not be
reported as the deployable L3B result.

## Protocol

- Native prefix: B05/T05+Boxer, 100 official ScanNet scenes, 1,788 boxes.
- Candidate geometry: one frozen HB medoid for every L3B T1 track.
- Admission, novelty filtering, and birth NMS: none.
- Added candidates: 28,156; total output: 29,944 boxes.
- Evaluation score: constant 1.0 for every native and appended box.

## Results

| Output | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| B05 prefix | 31.0130 | 26.7911 | 12.0669 |
| L3B-all stress | 3.7770 | 3.0868 | 1.2035 |
| Delta | -27.2360 | -23.7044 | -10.8633 |

The active pool raises recall but collapses precision: at IoU 0.50 the final
precision/recall are 2.3845%/49.8255%. This is expected when all candidate
tracks are treated as detections under a constant-score protocol.

## Interpretation

L3B is a shadow geometry selector, not a track-admission module. Its oracle
uses GT only to select which tracks are appended. A meaningful deployable
test therefore requires a separately frozen, GT-free, past-only admission and
novelty gate. The all-track output is not Cbest and does not measure that gate.

## Evidence

- Materialization manifest:
  `results/scannet_l3b_hbmedoid_all_active_score05/L3B_HBMEDOID_ALL_ACTIVE_PAPER100.json`
- Official evaluator log:
  `logs/cgf_paper100_constant_score/scannet_l3b_hbmedoid_all_active_score05_constant.log`
- L3B oracle:
  `reports/l3b_hbmedoid_t1_selector_paper100_oracle/L3B_HBMEDOID_T1_SELECTOR_PAPER100_ORACLE.json`

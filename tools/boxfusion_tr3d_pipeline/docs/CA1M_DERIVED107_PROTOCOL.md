# CA-1M corrected C0 derived107 protocol

`derived107` is an internal, non-canonical diagnostic. It is **not** the
paper's official CA-1M 107-scene evaluation and must never be presented as a
strict reproduction of the paper result.

The authors published `after_filter_boxes.npy` for 103 of the 107 validation
scenes. The following four scenes do not have that released filtered GT:

```text
45663164
47115469
47331311
47332000
```

This protocol keeps the corrected C0 model path unchanged (`score_thresh=0.4`,
real detector confidence, gap 20). Its data overlay contains:

- 103 scene-directory symlinks to the author-GT canonical data root;
- four physical scene directories with locally derived GT and immutable
  provenance manifests.

The runner hard-links the 103 existing canonical prediction pickles into an
independent prediction root and verifies same inode, SHA256 and pickle schema.
Only the four derived scenes are inferred. All 107 predictions are then
evaluated together with the unchanged official evaluator.

The frozen local GT proxy uses real depth points with `stride=4`,
`voxel_size=0.02 m`, `depth<10 m`, and the author's strict `distance<0.1 m`
corner-proximity rule.  On three independent scenes with author-published GT,
the aggregate proxy agreement was precision `0.98084`, recall `0.99225`, and
Jaccard `0.97338`.  This strong agreement validates the proxy as an internal
diagnostic, but it does not turn the four derived files into author GT.

Prerequisite: build and audit the overlay at
`/extra/ZhaoX/boxfusion_ca1m_derived107_v1`. It must contain a root
`derived_gt_manifest.json` and a `derived_gt_manifest.json` inside each of the
four physical derived scene directories. Every manifest must state:

```text
derived=true
official_comparable=false
paper_claim_permitted=false
```

Run:

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
bash scripts/run_ca1m_c0_score04_real_score_derived107.sh 0,1
```

Outputs are isolated under:

```text
results/ca1m_repro/c0_score04_real_score_derived107_v1
logs/ca1m_repro/c0_score04_real_score_derived107_v1
reports/ca1m_repro/c0_score04_real_score_derived107_v1
data/ca1m_eval_derived107_v1
```

The final run manifest forcibly records `official_comparable=false`,
`paper_claim_permitted=false`, `official_public_gt_subset=103/107`, and the
four derived scene IDs. If the author's missing four filtered-GT files become
available, use a new strict107 protocol instead of relabeling this run.

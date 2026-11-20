#!/usr/bin/env python3
"""CPU-only evidence audit for a prospective office framework illustration."""
from __future__ import annotations

import copy
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

import build_framework_scene0030_real_assets as source


ROOT = source.ROOT
SCENE = "scene0131_02"
OUT = ROOT / "reports/framework_scene0131_visual_audit_20261115"
NATIVE = source.NATIVE.with_name(SCENE + ".npz")
EVIDENCE = source.EVIDENCE.with_name(SCENE + ".npz")
READOUT = source.READOUT.with_name(SCENE + ".npz")
RECEIPT = source.RECEIPT.parent / SCENE
GT_PATH = ROOT / "evaluation/data_util/scannet_train_detection_data" / (SCENE + "_bbox.npy")
LABELS = {3:'cabinet',4:'bed',5:'chair',6:'sofa',7:'table',8:'door',9:'window',10:'bookshelf',
          11:'picture',12:'counter',14:'desk',16:'curtain',24:'refrigerator',28:'shower curtain',
          33:'toilet',34:'sink',36:'bathtub',39:'other furniture'}


def dump(name, value):
    source.save_json(OUT / name, value)


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coverage_rows(boxes, sources, gt, matrix, raw_native):
    aligned = boxes @ matrix[:3, :3].T + matrix[:3, 3]
    overlaps = source.common.iou_matrix(aligned, gt)
    raw_ov = source.common.iou_matrix(raw_native @ matrix[:3, :3].T + matrix[:3, 3], gt)
    result = []
    for g in range(len(gt)):
        row = {"gt_index": g}
        for branch in ("native", "m1p", "m1a"):
            ids = np.flatnonzero(np.asarray(sources) == branch)
            best = int(ids[np.argmax(overlaps[ids, g])]) if len(ids) else -1
            row[branch + "_best_index"] = best
            row[branch + "_max_iou"] = float(overlaps[best, g]) if best >= 0 else 0.
        row["prefilter_native_max_iou"] = float(raw_ov[:, g].max())
        result.append(row)
    return result, overlaps


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    native_rows = source.replay.load_native(NATIVE)
    evidence_rows = source.replay.load_evidence(EVIDENCE)
    assert [f for f, _ in native_rows] == [f for f, _ in evidence_rows]
    config_path = RECEIPT / "effective_config.yaml"
    cfg = copy.deepcopy(yaml.safe_load(config_path.read_text())["online_candidate_map"]["state"])
    cfg.update(mvsr_only=False, selected_calr_v1=True, audit_controls=False,
               audit_direct_plr=False, audit_raw_iou_max=False, audit_mvsr_factors=False,
               audit_plr_controls=False)
    cfg["m2"].update(strength_variant="no_visibility", support_mode="max")
    state = source.build_reliability_candidate_map(SCENE, cfg)
    lineage = {}
    original = state.m1p._try_birth

    def traced(track, frame_id, native_corners):
        result = original(track, frame_id, native_corners)
        birth = state.m1p.births.get(track.track_id)
        if birth is not None and birth.evidence_frame_ids == tuple(sorted(track.observations)):
            lineage[track.track_id] = [copy.deepcopy(vars(v)) for v in
                                     sorted(track.observations.values(), key=lambda o: o.frame_id)]
        return result

    state.m1p._try_birth = traced
    for (frame, native), (_, evidence) in zip(native_rows, evidence_rows):
        state.update(frame, native, evidence)
    keep = source.native_filter(native)
    snap = state.materialize_terminal(frame, native.ids[keep], native.corners[keep], native.scores[keep])
    native_count = int(keep.sum())
    native_geometry_unchanged = np.array_equal(snap.boxes[:native_count], native.corners[keep])
    assert native_geometry_unchanged, "HSR must not change native geometry"
    matrix = source.common.alignment(SCENE)
    gta = np.load(GT_PATH)
    gt = gta[:, None, :3] + (source.common.CORNER_CODES[None] - .5) * gta[:, None, 3:6]
    rows, overlaps = coverage_rows(snap.boxes, snap.sources, gt, matrix, native.corners)
    for i, row in enumerate(rows):
        row["nyu40_id"] = int(gta[i, -1])
        row["class"] = LABELS[int(gta[i, -1])]
    write_csv("cache_replay_gt_coverage.csv", rows)
    ptr_rows = state.m1p.terminal_rows(native.corners[keep])
    var_rows = state.m1a_control.rows()
    branch_detail = []
    for branch, exported in (("m1p", ptr_rows), ("m1a", var_rows)):
        indexes = np.flatnonzero(np.asarray(snap.sources) == branch)
        assert len(indexes) == len(exported)
        for index, item in zip(indexes, exported):
            assert np.array_equal(snap.boxes[index], item["box"])
            g = int(np.argmax(overlaps[index]))
            rec = dict(branch=branch, prediction_index=int(index), gt_index=g,
                       gt_class=LABELS[int(gta[g, -1])], iou=float(overlaps[index, g]),
                       native_max=rows[g]["native_max_iou"], state_row=item)
            if branch == "m1p":
                obs = lineage[item["track_id"]]
                assert tuple(o["frame_id"] for o in obs) == item["evidence_frame_ids"]
                assert any(np.array_equal(o["corners"], item["box"]) for o in obs)
                rec["observations"] = obs
            else:
                voxel = tuple(item["voxel_key"])
                support = sorted(state.m1a_control.births[voxel].frames)
                assert len(support) == item["support_frames"]
                representative = dict(evidence_rows)[int(item["frame_id"])]
                r = int(np.flatnonzero(representative.anchor_ids == item["anchor_id"])[0])
                assert np.array_equal(item["box"], representative.anchor_corners[r])
                rec["support_frame_ids"] = support
                rec["birth"] = next(ev for ev in state.m1a_control.events if tuple(ev["voxel_key"]) == voxel)
            branch_detail.append(rec)
    dump("branch_lineage.json", branch_detail)

    # Keep the archived end-to-end run separate from the frozen-cache replay.
    diag_path = RECEIPT / "online_diagnostics" / (SCENE + ".json")
    diag = json.loads(diag_path.read_text())
    counts = diag["state"]["terminal_counts"]
    archived_native = source.common.read_predictions(RECEIPT / "native" / (SCENE + "_boxes.pkl"), matrix)
    archived_full = source.common.read_predictions(RECEIPT / "recar" / (SCENE + "_boxes.pkl"), matrix)
    n = counts["native"]
    assert n == len(archived_native["raw"])
    assert np.array_equal(archived_native["raw"], archived_full["raw"][:n])
    archive_sources = ["native"]*n + ["m1p"]*counts["m1p"] + ["m1a"]*counts["m1a"]
    assert len(archive_sources) == len(archived_full["raw"])
    archive_rows, _ = coverage_rows(archived_full["raw"], archive_sources, gt, matrix, archived_native["raw"])
    for i, r in enumerate(archive_rows):
        r["class"] = LABELS[int(gta[i, -1])]
    write_csv("archived_run_gt_coverage.csv", archive_rows)

    h = np.load(READOUT)
    offsets = np.r_[0, np.cumsum(h["lengths"])]
    last = slice(offsets[-2], offsets[-1])
    assert np.array_equal(h["ids"][last], native.ids[keep])
    score_error = float(np.max(np.abs(h["scores_historical_max"][last] - snap.scores[:int(keep.sum())])))
    assert score_error < 1e-12, "HSR readout must agree with replay"
    hsr = []
    hsr_terminal = []
    evidence_by_frame = dict(evidence_rows)
    for ordinal, (fid, nf) in enumerate(native_rows):
        ef = evidence_by_frame[fid]
        sl = slice(offsets[ordinal], offsets[ordinal + 1])
        ids = h["ids"][sl]
        ns, hs = h["scores_native"][sl], h["scores_historical_max"][sl]
        matches = source.helper.exact_assignment(nf.corners, ef.proposal_boxes_2d,
                                               nf.camera_to_world, nf.intrinsic, nf.width, nf.height)
        local_ov = source.common.iou_matrix(nf.corners @ matrix[:3, :3].T + matrix[:3, 3], gt)
        for r, identity in enumerate(ids):
            native_index = int(np.flatnonzero(nf.ids == identity)[0])
            g = int(local_ov[native_index].argmax())
            nr = int(np.flatnonzero(np.argsort(-ns, kind="stable") == r)[0]) + 1
            hr = int(np.flatnonzero(np.argsort(-hs, kind="stable") == r)[0]) + 1
            if ordinal == len(native_rows)-1 and nr > hr and local_ov[native_index, g] >= .4:
                j = sl.start + r
                hsr_terminal.append(dict(frame=fid,native_id=int(identity),prediction_index=r,
                    gt_index=g,gt_class=LABELS[int(gta[g,-1])],gt_iou=float(local_ov[native_index,g]),
                    native_score=float(ns[r]),hsr_score=float(hs[r]),native_rank=nr,hsr_rank=hr,
                    rank_gain=nr-hr,current_support=float(h['current_support'][j]),
                    historical_support=float(h['max_support'][j]),currently_matched=native_index in matches))
            if native_index not in matches or nr <= hr or local_ov[native_index, g] < .4:
                continue
            pr, miou, projected_rect = matches[native_index]
            j = sl.start + r
            assert abs(miou - h["iou2d"][j]) < 1e-10
            hsr.append(dict(frame=fid, native_id=int(identity), gt_index=g,
                            gt_class=LABELS[int(gta[g, -1])], gt_iou=float(local_ov[native_index, g]),
                            native_score=float(ns[r]), hsr_score=float(hs[r]),
                            native_rank=nr, hsr_rank=hr, rank_gain=nr-hr,
                            current_support=float(h["current_support"][j]),
                            historical_support=float(h["max_support"][j]),
                            proposal_id=int(ef.proposal_ids[pr]), iou2d=float(miou),
                            native_projection_xyxy=np.asarray(projected_rect).tolist(),
                            proposal_xyxy=ef.proposal_boxes_2d[pr].tolist()))
    dump("hsr_verified_examples.json", hsr)
    dump("hsr_terminal_examples.json", hsr_terminal)
    thresholds = {}
    for t in [.15, .25, .5]:
        missed = [r for r in rows if r["native_max_iou"] < t]
        thresholds[str(t)] = {
            "native_output_missed_gt": [r["gt_index"] for r in missed],
            "ptr_covered_missed_gt": [r["gt_index"] for r in missed if r["m1p_max_iou"] >= t],
            "var_covered_missed_gt": [r["gt_index"] for r in missed if r["m1a_max_iou"] >= t],
        }
    pairs = []
    missed15 = [r for r in rows if r["native_max_iou"] < .15]
    for p in missed15:
        for v in missed15:
            if p["gt_index"] != v["gt_index"]:
                pairs.append({"ptr_gt": p["gt_index"], "var_gt": v["gt_index"],
                              "ptr_iou": p["m1p_max_iou"], "var_iou": v["m1a_max_iou"],
                              "minimum_iou": min(p["m1p_max_iou"], v["m1a_max_iou"])})
    pairs.sort(key=lambda r: -r["minimum_iou"])
    inputs = [NATIVE, EVIDENCE, READOUT, GT_PATH, config_path, diag_path,
              Path(archived_native["path"]), Path(archived_full["path"]),
              ROOT / "development/mvsr_multiview_audit_20261012_source/boxfusion/reliability_candidate_map.py"]
    result = dict(scene=SCENE, schema="carr3d.visual_scene_audit.v1", gpu_inference=False,
                  cache_keyframes=len(native_rows), final_frame=frame,
                  cached_replay_counts={k: snap.sources.count(k) for k in ["native", "m1p", "m1a"]},
                  archived_run_counts=counts, hsr_readout_max_score_error=score_error,
                  hsr_terminal_native_geometry_unchanged=bool(native_geometry_unchanged),
                  hsr_examples_count=len(hsr),
                  hsr_terminal_examples=hsr_terminal,
                  best_hsr_examples=sorted(hsr, key=lambda r: (-r["rank_gain"], -r["gt_iou"]))[:8],
                  thresholds=thresholds, best_distinct_target_pair=pairs[0] if pairs else None,
                  gt_coverage=rows, source_hashes={str(p): source.common.sha(p) for p in inputs},
                  notes=["GT max-IoU coverage audit, not one-to-one detection AP/recall.",
                         "Native output uses the three-axis 0.30 m extent filter.",
                         "Original end-to-end run and frozen-cache replay are reported separately.",
                         "Branch labels in cache replay come from explicit snapshot sources and verified state rows.",
                         "No object or box was moved or fabricated to satisfy a desired figure layout."])
    dump("audit_summary.json", result)
    np.savez_compressed(OUT / "verified_geometry.npz", boxes_world=snap.boxes,
                        scores=snap.scores, sources=np.asarray(snap.sources), axis_alignment=matrix,
                        gt_aligned=gt, native_ids=native.ids[keep])
    print(json.dumps(source.jsonable({k: result[k] for k in ["cached_replay_counts", "archived_run_counts",
                     "hsr_examples_count", "best_distinct_target_pair", "thresholds", "best_hsr_examples"]}), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default=SCENE)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    SCENE, OUT = args.scene, args.output
    NATIVE = source.NATIVE.with_name(SCENE+'.npz')
    EVIDENCE = source.EVIDENCE.with_name(SCENE+'.npz')
    READOUT = source.READOUT.with_name(SCENE+'.npz')
    RECEIPT = source.RECEIPT.parent/SCENE
    GT_PATH = ROOT/'evaluation/data_util/scannet_train_detection_data'/(SCENE+'_bbox.npy')
    main()

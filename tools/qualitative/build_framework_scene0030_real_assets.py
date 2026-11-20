#!/usr/bin/env python3
"""Audited, CPU-only framework assets from a single real ScanNet sequence.

No generative imagery, model inference, modified prediction geometry, or
hand-associated track evidence. The frozen state implementation is replayed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SCENE = "scene0030_01"
OUT = ROOT / "figure_assets/carr3d_framework_real_scene0030_01_20261114"
BASE_OUT = OUT
STROKES = {"rgb": 5, "rgb_halo": 8, "native_3d": 6, "recovery_3d": 9}
SOURCE = ROOT / "development/mvsr_multiview_audit_20261012_source"
NATIVE = ROOT / "development/final_controls/experiments/mvsr_multiview_audit_20261012/native_prefix" / f"{SCENE}.npz"
EVIDENCE = ROOT / "development/final_controls/experiments/mvsr_unified_samepass_20261010/evidence" / f"{SCENE}.npz"
READOUT = ROOT / "reports/scannet_hcsr_current_zero_vs_max_20261029/frame_readouts" / f"{SCENE}.npz"
RECEIPT = ROOT / "reports/official_github_boxfusion_vs_final_recar_public_cli_coldcache_nowarmup100_20261009/receipts/recar" / SCENE
CONFIG = RECEIPT / "effective_config.yaml"
RGB = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/color"
DEPTH = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/depth"
RAW = ROOT / "reports/scannet_seedless_full100_20260911/raw" / SCENE
BLUE, ORANGE, GOLD, PURPLE = "#2779B5", "#EF7624", "#C59308", "#8054AF"
INK = "#223348"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))
from tools import replay_scannet_online_prefix as replay
from boxfusion.reliability_candidate_map import build_reliability_candidate_map


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


common = module("framework_render_common", ROOT / "tools/qualitative/render_boxfusion_ctrl_recar_3d.py")
helper = module("framework_projection", ROOT / "tools/qualitative/build_cross_keyframe_real_assets_scene0599.py")


def native_filter(native):
    return np.all(np.ptp(native.corners, axis=1) >= .3, axis=1)


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [jsonable(v) for v in value]
    return value


def save_json(path, obj):
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def font(size, bold=False):
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def sheet(items, path, cols=4, tw=420, th=300):
    rows = (len(items)+cols-1)//cols
    canvas = Image.new("RGB", (cols*tw, rows*(th+50)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, im) in enumerate(items):
        im = Image.open(im) if isinstance(im, Path) else im
        im = im.convert("RGBA")
        im = ImageOps.contain(im, (tw-24, th-20), Image.Resampling.LANCZOS)
        x, y = idx % cols * tw, idx // cols * (th+50)
        canvas.paste(im, (x+(tw-im.width)//2, y+(th-im.height)//2), im)
        draw.text((x+12, y+th+8), label, fill=INK, font=font(18))
    canvas.save(path)


def probe():
    OUT.mkdir(exist_ok=True)
    for directory in ["01_input", "02_native", "03_auxiliary", "04_ptr", "05_var", "06_hsr", "07_output", "08_audit"]:
        (OUT/directory).mkdir(exist_ok=True)
    nr, er = replay.load_native(NATIVE), replay.load_evidence(EVIDENCE)
    assert [f for f, _ in nr] == [f for f, _ in er]
    cfg = copy.deepcopy(yaml.safe_load(CONFIG.read_text())["online_candidate_map"]["state"])
    cfg.update(mvsr_only=False, selected_calr_v1=True, audit_controls=False,
               audit_direct_plr=False, audit_raw_iou_max=False,
               audit_mvsr_factors=False, audit_plr_controls=False)
    cfg["m2"].update(strength_variant="no_visibility", support_mode="max")
    state = build_reliability_candidate_map(SCENE, cfg)
    lineage = {}
    original = state.m1p._try_birth

    def traced_birth(track, frame_id, native_corners):
        result = original(track, frame_id, native_corners)
        if track.track_id in state.m1p.births:
            birth = state.m1p.births[track.track_id]
            if birth.evidence_frame_ids == tuple(sorted(track.observations)):
                lineage[track.track_id] = [copy.deepcopy(vars(v)) for v in sorted(track.observations.values(), key=lambda r: r.frame_id)]
        return result

    state.m1p._try_birth = traced_birth
    for (fid, native), (_, evidence) in zip(nr, er):
        state.update(fid, native, evidence)
    keep = native_filter(native)
    snapshot = state.materialize_terminal(fid, native.ids[keep], native.corners[keep], native.scores[keep])
    matrix = common.alignment(SCENE)
    gt_arr = np.load(ROOT/"evaluation/data_util/scannet_train_detection_data"/f"{SCENE}_bbox.npy")
    gt = gt_arr[:, None, :3] + (common.CORNER_CODES[None]-.5)*gt_arr[:, None, 3:6]
    aligned = snapshot.boxes @ matrix[:3, :3].T + matrix[:3, 3]
    overlaps = common.iou_matrix(aligned, gt)
    native_n = int(keep.sum())
    pairs = []
    ptr_ids = [i for i, src in enumerate(snapshot.sources) if src == "m1p"]
    var_ids = [i for i, src in enumerate(snapshot.sources) if src == "m1a"]
    native_max = overlaps[:native_n].max(0)
    for p in ptr_ids:
        pg = int(overlaps[p].argmax())
        if native_max[pg] > .15 or overlaps[p, pg] < .55: continue
        for v in var_ids:
            vg = int(overlaps[v].argmax())
            if pg == vg or native_max[vg] > .15 or overlaps[v, vg] < .55: continue
            pairs.append((min(overlaps[p, pg], overlaps[v, vg]), overlaps[p, pg]+overlaps[v, vg], p, v, pg, vg))
    if not pairs: raise RuntimeError("No verified distinct PTR/VAR recovery pair")
    _, _, pi, vi, pg, vg = max(pairs)
    ptr_row = state.m1p.terminal_rows(native.corners[keep])[ptr_ids.index(pi)]
    var_row = state.m1a_control.rows()[var_ids.index(vi)]
    assert np.array_equal(ptr_row["box"], snapshot.boxes[pi])
    assert np.array_equal(var_row["box"], snapshot.boxes[vi])
    ptr_obs = lineage[ptr_row["track_id"]]
    assert tuple(o["frame_id"] for o in ptr_obs) == ptr_row["evidence_frame_ids"]
    assert any(np.array_equal(o["corners"], ptr_row["box"]) for o in ptr_obs)
    key = tuple(var_row["voxel_key"])
    var_obs = []
    support_frames = state.m1a_control.births[key].frames
    for frame, evidence in er:
        if frame not in support_frames: continue
        keys = np.floor(evidence.anchor_corners.mean(1)/.3).astype(int)
        rows = np.flatnonzero(np.all(keys == np.asarray(key), axis=1))
        ri = min(rows, key=lambda i: (-evidence.anchor_scores[i], int(evidence.anchor_ids[i])))
        var_obs.append(dict(frame_id=frame, source_id=int(evidence.anchor_ids[ri]),
                            corners=evidence.anchor_corners[ri].copy(), score=float(evidence.anchor_scores[ri])))
    assert len(var_obs) == var_row["support_frames"]
    assert any(np.array_equal(o["corners"], var_row["box"]) for o in var_obs)
    display = []
    used = {pg, vg}
    for i in np.argsort(-overlaps[:native_n].max(1), kind="stable"):
        g = int(overlaps[i].argmax())
        if g in used or overlaps[i, g] < .4: continue
        display.append(int(i)); used.add(g)
        if len(display) == 4: break
    payload = dict(config=cfg, native_rows=nr, evidence_rows=er, snapshot=snapshot,
                   matrix=matrix, gt=gt, ptr_row=ptr_row, var_row=var_row,
                   ptr_observations=ptr_obs, var_observations=var_obs,
                   ptr_index=pi, var_index=vi, ptr_gt=pg, var_gt=vg,
                   ptr_iou=float(overlaps[pi,pg]), var_iou=float(overlaps[vi,vg]),
                   native_display_indices=display, native_count=native_n,
                   native_display_ids=native.ids[keep][display],
                   var_birth=next(ev for ev in state.m1a_control.events if tuple(ev["voxel_key"]) == key))
    with (OUT/"08_audit/replay_payload.pkl").open("wb") as handle: pickle.dump(payload, handle)
    save_json(OUT/"08_audit/selection.json", {k:v for k,v in payload.items() if k not in ("native_rows","evidence_rows","snapshot")})
    frames = [f for f,_ in nr]
    sheet([(f"RGB frame {f}", RGB/f"{f}.jpg") for f in frames[::4]], OUT/"08_audit/rgb_candidates.png", cols=4)
    print(json.dumps(jsonable({k:payload[k] for k in ["ptr_row","var_row","ptr_iou","var_iou","native_display_ids","var_birth"]}), indent=2))
    print("PTR evidence:", [(r["frame_id"],r["source_id"]) for r in ptr_obs])
    print("VAR evidence:", [(r["frame_id"],r["source_id"]) for r in var_obs])


def export_rgb_overlay(frame, boxes, colors, destination, native, *, mode="3d", crop=True):
    """Project stored boxes without changing their geometry or camera pose."""
    image = Image.open(RGB/f"{frame}.jpg").convert("RGB")
    draw = ImageDraw.Draw(image)
    all_points = []
    scale = np.array([image.width/native.width, image.height/native.height])
    for box, color in zip(boxes, colors):
        if mode == "3d":
            uv, depth = helper.project_world(np.asarray(box)[None], native.camera_to_world, native.intrinsic)
            if np.any(depth <= .001): raise ValueError(f"frame {frame}: behind camera")
            uv = uv[0]*scale
            for a, b in helper.cuboid_edges(box):
                line = [tuple(uv[a]), tuple(uv[b])]
                draw.line(line, fill="white", width=STROKES["rgb_halo"])
                draw.line(line, fill=color, width=STROKES["rgb"])
            all_points.extend(uv.tolist())
        else:
            rect = np.asarray(box, float).reshape(2,2)
            if mode == "2d": rect = rect*scale
            draw.rectangle(tuple(rect.ravel()), outline="white", width=STROKES["rgb_halo"])
            draw.rectangle(tuple(rect.ravel()), outline=color, width=STROKES["rgb"])
            all_points.extend(rect.tolist())
    image.save(destination.with_name(destination.stem+"_full.png"), dpi=(400,400))
    if crop and all_points:
        image = image.crop(helper.crop_bounds(np.asarray(all_points), image.width, image.height))
    image.save(destination, dpi=(400,400))
    return destination


def depth_image(frame):
    values = np.asarray(Image.open(DEPTH/f"{frame}.png"), dtype=float)
    valid = values > 0
    low, high = np.percentile(values[valid], [3,97])
    gray = np.zeros(values.shape, dtype=np.uint8)
    gray[valid] = np.asarray(245-210*np.clip((values[valid]-low)/(high-low),0,1), np.uint8)
    result = Image.fromarray(gray).convert("RGB")
    return result, dict(frame=frame,raw_units="millimetres",display_near_mm=float(low),display_far_mm=float(high),invalid_pixel_value=0)


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    for directory in ["01_input", "02_native", "03_auxiliary", "04_ptr", "05_var", "06_hsr", "07_output", "08_audit"]:
        (OUT/directory).mkdir(parents=True, exist_ok=True)
    payload = OUT/"08_audit/replay_payload.pkl"
    if not payload.exists(): payload = BASE_OUT/"08_audit/replay_payload.pkl"
    with payload.open("rb") as handle: data = pickle.load(handle)
    nr, er = dict(data["native_rows"]), dict(data["evidence_rows"])
    final_frame = max(nr)
    final_native = nr[final_frame]
    snapshot = data["snapshot"]
    matrix = data["matrix"]
    transform = lambda x: np.asarray(x) @ matrix[:3,:3].T + matrix[:3,3]
    preview = []
    audit = dict(scene=SCENE, model="CaRR-3D", generative_images_used=False,
                 prediction_geometry_modified=False, GPU_inference_used=False,
                 sources=dict(native_cache=str(NATIVE),evidence_cache=str(EVIDENCE),
                              state_config=str(CONFIG),state_code=str(SOURCE),hsr_readout=str(READOUT)),
                 notes=["Single scene; component illustrations use different real keyframes, not one synchronous snapshot.",
                        "GT-assisted selection of illustrative targets; not all output boxes are displayed.",
                        "Mesh is visualization-only. It is not an online reconstruction or a model input."])
    audit["source_hashes"] = {str(p):common.sha(p) for p in (NATIVE,EVIDENCE,CONFIG,READOUT)}
    audit["stroke_width_pixels"] = dict(STROKES)
    audit["render_payload"] = {"path":str(payload),"sha256":common.sha(payload)}

    # Input stream: keep original pixels and store unnormalized depth separately.
    input_frames = [100,1100,1200]
    audit["input_frames"] = []
    for frame in input_frames:
        im = Image.open(RGB/f"{frame}.jpg").convert("RGB")
        im.save(OUT/f"01_input/rgb_f{frame:04d}.png",dpi=(400,400))
        imd, depmeta = depth_image(frame)
        imd.save(OUT/f"01_input/depth_f{frame:04d}.png",dpi=(400,400))
        Image.open(DEPTH/f"{frame}.png").save(OUT/f"01_input/depth_raw_mm_f{frame:04d}.png")
        audit["input_frames"].append(depmeta)
    preview += [("RGB input | f1100",OUT/"01_input/rgb_f1100.png"),
                ("Depth | same frame",OUT/"01_input/depth_f1100.png")]

    # Frontend examples from actual post-NMS and Top-300 lifted cached records.
    p_frame, a_frame = 1250, 1050
    anchor_chosen = [o for o in data["var_observations"] if o["frame_id"] == a_frame][0]
    proposal_chosen = [o for o in data["ptr_observations"] if o["frame_id"] == p_frame][0]
    p_path = export_rgb_overlay(p_frame,[proposal_chosen["corners"]],[ORANGE],OUT/"03_auxiliary/lifted_proposal_f1250.png",nr[p_frame])
    a_path = export_rgb_overlay(a_frame,[anchor_chosen["corners"]],[GOLD],OUT/"03_auxiliary/lifted_anchor_f1050.png",nr[a_frame])
    preview += [("Lifted proposal | f1250",p_path),("Lifted anchor | f1050",a_path)]
    audit["auxiliary_examples"] = dict(proposal_frame=p_frame,anchor_frame=a_frame,proposal_id=proposal_chosen["source_id"], anchor_id=anchor_chosen["source_id"], selection="one displayed cached candidate per level")

    # Genuine native detector rectangles, not mislabeled auxiliary proposals.
    logfile = RECEIPT/"native_boxer_diagnostics"/f"{SCENE}_boxer_lifting.jsonl"
    log = next(json.loads(line) for line in logfile.read_text().splitlines() if json.loads(line)["frame_id"] == 1100)
    rectangles = np.asarray(log["input_boxes_xyxy"])
    scores = np.asarray(log["detector_scores"])
    select = np.argsort(-scores,kind="stable")[:4]
    native_proposals = export_rgb_overlay(1100,rectangles[select],[BLUE]*len(select),OUT/"02_native/cutr_proposals_f1100.png",nr[1100],mode="2d",crop=False)
    audit["native_detector_example"] = dict(frame=1100,source=str(logfile),source_sha256=common.sha(logfile),display_rows=select,boxes_xyxy=rectangles[select],
        caveat="The CuTR illustration comes from the 20261009 archived frontend diagnostic; prefix/state illustrations come from the separately cached 20261012-native/20261010-evidence replay. This is a module illustration, not a claimed end-to-end single-run trace.")
    preview.append(("CuTR detections | f1100", native_proposals))

    # These are members of the actual final track/voxel, not nearest boxes.
    ptr_frames, var_frames = [1250,1275,1300], [1025,1050,1075]
    audit["ptr"] = dict(track_id=data["ptr_row"]["track_id"],confirmation_frame=data["ptr_row"]["confirmation_frame_id"],
                         final_iou=data["ptr_iou"],gt_index=data["ptr_gt"],
                         final_representative_frame=next(o["frame_id"] for o in data["ptr_observations"] if np.array_equal(o["corners"],data["ptr_row"]["box"])),
                         displayed_frames=ptr_frames, selection="3 consecutive supporting keyframes from the recorded track; not the first three observations")
    audit["var"] = dict(voxel_key=data["var_row"]["voxel_key"],voxel_m=.3,birth_frame=data["var_birth"]["frame_id"],
                         final_iou=data["var_iou"],gt_index=data["var_gt"],final_exemplar_frame=data["var_row"]["frame_id"],
                         final_exemplar_anchor=data["var_row"]["anchor_id"],displayed_frames=var_frames,
                         selection="highest-scoring contributing anchor in each of 3 consecutive supporting keyframes of the actual voxel; not the first three observations")
    for branch, frames, color in [("ptr",ptr_frames,ORANGE),("var",var_frames,GOLD)]:
        directory = "04_ptr" if branch == "ptr" else "05_var"
        obs_by_frame = {o["frame_id"]:o for o in data[f"{branch}_observations"]}
        rows = []
        for f in frames:
            obs = obs_by_frame[f]
            path = export_rgb_overlay(f,[obs["corners"]],[color],OUT/f"{directory}/{branch}_support_f{f:04d}.png",nr[f])
            preview.append((f"{branch.upper()} support | f{f}",path))
            entry = dict(frame=f,id=obs["source_id"],score=obs["score"],corners=obs["corners"])
            if branch == "var":
                entry["voxel_key"] = np.floor(obs["corners"].mean(0)/.3).astype(int)
                assert np.array_equal(entry["voxel_key"],data["var_row"]["voxel_key"])
            rows.append(entry)
        audit[branch]["displayed_observations"] = rows
        sheet([(f"Frame {f}",OUT/f"{directory}/{branch}_support_f{f:04d}.png") for f in frames], OUT/f"{directory}/{branch}_three_keyframes.png",cols=3,tw=480,th=330)

    # Historical support: exact exclusive proposal association and audited scores.
    hframe, identity = 1400, 354
    native, evidence = nr[hframe], er[hframe]
    ri = int(np.flatnonzero(native.ids == identity)[0])
    assignments = helper.exact_assignment(native.corners,evidence.proposal_boxes_2d,native.camera_to_world,native.intrinsic,native.width,native.height)
    pr, match_iou, native_rect = assignments[ri]
    match_path = export_rgb_overlay(hframe,[native_rect,evidence.proposal_boxes_2d[pr]],[BLUE,PURPLE],OUT/"06_hsr/native_proposal_match_f1400.png",native,mode="2d")
    preview.append(("HSR exact match | f1400",match_path))
    h = np.load(READOUT)
    off = np.r_[0,np.cumsum(h["lengths"])]
    history = []
    for f in [1325,1350,1375,1400,1425,1450,1475,1500]:
        idx = int(np.flatnonzero(h["frame_ids"] == f)[0]); sl = slice(off[idx],off[idx+1])
        row = int(np.flatnonzero(h["ids"][sl] == identity)[0]); j = sl.start+row
        ns, rs = h["scores_native"][sl], h["scores_historical_max"][sl]
        history.append(dict(frame=f,current_support=float(h["current_support"][j]),historical_max=float(h["max_support"][j]),
                            native_score=float(ns[row]),hsr_score=float(rs[row]),
                            native_rank=int(np.flatnonzero(np.argsort(-ns,kind="stable") == row)[0])+1,
                            hsr_rank=int(np.flatnonzero(np.argsort(-rs,kind="stable") == row)[0])+1))
    event = next(row for row in history if row["frame"] == hframe)
    audit["hsr"] = dict(native_id=identity,display_frame=hframe,matched_proposal_id=int(evidence.proposal_ids[pr]),iou2d=float(match_iou),history=history,ranking_scope="size-filtered native readout within the same frame",**{k:v for k,v in event.items() if k!="frame"})
    fig, ax = plt.subplots(figsize=(5.5,2.6))
    xx = np.arange(len(history))
    ax.bar(xx,[v["current_support"] for v in history],width=.6,color="#DCCBEF",label="Current support")
    ax.step(xx,[v["historical_max"] for v in history],where="post",color=PURPLE,linewidth=2.3,label="Historical max")
    ax.scatter(xx,[v["historical_max"] for v in history],color=PURPLE,s=18,zorder=4)
    ax.set_ylim(0,1.02); ax.set_xticks(xx,[str(v["frame"]) for v in history],rotation=45,fontsize=8)
    ax.set_xlabel("Frame"); ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,fontsize=8,loc="lower left")
    fig.tight_layout()
    for ext in ["png","pdf","svg"]: fig.savefig(OUT/f"06_hsr/actual_support_history.{ext}",dpi=400,transparent=True)
    plt.close(fig)
    preview.append(("HSR measured support",OUT/"06_hsr/actual_support_history.png"))

    # Common fixed perspective and common crop for all 3D state illustrations.
    renderer = common.load_renderer()
    original_manifest = json.loads((ROOT/"figure_assets/online_detection_readout_dual_branch_candidates_20261107/source_manifest.json").read_text())
    scene_meta = next(x for x in original_manifest["scenes"] if x["scene"] == SCENE)
    walls = scene_meta["removed_wall_instance_ids"]
    xyz, rgb, display_rgb, faces, mesh_info = renderer.load_mesh(SCENE,matrix,walls)
    camera = copy.deepcopy(scene_meta["camera"])
    scale = 2
    camera["focal_pixels"] *= scale
    camera["principal_point_pixels"] = (np.asarray(camera["principal_point_pixels"])*scale).tolist()
    camera["image_size"] = [int(x*scale) for x in camera["image_size"]]
    pargs = (np.asarray(camera["position"]),np.asarray(camera["look_at"]),camera["focal_pixels"],np.asarray(camera["principal_point_pixels"]))
    projected = renderer.project(xyz,*pargs)
    width,height = camera["image_size"]
    background = renderer.rasterize(projected,display_rgb,faces,width,height)
    image = Image.fromarray(background)
    native_boxes = snapshot.boxes[data["native_display_indices"]]
    ptr_box, var_box = data["ptr_row"]["box"], data["var_row"]["box"]
    shown_boxes = np.concatenate([native_boxes,ptr_box[None],var_box[None]])
    pts = renderer.project(transform(shown_boxes).reshape(-1,3),*pargs)[:,:2]
    bbox = image.getbbox()
    low = np.minimum(np.array(bbox[:2]),pts.min(0))-24
    high = np.maximum(np.array(bbox[2:]),pts.max(0))+24
    bounds = [int(low[0]),int(low[1]),int(high[0]),int(high[1])]
    # Never alter mesh vertices or box corners; only camera/crop/display shading.
    def scene_image(stem, boxes, colors, *, focus=None):
        out = image.copy()
        draw = ImageDraw.Draw(out)
        for box,color in zip(boxes,colors):
            uv = renderer.project(transform(box),*pargs)[:,:2]
            for a,b in helper.cuboid_edges(box):
                draw.line([tuple(uv[a]),tuple(uv[b])],fill=color,
                          width=STROKES["native_3d"] if color==BLUE else STROKES["recovery_3d"])
        out = out.crop(tuple(bounds))
        if focus is not None:
            uv = renderer.project(transform(focus),*pargs)[:,:2]-np.array(bounds[:2])
            # Consistent moderate context, not a fabricated object silhouette.
            box = helper.crop_bounds(uv,out.width,out.height,ratio=1.25)
            out = out.crop(box)
        transparent = OUT/(stem+".png")
        out.save(transparent,dpi=(400,400))
        white = Image.new("RGB",out.size,"white"); white.paste(out,mask=out.getchannel("A"))
        white.save(OUT/(stem+"_white.png"),dpi=(400,400))
        return transparent

    clean = scene_image("07_output/scene_mesh",[],[])
    current = scene_image("02_native/native_map_current_f1600",native_boxes,[BLUE]*len(native_boxes))
    prev = nr[1575]
    previous_indices = [int(np.flatnonzero(prev.ids==id)[0]) for id in data["native_display_ids"]]
    previous = scene_image("02_native/native_map_previous_f1575",prev.corners[previous_indices],[BLUE]*len(previous_indices))
    final = scene_image("07_output/carr3d_output_selected",shown_boxes,[BLUE]*len(native_boxes)+[ORANGE,GOLD])
    p3d = scene_image("04_ptr/ptr_recovery_context",[ptr_box],[ORANGE],focus=ptr_box)
    a3d = scene_image("05_var/var_recovery_context",[var_box],[GOLD],focus=var_box)
    h3d = scene_image("06_hsr/native_geometry_unchanged",[native.corners[ri]],[BLUE],focus=native.corners[ri])
    # An honest all-output audit exists alongside the compact diagram readout.
    scene_image("08_audit/all_replay_outputs",snapshot.boxes,[BLUE if s=="native" else ORANGE if s=="m1p" else GOLD for s in snapshot.sources])
    scene_image("08_audit/selected_gt_reference",[data["gt"][data["ptr_gt"]] @ np.linalg.inv(matrix[:3,:3]).T - matrix[:3,3] @ np.linalg.inv(matrix[:3,:3]).T,
                                                data["gt"][data["var_gt"]] @ np.linalg.inv(matrix[:3,:3]).T - matrix[:3,3] @ np.linalg.inv(matrix[:3,:3]).T],["#14A06C"]*2)
    preview += [("Previous native map | f1575",previous),("Native map | f1600",current),
                ("PTR recovery | 3D context",p3d),("VAR recovery | 3D context",a3d),
                ("HSR | fixed geometry",h3d),("CaRR-3D | selected output",final)]
    audit["mesh"] = mesh_info; audit["camera"] = camera
    audit["crop_bounds"] = bounds
    audit["readout"] = dict(frame=final_frame,counts={s:snapshot.sources.count(s) for s in ["native","m1p","m1a"]},
                             total=len(snapshot.boxes),selected_native_ids=data["native_display_ids"],
                             selected_ptr_track=data["ptr_row"]["track_id"],selected_var_voxel=data["var_row"]["voxel_key"],
                             size_filter="native AABB extent >= 0.30 m on all axes; terminal readout API, no future observations")
    np.savez_compressed(OUT/"08_audit/geometry_and_scores.npz",boxes_world=snapshot.boxes,scores=snapshot.scores,
                        sources=np.asarray(snapshot.sources),axis_alignment=matrix,
                        ptr_box=ptr_box,var_box=var_box,native_display=native_boxes,
                        ptr_evidence=np.stack([o["corners"] for o in data["ptr_observations"]]),
                        var_evidence=np.stack([o["corners"] for o in data["var_observations"]]))
    sheet(preview,OUT/"00_all_assets_preview.png",cols=4)
    sheet([("Native output",current),("CaRR-3D selected output",final)],OUT/"00_scene_comparison.png",cols=2,tw=1000,th=760)
    audit["recommended_replacements"] = [dict(label=label,path=str(path.relative_to(OUT))) for label,path in preview]
    with (OUT/"08_audit/hsr_history.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    save_json(OUT/"source_manifest.json",audit)
    instructions = f"""CaRR-3D 流程图真实图片包 — {SCENE}

推荐场景：书架、桌椅与黑板的空间轮廓清晰；PTR 与 VAR 的示例分别对应两个不同的书架。
全部采用真实 RGB-D、原生记录及冻结候选缓存。没有生成式填充，没有移动或缩放检测框几何，没有重新运行 GPU 模型。

替换位置
1. 输入 RGB、Depth：01_input/rgb_f1100.png、depth_f1100.png（同一帧；未对齐的原生传感器图像仅用于展示，不用于新增计算）。
2. Native path 的 CuTR proposals：02_native/cutr_proposals_f1100.png。
3. Previous Native Map / Native Map：02_native/native_map_previous_f1575.png、native_map_current_f1600.png。
4. Auxiliary path：03_auxiliary/lifted_proposal_f1250.png 和 lifted_anchor_f1050.png；分别是实际提升后的单个候选，不是完整候选集合。
5. PTR：04_ptr/ptr_support_f1250.png、f1275、f1300；后接 ptr_recovery_context.png。
6. VAR：05_var/var_support_f1025.png、f1050、f1075；后接 var_recovery_context.png。大书架在部分原始视图中被画面边缘截断，未补画不可见区域。
7. HSR：06_hsr/native_proposal_match_f1400.png、actual_support_history.png、native_geometry_unchanged.png。
8. 最右侧输出：07_output/carr3d_output_selected.png；同目录提供白底版本。

绘图与证据边界
- 本版绘制线宽（源图像素）：原生三维框 {STROKES['native_3d']}，恢复三维框 {STROKES['recovery_3d']}；RGB 框 {STROKES['rgb']}，白色描边总宽 {STROKES['rgb_halo']}。仅改变绘制线宽，不改变框坐标、相机、颜色或评分。
- 三维图片提供透明底和白底；RGB 不作生成式去背景或美化。所有三维图使用同一相机，局部图仅裁剪画布。
- 蓝色为原生框，橙色为 PTR，金色为 VAR；HSR 的紫色仅标记被匹配的 proposal，HSR 不改变原生几何。
- 图中只显示选定的 4 个原生框、1 个 PTR 和 1 个 VAR。完整输出为 {len(snapshot.boxes)} 条（各分支数量见 manifest），不可宣称展示全部预测。
- PTR 示意使用真实轨迹 {data['ptr_row']['track_id']} 的三个连续支持关键帧；VAR 使用真实体素 {data['var_row']['voxel_key']} 的三个连续支持关键帧。它们不是第一次确认所需的最早三个关键帧。流程图请标实际帧号或“selected supporting keyframes”，不要误写第一次确认时刻。
- PTR 和 VAR 终态代表框与所选 GT 的三维 AABB IoU 分别为 {data['ptr_iou']:.4f}、{data['var_iou']:.4f}。GT 仅辅助选择可视化案例，不参与算法。
- HSR：原生 ID={identity}，帧 {hframe}，分数 {event['native_score']:.3f} → {event['hsr_score']:.3f}，帧内原生排名 {event['native_rank']} → {event['hsr_rank']}；曲线是真实记录，不是任意绘制的示意数值。
- 原生前端 CuTR 小图来自 20261009 归档诊断；其余状态图来自 20261012 原生前缀与 20261010 冻结候选的共同重放。这里展示模块作用，不宣称是一个端到端运行的逐步骤同帧截屏。
- 各模块使用同一场景中不同关键帧的实例，不是同一个 t 的全流程同步快照。
- 场景网格仅供离线可视化；并非 CaRR-3D 在线构建的模型。仅做已标注墙面及顶棚剖切、光照显示调整，未新增物体或表面。详细来源、剖切规则、框坐标与文件校验值均已保存。

流程图下方建议加：Real observations and selected detections from one ScanNet sequence; module examples use different keyframes. Scene mesh is for visualization only.
不要把 Reliable Views 三张随意 RGB 图称作 PFO 的真实 Top-K 观测：本包未提供可追溯的 PFO 历史选择日志，保留文字或示意即可。
"""
    (OUT/"使用说明.txt").write_text(instructions,encoding="utf-8")
    save_json(OUT/"file_checksums.json",{str(p.relative_to(OUT)):common.sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name!="file_checksums.json"})
    with zipfile.ZipFile(OUT.with_suffix(".zip"),"w",zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p.name != "replay_payload.pkl": archive.write(p,Path(OUT.name)/p.relative_to(OUT))
    print("Exported",OUT)
    print("ZIP",OUT.with_suffix(".zip"))
    print("HSR",event)


def package_recommended():
    audit = json.loads((OUT/"source_manifest.json").read_text())
    selected = {OUT/r["path"] for r in audit["recommended_replacements"]}
    selected.update(OUT/p for p in ["00_all_assets_preview.png","00_scene_comparison.png",
        "使用说明.txt","source_manifest.json","04_ptr/ptr_three_keyframes.png",
        "05_var/var_three_keyframes.png","06_hsr/actual_support_history.pdf",
        "06_hsr/actual_support_history.svg","07_output/carr3d_output_selected_white.png",
        "07_output/scene_mesh.png","08_audit/geometry_and_scores.npz","08_audit/hsr_history.csv"])
    checksums = {str(p.relative_to(OUT)):common.sha(p) for p in sorted(selected)}
    save_json(OUT/"recommended_checksums.json",checksums)
    selected.add(OUT/"recommended_checksums.json")
    destination = OUT.with_name(OUT.name+"_ready_to_use.zip")
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(selected): archive.write(p,Path(OUT.name)/p.relative_to(OUT))
        archive.write(Path(__file__),Path(OUT.name)/"08_audit"/Path(__file__).name)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
    print(destination, "files",len(selected)+1,"bytes",destination.stat().st_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--bold", action="store_true",help="Export heavier diagram-ready strokes to a separate folder.")
    args = parser.parse_args()
    if args.bold:
        OUT = BASE_OUT.with_name(BASE_OUT.name+"_bold")
        STROKES.update(rgb=10,rgb_halo=14,native_3d=16,recovery_3d=22)
    if args.package: package_recommended()
    elif args.probe: probe()
    else:
        render()
        if args.bold: package_recommended()

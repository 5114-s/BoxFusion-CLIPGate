import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
from collections import Counter
import glob
import itertools
import numpy as np
import random
import rerun
import rerun.blueprint as rrb
import yaml
import torch
import torchvision
import sys
import uuid
import open3d as o3d
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from tools.utils import * 
import open_clip 
import torch.nn.functional as F
import time

from boxfusion.cubify_transformer import make_cubify_transformer

from boxfusion.instances import Instances3D
from boxfusion.preprocessor import Augmentor, Preprocessor

from boxfusion.box_manager import BoxManager
from boxfusion.box_fusion import BoxFusion
from boxfusion.boxer_gsa import build_boxer_gsa
from boxfusion.boxer_lifter import build_lifting_adapter
from boxfusion.boxer_mvpr import build_boxer_mvpr, isolated_rng
from boxfusion.edgetam_maskdepth import EdgeTAMMaskDepthProvider
from boxfusion.proposal_cache import build_proposal_cache
from boxfusion.sealed_boxer_proposal_cache import (
    build_proposal_cache as build_sealed_boxer_proposal_cache,
)
from boxfusion.graw_fragments import (
    RawFragmentExtractor,
    aligned_resize_affine as raw_aligned_resize_affine,
)
from boxfusion.graw_shadow import (
    GrawShadow,
    graw_result_to_dict,
    write_graw_shadow_diagnostics,
)
from boxfusion.gclean_shadow import (
    GcleanShadow,
    gclean_result_to_dict,
    write_gclean_shadow_diagnostics,
)
from boxfusion.puf_gclean_shadow import (
    PufGcleanShadow,
    puf_gclean_result_to_dict,
    write_puf_gclean_shadow_diagnostics,
)
from boxfusion.observer_track_adapter import build_observer_track_adapter
from boxfusion.observer_track_registry import IdentityResolution
from boxfusion.smov_fragments import (
    SMOVFragmentExtractor,
    aligned_resize_affine as smov_aligned_resize_affine,
    smov_batch_to_dict,
    write_smov_shadow_diagnostics,
)
from boxfusion.stream3dv2_live import (
    build_stream3dv2_live_route,
    tm_fpf_c1_view_abstention_reason,
)
from boxfusion.stream3dv3_live import build_stream3dv3_live_route
from boxfusion.tm_fpf_c1 import (
    TMFPFC1,
    TMFPFC1ContractError,
    make_target_mask_view,
    match_fastsam_target_masks,
)


def resolve_group3d_shadow_variant(cfg):
    """Resolve the output-inert fragment/association arm strictly."""

    association = cfg.get("association", {})
    if association is None:
        association = {}
    if not hasattr(association, "get"):
        raise ValueError("association configuration must be a mapping")
    section = association.get("group3d_lite", {})
    if section is None:
        section = {}
    if not hasattr(section, "get"):
        raise ValueError("association.group3d_lite must be a mapping")
    variant = section.get("shadow_variant", "graw")
    if not isinstance(variant, str) or variant not in (
        "graw",
        "smov",
        "gclean",
        "puf",
    ):
        raise ValueError(
            "association.group3d_lite.shadow_variant must be "
            "'graw', 'smov', 'gclean', or 'puf'"
        )
    return variant


def run(cfg, model, dataset, clip_model, preprocess, tokenized_text, text_features, augmentor, preprocessor, score_thresh=0.0, viz_on_gt_points=False, gap=25, re_vis=True):
    is_depth_model = "wide/depth" in augmentor.measurement_keys
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            contents=[
                rrb.Horizontal(
                    contents=([
                    rrb.Spatial3DView(
                        name="World",
                        contents=[
                            "+ $origin/**",
                            "+ /device/wide/pred_instances/**",
                            # "+ /world/image/**"
                        ],
                        origin="/world"),
                    ])),
                rrb.Horizontal(
                    contents=([
                        rrb.Spatial2DView(
                            name="Image",
                            origin="/device/wide/image",
                            contents=[
                                "+ $origin/**",
                                "+ /device/wide/pred_instances/**"
                            ])
                    ] + ([
                        # Only show this for RGB-D.
                        rrb.Spatial2DView(
                            name="Depth",
                            origin="/device/wide/depth")
                    ] if is_depth_model else [])),
                    name="Wide")
            ]))

    recording = None
    video_id = None

    device = model.pixel_mean
    lifting_adapter = build_lifting_adapter(
        cfg,
        device=str(device.device),
        code_root=str(Path(__file__).resolve().parent),
    )
    proposal_cache_builder = (
        build_sealed_boxer_proposal_cache
        if str(cfg.get("lifting", {}).get("backend", "cutr")).lower() == "boxer"
        else build_proposal_cache
    )
    proposal_cache = proposal_cache_builder(cfg, device=device.device)
    boxer_mvpr = build_boxer_mvpr(cfg, device=device.device)
    boxer_gsa = build_boxer_gsa(cfg, device=device.device)
    if boxer_mvpr is not None and boxer_gsa is not None:
        raise ValueError("Boxer-MVPR and Boxer-GSA are mutually exclusive")
    proposal_cache_scene_id = None
    stream3dv2_live = build_stream3dv2_live_route(
        cfg,
        lifting_adapter=lifting_adapter,
        device=str(device.device),
    )
    stream3dv3_live = build_stream3dv3_live_route(
        cfg,
        lifting_adapter=lifting_adapter,
        device=str(device.device),
    )
    if stream3dv2_live is not None and stream3dv3_live is not None:
        raise ValueError("Stream3Dv2 and Stream3Dv3 are mutually exclusive")
    strict_live_route = (
        stream3dv3_live if stream3dv3_live is not None else stream3dv2_live
    )
    tm_fpf_c1 = TMFPFC1(cfg.get("box_fusion", {}))
    if tm_fpf_c1.enabled:
        if stream3dv2_live is None or stream3dv3_live is not None:
            raise ValueError(
                "TM-FPF-C1 requires the Stream3Dv2 lightweight route"
            )
        if not stream3dv2_live.config.lightweight_enabled:
            raise ValueError("TM-FPF-C1 requires lightweight Stream3Dv2")
        if not stream3dv2_live.config.depth_trigger_enabled:
            raise ValueError("TM-FPF-C1 requires the lightweight depth trigger")
    tm_fpf_c1_views_by_init_id = {}
    tm_fpf_c1_native_init_ids = set()
    tm_fpf_c1_collection_abstentions = Counter()

    def proposal_cache_inputs(sample, image):
        return {
            "image": image,
            "depth": sample["wide"]["depth"][-1],
            "image_K": sample["sensor_info"].wide.image.K[-1],
            "depth_K": sample["sensor_info"].gt.depth.K[-1],
            "camera_to_world": sample["sensor_info"].gt.RT[-1],
        }

    def apply_lifting_if_configured(
        pred_instances,
        sample,
        image,
        scene_id,
        frame_id,
        stage,
        attempt_id,
    ):
        if lifting_adapter is None or lifting_adapter.apply_stage != stage:
            return pred_instances
        return lifting_adapter.apply(
            pred_instances,
            image=image,
            depth=sample["wide"]["depth"][-1],
            image_K=sample["sensor_info"].wide.image.K[-1],
            depth_K=sample["sensor_info"].gt.depth.K[-1],
            camera_to_world=sample["sensor_info"].gt.RT[-1],
            scene_id=scene_id,
            frame_id=frame_id,
            attempt_id=attempt_id,
        )

    count=0
    all_pred_box = None
    all_poses = None

    all_kf_pose = {}
    per_frame_ins = None #save every predicted boxes
    traj_xyz = []

    box_manager = BoxManager(cfg)
    Box_Fuser = BoxFusion(cfg)
    edgetam_maskdepth = EdgeTAMMaskDepthProvider(cfg["box_fusion"])
    terminal_clip_enabled = bool(
        stream3dv2_live is not None
        and stream3dv2_live.config.terminal_clip_enabled
    )
    if terminal_clip_enabled and (
        cfg.get("association", {}).get("appearance_gate", {}).get(
            "enabled", False
        )
        or box_manager.causal_hungarian.needs_appearance
    ):
        raise ValueError(
            "Terminal-batch CLIP cannot replace features required by association"
        )
    terminal_clip_crops = {}

    def cache_terminal_clip_crops(instances, rgb):
        if not terminal_clip_enabled or len(instances) == 0:
            return
        boxes = scale_boxes(
            instances.pred_boxes.detach().cpu().numpy(),
            rgb.shape[0],
            rgb.shape[1],
            scale=cfg['detection'].get('scale_box', 1.5),
        )
        image_pil = Image.fromarray(rgb)
        init_ids = instances.init_id.detach().cpu().tolist()
        for init_id, box in zip(init_ids, boxes):
            x1, y1, x2, y2 = (int(value) for value in box)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Terminal CLIP received an empty proposal crop")
            terminal_clip_crops[int(init_id)] = image_pil.crop((x1, y1, x2, y2))

    def prune_terminal_clip_crops(instances):
        if not terminal_clip_enabled or instances is None:
            return
        active = {
            int(value)
            for value in instances.init_id.detach().cpu().tolist()
        }
        for init_id in tuple(terminal_clip_crops):
            if init_id not in active:
                del terminal_clip_crops[init_id]

    def run_terminal_clip(instances):
        if not terminal_clip_enabled or instances is None or len(instances) == 0:
            return
        init_ids = [
            int(value)
            for value in instances.init_id.detach().cpu().tolist()
        ]
        missing = [value for value in init_ids if value not in terminal_clip_crops]
        if missing:
            raise RuntimeError(
                "Terminal CLIP crop cache is incomplete for active objects: "
                f"{missing[:8]}"
            )
        batch_size = stream3dv2_live.config.terminal_clip_batch_size
        categories = []
        started = time.perf_counter()
        batches = 0
        for start in range(0, len(init_ids), batch_size):
            crops = [
                terminal_clip_crops[value]
                for value in init_ids[start : start + batch_size]
            ]
            scores, _ = retriev(
                clip_model,
                preprocess,
                crops,
                text_features,
                device=str(device.device),
            )
            max_ids = torch.max(scores, dim=-1).indices.detach().cpu().numpy()
            categories.extend(tokenized_text[max_ids].tolist())
            batches += 1
        instances.categories = np.asarray(categories)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        stream3dv2_live.record_terminal_clip(
            proposal_count=len(init_ids),
            batch_count=batches,
            elapsed_ms=elapsed_ms,
        )
        print(
            "Terminal batch CLIP:",
            f"proposals={len(init_ids)}",
            f"batches={batches}",
            f"elapsed_ms={elapsed_ms:.3f}",
        )

    def cache_tm_fpf_c1_views(instances, mask_frame):
        """Bind same-frame automatic masks to immutable world-frame views."""

        if not tm_fpf_c1.enabled or len(instances) == 0:
            return
        init_ids = tuple(
            int(value) for value in instances.init_id.detach().cpu().tolist()
        )
        duplicate_ids = set(init_ids) & tm_fpf_c1_native_init_ids
        if duplicate_ids:
            raise RuntimeError(
                "TM-FPF-C1 received duplicate native init_id values: "
                f"{sorted(duplicate_ids)[:8]}"
            )
        tm_fpf_c1_native_init_ids.update(init_ids)
        # A depth-trigger abstention or non-upright keyframe has no automatic
        # masks by contract.  Its native rows remain valid, but contribute no
        # TM-FPF observation and therefore cause a terminal abstention unless
        # other associated frames provide enough evidence.
        if mask_frame is None:
            return
        if mask_frame.scene_id != scene_id or mask_frame.frame_id != count:
            raise RuntimeError("TM-FPF-C1 mask frame identity is misaligned")
        native_boxes = instances.pred_boxes.detach().cpu().numpy()
        if (
            mask_frame.native_boxes_xyxy.shape != native_boxes.shape
            or not np.array_equal(mask_frame.native_boxes_xyxy, native_boxes)
        ):
            raise RuntimeError("TM-FPF-C1 native rows changed after FastSAM")
        matched = match_fastsam_target_masks(
            native_boxes_xyxy=native_boxes,
            automatic_masks=mask_frame.masks,
            automatic_boxes_xyxy=mask_frame.automatic_boxes_xyxy,
            automatic_confidences=mask_frame.automatic_confidences,
            config=tm_fpf_c1.config,
        )
        boxes_world = instances.pred_boxes_3d.tensor.detach().cpu().numpy()
        rotations_world = instances.pred_boxes_3d.R.detach().cpu().numpy()
        for row_index, mask_index in enumerate(matched):
            if mask_index is None:
                continue
            init_id = init_ids[row_index]
            try:
                view = make_target_mask_view(
                    source_id=(
                        f"{scene_id}/frame_{count:06d}/native_{init_id:06d}"
                    ),
                    frame_id=count,
                    observation_box_xyzlhw=boxes_world[row_index],
                    observation_rotation=rotations_world[row_index],
                    target_mask=mask_frame.masks[mask_index],
                    depth_m=mask_frame.depth_m,
                    intrinsics=mask_frame.intrinsics,
                    camera_to_world=mask_frame.camera_to_world,
                    config=tm_fpf_c1.config,
                )
            except TMFPFC1ContractError as error:
                abstention_reason = tm_fpf_c1_view_abstention_reason(error)
                if abstention_reason is None:
                    raise
                tm_fpf_c1_collection_abstentions[abstention_reason] += 1
                continue
            if init_id in tm_fpf_c1_views_by_init_id:
                raise RuntimeError(
                    f"TM-FPF-C1 init_id {init_id} already owns an observation"
                )
            tm_fpf_c1_views_by_init_id[init_id] = view

    def run_terminal_tm_fpf_c1(instances):
        """Refine native terminal geometry without mutating online state."""

        if not tm_fpf_c1.enabled:
            return instances.pred_boxes_3d
        if len(box_manager.fusion_list) != len(instances):
            raise RuntimeError(
                "TM-FPF-C1 terminal tracks do not align with native rows"
            )
        track_views = []
        for fusion_ids in box_manager.fusion_list:
            row_views = []
            for raw_init_id in fusion_ids:
                init_id = int(raw_init_id)
                if init_id not in tm_fpf_c1_native_init_ids:
                    raise RuntimeError(
                        "TM-FPF-C1 fusion_list references an unknown init_id: "
                        f"{init_id}"
                    )
                view = tm_fpf_c1_views_by_init_id.get(init_id)
                if view is not None:
                    row_views.append(view)
            track_views.append(tuple(row_views))

        native_boxes = instances.pred_boxes_3d.tensor.detach().cpu().numpy()
        native_rotations = instances.pred_boxes_3d.R.detach().cpu().numpy()
        native_scores = instances.scores.detach().cpu().numpy()
        result = tm_fpf_c1.refine_terminal(
            boxes_xyzlhw=native_boxes,
            rotations=native_rotations,
            scores=native_scores,
            track_views=tuple(track_views),
        )
        if result.online_writeback:
            raise RuntimeError("TM-FPF-C1 attempted forbidden online writeback")
        if not np.array_equal(result.rotations, native_rotations):
            raise RuntimeError("TM-FPF-C1 changed native rotations or row order")
        if not np.array_equal(result.scores, native_scores):
            raise RuntimeError("TM-FPF-C1 changed native real scores or row order")
        if result.boxes_xyzlhw.shape != native_boxes.shape:
            raise RuntimeError("TM-FPF-C1 changed native row count")

        terminal_geometry = instances.pred_boxes_3d.clone()
        refined_tensor = torch.as_tensor(
            np.array(result.boxes_xyzlhw, copy=True),
            dtype=terminal_geometry.tensor.dtype,
            device=terminal_geometry.tensor.device,
        )
        terminal_geometry.tensor.copy_(refined_tensor)
        if not torch.equal(terminal_geometry.R, instances.pred_boxes_3d.R):
            raise RuntimeError("TM-FPF-C1 terminal copy changed rotations")
        reasons = Counter(row.reason for row in result.decisions if not row.accepted)
        reason_text = ",".join(
            f"{reason}:{reasons[reason]}" for reason in sorted(reasons)
        )
        collection_text = ",".join(
            f"{reason}:{tm_fpf_c1_collection_abstentions[reason]}"
            for reason in sorted(tm_fpf_c1_collection_abstentions)
        )
        print(
            "TM-FPF-C1 terminal |",
            f"native={len(instances)}",
            f"views={len(tm_fpf_c1_views_by_init_id)}",
            f"accepted={result.accepted_count}",
            f"abstained={len(instances) - result.accepted_count}",
            f"reasons={reason_text or 'none'}",
            f"view_abstentions={collection_text or 'none'}",
        )
        return terminal_geometry

    observer_adapter = None
    observer_scene_id = None
    shadow_variant = resolve_group3d_shadow_variant(cfg)
    graw_extractor = None
    graw_shadow = None
    graw_frame_records = []
    smov_extractor = None
    smov_frame_records = []
    gclean_shadow = None
    gclean_frame_records = []
    puf_gclean_shadow = None
    puf_gclean_frame_records = []

    def observer_native_fields(current_predictions=None):
        """Borrow only native arrays/lists that an observer must not change."""

        fields = {
            "all_poses": all_poses,
            "fusion_list": box_manager.fusion_list,
            "last_fusion_frame": box_manager.last_fusion_frame,
            "fusion_flag": box_manager.fusion_flag,
            "already_fusion": box_manager.already_fusion,
            "num_record": tuple(
                sorted((int(key), int(value)) for key, value in box_manager.num_record.items())
            ),
            "merge_log": box_manager.merge_log,
        }

        def add_instances(prefix, instances):
            if instances is None:
                return
            for name in (
                "scores",
                "init_id",
                "valid_num",
                "frame_id",
                "pred_boxes",
                "appearance_features",
            ):
                if instances.has(name):
                    fields[prefix + "." + name] = instances.get(name)
            if instances.has("categories"):
                fields[prefix + ".categories"] = np.asarray(
                    instances.get("categories")
                )
            if instances.has("pred_boxes_3d"):
                boxes_3d = instances.get("pred_boxes_3d")
                fields[prefix + ".boxes3d.tensor"] = boxes_3d.tensor
                fields[prefix + ".boxes3d.R"] = boxes_3d.R

        add_instances("all", all_pred_box)
        add_instances("history", per_frame_ins)
        add_instances("current", current_predictions)
        return fields

    def finalize_group3d_shadow_frame(
        current_predictions,
        observer_token,
        shadow_token,
        prepared_fragment,
        attempt_id,
    ):
        if observer_token is None or observer_adapter is None:
            return
        trace = None
        result = None
        with observer_adapter.observer_boundary(
            observer_native_fields(current_predictions),
            label=f"finish_keyframe:{observer_token.frame_id}",
        ):
            trace = observer_adapter.finalize(box_manager, observer_token)
            if trace is not None and shadow_token is not None and prepared_fragment is not None:
                resolution = IdentityResolution(
                    frame_id=trace.frame_id,
                    proposal_ids=trace.proposal_ids,
                    proposal_track_ids=trace.proposal_track_ids,
                    active_track_ids=trace.active_track_ids,
                    track_aliases=trace.track_aliases,
                )
                if shadow_variant == "graw":
                    result = graw_shadow.finish_keyframe(
                        shadow_token,
                        batch=prepared_fragment,
                        resolution=resolution,
                        reserved_past_track_ids=trace.reserved_past_track_ids,
                    )
                elif shadow_variant == "gclean":
                    result = gclean_shadow.finish_keyframe(
                        shadow_token,
                        batch=prepared_fragment,
                        resolution=resolution,
                        reserved_past_track_ids=trace.reserved_past_track_ids,
                        unmatched_retained_proposal_ids=(
                            trace.native_unmatched_retained_proposal_ids
                        ),
                    )
                elif shadow_variant == "puf":
                    result = puf_gclean_shadow.finish_keyframe(
                        shadow_token,
                        batch=prepared_fragment,
                        resolution=resolution,
                        reserved_past_track_ids=trace.reserved_past_track_ids,
                        unmatched_retained_proposal_ids=(
                            trace.native_unmatched_retained_proposal_ids
                        ),
                    )
        if shadow_variant == "graw":
            if shadow_token is not None and graw_shadow.pending:
                graw_shadow.abort_keyframe(shadow_token)
        elif shadow_variant == "gclean":
            if shadow_token is not None and gclean_shadow.pending:
                gclean_shadow.abort_keyframe(shadow_token)
        elif shadow_variant == "puf":
            if shadow_token is not None and puf_gclean_shadow.pending:
                puf_gclean_shadow.abort_keyframe(shadow_token)
        if (
            prepared_fragment is not None
            and shadow_variant in ("smov", "gclean", "puf")
            and observer_adapter.trace_valid
        ):
            smov_record = smov_batch_to_dict(prepared_fragment)
            smov_record["source_attempt_id"] = str(attempt_id)
            smov_frame_records.append(smov_record)
        if result is not None and observer_adapter.trace_valid and shadow_variant == "graw":
            record = graw_result_to_dict(
                result,
                raw_prepare_elapsed_ms=prepared_fragment.elapsed_ms,
            )
            record["source_attempt_id"] = str(attempt_id)
            graw_frame_records.append(record)
        elif result is not None and observer_adapter.trace_valid and shadow_variant == "gclean":
            record = gclean_result_to_dict(result)
            record["source_attempt_id"] = str(attempt_id)
            gclean_frame_records.append(record)
        elif result is not None and observer_adapter.trace_valid and shadow_variant == "puf":
            record = puf_gclean_result_to_dict(result)
            record["source_attempt_id"] = str(attempt_id)
            puf_gclean_frame_records.append(record)

    box_count = 0
    start_time = time.time()
    if stream3dv3_live is not None:
        stream3dv3_live.start_pipeline_clock(
            dataset_frame_count=len(dataset),
            keyframe_gap=gap,
        )
    
    
    
    for sample in dataset:
        sample_video_id = sample["meta"]["video_id"] #(['sensor_info', 'wide', 'gt', 'meta'])
        scene_id = (
            str(sample_video_id[0])
            if isinstance(sample_video_id, (list, tuple, np.ndarray))
            else str(sample_video_id)
        )
        if observer_adapter is None:
            observer_adapter = build_observer_track_adapter(
                cfg, scene_id=scene_id
            )
            observer_scene_id = scene_id
            if observer_adapter.config.mode == "shadow":
                if shadow_variant == "graw":
                    graw_extractor = RawFragmentExtractor()
                    graw_shadow = GrawShadow()
                else:
                    smov_extractor = SMOVFragmentExtractor()
                    if shadow_variant == "gclean":
                        gclean_shadow = GcleanShadow()
                    elif shadow_variant == "puf":
                        puf_gclean_shadow = PufGcleanShadow()
        elif observer_scene_id != scene_id:
            raise ValueError(
                "One BoxFusion run cannot mix observer scenes: "
                f"{observer_scene_id} != {scene_id}"
            )
        if observer_adapter.enabled and count % gap != 0:
            observer_adapter.mark_non_keyframe(count)
        if proposal_cache is not None:
            if proposal_cache_scene_id is None:
                proposal_cache.bind_scene(
                    scene_id,
                    dataset_length=len(dataset),
                    gap=gap,
                )
                proposal_cache_scene_id = scene_id
            elif proposal_cache_scene_id != scene_id:
                raise ValueError(
                    "One BoxFusion run cannot mix proposal-cache scenes: "
                    f"{proposal_cache_scene_id} != {scene_id}"
                )
        if boxer_mvpr is not None and boxer_mvpr.scene_id is None:
            boxer_mvpr.bind_scene(
                scene_id,
                dataset_length=len(dataset),
                gap=gap,
            )
        if boxer_gsa is not None and boxer_gsa.scene_id is None:
            boxer_gsa.bind_scene(
                scene_id,
                dataset_length=len(dataset),
                gap=gap,
            )
        pose = sample['sensor_info'].gt.RT
        
        video_id = sample_video_id
        if ((recording is None) or (video_id != sample_video_id)) and re_vis:
            new_recording = rerun.new_recording(
                application_id=str(sample_video_id), recording_id=uuid.uuid4(), make_default=True)
            new_recording.send_blueprint(blueprint, make_active=True)
            rerun.spawn()
            recording = new_recording
        
        pose_np = pose.squeeze().cpu().numpy()

        if re_vis:
            rerun.set_time_seconds("pts", sample["meta"]["timestamp"], recording=recording)

        # -> channels last.
        image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)  #[H,W,3]
        if strict_live_route is not None:
            strict_live_route.bind_scene(scene_id)
            strict_live_route.poll(count)

        if re_vis:
            color_camera = rerun.Pinhole(
                image_from_camera=sample["sensor_info"].wide.image.K[-1].numpy(), resolution=sample["sensor_info"].wide.image.size)

        if is_depth_model and re_vis:
            # Show the depth being sent to the model.            
            depth_camera = rerun.Pinhole(
                image_from_camera=sample["sensor_info"].wide.depth.K[-1].numpy(), resolution=sample["sensor_info"].wide.depth.size)

        if Box_Fuser.update_K_flag == False:
            Box_Fuser.update_intrinsics(sample["sensor_info"].wide.image.size,sample["sensor_info"].wide.image.K[-1].numpy()) #size:[W,H]

        xyzrgb = None
        if re_vis and viz_on_gt_points and sample["sensor_info"].has("gt"):
            # Backproject GT depth to world so we can compare our predictions.
            depth_gt = sample["wide"]["depth"][-1]
            matched_image = torch.tensor(np.array(Image.fromarray(image).resize((depth_gt.shape[1], depth_gt.shape[0]))))
            # Feel free to change max_depth, but know CA is only trained up to 5m.
            xyz, valid = unproject(depth_gt, sample["sensor_info"].gt.depth.K[-1], pose.squeeze(), max_depth=10.0)
            xyzrgb = torch.cat((xyz, matched_image / 255.0), dim=-1)[valid]            
                    
        # Every gap nth frame is selected as keyframe
        native_target_mask_frame = None
        if count % gap == 0:
            if proposal_cache is not None and proposal_cache.is_replay:
                pred_instances, source_attempt_id = proposal_cache.replay(
                    scene_id,
                    count,
                    inputs=proposal_cache_inputs(sample, image),
                )
                pred_instances = apply_lifting_if_configured(
                    pred_instances,
                    sample,
                    image,
                    scene_id,
                    count,
                    "post_filter",
                    source_attempt_id,
                )
                if boxer_mvpr is not None:
                    low_instances, _ = boxer_mvpr.replay_low_candidates(
                        scene_id,
                        count,
                        inputs=proposal_cache_inputs(sample, image),
                        native_instances=pred_instances,
                    )
                    with isolated_rng(device.device):
                        low_instances = apply_lifting_if_configured(
                            low_instances,
                            sample,
                            image,
                            scene_id,
                            count,
                            "post_filter",
                            "mvpr",
                        )
                    recovered_instances = boxer_mvpr.recover(
                        count,
                        low_instances,
                        camera_to_world=sample["sensor_info"].gt.RT[-1],
                    )
                    if len(recovered_instances) > 0:
                        pred_instances = Instances3D.cat(
                            [pred_instances, recovered_instances]
                        )
                    print(
                        "Boxer-MVPR:",
                        f"frame={count}",
                        f"low={len(low_instances)}",
                        f"recovered={len(recovered_instances)}",
                    )
                if boxer_gsa is not None:
                    low_instances, _ = boxer_gsa.replay_low_candidates(
                        scene_id,
                        count,
                        inputs=proposal_cache_inputs(sample, image),
                        native_instances=pred_instances,
                    )
                    with isolated_rng(device.device):
                        low_instances = apply_lifting_if_configured(
                            low_instances,
                            sample,
                            image,
                            scene_id,
                            count,
                            "post_filter",
                            "gsa",
                        )
                    recovered_instances = boxer_gsa.recover(
                        count,
                        low_instances,
                        camera_to_world=sample["sensor_info"].gt.RT[-1],
                    )
                    if len(recovered_instances) > 0:
                        pred_instances = Instances3D.cat(
                            [pred_instances, recovered_instances]
                        )
                    print(
                        "Boxer-GSA:",
                        f"frame={count}",
                        f"low={len(low_instances)}",
                        f"recovered={len(recovered_instances)}",
                    )
            else:
                # CuTR consumes keyframes only.  Keep packaging, device transfer
                # and depth standardization inside that same boundary so the
                # other gap-1 raw frames do not pay an output-inert cost.
                packaged = augmentor.package(sample)
                packaged = move_input_to_current_device(packaged, device)
                packaged = preprocessor.preprocess([packaged])
                source_attempt_id = "primary"
                with torch.no_grad():
                    pred_instances = model(packaged)[0]

                pred_instances = pred_instances[
                    pred_instances.scores >= float(score_thresh)
                ]
                pred_instances = apply_lifting_if_configured(
                    pred_instances,
                    sample,
                    image,
                    scene_id,
                    count,
                    "pre_filter",
                    source_attempt_id,
                )

                if cfg["detection"]["uv_bound"]:
                    uv_mask = box_manager.check_uv_bounds(
                        pred_instances.pred_proj_xy,
                        image.shape[1],
                        image.shape[0],
                        ratio=cfg["detection"]["uv_bound_value"],
                    )
                    pred_instances = pred_instances[uv_mask]
                if cfg["detection"]["floor_mask"]:
                    floor_mask = box_manager.check_floor_mask(
                        pred_instances.pred_boxes_3d.tensor,
                        ratio=cfg["detection"]["floor_ratio"],
                    )
                    pred_instances = pred_instances[~floor_mask]
                pred_instances = apply_lifting_if_configured(
                    pred_instances,
                    sample,
                    image,
                    scene_id,
                    count,
                    "post_filter",
                    source_attempt_id,
                )

                # Preserve the released frame-0 retry contract: lower score
                # threshold plus UV filtering, but no floor filtering.
                if len(pred_instances) == 0 and count == 0:
                    source_attempt_id = "retry"
                    with torch.no_grad():
                        pred_instances = model(packaged)[0]
                    pred_instances = pred_instances[
                        pred_instances.scores
                        >= float(cfg["detection"]["score_thresh"] / 4)
                    ]
                    pred_instances = apply_lifting_if_configured(
                        pred_instances,
                        sample,
                        image,
                        scene_id,
                        count,
                        "pre_filter",
                        source_attempt_id,
                    )
                    print("again", count, "pred_instances", len(pred_instances))
                    if cfg["detection"]["uv_bound"]:
                        uv_mask = box_manager.check_uv_bounds(
                            pred_instances.pred_proj_xy,
                            image.shape[1],
                            image.shape[0],
                            ratio=cfg["detection"]["uv_bound_value"],
                        )
                        pred_instances = pred_instances[uv_mask]
                    pred_instances = apply_lifting_if_configured(
                        pred_instances,
                        sample,
                        image,
                        scene_id,
                        count,
                        "post_filter",
                        source_attempt_id,
                    )
                    print("again", count, "pred_instances", len(pred_instances))

                # Cache boundary: ordered camera-frame CuTR rows after score,
                # UV, floor and frame-0 retry filtering, before any observer,
                # CLIP, world transform, association or fusion.
                if proposal_cache is not None and proposal_cache.is_record:
                    pred_instances = proposal_cache.record(
                        scene_id,
                        count,
                        pred_instances,
                        attempt_id=source_attempt_id,
                        inputs=proposal_cache_inputs(sample, image),
                    )

            if strict_live_route is not None:
                native_target_mask_frame = strict_live_route.process_keyframe(
                    scene_id=scene_id,
                    frame_id=count,
                    rgb=image,
                    depth_m=sample["wide"]["depth"][-1],
                    intrinsics=sample["sensor_info"].gt.depth.K[-1],
                    camera_to_world=sample["sensor_info"].gt.RT[-1],
                    native_boxes_xyxy=pred_instances.pred_boxes.detach().cpu().numpy(),
                )

        # Hold off on logging anything until now, since the delay might confuse the user in the visualizer.
        RT = sample["sensor_info"].gt.RT[-1].numpy()
        if re_vis:
            pose_transform = rerun.Transform3D(
                translation=RT[:3, 3],
                rotation=rerun.Quaternion(xyzw=Rotation.from_matrix(RT[:3, :3]).as_quat()))
            rerun.log("/world/image", pose_transform)
            rerun.log("/world/image", color_camera)

            rerun.log("/device/wide/image", pose_transform)
            rerun.log("/device/wide/image", rerun.Image(image).compress())
            rerun.log("/device/wide/image", color_camera)
        traj_xyz.append(RT[:3, 3])
            

        if is_depth_model and re_vis:
            rerun.log("/device/wide/depth", rerun.DepthImage(sample["wide"]["depth"][-1].numpy()))
            rerun.log("/device/wide/depth", depth_camera)
        
        if xyzrgb is not None and re_vis:
            rerun.log("/world/xyz", rerun.Points3D(positions=xyzrgb[..., :3], colors=xyzrgb[..., 3:], radii=None))        

        # visualize the trajectory
        if cfg["vis"]["trajectory"] and re_vis:
            rerun.log("/world/trajectory", rerun.LineStrips3D([np.array(traj_xyz)[:count]], colors=[84,255,159]))

        observer_frame_token = None
        shadow_frame_token = None
        fragment_keyframe = None

        # only process keyframes
        if count % gap ==0 or count == len(dataset)-1:
            
            all_kf_pose[count] = pose_np
            pose_np = np.expand_dims(pose_np,axis=0)
            pose_np = np.repeat(pose_np, repeats=len(pred_instances), axis=0) 
            
            if len(pred_instances)==0:
                if observer_adapter.enabled and count % gap == 0:
                    with observer_adapter.observer_boundary(
                        observer_native_fields(pred_instances),
                        label=f"empty_keyframe:{count}",
                    ):
                        empty_token = observer_adapter.begin_keyframe(count, ())
                        observer_adapter.finalize(box_manager, empty_token)
                all_pred_box = all_pred_box
                all_poses = all_poses
                box_count += len(pred_instances)
                box_manager.num_record[count] = box_count
                count+=1
                continue
            
            # add new properties for Instance3D predictions
            pred_instances.categories = np.array(['None'] * len(pred_instances)) # Initialize category labels as 'None' for all predicted instances
            pred_instances.cam_pose = torch.from_numpy(pose_np) # Convert camera pose from numpy to tensor and assign to instances
            pred_instances.frame_id = torch.tensor([count]).repeat(pose_np.shape[0]) # Assign current frame ID to all instances in this frame
            pred_instances.init_id = box_count+torch.arange(len(pred_instances)) # Create unique initial IDs for each instance based on global box count
            pred_instances.valid_num = torch.zeros(len(pred_instances)) # Initialize validation counter to zero for all instances
            cache_terminal_clip_crops(pred_instances, image)
            pred_instances.pred_boxes_3d.transform2world(pred_instances.cam_pose) # Transform 3D bounding boxes from camera coordinates to world coordinates
            cache_tm_fpf_c1_views(pred_instances, native_target_mask_frame)
            pred_instances.project_3d_boxes(sample["sensor_info"].wide.depth.K[-1].numpy(), H=image.shape[0],W=image.shape[1]) # Project 3D boxes to 2D image coordinates using camera intrinsics
            if Box_Fuser.capf.enabled:
                Box_Fuser.capf.attach_observations(
                    pred_instances,
                    depth_m=sample["wide"]["depth"][-1],
                    intrinsics=sample["sensor_info"].gt.depth.K[-1],
                    image_height=image.shape[0],
                    image_width=image.shape[1],
                    camera_to_world=sample["sensor_info"].gt.RT[-1],
                )
            if Box_Fuser.vapf_lite.enabled:
                Box_Fuser.vapf_lite.attach_observations(
                    pred_instances,
                    depth_m=sample["wide"]["depth"][-1],
                    image_height=image.shape[0],
                    image_width=image.shape[1],
                    camera_to_world=sample["sensor_info"].gt.RT[-1],
                )
            if edgetam_maskdepth.enabled:
                depth_frame = sample["wide"]["depth"][-1]
                depth_np = (
                    depth_frame.detach().cpu().numpy()
                    if torch.is_tensor(depth_frame)
                    else np.asarray(depth_frame)
                )
                edgetam_maskdepth.attach(
                    pred_instances,
                    image_rgb=image,
                    depth_m=depth_np,
                    intrinsics=sample["sensor_info"].gt.depth.K[-1].numpy(),
                    camera_to_world=sample["sensor_info"].gt.RT[-1].numpy(),
                )
            appearance_gate_cfg = cfg.get('association', {}).get(
                'appearance_gate', {}
            )
            appearance_gate_enabled = appearance_gate_cfg.get(
                'enabled', False
            )
            association_features_enabled = (
                appearance_gate_enabled
                or box_manager.causal_hungarian.needs_appearance
            )
            if association_features_enabled:
                # Extract once per new proposal. The normalized image feature is
                # retained as instance appearance and reused only after geometry
                # has produced a plausible association candidate.
                boxes = pred_instances.pred_boxes.cpu().numpy()
                boxes = scale_boxes(
                    boxes,
                    image.shape[0],
                    image.shape[1],
                    scale=cfg['detection'].get('scale_box', 1.5),
                )
                class_results, box_features = text_prompt(
                    boxes,
                    tokenized_text,
                    text_features,
                    image,
                    clip_model,
                    preprocess,
                )
                pred_instances.categories = class_results
                pred_instances.appearance_features = (
                    box_features.detach().float().cpu()
                )

            if observer_adapter.enabled and count % gap == 0:
                with observer_adapter.observer_boundary(
                    observer_native_fields(pred_instances),
                    label=f"begin_keyframe:{count}",
                ):
                    begin_active_track_ids = observer_adapter.active_track_ids
                    if shadow_variant == "graw":
                        shadow_frame_token = graw_shadow.begin_keyframe(
                            count,
                            active_track_ids=begin_active_track_ids,
                        )
                    elif shadow_variant == "gclean":
                        shadow_frame_token = gclean_shadow.begin_keyframe(
                            count,
                            active_track_ids=begin_active_track_ids,
                        )
                    elif shadow_variant == "puf":
                        shadow_frame_token = puf_gclean_shadow.begin_keyframe(
                            count,
                            active_track_ids=begin_active_track_ids,
                        )
                    depth_frame = sample["wide"]["depth"][-1]
                    depth_np = (
                        depth_frame.detach().cpu().numpy()
                        if torch.is_tensor(depth_frame)
                        else np.asarray(depth_frame)
                    )
                    depth_intrinsics = sample["sensor_info"].gt.depth.K[-1]
                    depth_intrinsics_np = (
                        depth_intrinsics.detach().cpu().numpy()
                        if torch.is_tensor(depth_intrinsics)
                        else np.asarray(depth_intrinsics)
                    )
                    camera_to_world = sample["sensor_info"].gt.RT[-1]
                    camera_to_world_np = (
                        camera_to_world.detach().cpu().numpy()
                        if torch.is_tensor(camera_to_world)
                        else np.asarray(camera_to_world)
                    )
                    fragment_extractor = (
                        graw_extractor
                        if shadow_variant == "graw"
                        else smov_extractor
                    )
                    registration = (
                        raw_aligned_resize_affine(
                            (image.shape[0], image.shape[1]), depth_np.shape
                        )
                        if shadow_variant == "graw"
                        else smov_aligned_resize_affine(
                            (image.shape[0], image.shape[1]), depth_np.shape
                        )
                    )
                    fragment_keyframe = fragment_extractor.prepare_keyframe(
                        scene_id=scene_id,
                        frame_id=count,
                        proposal_ids=pred_instances.init_id.detach().cpu().numpy(),
                        boxes_xyxy=pred_instances.pred_boxes.detach().cpu().numpy(),
                        proposal_scores=pred_instances.scores.detach().cpu().numpy(),
                        proposal_image_shape=(image.shape[0], image.shape[1]),
                        proposal_to_depth_affine=registration,
                        depth_m=depth_np,
                        intrinsics=depth_intrinsics_np,
                        camera_to_world=camera_to_world_np,
                    )
                    observer_frame_token = observer_adapter.begin_keyframe(
                        count,
                        tuple(
                            int(value)
                            for value in pred_instances.init_id.detach().cpu().tolist()
                        ),
                    )
                if (
                    observer_frame_token is None
                    or fragment_keyframe is None
                    or not observer_adapter.trace_valid
                ):
                    if (
                        shadow_variant == "graw"
                        and shadow_frame_token is not None
                        and graw_shadow.pending
                    ):
                        graw_shadow.abort_keyframe(shadow_frame_token)
                    elif (
                        shadow_variant == "gclean"
                        and shadow_frame_token is not None
                        and gclean_shadow.pending
                    ):
                        gclean_shadow.abort_keyframe(shadow_frame_token)
                    elif (
                        shadow_variant == "puf"
                        and shadow_frame_token is not None
                        and puf_gclean_shadow.pending
                    ):
                        puf_gclean_shadow.abort_keyframe(shadow_frame_token)
                    shadow_frame_token = None

            # record how many boxes each keyframe has, so we know which box belongs to which frame
            box_count += len(pred_instances)
            box_manager.num_record[count] = box_count
 
            # first keyframe, initialize some data structures
            if all_pred_box is None and count<gap:
                
                #predict the semantic classes
                if not association_features_enabled and not terminal_clip_enabled:
                    boxes = pred_instances.pred_boxes.cpu().numpy()
                    #scale the boxes by
                    boxes = scale_boxes(boxes,image.shape[0],image.shape[1],scale=cfg['detection'].get('scale_box', 1.5))

                    class_results, box_features = text_prompt(boxes, tokenized_text, text_features, image, clip_model, preprocess) #[N_box]
                    pred_instances.categories = class_results

                all_pred_box = pred_instances
                all_poses = pose_np
                per_frame_ins = pred_instances
 
                #record the current frame boxes info
                box_manager.init_new_predictions(len(pred_instances),0)
                observer_adapter.attach(box_manager, observer_frame_token)
                finalize_group3d_shadow_frame(
                    pred_instances,
                    observer_frame_token,
                    shadow_frame_token,
                    fragment_keyframe,
                    source_attempt_id,
                )

            else:
                
                box_manager.init_new_predictions(len(pred_instances),len(per_frame_ins))
                observer_adapter.attach(box_manager, observer_frame_token)

                num_before_cat = len(all_pred_box)
                cur_global_pred_box = all_pred_box

                all_pred_box = Instances3D.cat([all_pred_box,pred_instances])
                per_frame_ins = Instances3D.cat([per_frame_ins,pred_instances])

                all_poses = np.concatenate((all_poses, pose_np), axis=0)  

                print("\ncur frame id:",count)
                '''
                STEP1: spatial association using 3D OBB NMS
                '''
                mask, success_mask = Instances3D.spatial_association(all_pred_box,cfg["box_fusion"]["nms_threshold"],box_manager,per_frame_ins.cam_pose)
                
                cur_keep_idx = [i-num_before_cat for i in mask if i>=num_before_cat]
                cur_success_nms = [i-num_before_cat for i in success_mask if i>=num_before_cat]
                
 
                keep_idx = np.asarray(mask)
                if len(cur_keep_idx)>0:
                    '''
                    STEP2: correspondence association for small objects
                    '''
                    all_pred_box,all_poses,keep_idx = Instances3D.correspondence_association(
                        cfg, 
                        box_manager, 
                        cur_keep_idx, 
                        cur_success_nms,
                        pred_instances, 
                        cur_global_pred_box, 
                        all_pred_box,all_poses, 
                        per_frame_ins.cam_pose, 
                        count,
                        mask,
                        sample["sensor_info"].gt.depth.K[-1],
                        all_kf_pose,
                        threshold=cfg['association']['small_threshold'],
                        H=image.shape[0],
                        W=image.shape[1]
                        )

                    # update the fusion list based on keep_idx
                    box_manager.update(keep_idx)
                
                    print(count," box_manager",box_manager.fusion_list)

                    #filter those evident wrong boxes that valid_num=0
                    if cfg['box_fusion']['check_valid']:
                        all_pred_box = box_manager.check_valid_num(all_pred_box, count, gap)

                    finalize_group3d_shadow_frame(
                        pred_instances,
                        observer_frame_token,
                        shadow_frame_token,
                        fragment_keyframe,
                        source_attempt_id,
                    )

                    '''
                    multi-view box fusion
                    '''
                    print("frame_id:box_num",box_manager.num_record)
                    if cfg['box_fusion']['use']:
                        Box_Fuser.boxfusion(all_pred_box, per_frame_ins, box_manager)
                
                    #predict the semantic classes of remaining new boxes
                    cur_keep_idx = [i-num_before_cat for i in keep_idx if i>=num_before_cat]
                    cur_keep_idx_in_all = [i for i in range(keep_idx.shape[0]) if keep_idx[i]>=num_before_cat]

                    if (
                        len(cur_keep_idx)>0
                        and not association_features_enabled
                        and not terminal_clip_enabled
                    ):
                        boxes = pred_instances.pred_boxes.cpu().numpy()
                        boxes = boxes[cur_keep_idx]
                        # scale the boxes
                        boxes = scale_boxes(boxes,image.shape[0],image.shape[1],scale=cfg['detection'].get('scale_box', 1.5))
                        # if len(pred_instances)>0:
                        class_results, box_features = text_prompt(boxes, tokenized_text, text_features, image, clip_model, preprocess) #[N_box]
                        all_pred_box.categories[cur_keep_idx_in_all] = class_results

                else: # no new box
                    all_pred_box = all_pred_box[mask]
                    all_poses = all_poses[mask]
                    box_manager.update(keep_idx)
                    finalize_group3d_shadow_frame(
                        pred_instances,
                        observer_frame_token,
                        shadow_frame_token,
                        fragment_keyframe,
                        source_attempt_id,
                    )
                    print(count, "new boxes have all been nms"," box_manager",box_manager.fusion_list)

            if re_vis:
                visualize_online_boxes(all_pred_box, prefix="/device/wide", boxes_3d_name="pred_boxes_3d", log_instances_name="pred_instances",count=count,save=False,show_class=cfg["vis"]["show_class"],show_label=cfg["vis"]["show_label"]) 

            prune_terminal_clip_crops(all_pred_box)

        count+=1
        
        # save the results
        if count == len(dataset)-1 or (count+gap)>len(dataset)-1:
            run_terminal_clip(all_pred_box)
            terminal_geometry = run_terminal_tm_fpf_c1(all_pred_box)
            terminal_boxes_3d = terminal_geometry.corners.cpu().numpy()
            terminal_scores = all_pred_box.scores.detach().cpu().numpy()
            terminal_valid_mask = np.ones(terminal_boxes_3d.shape[0], dtype=bool)
            if cfg['dataset'] == 'scannet':
                terminal_boxes_3d, terminal_valid_mask = post_process(
                    terminal_boxes_3d, return_mask=True
                )
                terminal_scores = terminal_scores[terminal_valid_mask]
            if Box_Fuser.capf.oracle_shadow:
                terminal_track_keys = [
                    box_manager.fusion_list[int(index)]
                    for index in np.flatnonzero(terminal_valid_mask)
                ]
                capf_oracle_path = Box_Fuser.capf.write_oracle_diagnostics(
                    scene_id=scene_id,
                    final_track_keys=terminal_track_keys,
                    final_corners_world=terminal_boxes_3d,
                    final_scores=terminal_scores,
                )
                print("CAPF oracle shadow saved:", capf_oracle_path)
            live_terminal = None
            if strict_live_route is not None:
                live_terminal = strict_live_route.finalize(
                    native_boxes_3d=terminal_boxes_3d,
                    native_scores=terminal_scores,
                    final_frame_id=count - 1,
                )
                terminal_boxes_3d = live_terminal.boxes_3d
                terminal_scores = live_terminal.scores
            end_time = time.time()
            duration = end_time - start_time  
            fps = count / duration
            print(f"Cost: {duration:.2f} s", f"Average FPS: {fps:.2f}")
            if cfg.get('association', {}).get(
                'appearance_gate', {}
            ).get('enabled', False):
                print(box_manager.appearance_gate_summary())
            if box_manager.causal_hungarian.enabled:
                print(box_manager.causal_hungarian.summary())
            if Box_Fuser.reliable_view_cfg["enabled"]:
                print(Box_Fuser.reliable_view_summary())
            if edgetam_maskdepth.enabled:
                print(edgetam_maskdepth.summary())
            if Box_Fuser.maskdepth_pfo_cfg["enabled"]:
                print(Box_Fuser.maskdepth_pfo_summary())
            if Box_Fuser.vapf_lite.enabled:
                print(Box_Fuser.vapf_lite.summary())
            if Box_Fuser.capf.enabled:
                print(Box_Fuser.capf.summary())
            if lifting_adapter is not None:
                print(lifting_adapter.summary())
            if boxer_mvpr is not None:
                mvpr_summary = boxer_mvpr.finalize()
                print("Boxer-MVPR summary:", mvpr_summary["stats"])
            if boxer_gsa is not None:
                gsa_summary = boxer_gsa.finalize()
                print("Boxer-GSA summary:", gsa_summary["stats"])
            if live_terminal is not None:
                live_diag = live_terminal.diagnostics
                live_counts = live_diag["counts"]
                live_timing = live_diag["timing_ms"]["keyframe_total"]
                if stream3dv3_live is not None:
                    live_runtime = live_diag["runtime"]
                    print(
                        "Stream3Dv3 live summary |",
                        "past_only=True",
                        "future_access=0",
                        f"deadline_misses={live_counts.get('addon_deadline_misses', 0)}",
                        f"keyframe_p50/p95_ms={live_timing['p50']:.3f}/{live_timing['p95']:.3f}",
                        f"births={live_terminal.birth_count}",
                        f"overlays={live_terminal.overlay_count}",
                        f"exact_fps={live_runtime['end_to_end_fps']:.6f}",
                        f"raw_frames={live_runtime['raw_frame_count']}",
                        f"f4_attempts={live_counts.get('f4_attempts', 0)}",
                        f"cuda_peak_allocated_mb={live_diag['peak_cuda_allocated_bytes'] / (1024 ** 2):.1f}",
                    )
                else:
                    live_sam3 = live_diag["sam3"]
                    print(
                        "Strict live summary |",
                        "past_only=True",
                        "future_access=0",
                        "queue_capacity=1",
                        f"queue_max={live_sam3['max_queue_depth']}",
                        f"queue_drops={live_sam3['drop_count']}",
                        f"deadline_misses={live_counts.get('deadline_misses', 0)}",
                        f"keyframe_p50/p95_ms={live_timing['p50']:.3f}/{live_timing['p95']:.3f}",
                        f"births={live_terminal.birth_count}",
                        f"overlays={live_terminal.overlay_count}",
                        f"cuda_peak_allocated_mb={live_diag['peak_cuda_allocated_bytes'] / (1024 ** 2):.1f}",
                    )

            if proposal_cache is not None and proposal_cache.is_replay:
                baseline_root = cfg["lifting"]["proposal_cache"].get(
                    "baseline_prediction_root", ""
                )
                proposal_cache.verify_replay_complete(
                    scene_id,
                    baseline_prediction_path=os.path.join(
                        baseline_root, scene_id + "_boxes.pkl"
                    ),
                )
            
            # save global boxes for evaluation
            if cfg['data']['output_dir'] is not None and cfg["eval"]:
                output_path = os.path.join(
                    cfg['data']['output_dir'], scene_id + "_boxes.pkl"
                )
                class_list = tokenized_text.tolist()
                class_idx = np.array([class_list.index(c) for c in all_pred_box.categories]) #[N]

                boxes_3d = terminal_boxes_3d
                scores = terminal_scores
                valid_mask = terminal_valid_mask

                terminal_identity = None
                if observer_adapter is not None and observer_adapter.config.mode == "shadow":
                    with observer_adapter.observer_boundary(
                        observer_native_fields(),
                        label="terminal_mapping",
                    ):
                        terminal_identity = observer_adapter.terminal_mapping(
                            np.flatnonzero(valid_mask),
                            native_row_count=len(all_pred_box),
                            current_frame_id=count - 1,
                            close=True,
                        )

                assert boxes_3d.shape[0] == scores.shape[0], \
                    "Saved boxes and confidence scores must stay aligned"
                    
                if boxes_3d.shape[0]>0:
                    print(
                        "Saving score-preserving predictions:",
                        f"count={scores.shape[0]}",
                        f"min={scores.min():.6f}",
                        f"max={scores.max():.6f}",
                        f"std={scores.std():.6f}",
                    )
                    save_list = [[
                        (int(0), boxes_3d[n], float(scores[n]))
                        for n in range(boxes_3d.shape[0])
                    ]] # list of tuples class_idx[n]

                    save_box(save_list, output_path)
                elif proposal_cache is not None:
                    print("Saving score-preserving predictions: count=0")
                    save_box([[]], output_path)

                if observer_adapter is not None and observer_adapter.config.mode == "shadow":
                    observer_adapter.write_diagnostics()
                    trace_valid = (
                        observer_adapter.trace_valid
                        and terminal_identity is not None
                    )
                    diagnostics_root = observer_adapter.config.diagnostics_root

                    if (
                        shadow_variant in ("smov", "gclean", "puf")
                        and diagnostics_root is not None
                    ):
                        prepare_samples = [
                            float(record["prepare_elapsed_ms"])
                            for record in smov_frame_records
                        ]
                        failure_reasons = {}
                        for record in smov_frame_records:
                            for reason, value in record["failure_reasons"].items():
                                failure_reasons[reason] = (
                                    failure_reasons.get(reason, 0) + int(value)
                                )
                        smov_summary = {
                            "mode": "shadow",
                            "fragment_source": "smov_clean",
                            "pending": False,
                            "keyframes": len(smov_frame_records),
                            "proposal_count": sum(
                                int(record["proposal_count"])
                                for record in smov_frame_records
                            ),
                            "selected_count": sum(
                                int(record["selected_count"])
                                for record in smov_frame_records
                            ),
                            "accepted_count": sum(
                                int(record["accepted_count"])
                                for record in smov_frame_records
                            ),
                            "abstained_count": sum(
                                int(record["abstained_count"])
                                for record in smov_frame_records
                            ),
                            "capped_count": sum(
                                int(record["capped_count"])
                                for record in smov_frame_records
                            ),
                            "failure_reasons": dict(sorted(failure_reasons.items())),
                            "prepare_timing_ms": {
                                "count": len(prepare_samples),
                                "p50": (
                                    float(np.percentile(prepare_samples, 50))
                                    if prepare_samples
                                    else 0.0
                                ),
                                "p95": (
                                    float(np.percentile(prepare_samples, 95))
                                    if prepare_samples
                                    else 0.0
                                ),
                                "max": max(prepare_samples, default=0.0),
                                "sum": float(sum(prepare_samples)),
                            },
                            "observer_errors": list(observer_adapter.errors),
                        }
                        try:
                            write_smov_shadow_diagnostics(
                                os.path.join(
                                    diagnostics_root,
                                    scene_id + ".smov_shadow.json",
                                ),
                                scene_id,
                                smov_frame_records,
                                smov_summary,
                                trace_valid,
                            )
                        except Exception as error:
                            print("SMOV-shadow diagnostic write failed:", repr(error))
                        print(
                            "SMOV-shadow summary |",
                            f"trace_valid={trace_valid}",
                            f"keyframes={smov_summary['keyframes']}",
                            f"proposals={smov_summary['proposal_count']}",
                            f"accepted={smov_summary['accepted_count']}",
                            f"abstained={smov_summary['abstained_count']}",
                        )

                    if graw_shadow is not None and diagnostics_root is not None:
                        graw_diagnostics = graw_shadow.diagnostics()
                        graw_summary = {
                            "pending": bool(graw_diagnostics["pending"]),
                            "last_frame": graw_diagnostics["last_frame"],
                            "memory_track_ids": list(
                                graw_diagnostics["memory_track_ids"]
                            ),
                            "stats": dict(graw_diagnostics["stats"]),
                            "observer_errors": list(observer_adapter.errors),
                        }
                        try:
                            write_graw_shadow_diagnostics(
                                os.path.join(
                                    diagnostics_root,
                                    scene_id + ".graw_shadow.json",
                                ),
                                scene_id=scene_id,
                                results=graw_frame_records,
                                summary=graw_summary,
                                trace_valid=trace_valid,
                            )
                        except Exception as error:
                            print("Graw-shadow diagnostic write failed:", repr(error))
                        graw_stats = dict(graw_diagnostics["stats"])
                        print(
                            "Graw-shadow summary |",
                            f"trace_valid={trace_valid}",
                            f"keyframes={graw_stats.get('keyframes', 0)}",
                            f"candidates={graw_stats.get('candidate_proposals', 0)}",
                            f"associations={graw_stats.get('counterfactual_associations', 0)}",
                            f"fail_open={graw_stats.get('matcher_fail_open', 0)}",
                        )

                    if gclean_shadow is not None and diagnostics_root is not None:
                        gclean_diagnostics = gclean_shadow.diagnostics()
                        gclean_summary = {
                            **dict(gclean_diagnostics),
                            "observer_errors": list(observer_adapter.errors),
                        }
                        try:
                            write_gclean_shadow_diagnostics(
                                os.path.join(
                                    diagnostics_root,
                                    scene_id + ".gclean_shadow.json",
                                ),
                                scene_id=scene_id,
                                results=gclean_frame_records,
                                summary=gclean_summary,
                                trace_valid=trace_valid,
                            )
                        except Exception as error:
                            print("Gclean-shadow diagnostic write failed:", repr(error))
                        gclean_stats = dict(gclean_diagnostics["stats"])
                        print(
                            "Gclean-shadow summary |",
                            f"trace_valid={trace_valid}",
                            f"keyframes={gclean_stats.get('keyframes', 0)}",
                            f"candidates={gclean_stats.get('candidate_proposals', 0)}",
                            f"associations={gclean_stats.get('counterfactual_associations', 0)}",
                            f"fail_open={gclean_stats.get('matcher_fail_open', 0)}",
                        )

                    if puf_gclean_shadow is not None and diagnostics_root is not None:
                        puf_diagnostics = puf_gclean_shadow.diagnostics()
                        puf_summary = {
                            **dict(puf_diagnostics),
                            "observer_errors": list(observer_adapter.errors),
                        }
                        try:
                            write_puf_gclean_shadow_diagnostics(
                                os.path.join(
                                    diagnostics_root,
                                    scene_id + ".puf_gclean_shadow.json",
                                ),
                                scene_id=scene_id,
                                results=puf_gclean_frame_records,
                                summary=puf_summary,
                                trace_valid=trace_valid,
                            )
                        except Exception as error:
                            print(
                                "PUF-Gclean-shadow diagnostic write failed:",
                                repr(error),
                            )
                        puf_stats = dict(puf_diagnostics["stats"])
                        print(
                            "PUF-Gclean-shadow summary |",
                            f"trace_valid={trace_valid}",
                            f"keyframes={puf_stats.get('keyframes', 0)}",
                            f"proposals={puf_stats.get('proposals', 0)}",
                            f"directives={puf_stats.get('paper_directives', 0)}",
                            f"actionable={puf_stats.get('active_safe_associations', 0)}",
                            f"fail_open={puf_stats.get('fail_open_keyframes', 0)}",
                        )

            if proposal_cache is not None and proposal_cache.is_record:
                if cfg['data']['output_dir'] is None or not cfg["eval"]:
                    raise ValueError(
                        "Proposal-cache record requires an evaluated output file"
                    )
                proposal_cache.finalize(scene_id, prediction_path=output_path)
                    
            exit(0)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_path", help="Path to the directory containing the .tar files, the full path to a single tar file (recommended), or a path to a txt file containing HTTP links. Using the value \"stream\" will attempt to stream from your device using the NeRFCapture app")
    parser.add_argument("--model-path", help="Path to the model to load")
    parser.add_argument("--config", default=None, type=str, help="config_path")
    parser.add_argument("--clip_path", default='./models/open_clip_pytorch_model.bin', type=str, help="Path to the CLIP model")
    parser.add_argument("--seq", default='None', type=str, help="config_path")
    parser.add_argument("--class_txt", default='./data/panoptic_categories_nomerge.txt', type=str, help="config_path")
    parser.add_argument("--every-nth-frame", default=None, type=int, help="Load every `n` frames")
    parser.add_argument("--viz-on-gt-points", default=True, action="store_true", help="Backproject the GT depth to form a point cloud in order to visualize the predictions")
    parser.add_argument("--device", default="cpu", help="Which device to push the model to (cpu, mps, cuda)")
    parser.add_argument("--video-ids", nargs="+", help="Subset of videos to execute on. By default, all. Ignored if a tar file is explicitly given or in stream mode.")

    args = parser.parse_args()
    print("Command Line Args:", args)

    dataset_path = args.dataset_path
    use_cache = False
    
    if dataset_path.lower() in ["scannet", "ca1m", 'online']:
        if not os.path.exists(args.config):
            raise ValueError("Missing config path")
        else:
            with open(args.config, 'r') as  f:
                cfg = yaml.full_load(f)
        experiment_seed = int(cfg.get("experiment", {}).get("seed", 0))
        random.seed(experiment_seed)
        np.random.seed(experiment_seed)
        torch.manual_seed(experiment_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(experiment_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("Deterministic experiment seed:", experiment_seed)
        # load the customized sequence if given by the user
        if args.seq is not None:
            if dataset_path.lower()=='ca1m':
                if 'example' in cfg['data']['datadir']:
                    current_file_path = os.path.abspath(__file__)
                    current_dir = os.path.dirname(current_file_path)
                    cfg['data']['datadir'] = os.path.join(current_dir, cfg['data']['datadir'])

                else:
                    new_datadir = os.path.join(os.path.dirname(os.path.dirname(cfg['data']['datadir'])),  args.seq+'/')
                    cfg['data']['datadir'] = new_datadir

                
            else:
                new_datadir = os.path.join(os.path.dirname(os.path.dirname(cfg['data']['datadir'])),  args.seq+'/frames/')
                cfg['data']['datadir'] = new_datadir
                
            # eval only
            if os.path.exists(os.path.join(cfg['data']['output_dir'],args.seq+"_boxes.pkl")) and cfg["eval"]:
                print("Results for boxes already exist, skip evaluation")
                sys.exit(0)
        
        dataset = get_dataset(cfg)

    assert args.model_path is not None
    checkpoint = torch.load(args.model_path, map_location=args.device or "cpu")["model"]
    backbone_embedding_dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
        
    is_depth_model = True 
    model = make_cubify_transformer(dimension=backbone_embedding_dimension, depth_model=is_depth_model).eval()
    model.load_state_dict(checkpoint)

    dataset.load_arkit_depth = True
    if args.every_nth_frame is not None:
        dataset = itertools.islice(dataset, 0, None, args.every_nth_frame)

    augmentor = Augmentor(("wide/image", "wide/depth"))
    preprocessor = Preprocessor()
    
    if args.device is not None:
        model = model.to(args.device)
        clip_model, preprocess = load_clip(args.clip_path)
        text_class = np.genfromtxt(args.class_txt, delimiter='\n', dtype=str) 
        text_features = torch.load('./data/class_features.pt').cuda()

    run(cfg, model, dataset, clip_model, preprocess, text_class, text_features, augmentor, preprocessor, score_thresh=cfg['detection']['score_thresh'], viz_on_gt_points=args.viz_on_gt_points, gap=cfg["data"]["gap"], re_vis=cfg['vis']['rerun'])

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import glob
import itertools
import random
import numpy as np
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
from boxfusion.clip_instance_features import ClipInstanceFeatureEncoder
from boxfusion.online_refinement import (
    build_online_refinement_controller,
    supplemental_extent_is_valid,
)
from boxfusion.online_ablation import (
    ONLINE_ABLATION_PROFILES,
    apply_online_ablation_profile,
)
from boxfusion.stable_ids import resolve_fusion_stable_ids
from boxfusion.trifusion_cli import configure_trifusion_ap50_gate
from boxfusion.yidu_cli import configure_yidu_ap50_gate


class StridedDataset:
    """Sized iterable view used by ``--every-nth-frame``."""

    def __init__(self, dataset, stride):
        if stride < 1:
            raise ValueError("stride must be at least one")
        self.dataset = dataset
        self.stride = int(stride)

    def __iter__(self):
        return itertools.islice(iter(self.dataset), 0, None, self.stride)

    def __len__(self):
        length = len(self.dataset)
        return (length + self.stride - 1) // self.stride


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

    count=0
    all_pred_box = None
    all_poses = None

    all_kf_pose = {}
    per_frame_ins = None #save every predicted boxes
    traj_xyz = []

    box_manager = BoxManager(cfg)
    Box_Fuser = BoxFusion(cfg)
    online_cfg = cfg.get("online_refinement", {})
    use_online_appearance = bool(
        online_cfg.get("enabled", False)
        and online_cfg.get("appearance_memory", {}).get("enabled", True)
    )
    appearance_encoder = (
        ClipInstanceFeatureEncoder(
            clip_model,
            preprocess,
            masked_crop=online_cfg.get("appearance_memory", {}).get(
                "masked_crop", True
            ),
        )
        if use_online_appearance
        else None
    )
    online_refiner = build_online_refinement_controller(
        cfg,
        device=str(model.pixel_mean.device),
        appearance_encoder=appearance_encoder,
    )

    box_count = 0
    start_time = time.time()
    reported_stable_id_repairs = set()

    def online_snapshot():
        """Return immutable final-output inputs for the optional controller."""
        if all_pred_box is None:
            return (
                np.empty((0, 8, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        corners = all_pred_box.pred_boxes_3d.corners.detach().cpu().numpy()
        scores = all_pred_box.scores.detach().cpu().numpy()
        if len(box_manager.fusion_list) != len(all_pred_box):
            raise RuntimeError(
                "BoxManager fusion_list is not aligned with global boxes"
            )
        raw_stable_ids = np.asarray(
            [min(indices) for indices in box_manager.fusion_list],
            dtype=np.int64,
        )
        stable_ids = resolve_fusion_stable_ids(box_manager.fusion_list)
        if not np.array_equal(stable_ids, raw_stable_ids):
            changes = tuple(
                (
                    tuple(box_manager.fusion_list[index]),
                    int(raw_stable_ids[index]),
                    int(stable_ids[index]),
                )
                for index in range(len(stable_ids))
                if stable_ids[index] != raw_stable_ids[index]
            )
            if changes not in reported_stable_id_repairs:
                reported_stable_id_repairs.add(changes)
                printable_changes = [
                    (list(group), raw_id, resolved_id)
                    for group, raw_id, resolved_id in changes
                ]
                print(
                    "Resolved duplicate fusion stable IDs | "
                    f"changes={printable_changes}"
                )
        return corners, scores, stable_ids

    def observe_online_keyframe(
        frame_index,
        scene_identifier,
        source_frame_identifier,
        pose_matrix,
    ):
        if not online_refiner.enabled:
            return
        global_corners, global_scores, stable_ids = online_snapshot()
        depth = sample["wide"]["depth"][-1].detach().cpu().numpy()
        depth_intrinsics = (
            sample["sensor_info"].wide.depth.K[-1]
            .detach()
            .cpu()
            .numpy()
        )
        online_refiner.process_keyframe(
            image=image,
            depth=depth,
            intrinsics=depth_intrinsics,
            camera_to_world=pose_matrix,
            frame_id=int(frame_index),
            scene_id=str(scene_identifier),
            cache_frame_id=str(source_frame_identifier),
            global_corners=global_corners,
            global_scores=global_scores,
            stable_ids=stable_ids,
        )
    
    
    
    for sample in dataset:
        sample_video_id = sample["meta"]["video_id"] #(['sensor_info', 'wide', 'gt', 'meta'])
        pose = sample['sensor_info'].gt.RT
        
        video_id = sample_video_id
        if ((recording is None) or (video_id != sample_video_id)) and re_vis:
            new_recording = rerun.new_recording(
                application_id=str(sample_video_id), recording_id=uuid.uuid4(), make_default=True)
            new_recording.send_blueprint(blueprint, make_active=True)
            rerun.spawn()
            recording = new_recording
        
        pose_np = pose.squeeze().cpu().numpy()
        frame_pose_np = pose_np.copy()

        if re_vis:
            rerun.set_time_seconds("pts", sample["meta"]["timestamp"], recording=recording)

        # -> channels last.
        image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)  #[H,W,3]

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
                    
        packaged = augmentor.package(sample)
        packaged = move_input_to_current_device(packaged, device)
        packaged = preprocessor.preprocess([packaged])

        # Every gap nth frame is selected as keyframe
        if count % gap == 0:
            with torch.no_grad():
                pred_instances = model(packaged)[0] 

            pred_instances = pred_instances[pred_instances.scores >= float(score_thresh)]
 
            if cfg["detection"]["uv_bound"]:
                uv_mask = box_manager.check_uv_bounds(pred_instances.pred_proj_xy,image.shape[1],image.shape[0],ratio=cfg["detection"]["uv_bound_value"]) #[N]
                pred_instances = pred_instances[uv_mask]
            if cfg["detection"]["floor_mask"]:
                floor_mask = box_manager.check_floor_mask(pred_instances.pred_boxes_3d.tensor, ratio=cfg["detection"]["floor_ratio"])
                pred_instances = pred_instances[~floor_mask]

           # avoid first frame empty predictions
            if len(pred_instances) == 0 and count ==0:
                with torch.no_grad():
                    pred_instances = model(packaged)[0]
                pred_instances = pred_instances[pred_instances.scores >= float(cfg['detection']['score_thresh']/4)]
                print("again",count,"pred_instances",len(pred_instances))
                if cfg["detection"]["uv_bound"]:
                    uv_mask = box_manager.check_uv_bounds(pred_instances.pred_proj_xy,image.shape[1],image.shape[0],ratio=cfg["detection"]["uv_bound_value"]) #[N]
                    pred_instances = pred_instances[uv_mask]
                print("again",count,"pred_instances",len(pred_instances))

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

        # only process keyframes
        if count % gap ==0 or count == len(dataset)-1:
            
            all_kf_pose[count] = pose_np
            pose_np = np.expand_dims(pose_np,axis=0)
            pose_np = np.repeat(pose_np, repeats=len(pred_instances), axis=0) 
            
            if len(pred_instances)==0:
                all_pred_box = all_pred_box
                all_poses = all_poses
                box_count += len(pred_instances)
                box_manager.num_record[count] = box_count
                scene_identifier = (
                    video_id[0]
                    if isinstance(video_id, (tuple, list, np.ndarray))
                    else video_id
                )
                observe_online_keyframe(
                    count,
                    scene_identifier,
                    sample["meta"]["timestamp"],
                    frame_pose_np,
                )
                count+=1
                continue
            
            # add new properties for Instance3D predictions
            pred_instances.categories = np.array(['None'] * len(pred_instances)) # Initialize category labels as 'None' for all predicted instances
            pred_instances.cam_pose = torch.from_numpy(pose_np) # Convert camera pose from numpy to tensor and assign to instances
            pred_instances.frame_id = torch.tensor([count]).repeat(pose_np.shape[0]) # Assign current frame ID to all instances in this frame
            pred_instances.init_id = box_count+torch.arange(len(pred_instances)) # Create unique initial IDs for each instance based on global box count
            pred_instances.valid_num = torch.zeros(len(pred_instances)) # Initialize validation counter to zero for all instances
            pred_instances.pred_boxes_3d.transform2world(pred_instances.cam_pose) # Transform 3D bounding boxes from camera coordinates to world coordinates
            pred_instances.project_3d_boxes(sample["sensor_info"].wide.depth.K[-1].numpy(), H=image.shape[0],W=image.shape[1]) # Project 3D boxes to 2D image coordinates using camera intrinsics
            appearance_gate_cfg = cfg.get('association', {}).get(
                'appearance_gate', {}
            )
            appearance_gate_enabled = appearance_gate_cfg.get(
                'enabled', False
            )
            if appearance_gate_enabled:
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

            # record how many boxes each keyframe has, so we know which box belongs to which frame
            box_count += len(pred_instances)
            box_manager.num_record[count] = box_count
 
            # first keyframe, initialize some data structures
            if all_pred_box is None and count<gap:
                
                #predict the semantic classes
                if not appearance_gate_enabled:
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

            else:
                
                box_manager.init_new_predictions(len(pred_instances),len(per_frame_ins))

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

                    '''
                    multi-view box fusion
                    '''
                    print("frame_id:box_num",box_manager.num_record)
                    if cfg['box_fusion']['use']:
                        Box_Fuser.boxfusion(all_pred_box, per_frame_ins, box_manager)
                
                    #predict the semantic classes of remaining new boxes
                    cur_keep_idx = [i-num_before_cat for i in keep_idx if i>=num_before_cat]
                    cur_keep_idx_in_all = [i for i in range(keep_idx.shape[0]) if keep_idx[i]>=num_before_cat]

                    if len(cur_keep_idx)>0 and not appearance_gate_enabled:
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
                    print(count, "new boxes have all been nms"," box_manager",box_manager.fusion_list)

            scene_identifier = (
                video_id[0]
                if isinstance(video_id, (tuple, list, np.ndarray))
                else video_id
            )
            observe_online_keyframe(
                count,
                scene_identifier,
                sample["meta"]["timestamp"],
                frame_pose_np,
            )

            if re_vis:
                visualize_online_boxes(all_pred_box, prefix="/device/wide", boxes_3d_name="pred_boxes_3d", log_instances_name="pred_instances",count=count,save=False,show_class=cfg["vis"]["show_class"],show_label=cfg["vis"]["show_label"]) 

        count+=1
        
        # save the results
        if count == len(dataset)-1 or (count+gap)>len(dataset)-1:
            if cfg.get('association', {}).get(
                'appearance_gate', {}
            ).get('enabled', False):
                print(box_manager.appearance_gate_summary())
            if Box_Fuser.reliable_view_cfg["enabled"]:
                print(Box_Fuser.reliable_view_summary())

            # save global boxes for evaluation
            timing_printed = False
            if cfg['data']['output_dir'] is not None and cfg["eval"]:
                if online_refiner.enabled:
                    base_corners, base_scores, stable_ids = (
                        online_snapshot()
                    )
                    scene_identifier = (
                        video_id[0]
                        if isinstance(video_id, (tuple, list, np.ndarray))
                        else video_id
                    )
                    refinement_result = online_refiner.finalize(
                        global_corners=base_corners,
                        global_scores=base_scores,
                        stable_ids=stable_ids,
                        scene_id=str(scene_identifier),
                    )
                    boxes_3d = refinement_result.corners
                    scores = refinement_result.scores
                    print(online_refiner.summary_text())
                else:
                    # Preserve the original export path and its invariants.
                    boxes_3d = (
                        all_pred_box.pred_boxes_3d.corners.cpu().numpy()
                    )
                    scores = all_pred_box.scores.detach().cpu().numpy()
                if cfg['dataset'] == 'scannet':
                    minimum_extent = float(
                        cfg.get("data", {}).get(
                            "post_process_min_extent", 0.30
                        )
                    )
                    supplemental_cfg = (
                        online_refiner.config["supplemental_output"]
                        if online_refiner.enabled
                        else {}
                    )
                    if (
                        online_refiner.enabled
                        and supplemental_cfg.get(
                            "class_aware_extent", False
                        )
                    ):
                        # Preserve the exact ScanNet extent gate for every
                        # upstream/global row.  Only C1 supplemental rows may
                        # use the explicit sink/door/window allow-list.
                        ranges = (
                            np.max(boxes_3d, axis=1)
                            - np.min(boxes_3d, axis=1)
                        )
                        # The controller validates C2 in center/size form.
                        # A float32 corner round-trip at non-zero world
                        # coordinates may undershoot an exact extent by a few
                        # ulps, so reuse the authoritative C2 dimensions.  Do
                        # not relax the legacy/global extent contract.
                        c2_ranges = ranges.copy()
                        for row_index, refit_reason in enumerate(
                            refinement_result.refit_reasons
                        ):
                            if str(refit_reason).startswith(
                                "c2_"
                            ) and str(refit_reason).endswith("_accepted"):
                                c2_ranges[row_index] = (
                                    refinement_result.boxes[
                                        row_index, 3:6
                                    ]
                                )
                        valid_mask = np.zeros(
                            boxes_3d.shape[0], dtype=bool
                        )
                        for row_index, (
                            dimensions,
                            source_index,
                            label,
                            refit_reason,
                        ) in enumerate(
                            zip(
                                c2_ranges,
                                refinement_result.source_indices,
                                refinement_result.labels,
                                refinement_result.refit_reasons,
                            )
                        ):
                            if int(source_index) >= 0:
                                valid_mask[row_index] = bool(
                                    np.all(
                                        dimensions >= minimum_extent
                                    )
                                )
                            else:
                                if refit_reason == "c2_planar_accepted":
                                    c2_cfg = online_refiner.config[
                                        "supplemental_geometry_refiner"
                                    ]
                                    valid_mask[row_index] = bool(
                                        np.all(
                                            np.sort(dimensions) + 1e-6
                                            >= np.asarray(
                                                [
                                                    c2_cfg[
                                                        "refined_planar_minimum_extent"
                                                    ],
                                                    supplemental_cfg[
                                                        "planar_middle_extent"
                                                    ],
                                                    supplemental_cfg[
                                                        "planar_max_extent"
                                                    ],
                                                ],
                                                dtype=np.float64,
                                            )
                                        )
                                    )
                                else:
                                    valid_mask[row_index] = (
                                        supplemental_extent_is_valid(
                                        dimensions,
                                        label,
                                        supplemental_cfg,
                                        default_minimum_extent=minimum_extent,
                                    )
                                    )
                        boxes_3d = boxes_3d[valid_mask]
                    else:
                        boxes_3d, valid_mask = post_process(
                            boxes_3d,
                            threshold=minimum_extent,
                            return_mask=True,
                        )
                    scores = scores[valid_mask]

                assert boxes_3d.shape[0] == scores.shape[0], \
                    "Saved boxes and confidence scores must stay aligned"

                end_time = time.time()
                duration = end_time - start_time
                fps = count / duration
                print(f"Cost: {duration:.2f} s", f"Average FPS: {fps:.2f}")
                timing_printed = True

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
                    ]]

                    save_box(save_list, os.path.join(cfg['data']['output_dir'], video_id[0]+"_boxes.pkl"))

            if not timing_printed:
                end_time = time.time()
                duration = end_time - start_time
                fps = count / duration
                print(f"Cost: {duration:.2f} s", f"Average FPS: {fps:.2f}")

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
    parser.add_argument("--class-features", default='./data/class_features.pt', type=str, help="Path to cached CLIP text features")
    parser.add_argument("--output-dir", default=None, type=str, help="Override config data.output_dir")
    parser.add_argument("--diagnostics-root", default=None, type=str, help="Override online-refinement diagnostics root")
    parser.add_argument("--online-proposal-checkpoint", default=None, type=str, help="Override the supplemental proposal checkpoint")
    parser.add_argument("--online-proposal-provider", choices=["yoloe", "cache_only"], default=None, help="Use the lightweight online provider or strict offline-teacher cache replay")
    parser.add_argument("--online-proposal-cache-directory", default=None, type=str, help="Override the proposal-cache directory")
    parser.add_argument("--online-proposal-cache-namespace", default=None, type=str, help="Immutable namespace used by strict teacher-cache replay")
    parser.add_argument("--online-proposal-cache-missing-policy", choices=["error", "empty"], default=None, help="Strict teacher-cache miss behavior")
    parser.add_argument("--c4-proposal-cache-directory", default=None, type=str, help="Read-only SAM3 proposal cache used only by the C4 local-geometry observer")
    parser.add_argument("--c4-proposal-cache-namespace", default=None, type=str, help="Immutable namespace for C4 SAM3 cache replay")
    parser.add_argument("--c4-proposal-cache-missing-policy", choices=["error", "empty"], default=None, help="Strict C4 SAM3 cache miss behavior")
    parser.add_argument("--online-refiner-checkpoint", default=None, type=str, help="Enable and load a trained BoxRefiner checkpoint")
    parser.add_argument("--online-joint-checkpoint", default=None, type=str, help="Enable and load a strict B3 -> B5 + B6-v2 joint-local-head checkpoint")
    parser.add_argument("--online-joint-detector-blend", default=None, type=float, help="Weight of the original detector score in joint-head final ranking")
    parser.add_argument("--online-quality-checkpoint", default=None, type=str, help="Load a learned quality-calibration checkpoint")
    parser.add_argument("--online-quality-mode", choices=["linear", "mlp", "iou_mlp"], default=None, help="Learned quality checkpoint type")
    parser.add_argument("--online-quality-detector-blend", default=None, type=float, help="Weight of the original detector score in final quality ranking")
    parser.add_argument("--p1-residual-checkpoint", default=None, type=str, help="Train-only class-agnostic P1 residual proposal checkpoint")
    parser.add_argument("--p1-residual-mode", choices=["infer", "collect"], default=None, help="Run P1 inference or collect train-only sparse voxel inputs")
    parser.add_argument("--p1-collect-voxel-inputs", action="store_true", help="Persist P1 ragged sparse voxel inputs in diagnostics")
    parser.add_argument("--p2-occupancy-checkpoint", default=None, type=str, help="Train-only P2 foreground-occupancy Top-K checkpoint")
    parser.add_argument("--trifusion-ap50-gate-checkpoint", default=None, type=str, help="Strict NumPy AP50 safety-gate checkpoint; valid only with trifusion_plus10_observer")
    parser.add_argument("--yidu-ap50-gate-checkpoint", default=None, type=str, help="Strict train-only NumPy AP50 safety-gate checkpoint; valid only with yidu A6")
    parser.add_argument("--scannet-min-extent", default=None, type=float, help="Override both ScanNet export and online supplemental minimum box extent")
    parser.add_argument("--scannet-post-process-min-extent", default=None, type=float, help="Override only the final ScanNet export extent; keep online/global identity filters unchanged")
    parser.add_argument("--scannet-frames-root", default=None, type=str, help="Root containing <scene>/frames for ScanNet")
    parser.add_argument("--online-proposal-every-keyframes", default=None, type=int, help="Override supplemental proposal scheduling")
    parser.add_argument("--online-candidate-ttl-clock", choices=["keyframe", "provider_call"], default=None, help="Measure candidate TTL in BoxFusion keyframes or successful proposal-provider calls")
    parser.add_argument("--online-candidate-track-ttl", default=None, type=int, help="Override the number of missed candidate-lifecycle steps allowed before expiry")
    archive_group = parser.add_mutually_exclusive_group()
    archive_group.add_argument("--online-archive-confirmed-tracks", dest="online_archive_confirmed_tracks", action="store_true", help="Freeze confirmed candidates after TTL for final supplemental output")
    archive_group.add_argument("--no-online-archive-confirmed-tracks", dest="online_archive_confirmed_tracks", action="store_false", help="Discard every candidate after TTL (legacy behavior)")
    parser.set_defaults(online_archive_confirmed_tracks=None)
    parser.add_argument("--online-ablation-profile", choices=ONLINE_ABLATION_PROFILES, default=None, help="Apply a reproducible online-refinement ablation profile")
    parser.add_argument("--disable-online-refinement", action="store_true", help="Run the exact parent path from an online-refinement config")
    parser.add_argument("--every-nth-frame", default=None, type=int, help="Load every `n` frames")
    parser.add_argument("--viz-on-gt-points", default=True, action="store_true", help="Backproject the GT depth to form a point cloud in order to visualize the predictions")
    parser.add_argument("--device", default="cpu", help="Which device to push the model to (cpu, mps, cuda)")
    parser.add_argument("--seed", default=None, type=int, help="Optional deterministic inference seed")
    parser.add_argument("--video-ids", nargs="+", help="Subset of videos to execute on. By default, all. Ignored if a tar file is explicitly given or in stream mode.")

    args = parser.parse_args()
    print("Command Line Args:", args)
    if args.seed is not None:
        if args.seed < 0:
            parser.error("--seed must be a non-negative integer")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f"Inference seed: {args.seed}")

    dataset_path = args.dataset_path
    use_cache = False
    
    if dataset_path.lower() in ["scannet", "ca1m", 'online']:
        if not os.path.exists(args.config):
            raise ValueError("Missing config path")
        else:
            with open(args.config, 'r') as  f:
                cfg = yaml.full_load(f)
        if args.output_dir is not None:
            cfg["data"]["output_dir"] = args.output_dir
        online_cfg = cfg.setdefault("online_refinement", {})
        if (
            args.disable_online_refinement
            and args.online_ablation_profile is not None
        ):
            parser.error(
                "--disable-online-refinement and "
                "--online-ablation-profile are mutually exclusive"
            )
        if args.disable_online_refinement:
            online_cfg["enabled"] = False
        if args.diagnostics_root is not None:
            online_cfg.setdefault("diagnostics", {})["root"] = (
                args.diagnostics_root
            )
        if args.online_proposal_checkpoint is not None:
            online_cfg.setdefault("supplemental_proposals", {})[
                "checkpoint"
            ] = args.online_proposal_checkpoint
        if args.online_proposal_provider is not None:
            online_cfg.setdefault("supplemental_proposals", {})[
                "provider"
            ] = args.online_proposal_provider
        if any(
            value is not None
            for value in (
                args.online_proposal_cache_directory,
                args.online_proposal_cache_namespace,
                args.online_proposal_cache_missing_policy,
            )
        ):
            cache_cfg = online_cfg.setdefault(
                "supplemental_proposals", {}
            ).setdefault("cache", {})
            cache_cfg["enabled"] = True
            if args.online_proposal_cache_directory is not None:
                cache_cfg["directory"] = (
                    args.online_proposal_cache_directory
                )
            if args.online_proposal_cache_namespace is not None:
                cache_cfg["namespace"] = (
                    args.online_proposal_cache_namespace
                )
            if args.online_proposal_cache_missing_policy is not None:
                cache_cfg["missing_policy"] = (
                    args.online_proposal_cache_missing_policy
                )
        if args.online_refiner_checkpoint is not None:
            refiner_cfg = online_cfg.setdefault("box_refiner", {})
            refiner_cfg["enabled"] = True
            refiner_cfg["checkpoint"] = args.online_refiner_checkpoint
        if args.online_joint_checkpoint is not None:
            if (
                args.online_refiner_checkpoint is not None
                or args.online_quality_checkpoint is not None
            ):
                parser.error(
                    "--online-joint-checkpoint is mutually exclusive with "
                    "legacy refiner/quality checkpoints"
                )
            joint_cfg = online_cfg.setdefault("joint_local_head", {})
            joint_cfg["enabled"] = True
            joint_cfg["checkpoint"] = args.online_joint_checkpoint
            joint_cfg["mutate_geometry"] = True
            joint_cfg["mutate_scores"] = True
            joint_cfg["collect_diagnostics"] = True
        if args.online_quality_checkpoint is not None:
            if args.online_quality_mode is None:
                parser.error(
                    "--online-quality-mode is required with "
                    "--online-quality-checkpoint"
                )
            quality_cfg = online_cfg.setdefault("quality", {})
            quality_cfg["enabled"] = True
            quality_cfg["mode"] = args.online_quality_mode
            quality_cfg["checkpoint"] = args.online_quality_checkpoint
        elif args.online_quality_mode is not None:
            parser.error(
                "--online-quality-checkpoint is required with "
                "--online-quality-mode"
            )
        if args.online_proposal_every_keyframes is not None:
            if args.online_proposal_every_keyframes < 1:
                parser.error(
                    "--online-proposal-every-keyframes must be at least 1"
                )
            online_cfg["inference_every_keyframes"] = (
                args.online_proposal_every_keyframes
            )
        if args.online_candidate_ttl_clock is not None:
            online_cfg.setdefault("candidate_lifecycle", {})[
                "ttl_clock"
            ] = args.online_candidate_ttl_clock
        if args.online_candidate_track_ttl is not None:
            if args.online_candidate_track_ttl < 0:
                parser.error(
                    "--online-candidate-track-ttl must be non-negative"
                )
            online_cfg.setdefault("object_memory", {})[
                "track_ttl"
            ] = args.online_candidate_track_ttl
        if args.online_archive_confirmed_tracks is not None:
            online_cfg.setdefault("candidate_lifecycle", {})[
                "archive_confirmed"
            ] = args.online_archive_confirmed_tracks
        if args.online_ablation_profile is not None:
            cfg = apply_online_ablation_profile(
                cfg, args.online_ablation_profile
            )
            print(
                "Online ablation profile:",
                args.online_ablation_profile,
            )
        p1_cli_requested = (
            args.p1_residual_checkpoint is not None
            or args.p1_residual_mode is not None
            or args.p1_collect_voxel_inputs
        )
        if p1_cli_requested:
            if (
                args.online_ablation_profile
                not in {
                    "p1_residual_proposal_observer",
                    "p2_occupancy_topk_observer",
                    "p2v2_local_component_mask_rgbd_observer",
                }
            ):
                parser.error(
                    "P1 residual options require "
                    "a P1, P2, or P2V2 online-ablation profile"
                )
            p1_cfg = cfg.setdefault(
                "online_refinement", {}
            ).setdefault("residual_proposal", {})
            if args.p1_residual_mode is not None:
                p1_cfg["mode"] = args.p1_residual_mode
            if args.p1_residual_checkpoint is not None:
                p1_cfg["checkpoint"] = args.p1_residual_checkpoint
            if args.p1_collect_voxel_inputs:
                p1_cfg["collect_voxel_inputs"] = True
        if args.online_ablation_profile in {
            "p1_residual_proposal_observer",
            "p2_occupancy_topk_observer",
            "p2v2_local_component_mask_rgbd_observer",
        }:
            p1_cfg = cfg["online_refinement"]["residual_proposal"]
            if p1_cfg.get("mode", "infer") == "collect":
                if (
                    args.online_ablation_profile
                    in {
                        "p2_occupancy_topk_observer",
                        "p2v2_local_component_mask_rgbd_observer",
                    }
                ):
                    parser.error("P2/P2V2 require P1 infer mode")
                p1_cfg["collect_diagnostics"] = True
                p1_cfg["collect_voxel_inputs"] = True
                p1_cfg["checkpoint"] = None
            elif not p1_cfg.get("checkpoint"):
                parser.error(
                    "P1 infer mode requires --p1-residual-checkpoint"
                )
        if args.p2_occupancy_checkpoint is not None:
            if (
                args.online_ablation_profile
                not in {
                    "p2_occupancy_topk_observer",
                    "p2v2_local_component_mask_rgbd_observer",
                }
            ):
                parser.error(
                    "P2 occupancy options require "
                    "a P2 or P2V2 online-ablation profile"
                )
            cfg["online_refinement"]["occupancy_topk"][
                "checkpoint"
            ] = args.p2_occupancy_checkpoint
        if (
            args.online_ablation_profile
            in {
                "p2_occupancy_topk_observer",
                "p2v2_local_component_mask_rgbd_observer",
            }
            and not cfg["online_refinement"]["occupancy_topk"].get(
                "checkpoint"
            )
        ):
            parser.error(
                "P2/P2V2 require --p2-occupancy-checkpoint"
            )
        if args.trifusion_ap50_gate_checkpoint is not None:
            try:
                configure_trifusion_ap50_gate(
                    cfg,
                    args.trifusion_ap50_gate_checkpoint,
                    cli_profile=args.online_ablation_profile,
                )
            except (TypeError, ValueError, FileNotFoundError) as error:
                parser.error(str(error))
        if args.yidu_ap50_gate_checkpoint is not None:
            try:
                configure_yidu_ap50_gate(
                    cfg,
                    args.yidu_ap50_gate_checkpoint,
                    cli_profile=args.online_ablation_profile,
                )
            except (TypeError, ValueError, FileNotFoundError) as error:
                parser.error(str(error))
        c4_cache_values = (
            args.c4_proposal_cache_directory,
            args.c4_proposal_cache_namespace,
            args.c4_proposal_cache_missing_policy,
        )
        if any(value is not None for value in c4_cache_values):
            if args.c4_proposal_cache_directory is None:
                parser.error(
                    "--c4-proposal-cache-directory is required when "
                    "configuring C4 cache replay"
                )
            if args.c4_proposal_cache_namespace is None:
                parser.error(
                    "--c4-proposal-cache-namespace is required when "
                    "configuring C4 cache replay"
                )
            secondary_cfg = cfg.setdefault(
                "online_refinement", {}
            ).setdefault(
                "generic_local_geometry_refiner", {}
            ).setdefault(
                "secondary_proposals", {}
            )
            secondary_cfg["enabled"] = True
            secondary_cfg["provider"] = "cache_only"
            c4_cache_cfg = secondary_cfg.setdefault("cache", {})
            c4_cache_cfg["enabled"] = True
            c4_cache_cfg["write"] = False
            c4_cache_cfg["directory"] = (
                args.c4_proposal_cache_directory
            )
            c4_cache_cfg["namespace"] = (
                args.c4_proposal_cache_namespace
            )
            if args.c4_proposal_cache_missing_policy is not None:
                c4_cache_cfg["missing_policy"] = (
                    args.c4_proposal_cache_missing_policy
                )
        if args.scannet_min_extent is not None:
            if (
                not np.isfinite(args.scannet_min_extent)
                or args.scannet_min_extent < 0.0
            ):
                parser.error(
                    "--scannet-min-extent must be finite and non-negative"
                )
            cfg.setdefault("data", {})["post_process_min_extent"] = float(
                args.scannet_min_extent
            )
            cfg.setdefault("online_refinement", {}).setdefault(
                "output_filter", {}
            )["minimum_extent"] = float(args.scannet_min_extent)
        if args.scannet_post_process_min_extent is not None:
            if (
                not np.isfinite(args.scannet_post_process_min_extent)
                or args.scannet_post_process_min_extent < 0.0
            ):
                parser.error(
                    "--scannet-post-process-min-extent must be finite "
                    "and non-negative"
                )
            cfg.setdefault("data", {})["post_process_min_extent"] = float(
                args.scannet_post_process_min_extent
            )
        if args.online_quality_detector_blend is not None:
            if (
                not np.isfinite(args.online_quality_detector_blend)
                or not 0.0 <= args.online_quality_detector_blend <= 1.0
            ):
                parser.error(
                    "--online-quality-detector-blend must lie in [0, 1]"
                )
            cfg.setdefault("online_refinement", {}).setdefault(
                "quality", {}
            )["blend_with_detector"] = float(
                args.online_quality_detector_blend
            )
        if args.online_joint_detector_blend is not None:
            if (
                not np.isfinite(args.online_joint_detector_blend)
                or not 0.0
                <= args.online_joint_detector_blend
                <= 1.0
            ):
                parser.error(
                    "--online-joint-detector-blend must lie in [0, 1]"
                )
            cfg.setdefault("online_refinement", {}).setdefault(
                "joint_local_head", {}
            )["detector_blend"] = float(
                args.online_joint_detector_blend
            )
        if dataset_path.lower() == "scannet":
            # Give C1 the exact final export threshold before the controller
            # is constructed. This keeps its global duplicate reference,
            # source-aware extent policy, diagnostics, and saved rows aligned.
            cfg.setdefault("online_refinement", {}).setdefault(
                "output_filter", {}
            )["final_minimum_extent"] = float(
                cfg.get("data", {}).get(
                    "post_process_min_extent", 0.30
                )
            )
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
                frames_root = (
                    args.scannet_frames_root
                    if args.scannet_frames_root is not None
                    else os.path.dirname(
                        os.path.dirname(cfg["data"]["datadir"])
                    )
                )
                new_datadir = os.path.join(
                    frames_root, args.seq, "frames"
                )
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
        dataset = StridedDataset(dataset, args.every_nth_frame)

    augmentor = Augmentor(("wide/image", "wide/depth"))
    preprocessor = Preprocessor()
    
    if args.device is not None:
        model = model.to(args.device)
        clip_model, preprocess = load_clip(
            args.clip_path,
            device=args.device,
        )
        text_class = np.genfromtxt(args.class_txt, delimiter='\n', dtype=str) 
        text_features = torch.load(
            args.class_features,
            map_location=args.device,
        ).to(args.device)

    run(cfg, model, dataset, clip_model, preprocess, text_class, text_features, augmentor, preprocessor, score_thresh=cfg['detection']['score_thresh'], viz_on_gt_points=args.viz_on_gt_points, gap=cfg["data"]["gap"], re_vis=cfg['vis']['rerun'])

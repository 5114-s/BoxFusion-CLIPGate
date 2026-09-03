from dataclasses import dataclass
from typing import List, Dict, Set
from boxfusion.instances import Instances3D
from boxfusion.causal_hungarian_association import CausalHungarianAssociator
import operator
import numpy as np
import torch
import copy 


def _copy_observer_index(value):
    """Return a detached Python integer for an observer-only index."""
    if torch.is_tensor(value):
        if value.ndim != 0:
            raise ValueError("observer tensor indices must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError("observer array indices must be scalar")
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("observer indices must be integers, not booleans")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError("observer indices must be integers") from error
    result = int(result)
    if result < 0:
        raise ValueError("observer indices must be non-negative")
    return result


class BoxManager:

    def __init__(self,cfg):
        # self.box_registry: Dict[str, Instances3D] = {}  
        self.fusion_list = []  #record the candiates frame idx for per object box fusion 
        self.last_fusion_frame = [] # record the last fusion timestamp for each object
        self.fusion_flag = []
        self.already_fusion = []
        self.num_record = {}
        self.cfg = cfg
        self.causal_hungarian = CausalHungarianAssociator(cfg)
        self.rotation_gap = self.cfg['association']['rotation_gap']
        self.translation_gap = self.cfg['association']['translation_gap']
        self.small_size = self.cfg['box_fusion']['small_size'] 
        self.merge_log: List[Dict] = []           
        # Optional observer-only identity hook. It must never participate in
        # native association, fusion, scoring, or geometry.
        self.observer_track_registry = None
        self.observer_track_token = None
        self.observer_track_error = None
        self.appearance_gate_stats = {
            stage: {
                "candidates": 0,
                "accepted": 0,
                "protected": 0,
                "promoted": 0,
                "hard_overrides": 0,
                "penalized": 0,
                "bonused": 0,
                "similarities": [],
            }
            for stage in ("spatial", "correspondence")
        }

    def attach_observer_track_registry(self, registry, token):
        if (
            self.observer_track_registry is not None
            or self.observer_track_token is not None
        ):
            raise RuntimeError("observer track registry is already attached")
        try:
            registry.assert_native_row_count(token, len(self.fusion_list))
        except Exception as error:
            self._observer_track_fail_open(registry, token, error)
            return False
        self.observer_track_registry = registry
        self.observer_track_token = token
        self.observer_track_error = None
        return True

    def detach_observer_track_registry(self):
        registry = self.observer_track_registry
        token = self.observer_track_token
        error = self.observer_track_error
        self.observer_track_registry = None
        self.observer_track_token = None
        return registry, token, error

    def _observer_track_fail_open(self, registry, token, error):
        try:
            self.observer_track_error = repr(error)
        except Exception:
            self.observer_track_error = "unprintable observer error"
        try:
            registry.abort_keyframe(token)
        except Exception:
            pass
        self.observer_track_registry = None
        self.observer_track_token = None

    def _observer_track_call(self, method, *args, **kwargs):
        registry = self.observer_track_registry
        token = self.observer_track_token
        if registry is None or token is None:
            return
        try:
            # Observer code receives immutable snapshots of native index
            # containers. A one-shot iterator is rejected without advancing
            # it, so fail-open cannot consume or mutate native inputs.
            index_positions = kwargs.pop("_observer_index_positions", ())
            scalar_positions = kwargs.pop("_observer_scalar_positions", ())
            copied_args = list(args)
            for position in scalar_positions:
                copied_args[position] = _copy_observer_index(
                    copied_args[position]
                )
            for position in index_positions:
                value = copied_args[position]
                if isinstance(value, (str, bytes)):
                    raise ValueError(
                        "observer indices must be a bounded indexable container"
                    )
                try:
                    length = len(value)
                    get_item = value.__getitem__
                except (AttributeError, TypeError) as error:
                    raise ValueError(
                        "observer indices must be a bounded indexable container"
                    ) from error
                if length > 5120:
                    raise ValueError("observer indices exceed the hard cap of 5120")
                copied_args[position] = tuple(
                    _copy_observer_index(get_item(index))
                    for index in range(length)
                )
            registry.assert_native_row_count(token, len(self.fusion_list))
            getattr(registry, method)(token, *copied_args, **kwargs)
        except Exception as error:
            # Fail open: native BoxManager continues exactly as before.
            self._observer_track_fail_open(registry, token, error)

    def _observer_track_verify_rows(self):
        registry = self.observer_track_registry
        token = self.observer_track_token
        if registry is None or token is None:
            return
        try:
            registry.assert_native_row_count(token, len(self.fusion_list))
        except Exception as error:
            # A post-reindex mismatch also disables only the observer.
            self._observer_track_fail_open(registry, token, error)

    def init_new_predictions(self,box_num,all_num):
        for i in range(box_num):
            self.fusion_list.append([i+all_num])
            self.last_fusion_frame.append([0])
            self.fusion_flag.append(0)

    def record_appearance_gate(self, stage, gate_result):
        """Accumulate diagnostics without coupling association to logging."""
        if not gate_result.get("appearance_used", False):
            return
        if stage not in self.appearance_gate_stats:
            raise ValueError(f"Unknown appearance-gate stage: {stage}")

        candidate_mask = gate_result["geometry_candidates"]
        stats = self.appearance_gate_stats[stage]
        candidate_count = int(np.count_nonzero(candidate_mask))
        stats["candidates"] += candidate_count
        stats["accepted"] += int(
            np.count_nonzero(gate_result["accepted"] & candidate_mask)
        )
        stats["protected"] += int(
            np.count_nonzero(gate_result["protected"])
        )
        stats["promoted"] += int(
            np.count_nonzero(gate_result["promoted"])
        )
        stats["hard_overrides"] += int(
            np.count_nonzero(gate_result["hard_overrides"])
        )
        stats["penalized"] += int(
            np.count_nonzero(
                gate_result["penalized"] & candidate_mask
            )
        )
        stats["bonused"] += int(
            np.count_nonzero(
                gate_result["bonused"] & candidate_mask
            )
        )
        if candidate_count:
            stats["similarities"].extend(
                gate_result["similarities"][candidate_mask].tolist()
            )

    def appearance_gate_summary(self):
        lines = []
        for stage, stats in self.appearance_gate_stats.items():
            candidates = stats["candidates"]
            similarity_values = np.asarray(
                stats["similarities"], dtype=np.float32
            )
            if similarity_values.size:
                q10, q50, q90 = np.quantile(
                    similarity_values, [0.1, 0.5, 0.9]
                )
                similarity_summary = (
                    f"clip_q10/q50/q90="
                    f"{q10:.4f}/{q50:.4f}/{q90:.4f}"
                )
            else:
                similarity_summary = "clip_q10/q50/q90=nan/nan/nan"
            lines.append(
                f"{stage}: candidates={candidates}, "
                f"accepted={stats['accepted']}, "
                f"protected={stats['protected']}, "
                f"promoted={stats['promoted']}, "
                f"hard_overrides={stats['hard_overrides']}, "
                f"penalized={stats['penalized']}, "
                f"bonused={stats['bonused']}, "
                f"{similarity_summary}"
            )
        return "Appearance gate summary | " + " | ".join(lines)


    def add_fusion_ind(self, idx_list):
        self.already_fusion.append(copy.deepcopy(idx_list))

    def check_if_fusion(self, idx_list):
        if idx_list in self.already_fusion:
            return True
        else:
            return False

    def record(self, cur_id, fusion_inds, init_id, cam_poses, box_size, keep, box_centers):
        '''
        Note: cur_id is consistent to 'all_pred_box', idx is according to 'per_frame_box'
        '''
        if (
            self.observer_track_registry is not None
            and self.observer_track_token is not None
        ):
            self._observer_track_call(
                "record_association",
                cur_id,
                fusion_inds,
                stage="spatial",
                _observer_scalar_positions=(0,),
                _observer_index_positions=(1,),
            )
        cur_box_size = box_size[cur_id,:3]
        small = False
        if np.max(cur_box_size)<self.small_size:
            small = True
        for idx in fusion_inds:
            # old boxes nms new box
            if len(self.fusion_list[idx]) == 1:
                count = 0 
                for i in self.fusion_list[cur_id]: 
                    baseline_gap, rotation_gap, disparity_score, center_dis= self.compute_pose_center_disparity(cam_poses[i], cam_poses[init_id[idx]], box_centers[cur_id], box_centers[idx])
            
                    if (baseline_gap > self.translation_gap or rotation_gap > self.rotation_gap) or center_dis>0.5:
                        count+=1

                # different from all the corresponding key boxes
                if count == len(self.fusion_list[cur_id]) and len(self.fusion_list[cur_id])<5:
       
                    self.fusion_list[cur_id] += [init_id[idx]]
                    self.fusion_list[cur_id].sort()
    
            # new box nms old boxes
            else:
                count = 0 
                for i in self.fusion_list[idx]: 
                    
                    baseline_gap, rotation_gap, disparity_score,center_dis = self.compute_pose_center_disparity(cam_poses[i], cam_poses[init_id[cur_id]], box_centers[cur_id], box_centers[idx])

                    if (baseline_gap > self.translation_gap or rotation_gap > self.rotation_gap) or center_dis>0.5:
                        count+=1

                # different from all the corresponding key boxes
                if count == len(self.fusion_list[idx]) and len(self.fusion_list[idx])<5:
    
                    self.fusion_list[cur_id] += self.fusion_list[idx]
                    self.fusion_list[cur_id].sort()
                else:
                    if cur_id in keep:
                        print("extra remove","cur_id",cur_id,'add:',idx)
                        keep.remove(cur_id)
                        keep.append(idx)
                
                if self.fusion_flag[idx] == 1:
                    self.fusion_flag[cur_id] = 1

        return keep

    def record_corr(self, cur_id, fusion_inds, init_id, cam_poses, keep):
        '''
        Note: cur_id is consistent to 'all_pred_box', idx is according to 'per_frame_box'
        '''
        if (
            self.observer_track_registry is not None
            and self.observer_track_token is not None
        ):
            self._observer_track_call(
                "record_association",
                cur_id,
                fusion_inds,
                stage="correspondence",
                _observer_scalar_positions=(0,),
                _observer_index_positions=(1,),
            )
        for idx in fusion_inds:
            # completely new boxes and no nms is valid
            if len(self.fusion_list[idx]) == 1:

                count = 0 
                for i in self.fusion_list[cur_id]: 
                    baseline_gap, rotation_gap, disparity_score = self.compute_pose_disparity(cam_poses[i], cam_poses[init_id[idx]])
        
                    if rotation_gap > self.rotation_gap or baseline_gap>self.translation_gap: #10: #30
                        count+=1
                # different from all the corresponding key boxes
                if count == len(self.fusion_list[cur_id]) and len(self.fusion_list[cur_id])<5:
                    self.fusion_list[cur_id] += [init_id[idx]]
                    self.fusion_list[cur_id].sort()

            # new box nms old boxes
            else:
                count = 0 
                for i in self.fusion_list[idx]: 
                    baseline_gap, rotation_gap, disparity_score = self.compute_pose_disparity(cam_poses[i], cam_poses[init_id[cur_id]])
 
                    if rotation_gap > self.rotation_gap or baseline_gap > self.translation_gap: #10: #30
                        count += 1
                # different from all the corresponding key boxes
                if count == len(self.fusion_list[idx]) and len(self.fusion_list[idx]) < 5:
                    self.fusion_list[cur_id] += self.fusion_list[idx]
                    self.fusion_list[cur_id].sort()
                else:
                    if cur_id in keep:
                        keep[keep == cur_id] = idx 


                if self.fusion_flag[idx] == 1:
                    self.fusion_flag[cur_id] = 1

        return keep
    
    def update(self, keep_idx):

        if (
            self.observer_track_registry is not None
            and self.observer_track_token is not None
        ):
            self._observer_track_call(
                "apply_keep",
                keep_idx,
                _observer_index_positions=(0,),
            )
        self.fusion_list = [self.fusion_list[i] for i in keep_idx] 
        if (
            self.observer_track_registry is not None
            and self.observer_track_token is not None
        ):
            self._observer_track_verify_rows()
        
    def update_fusion_flag(self, idx):
        self.fusion_flag[idx] = 1

    def get_fusion_idx(self):

        fusion_idx = [idx for idx in range(len(self.fusion_flag)) if self.fusion_flag[idx] == 1]
        
        return fusion_idx
    
    def get_nofusion_idx(self):

        fusion_idx = [idx for idx in range(len(self.fusion_flag)) if self.fusion_flag[idx] == 0]


        return fusion_idx
    
    def check_valid_num(self, all_pred_box, count, gap):

        box_frame_ids = all_pred_box.frame_id #[N] tenor
        valid_num = all_pred_box.valid_num
        zero_boxid = torch.where((valid_num == 0) & (box_frame_ids < (count - gap)))[0] # not seen two times

        valid_boxid = torch.arange(len(all_pred_box))
        if zero_boxid.shape[0] > 0:
            for idx in zero_boxid:
                valid_boxid = valid_boxid[valid_boxid != idx]

        if (
            self.observer_track_registry is not None
            and self.observer_track_token is not None
        ):
            self._observer_track_call(
                "apply_keep",
                valid_boxid,
                _observer_index_positions=(0,),
            )
        # update fusion_list
        self.fusion_list = [self.fusion_list[int(i)] for i in valid_boxid] 
        if (
            self.observer_track_registry is not None
            and self.observer_track_token is not None
        ):
            self._observer_track_verify_rows()

        all_pred_box = all_pred_box[valid_boxid]
        return all_pred_box

    def compute_pose_disparity(self, pose1, pose2):

        R1 = pose1[:3, :3]
        t1 = pose1[:3, 3]
        R2 = pose2[:3, :3]
        t2 = pose2[:3, 3]

        baseline = torch.norm(t2 - t1, p=2)

        R_rel = R2 @ R1.T  # R_rel = R2 * R1^T


        trace = torch.clamp((torch.trace(R_rel) - 1) / 2, min=-1.0, max=1.0)  
        rotation_angle = torch.arccos(trace) * 180 / torch.pi 


        disparity_score = 0.6 * baseline + 0.4 * rotation_angle

        return baseline, rotation_angle, disparity_score
    
    def compute_pose_center_disparity(self, pose1, pose2, center1, center2):


        R1 = pose1[:3, :3]
        t1 = pose1[:3, 3]
        R2 = pose2[:3, :3]
        t2 = pose2[:3, 3]


        baseline = torch.norm(t2 - t1, p=2)


        R_rel = R2 @ R1.T  # R_rel = R2 * R1^T

  
        trace = torch.clamp((torch.trace(R_rel) - 1) / 2, min=-1.0, max=1.0) 
        rotation_angle = torch.arccos(trace) * 180 / torch.pi 

  
        disparity_score = 0.6 * baseline + 0.4 * rotation_angle

        center_dis = self.euclidean_distance_3d(center1, center2)


        return baseline, rotation_angle, disparity_score, center_dis
    
    def euclidean_distance_3d(self,point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def check_uv_bounds(self, uv_coords, W, H, ratio=1.0):
        gap_W = int((1-ratio)*W)
        gap_H = int((1-ratio)*H)
        # uv_coords: [N, 2] array
        u = uv_coords[:, 0]
        v = uv_coords[:, 1]
        mask = (u > gap_W) & (u < (W-gap_W)) & (v > gap_H) & (v < (H-gap_H))
        
        return mask #.astype(int)

    def check_floor_mask(self, box_3d, ratio=20):
        
        bos_size = box_3d[:, 3:]
        max_values = torch.amax(bos_size, dim=1)  
        min_values = torch.amin(bos_size, dim=1) 
        second_values = torch.sort(bos_size, dim=1, descending=True)[0][:, 1]
        # mask = (max_values > 2) & (min_values<0.15)
        mask = (max_values/min_values > ratio)
        second_mask = (max_values/min_values > ratio/2) & (max_values/second_values > ratio/2) & (second_values/min_values<2.0) & (second_values<0.15) & (min_values<0.15)
        mask = mask | second_mask
        return mask #.astype(int)

   

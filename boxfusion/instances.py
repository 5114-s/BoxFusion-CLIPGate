# Based on D2's Instances.
import itertools
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial import ConvexHull
import torchvision.transforms as tvf

import torchvision.transforms as T
import matplotlib.pyplot as plt

ImgNorm = tvf.Compose([
    # tvf.ToTensor(),  
    tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])


def _as_float_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def cosine_similarity_to_many(query_feature, candidate_features, eps=1e-8):
    """Cosine similarity between one CLIP feature and a feature matrix."""
    query = _as_float_numpy(query_feature).reshape(-1)
    candidates = _as_float_numpy(candidate_features)
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)
    if candidates.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if candidates.shape[1] != query.shape[0]:
        raise ValueError(
            "Appearance feature dimension mismatch: "
            f"{query.shape[0]} versus {candidates.shape[1]}"
        )

    query_norm = max(float(np.linalg.norm(query)), eps)
    candidate_norms = np.maximum(
        np.linalg.norm(candidates, axis=1),
        eps,
    )
    similarities = (candidates @ query) / (candidate_norms * query_norm)
    return np.clip(similarities, -1.0, 1.0).astype(np.float32)


def resolve_appearance_gate_config(gate_cfg, stage):
    """Merge shared gate settings with optional stage-specific overrides."""
    gate_cfg = gate_cfg or {}
    resolved = {
        key: value
        for key, value in gate_cfg.items()
        if key not in ("spatial", "correspondence")
    }
    stage_cfg = gate_cfg.get(stage, {})
    if stage_cfg is not None:
        if not isinstance(stage_cfg, dict):
            raise ValueError(
                f"appearance_gate.{stage} must be a mapping"
            )
        resolved.update(stage_cfg)
    return resolved


def appearance_gate_decisions(
    geometry_scores,
    similarities,
    query_score,
    candidate_scores,
    base_threshold,
    gate_cfg=None,
):
    """
    Apply a confidence-aware soft appearance gate to geometric associations.

    Low CLIP similarity raises the required IoU instead of hard-rejecting a
    match. High similarity may lower it when ``max_iou_bonus`` is non-zero.
    Detection confidence controls how strongly appearance can change the
    geometric threshold. A hard geometric overlap always remains sufficient.
    """
    geometry = _as_float_numpy(geometry_scores).reshape(-1)
    candidate_scores = _as_float_numpy(candidate_scores).reshape(-1)
    if candidate_scores.size == 1 and geometry.size != 1:
        candidate_scores = np.full_like(geometry, candidate_scores.item())
    if candidate_scores.shape != geometry.shape:
        raise ValueError(
            "candidate_scores and geometry_scores must have the same shape"
        )

    cfg = gate_cfg or {}
    enabled = bool(cfg.get("enabled", False)) and similarities is not None
    thresholds = np.full_like(geometry, float(base_threshold))
    geometry_min = float(cfg.get("geometry_min_iou", base_threshold))
    hard_geometry = float(cfg.get("hard_geometry_iou", 1.0))

    if not enabled:
        accepted = geometry > float(base_threshold)
        return {
            "accepted": accepted,
            "thresholds": thresholds,
            "geometry_candidates": accepted.copy(),
            "similarities": None,
            "reliability": np.zeros_like(geometry),
            "appearance_used": False,
            "protected": np.zeros_like(accepted),
            "promoted": np.zeros_like(accepted),
            "hard_overrides": np.zeros_like(accepted),
            "penalized": np.zeros_like(accepted),
            "bonused": np.zeros_like(accepted),
        }

    similarities = _as_float_numpy(similarities).reshape(-1)
    if similarities.shape != geometry.shape:
        raise ValueError(
            "similarities and geometry_scores must have the same shape"
        )

    low_similarity = float(cfg.get("low_similarity", 0.45))
    high_similarity = float(cfg.get("high_similarity", 0.75))
    if not low_similarity < high_similarity:
        raise ValueError(
            "appearance_gate.low_similarity must be smaller than "
            "appearance_gate.high_similarity"
        )
    if geometry_min > float(base_threshold):
        raise ValueError(
            "appearance_gate.geometry_min_iou cannot exceed the base IoU threshold"
        )

    confidence_floor = float(cfg.get("confidence_floor", 0.35))
    confidence_full = float(cfg.get("confidence_full", 0.75))
    if confidence_full <= confidence_floor:
        raise ValueError(
            "appearance_gate.confidence_full must exceed confidence_floor"
        )

    query_scores = np.full_like(geometry, float(query_score))
    pair_confidence = np.minimum(query_scores, candidate_scores)
    reliability = np.clip(
        (pair_confidence - confidence_floor)
        / (confidence_full - confidence_floor),
        0.0,
        1.0,
    )

    # Map cosine similarity to [-1, 1]: negative values protect against
    # false merges, positive values can promote a geometrically plausible pair.
    evidence = np.interp(
        similarities,
        [low_similarity, high_similarity],
        [-1.0, 1.0],
    ).astype(np.float32)
    penalty = (
        np.maximum(-evidence, 0.0)
        * float(cfg.get("max_iou_penalty", 0.10))
        * reliability
    )
    bonus = (
        np.maximum(evidence, 0.0)
        * float(cfg.get("max_iou_bonus", 0.0))
        * reliability
    )
    thresholds = np.clip(
        float(base_threshold) + penalty - bonus,
        geometry_min,
        1.0,
    )

    geometry_candidates = geometry > geometry_min
    hard_overrides = geometry_candidates & (geometry >= hard_geometry)
    accepted = geometry_candidates & (
        (geometry > thresholds) | hard_overrides
    )
    baseline_accepted = geometry > float(base_threshold)

    return {
        "accepted": accepted,
        "thresholds": thresholds,
        "geometry_candidates": geometry_candidates,
        "similarities": similarities,
        "reliability": reliability,
        "appearance_used": True,
        "protected": baseline_accepted & ~accepted,
        "promoted": ~baseline_accepted & accepted,
        "hard_overrides": hard_overrides & (geometry <= thresholds),
        "penalized": thresholds > float(base_threshold) + 1e-8,
        "bonused": thresholds < float(base_threshold) - 1e-8,
    }


def nms_3d(instance_lists, box_manager, boxes, scores, init_id, cam_poses, box_size, iou_threshold=0.5, merge_upper=0.7, merge_lower=0.3): #merge_lower=0.3/0.1
    """
    Performs 3D Non-Maximum Suppression (NMS) on bounding boxes.
    This function implements 3D NMS to filter overlapping 3D bounding boxes based on their scores and IoU.
    Boxes with higher scores are given priority, and boxes with IoU above the threshold are suppressed.
    Args:
        instance_lists: Object containing instance tracking information
        box_manager: Manager object for handling box operations and recording fusion history
        boxes (numpy.ndarray): Array of 3D bounding boxes, shape [N, 8, 3] where N is number of boxes
        scores (numpy.ndarray): Detection confidence scores for each box, shape [N]
        init_id (numpy.ndarray): Initial IDs for each box
        cam_poses: Camera poses corresponding to each detection
        box_size: Size parameters for the bounding boxes
        iou_threshold (float, optional): IoU threshold for suppression. Defaults to 0.5
        merge_upper (float, optional): Upper threshold for merging. Defaults to 0.7
        merge_lower (float, optional): Lower threshold for merging. Defaults to 0.3
    Returns:
        tuple: A pair of numpy arrays:
            - keep: Indices of boxes to keep after NMS
            - success_nms: Indices of boxes that were successfully merged during NMS
    Notes:
        The function processes boxes in descending order of scores, keeping high-scoring boxes
        and removing lower-scoring boxes that have high IoU with already-selected boxes.
        It also tracks fusion history through the box_manager.
    """

    #boxes_center
    boxes_centers = np.mean(boxes,axis=1) #[N,3]

    # sort according to scores
    order = scores.argsort()[::-1] #index large->small
    order_init_id = init_id.tolist()

    keep = []
    success_nms = []
    gate_cfg = resolve_appearance_gate_config(
        box_manager.cfg.get("association", {}).get(
            "appearance_gate", {}
        ),
        "spatial",
    )
    appearance_features = None
    if (
        gate_cfg.get("enabled", False)
        and instance_lists.has("appearance_features")
    ):
        appearance_features = _as_float_numpy(
            instance_lists.appearance_features
        )

    while order.size > 0:
        nms_box_inds = []
        
        i = order[0]
        
        keep.append(i)
        
        temp_order = order[1:]
        ious = calculate_obb_iou(boxes[i], boxes[order[1:]])

        similarities = None
        if appearance_features is not None:
            similarities = cosine_similarity_to_many(
                appearance_features[i],
                appearance_features[temp_order],
            )
        gate_result = appearance_gate_decisions(
            ious,
            similarities,
            scores[i],
            scores[temp_order],
            iou_threshold,
            gate_cfg,
        )
        box_manager.record_appearance_gate("spatial", gate_result)

        inds = np.where(~gate_result["accepted"])[0]
        # i nms others, and is valid
        associate_inds = np.where(gate_result["accepted"])[0]
        if associate_inds.shape[0]>=1:
            instance_lists.valid_num[i] +=1
            print(
                'nms', i, '->', order[1 + associate_inds],
                'iou:', np.round(ious[associate_inds], 3),
                'clip:', (
                    np.round(similarities[associate_inds], 3)
                    if similarities is not None else 'disabled'
                ),
            )

        nms_inds = associate_inds
        nms_inds = np.asarray(nms_inds)
        '''
        record the fusion history
        '''
        if len(nms_inds)>0:

            success_nms.append(i)

            for j in temp_order[nms_inds]:
                nms_box_inds.append(j)
   
            '''
            record and update the fusion list
            '''
            keep = box_manager.record(i, nms_box_inds, order_init_id, cam_poses, box_size, keep, boxes_centers)        

        order = order[inds + 1] # +1 because inds is for temp_order

        if order.size == 1:
            keep.append(order[0])
            break

    keep.sort()
    success_nms.sort()
    return np.array(keep), np.array(success_nms)




def calculate_obb_iou(corners1, corners_others):
    """
    Calculate Intersection over Union (IoU) between one oriented bounding box and multiple others.
    Args:
        corners1 (numpy.ndarray): Corners of the reference oriented bounding box.
            Shape should be (8, 3) representing 8 corners with XYZ coordinates.
        corners_others (numpy.ndarray): Corners of multiple oriented bounding boxes to compare with.
            Shape should be (N, 8, 3) where N is the number of boxes.
    Returns:
        numpy.ndarray: Array of IoU values between the reference box and each other box.
            Shape is (N,) where N is the number of boxes in corners_others.
    Note:
        Uses Instances3D.obb_iou for individual IoU calculations between pairs of boxes.
    """
    
    iou = [Instances3D.obb_iou(corners1,corners_others[i]) for i in range(corners_others.shape[0])]

    iou = np.asarray(iou) 

    return iou

# Provides basic compatibility with D2.
class Instances3D:
    """
    This class represents a list of instances in _the world_.
    """
    def __init__(self, image_size: Tuple[int, int] = (0, 0), **kwargs: Any):
        # image_size is here for Detectron2 compatibility.
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width (note: opposite of cubifycore).

        Here for D2 compatibility. You probably shouldn't be using this.
        """
        return self._image_size            

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances3D!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        with warnings.catch_warnings(record=True):
            data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances3D of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances3D":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances3D(image_size=self._image_size)
        # Copy fields that were explicitly added to this object (e.g., hidden fields)
        for name, value in self.__dict__.items():
            if (name not in ["_fields"]) and name.startswith("_"):
                setattr(ret, name, value.to(*args, **kwargs) if hasattr(value, "to") else value)
        
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)

        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances3D":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances3D` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances3D index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances3D(image_size=self.image_size)
        for name, value in self.__dict__.items():
            if (name not in ["_fields"]) and name.startswith("_"):
                setattr(ret, name, value)
        
        for k, v in self._fields.items():
            if isinstance(v, (torch.Tensor, np.ndarray)) or hasattr(v, "tensor"):
                # assume if has .tensor, then this is piped into __getitem__.
                # Make sure to match underlying types.
                if isinstance(v, np.ndarray) and isinstance(item, torch.Tensor):
                    ret.set(k, v[item.cpu().numpy()])
                else:
                    ret.set(k, v[item])
            elif hasattr(v, "__iter__"):
                # handle non-Tensor types like lists, etc.
                if isinstance(item, np.ndarray) and (item.dtype == np.bool_):
                    ret.set(k, [v_ for i_, v_ in enumerate(v) if item[i_]])                    
                elif isinstance(item, torch.BoolTensor) or (isinstance(item, torch.Tensor) and (item.dtype == torch.bool)):
                    ret.set(k, [v_ for i_, v_ in enumerate(v) if item[i_].item()])
                elif isinstance(item, torch.LongTensor) or (isinstance(item, torch.Tensor) and (item.dtype == torch.int64)):
                    # Can this be right?
                    ret.set(k, [v[i_.item()] for i_ in item])
                elif isinstance(item, slice):
                    ret.set(k, v[item])
                else:
                    raise ValueError("Expected Bool or Long Tensor")
            else:
                raise ValueError("Not supported!")
                
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances3D does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances3D` object is not iterable!")

    def split(self, split_size_or_sections):
        indexes = torch.arange(len(self))
        splits = torch.split(indexes, split_size_or_sections)

        return [self[split] for split in splits]

    def clone(self):
        import copy

        ret = Instances3D(image_size=self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "clone"):
                v = v.clone()
            elif isinstance(v, np.ndarray):
                v = np.copy(v)
            elif isinstance(v, (str, list, tuple)):
                v = copy.copy(v)
            elif hasattr(v, "tensor"):
                v = type(v)(v.tensor.clone())
            else:
                raise NotImplementedError

            ret.set(k, v)

        return ret

    @staticmethod
    def cat(instance_lists: List["Instances3D"]) -> "Instances3D":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances3D) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        ret = Instances3D(image_size=instance_lists[0]._image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):      
                values = np.concatenate(values, axis=0)  
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret
    
    def project_3d_boxes(self, K, H=480, W=640):
        """
        Project 3D boxes to 2D image plane.
        Returns:
            projected_boxes: (8,3) numpy array, where N is the number of boxes.
                              Each box is represented as [x1, y1, x2, y2].
        """
        
        boxes = self.get('pred_boxes_3d')
        corners = boxes.corners #[N,8,3]
        cam_pose = self.cam_pose #[N,4,4]

        N = corners.shape[0]
        
        ones = torch.ones((N, 8, 1), device=corners.device)
        boxes_homo = torch.cat([corners, ones], dim=2)  # [N,8,4]
        
        pose_inv = torch.linalg.inv(cam_pose).to(corners.device)
        
        boxes_cam = torch.einsum('nij,nkj->nki', pose_inv, boxes_homo)  
        
        X = boxes_cam[..., 0]
        Y = boxes_cam[..., 1]
        Z = boxes_cam[..., 2]

        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = (fx * X / Z) + cx
        v = (fy * Y / Z) + cy

        u = torch.clamp(u,0,W)
        v = torch.clamp(v,0,H)
        
        projected_boxes = torch.stack([u, v], dim=-1) #[N,8,2]

        self.projected_boxes = projected_boxes


    def spatial_association(instance_lists, threshold, box_manager, cam_poses) :
        """
        Args:
            instance Instances

        Returns:
            Instances
        """
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists

        if box_manager.causal_hungarian.enabled:
            result = box_manager.causal_hungarian.associate(
                instance_lists,
                box_manager,
                cam_poses,
            )
            return (
                result.keep_indices.tolist(),
                result.correspondence_skip_indices.tolist(),
            )

        boxes_now = instance_lists.get('pred_boxes_3d')

        boxes_corners = boxes_now.corners.cpu().numpy() #[N,8,3]
        box_size = boxes_now.dims.cpu().numpy()  

        scores = instance_lists.scores.cpu().numpy()  # confidence
        init_id = instance_lists.init_id.cpu().numpy() 

        # Execute spatial association
        keep, success_nms = nms_3d(instance_lists, box_manager, boxes_corners, scores, init_id, cam_poses, box_size, iou_threshold=threshold)
        keep = sorted(keep)
        success_nms = sorted(success_nms)

        return  keep,success_nms #, nms_inds


    def modify_instance(ind_old,ind_new,old_ins,new_ins):
        old_ins.scores[ind_old] = new_ins.scores[ind_new]
        old_ins.pred_classes[ind_old] = new_ins.pred_classes[ind_new]
        old_ins.pred_boxes[ind_old] = new_ins.pred_boxes[ind_new]
        old_ins.pred_logits[ind_old] = new_ins.pred_logits[ind_new]
        old_ins.pred_boxes_3d.tensor[ind_old] = new_ins.pred_boxes_3d.tensor[ind_new]
        old_ins.pred_boxes_3d.R[ind_old] = new_ins.pred_boxes_3d.R[ind_new]
        old_ins.object_desc[ind_old] = new_ins.object_desc[ind_new]
        old_ins.pred_proj_xy[ind_old] = new_ins.pred_proj_xy[ind_new]


    def correspondence_association(cfg, box_manager, cur_keep_idx, cur_success_nms, pred_instances, global_pred_box, all_pred_box, all_poses, per_frame_ins_cam_pose, frame_id, mask, intrinsic, all_kf_pose, threshold=0.33, H=480, W=640):  

        N_glo = len(global_pred_box)
    
        cur_2d_box = pred_instances.pred_boxes.cpu().numpy()
        cur_2d_box_scores = pred_instances.scores.cpu().numpy()
        global_box_scores = global_pred_box.scores.cpu().numpy()
        gate_cfg = resolve_appearance_gate_config(
            cfg.get("association", {}).get("appearance_gate", {}),
            "correspondence",
        )
        current_appearance = None
        global_appearance = None
        if (
            gate_cfg.get("enabled", False)
            and pred_instances.has("appearance_features")
            and global_pred_box.has("appearance_features")
        ):
            current_appearance = _as_float_numpy(
                pred_instances.appearance_features
            )
            global_appearance = _as_float_numpy(
                global_pred_box.appearance_features
            )
        pred_instances_pred_boxes_3d = pred_instances.get("pred_boxes_3d")
        pred_box_size = pred_instances_pred_boxes_3d.dims.cpu().numpy()

        init_id = all_pred_box.init_id.cpu().numpy()  # 

        keep_idx = copy.deepcopy(np.asarray(mask))
        global_keep_idx = keep_idx[keep_idx<N_glo]

        small_idx = []
     
        for idx in cur_keep_idx:
            cur_box_size = pred_box_size[idx,:3]
            '''
            only deal with the small ones length/weight/height < 35cm
            '''
            if np.max(cur_box_size)>cfg['box_fusion']['small_size'] or idx in cur_success_nms:
                continue
            small_idx.append(idx)


        if len(small_idx) == 0:
            keep_idx = np.sort(keep_idx)
            all_pred_box = all_pred_box[keep_idx]
            all_poses = all_poses[keep_idx]
            return all_pred_box, all_poses, keep_idx

        cur_pose = all_kf_pose[frame_id] #[4,4]

        for idx in small_idx:
   
            boxes_global = global_pred_box.get('pred_boxes_3d')
            boxes_3d = boxes_global.corners.cpu().numpy()[global_keep_idx,...] # [N,8,3]


            boxes_2d = Instances3D.project_3d_to_2d_box(boxes_3d, intrinsic.cpu().numpy(), cur_pose, H, W, frame_id=frame_id) #[N_glo-nms,4]

            cur_small_box_2d = cur_2d_box[idx]
            if len(boxes_2d) == 0:
                continue
            box_iou = Instances3D.IoU_2D_box(cur_small_box_2d, boxes_2d) #[N_glo-nms]

            boxes_3d_dims = boxes_global.dims.cpu().numpy()[global_keep_idx, ...] # #[N_glo-nms, 3]
            global_small_mask = np.max(boxes_3d_dims, axis=1) < cfg['box_fusion']['small_size'] + 0.1 # [N_glo-nms]
            box_iou = box_iou * global_small_mask

            similarities = None
            if current_appearance is not None and global_appearance is not None:
                similarities = cosine_similarity_to_many(
                    current_appearance[idx],
                    global_appearance[global_keep_idx],
                )
            gate_result = appearance_gate_decisions(
                box_iou,
                similarities,
                cur_2d_box_scores[idx],
                global_box_scores[global_keep_idx],
                threshold,
                gate_cfg,
            )
            box_manager.record_appearance_gate(
                "correspondence", gate_result
            )

            accepted_ids = np.flatnonzero(gate_result["accepted"])
            if accepted_ids.size == 0:
                continue
            association_margin = (
                box_iou[accepted_ids]
                - gate_result["thresholds"][accepted_ids]
            )
            corresponding_boxid = accepted_ids[
                np.argmax(association_margin)
            ]

            '''
            box with large IoU with old boxes in past keyframes
            '''
            if gate_result["accepted"][corresponding_boxid]:
                corresponding_idx = global_keep_idx[corresponding_boxid]
                clip_similarity = (
                    float(similarities[corresponding_boxid])
                    if similarities is not None else None
                )
                
                if global_box_scores[corresponding_idx] < cur_2d_box_scores[idx]:
                    print("frame_id:", frame_id, "find corr", idx, "remove old", corresponding_idx, 'iou:', box_iou[corresponding_boxid], 'clip:', clip_similarity, "better:", idx)
                    keep_idx = keep_idx[keep_idx != corresponding_idx]

                    # record box manager
                    all_pred_box.valid_num[idx + N_glo] += 1
                    keep_idx = box_manager.record_corr(idx + N_glo, [corresponding_idx], init_id, per_frame_ins_cam_pose, keep_idx)
                else:
                    print("frame_id:", frame_id, "find corr", idx, "remove new", idx + N_glo, "old:", corresponding_idx, 'iou:', box_iou[corresponding_boxid], 'clip:', clip_similarity, 'worse')
                    keep_idx = keep_idx[keep_idx != (idx + N_glo)]
                    # record box manager
                    all_pred_box.valid_num[corresponding_idx] += 1
                    keep_idx = box_manager.record_corr(corresponding_idx, [idx+N_glo], init_id, per_frame_ins_cam_pose,keep_idx)


        keep_idx = np.sort(keep_idx)
        all_pred_box = all_pred_box[keep_idx]
        all_poses = all_poses[keep_idx]

        return all_pred_box, all_poses, keep_idx
    
 
    def augment_vertices(corners):
        
        edges = [
            [0, 1], [0, 4], [1, 5], [4, 5],
            [2, 3], [2, 6], [6, 7], [3, 7],
            [0, 3], [4, 7], [1, 2], [5, 6]
        ]

        
        midpoints = []
        for edge in edges:
            v1 = corners[edge[0]]
            v2 = corners[edge[1]]
            midpoint = (v1 + v2) / 2
            midpoints.append(midpoint)

        # Combine original vertices with midpoints
        combined = np.vstack([corners, midpoints])
        
        return combined

    def check_intersection(corners1, corners2):
        """
        Check if two 3D convex hulls intersect based on their corner points.
        This function uses the Separating Axis Theorem (SAT) to determine if two convex
        hulls defined by their corner points intersect in 3D space.
        Args:
            corners1 (numpy.ndarray): Corner points of the first convex hull, shape (N,3)
            corners2 (numpy.ndarray): Corner points of the second convex hull, shape (M,3)
        Returns:
            bool: True if the convex hulls intersect, False otherwise
        Notes:
            - Uses ConvexHull from scipy.spatial to compute hull equations
            - Performs intersection test by checking if any points from one hull
              lie inside the other hull using hull equations
            - A small epsilon (1e-6) is used for numerical stability in comparisons
        """
        
        
        hull1 = ConvexHull(corners1)
        hull2 = ConvexHull(corners2)

        corners1 = Instances3D.augment_vertices(corners1)
        corners2 = Instances3D.augment_vertices(corners2)

        equations1 = hull1.equations  #  [K,4]
        equations2 = hull2.equations 
        


        dot_products1 = np.dot(corners1, equations2[:, :3].T) + equations2[:, 3]  

        mask1 = np.all(dot_products1 <= 1e-6, axis=1) 


        dot_products2 = np.dot(corners2, equations1[:, :3].T) + equations1[:, 3]  

        mask2 = np.all(dot_products2 <= 1e-6, axis=1)  

        sum_of_mask = np.sum(mask1) + np.sum(mask2)

        if sum_of_mask > 0:
            return True
        else:
            return False
    


    def batch_in_convex_hull_3d(points, corners):

        hull = ConvexHull(corners)
        equations = hull.equations  

        dot_products = np.dot(points, equations[:, :3].T) + equations[:, 3]  
        

        mask = np.all(dot_products <= 1e-6, axis=1)  
        
        return mask

    def obb_iou(corners1,corners2):
  
        results = Instances3D.check_intersection(corners1,corners2)

        if results:

            all_corners = np.concatenate([corners1, corners2], axis=0)

            xmin, ymin, zmin = np.min(all_corners, axis=0)
            xmax, ymax, zmax = np.max(all_corners, axis=0)

           
            num_samples_per_axis = 10

            
            x_samples = np.linspace(xmin, xmax, num_samples_per_axis)
            y_samples = np.linspace(ymin, ymax, num_samples_per_axis)
            z_samples = np.linspace(zmin, zmax, num_samples_per_axis)

            
            xx, yy, zz = np.meshgrid(x_samples, y_samples, z_samples, indexing='ij')

            
            sampled_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

            
            #TODO:easy to stuck here, if too many boxes
            mask1 = Instances3D.batch_in_convex_hull_3d(sampled_points, corners1)
            mask2 = Instances3D.batch_in_convex_hull_3d(sampled_points, corners2)
            common_mask = mask1 * mask2
            count1 = np.sum(mask1)
            count2 = np.sum(mask2)
            common_count = np.sum(common_mask)
 
            # compute IoU according to the mask
            IoU = common_count/(count1+count2-common_count+1e-6)

            return IoU
        else:
            IoU = 0.0
            return IoU


    def IoU_2D(A, B):
        
        A = A.astype(np.float64)
        x_min_A, y_min_A = np.min(A, axis=0)
        x_max_A, y_max_A = np.max(A, axis=0)
        area_A = (x_max_A - x_min_A) * (y_max_A - y_min_A)

        x_min_B = B[:, 0]
        y_min_B = B[:, 1]
        x_max_B = B[:, 2]
        y_max_B = B[:, 3]
        area_B = (x_max_B - x_min_B) * (y_max_B - y_min_B)

        x_min_inter = np.maximum(x_min_A, x_min_B)
        y_min_inter = np.maximum(y_min_A, y_min_B)
        x_max_inter = np.minimum(x_max_A, x_max_B)
        y_max_inter = np.minimum(y_max_A, y_max_B)

        inter_width = np.maximum(0, x_max_inter - x_min_inter)
        inter_height = np.maximum(0, y_max_inter - y_min_inter)
        inter_area = inter_width * inter_height

        union_area = area_A + area_B - inter_area
        iou = inter_area / (union_area + 1e-6)
        overlap_A = inter_area / (area_A+1e-6)
        return iou, overlap_A
    
    def IoU_2D_box(A, B):
        
        A = A.astype(np.float64)
        x_min_A, y_min_A, x_max_A, y_max_A  = A[0],A[1],A[2],A[3]

        area_A = (x_max_A - x_min_A) * (y_max_A - y_min_A)

        x_min_B = B[:, 0]
        y_min_B = B[:, 1]
        x_max_B = B[:, 2]
        y_max_B = B[:, 3]
        area_B = (x_max_B - x_min_B) * (y_max_B - y_min_B)

        x_min_inter = np.maximum(x_min_A, x_min_B)
        y_min_inter = np.maximum(y_min_A, y_min_B)
        x_max_inter = np.minimum(x_max_A, x_max_B)
        y_max_inter = np.minimum(y_max_A, y_max_B)

        inter_width = np.maximum(0, x_max_inter - x_min_inter)
        inter_height = np.maximum(0, y_max_inter - y_min_inter)
        inter_area = inter_width * inter_height

        union_area = area_A + area_B - inter_area
        iou = inter_area / (union_area + 1e-6)
        # overlap_A = inter_area / (area_A+1e-6)
        return iou

    def project_3d_to_2d_box(boxes_3d, K, pose, H, W, frame_id=None):
        
        N = boxes_3d.shape[0]
        boxes_2d = np.zeros((N, 4))
        
        # [N, 8, 4]
        ones = np.ones((N, 8, 1))
        boxes_homo = np.concatenate([boxes_3d, ones], axis=2)
        
        # [N, 8, 4]
        pose_inv = np.linalg.inv(pose)
        boxes_cam = np.dot(boxes_homo, pose_inv.T)
        
        # [N, 8, 3]
        X = boxes_cam[..., 0]
        Y = boxes_cam[..., 1]
        Z = boxes_cam[..., 2]
        
        # [N, 8, 2]
        u = (K[0, 0] * X / Z) + K[0, 2]
        v = (K[1, 1] * Y / Z) + K[1, 2]
        
        # 
        valid_mask = (Z > 0) * (u>0) * (u<W) * (v>0) * (v<H)

        for i in range(N):

            valid_u = u[i][valid_mask[i]]
            valid_v = v[i][valid_mask[i]]
            if len(valid_u) == 0:
                boxes_2d[i] = [0, 0, 0, 0]  
            else:
                valid_z = (Z > 0) * (Z<8) 
                if len(valid_z) == 0:
                    boxes_2d[i] = [0, 0, 0, 0]  
                else:
                    valid_u = u[i][valid_z[i]]
                    valid_v = v[i][valid_z[i]]
                    if len(valid_u) == 0 or len(valid_v)==0:
                        boxes_2d[i] = [0, 0, 0, 0]
                    else:
                        valid_u = np.clip(valid_u, 0, W)
                        valid_v = np.clip(valid_v, 0, H)
                        
                        boxes_2d[i] = [np.min(valid_u), np.min(valid_v), 
                                        np.max(valid_u), np.max(valid_v)]
        
        return boxes_2d
    
    
    def translate(self, translation):
        # in-place.
        for field_name, field in self._fields.items():
            if hasattr(field, "translate"):
                field.translate(translation)

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__

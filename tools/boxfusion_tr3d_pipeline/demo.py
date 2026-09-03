import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import glob
import itertools
import json
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
from contextlib import contextmanager

from boxfusion.cubify_transformer import make_cubify_transformer

from boxfusion.instances import Instances3D
from boxfusion.preprocessor import Augmentor, Preprocessor

from boxfusion.box_manager import BoxManager
from boxfusion.box_fusion import BoxFusion
from boxfusion.clip_instance_features import ClipInstanceFeatureEncoder
from boxfusion.online_refinement import (
    build_online_refinement_controller,
)
from boxfusion.online_ablation import (
    ONLINE_ABLATION_PROFILES,
    apply_online_ablation_profile,
)
from boxfusion.stable_ids import resolve_fusion_stable_ids
from boxfusion.moon_qim_lite import (
    CausalFusionIdRegistry,
    build_moon_qim_lite,
    derive_native_target_track_ids,
)
from boxfusion.puf_lite import build_puf_lite
from boxfusion.puf_arbitration_lite import build_puf_arbitration_lite
from boxfusion.depth_guide_geometry import sample_depth_guide_points_batch
from boxfusion.mv3dis_depth_lite import (
    build_mv3dis_depth_lite,
    derive_committed_track_ids,
)
from boxfusion.third_view_birth_lite import build_third_view_birth_lite
from boxfusion.side_birth_probation_lite import (
    SideBirthSeedEvent,
    build_side_birth_probation_lite,
)
from boxfusion.cutr_residual_birth_lite import (
    ResidualObservation,
    build_cutr_residual_birth_lite,
    partition_scores,
)
from boxfusion.cutr_residual_cross_view_r1 import (
    ResidualCrossViewEvidence,
    build_cutr_residual_cross_view_r1,
)
from boxfusion.boxer_lifter import build_lifting_adapter
from boxfusion.proposal_cache import build_proposal_cache
from boxfusion.tr3d_c3_online_identity import build_c3_online_identity_observer
from boxfusion.tr3d_terminal_active import (
    link_prediction_create_only,
    save_prediction_create_only,
)
from boxfusion.ca1m_native_b6_observer import (
    build_ca1m_native_b6_observer,
)
from boxfusion.openbox_smov_r2 import (
    build_openbox_smov_r2,
    save_r2_shadow_sidecar_create_only,
)


@contextmanager
def preserved_observer_rng_state():
    """Restore Python, NumPy, and Torch RNG streams after observer work."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


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


def run(
    cfg,
    model,
    dataset,
    clip_model,
    preprocess,
    tokenized_text,
    text_features,
    augmentor,
    preprocessor,
    score_thresh=0.0,
    viz_on_gt_points=False,
    gap=25,
    re_vis=True,
    online_same_run_anchor_root=None,
    ca1m_native_b6_same_run_anchor_root=None,
    prediction_same_run_anchor_root=None,
):
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
    proposal_cache = build_proposal_cache(cfg, device=device.device)
    cutr_residual_birth_lite = build_cutr_residual_birth_lite(cfg)
    cutr_residual_cross_view_r1 = build_cutr_residual_cross_view_r1(cfg)
    if cutr_residual_birth_lite.enabled:
        if proposal_cache is not None:
            raise ValueError(
                "CuTR residual birth-lite requires live proposals; proposal "
                "cache record/replay is unsupported"
            )
        configured_ceiling = float(
            cfg["cutr_residual_birth_lite"]["score_ceiling"]
        )
        native_threshold = float(cfg["detection"]["score_thresh"])
        if configured_ceiling != native_threshold:
            raise ValueError(
                "cutr_residual_birth_lite.score_ceiling must exactly equal "
                "detection.score_thresh"
            )

    def proposal_cache_inputs(sample, image):
        # Insertion order is part of the immutable cache schema.
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

    def make_cutr_residual_side(
        raw_predictions,
        current_image,
        actual_native_threshold=None,
        apply_floor_filter=True,
    ):
        """Copy the frozen low-score band before native code mutates rows."""

        if not cutr_residual_birth_lite.enabled:
            return None, np.empty((0,), dtype=np.int64)
        score_ceiling = (
            float(cfg["cutr_residual_birth_lite"]["score_ceiling"])
            if actual_native_threshold is None
            else float(actual_native_threshold)
        )
        raw_scores = np.array(
            raw_predictions.scores.detach().cpu().numpy(),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if score_ceiling <= 0.10:
            # The retry native band reaches the frozen residual floor, so no
            # score can be both residual and excluded from native output.
            raw_row_indices = np.empty((0,), dtype=np.int64)
        else:
            partition = partition_scores(
                raw_scores, score_ceiling=score_ceiling
            )
            raw_row_indices = np.asarray(
                partition.residual_indices, dtype=np.int64
            )
        row_selector = torch.as_tensor(
            raw_row_indices,
            dtype=torch.long,
            device=raw_predictions.scores.device,
        )
        residual_predictions = raw_predictions[row_selector].clone()

        # These are the only native proposal filters intentionally mirrored
        # onto the side clone.  The raw row numbers stay outside Instances3D
        # so they can never leak into CLIP, lifting, QIM, or BoxFusion.
        if cfg["detection"]["uv_bound"]:
            uv_mask = box_manager.check_uv_bounds(
                residual_predictions.pred_proj_xy,
                current_image.shape[1],
                current_image.shape[0],
                ratio=cfg["detection"]["uv_bound_value"],
            )
            residual_predictions = residual_predictions[uv_mask]
            raw_row_indices = raw_row_indices[
                np.asarray(uv_mask.detach().cpu().numpy(), dtype=np.bool_)
            ]
        if apply_floor_filter and cfg["detection"]["floor_mask"]:
            floor_mask = box_manager.check_floor_mask(
                residual_predictions.pred_boxes_3d.tensor,
                ratio=cfg["detection"]["floor_ratio"],
            )
            residual_predictions = residual_predictions[~floor_mask]
            raw_row_indices = raw_row_indices[
                ~np.asarray(
                    floor_mask.detach().cpu().numpy(), dtype=np.bool_
                )
            ]
        return residual_predictions, raw_row_indices

    def observe_cutr_residual_keyframe(
        frame_index,
        residual_predictions,
        raw_row_indices,
        frame_pose,
        native_predictions,
        sample,
    ):
        """Observe one true CuTR keyframe without touching native state."""

        if not cutr_residual_birth_lite.enabled:
            return
        if residual_predictions is None:
            raise RuntimeError(
                "enabled CuTR residual observer is missing live side rows"
            )

        def snapshot_native_fields(instances):
            """Copy every array-valued CuTR field for the observer guard."""

            snapshot = []
            for name, value in sorted(instances.get_fields().items()):
                if isinstance(value, torch.Tensor):
                    arrays = (value.detach().cpu().numpy(),)
                elif isinstance(value, np.ndarray):
                    arrays = (value,)
                elif hasattr(value, "tensor") and hasattr(value, "R"):
                    arrays = (
                        value.tensor.detach().cpu().numpy(),
                        value.R.detach().cpu().numpy(),
                    )
                else:
                    raise RuntimeError(
                        "CuTR native identity guard cannot snapshot field "
                        + str(name)
                    )
                snapshot.append(
                    (
                        str(name),
                        tuple(np.array(array, order="C", copy=True) for array in arrays),
                    )
                )
            return tuple(snapshot)

        native_before = snapshot_native_fields(native_predictions)
        side_pose = np.repeat(
            np.expand_dims(
                np.array(frame_pose, dtype=np.float32, copy=True), axis=0
            ),
            repeats=len(residual_predictions),
            axis=0,
        )
        # World lifting is performed only on the independent side clone.
        residual_predictions.pred_boxes_3d.transform2world(side_pose)
        side_scores = np.array(
            residual_predictions.scores.detach().cpu().numpy(),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        side_corners = np.array(
            residual_predictions.pred_boxes_3d.corners.detach()
            .cpu()
            .numpy(),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if not (
            len(raw_row_indices) == len(side_scores) == len(side_corners)
        ):
            raise RuntimeError("CuTR residual side rows lost alignment")
        observations = tuple(
            ResidualObservation(
                frame_id=int(frame_index),
                raw_index=int(raw_row_indices[row]),
                score=float(side_scores[row]),
                corners=side_corners[row],
            )
            for row in range(len(side_scores))
        )
        with preserved_observer_rng_state():
            base_result = cutr_residual_birth_lite.observe(
                frame_id=int(frame_index), observations=observations
            )

        if cutr_residual_cross_view_r1.enabled:
            wrapper_started = time.perf_counter()
            side_row_by_raw = {
                int(raw_index): int(row)
                for row, raw_index in enumerate(raw_row_indices)
            }
            if len(side_row_by_raw) != len(raw_row_indices):
                raise RuntimeError("CuTR residual raw row ids are not unique")
            assigned_side_rows = tuple(
                side_row_by_raw[assignment.raw_index]
                for assignment in base_result.assignments
            )
            if len(assigned_side_rows) > 64:
                raise RuntimeError("R1 descriptor copy exceeded the frozen cap")

            depth_m = np.array(
                sample["wide"]["depth"][-1].numpy(),
                dtype=np.float32,
                order="C",
                copy=True,
            )
            depth_K = np.array(
                sample["sensor_info"].wide.depth.K[-1].numpy(),
                dtype=np.float64,
                order="C",
                copy=True,
            )
            camera_to_world = np.array(
                frame_pose, dtype=np.float64, order="C", copy=True
            )
            evidence_rows = []
            if assigned_side_rows:
                selector = torch.as_tensor(
                    assigned_side_rows,
                    dtype=torch.long,
                    device=residual_predictions.scores.device,
                )
                try:
                    descriptors = np.array(
                        residual_predictions.object_desc[selector]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                        dtype=np.float32,
                        order="C",
                        copy=True,
                    )
                    raw_boxes_xyxy = np.array(
                        residual_predictions.pred_boxes[selector]
                        .detach()
                        .cpu()
                        .numpy(),
                        dtype=np.float64,
                        order="C",
                        copy=True,
                    )
                    world_boxes = residual_predictions.pred_boxes_3d
                    centers_world = np.array(
                        world_boxes.tensor[selector, :3]
                        .detach()
                        .cpu()
                        .numpy(),
                        dtype=np.float64,
                        order="C",
                        copy=True,
                    )
                    dimensions = np.array(
                        world_boxes.tensor[selector, 3:6]
                        .detach()
                        .cpu()
                        .numpy(),
                        dtype=np.float64,
                        order="C",
                        copy=True,
                    )
                    rotations_world = np.array(
                        world_boxes.R[selector].detach().cpu().numpy(),
                        dtype=np.float64,
                        order="C",
                        copy=True,
                    )
                    guides = sample_depth_guide_points_batch(
                        depth_m,
                        depth_K,
                        camera_to_world,
                        raw_boxes_xyxy,
                        centers_world,
                        dimensions,
                        rotations_world,
                    )
                    if not (
                        len(descriptors)
                        == len(raw_boxes_xyxy)
                        == len(guides)
                        == len(base_result.assignments)
                    ):
                        raise ValueError("R1 copied rows lost alignment")
                    for row, assignment in enumerate(base_result.assignments):
                        guide = guides[row]
                        if guide is None:
                            evidence_rows.append(
                                ResidualCrossViewEvidence.abstain(
                                    int(frame_index),
                                    assignment.raw_index,
                                    "invalid_depth_guide",
                                )
                            )
                            continue
                        try:
                            evidence_rows.append(
                                ResidualCrossViewEvidence(
                                    frame_id=int(frame_index),
                                    raw_index=assignment.raw_index,
                                    descriptor=descriptors[row],
                                    camera_to_world=camera_to_world,
                                    raw_box_xyxy=raw_boxes_xyxy[row],
                                    guide_points_world=guide.points_world,
                                )
                            )
                        except ValueError:
                            evidence_rows.append(
                                ResidualCrossViewEvidence.abstain(
                                    int(frame_index),
                                    assignment.raw_index,
                                    "invalid_r1_evidence",
                                )
                            )
                except (AttributeError, IndexError, RuntimeError, ValueError):
                    evidence_rows = [
                        ResidualCrossViewEvidence.abstain(
                            int(frame_index),
                            assignment.raw_index,
                            "r1_batch_extraction_failed",
                        )
                        for assignment in base_result.assignments
                    ]

            with preserved_observer_rng_state():
                cutr_residual_cross_view_r1.observe(
                    frame_id=int(frame_index),
                    evidence_rows=tuple(evidence_rows),
                    base_result=base_result,
                    depth_m=depth_m,
                    K=depth_K,
                    T_wc=camera_to_world,
                )
            cutr_residual_cross_view_r1.record_wrapper_timing(
                (time.perf_counter() - wrapper_started) * 1000.0
            )

        native_after = snapshot_native_fields(native_predictions)
        native_identity = (
            len(native_before) == len(native_after)
            and all(
                before_name == after_name
                and len(before_arrays) == len(after_arrays)
                and all(
                    np.array_equal(before, after)
                    for before, after in zip(before_arrays, after_arrays)
                )
                for (before_name, before_arrays), (
                    after_name,
                    after_arrays,
                ) in zip(native_before, native_after)
            )
        )
        if not native_identity:
            raise RuntimeError(
                "CuTR residual keyframe observer mutated native proposals"
            )
        return base_result

    count=0
    all_pred_box = None
    all_poses = None

    all_kf_pose = {}
    per_frame_ins = None #save every predicted boxes
    traj_xyz = []

    box_manager = BoxManager(cfg)
    Box_Fuser = BoxFusion(cfg)
    moon_qim_lite = build_moon_qim_lite(cfg)
    puf_lite = build_puf_lite(cfg)
    puf_arbitration_lite = build_puf_arbitration_lite(cfg)
    mv3dis_depth_lite = build_mv3dis_depth_lite(cfg)
    third_view_birth_lite = build_third_view_birth_lite(cfg)
    side_birth_probation_lite = build_side_birth_probation_lite(cfg)
    openbox_smov_r2 = build_openbox_smov_r2(cfg)
    if puf_lite.enabled and not moon_qim_lite.enabled:
        raise ValueError("PUF-lite requires enabled Moon-QIM-lite")
    if puf_arbitration_lite.enabled and not puf_lite.enabled:
        raise ValueError("PUF arbitration-lite requires enabled PUF-lite")
    if mv3dis_depth_lite.enabled and not moon_qim_lite.enabled:
        raise ValueError("MV3DIS-Depth-lite requires enabled Moon-QIM-lite")
    if third_view_birth_lite.enabled and not moon_qim_lite.enabled:
        raise ValueError(
            "third-view birth-lite requires enabled Moon-QIM-lite"
        )
    if side_birth_probation_lite.enabled and not puf_arbitration_lite.enabled:
        raise ValueError(
            "side-birth probation-lite requires enabled PUF arbitration-lite"
        )
    moon_qim_identity = CausalFusionIdRegistry()
    # R2 owns an independent stable-ID registry, but it consumes the same
    # immutable native association events as Moon.  Recording must therefore
    # remain enabled when R2 is run as an isolated ablation.
    box_manager.record_merge_events = (
        moon_qim_lite.enabled or openbox_smov_r2.enabled
    )
    online_cfg = cfg.get("online_refinement", {})
    c3_online_observer = build_c3_online_identity_observer(cfg)
    ca1m_native_b6_observer = build_ca1m_native_b6_observer(cfg)
    if openbox_smov_r2.enabled:
        if online_cfg.get("enabled", False):
            raise ValueError(
                "OpenBox-SMOV R2 shadow and online refinement are "
                "mutually exclusive"
            )
        if proposal_cache is not None:
            raise ValueError(
                "OpenBox-SMOV R2 shadow requires the live CuTR stream; "
                "proposal cache record/replay is forbidden"
            )
        openbox_diagnostics_root = openbox_smov_r2.config.get(
            "diagnostics", {}
        ).get("root")
        output_root = cfg.get("data", {}).get("output_dir")
        if not openbox_diagnostics_root or output_root is None:
            raise ValueError(
                "OpenBox-SMOV R2 shadow requires separate native-output "
                "and create-only diagnostics roots"
            )
        if Path(openbox_diagnostics_root).resolve() == Path(
            output_root
        ).resolve():
            raise ValueError(
                "OpenBox-SMOV R2 diagnostics root must differ from the "
                "native prediction root"
            )
    if c3_online_observer.enabled and not online_cfg.get("enabled", False):
        raise ValueError("C3 online identity observer requires online refinement")
    if ca1m_native_b6_observer.enabled:
        if str(cfg.get("dataset", "")).lower() != "ca1m":
            raise ValueError("CA1M native B6 observer requires dataset=CA1M")
        if online_cfg.get("enabled", False):
            raise ValueError(
                "CA1M native B6 observer and online refinement are "
                "mutually exclusive"
            )
        if online_same_run_anchor_root is not None:
            raise ValueError(
                "CA1M native B6 observer cannot use the online-refinement "
                "same-run anchor"
            )
        if ca1m_native_b6_same_run_anchor_root is None:
            raise ValueError(
                "enabled CA1M native B6 observer requires "
                "--ca1m-native-b6-same-run-anchor-root"
            )
        output_root = cfg.get("data", {}).get("output_dir")
        if output_root is None:
            raise ValueError(
                "CA1M native B6 same-run identity requires an output directory"
            )
        anchor_root = Path(ca1m_native_b6_same_run_anchor_root).resolve()
        if anchor_root == Path(output_root).resolve():
            raise ValueError(
                "CA1M native B6 anchor root must differ from the output root"
            )
        ca1m_native_b6_same_run_anchor_root = str(anchor_root)
    elif ca1m_native_b6_same_run_anchor_root is not None:
        raise ValueError(
            "--ca1m-native-b6-same-run-anchor-root requires an enabled "
            "CA1M native B6 observer"
        )
    if prediction_same_run_anchor_root is not None:
        if str(cfg.get("dataset", "")).lower() != "ca1m":
            raise ValueError(
                "generic prediction same-run identity is restricted to CA1M"
            )
        if (
            online_same_run_anchor_root is not None
            or ca1m_native_b6_same_run_anchor_root is not None
        ):
            raise ValueError(
                "generic prediction identity cannot be combined with an "
                "observer-specific same-run anchor"
            )
        output_root = cfg.get("data", {}).get("output_dir")
        if output_root is None:
            raise ValueError(
                "prediction same-run identity requires an output directory"
            )
        anchor_root = Path(prediction_same_run_anchor_root).resolve()
        if anchor_root == Path(output_root).resolve():
            raise ValueError(
                "prediction identity root must differ from the output root"
            )
        prediction_same_run_anchor_root = str(anchor_root)
    use_online_appearance = bool(
        online_cfg.get("enabled", False)
        and online_cfg.get("appearance_memory", {}).get("enabled", True)
    )
    def build_online_components():
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
        controller = build_online_refinement_controller(
            cfg,
            device=str(model.pixel_mean.device),
            appearance_encoder=appearance_encoder,
            proposal_observer=(
                c3_online_observer if c3_online_observer.enabled else None
            ),
        )
        return appearance_encoder, controller

    if online_same_run_anchor_root is not None:
        with preserved_observer_rng_state():
            appearance_encoder, online_refiner = build_online_components()
    else:
        appearance_encoder, online_refiner = build_online_components()
    if online_same_run_anchor_root is not None:
        if not online_refiner.enabled:
            raise ValueError(
                "--online-same-run-anchor-root requires online refinement"
            )
        output_root = cfg.get("data", {}).get("output_dir")
        if output_root is None:
            raise ValueError(
                "--online-same-run-anchor-root requires an output directory"
            )
        anchor_root = Path(online_same_run_anchor_root).resolve()
        if anchor_root == Path(output_root).resolve():
            raise ValueError(
                "same-run anchor root must differ from the output root"
            )
        online_same_run_anchor_root = str(anchor_root)
    if prediction_same_run_anchor_root is not None:
        if online_refiner.enabled or ca1m_native_b6_observer.enabled:
            raise ValueError(
                "generic prediction identity requires the plain finalizer "
                "without online or native-B6 observers"
            )

    box_count = 0
    side_birth_keyframe_step = -1
    start_time = time.time()
    reported_stable_id_repairs = set()
    openbox_smov_r2_wrapper_starts = {}

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

    def prepare_openbox_smov_r2_keyframe(
        frame_index,
        scene_identifier,
        current_predictions,
        current_sample,
        camera_to_world,
    ):
        """Freeze R2 evidence before native association mutates proposals."""

        if not openbox_smov_r2.enabled:
            return None
        wrapper_started = time.perf_counter_ns()
        previous_groups = tuple(
            tuple(int(value) for value in group)
            for group in box_manager.fusion_list
        )
        proposal_ids = np.array(
            current_predictions.init_id.detach().cpu().numpy(),
            dtype=np.int64,
            order="C",
            copy=True,
        )
        raw_boxes_xyxy = np.array(
            current_predictions.pred_boxes.detach().cpu().numpy(),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        proposal_scores = np.array(
            current_predictions.scores.detach().cpu().numpy(),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        expected_ids = proposal_ids.copy()
        expected_boxes = raw_boxes_xyxy.copy()
        expected_scores = proposal_scores.copy()
        depth_meters = np.array(
            current_sample["wide"]["depth"][-1].detach().cpu().numpy(),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        intrinsics = np.array(
            current_sample["sensor_info"].wide.depth.K[-1]
            .detach()
            .cpu()
            .numpy(),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        camera_to_world = np.array(
            camera_to_world,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        expected_depth = depth_meters.copy()
        prepared = openbox_smov_r2.prepare_keyframe(
            scene_id=str(scene_identifier),
            frame_id=int(frame_index),
            proposal_ids=proposal_ids.copy(),
            boxes_xyxy=raw_boxes_xyxy.copy(),
            proposal_scores=proposal_scores.copy(),
            proposal_image_shape=tuple(
                int(value) for value in current_predictions.image_size
            ),
            depth_m=depth_meters.copy(),
            intrinsics=intrinsics.copy(),
            camera_to_world=camera_to_world.copy(),
            previous_fusion_groups=previous_groups,
        )
        # The observer received only independent CPU copies.  Checking those
        # source copies is sufficient and avoids four redundant GPU downloads
        # (and their implicit synchronizations) on every CuTR keyframe.
        if (
            not np.array_equal(proposal_ids, expected_ids)
            or not np.array_equal(raw_boxes_xyxy, expected_boxes)
            or not np.array_equal(proposal_scores, expected_scores)
            or not np.array_equal(depth_meters, expected_depth)
        ):
            raise RuntimeError(
                "OpenBox-SMOV R2 prepare mutated native proposal state"
            )
        openbox_smov_r2_wrapper_starts[id(prepared)] = (
            time.perf_counter_ns() - wrapper_started
        ) / 1e6
        return prepared

    def prepare_openbox_smov_r2_abstain(
        frame_index,
        scene_identifier,
        reason,
    ):
        """Advance the R2 clock without fabricating a stale observation."""

        if not openbox_smov_r2.enabled:
            return None
        wrapper_started = time.perf_counter_ns()
        previous_groups = tuple(
            tuple(int(value) for value in group)
            for group in box_manager.fusion_list
        )
        prepared = openbox_smov_r2.prepare_abstain(
            scene_id=str(scene_identifier),
            frame_id=int(frame_index),
            proposal_ids=np.empty((0,), dtype=np.int64),
            previous_fusion_groups=previous_groups,
            reason=str(reason),
        )
        openbox_smov_r2_wrapper_starts[id(prepared)] = (
            time.perf_counter_ns() - wrapper_started
        ) / 1e6
        return prepared

    def commit_openbox_smov_r2(prepared):
        """Commit R2 only after native association/fusion fixes track IDs."""

        if not openbox_smov_r2.enabled:
            return None
        wrapper_prefix_ms = openbox_smov_r2_wrapper_starts.pop(
            id(prepared), 0.0
        )
        wrapper_started = time.perf_counter_ns()
        current_groups = tuple(
            tuple(int(value) for value in group)
            for group in box_manager.fusion_list
        )
        native_event_signature = tuple(
            (
                tuple(int(value) for value in event.get("winner_members", ())),
                tuple(int(value) for value in event.get("loser_members", ())),
            )
            for event in box_manager.merge_log
        )
        # Pass fresh mappings containing only the two geometry-lineage fields
        # consumed by R2.  No mutable BoxManager object crosses this boundary.
        association_events = tuple(
            {
                "winner_members": winner,
                "loser_members": loser,
            }
            for winner, loser in native_event_signature
        )
        receipt = openbox_smov_r2.commit_keyframe(
            prepared,
            current_fusion_groups=current_groups,
            association_events=association_events,
        )
        after_groups = tuple(
            tuple(int(value) for value in group)
            for group in box_manager.fusion_list
        )
        after_event_signature = tuple(
            (
                tuple(int(value) for value in event.get("winner_members", ())),
                tuple(int(value) for value in event.get("loser_members", ())),
            )
            for event in box_manager.merge_log
        )
        if (
            after_groups != current_groups
            or after_event_signature != native_event_signature
        ):
            raise RuntimeError(
                "OpenBox-SMOV R2 commit mutated native BoxFusion state"
            )
        openbox_smov_r2.record_wrapper_timing(
            wrapper_prefix_ms
            + (time.perf_counter_ns() - wrapper_started) / 1e6
        )
        return receipt

    def query_moon_qim_lite(
        frame_index,
        scene_identifier,
        current_predictions,
    ):
        """Query history before native association mutates BoxManager state."""

        if not moon_qim_lite.enabled:
            return None
        pipeline_start = time.perf_counter_ns()
        previous_groups = tuple(
            tuple(int(value) for value in group)
            for group in box_manager.fusion_list
        )
        previous_stable_ids = moon_qim_identity.ids_for(previous_groups)
        # Bound the native association trace to this one keyframe.
        box_manager.merge_log.clear()
        proposal_ids = np.array(
            current_predictions.init_id.detach().cpu().numpy(),
            dtype=np.int64,
            order="C",
            copy=True,
        )
        proposal_corners = np.array(
            current_predictions.pred_boxes_3d.corners.detach().cpu().numpy(),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        batch = moon_qim_lite.query(
            scene_id=str(scene_identifier),
            frame_id=int(frame_index),
            proposal_ids=proposal_ids,
            proposal_corners_world=proposal_corners,
        )
        moon_qim_lite.record_pipeline_timing(
            query_ms=(time.perf_counter_ns() - pipeline_start) / 1e6
        )
        puf_batch = None
        if puf_lite.enabled:
            puf_pipeline_start = time.perf_counter_ns()
            if all_pred_box is None:
                active_corners = np.empty((0, 8, 3), dtype=np.float32)
            else:
                if len(all_pred_box) != len(previous_groups):
                    raise RuntimeError(
                        "PUF pre-association tracks are not aligned with "
                        "BoxManager fusion groups"
                    )
                active_corners = np.array(
                    all_pred_box.pred_boxes_3d.corners.detach().cpu().numpy(),
                    dtype=np.float32,
                    order="C",
                    copy=True,
                )
            puf_batch = puf_lite.query(
                qim_batch=batch,
                proposal_corners_world=proposal_corners,
                active_track_ids=np.array(
                    previous_stable_ids,
                    dtype=np.int64,
                    order="C",
                    copy=True,
                ),
                active_track_corners_world=active_corners,
            )
            puf_lite.record_pipeline_timing(
                query_ms=(time.perf_counter_ns() - puf_pipeline_start) / 1e6
            )
        arbitration_batch = None
        if puf_arbitration_lite.enabled:
            if puf_batch is None:
                raise RuntimeError(
                    "PUF arbitration-lite requires a frozen PUF query"
                )
            arbitration_start = time.perf_counter_ns()
            arbitration_batch = puf_arbitration_lite.query(
                puf_batch=puf_batch
            )
            puf_arbitration_lite.record_pipeline_timing(
                query_ms=(time.perf_counter_ns() - arbitration_start) / 1e6
            )
        return (
            batch,
            previous_groups,
            previous_stable_ids,
            puf_batch,
            arbitration_batch,
        )

    def query_mv3dis_depth_lite(
        frame_index,
        scene_identifier,
        current_predictions,
        current_sample,
        qim_batch,
    ):
        """Extract current guides, then query only committed QIM history."""

        if not mv3dis_depth_lite.enabled:
            return None
        if qim_batch is None:
            raise RuntimeError(
                "MV3DIS-Depth-lite requires the current Moon-QIM query"
            )
        pipeline_start = time.perf_counter_ns()

        depth_value = current_sample["wide"]["depth"][-1]
        depth_m = np.array(
            depth_value.detach().cpu().numpy()
            if torch.is_tensor(depth_value)
            else depth_value,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        depth_K_value = current_sample["sensor_info"].wide.depth.K[-1]
        depth_K = np.array(
            depth_K_value.detach().cpu().numpy()
            if torch.is_tensor(depth_K_value)
            else depth_K_value,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        pose_value = current_sample["sensor_info"].gt.RT[-1]
        camera_to_world = np.array(
            pose_value.detach().cpu().numpy()
            if torch.is_tensor(pose_value)
            else pose_value,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        raw_boxes = np.array(
            current_predictions.pred_boxes.detach().cpu().numpy(),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        structured_boxes = np.array(
            current_predictions.pred_boxes_3d.tensor.detach().cpu().numpy(),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        world_rotations = np.array(
            current_predictions.pred_boxes_3d.R.detach().cpu().numpy(),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if len(qim_batch.proposal_ids) != len(raw_boxes):
            raise RuntimeError(
                "MV3DIS proposal rows are not aligned with Moon-QIM"
            )

        try:
            guides = sample_depth_guide_points_batch(
                depth_m,
                depth_K,
                camera_to_world,
                raw_boxes,
                structured_boxes[:, :3],
                structured_boxes[:, 3:6],
                world_rotations,
            )
        except ValueError:
            # A malformed sensor/OBB batch is a batch-wide abstention. Native
            # BoxFusion remains the only association and fusion path.
            guides = (None,) * len(raw_boxes)
        proposal_points = tuple(
            np.empty((0, 3), dtype=np.float64)
            if guide is None
            else guide.points_world
            for guide in guides
        )

        with preserved_observer_rng_state():
            depth_batch = mv3dis_depth_lite.query(
                qim_batch=qim_batch,
                proposal_points_world=proposal_points,
                depth_m=depth_m,
                K=depth_K,
                T_wc=camera_to_world,
                proposal_boxes_xyxy=tuple(
                    tuple(float(value) for value in row)
                    for row in raw_boxes
                ),
            )
        mv3dis_depth_lite.record_pipeline_timing(
            query_ms=(time.perf_counter_ns() - pipeline_start) / 1e6
        )
        return depth_batch

    def commit_mv3dis_depth_lite(
        depth_batch,
        native_targets,
        current_groups,
        current_stable_ids,
        association_events,
    ):
        """Commit guides only after native association has fixed stable IDs."""

        if not mv3dis_depth_lite.enabled:
            return
        if depth_batch is None:
            raise RuntimeError(
                "MV3DIS-Depth-lite commit requires its frozen query batch"
            )
        pipeline_start = time.perf_counter_ns()
        committed_track_ids = derive_committed_track_ids(
            proposal_ids=np.asarray(depth_batch.proposal_ids, dtype=np.int64),
            current_fusion_groups=current_groups,
            current_stable_ids=np.asarray(
                current_stable_ids, dtype=np.int64
            ),
            association_events=association_events,
        )
        with preserved_observer_rng_state():
            mv3dis_depth_lite.commit(
                depth_batch,
                committed_track_ids=committed_track_ids,
                native_target_track_ids=native_targets,
            )
        mv3dis_depth_lite.record_pipeline_timing(
            commit_ms=(time.perf_counter_ns() - pipeline_start) / 1e6
        )

    def commit_moon_qim_lite(
        frame_index,
        scene_identifier,
        query_context=None,
        mv3dis_query_context=None,
    ):
        """Score the shadow query, then commit only the post-association state."""

        nonlocal side_birth_keyframe_step
        if not moon_qim_lite.enabled:
            return
        pipeline_start = time.perf_counter_ns()
        current_groups = tuple(
            tuple(int(value) for value in group)
            for group in box_manager.fusion_list
        )
        association_events = tuple(box_manager.merge_log)
        native_targets = None
        if query_context is not None:
            (
                batch,
                previous_groups,
                previous_stable_ids,
                puf_batch,
                arbitration_batch,
            ) = query_context
            native_targets = derive_native_target_track_ids(
                proposal_ids=np.asarray(batch.proposal_ids, dtype=np.int64),
                previous_fusion_groups=previous_groups,
                previous_stable_ids=previous_stable_ids,
                current_fusion_groups=current_groups,
                association_events=association_events,
            )
            moon_qim_lite.observe_native_targets(batch, native_targets)
            qim_prefix_ms = (
                time.perf_counter_ns() - pipeline_start
            ) / 1e6
            if puf_batch is not None:
                puf_observe_start = time.perf_counter_ns()
                puf_lite.observe_native_targets(puf_batch, native_targets)
                puf_lite.record_pipeline_timing(
                    observe_ms=(
                        time.perf_counter_ns() - puf_observe_start
                    )
                    / 1e6
                )
            if arbitration_batch is not None:
                arbitration_observe_start = time.perf_counter_ns()
                puf_arbitration_lite.observe_native_targets(
                    arbitration_batch, native_targets
                )
                puf_arbitration_lite.record_pipeline_timing(
                    observe_ms=(
                        time.perf_counter_ns() - arbitration_observe_start
                    )
                    / 1e6
                )
            pipeline_start = time.perf_counter_ns()
        else:
            qim_prefix_ms = 0.0
        stable_ids = moon_qim_identity.update(current_groups)
        committed_track_ids = None
        side_birth_pipeline_start = None
        if side_birth_probation_lite.enabled:
            side_birth_pipeline_start = time.perf_counter_ns()
            side_birth_keyframe_step += 1
            if query_context is not None:
                committed_track_ids = derive_committed_track_ids(
                    proposal_ids=np.asarray(
                        batch.proposal_ids, dtype=np.int64
                    ),
                    current_fusion_groups=current_groups,
                    current_stable_ids=np.asarray(
                        stable_ids, dtype=np.int64
                    ),
                    association_events=association_events,
                )
        if all_pred_box is None:
            global_corners = np.empty((0, 8, 3), dtype=np.float32)
        else:
            if len(current_groups) != len(all_pred_box):
                raise RuntimeError(
                    "QIM fusion groups are not aligned with global boxes"
                )
            global_corners = (
                all_pred_box.pred_boxes_3d.corners.detach().cpu().numpy()
            )
        if len(global_corners) != len(stable_ids):
            raise RuntimeError(
                "QIM causal IDs are not aligned with global boxes"
            )
        moon_qim_lite.update(
            scene_id=str(scene_identifier),
            frame_id=int(frame_index),
            track_ids=np.array(
                stable_ids, dtype=np.int64, order="C", copy=True
            ),
            track_corners_world=np.array(
                global_corners, dtype=np.float32, order="C", copy=True
            ),
        )
        moon_qim_lite.record_pipeline_timing(
            update_ms=(
                qim_prefix_ms
                + (time.perf_counter_ns() - pipeline_start) / 1e6
            )
        )
        if third_view_birth_lite.enabled:
            third_view_start = time.perf_counter_ns()
            if per_frame_ins is None:
                if current_groups:
                    raise RuntimeError(
                        "third-view birth-lite has groups without source rows"
                    )
                source_frame_ids = {}
            else:
                required_source_ids = sorted(
                    {
                        int(source_id)
                        for group in current_groups
                        for source_id in group
                    }
                )
                if required_source_ids:
                    source_index = torch.as_tensor(
                        required_source_ids,
                        dtype=torch.long,
                        device=per_frame_ins.init_id.device,
                    )
                    selected_init_ids = np.asarray(
                        per_frame_ins.init_id[source_index]
                        .detach()
                        .cpu()
                        .numpy(),
                        dtype=np.int64,
                    )
                    selected_frame_ids = np.asarray(
                        per_frame_ins.frame_id[source_index]
                        .detach()
                        .cpu()
                        .numpy(),
                        dtype=np.int64,
                    )
                else:
                    selected_init_ids = np.empty((0,), dtype=np.int64)
                    selected_frame_ids = np.empty((0,), dtype=np.int64)
                expected_init_ids = np.asarray(
                    required_source_ids, dtype=np.int64
                )
                if not np.array_equal(selected_init_ids, expected_init_ids):
                    raise RuntimeError(
                        "third-view source init/frame lookup is not causally "
                        "aligned"
                    )
                source_frame_ids = dict(
                    zip(required_source_ids, selected_frame_ids.tolist())
                )
            with preserved_observer_rng_state():
                third_view_birth_lite.observe(
                    scene_id=str(scene_identifier),
                    frame_id=int(frame_index),
                    current_fusion_groups=current_groups,
                    current_stable_ids=np.asarray(
                        stable_ids, dtype=np.int64
                    ),
                    source_frame_ids=source_frame_ids,
                )
                third_view_result = third_view_birth_lite.finalize(
                    final_stable_ids=np.asarray(stable_ids, dtype=np.int64)
                )
            if not all(third_view_result.keep_mask):
                raise RuntimeError(
                    "third-view birth-lite attempted to filter native boxes"
                )
            third_view_birth_lite.record_pipeline_timing(
                (time.perf_counter_ns() - third_view_start) / 1e6
            )
        if side_birth_probation_lite.enabled:
            seed_events = []
            observed_stable_ids = []
            if query_context is not None:
                if (
                    arbitration_batch is None
                    or native_targets is None
                    or committed_track_ids is None
                ):
                    raise RuntimeError(
                        "side-birth probation lost its frozen association rows"
                    )
                if not (
                    len(arbitration_batch.rows)
                    == len(native_targets)
                    == len(committed_track_ids)
                ):
                    raise RuntimeError(
                        "side-birth probation rows are not aligned"
                    )
                observed_stable_ids = [
                    int(value)
                    for value in committed_track_ids
                    if value is not None
                ]
                for row, raw_targets, committed_id in zip(
                    arbitration_batch.rows,
                    native_targets,
                    committed_track_ids,
                ):
                    if row.action != "birth":
                        continue
                    if row.top_probability is None or row.margin is None:
                        raise RuntimeError(
                            "selected birth lacks a frozen probability/margin"
                        )
                    if raw_targets is None:
                        native_kind = "unresolved"
                        target_ids = ()
                    else:
                        target_ids = tuple(
                            sorted({int(value) for value in raw_targets})
                        )
                        if not target_ids:
                            native_kind = "birth"
                        elif len(target_ids) == 1:
                            native_kind = "unique_history"
                        else:
                            native_kind = "ambiguous_history"
                    seed_events.append(
                        SideBirthSeedEvent(
                            proposal_id=int(row.proposal_id),
                            committed_stable_id=(
                                None
                                if committed_id is None
                                else int(committed_id)
                            ),
                            top_probability=float(row.top_probability),
                            margin=float(row.margin),
                            native_target_kind=native_kind,
                            native_target_ids=target_ids,
                        )
                    )
            with preserved_observer_rng_state():
                side_birth_probation_lite.observe_true_cutr_keyframe(
                    scene_id=str(scene_identifier),
                    frame_id=int(frame_index),
                    keyframe_step=int(side_birth_keyframe_step),
                    birth_events=tuple(seed_events),
                    observed_stable_ids=tuple(observed_stable_ids),
                )
            side_birth_probation_lite.record_pipeline_timing(
                (time.perf_counter_ns() - side_birth_pipeline_start) / 1e6
            )
        if mv3dis_query_context is not None:
            if native_targets is None:
                raise RuntimeError(
                    "MV3DIS-Depth-lite commit requires native targets"
                )
            commit_mv3dis_depth_lite(
                mv3dis_query_context,
                native_targets,
                current_groups,
                stable_ids,
                association_events,
            )
        elif mv3dis_depth_lite.enabled and query_context is not None:
            raise RuntimeError(
                "MV3DIS-Depth-lite query context was lost before commit"
            )
        box_manager.merge_log.clear()

    def observe_online_keyframe(
        frame_index,
        scene_identifier,
        source_frame_identifier,
        pose_matrix,
    ):
        if not online_refiner.enabled:
            return
        global_corners, global_scores, stable_ids = online_snapshot()
        global_corners = np.array(
            global_corners, dtype=np.float32, order="C", copy=True
        )
        global_scores = np.array(
            global_scores, dtype=np.float32, order="C", copy=True
        )
        stable_ids = np.array(stable_ids, dtype=np.int64, order="C", copy=True)
        expected_corners = global_corners.copy()
        expected_scores = global_scores.copy()
        expected_stable_ids = stable_ids.copy()
        depth = sample["wide"]["depth"][-1].detach().cpu().numpy()
        depth_intrinsics = (
            sample["sensor_info"].wide.depth.K[-1]
            .detach()
            .cpu()
            .numpy()
        )
        if online_same_run_anchor_root is not None:
            # The identity route must not consume random numbers that the
            # remaining BoxFusion stream would otherwise use.  This scope is
            # intentionally opt-in so active online-refinement routes retain
            # their historical runtime semantics.
            with preserved_observer_rng_state():
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
            after_corners, after_scores, after_stable_ids = online_snapshot()
            if (
                not np.array_equal(global_corners, expected_corners)
                or not np.array_equal(global_scores, expected_scores)
                or not np.array_equal(stable_ids, expected_stable_ids)
                or not np.array_equal(after_corners, expected_corners)
                or not np.array_equal(after_scores, expected_scores)
                or not np.array_equal(after_stable_ids, expected_stable_ids)
            ):
                raise RuntimeError(
                    "online observer directly mutated BoxFusion state"
                )
            return

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

    def observe_ca1m_native_b6_keyframe(
        frame_index,
        scene_identifier,
        source_frame_identifier,
        pose_matrix,
    ):
        """Cache one causal RGB-D keyframe without changing BoxFusion state."""

        if not ca1m_native_b6_observer.enabled:
            return
        before_corners, before_scores, before_stable_ids = online_snapshot()
        before_corners = np.array(
            before_corners, dtype=np.float32, order="C", copy=True
        )
        before_scores = np.array(
            before_scores, dtype=np.float32, order="C", copy=True
        )
        before_stable_ids = np.array(
            before_stable_ids, dtype=np.int64, order="C", copy=True
        )
        depth = np.array(
            sample["wide"]["depth"][-1].detach().cpu().numpy(),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        depth_intrinsics = np.array(
            sample["sensor_info"].wide.depth.K[-1]
            .detach()
            .cpu()
            .numpy(),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        with preserved_observer_rng_state():
            ca1m_native_b6_observer.record_keyframe(
                scene_id=str(scene_identifier),
                frame_id=int(frame_index),
                source_frame_id=str(source_frame_identifier),
                depth_meters=depth,
                intrinsics=depth_intrinsics,
                camera_to_world=np.array(
                    pose_matrix,
                    dtype=np.float64,
                    order="C",
                    copy=True,
                ),
            )
        after_corners, after_scores, after_stable_ids = online_snapshot()
        if (
            not np.array_equal(after_corners, before_corners)
            or not np.array_equal(after_scores, before_scores)
            or not np.array_equal(after_stable_ids, before_stable_ids)
        ):
            raise RuntimeError(
                "CA1M native B6 keyframe observer mutated BoxFusion state"
            )
    
    
    
    for sample in dataset:
        # Per-frame side state must never fall through to the legacy terminal
        # branch, which intentionally reuses the last native predictions.
        cutr_residual_predictions = None
        cutr_residual_raw_row_indices = np.empty((0,), dtype=np.int64)
        sample_video_id = sample["meta"]["video_id"] #(['sensor_info', 'wide', 'gt', 'meta'])
        scene_id = (
            str(sample_video_id[0])
            if isinstance(sample_video_id, (list, tuple, np.ndarray))
            else str(sample_video_id)
        )
        if proposal_cache is not None:
            proposal_cache.bind_scene(
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
            else:
                source_attempt_id = "primary"
                with torch.no_grad():
                    raw_pred_instances = model(packaged)[0]

                (
                    cutr_residual_predictions,
                    cutr_residual_raw_row_indices,
                ) = make_cutr_residual_side(raw_pred_instances, image)
                pred_instances = raw_pred_instances

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
                    "primary",
                )

                if cfg["detection"]["uv_bound"]:
                    uv_mask = box_manager.check_uv_bounds(pred_instances.pred_proj_xy,image.shape[1],image.shape[0],ratio=cfg["detection"]["uv_bound_value"]) #[N]
                    pred_instances = pred_instances[uv_mask]
                if cfg["detection"]["floor_mask"]:
                    floor_mask = box_manager.check_floor_mask(pred_instances.pred_boxes_3d.tensor, ratio=cfg["detection"]["floor_ratio"])
                    pred_instances = pred_instances[~floor_mask]
                pred_instances = apply_lifting_if_configured(
                    pred_instances,
                    sample,
                    image,
                    scene_id,
                    count,
                    "post_filter",
                    "primary",
                )

               # avoid first frame empty predictions
                if len(pred_instances) == 0 and count ==0:
                    source_attempt_id = "retry"
                    # The retry is the sole frame-0 CuTR observation.  Drop
                    # every row copied from the primary attempt first.
                    cutr_residual_predictions = None
                    cutr_residual_raw_row_indices = np.empty(
                        (0,), dtype=np.int64
                    )
                    with torch.no_grad():
                        raw_pred_instances = model(packaged)[0]
                    (
                        cutr_residual_predictions,
                        cutr_residual_raw_row_indices,
                    ) = make_cutr_residual_side(
                        raw_pred_instances,
                        image,
                        actual_native_threshold=float(
                            cfg['detection']['score_thresh']/4
                        ),
                        # The released retry path mirrors only UV filtering;
                        # the observer must follow that exact native contract
                        # without changing native retry behavior.
                        apply_floor_filter=False,
                    )
                    pred_instances = raw_pred_instances
                    pred_instances = pred_instances[
                        pred_instances.scores
                        >= float(cfg['detection']['score_thresh']/4)
                    ]
                    pred_instances = apply_lifting_if_configured(
                        pred_instances,
                        sample,
                        image,
                        scene_id,
                        count,
                        "pre_filter",
                        "retry",
                    )
                    print("again",count,"pred_instances",len(pred_instances))
                    if cfg["detection"]["uv_bound"]:
                        uv_mask = box_manager.check_uv_bounds(pred_instances.pred_proj_xy,image.shape[1],image.shape[0],ratio=cfg["detection"]["uv_bound_value"]) #[N]
                        pred_instances = pred_instances[uv_mask]
                    pred_instances = apply_lifting_if_configured(
                        pred_instances,
                        sample,
                        image,
                        scene_id,
                        count,
                        "post_filter",
                        "retry",
                    )
                    print("again",count,"pred_instances",len(pred_instances))

                if proposal_cache is not None and proposal_cache.is_record:
                    pred_instances = proposal_cache.record(
                        scene_id,
                        count,
                        pred_instances,
                        attempt_id=source_attempt_id,
                        inputs=proposal_cache_inputs(sample, image),
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

        # only process keyframes
        if count % gap ==0 or count == len(dataset)-1:
            # ``pred_instances`` is produced only on true CuTR keyframes.  The
            # legacy terminal-frame branch may reuse the previous value, so
            # every causal side index must guard on this explicit flag.
            has_current_cutr_proposals = count % gap == 0
            moon_qim_query_context = None
            mv3dis_query_context = None
            openbox_smov_r2_batch = None

            if has_current_cutr_proposals:
                # One native trace per true CuTR keyframe.  Every observer must
                # consume it before the post-association clear below.
                box_manager.merge_log.clear()
                observe_cutr_residual_keyframe(
                    count,
                    cutr_residual_predictions,
                    cutr_residual_raw_row_indices,
                    frame_pose_np,
                    pred_instances,
                    sample,
                )

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
                if has_current_cutr_proposals:
                    openbox_smov_r2_batch = (
                        prepare_openbox_smov_r2_abstain(
                            count,
                            scene_identifier,
                            "empty_current_cutr_proposals",
                        )
                    )
                    commit_openbox_smov_r2(openbox_smov_r2_batch)
                    commit_moon_qim_lite(
                        count,
                        scene_identifier,
                    )
                    box_manager.merge_log.clear()
                observe_online_keyframe(
                    count,
                    scene_identifier,
                    sample["meta"]["timestamp"],
                    frame_pose_np,
                )
                observe_ca1m_native_b6_keyframe(
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
            if has_current_cutr_proposals:
                openbox_smov_r2_batch = prepare_openbox_smov_r2_keyframe(
                    count,
                    scene_id,
                    pred_instances,
                    sample,
                    frame_pose_np,
                )
                moon_qim_query_context = query_moon_qim_lite(
                    count,
                    scene_id,
                    pred_instances,
                )
                if mv3dis_depth_lite.enabled:
                    mv3dis_query_context = query_mv3dis_depth_lite(
                        count,
                        scene_id,
                        pred_instances,
                        sample,
                        moon_qim_query_context[0],
                    )
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
            if has_current_cutr_proposals:
                commit_openbox_smov_r2(openbox_smov_r2_batch)
                commit_moon_qim_lite(
                    count,
                    scene_identifier,
                    moon_qim_query_context,
                    mv3dis_query_context,
                )
                box_manager.merge_log.clear()
            observe_online_keyframe(
                count,
                scene_identifier,
                sample["meta"]["timestamp"],
                frame_pose_np,
            )
            observe_ca1m_native_b6_keyframe(
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
            if moon_qim_lite.enabled:
                print(moon_qim_lite.summary_line())
                print(
                    "Moon-QIM-lite observer JSON | "
                    + json.dumps(
                        moon_qim_lite.summary(),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            if puf_lite.enabled:
                print(puf_lite.summary_line())
                print(
                    "PUF-lite shadow JSON | "
                    + json.dumps(
                        puf_lite.summary(),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            if puf_arbitration_lite.enabled:
                print(puf_arbitration_lite.summary_line())
                print(
                    "PUF-arbitration-lite shadow JSON | "
                    + json.dumps(
                        puf_arbitration_lite.summary(),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            if mv3dis_depth_lite.enabled:
                print(mv3dis_depth_lite.summary_line())
                print(
                    "MV3DIS-Depth-lite S0 shadow JSON | "
                    + json.dumps(
                        mv3dis_depth_lite.summary(),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            if third_view_birth_lite.enabled:
                print(third_view_birth_lite.summary_line())
                print(
                    "Third-view-birth-lite shadow JSON | "
                    + json.dumps(
                        third_view_birth_lite.summary(),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            if side_birth_probation_lite.enabled:
                terminal_groups = tuple(
                    tuple(int(value) for value in group)
                    for group in box_manager.fusion_list
                )
                terminal_stable_ids = moon_qim_identity.ids_for(
                    terminal_groups
                )
                side_birth_probation_lite.close_scene(
                    scene_id=str(scene_id),
                    terminal_frame_id=max(int(count) - 1, 0),
                    active_stable_ids=terminal_stable_ids,
                )
                print(
                    "Side-birth-probation-lite shadow JSON | "
                    + json.dumps(
                        side_birth_probation_lite.summary(),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            if lifting_adapter is not None:
                print(lifting_adapter.summary())

            output_path = None
            if cfg['data']['output_dir'] is not None:
                output_path = os.path.join(
                    cfg['data']['output_dir'],
                    video_id[0] + "_boxes.pkl",
                )
            if proposal_cache is not None and proposal_cache.is_replay:
                baseline_root = cfg["lifting"]["proposal_cache"].get(
                    "baseline_prediction_root", ""
                )
                if not baseline_root:
                    raise ValueError(
                        "Replay requires proposal_cache.baseline_prediction_root"
                    )
                proposal_cache.verify_replay_complete(
                    scene_id,
                    baseline_prediction_path=os.path.join(
                        baseline_root,
                        video_id[0] + "_boxes.pkl",
                    ),
                )

            # save global boxes for evaluation
            timing_printed = False
            if cfg['data']['output_dir'] is not None and cfg["eval"]:
                scene_identifier = (
                    video_id[0]
                    if isinstance(video_id, (tuple, list, np.ndarray))
                    else video_id
                )
                if online_refiner.enabled:
                    base_corners, base_scores, stable_ids = (
                        online_snapshot()
                    )
                    base_corners = np.array(
                        base_corners, dtype=np.float32, order="C", copy=True
                    )
                    base_scores = np.array(
                        base_scores, dtype=np.float32, order="C", copy=True
                    )
                    stable_ids = np.array(
                        stable_ids, dtype=np.int64, order="C", copy=True
                    )
                    expected_base_corners = base_corners.copy()
                    expected_base_scores = base_scores.copy()
                    expected_stable_ids = stable_ids.copy()
                    anchor_path = None
                    if online_same_run_anchor_root is not None:
                        scene_identifier = (
                            video_id[0]
                            if isinstance(video_id, (tuple, list, np.ndarray))
                            else video_id
                        )
                        anchor_path = (
                            Path(online_same_run_anchor_root).resolve()
                            / f"{scene_identifier}_boxes.pkl"
                        )
                    scene_identifier = (
                        video_id[0]
                        if isinstance(video_id, (tuple, list, np.ndarray))
                        else video_id
                    )
                    refinement_result = online_refiner.finalize(
                        global_corners=base_corners.copy(),
                        global_scores=base_scores.copy(),
                        stable_ids=stable_ids.copy(),
                        scene_id=str(scene_identifier),
                    )
                    if online_same_run_anchor_root is not None:
                        expected_indices = np.arange(
                            len(base_corners), dtype=np.int64
                        )
                        if (
                            not np.array_equal(
                                base_corners, expected_base_corners
                            )
                            or not np.array_equal(
                                base_scores, expected_base_scores
                            )
                            or not np.array_equal(
                                stable_ids, expected_stable_ids
                            )
                            or
                            not np.array_equal(
                                refinement_result.corners,
                                expected_base_corners,
                            )
                            or not np.array_equal(
                                refinement_result.scores,
                                expected_base_scores,
                            )
                            or not np.array_equal(
                                refinement_result.source_indices,
                                expected_indices,
                            )
                            or not np.array_equal(
                                refinement_result.stable_ids,
                                expected_stable_ids,
                            )
                        ):
                            raise RuntimeError(
                                "online observer finalizer violated "
                                "same-run identity"
                            )
                        save_prediction_create_only(
                            expected_base_corners,
                            expected_base_scores,
                            anchor_path,
                        )
                        print(
                            "Online same-run anchor saved atomically to",
                            anchor_path,
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
                    openbox_smov_r2_stable_ids = None
                    if openbox_smov_r2.enabled:
                        terminal_groups = tuple(
                            tuple(int(value) for value in group)
                            for group in box_manager.fusion_list
                        )
                        openbox_smov_r2_stable_ids = (
                            openbox_smov_r2.current_stable_ids(
                                terminal_groups
                            )
                        )
                        openbox_smov_r2_stable_ids = np.array(
                            openbox_smov_r2_stable_ids,
                            dtype=np.int64,
                            order="C",
                            copy=True,
                        )
                        if len(openbox_smov_r2_stable_ids) != len(boxes_3d):
                            raise RuntimeError(
                                "OpenBox-SMOV R2 stable IDs are not aligned "
                                "with native terminal rows"
                            )
                    if ca1m_native_b6_observer.enabled:
                        base_corners, base_scores, stable_ids = online_snapshot()
                        base_corners = np.array(
                            base_corners,
                            dtype=np.float32,
                            order="C",
                            copy=True,
                        )
                        base_scores = np.array(
                            base_scores,
                            dtype=np.float32,
                            order="C",
                            copy=True,
                        )
                        stable_ids = np.array(
                            stable_ids,
                            dtype=np.int64,
                            order="C",
                            copy=True,
                        )
                        expected_corners = base_corners.copy()
                        expected_scores = base_scores.copy()
                        expected_stable_ids = stable_ids.copy()
                        scene_identifier = (
                            video_id[0]
                            if isinstance(
                                video_id, (tuple, list, np.ndarray)
                            )
                            else video_id
                        )
                        with preserved_observer_rng_state():
                            native_summary = ca1m_native_b6_observer.finalize(
                                scene_id=str(scene_identifier),
                                corners=base_corners.copy(),
                                scores=base_scores.copy(),
                                stable_ids=stable_ids.copy(),
                            )
                        after_corners, after_scores, after_stable_ids = (
                            online_snapshot()
                        )
                        if (
                            not np.array_equal(
                                base_corners, expected_corners
                            )
                            or not np.array_equal(
                                base_scores, expected_scores
                            )
                            or not np.array_equal(
                                stable_ids, expected_stable_ids
                            )
                            or not np.array_equal(
                                after_corners, expected_corners
                            )
                            or not np.array_equal(
                                after_scores, expected_scores
                            )
                            or not np.array_equal(
                                after_stable_ids, expected_stable_ids
                            )
                        ):
                            raise RuntimeError(
                                "CA1M native B6 final observer violated "
                                "same-run identity"
                            )
                        anchor_path = (
                            Path(ca1m_native_b6_same_run_anchor_root)
                            / f"{scene_identifier}_boxes.pkl"
                        )
                        save_prediction_create_only(
                            expected_corners,
                            expected_scores,
                            anchor_path,
                        )
                        boxes_3d = expected_corners
                        scores = expected_scores
                        print(
                            ca1m_native_b6_observer.summary_text(
                                native_summary
                            )
                        )
                if cfg['dataset'] == 'scannet':
                    minimum_extent = float(
                        cfg.get("data", {}).get(
                            "post_process_min_extent", 0.30
                        )
                    )
                    boxes_3d, valid_mask = post_process(
                        boxes_3d,
                        threshold=minimum_extent,
                        return_mask=True,
                    )
                    scores = scores[valid_mask]
                    if openbox_smov_r2.enabled:
                        openbox_smov_r2_stable_ids = (
                            openbox_smov_r2_stable_ids[valid_mask]
                        )

                if openbox_smov_r2.enabled:
                    expected_native_corners = np.array(
                        boxes_3d,
                        dtype=np.float32,
                        order="C",
                        copy=True,
                    )
                    expected_native_scores = np.array(
                        scores,
                        dtype=np.float32,
                        order="C",
                        copy=True,
                    )
                    expected_stable_ids = np.array(
                        openbox_smov_r2_stable_ids,
                        dtype=np.int64,
                        order="C",
                        copy=True,
                    )
                    before_global_corners, before_global_scores, _ = (
                        online_snapshot()
                    )
                    openbox_smov_r2_result = (
                        openbox_smov_r2.finalize_shadow(
                            scene_id=str(scene_identifier),
                            native_corners=expected_native_corners.copy(),
                            native_scores=expected_native_scores.copy(),
                            stable_ids=expected_stable_ids.copy(),
                        )
                    )
                    after_global_corners, after_global_scores, _ = (
                        online_snapshot()
                    )
                    after_openbox_stable_ids = np.array(
                        openbox_smov_r2.current_stable_ids(
                            terminal_groups
                        ),
                        dtype=np.int64,
                        order="C",
                        copy=True,
                    )
                    if cfg['dataset'] == 'scannet':
                        after_openbox_stable_ids = (
                            after_openbox_stable_ids[valid_mask]
                        )
                    if (
                        not np.array_equal(
                            boxes_3d, expected_native_corners
                        )
                        or not np.array_equal(
                            scores, expected_native_scores
                        )
                        or not np.array_equal(
                            openbox_smov_r2_stable_ids,
                            expected_stable_ids,
                        )
                        or not np.array_equal(
                            after_openbox_stable_ids,
                            expected_stable_ids,
                        )
                        or not np.array_equal(
                            before_global_corners, after_global_corners
                        )
                        or not np.array_equal(
                            before_global_scores, after_global_scores
                        )
                        or not np.array_equal(
                            openbox_smov_r2_result.native_corners,
                            expected_native_corners,
                        )
                        or not np.array_equal(
                            openbox_smov_r2_result.native_scores,
                            expected_native_scores,
                        )
                        or not np.array_equal(
                            openbox_smov_r2_result.stable_ids,
                            expected_stable_ids,
                        )
                    ):
                        raise RuntimeError(
                            "OpenBox-SMOV R2 finalizer violated native "
                            "geometry/score/order identity"
                        )
                    diagnostics_root = (
                        cfg["openbox_smov_r2"]
                        .get("diagnostics", {})
                        .get("root")
                    )
                    if diagnostics_root:
                        openbox_smov_r2_path = (
                            Path(diagnostics_root)
                            / (
                                f"{scene_identifier}_"
                                "openbox_smov_r2_shadow.npz"
                            )
                        )
                        save_r2_shadow_sidecar_create_only(
                            openbox_smov_r2_result,
                            openbox_smov_r2_path,
                        )
                    openbox_smov_r2_summary = dict(
                        openbox_smov_r2.summary()
                    )
                    openbox_smov_r2_summary["terminal"] = (
                        openbox_smov_r2_result.summary.as_dict()
                    )
                    print(
                        "OpenBox-SMOV R2 shadow JSON | "
                        + json.dumps(
                            openbox_smov_r2_summary,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )

                if cutr_residual_birth_lite.enabled:
                    expected_native_corners = np.array(
                        boxes_3d, order="C", copy=True
                    )
                    expected_native_scores = np.array(
                        scores, order="C", copy=True
                    )
                    with preserved_observer_rng_state():
                        residual_close = cutr_residual_birth_lite.close(
                            native_corners=expected_native_corners.copy(),
                            native_scores=expected_native_scores.copy(),
                        )
                    if (
                        not np.array_equal(
                            boxes_3d, expected_native_corners
                        )
                        or not np.array_equal(
                            scores, expected_native_scores
                        )
                    ):
                        raise RuntimeError(
                            "CuTR residual final observer mutated native output"
                        )
                    residual_summary = dict(
                        cutr_residual_birth_lite.summary()
                    )
                    residual_summary.update(
                        {
                            "counterfactual_candidate_count": len(
                                residual_close.candidates
                            ),
                            "counterfactual_candidate_track_ids": [
                                int(candidate.track_id)
                                for candidate in residual_close.candidates
                            ],
                            "native_export_appended": False,
                        }
                    )
                    print(
                        "CuTR-residual-birth-lite shadow JSON | "
                        + json.dumps(
                            residual_summary,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    if cutr_residual_cross_view_r1.enabled:
                        with preserved_observer_rng_state():
                            residual_r1_close = (
                                cutr_residual_cross_view_r1.close(
                                    residual_close
                                )
                            )
                        if (
                            not np.array_equal(
                                boxes_3d, expected_native_corners
                            )
                            or not np.array_equal(
                                scores, expected_native_scores
                            )
                        ):
                            raise RuntimeError(
                                "CuTR residual R1 final observer mutated "
                                "native output"
                            )
                        residual_r1_summary = dict(
                            cutr_residual_cross_view_r1.summary()
                        )
                        residual_r1_summary.update(
                            {
                                "base_counterfactual_candidate_count": len(
                                    residual_close.candidates
                                ),
                                "counterfactual_candidate_count": len(
                                    residual_r1_close.candidates
                                ),
                                "counterfactual_candidate_track_ids": [
                                    int(candidate.track_id)
                                    for candidate in residual_r1_close.candidates
                                ],
                                "native_export_appended": False,
                            }
                        )
                        print(
                            "CuTR-residual-cross-view-R1 shadow JSON | "
                            + json.dumps(
                                residual_r1_summary,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        )

                if c3_online_observer.enabled:
                    c3_summary = c3_online_observer.finalize(
                        scene_id=str(scene_identifier),
                        prediction_corners=boxes_3d,
                        prediction_scores=scores,
                    )
                    print(c3_online_observer.summary_text(c3_summary))

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
                else:
                    print("Saving score-preserving predictions: count=0")

                if (
                    online_same_run_anchor_root is not None
                    or ca1m_native_b6_same_run_anchor_root is not None
                    or prediction_same_run_anchor_root is not None
                    or (
                        proposal_cache is not None
                        and proposal_cache.is_record
                    )
                ):
                    save_prediction_create_only(boxes_3d, scores, output_path)
                    if prediction_same_run_anchor_root is not None:
                        identity_path = (
                            Path(prediction_same_run_anchor_root)
                            / f"{scene_identifier}_boxes.pkl"
                        )
                        link_prediction_create_only(output_path, identity_path)
                        print(
                            "Prediction same-run byte-identity anchor saved to",
                            identity_path,
                        )
                elif boxes_3d.shape[0] > 0:
                    save_list = [[
                        (int(0), boxes_3d[n], float(scores[n]))
                        for n in range(boxes_3d.shape[0])
                    ]]
                    save_box(save_list, output_path)

            if proposal_cache is not None and proposal_cache.is_record:
                cache_manifest = proposal_cache.finalize(
                    scene_id,
                    prediction_path=output_path,
                )
                print("Finalized immutable CuTR proposal cache:", cache_manifest)

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
    parser.add_argument("--openbox-smov-r2-diagnostics-root", default=None, type=str, help="Override the create-only OpenBox-SMOV R2 shadow diagnostics root")
    parser.add_argument("--online-same-run-anchor-root", default=None, type=str, help="Create-only pre-online-refinement prediction root")
    parser.add_argument("--ca1m-native-b6-diagnostics-root", default=None, type=str, help="Create-only CA1M native B6 observer diagnostic root")
    parser.add_argument("--ca1m-native-b6-same-run-anchor-root", default=None, type=str, help="Create-only pre-observer CA1M prediction root")
    parser.add_argument("--prediction-same-run-anchor-root", default=None, type=str, help="Create-only hard-link identity root for one plain CA1M finalizer output")
    parser.add_argument("--boxer-diagnostics-root", default=None, type=str, help="Override selective-Boxer diagnostics root")
    parser.add_argument("--boxer-selective-max-center-shift-m", default=None, type=float, help="Override the selective-Boxer maximum center shift in metres")
    parser.add_argument("--boxer-selective-min-volume-ratio", default=None, type=float, help="Override the selective-Boxer minimum Boxer/CuTR volume ratio")
    parser.add_argument("--boxer-selective-max-volume-ratio", default=None, type=float, help="Override the selective-Boxer maximum Boxer/CuTR volume ratio")
    parser.add_argument("--tr3d-c3-online-parent-cache-root", default=None, type=str, help="Immutable terminal TR3D parent-cache root for C3 collection")
    parser.add_argument("--tr3d-c3-online-diagnostics-root", default=None, type=str, help="Create-only C3 online identity diagnostic root")
    parser.add_argument("--tr3d-c3-online-candidate-source", choices=["parent_score"], default="parent_score", help="Direct train C3 candidate source")
    parser.add_argument("--online-proposal-checkpoint", default=None, type=str, help="Override the supplemental proposal checkpoint")
    parser.add_argument("--online-refiner-checkpoint", default=None, type=str, help="Enable and load a trained BoxRefiner checkpoint")
    parser.add_argument("--online-quality-checkpoint", default=None, type=str, help="Load a learned quality-calibration checkpoint")
    parser.add_argument("--online-quality-mode", choices=["linear", "mlp", "iou_mlp"], default=None, help="Learned quality checkpoint type")
    parser.add_argument("--online-quality-detector-blend", default=None, type=float, help="Weight of the original detector score in final quality ranking")
    parser.add_argument("--scannet-min-extent", default=None, type=float, help="Override both ScanNet export and online supplemental minimum box extent")
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
        if args.openbox_smov_r2_diagnostics_root is not None:
            r2_cfg = cfg.setdefault("openbox_smov_r2", {})
            if not r2_cfg.get("enabled", False):
                parser.error(
                    "--openbox-smov-r2-diagnostics-root requires an "
                    "enabled OpenBox-SMOV R2 observer"
                )
            r2_cfg.setdefault("diagnostics", {})["root"] = (
                args.openbox_smov_r2_diagnostics_root
            )
        if args.ca1m_native_b6_diagnostics_root is not None:
            native_cfg = cfg.setdefault("ca1m_native_b6_observer", {})
            if not native_cfg.get("enabled", False):
                parser.error(
                    "--ca1m-native-b6-diagnostics-root requires an enabled "
                    "CA1M native B6 observer"
                )
            native_cfg.setdefault("diagnostics", {})["root"] = (
                args.ca1m_native_b6_diagnostics_root
            )
        if args.boxer_diagnostics_root is not None:
            cfg.setdefault("lifting", {}).setdefault("boxer", {})[
                "diagnostics_dir"
            ] = args.boxer_diagnostics_root
        c3_values = (
            args.tr3d_c3_online_parent_cache_root,
            args.tr3d_c3_online_diagnostics_root,
        )
        if any(value is not None for value in c3_values):
            if not all(value is not None for value in c3_values):
                parser.error("both C3 parent-cache and diagnostics roots are required")
            cfg["tr3d_c3_online_observer"] = {
                "enabled": True,
                "c2_cache_root": "",
                "parent_cache_root": args.tr3d_c3_online_parent_cache_root,
                "diagnostics_root": args.tr3d_c3_online_diagnostics_root,
                "prefix_id": "p100",
                "source_rank_max": 5,
                "gate_name": "mask2_depth",
                "candidate_source": args.tr3d_c3_online_candidate_source,
            }
        boxer_gate_values = (
            args.boxer_selective_max_center_shift_m,
            args.boxer_selective_min_volume_ratio,
            args.boxer_selective_max_volume_ratio,
        )
        if any(value is not None for value in boxer_gate_values):
            if not all(value is not None for value in boxer_gate_values):
                parser.error(
                    "all three selective-Boxer gate overrides are required "
                    "together"
                )
            center_shift, min_volume, max_volume = boxer_gate_values
            if not np.isfinite(center_shift) or center_shift < 0.0:
                parser.error(
                    "--boxer-selective-max-center-shift-m must be finite "
                    "and non-negative"
                )
            if not np.isfinite(min_volume) or min_volume <= 0.0:
                parser.error(
                    "--boxer-selective-min-volume-ratio must be finite "
                    "and positive"
                )
            if not np.isfinite(max_volume) or max_volume < min_volume:
                parser.error(
                    "--boxer-selective-max-volume-ratio must be finite "
                    "and no smaller than the minimum volume ratio"
                )
            lifting_cfg = cfg.setdefault("lifting", {})
            boxer_cfg = lifting_cfg.setdefault("boxer", {})
            if lifting_cfg.get("backend") != "boxer":
                parser.error(
                    "selective-Boxer gate overrides require "
                    "lifting.backend=boxer"
                )
            gate_cfg = boxer_cfg.setdefault("selective_gate", {})
            if not gate_cfg.get("enabled", False):
                parser.error(
                    "selective-Boxer gate overrides require an enabled "
                    "selective gate"
                )
            gate_cfg.update(
                {
                    "max_center_shift_m": float(center_shift),
                    "min_volume_ratio": float(min_volume),
                    "max_volume_ratio": float(max_volume),
                }
            )
            print(
                "Selective Boxer gate override:",
                f"center<={center_shift:.6g}m,",
                f"volume=[{min_volume:.6g},{max_volume:.6g}]",
            )
        if args.online_proposal_checkpoint is not None:
            online_cfg.setdefault("supplemental_proposals", {})[
                "checkpoint"
            ] = args.online_proposal_checkpoint
        if args.online_refiner_checkpoint is not None:
            refiner_cfg = online_cfg.setdefault("box_refiner", {})
            refiner_cfg["enabled"] = True
            refiner_cfg["checkpoint"] = args.online_refiner_checkpoint
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

    run(
        cfg,
        model,
        dataset,
        clip_model,
        preprocess,
        text_class,
        text_features,
        augmentor,
        preprocessor,
        score_thresh=cfg['detection']['score_thresh'],
        viz_on_gt_points=args.viz_on_gt_points,
        gap=cfg["data"]["gap"],
        re_vis=cfg['vis']['rerun'],
        online_same_run_anchor_root=args.online_same_run_anchor_root,
        ca1m_native_b6_same_run_anchor_root=(
            args.ca1m_native_b6_same_run_anchor_root
        ),
        prediction_same_run_anchor_root=(
            args.prediction_same_run_anchor_root
        ),
    )

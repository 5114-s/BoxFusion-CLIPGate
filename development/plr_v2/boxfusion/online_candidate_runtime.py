"""Live WeDetect/Boxer provider and BoxFusion runtime bridge.

This is the executable bridge for :mod:`boxfusion.online_candidate_map`.
One frozen WeDetect-Uni forward and one shared Boxer lift are run on the
current keyframe.  The resulting post-NMS proposals and top-M dense anchors
are committed immediately; no terminal replay is performed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import queue
import threading
import time
from typing import Mapping

import numpy as np
from PIL import Image

from boxfusion.online_candidate_map import (
    EvidenceFrame,
    NativeFrame,
    OnlineCandidateMap,
    OnlineMapSnapshot,
    build_online_candidate_map,
)


RUNTIME_SCHEMA = "boxfusion.online_candidate_runtime.v1"


def _numpy(value: object, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


class FrozenWeDetectBoxerProvider:
    """Fresh per-keyframe evidence from the frozen final experiment models."""

    def __init__(
        self,
        *,
        score_threshold: float = 0.05,
        top_proposals: int = 150,
        top_anchors: int = 300,
        diagnostics_root: str = "/tmp/boxfusion_online_candidate_map",
        device: str = "cuda:0",
        warmup: bool = True,
    ) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must lie in [0,1]")
        if top_proposals < 1 or top_anchors < 1:
            raise ValueError("provider caps must be positive")
        self.score_threshold = float(score_threshold)
        self.top_proposals = int(top_proposals)
        self.top_anchors = int(top_anchors)
        self.diagnostics_root = str(diagnostics_root)
        self.device = str(device)
        self.warmup = bool(warmup)
        self.warmup_seconds = 0.0
        self.detector = None
        self.lifter = None
        self.forward_seconds = []
        self.lifting_seconds = []

    def _load(self) -> None:
        if self.detector is not None:
            return
        import torch
        from tools.validate_ca1m_prenms_query import DenseCapture, build_lifter

        if self.device.startswith("cuda"):
            torch.cuda.set_device(torch.device(self.device))
        directory = Path(self.diagnostics_root)
        directory.mkdir(parents=True, exist_ok=True)
        self.detector = DenseCapture()
        self.lifter = build_lifter(directory)
        if self.warmup:
            from tools.validate_ca1m_prenms_query import lift

            started = time.perf_counter()
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            depth = np.ones((480, 640), dtype=np.float32)
            intrinsic = np.asarray(
                [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
            )
            self.detector.forward(Image.fromarray(rgb))
            lift(
                self.lifter,
                "__online_warmup__",
                0,
                rgb,
                depth,
                intrinsic,
                intrinsic,
                np.eye(4),
                np.asarray([[200.0, 150.0, 440.0, 350.0]], dtype=np.float32),
            )
            self.warmup_seconds = time.perf_counter() - started

    def process(
        self,
        *,
        scene_id: str,
        frame_id: int,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        image_intrinsic: np.ndarray,
        depth_intrinsic: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> EvidenceFrame:
        self._load()
        assert self.detector is not None and self.lifter is not None
        rgb_array = _numpy(rgb, dtype=np.uint8)
        depth_array = _numpy(depth_m, dtype=np.float32).squeeze()
        pil = Image.fromarray(rgb_array).convert("RGB")

        started = time.perf_counter()
        raw = self.detector.forward(pil)
        self.forward_seconds.append(time.perf_counter() - started)
        post_scores_all = np.asarray(raw["post_scores"], dtype=np.float64)
        post_keep = np.flatnonzero(post_scores_all >= self.score_threshold)
        if len(post_keep) > self.top_proposals:
            order = np.argsort(
                -post_scores_all[post_keep], kind="stable"
            )[: self.top_proposals]
            post_keep = post_keep[order]
        proposal_boxes = np.asarray(raw["post_boxes"], dtype=np.float32)[post_keep]
        proposal_scores = post_scores_all[post_keep]
        proposal_ids = np.asarray(raw["post_ids"], dtype=np.int64)[post_keep]

        anchor_scores_all = np.asarray(raw["scores"], dtype=np.float64)
        anchor_ids = np.argsort(-anchor_scores_all, kind="stable")[: self.top_anchors]
        anchor_boxes = np.asarray(raw["boxes"], dtype=np.float32)[anchor_ids]
        anchor_scores = anchor_scores_all[anchor_ids]
        combined = np.concatenate([proposal_boxes, anchor_boxes], axis=0)

        if len(combined):
            from tools.validate_ca1m_prenms_query import lift

            started = time.perf_counter()
            lifted, _, _ = lift(
                self.lifter,
                str(scene_id),
                int(frame_id),
                rgb_array,
                depth_array,
                _numpy(image_intrinsic),
                _numpy(depth_intrinsic),
                _numpy(camera_to_world),
                combined,
            )
            self.lifting_seconds.append(time.perf_counter() - started)
        else:
            lifted = np.empty((0, 8, 3), dtype=np.float64)
            self.lifting_seconds.append(0.0)
        split = len(proposal_boxes)
        proposal_corners = np.asarray(lifted[:split], dtype=np.float64)
        anchor_corners = np.asarray(lifted[split:], dtype=np.float64)
        proposal_valid = (
            np.isfinite(proposal_corners).all(axis=(1, 2))
            & (np.ptp(proposal_corners, axis=1) > 0.0).all(axis=1)
            if len(proposal_corners)
            else np.empty(0, dtype=bool)
        )
        anchor_valid = (
            np.isfinite(anchor_corners).all(axis=(1, 2))
            & (np.ptp(anchor_corners, axis=1) > 0.0).all(axis=1)
            if len(anchor_corners)
            else np.empty(0, dtype=bool)
        )
        return EvidenceFrame(
            proposal_ids=proposal_ids[proposal_valid],
            proposal_boxes_2d=proposal_boxes[proposal_valid],
            proposal_corners=proposal_corners[proposal_valid],
            proposal_scores=proposal_scores[proposal_valid],
            anchor_ids=anchor_ids[anchor_valid],
            anchor_corners=anchor_corners[anchor_valid],
            anchor_scores=anchor_scores[anchor_valid],
        )

    def diagnostics(self) -> dict:
        def timing(values):
            milliseconds = np.asarray(values, dtype=np.float64) * 1000.0
            return {
                "count": int(len(milliseconds)),
                "mean_ms": float(milliseconds.mean()) if len(milliseconds) else 0.0,
                "p50_ms": (
                    float(np.percentile(milliseconds, 50)) if len(milliseconds) else 0.0
                ),
                "p95_ms": (
                    float(np.percentile(milliseconds, 95)) if len(milliseconds) else 0.0
                ),
                "max_ms": float(milliseconds.max()) if len(milliseconds) else 0.0,
            }

        return {
            "frozen": True,
            "score_threshold": self.score_threshold,
            "top_proposals": self.top_proposals,
            "top_anchors": self.top_anchors,
            "device": self.device,
            "synthetic_warmup": self.warmup,
            "warmup_seconds_before_stream": self.warmup_seconds,
            "detector_forward": timing(self.forward_seconds),
            "boxer_lifting": timing(self.lifting_seconds),
        }


class OnlineCandidateRuntime:
    """Bind the causal state to BoxFusion and expose a live output snapshot."""

    def __init__(
        self,
        *,
        provider: FrozenWeDetectBoxerProvider,
        state_config: Mapping[str, object] | None = None,
        output_root: str | None = None,
        diagnostics_root: str | None = None,
        audit_output_root: str | None = None,
        write_every_keyframe: bool = True,
        asynchronous: bool = False,
        queue_capacity: int = 2,
        keyframe_deadline_ms: float = 833.333333,
        preload_provider: bool = True,
    ) -> None:
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        if keyframe_deadline_ms <= 0.0:
            raise ValueError("keyframe_deadline_ms must be positive")
        self.provider = provider
        self.state_config = dict(state_config or {})
        self.output_root = None if not output_root else Path(output_root)
        self.diagnostics_root = (
            None if not diagnostics_root else Path(diagnostics_root)
        )
        self.audit_output_root = (
            None if not audit_output_root else Path(audit_output_root)
        )
        self.write_every_keyframe = bool(write_every_keyframe)
        self.asynchronous = bool(asynchronous)
        self.queue_capacity = int(queue_capacity)
        self.keyframe_deadline_ms = float(keyframe_deadline_ms)
        self.preload_provider = bool(preload_provider)
        self.scene_id = None
        self.state = None
        self.last_snapshot = None
        self.keyframe_seconds = []
        self.end_to_end_seconds = []
        self.enqueue_seconds = []
        self.serialisation_seconds = []
        self.max_queue_depth = 0
        self.pending_at_close = 0
        self.close_drain_seconds = 0.0
        self.terminal_assembly_seconds = []
        self.terminal_materialized = False
        self._closed = False
        self._worker_error = None
        self._worker_ready = threading.Event()
        self._pending_submissions = {}
        self._last_submitted_frame_id = -1
        self._queue = None
        self._worker = None
        if self.asynchronous:
            self._queue = queue.Queue(maxsize=self.queue_capacity)
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="boxfusion-online-candidate-map",
                daemon=True,
            )
            self._worker.start()
            self._worker_ready.wait()
            self._raise_worker_error()

    def _bind(self, scene_id: str) -> None:
        if self.state is None:
            self.scene_id = str(scene_id)
            if self.state_config.get("variant") == "reliability_v2":
                from boxfusion.reliability_candidate_map import (
                    build_reliability_candidate_map,
                )

                self.state = build_reliability_candidate_map(
                    self.scene_id, self.state_config
                )
            else:
                self.state = build_online_candidate_map(
                    self.scene_id, self.state_config
                )
            if self.output_root is not None:
                self.output_root.mkdir(parents=True, exist_ok=True)
            if self.diagnostics_root is not None:
                self.diagnostics_root.mkdir(parents=True, exist_ok=True)
            if self.audit_output_root is not None:
                self.audit_output_root.mkdir(parents=True, exist_ok=True)
        elif self.scene_id != str(scene_id):
            raise ValueError("one online candidate runtime cannot mix scenes")

    def process_keyframe(
        self,
        *,
        scene_id: str,
        frame_id: int,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        image_intrinsic: np.ndarray,
        depth_intrinsic: np.ndarray,
        camera_to_world: np.ndarray,
        native_ids: np.ndarray,
        native_corners: np.ndarray,
        native_scores: np.ndarray,
    ) -> OnlineMapSnapshot | None:
        self.begin_keyframe(
            scene_id=scene_id,
            frame_id=frame_id,
            rgb=rgb,
            depth_m=depth_m,
            image_intrinsic=image_intrinsic,
            depth_intrinsic=depth_intrinsic,
            camera_to_world=camera_to_world,
        )
        return self.finish_keyframe(
            frame_id=frame_id,
            native_ids=native_ids,
            native_corners=native_corners,
            native_scores=native_scores,
        )

    def begin_keyframe(
        self,
        *,
        scene_id: str,
        frame_id: int,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        image_intrinsic: np.ndarray,
        depth_intrinsic: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> None:
        """Start frozen evidence extraction before native inference completes."""
        if self._closed:
            raise RuntimeError("online candidate runtime is closed")
        self._raise_worker_error()
        self._bind(scene_id)
        frame_id = int(frame_id)
        if frame_id <= self._last_submitted_frame_id:
            raise ValueError("submitted frame_id must increase strictly")
        if frame_id in self._pending_submissions:
            raise ValueError("keyframe is already pending")
        task = {
            "enqueued_at": time.perf_counter(),
            "scene_id": str(scene_id),
            "frame_id": frame_id,
            "rgb": _numpy(rgb, dtype=np.uint8).copy(),
            "depth_m": _numpy(depth_m, dtype=np.float32).squeeze().copy(),
            "image_intrinsic": _numpy(image_intrinsic).copy(),
            "depth_intrinsic": _numpy(depth_intrinsic).copy(),
            "camera_to_world": _numpy(camera_to_world).copy(),
            "native_ready": threading.Event(),
        }
        self._pending_submissions[frame_id] = task
        self._last_submitted_frame_id = frame_id
        if self.asynchronous:
            assert self._queue is not None
            started = time.perf_counter()
            self._queue.put(task)
            self.enqueue_seconds.append(time.perf_counter() - started)
            self.max_queue_depth = max(self.max_queue_depth, self._queue.qsize())
            self._raise_worker_error()
            return
        task["evidence"] = self._extract_evidence(task)

    def finish_keyframe(
        self,
        *,
        frame_id: int,
        native_ids: np.ndarray,
        native_corners: np.ndarray,
        native_scores: np.ndarray,
        child_ids: np.ndarray = np.empty(0, dtype=np.int64),
        child_corners: np.ndarray = np.empty((0, 8, 3)),
        child_scores: np.ndarray = np.empty(0),
    ) -> OnlineMapSnapshot | None:
        """Attach the causal native prefix and make this keyframe committable."""
        self._raise_worker_error()
        frame_id = int(frame_id)
        if frame_id not in self._pending_submissions:
            raise ValueError("keyframe was not begun")
        task = self._pending_submissions.pop(frame_id)
        task["native_ids"] = _numpy(native_ids, dtype=np.int64).reshape(-1).copy()
        task["native_corners"] = _numpy(native_corners).reshape(-1, 8, 3).copy()
        task["native_scores"] = _numpy(native_scores).reshape(-1).copy()
        task["child_ids"] = _numpy(child_ids, dtype=np.int64).reshape(-1).copy()
        task["child_corners"] = _numpy(child_corners).reshape(-1, 8, 3).copy()
        task["child_scores"] = _numpy(child_scores).reshape(-1).copy()
        task["native_ready"].set()
        if self.asynchronous:
            return self.last_snapshot
        return self._commit_task(task, task["evidence"])

    def _extract_evidence(self, task: Mapping[str, object]) -> EvidenceFrame:
        return self.provider.process(
            scene_id=task["scene_id"],
            frame_id=task["frame_id"],
            rgb=task["rgb"],
            depth_m=task["depth_m"],
            image_intrinsic=task["image_intrinsic"],
            depth_intrinsic=task["depth_intrinsic"],
            camera_to_world=task["camera_to_world"],
        )

    def _commit_task(
        self, task: Mapping[str, object], evidence: EvidenceFrame
    ) -> OnlineMapSnapshot:
        assert self.state is not None
        started = time.perf_counter()
        if len(task["child_ids"]):
            evidence = EvidenceFrame(
                proposal_ids=evidence.proposal_ids,
                proposal_boxes_2d=evidence.proposal_boxes_2d,
                proposal_corners=evidence.proposal_corners,
                proposal_scores=evidence.proposal_scores,
                anchor_ids=evidence.anchor_ids,
                anchor_corners=evidence.anchor_corners,
                anchor_scores=evidence.anchor_scores,
                child_ids=task["child_ids"],
                child_corners=task["child_corners"],
                child_scores=task["child_scores"],
            )
        native = NativeFrame(
            ids=task["native_ids"],
            corners=task["native_corners"],
            scores=task["native_scores"],
            camera_to_world=task["camera_to_world"],
            intrinsic=task["image_intrinsic"],
            width=int(np.asarray(task["rgb"]).shape[1]),
            height=int(np.asarray(task["rgb"]).shape[0]),
        )
        snapshot = self.state.update(int(task["frame_id"]), native, evidence)
        self.last_snapshot = snapshot
        if self.write_every_keyframe:
            self._write(snapshot)
        self.keyframe_seconds.append(time.perf_counter() - started)
        self.end_to_end_seconds.append(
            time.perf_counter() - float(task["enqueued_at"])
        )
        return snapshot

    def _worker_loop(self) -> None:
        assert self._queue is not None
        try:
            if self.preload_provider:
                loader = getattr(self.provider, "_load", None)
                if callable(loader):
                    loader()
        except BaseException as error:
            self._worker_error = error
            self._worker_ready.set()
            return
        self._worker_ready.set()
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                evidence = self._extract_evidence(task)
                task["native_ready"].wait()
                self._commit_task(task, evidence)
            except BaseException as error:
                self._worker_error = error
                return
            finally:
                self._queue.task_done()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("online candidate worker failed") from self._worker_error

    def _write(self, snapshot: OnlineMapSnapshot) -> None:
        if self.output_root is None:
            return
        assert self.scene_id is not None
        started = time.perf_counter()
        output_path = self.output_root / f"{self.scene_id}_boxes.pkl"
        temporary = output_path.with_name(output_path.name + ".tmp")
        payload = [[
            (0, np.array(box, copy=True), float(score))
            for box, score in zip(snapshot.boxes, snapshot.scores)
        ]]
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        self.serialisation_seconds.append(time.perf_counter() - started)

    def _write_diagnostics(self, report: Mapping[str, object]) -> None:
        if self.diagnostics_root is None or self.scene_id is None:
            return
        path = self.diagnostics_root / f"{self.scene_id}.json"
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)

    def _write_audit_components(
        self, components: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        if self.audit_output_root is None or self.scene_id is None:
            return
        counts = {}
        for name, (boxes_value, scores_value) in components.items():
            boxes = np.asarray(boxes_value, dtype=np.float64).reshape(-1, 8, 3)
            scores = np.asarray(scores_value, dtype=np.float64).reshape(-1)
            if len(boxes) != len(scores):
                raise ValueError(f"audit component {name} is misaligned")
            directory = self.audit_output_root / name
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"{self.scene_id}_boxes.pkl"
            temporary = destination.with_name(destination.name + ".tmp")
            payload = [[
                (0, np.array(box, copy=True), float(score))
                for box, score in zip(boxes, scores)
            ]]
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            counts[name] = len(boxes)
        manifest = self.audit_output_root / f"{self.scene_id}.json"
        temporary = manifest.with_name(manifest.name + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": "boxfusion.reliability_v2.components.v1",
                    "scene_id": self.scene_id,
                    "counts": counts,
                    "single_online_pass": True,
                    "future_frames_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest)

    def materialize_terminal(
        self,
        *,
        frame_id: int,
        native_ids: np.ndarray,
        native_corners: np.ndarray,
        native_scores: np.ndarray,
    ) -> tuple[OnlineMapSnapshot, dict]:
        """Write the causal state on the host mapper's normal final map."""
        if not self._closed:
            raise RuntimeError("terminal materialization requires a drained runtime")
        if self.state is None:
            raise RuntimeError("terminal materialization requires initialized state")
        started = time.perf_counter()
        snapshot = self.state.materialize_terminal(
            int(frame_id), native_ids, native_corners, native_scores
        )
        component_reader = getattr(self.state, "experiment_components", None)
        if self.audit_output_root is not None:
            if not callable(component_reader):
                raise RuntimeError("audit output requires experiment_components")
            self._write_audit_components(
                component_reader(native_ids, native_corners, native_scores)
            )
        self.last_snapshot = snapshot
        self._write(snapshot)
        self.terminal_assembly_seconds.append(time.perf_counter() - started)
        self.terminal_materialized = True
        report = self.diagnostics()
        self._write_diagnostics(report)
        return snapshot, report

    def close(self) -> dict:
        """Persist the current prefix; no scene-end inference is performed."""
        if self._closed:
            return self.diagnostics()
        if self._pending_submissions:
            raise RuntimeError(
                "cannot close with unfinished keyframe submissions: "
                f"{sorted(self._pending_submissions)}"
            )
        self._closed = True
        if self.asynchronous:
            assert self._queue is not None and self._worker is not None
            self._raise_worker_error()
            self.pending_at_close = self._queue.qsize()
            started = time.perf_counter()
            self._queue.put(None)
            self._worker.join()
            self.close_drain_seconds = time.perf_counter() - started
            self._raise_worker_error()
        if self.last_snapshot is not None and not self.write_every_keyframe:
            self._write(self.last_snapshot)
        report = self.diagnostics()
        self._write_diagnostics(report)
        return report

    def diagnostics(self) -> dict:
        milliseconds = np.asarray(self.keyframe_seconds, dtype=np.float64) * 1000.0
        end_to_end = np.asarray(
            self.end_to_end_seconds, dtype=np.float64
        ) * 1000.0
        enqueue = np.asarray(self.enqueue_seconds, dtype=np.float64) * 1000.0
        serialisation = np.asarray(
            self.serialisation_seconds, dtype=np.float64
        ) * 1000.0
        terminal_assembly = np.asarray(
            self.terminal_assembly_seconds, dtype=np.float64
        ) * 1000.0

        def summary(values):
            return {
                "count": int(len(values)),
                "mean_ms": float(values.mean()) if len(values) else 0.0,
                "p50_ms": float(np.percentile(values, 50)) if len(values) else 0.0,
                "p95_ms": float(np.percentile(values, 95)) if len(values) else 0.0,
                "max_ms": float(values.max()) if len(values) else 0.0,
            }

        return {
            "schema": RUNTIME_SCHEMA,
            "scene_id": self.scene_id,
            "strictly_causal": True,
            "online_incremental": True,
            "scene_end_inference": False,
            "scene_end_assembly": self.terminal_materialized,
            "write_every_keyframe": self.write_every_keyframe,
            "asynchronous": self.asynchronous,
            "queue_capacity": self.queue_capacity,
            "max_queue_depth": self.max_queue_depth,
            "pending_at_close": self.pending_at_close,
            "close_drain_ms": self.close_drain_seconds * 1000.0,
            "keyframe_deadline_ms": self.keyframe_deadline_ms,
            "provider_preloaded_before_stream": self.preload_provider,
            "audit_output_root": (
                None if self.audit_output_root is None else str(self.audit_output_root)
            ),
            "deadline_misses": int(
                np.count_nonzero(end_to_end > self.keyframe_deadline_ms)
            ),
            "keyframe_total": summary(milliseconds),
            "keyframe_end_to_end": summary(end_to_end),
            "enqueue_blocking": summary(enqueue),
            "serialisation": summary(serialisation),
            "terminal_assembly": summary(terminal_assembly),
            "provider": self.provider.diagnostics(),
            "state": None if self.state is None else self.state.diagnostics(),
        }


def build_online_candidate_runtime(
    config: Mapping[str, object] | None,
) -> OnlineCandidateRuntime | None:
    section = {} if config is None else dict(config)
    if not bool(section.get("enabled", False)):
        return None
    provider_row = dict(section.get("provider", {}) or {})
    provider = FrozenWeDetectBoxerProvider(
        score_threshold=float(provider_row.get("score_threshold", 0.05)),
        top_proposals=int(provider_row.get("top_proposals", 150)),
        top_anchors=int(provider_row.get("top_anchors", 300)),
        diagnostics_root=str(
            provider_row.get(
                "diagnostics_root", "/tmp/boxfusion_online_candidate_map"
            )
        ),
        device=str(provider_row.get("device", "cuda:0")),
        warmup=bool(provider_row.get("warmup", True)),
    )
    return OnlineCandidateRuntime(
        provider=provider,
        state_config=dict(section.get("state", {}) or {}),
        output_root=section.get("output_root"),
        diagnostics_root=section.get("diagnostics_root"),
        audit_output_root=section.get("audit_output_root"),
        write_every_keyframe=bool(section.get("write_every_keyframe", True)),
        asynchronous=bool(section.get("asynchronous", False)),
        queue_capacity=int(section.get("queue_capacity", 2)),
        keyframe_deadline_ms=float(section.get("keyframe_deadline_ms", 833.333333)),
        preload_provider=bool(section.get("preload_provider", True)),
    )

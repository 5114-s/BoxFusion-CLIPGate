"""Immutable-manifest contracts for the isolated P0/P1/P2 runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch


_TOOL = Path(__file__).resolve().parents[1] / "tools" / "build_p_run_manifest.py"
_SPEC = importlib.util.spec_from_file_location("build_p_run_manifest", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    for relative in (
        "boxfusion/residual_proposal.py",
        "boxfusion/occupancy_topk.py",
        "boxfusion/online_refinement.py",
        "boxfusion/p_ablation.py",
        "demo.py",
        "scripts/run_scannet_online_refinement.sh",
        "scripts/run_scannet_p_ablation.sh",
        "tools/build_p_run_manifest.py",
    ):
        _write(root / relative, relative.encode("utf-8"))
    return root


def _args(tmp_path: Path, *, stage: str = "P1") -> argparse.Namespace:
    root = _root(tmp_path)
    for directory in ("live", "frames", "gt", "scans"):
        (root / directory).mkdir(parents=True)
    b6_checkpoint = _write(root / "b6.npz", b"b6")
    forbidden_scene_list = _write(
        root / "validation.txt", b"scene0700_00\n"
    )
    p1_checkpoint = None
    if stage in {"P1", "P2"}:
        p1_checkpoint = root / "p1.pt"
        torch.save(
            {
                "provenance": {
                    "train_scene_ids": ["scene0001_00"],
                    "forbidden_overlap": [],
                    "train_scene_list_sha256": "1" * 64,
                    "forbidden_scene_list_sha256": hashlib.sha256(
                        forbidden_scene_list.read_bytes()
                    ).hexdigest(),
                    "b6_checkpoint_sha256": hashlib.sha256(
                        b6_checkpoint.read_bytes()
                    ).hexdigest(),
                }
            },
            p1_checkpoint,
        )
    p2_checkpoint = None
    if stage == "P2":
        p2_checkpoint = root / "p2.pt"
        torch.save(
            {
                "schema": _MODULE._P2_HEAD_SCHEMA,
                "feature_names": list(_MODULE._P1_FEATURE_NAMES),
                "model_config": {"input_dim": 14, "hidden_dim": 32},
                "state_dict": {"weight": torch.zeros(1)},
                "provenance": {
                    "train_scene_ids": ["scene0001_00"],
                    "forbidden_overlap": [],
                    "train_scene_list_sha256": "1" * 64,
                    "forbidden_scene_list_sha256": hashlib.sha256(
                        forbidden_scene_list.read_bytes()
                    ).hexdigest(),
                    "p1_checkpoint_sha256": hashlib.sha256(
                        p1_checkpoint.read_bytes()
                    ).hexdigest(),
                    "b6_checkpoint_sha256": hashlib.sha256(
                        b6_checkpoint.read_bytes()
                    ).hexdigest(),
                },
            },
            p2_checkpoint,
        )
    return argparse.Namespace(
        root=root,
        stage=stage,
        profile=(
            {
                "P0": "p0_frozen_b6",
                "P1": "p1_residual_proposal_observer",
                "P2": "p2_occupancy_topk_observer",
            }[stage]
        ),
        scene_list=_write(root / "scenes.txt", b"scene0001_00\n"),
        config=_write(root / "config.yaml", b"dataset: scannet\n"),
        b6_checkpoint=b6_checkpoint,
        yoloe_checkpoint=_write(root / "yoloe.pt", b"yoloe"),
        cutr_checkpoint=_write(root / "cutr.pth", b"cutr"),
        clip_checkpoint=_write(root / "clip.bin", b"clip"),
        class_features=_write(root / "features.pt", b"features"),
        class_list=_write(root / "classes.txt", b"chair\n"),
        pst_texture=_write(root / "pst.tiff", b"pst"),
        p1_checkpoint=p1_checkpoint,
        p2_checkpoint=p2_checkpoint,
        forbidden_scene_list=(
            forbidden_scene_list if stage in {"P1", "P2"} else None
        ),
        b6_detector_blend=0.4,
        minimum_extent=0.4,
        post_minimum_extent="",
        proposal_interval=5,
        quality_mode="iou_mlp",
        proposal_provider="yoloe",
        candidate_ttl_clock="provider_call",
        candidate_track_ttl=3,
        archive_confirmed_tracks=0,
        inference_seed=0,
        evaluation_seed=0,
        live_root=root / "live",
        frames_root=root / "frames",
        ground_truth_root=root / "gt",
        scans_root=root / "scans",
        python_executable=Path(__import__("sys").executable),
        prediction_root=root / "results" / stage.lower(),
        log_root=root / "logs" / stage.lower(),
        diagnostics_root=root / "diagnostics" / stage.lower(),
        evaluation_root=root / "evaluation" / stage.lower(),
        manifest=root / "logs" / stage.lower() / "run_manifest.json",
    )


def test_p1_manifest_binds_all_models_code_and_output_roots(tmp_path):
    args = _args(tmp_path)
    payload = _MODULE.build_manifest(args)

    assert payload["stage"] == "P1"
    assert payload["scene_count"] == 1
    assert len(payload["b6_checkpoint_sha256"]) == 64
    assert len(payload["yoloe_checkpoint_sha256"]) == 64
    assert len(payload["p1_checkpoint_sha256"]) == 64
    assert payload["log_root"] == str(args.log_root.resolve())
    assert payload["evaluation_root"] == str(args.evaluation_root.resolve())
    assert "tools/build_p_run_manifest.py" in payload["code_sha256"]


def test_manifest_resume_is_exact_and_changed_input_fails_closed(tmp_path):
    args = _args(tmp_path)
    payload = _MODULE.build_manifest(args)
    assert _MODULE.write_or_verify(
        payload,
        manifest_path=args.manifest,
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
    ) == "created"
    assert _MODULE.write_or_verify(
        payload,
        manifest_path=args.manifest,
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
    ) == "verified"

    args.config.write_text("dataset: changed\n", encoding="utf-8")
    changed = _MODULE.build_manifest(args)
    with pytest.raises(ValueError, match="manifest disagrees"):
        _MODULE.write_or_verify(
            changed,
            manifest_path=args.manifest,
            prediction_root=args.prediction_root,
            diagnostics_root=args.diagnostics_root,
        )


def test_p2_manifest_binds_exact_p1_p2_and_b6_chain(tmp_path):
    args = _args(tmp_path, stage="P2")
    payload = _MODULE.build_manifest(args)

    assert payload["stage"] == "P2"
    assert payload["profile"] == "p2_occupancy_topk_observer"
    assert payload["p2_checkpoint_sha256"] == hashlib.sha256(
        args.p2_checkpoint.read_bytes()
    ).hexdigest()
    assert payload["p2_training_provenance"][
        "p1_checkpoint_sha256"
    ] == payload["p1_checkpoint_sha256"]
    assert payload["p2_training_provenance"][
        "b6_checkpoint_sha256"
    ] == payload["b6_checkpoint_sha256"]
    assert "boxfusion/occupancy_topk.py" in payload["code_sha256"]


def test_artifacts_without_manifest_are_rejected(tmp_path):
    args = _args(tmp_path, stage="P0")
    payload = _MODULE.build_manifest(args)
    _write(args.prediction_root / "scene0001_00_boxes.pkl", b"old")

    with pytest.raises(ValueError, match="artifacts exist without a manifest"):
        _MODULE.write_or_verify(
            payload,
            manifest_path=args.manifest,
            prediction_root=args.prediction_root,
            diagnostics_root=args.diagnostics_root,
        )

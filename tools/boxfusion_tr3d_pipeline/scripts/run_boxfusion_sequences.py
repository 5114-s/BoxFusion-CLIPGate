#!/usr/bin/env python3
"""Run BoxFusion demo.py over a list of sequences with resume-friendly logs."""

from __future__ import annotations

import argparse
from collections import Counter
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PIPELINE_ROOT.parents[1]
R2_SIDECAR_SUFFIX = "_openbox_smov_r2_shadow.npz"
R2_EXPECTED_SCHEMA = "boxfusion.openbox_smov_r2_shadow.v2"


def resolve_demo_config_path(value: str) -> Path:
    """Resolve a config path exactly as demo.py does from its fixed cwd."""

    path = Path(value)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def duplicate_sequences(seqs: list[str]) -> list[str]:
    counts = Counter(seqs)
    return sorted(seq for seq, count in counts.items() if count > 1)


def r2_diagnostics_root(
    cfg: Mapping[str, object], override: str | None
) -> tuple[bool, Path | None]:
    section = cfg.get("openbox_smov_r2", {})
    if not isinstance(section, Mapping):
        raise ValueError("openbox_smov_r2 must be a mapping")
    enabled = bool(section.get("enabled", False))
    if override is not None and not enabled:
        raise ValueError(
            "--openbox-smov-r2-diagnostics-root requires an enabled "
            "OpenBox-SMOV R2 observer"
        )
    if not enabled:
        return False, None

    diagnostics = section.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        raise ValueError("openbox_smov_r2.diagnostics must be a mapping")
    value = override if override is not None else diagnostics.get("root")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "enabled OpenBox-SMOV R2 requires a non-empty diagnostics root"
        )
    if override is not None:
        return True, Path(value).resolve()
    return True, resolve_demo_config_path(value)


def r2_artifact_pair(
    output_root: Path, diagnostics_root: Path, seq: str
) -> tuple[Path, Path]:
    return (
        output_root / f"{seq}_boxes.pkl",
        diagnostics_root / f"{seq}{R2_SIDECAR_SUFFIX}",
    )


def paired_state(prediction: Path, sidecar: Path) -> str:
    present = (prediction.exists(), sidecar.exists())
    if present == (True, True):
        return "complete"
    if present == (False, False):
        return "missing"
    return "partial"


def r2_sidecar_schema(sidecar: Path) -> str:
    """Read only the scalar provenance needed for safe paired resume."""

    if sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError(f"R2 sidecar is not a regular file: {sidecar}")
    try:
        with np.load(sidecar, allow_pickle=False) as archive:
            if "schema" not in archive.files:
                raise ValueError("missing schema member")
            raw = np.asarray(archive["schema"])
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read R2 sidecar schema: {sidecar}") from error
    if raw.shape != () or raw.dtype.kind not in "US":
        raise ValueError(f"invalid R2 sidecar schema scalar: {sidecar}")
    return str(raw.item())


def read_sequences(path: Path, dataset: str) -> list[str]:
    if dataset.lower() == "ca1m":
        seqs = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            stem = Path(line).stem
            seqs.append(stem.replace("ca1m-val-", ""))
        return seqs
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_sequence_datadir(config_datadir: str, dataset: str, seq: str) -> Path:
    if dataset.lower() == "ca1m":
        if "example" in config_datadir:
            return Path(config_datadir)
        return Path(config_datadir).parent.parent / seq
    return Path(config_datadir).parent.parent / seq / "frames"


def missing_reason(datadir: Path, dataset: str) -> str | None:
    if dataset.lower() == "ca1m":
        required_files = [
            datadir / "K_depth.txt",
            datadir / "K_rgb.txt",
            datadir / "all_poses.npy",
            datadir / "after_filter_boxes.npy",
        ]
        required_dirs = [datadir / "rgb", datadir / "depth"]
    else:
        required_files = [datadir / "intrinsic" / "intrinsic_depth.txt"]
        required_dirs = [datadir / "color", datadir / "depth", datadir / "pose"]

    missing = [str(p) for p in required_files if not p.exists()]
    missing.extend(str(p) for p in required_dirs if not p.is_dir())
    if missing:
        return "missing " + ", ".join(missing[:4])
    if dataset.lower() == "ca1m":
        rgb_count = len(list((datadir / "rgb").glob("*.png")))
        depth_count = len(list((datadir / "depth").glob("*.png")))
        try:
            import numpy as np

            expected = int(np.load(datadir / "all_poses.npy", mmap_mode="r").shape[0])
        except Exception:
            expected = None
        if rgb_count == 0 or depth_count == 0:
            return f"incomplete frames rgb={rgb_count} depth={depth_count}"
        if rgb_count != depth_count:
            return f"incomplete frames rgb={rgb_count} depth={depth_count}"
        if expected is not None and rgb_count < expected:
            return f"incomplete frames rgb={rgb_count} depth={depth_count} expected={expected}"
    else:
        color_count = len(list((datadir / "color").glob("*.jpg")))
        depth_count = len(list((datadir / "depth").glob("*.png")))
        pose_count = len(list((datadir / "pose").glob("*.txt")))
        if color_count == 0 or depth_count == 0 or pose_count == 0:
            return f"incomplete frames color={color_count} depth={depth_count} pose={pose_count}"
        if color_count != depth_count or color_count != pose_count:
            return f"incomplete frames color={color_count} depth={depth_count} pose={pose_count}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["CA1M", "scannet"], required=True)
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", default="./models/cutr_rgbd.pth")
    parser.add_argument(
        "--clip-path",
        default="./models/open_clip_pytorch_model.bin",
    )
    parser.add_argument(
        "--class-txt",
        default="./data/panoptic_categories_nomerge.txt",
    )
    parser.add_argument(
        "--class-features",
        default="./data/class_features.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override config data.output_dir and keep paired runs separate",
    )
    parser.add_argument(
        "--openbox-smov-r2-diagnostics-root",
        default=None,
        help=(
            "Override the create-only OpenBox-SMOV R2 diagnostics root; "
            "paired resume requires one sidecar per native prediction"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Forward a deterministic inference seed to demo.py",
    )
    parser.add_argument(
        "--scannet-frames-root",
        default=None,
        help="Forward an explicit <scene>/frames root to demo.py",
    )
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")

    config_path = Path(args.config).resolve()
    with config_path.open("r") as f:
        cfg = yaml.full_load(f)
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.output_dir is not None:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = resolve_demo_config_path(cfg["data"]["output_dir"])
    try:
        r2_enabled, r2_diag_dir = r2_diagnostics_root(
            cfg, args.openbox_smov_r2_diagnostics_root
        )
    except ValueError as error:
        parser.error(str(error))
    if r2_enabled and out_dir == r2_diag_dir:
        parser.error(
            "OpenBox-SMOV R2 native output and diagnostics roots must differ"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if r2_diag_dir is not None:
        r2_diag_dir.mkdir(parents=True, exist_ok=True)

    all_seqs = read_sequences(Path(args.seq_list).resolve(), args.dataset)
    if r2_enabled:
        duplicates = duplicate_sequences(all_seqs)
        if duplicates:
            parser.error(
                "OpenBox-SMOV R2 sequence list contains duplicate scene IDs: "
                + " ".join(duplicates)
            )
    seqs = [seq for i, seq in enumerate(all_seqs) if i % args.num_shards == args.shard_index]
    if r2_enabled:
        partial_pairs = []
        wrong_schema_pairs = []
        for seq in seqs:
            pred, sidecar = r2_artifact_pair(out_dir, r2_diag_dir, seq)
            state = paired_state(pred, sidecar)
            if state == "partial":
                partial_pairs.append(
                    f"{seq}(prediction={pred.exists()},sidecar={sidecar.exists()})"
                )
            elif state == "complete":
                try:
                    observed_schema = r2_sidecar_schema(sidecar)
                except ValueError as error:
                    wrong_schema_pairs.append(f"{seq}({error})")
                else:
                    if observed_schema != R2_EXPECTED_SCHEMA:
                        wrong_schema_pairs.append(
                            f"{seq}(schema={observed_schema})"
                        )
        if partial_pairs:
            parser.error(
                "OpenBox-SMOV R2 found incomplete prediction/sidecar pairs; "
                "refusing create-only resume: " + " ".join(partial_pairs)
            )
        if wrong_schema_pairs:
            parser.error(
                "OpenBox-SMOV R2 paired resume requires schema "
                f"{R2_EXPECTED_SCHEMA}; use fresh v2 roots: "
                + " ".join(wrong_schema_pairs)
            )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("MPLCONFIGDIR", "/tmp/boxfusion_mpl")
    env.setdefault("XDG_CACHE_HOME", "/tmp/boxfusion_xdg")

    failures: list[str] = []
    skipped_missing: list[str] = []
    for idx, seq in enumerate(seqs, start=1):
        pred = out_dir / f"{seq}_boxes.pkl"
        r2_sidecar = None
        if r2_enabled:
            pred, r2_sidecar = r2_artifact_pair(out_dir, r2_diag_dir, seq)
            if paired_state(pred, r2_sidecar) == "complete":
                print(
                    f"[{idx}/{len(seqs)}] skip {seq}: paired R2 artifacts exist",
                    flush=True,
                )
                continue
        elif pred.exists():
            print(f"[{idx}/{len(seqs)}] skip {seq}: {pred} exists", flush=True)
            continue

        if args.skip_missing:
            datadir = resolve_sequence_datadir(cfg["data"]["datadir"], args.dataset, seq)
            reason = missing_reason(datadir, args.dataset)
            if reason is not None:
                skipped_missing.append(seq)
                print(f"[{idx}/{len(seqs)}] skip {seq}: {reason}", flush=True)
                continue

        log_path = log_dir / f"{seq}.log"
        cmd = [
            sys.executable,
            str(PIPELINE_ROOT / "demo.py"),
            args.dataset,
            "--model-path",
            str(Path(args.model_path).resolve()),
            "--clip_path",
            str(Path(args.clip_path).resolve()),
            "--class_txt",
            str(Path(args.class_txt).resolve()),
            "--class-features",
            str(Path(args.class_features).resolve()),
            "--config",
            str(config_path),
            "--device",
            args.device,
            "--seq",
            seq,
        ]
        if args.output_dir is not None:
            cmd.extend(("--output-dir", str(out_dir)))
        if r2_enabled:
            cmd.extend(
                (
                    "--openbox-smov-r2-diagnostics-root",
                    str(r2_diag_dir),
                )
            )
        if args.seed is not None:
            cmd.extend(("--seed", str(args.seed)))
        if args.scannet_frames_root is not None:
            cmd.extend(
                ("--scannet-frames-root", args.scannet_frames_root)
            )
        print(f"[{idx}/{len(seqs)}] run {seq}; log={log_path}", flush=True)
        with open(log_path, "w") as log:
            proc = subprocess.run(
                cmd,
                cwd=REPOSITORY_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            failures.append(seq)
            print(f"[{idx}/{len(seqs)}] FAILED {seq} rc={proc.returncode}", flush=True)
        elif r2_enabled:
            if paired_state(pred, r2_sidecar) != "complete":
                failures.append(seq)
                print(
                    f"[{idx}/{len(seqs)}] FAILED {seq}: demo returned 0 but "
                    "the native prediction and R2 sidecar are not both present",
                    flush=True,
                )
            else:
                try:
                    observed_schema = r2_sidecar_schema(r2_sidecar)
                except ValueError as error:
                    failures.append(seq)
                    print(
                        f"[{idx}/{len(seqs)}] FAILED {seq}: {error}",
                        flush=True,
                    )
                else:
                    if observed_schema != R2_EXPECTED_SCHEMA:
                        failures.append(seq)
                        print(
                            f"[{idx}/{len(seqs)}] FAILED {seq}: sidecar "
                            f"schema {observed_schema} != {R2_EXPECTED_SCHEMA}",
                            flush=True,
                        )
                    else:
                        print(f"[{idx}/{len(seqs)}] done {seq}", flush=True)
        else:
            print(f"[{idx}/{len(seqs)}] done {seq}", flush=True)

    if failures:
        print("failed sequences:", " ".join(failures), flush=True)
        return 1
    if skipped_missing:
        print("skipped missing sequences:", " ".join(skipped_missing), flush=True)
    print("all requested sequences finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

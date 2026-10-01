#!/usr/bin/env python3
"""Run exact100 CA terminal observation from a sealed CA-native checkpoint.

This v3 entry point reuses the established CA terminal-cache schema and core
geometry implementation.  It deliberately does not accept raw model config or
checkpoint arguments: both are resolved only from the independently rehashed
CA scratch-training binding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_checkpoint_binding import (  # noqa: E402
    load_checkpoint_binding,
)
from boxfusion.ca1m_tr3d_terminal import sha256_file  # noqa: E402
from tools import run_ca1m_tr3d_terminal_observer as shared  # noqa: E402
from tools.preflight_ca1m_tr3d_terminal_train100_v3 import (  # noqa: E402
    validate_config,
)


def parser() -> argparse.ArgumentParser:
    value = shared.parser()
    value.description = __doc__
    for action in value._actions:
        if action.dest in {"tr3d_config", "tr3d_checkpoint"}:
            action.required = False
            action.default = None
            action.help = argparse.SUPPRESS
    value.add_argument(
        "--collection-config",
        type=Path,
        default=ROOT / "config/ca1m_tr3d_terminal_train100_v3.json",
    )
    value.add_argument("--tr3d-binding-manifest", type=Path, required=True)
    return value


def _code_sources(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "runner_v3": Path(__file__),
        "observer_shared": ROOT / "tools/run_ca1m_tr3d_terminal_observer.py",
        "checkpoint_binding_core": ROOT / "boxfusion/ca1m_tr3d_checkpoint_binding.py",
        "collection_preflight": (
            ROOT / "tools/preflight_ca1m_tr3d_terminal_train100_v3.py"
        ),
        "collection_config": args.collection_config,
        "checkpoint_binding": args.tr3d_binding_manifest,
        "worker": args.worker_script,
        "terminal_core": ROOT / "boxfusion/ca1m_tr3d_terminal.py",
        "worker_client": ROOT / "boxfusion/ca1m_tr3d_worker_client.py",
        "native_b6_score": ROOT / "boxfusion/ca1m_native_b6_score.py",
        "rgbd_backprojection": ROOT / "boxfusion/tr3d_incremental_online.py",
        "official_adapter": args.runtime_root / "boxfusion/tr3d_inference.py",
    }


def _code_manifest(args: argparse.Namespace) -> str:
    files = {
        name: sha256_file(shared._regular(path, f"v3 code source {name}"))
        for name, path in sorted(_code_sources(args).items())
    }
    return json.dumps(
        {"schema": "boxfusion.ca1m_tr3d_terminal_code_manifest.v1", "files": files},
        sort_keys=True,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.tr3d_config is not None or args.tr3d_checkpoint is not None:
        raise ValueError("v3 forbids raw --tr3d-config/--tr3d-checkpoint arguments")
    if args.synthetic:
        raise ValueError("formal v3 collection requires genuine CA-native TR3D")
    binding = load_checkpoint_binding(args.tr3d_binding_manifest)
    preflight = validate_config(args.collection_config, args.tr3d_binding_manifest)
    if not preflight["ready_for_gpu"] or preflight["scene_count"] != 100:
        raise ValueError("v3 runtime preflight did not authorize exact100 collection")
    args.collection_config = Path(args.collection_config).resolve()
    args.tr3d_binding_manifest = binding.manifest_path
    args.tr3d_config = binding.effective_config_path
    args.tr3d_checkpoint = binding.checkpoint_path
    original_checkpoint = shared.EXPECTED_CHECKPOINT_SHA256
    original_config = shared.EXPECTED_CONFIG_SHA256
    original_manifest = shared._code_manifest
    shared.EXPECTED_CHECKPOINT_SHA256 = binding.checkpoint_sha256
    shared.EXPECTED_CONFIG_SHA256 = binding.effective_config_sha256
    shared._code_manifest = _code_manifest
    try:
        result = shared.run(args)
    finally:
        shared.EXPECTED_CHECKPOINT_SHA256 = original_checkpoint
        shared.EXPECTED_CONFIG_SHA256 = original_config
        shared._code_manifest = original_manifest
    result = dict(result)
    result.update(
        {
            "schema": "boxfusion.ca1m_tr3d_terminal_observer_run.v3",
            "collection_config_sha256": sha256_file(args.collection_config),
            "checkpoint_binding_sha256": binding.manifest_sha256,
            "ca_native_checkpoint": True,
        }
    )
    return result


def main() -> int:
    result = run(parser().parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

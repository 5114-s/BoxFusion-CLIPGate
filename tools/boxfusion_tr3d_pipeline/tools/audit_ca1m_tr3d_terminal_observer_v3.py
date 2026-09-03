#!/usr/bin/env python3
"""Independently audit v3 CA-native exact100 terminal observer caches."""

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
from tools import audit_ca1m_tr3d_terminal_observer as shared_audit  # noqa: E402
from tools import run_ca1m_tr3d_terminal_observer_v3 as runner_v3  # noqa: E402
from tools.preflight_ca1m_tr3d_terminal_train100_v3 import (  # noqa: E402
    validate_config,
)


def parser() -> argparse.ArgumentParser:
    value = shared_audit.parser()
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.tr3d_config is not None or args.tr3d_checkpoint is not None:
        raise ValueError("v3 audit forbids raw TR3D config/checkpoint arguments")
    binding = load_checkpoint_binding(args.tr3d_binding_manifest)
    preflight = validate_config(args.collection_config, args.tr3d_binding_manifest)
    if not preflight["ready_for_gpu"] or preflight["scene_count"] != 100:
        raise ValueError("v3 runtime preflight did not authorize exact100 audit")
    args.collection_config = Path(args.collection_config).resolve()
    args.tr3d_binding_manifest = binding.manifest_path
    args.tr3d_config = binding.effective_config_path
    args.tr3d_checkpoint = binding.checkpoint_path
    requested_output = args.output
    args.output = None
    original_checkpoint = shared_audit.EXPECTED_CHECKPOINT_SHA256
    original_config = shared_audit.EXPECTED_CONFIG_SHA256
    original_sources = shared_audit._code_sources
    shared_audit.EXPECTED_CHECKPOINT_SHA256 = binding.checkpoint_sha256
    shared_audit.EXPECTED_CONFIG_SHA256 = binding.effective_config_sha256
    shared_audit._code_sources = runner_v3._code_sources
    try:
        result = shared_audit.run(args)
    finally:
        shared_audit.EXPECTED_CHECKPOINT_SHA256 = original_checkpoint
        shared_audit.EXPECTED_CONFIG_SHA256 = original_config
        shared_audit._code_sources = original_sources
        args.output = requested_output
    result = dict(result)
    result.update(
        {
            "collection_schema": "boxfusion.ca1m_tr3d_terminal_collection.v3",
            "collection_config_sha256": sha256_file(args.collection_config),
            "checkpoint_binding_sha256": binding.manifest_sha256,
            "ca_native_checkpoint": True,
            "old_terminal_artifact_reuse": False,
        }
    )
    if requested_output is not None:
        shared_audit._write_json_create_only(requested_output, result)
    return result


def main() -> int:
    print(json.dumps(run(parser().parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only GT/GPU-free static preflight for the unsealed L6 v3 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_e961_incremental_l6_v3 import (
    NAMESPACE, PENDING_SCHEMA, R4_PROTOCOL_SCHEMA, R4_PROTOCOL_SHA256,
    R6_PREREGISTRATION_SCHEMA, R6_PREREGISTRATION_SHA256,
    V2_INVALID_SCHEMA, V2_INVALID_SHA256, sha256_file,
)


DEFAULT_CONFIG = ROOT / "config/ca1m_e961_incremental_l6_v3_pending.json"
CONFIG_SHA256 = "c28747da6b0a35e36945c32b9e5dd6bc18226eb03a91bd77b0e12cfba8b87755"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_json(record: Any, name: str, *, schema: str, sha256: str) -> None:
    value = _mapping(record, name)
    if set(value) != {"path", "schema", "sha256"}:
        raise ValueError(f"{name} fields differ")
    if value["schema"] != schema or value["sha256"] != sha256:
        raise ValueError(f"{name} binding differs")
    path = Path(value["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise ValueError(f"{name} must be an immutable regular file")
    if sha256_file(path) != sha256:
        raise ValueError(f"{name} SHA256 differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ValueError(f"{name} schema differs")


def validate_static_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Validate only frozen static bytes; never open any dynamic blocker path."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("L6 v3 config must be a regular file")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("L6 v3 config bytes differ from the reviewed candidate")
    cfg = json.loads(raw.decode("utf-8"))
    if not isinstance(cfg, Mapping):
        raise ValueError("L6 v3 config must be an object")
    if (
        cfg["schema"] != PENDING_SCHEMA or cfg["namespace"] != NAMESPACE
        or cfg["status"] != "static_candidate_pending_independent_review_not_sealed"
        or cfg["dataset"] != "ca1m_train_only"
    ):
        raise ValueError("L6 v3 identity differs")
    if cfg["access"] != {
        "run_authorized": False,
        "ready_sealed": False,
        "static_protocol_sealed": False,
        "gpu_started": False,
        "ground_truth_access_at_static_stage": False,
        "fold1_path_or_loader_present": False,
        "official_validation_path_or_loader_present": False,
    }:
        raise ValueError("L6 v3 static access boundary differs")

    bindings = _mapping(cfg["static_bindings"], "static bindings")
    if set(bindings) != {"r4_terminal_protocol", "r6_terminal_inputs_preregistration"}:
        raise ValueError("L6 v3 static binding inventory differs")
    _exact_json(
        bindings["r4_terminal_protocol"], "R4 protocol",
        schema=R4_PROTOCOL_SCHEMA, sha256=R4_PROTOCOL_SHA256,
    )
    _exact_json(
        bindings["r6_terminal_inputs_preregistration"], "R6 preregistration",
        schema=R6_PREREGISTRATION_SCHEMA, sha256=R6_PREREGISTRATION_SHA256,
    )

    tombstone = _mapping(cfg["v2_tombstone"], "v2 tombstone")
    if tombstone != {
        "path": os.fspath(ROOT / "manifests/ca1m_e961_incremental_l6_v2_INVALID.json"),
        "schema": V2_INVALID_SCHEMA,
        "sha256": V2_INVALID_SHA256,
        "invalid": True,
        "v2_protocol_may_be_sealed": False,
    }:
        raise ValueError("L6 v2 tombstone binding differs")
    tombstone_path = Path(tombstone["path"])
    if tombstone_path.is_symlink() or not tombstone_path.is_file():
        raise ValueError("L6 v2 tombstone is unavailable")
    if sha256_file(tombstone_path) != V2_INVALID_SHA256:
        raise ValueError("L6 v2 tombstone SHA256 differs")
    tombstone_payload = json.loads(tombstone_path.read_text(encoding="utf-8"))
    if (
        tombstone_payload.get("schema") != V2_INVALID_SCHEMA
        or tombstone_payload.get("invalid") is not True
        or tombstone_payload.get("static_protocol_was_sealed") is not False
        or tombstone_payload.get("run_was_authorized") is not False
    ):
        raise ValueError("L6 v2 tombstone semantics differ")

    expected_dependencies = {
        "v3_core": (ROOT / "boxfusion/ca1m_e961_incremental_l6_v3.py", "340cc6c2f31ca87ae199ebec2c16475fb67e4e54ae313ba352c5f07d74dc6089"),
        "v3_provider": (ROOT / "boxfusion/ca1m_e961_incremental_provider_v3.py", "1106194292f8bf3bfe7b66a98f8bc1872426867e292ab944bf089b72cb24ded9"),
        "observer": (ROOT / "boxfusion/tr3d_incremental_online.py", "f2017f86187ab671df2bba9c3de0db82ad85092a7bf4cfba8690fa0dcef376f7"),
        "lightweight_stage6_observer": (ROOT / "boxfusion/tr3d_lightweight_fusion.py", "eb07218f2b6851704099c66cfff7382f88dca70153db6e974185168f972c162b"),
        "lightweight_depth_geometry": (ROOT / "boxfusion/tr3d_r2_geometry.py", "f617757e68480697a8485529efd241c4faf4b1a012230c8564924351f79728c7"),
        "lightweight_yaw_geometry": (ROOT / "boxfusion/tr3d_r4_smov_observer.py", "72d0fecdc3355327ff8c6cf47b26483b365dbe7b0efe36062f5d3731430d8464"),
        "ca_worker_client": (ROOT / "boxfusion/ca1m_tr3d_worker_client.py", "aad34038e2df45b8ac154196ed4bcd154b9eb225b2ff5a466068f84b835bdb6b"),
        "ca_worker": (ROOT / "tools/ca1m_tr3d_terminal_worker.py", "e01c8bcea1a00bcb30e2553787e50f8086fa2b73787e86b7d3cd94a039f770d0"),
        "ca_inference_contract": (ROOT / "boxfusion/ca1m_tr3d_inference_contract.py", "06068eee518a37bf091ecfed79f202e4f9a9dc9660ed06d59cb7a4231b167ced"),
        "point_inference_config": (Path("/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/config/tr3d/tr3d_ca1m_foreground_point_inference_xfit_r2.py"), "479f7e61eff9fd23fc086ebc2603e161caa876defe73c556a0e671a8fd35c052"),
        "native_visibility_observer": (ROOT / "boxfusion/ca1m_native_b6_observer.py", "e22965b5527d28369faa3848cbc2d92c4c927905ac29c4e605d338de73464280"),
        "native_feature_contract": (ROOT / "boxfusion/ca1m_native_b6_score.py", "6daea10fe05ad531245a3007839fafe40b380bf1ab201f9ff9d612ee2abb8750"),
        "r4_generic_gate_selection": (ROOT / "boxfusion/ca1m_tr3d_terminal_gate_v5.py", "818b3aa60e1706f8dc03fde6bb872d20e41f31b18e6df8c6dd4ee45ddc1e812d"),
        "scannet_feature_reference": (ROOT / "boxfusion/tr3d_incremental_gate.py", "96b6bcd7ac89b7e388c5336e97f4fa562aa4109498631065332201bcb48f390c"),
        "scannet_label_reference": (ROOT / "tools/build_tr3d_incremental_novelty_dataset.py", "598d260f471749ad31e1022dcbddd2cf136bde5f942eb7b84d08ab84da180c39"),
        "scannet_materializer_reference": (ROOT / "tools/materialize_tr3d_lightweight_active.py", "5772fbd961753310ea6a4c47b95466d3242aefb62d38bc2ac570660a5f3e3cc3"),
    }
    dependencies = _mapping(cfg["implementation_dependencies"], "dependencies")
    if set(dependencies) != set(expected_dependencies):
        raise ValueError("L6 v3 dependency inventory differs")
    for name, (expected_path, expected_sha) in expected_dependencies.items():
        record = _mapping(dependencies[name], f"{name} dependency")
        if record != {"path": os.fspath(expected_path), "sha256": expected_sha}:
            raise ValueError(f"{name} dependency binding differs")
        if expected_path.is_symlink() or not expected_path.is_file():
            raise ValueError(f"{name} dependency is unavailable")
        if sha256_file(expected_path) != expected_sha:
            raise ValueError(f"{name} dependency SHA256 differs")

    # Full reviewed-byte equality above closes every top-level and nested key/value.
    # These explicit checks document the most security-sensitive subtree semantics.
    states = cfg["terminal_upstream_states"]
    if (
        states["pass"]["status"] != "PASS_EXPLORATORY_FOLD0_DIAGNOSTIC_COMPLETE"
        or states["pass"]["l6_allowed"] is not True
        or states["scientific_stop"]["status"] != "STOP_FOLD234_OOF_GATE_FAIL"
        or states["scientific_stop"]["l6_allowed"] is not True
        or states["provenance_or_implementation_failure"]["l6_allowed"] is not False
    ):
        raise ValueError("L6 v3 terminal-state semantics differ")
    if cfg["candidate_universe"]["roles"] != {
        "inner_holdout2": {"train_folds": [3, 4], "output_fold": 2},
        "inner_holdout3": {"train_folds": [2, 4], "output_fold": 3},
        "inner_holdout4": {"train_folds": [2, 3], "output_fold": 4},
        "outer_dev": {"train_folds": [2, 3, 4], "output_fold": 0},
    }:
        raise ValueError("L6 v3 fold roles differ")
    return {
        "ok": True,
        "config_sha256": CONFIG_SHA256,
        "namespace": NAMESPACE,
        "static_protocol_sealed": False,
        "dynamic_ready": False,
        "run_authorized": False,
        "gt_read": False,
        "fold1_read": False,
        "official_validation_read": False,
        "gpu_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.fspath(DEFAULT_CONFIG))
    args = parser.parse_args()
    print(json.dumps(validate_static_config(args.config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

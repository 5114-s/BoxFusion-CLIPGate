from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import boxfusion.ca1m_tr3d_e961_terminal_inputs_v5_r3 as route


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "tools/preflight_ca1m_tr3d_e961_terminal_inputs_v5_r3.py"
RUNNER = ROOT / "tools/run_ca1m_tr3d_e961_terminal_inputs_v5_r3.py"
R2_CORE = ROOT / "boxfusion/ca1m_tr3d_e961_terminal_inputs_v5_r2.py"


def test_frozen_r2_is_unchanged_and_r3_binds_it() -> None:
    assert route.sha256_file(R2_CORE) == "e44114b18c79176a8c1ddb992e878d025c7ff523c3087e535b8fa00bcc8c3826"
    cfg = json.loads(route.DEFAULT_CONFIG.read_text())
    assert cfg["implementation"]["r2_execution_core"]["sha256"] == route.sha256_file(R2_CORE)


def _pending() -> dict:
    return json.loads(route.DEFAULT_CONFIG.read_text())


def _bound(schema: str, index: int) -> dict:
    return {"state": "bound", "path": f"/canonical/{index}.json", "sha256": f"{index:064x}", "schema": schema}


def _synthetic_ready() -> tuple[dict, dict]:
    pending = _pending()
    ready = copy.deepcopy(pending)
    for index, role in enumerate(route.ROLE_ORDER, 1):
        schema = route.OUTER_SCHEMA if role == "outer_dev" else route.INNER_SCHEMA
        ready["scene_contract"]["roles"][role]["source_success_receipt"] = _bound(schema, index)
    ready["continuation_receipt"] = _bound(route.CONTINUATION_SCHEMA, 5)
    ready["run_authorization"] = _bound(route.AUTH_SCHEMA, 6)
    return pending, ready


def test_ready_delta_allows_exactly_six_replacements() -> None:
    pending, ready = _synthetic_ready()
    bound = route._ready_delta(pending, ready)
    assert set(bound) == {*route.ROLE_ORDER, "continuation_receipt", "run_authorization"}


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["runtime"].__setitem__("formal_device", "cuda:1"),
        lambda value: value["access"].__setitem__("fold1_path_present", True),
        lambda value: value["outputs"].__setitem__("namespace_root", "/tmp/escape"),
        lambda value: value["scene_contract"].__setitem__("scene_count", 79),
    ],
)
def test_ready_delta_rejects_every_non_dynamic_change(mutator) -> None:
    pending, ready = _synthetic_ready()
    mutator(ready)
    with pytest.raises(PermissionError, match="outside the six"):
        route._ready_delta(pending, ready)


def test_pending_operational_stops_before_static_receipt_probe_or_output(monkeypatch) -> None:
    monkeypatch.setattr(route, "validate_preregistration", lambda *_: pytest.fail("prereg reached"))
    monkeypatch.setattr(route, "_host_target_probe", lambda *_: pytest.fail("host probe reached"))
    monkeypatch.setattr(route, "_claim_writer", lambda *_: pytest.fail("claim reached"))
    with pytest.raises(route.PendingOperationalInputs):
        route.validate_operational_ready(route.DEFAULT_CONFIG)


@pytest.mark.parametrize(
    ("program", "mode"),
    [(PREFLIGHT, "--operational"), (RUNNER, "--operational-preflight")],
)
def test_pending_cli_exit3_stdout_empty_and_no_namespace(program: Path, mode: str) -> None:
    namespace = Path(_pending()["outputs"]["namespace_root"])
    existed = namespace.exists()
    result = subprocess.run(
        [sys.executable, str(program), "--config", str(route.DEFAULT_CONFIG), mode],
        cwd=ROOT, text=False, capture_output=True, check=False,
    )
    assert result.returncode == 3
    assert result.stdout == b""
    assert json.loads(result.stderr)["status"] == "BLOCKED_PENDING"
    assert namespace.exists() is existed


def test_parent_chain_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"; real.mkdir()
    (real / "value").write_bytes(b"ok")
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        route.stable_bytes(tmp_path / "link/value", "symlinked parent")


def test_stable_read_rechecks_parent_identity(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "value"; path.write_bytes(b"stable")
    reached = {"value": False}
    original = route._verify_dir_chain
    def check(parent, expected, name):
        reached["value"] = True
        return original(parent, expected, name)
    monkeypatch.setattr(route, "_verify_dir_chain", check)
    assert route.stable_bytes(path, "stable") == b"stable"
    assert reached["value"] is True


def test_host_probe_hardlinks_fsyncs_and_cleans(tmp_path: Path) -> None:
    route._host_target_probe(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_host_probe_failure_cleans_only_dedicated_dir(tmp_path: Path, monkeypatch) -> None:
    keep = tmp_path / "keep"; keep.write_bytes(b"user")
    monkeypatch.setattr(route.os, "link", lambda *_a, **_k: (_ for _ in ()).throw(OSError("no links")))
    with pytest.raises(OSError, match="no links"):
        route._host_target_probe(tmp_path)
    assert keep.read_bytes() == b"user"
    assert [x.name for x in tmp_path.iterdir()] == ["keep"]


def test_create_only_is_fuse_compatible_and_never_chmods(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(os, "chmod", lambda *_a, **_k: pytest.fail("chmod reached"))
    monkeypatch.setattr(os, "fchmod", lambda *_a, **_k: pytest.fail("fchmod reached"))
    target = tmp_path / "nested/value.bin"
    route.write_bytes_exclusive(target, b"payload")
    assert target.read_bytes() == b"payload"
    assert target.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        route.write_bytes_exclusive(target, b"replace")


def test_receipt_path_rejects_legacy_and_noncanonical() -> None:
    record = _bound(route.LEGACY_INNER_SCHEMA, 1)
    with pytest.raises(ValueError, match="legacy"):
        route._receipt_path(record, Path("/canonical"), "inner_holdout2", route.INNER_SCHEMA)
    record = _bound(route.INNER_SCHEMA, 1)
    with pytest.raises(ValueError, match="canonical leaf"):
        route._receipt_path(record, Path("/canonical"), "inner_holdout2", route.INNER_SCHEMA)


def _readonly_json(path: Path, value: dict) -> Path:
    path.write_bytes(route.canonical_json(value)); path.chmod(0o444); return path


def test_eval_pass_requires_every_frozen_check(tmp_path: Path, monkeypatch) -> None:
    children = {}
    for key in ("preregistration", "checkpoint_binding", "proposal_collection", "evaluation_report"):
        child = _readonly_json(tmp_path / f"{key}.json", {"complete": True})
        children[key] = {"path": str(child), "sha256": route.sha256_file(child)}
    checks = {
        "proposal_exact20_finite_ca_only": True,
        "same_gt_gain_ge_0_05_replacements_ge_10": True,
        "same_gt_gain_ge_0_05_scenes_ge_5": True,
        "oracle_delta_ap15_nonnegative": True,
        "oracle_delta_ap25_nonnegative": True,
        "oracle_delta_ap50_at_least_0_005": True,
    }
    continuation = _readonly_json(tmp_path / "CONTINUATION.json", {
        "schema": route.CONTINUATION_SCHEMA, "complete": True, "create_only": True,
        "pass": True, "authorized_roles": list(route.ROLE_ORDER[1:]), "scene_count": 20,
        "checkpoint_selection": False, "fold1_access": False, "official_validation_access": False,
        "continuation_gate": {"pass": True, "authorized_inner_roles": list(route.ROLE_ORDER[1:]), "checks": checks},
        **children,
    })
    monkeypatch.setattr(route, "CONTINUATION_CANONICAL_PATH", continuation)
    record = {"path": str(continuation), "sha256": route.sha256_file(continuation), "schema": route.CONTINUATION_SCHEMA}
    assert route._verify_continuation(record)[0] == continuation
    bad = json.loads(continuation.read_text()); bad["continuation_gate"]["checks"]["oracle_delta_ap50_at_least_0_005"] = False
    continuation.chmod(0o644); continuation.write_bytes(route.canonical_json(bad)); continuation.chmod(0o444)
    record["sha256"] = route.sha256_file(continuation)
    with pytest.raises(PermissionError, match="not an exact PASS"):
        route._verify_continuation(record)


def test_r2_stage_p_is_real_delegation(monkeypatch) -> None:
    called = {}
    fake_r2 = SimpleNamespace(
        write_bytes_exclusive=object(), ensure_directory=object(),
        run_stage_p=lambda ctx, role, device, **kwargs: called.update(ctx=ctx, role=role, device=device) or {"ok": True},
    )
    fake_ctx = SimpleNamespace(
        config_path=route.READY_CONFIG_PATH, authorization_path=Path("/unused"),
        authorization_sha256="a" * 64, config={}, r2_context="R2_CTX",
    )
    monkeypatch.setattr(route, "_pre_operation_guard", lambda *_: None)
    monkeypatch.setattr(route, "_import_r2", lambda *_: fake_r2)
    monkeypatch.delitem(sys.modules, "_r3_frozen_r2_execution", raising=False)
    assert route.run_stage_p(fake_ctx, "outer_dev") == {"ok": True}
    assert called == {"ctx": "R2_CTX", "role": "outer_dev", "device": "cuda:0"}


def test_static_preregistration_binds_exact100_inventory_and_complete_code() -> None:
    report = route.validate_static_config()
    assert report["status"] == "PASS_STATIC_PENDING"
    prereg = json.loads(route.PREREGISTRATION_PATH.read_text())
    inventory = prereg["processed_point_inventory"]
    assert inventory["scene_count"] == 100
    assert inventory["exact_visible_scene_count"] == 100
    assert len(inventory["scenes"]) == 100
    assert inventory["ground_truth_access"] is False
    assert set(prereg["implementation"]) == set(_pending()["implementation"])
    assert prereg["canonical_dynamic_paths"]["continuation_receipt"] == str(
        ROOT / "reports/ca1m_tr3d_e961_outer_dev_eval_v1/CONTINUATION_RECEIPT.json"
    )
    invalid = json.loads(route.V1_INVALID_PATH.read_text())
    assert invalid["invalid"] is True
    assert invalid["predecessor"]["sha256"] == route.V1_PREREGISTRATION_SHA256


def test_source_has_no_ground_truth_or_old_terminal_loader() -> None:
    text = Path(route.__file__).read_text()
    assert "load_ground_truth" not in text
    assert "terminal_inputs_v5.py" not in text
    assert "terminal_inputs_v5_r2.py" not in text  # loaded only by SHA-bound config record
    cfg = _pending()
    assert cfg["access"]["ground_truth_access"] is False
    assert cfg["access"]["fold1_path_present"] is False
    assert cfg["access"]["official_validation_path_present"] is False

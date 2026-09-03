from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import boxfusion.ca1m_tr3d_e961_terminal_inputs_v5_r4 as route


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "tools/preflight_ca1m_tr3d_e961_terminal_inputs_v5_r4.py"
RUNNER = ROOT / "tools/run_ca1m_tr3d_e961_terminal_inputs_v5_r4.py"


def test_real_bound_outer_inner_verifier_import_from_pipeline_cwd() -> None:
    cfg = json.loads(route.DEFAULT_CONFIG.read_text())
    _, base = route._load_base(cfg)
    old_path = list(sys.path)
    alias = "tr3d_ca1m_e961_outer_train_r2"
    fake = object(); existed = alias in sys.modules; previous = sys.modules.get(alias)
    sys.modules[alias] = fake
    try:
        outer, inner = route.load_bound_producer_verifiers(base)
        assert inner.outer is outer
        assert callable(outer.verify_success_receipt)
        assert callable(inner.verify_success_receipt)
        assert sys.modules[alias] is fake
        assert list(sys.path) == old_path
        assert set(route.ROLE_ORDER[1:]) == {"inner_holdout2", "inner_holdout3", "inner_holdout4"}
    finally:
        if existed: sys.modules[alias] = previous
        else: sys.modules.pop(alias, None)


def _make_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> route.ReadyContext:
    parent = tmp_path / "host"; parent.mkdir()
    outputs = {
        "namespace_root": str(parent / route.NAMESPACE),
        "proposal_root": str(parent / route.NAMESPACE / "P_anchor_free"),
        "overlay_root": str(parent / route.NAMESPACE / "O_oof_overlay"),
        "candidate_diagnostic_root": str(parent / route.NAMESPACE / "E_candidate_native"),
        "evidence_root": str(parent / route.NAMESPACE / "evidence"),
        "receipt_root": str(parent / route.NAMESPACE / "normalized_receipts"),
        "manifest_root": str(parent / route.NAMESPACE / "manifests"),
        "combined_manifest": str(parent / route.NAMESPACE / "manifests/CANDIDATE_COLLECTION_EXACT80.json"),
    }
    ready = {"outputs": copy.deepcopy(outputs)}
    r2_config = {"outputs": copy.deepcopy(outputs), "frozen": True}
    monkeypatch.setattr(route, "_expected_outputs", lambda: copy.deepcopy(outputs))
    monkeypatch.setattr(route, "_r2_config", lambda _ready: copy.deepcopy(r2_config))
    ready_path = tmp_path / "READY_CONFIG.json"; auth_path = tmp_path / "RUN_AUTHORIZATION.json"; bundle_path = tmp_path / "AUTHORIZATION_BUNDLE.json"
    ready_bytes = route.canonical_json(ready); auth_bytes = route.canonical_json({"schema": route.AUTH_SCHEMA}); bundle_bytes = route.canonical_json({"schema": route.BUNDLE_SCHEMA, "commit_id": "c" * 64})
    for path, data in ((ready_path, ready_bytes), (auth_path, auth_bytes), (bundle_path, bundle_bytes)):
        route.r3.write_bytes_exclusive(path, data)
    monkeypatch.setattr(route, "RUN_AUTHORIZATION_PATH", auth_path)
    monkeypatch.setattr(route, "AUTHORIZATION_BUNDLE_PATH", bundle_path)
    claim_path = parent / ".claim"
    writer_fd = route.r3._claim_writer(parent, claim_path.name, route.sha256_bytes(bundle_bytes))
    parent_fd, chain = route.r3._open_dir_chain(parent, "test parent")
    parent_stat = os.fstat(parent_fd)
    roles = {
        role: SimpleNamespace(
            train_folds=route.ROLE_SPECS[role][0], heldout_fold=route.ROLE_SPECS[role][1],
            scenes=(f"{index:08d}",), receipt_path=tmp_path / f"{role}.receipt",
            receipt_sha256=f"{index + 1:064x}", checkpoint_path=tmp_path / "iter_11268.pth",
            checkpoint_sha256=f"{index + 11:064x}",
        ) for index, role in enumerate(route.ROLE_ORDER)
    }
    snapshot = route.R2ContextSnapshot(
        ready_path, copy.deepcopy(r2_config), bundle_path, route.sha256_bytes(bundle_bytes),
        tmp_path / "continuation", "d" * 64, roles,
    )
    r2_path = ROOT / "boxfusion/ca1m_tr3d_e961_terminal_inputs_v5_r2.py"
    return route.ReadyContext(
        ready_path, ready, ready_bytes, route.sha256_bytes(ready_bytes),
        {"schema": route.AUTH_SCHEMA}, auth_bytes, route.sha256_bytes(auth_bytes),
        {"schema": route.BUNDLE_SCHEMA, "commit_id": "c" * 64}, bundle_bytes,
        route.sha256_bytes(bundle_bytes), snapshot.continuation_path,
        snapshot.continuation_sha256, roles, route._roles_binding(roles), snapshot, route.canonical_json(r2_config),
        route.sha256_bytes(route.canonical_json(r2_config)), r2_path,
        route.sha256_file(r2_path), route._identity(r2_path), writer_fd, claim_path,
        route._identity(claim_path), parent_fd, parent,
        (parent_stat.st_dev, parent_stat.st_ino), tuple(chain),
    )


def _assert_stage_guard_rejects(ctx: route.ReadyContext, monkeypatch: pytest.MonkeyPatch, match: str) -> None:
    monkeypatch.setattr(route, "_fresh_r2_module", lambda *_: pytest.fail("R2 stage/module reached"))
    with pytest.raises(PermissionError, match=match):
        route.run_stage_p(ctx, "outer_dev")


def test_closed_writer_fd_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _make_context(tmp_path, monkeypatch); os.close(ctx.writer_fd); ctx.writer_fd = -1
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "descriptor is closed")
    finally: ctx.close()


def test_writer_claim_inode_swap_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _make_context(tmp_path, monkeypatch)
    moved = ctx.writer_claim_path.with_suffix(".old"); ctx.writer_claim_path.rename(moved)
    ctx.writer_claim_path.write_bytes(moved.read_bytes())
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "canonical writer claim")
    finally: ctx.close()


def test_unlocked_writer_fd_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _make_context(tmp_path, monkeypatch); fcntl.flock(ctx.writer_fd, fcntl.LOCK_UN)
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "no longer holds")
    finally: ctx.close()


def test_closed_parent_fd_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _make_context(tmp_path, monkeypatch); os.close(ctx.parent_fd); ctx.parent_fd = -1
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "descriptor is closed")
    finally: ctx.close()


def test_parent_path_swap_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _make_context(tmp_path, monkeypatch)
    old = ctx.parent_path.with_suffix(".old"); ctx.parent_path.rename(old); ctx.parent_path.mkdir()
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "canonical writer claim|output-parent")
    finally: ctx.close()


def test_r2_context_config_diversion_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _make_context(tmp_path, monkeypatch); ctx.r2_context.config["diverted"] = True
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "r2_context config")
    finally: ctx.close()


def test_r2_context_output_diversion_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _make_context(tmp_path, monkeypatch); ctx.r2_context.config["outputs"]["namespace_root"] = str(tmp_path / "escape")
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "r2_context config|output namespace")
    finally: ctx.close()


def test_role_snapshot_diversion_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _make_context(tmp_path, monkeypatch)
    ctx.roles["outer_dev"] = copy.copy(ctx.roles["outer_dev"])
    ctx.roles["outer_dev"].receipt_sha256 = "f" * 64
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "role snapshot")
    finally: ctx.close()


def test_sys_modules_fake_r2_injection_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _make_context(tmp_path, monkeypatch)
    monkeypatch.setattr(route.secrets, "token_hex", lambda *_: "feedface")
    name = f"boxfusion._r4_frozen_r2_{ctx.r2_module_sha256[:12]}_feedface"
    sys.modules[name] = SimpleNamespace(run_stage_p=lambda *_a, **_k: pytest.fail("fake executed"))
    try:
        with pytest.raises(RuntimeError, match="collision"):
            route.run_stage_p(ctx, "outer_dev")
    finally:
        sys.modules.pop(name, None); ctx.close()


def test_fresh_r2_load_ignores_generic_fake_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _make_context(tmp_path, monkeypatch)
    sys.modules["_r3_frozen_r2_execution"] = SimpleNamespace(run_stage_p=lambda *_: None)
    try:
        module = route._fresh_r2_module(ctx)
        assert Path(module.__file__).resolve() == ctx.r2_module_path.resolve()
        assert module.run_stage_p.__module__ == module.__name__
    finally:
        sys.modules.pop("_r3_frozen_r2_execution", None); ctx.close()


def test_post_call_guard_rejects_stage_time_context_diversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _make_context(tmp_path, monkeypatch)
    module_name = "boxfusion._r4_fault_injection"

    def divert(r2_context: route.R2ContextSnapshot, _role: str, **_kwargs: object) -> dict[str, bool]:
        r2_context.config["diverted_during_stage"] = True
        return {"stage_returned": True}

    divert.__module__ = module_name
    fake_module = SimpleNamespace(__name__=module_name, run_stage_p=divert)
    monkeypatch.setattr(route, "_fresh_r2_module", lambda _ctx: fake_module)
    try:
        with pytest.raises(PermissionError, match="r2_context config"):
            route.run_stage_p(ctx, "outer_dev")
    finally:
        ctx.close()


def test_bundle_missing_or_mismatched_never_commits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_path = tmp_path / "READY_CONFIG.json"; auth_path = tmp_path / "RUN_AUTHORIZATION.json"; bundle_path = tmp_path / "AUTHORIZATION_BUNDLE.json"
    commit = "a" * 64
    ready = {"run_authorization": {"state": "committed_by_bundle", "path": str(bundle_path), "commit_id": commit, "schema": route.BUNDLE_SCHEMA}}
    ready_bytes = route.canonical_json(ready); auth_bytes = route.canonical_json({"schema": route.AUTH_SCHEMA, "commit_id": commit})
    route.r3.write_bytes_exclusive(ready_path, ready_bytes); route.r3.write_bytes_exclusive(auth_path, auth_bytes)
    monkeypatch.setattr(route, "READY_CONFIG_PATH", ready_path); monkeypatch.setattr(route, "RUN_AUTHORIZATION_PATH", auth_path); monkeypatch.setattr(route, "AUTHORIZATION_BUNDLE_PATH", bundle_path)
    with pytest.raises(route.PendingOperationalInputs): route._load_committed_ready(ready_path)
    bundle = route._bundle_payload(ready_bytes, auth_bytes, commit); bundle["commit_id"] = "b" * 64
    route.r3.write_bytes_exclusive(bundle_path, route.canonical_json(bundle))
    with pytest.raises(PermissionError, match="bundle/leaf"):
        route._load_committed_ready(ready_path)


def test_crash_replay_fills_exact_leaves_then_publishes_bundle_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_path = tmp_path / "READY_CONFIG.json"; auth_path = tmp_path / "RUN_AUTHORIZATION.json"; bundle_path = tmp_path / "AUTHORIZATION_BUNDLE.json"
    monkeypatch.setattr(route, "READY_CONFIG_PATH", ready_path); monkeypatch.setattr(route, "RUN_AUTHORIZATION_PATH", auth_path); monkeypatch.setattr(route, "AUTHORIZATION_BUNDLE_PATH", bundle_path)
    commit = "c" * 64
    ready = {"run_authorization": {"state": "committed_by_bundle", "path": str(bundle_path), "commit_id": commit, "schema": route.BUNDLE_SCHEMA}}
    ready_bytes = route.canonical_json(ready)
    auth_bytes = route.canonical_json({"schema": route.AUTH_SCHEMA, "commit_id": commit})
    bundle_bytes = route.canonical_json(route._bundle_payload(ready_bytes, auth_bytes, commit))

    # Simulate a crash after the first leaf: it is not an operational commit.
    route.r3.create_or_verify(ready_path, ready_bytes, "R4 ready leaf")
    with pytest.raises(route.PendingOperationalInputs):
        route._load_committed_ready(ready_path)

    # Exact replay verifies the existing leaf, fills the second leaf, and only
    # then publishes the sole operational gate.
    route.r3.create_or_verify(ready_path, ready_bytes, "R4 ready leaf")
    route.r3.create_or_verify(auth_path, auth_bytes, "R4 authorization leaf")
    with pytest.raises(route.PendingOperationalInputs):
        route._load_committed_ready(ready_path)
    route.r3.create_or_verify(bundle_path, bundle_bytes, "R4 authorization bundle")
    loaded = route._load_committed_ready(ready_path)
    assert loaded[1] == ready_bytes and loaded[3] == auth_bytes and loaded[5] == bundle_bytes

    # A conflicting replay never replaces a published leaf or bundle.
    with pytest.raises(ValueError, match="resume bytes differ"):
        route.r3.create_or_verify(auth_path, auth_bytes + b" ", "R4 authorization leaf")
    assert auth_path.read_bytes() == auth_bytes and bundle_path.read_bytes() == bundle_bytes


@pytest.mark.parametrize("program,mode", [(PREFLIGHT,"--operational"),(RUNNER,"--operational-preflight")])
def test_pending_cli_stdout_empty_exit3_no_namespace(program: Path, mode: str) -> None:
    result=subprocess.run([sys.executable,str(program),"--config",str(route.DEFAULT_CONFIG),mode],cwd=ROOT,capture_output=True)
    assert result.returncode==3 and result.stdout==b""
    assert json.loads(result.stderr)["status"]=="BLOCKED_PENDING"
    assert not Path("/extra/ZhaoX") .joinpath(route.NAMESPACE).exists()


def test_static_preregistration_binds_r3_invalid_and_r4_invariants() -> None:
    report=route.validate_static_config(); assert report["status"]=="PASS_STATIC_PENDING"
    prereg=json.loads(route.PREREGISTRATION_PATH.read_text()); assert prereg["r3_invalidation"]["sha256"]==route.sha256_file(route.R3_INVALID_PATH)
    relationship = prereg["invalid_r3_dependency_relationship"]
    assert relationship["predecessor_preregistration_invalid"] is True
    assert relationship["r4_rehashes_every_canonical_dependency"] is True
    assert relationship["changed_dependencies"] == [{
        "name": "v5_manifest_runtime",
        "path": str(ROOT / "boxfusion/ca1m_tr3d_terminal_gate_v5.py"),
        "invalid_r3_sha256": "a1128faaf2eeb253838bdf89faea4158ae686b14f22f8e728a27f19ec9972e6a",
        "r4_preregistered_sha256": "818b3aa60e1706f8dc03fde6bb872d20e41f31b18e6df8c6dd4ee45ddc1e812d",
    }]
    assert all(prereg["runtime_invariants"].values())

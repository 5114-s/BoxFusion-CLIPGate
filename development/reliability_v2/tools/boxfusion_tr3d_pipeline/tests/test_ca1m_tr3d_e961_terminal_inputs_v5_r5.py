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

import boxfusion.ca1m_tr3d_e961_terminal_inputs_v5_r5 as route


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "tools/preflight_ca1m_tr3d_e961_terminal_inputs_v5_r5.py"
RUNNER = ROOT / "tools/run_ca1m_tr3d_e961_terminal_inputs_v5_r5.py"
R4_RUNNER = ROOT / "tools/run_ca1m_tr3d_e961_terminal_inputs_v5_r4.py"


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
    finally:
        if existed: sys.modules[alias] = previous
        else: sys.modules.pop(alias, None)


def test_invalidated_r4_runner_is_tombstoned_and_bound_core_fails_closed() -> None:
    result = subprocess.run(
        [sys.executable, str(R4_RUNNER), "--run-all"], cwd=ROOT, capture_output=True,
    )
    assert result.returncode == 66 and result.stdout == b""
    assert json.loads(result.stderr)["status"] == "INVALIDATED_R4_CODE_BLOCK"
    with pytest.raises((ValueError, PermissionError), match="implementation current_runner|SHA"):
        route.r4.validate_static_config()


def _make_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[route.ReadyContext, dict[str, object]]:
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
    ready_path = tmp_path / "READY_CONFIG.json"
    auth_path = tmp_path / "RUN_AUTHORIZATION.json"
    bundle_path = tmp_path / "AUTHORIZATION_BUNDLE.json"
    prereg_path = tmp_path / "PREREGISTRATION.json"
    r4_invalid_path = tmp_path / "R4_INVALID.json"
    config_path = tmp_path / "pending.json"
    continuation_path = tmp_path / "continuation.json"
    dependency_path = tmp_path / "scientific_dependency.py"
    continuation_path.write_text("continuation\n")
    dependency_path.write_text("VALUE = 1\n")
    config_path.write_text("{}\n")

    monkeypatch.setattr(route, "READY_CONFIG_PATH", ready_path)
    monkeypatch.setattr(route, "RUN_AUTHORIZATION_PATH", auth_path)
    monkeypatch.setattr(route, "AUTHORIZATION_BUNDLE_PATH", bundle_path)
    monkeypatch.setattr(route, "PREREGISTRATION_PATH", prereg_path)
    monkeypatch.setattr(route, "R4_INVALID_PATH", r4_invalid_path)
    monkeypatch.setattr(route, "DEFAULT_CONFIG", config_path)
    monkeypatch.setattr(route, "OUTPUT_PARENT_PATH", parent)
    monkeypatch.setattr(route, "WRITER_CLAIM_PATH", parent / f".{route.NAMESPACE}.writer.claim")
    monkeypatch.setattr(route, "_expected_outputs", lambda: copy.deepcopy(outputs))

    r2_path = ROOT / "boxfusion/ca1m_tr3d_e961_terminal_inputs_v5_r2.py"
    implementation = {
        "r2_execution_core": {"path": str(r2_path), "sha256": route.sha256_file(r2_path)},
    }
    pending = {
        "schema": route.CONFIG_SCHEMA, "namespace": route.NAMESPACE,
        "producer_success_receipts": {
            role: route._pending(route.OUTER_SCHEMA if role == "outer_dev" else route.INNER_SCHEMA)
            for role in route.ROLE_ORDER
        },
        "continuation_receipt": route._pending(route.CONTINUATION_SCHEMA),
        "run_authorization": route._pending(route.BUNDLE_SCHEMA, bundle=True),
        "outputs": copy.deepcopy(outputs), "implementation": copy.deepcopy(implementation),
    }
    monkeypatch.setattr(route, "_load_pending", lambda: (config_path, copy.deepcopy(pending)))
    r2_config = {"outputs": copy.deepcopy(outputs), "frozen": True}
    monkeypatch.setattr(route, "_r2_config", lambda _ready: copy.deepcopy(r2_config))

    roles = {
        role: SimpleNamespace(
            train_folds=route.ROLE_SPECS[role][0], heldout_fold=route.ROLE_SPECS[role][1],
            scenes=(f"{index:08d}",), receipt_path=tmp_path / f"{role}.receipt",
            receipt_sha256=f"{index + 1:064x}", checkpoint_path=tmp_path / "iter_11268.pth",
            checkpoint_sha256=f"{index + 11:064x}",
        ) for index, role in enumerate(route.ROLE_ORDER)
    }
    state: dict[str, object] = {"roles": roles, "continuation": continuation_path}
    monkeypatch.setattr(
        route, "_verify_dynamic", lambda _ready: (state["continuation"], state["roles"]),
    )

    route.r3.write_bytes_exclusive(r4_invalid_path, route.canonical_json({
        "schema": route.R4_INVALID_SCHEMA, "invalid": True,
    }))
    preregistration = {
        "schema": route.PREREGISTRATION_SCHEMA, "complete": True,
        "create_only": True, "static_only": True, "namespace": route.NAMESPACE,
        "pending_config": {
            "path": str(config_path), "sha256": route.sha256_file(config_path),
            "schema": route.CONFIG_SCHEMA,
        },
        "implementation": copy.deepcopy(implementation),
        "base_execution_dependencies": {
            **copy.deepcopy(implementation),
            "scientific_dependency": {
                "path": str(dependency_path), "sha256": route.sha256_file(dependency_path),
            },
        },
        "r4_invalidation": {
            "path": str(r4_invalid_path), "sha256": route.sha256_file(r4_invalid_path),
            "schema": route.R4_INVALID_SCHEMA,
        },
    }
    route.r3.write_bytes_exclusive(prereg_path, route.canonical_json(preregistration))

    commit = "c" * 64
    ready = copy.deepcopy(pending)
    for index, role in enumerate(route.ROLE_ORDER):
        ready["producer_success_receipts"][role] = {
            "state": "bound", "path": str(tmp_path / f"{role}.receipt"),
            "sha256": f"{index + 1:064x}",
            "schema": route.OUTER_SCHEMA if role == "outer_dev" else route.INNER_SCHEMA,
        }
    ready["continuation_receipt"] = {
        "state": "bound", "path": str(continuation_path),
        "sha256": route.sha256_file(continuation_path), "schema": route.CONTINUATION_SCHEMA,
    }
    ready["run_authorization"] = {
        "state": "committed_by_bundle", "path": str(bundle_path),
        "commit_id": commit, "schema": route.BUNDLE_SCHEMA,
    }
    auth = route._authorization_payload(ready, continuation_path, roles, commit)
    ready_bytes = route.canonical_json(ready)
    auth_bytes = route.canonical_json(auth)
    bundle = route._bundle_payload(ready_bytes, auth_bytes, commit)
    bundle_bytes = route.canonical_json(bundle)
    for path, data in ((ready_path, ready_bytes), (auth_path, auth_bytes), (bundle_path, bundle_bytes)):
        route.r3.write_bytes_exclusive(path, data)

    derived = route._derive_canonical_payloads()
    parent_fd, parent_chain = route.r3._open_dir_chain(parent, "test fixed parent")
    parent_stat = os.fstat(parent_fd)
    writer_fd = route._claim_runtime_writer(derived.bundle_sha256)
    token = "test-" + os.urandom(16).hex()
    route._RUNTIME_AUTHORITIES[token] = route.CanonicalAuthority(
        derived.ready_bytes, derived.ready_sha256,
        derived.authorization_bytes, derived.authorization_sha256,
        derived.bundle_bytes, derived.bundle_sha256,
        derived.preregistration_bytes, derived.preregistration_sha256,
        route._identity(route.WRITER_CLAIM_PATH),
        (parent_stat.st_dev, parent_stat.st_ino), tuple(parent_chain),
    )
    ctx = route.ReadyContext(token, writer_fd, parent_fd)
    route._guard_context(ctx)
    state.update({
        "pending": pending, "ready": ready, "derived": derived,
        "dependency_path": dependency_path,
    })
    return ctx, state


def _assert_stage_guard_rejects(ctx: object, monkeypatch: pytest.MonkeyPatch, match: str) -> None:
    monkeypatch.setattr(route, "_fresh_r2_module", lambda *_: pytest.fail("R2 stage/module reached"))
    with pytest.raises(PermissionError, match=match):
        route.run_stage_p(ctx, "outer_dev")


def test_closed_writer_fd_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _ = _make_context(tmp_path, monkeypatch); os.close(ctx.writer_fd); object.__setattr__(ctx, "writer_fd", -1)
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "descriptor is closed")
    finally: ctx.close()


def test_writer_claim_inode_swap_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _ = _make_context(tmp_path, monkeypatch)
    moved = route.WRITER_CLAIM_PATH.with_suffix(".old"); route.WRITER_CLAIM_PATH.rename(moved)
    route.WRITER_CLAIM_PATH.write_bytes(moved.read_bytes())
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "fixed writer claim")
    finally: ctx.close()


def test_unlocked_writer_fd_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _ = _make_context(tmp_path, monkeypatch); fcntl.flock(ctx.writer_fd, fcntl.LOCK_UN)
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "no longer holds")
    finally: ctx.close()


def test_same_inode_non_lock_owning_fd_swap_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _ = _make_context(tmp_path, monkeypatch)
    original = ctx.writer_fd
    replacement = os.open(route.WRITER_CLAIM_PATH, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    object.__setattr__(ctx, "writer_fd", replacement)
    try:
        _assert_stage_guard_rejects(ctx, monkeypatch, "lock-owning open description")
    finally:
        object.__setattr__(ctx, "writer_fd", original); os.close(replacement); ctx.close()


def test_parent_path_swap_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _ = _make_context(tmp_path, monkeypatch)
    old = route.OUTPUT_PARENT_PATH.with_suffix(".old"); route.OUTPUT_PARENT_PATH.rename(old); route.OUTPUT_PARENT_PATH.mkdir()
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "fixed writer claim|fixed output-parent")
    finally: ctx.close()


def test_whole_parent_writer_fd_group_swap_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _ = _make_context(tmp_path, monkeypatch)
    original_writer, original_parent = ctx.writer_fd, ctx.parent_fd
    alternate = tmp_path / "alternate"; alternate.mkdir()
    alternate_parent, _ = route.r3._open_dir_chain(alternate, "alternate parent")
    alternate_writer = route.r3._claim_writer(alternate, ".alternate.claim", "f" * 64)
    object.__setattr__(ctx, "writer_fd", alternate_writer); object.__setattr__(ctx, "parent_fd", alternate_parent)
    try:
        _assert_stage_guard_rejects(ctx, monkeypatch, "registered canonical claim|registered canonical parent")
    finally:
        object.__setattr__(ctx, "writer_fd", original_writer); object.__setattr__(ctx, "parent_fd", original_parent)
        os.close(alternate_writer); os.close(alternate_parent); ctx.close()


def _legacy_proxy(ctx: route.ReadyContext, **fields: object) -> SimpleNamespace:
    return SimpleNamespace(
        authority_token=ctx.authority_token, writer_fd=ctx.writer_fd,
        parent_fd=ctx.parent_fd, **fields,
    )


def test_tmp_r2_module_triple_replacement_is_not_a_context_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, _ = _make_context(tmp_path, monkeypatch)
    fake_module = tmp_path / "fake_r2.py"; fake_module.write_text("def run_stage_p(*a, **k):\n    raise AssertionError('BYPASS')\n")
    proxy = _legacy_proxy(
        ctx, r2_module_path=fake_module, r2_module_sha256=route.sha256_file(fake_module),
        r2_module_identity=route._identity(fake_module),
    )
    try: _assert_stage_guard_rejects(proxy, monkeypatch, "context type differs")
    finally: ctx.close()


def test_ready_config_and_r2_context_group_replacement_is_not_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, _ = _make_context(tmp_path, monkeypatch)
    alternate = tmp_path / "alternate_ready.json"; alternate.write_text('{"outputs":{}}\n')
    proxy = _legacy_proxy(
        ctx, config_path=alternate, ready={"outputs": {}}, ready_bytes=alternate.read_bytes(),
        ready_sha256=route.sha256_file(alternate), r2_config_bytes=b"{}\n",
        r2_config_sha256=route.sha256_bytes(b"{}\n"),
        r2_context=SimpleNamespace(config={"outputs": {}}),
    )
    try: _assert_stage_guard_rejects(proxy, monkeypatch, "context type differs")
    finally: ctx.close()


def test_roles_group_rederived_from_canonical_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, state = _make_context(tmp_path, monkeypatch)
    altered = copy.deepcopy(state["roles"])
    altered["outer_dev"].receipt_sha256 = "f" * 64
    state["roles"] = altered
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "authorization payload differs")
    finally: ctx.close()


def test_continuation_group_rederived_from_canonical_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, state = _make_context(tmp_path, monkeypatch)
    alternate = tmp_path / "alternate_continuation.json"; alternate.write_text("alternate\n")
    state["continuation"] = alternate
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "authorization payload differs")
    finally: ctx.close()


def test_preregistered_dependency_rehashed_before_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, state = _make_context(tmp_path, monkeypatch)
    state["dependency_path"].write_text("VALUE = 2\n")
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "scientific_dependency|SHA")
    finally: ctx.close()


def test_unregistered_authority_token_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _ = _make_context(tmp_path, monkeypatch)
    original = ctx.authority_token; object.__setattr__(ctx, "authority_token", "forged-token")
    try: _assert_stage_guard_rejects(ctx, monkeypatch, "no registered canonical authority")
    finally:
        object.__setattr__(ctx, "authority_token", original); ctx.close()


def test_sys_modules_fake_r2_injection_rejected_before_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _ = _make_context(tmp_path, monkeypatch)
    derived = route._guard_context(ctx)
    monkeypatch.setattr(route.secrets, "token_hex", lambda *_: "feedface")
    name = f"boxfusion._r5_frozen_r2_{derived.r2_module_sha256[:12]}_feedface"
    sys.modules[name] = SimpleNamespace(run_stage_p=lambda *_a, **_k: pytest.fail("fake executed"))
    try:
        with pytest.raises(RuntimeError, match="collision"):
            route.run_stage_p(ctx, "outer_dev")
    finally:
        sys.modules.pop(name, None); ctx.close()


def test_post_call_guard_rederives_roles_before_return(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, state = _make_context(tmp_path, monkeypatch)
    module_name = "boxfusion._r5_fault_injection"

    def divert(_r2_context: route.R2ContextSnapshot, _role: str, **_kwargs: object) -> dict[str, bool]:
        altered = copy.deepcopy(state["roles"])
        altered["outer_dev"].checkpoint_sha256 = "e" * 64
        state["roles"] = altered
        return {"stage_returned": True}

    divert.__module__ = module_name
    fake = SimpleNamespace(__name__=module_name, run_stage_p=divert)
    monkeypatch.setattr(route, "_fresh_r2_module", lambda _authority: fake)
    try:
        with pytest.raises(PermissionError, match="authorization payload differs"):
            route.run_stage_p(ctx, "outer_dev")
    finally:
        ctx.close()


def test_bundle_missing_or_mismatched_never_commits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_path = tmp_path / "READY_CONFIG.json"; auth_path = tmp_path / "RUN_AUTHORIZATION.json"; bundle_path = tmp_path / "AUTHORIZATION_BUNDLE.json"
    monkeypatch.setattr(route, "READY_CONFIG_PATH", ready_path); monkeypatch.setattr(route, "RUN_AUTHORIZATION_PATH", auth_path); monkeypatch.setattr(route, "AUTHORIZATION_BUNDLE_PATH", bundle_path)
    commit = "a" * 64
    ready = {"run_authorization": {"state": "committed_by_bundle", "path": str(bundle_path), "commit_id": commit, "schema": route.BUNDLE_SCHEMA}}
    ready_bytes = route.canonical_json(ready); auth_bytes = route.canonical_json({"schema": route.AUTH_SCHEMA, "commit_id": commit})
    route.r3.write_bytes_exclusive(ready_path, ready_bytes); route.r3.write_bytes_exclusive(auth_path, auth_bytes)
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
    ready_bytes = route.canonical_json(ready); auth_bytes = route.canonical_json({"schema": route.AUTH_SCHEMA, "commit_id": commit})
    bundle_bytes = route.canonical_json(route._bundle_payload(ready_bytes, auth_bytes, commit))
    route.r3.create_or_verify(ready_path, ready_bytes, "R5 ready leaf")
    with pytest.raises(route.PendingOperationalInputs): route._load_committed_ready(ready_path)
    route.r3.create_or_verify(auth_path, auth_bytes, "R5 authorization leaf")
    with pytest.raises(route.PendingOperationalInputs): route._load_committed_ready(ready_path)
    route.r3.create_or_verify(bundle_path, bundle_bytes, "R5 authorization bundle")
    loaded = route._load_committed_ready(ready_path)
    assert loaded[1] == ready_bytes and loaded[3] == auth_bytes and loaded[5] == bundle_bytes


@pytest.mark.parametrize("program,mode", [(PREFLIGHT,"--operational"),(RUNNER,"--operational-preflight")])
def test_pending_cli_stdout_empty_exit3_no_namespace(program: Path, mode: str) -> None:
    result=subprocess.run([sys.executable,str(program),"--config",str(route.DEFAULT_CONFIG),mode],cwd=ROOT,capture_output=True)
    assert result.returncode==3 and result.stdout==b""
    assert json.loads(result.stderr)["status"]=="BLOCKED_PENDING"
    assert not Path("/extra/ZhaoX").joinpath(route.NAMESPACE).exists()


def test_static_preregistration_binds_r4_invalid_and_r5_invariants() -> None:
    report=route.validate_static_config(); assert report["status"]=="PASS_STATIC_PENDING"
    prereg=json.loads(route.PREREGISTRATION_PATH.read_text())
    assert prereg["r4_invalidation"]["sha256"]==route.sha256_file(route.R4_INVALID_PATH)
    assert prereg["inherited_base_dependency_relationship"]["r4_predecessor_preregistration_invalid"] is True
    assert all(prereg["runtime_invariants"].values())

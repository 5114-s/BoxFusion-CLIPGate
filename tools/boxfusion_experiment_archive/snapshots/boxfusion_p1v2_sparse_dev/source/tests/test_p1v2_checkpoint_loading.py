"""Strict checkpoint-loading contracts for the P1/P1R/P1S heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from boxfusion.p1_spatial_residual import (  # noqa: E402
    NativeSparseResidualProposalHead,
)
from boxfusion.residual_proposal import (  # noqa: E402
    P1_FEATURE_DIM,
    P1_FEATURE_NAMES,
    P1_HEAD_SCHEMA,
    P1R_HEAD_SCHEMA,
    P1S_HEAD_SCHEMA,
    ResidualProposalConfig,
    ResidualVoxelProposalHead,
    load_residual_proposal_head,
    sha256_file,
)


_B6_SHA256 = "a" * 64
_HIDDEN_DIM = 8


@dataclass(frozen=True)
class _CheckpointContract:
    stage: str
    schema: str
    architecture: str
    target_scope: str
    model_type: type[torch.nn.Module]
    legacy_metadata: bool = False


_CONTRACTS = (
    _CheckpointContract(
        stage="P1",
        schema=P1_HEAD_SCHEMA,
        architecture="per_voxel_mlp",
        target_scope="scene_global",
        model_type=ResidualVoxelProposalHead,
        legacy_metadata=True,
    ),
    _CheckpointContract(
        stage="P1R",
        schema=P1R_HEAD_SCHEMA,
        architecture="per_voxel_mlp",
        target_scope="snapshot_inside_only",
        model_type=ResidualVoxelProposalHead,
    ),
    _CheckpointContract(
        stage="P1S",
        schema=P1S_HEAD_SCHEMA,
        architecture="native_sparse_context_v1",
        target_scope="snapshot_inside_only",
        model_type=NativeSparseResidualProposalHead,
    ),
)
_CONTRACT_IDS = tuple(contract.stage for contract in _CONTRACTS)


def _new_head(contract: _CheckpointContract) -> torch.nn.Module:
    if contract.architecture == "native_sparse_context_v1":
        return NativeSparseResidualProposalHead(
            input_dim=P1_FEATURE_DIM,
            hidden_dim=_HIDDEN_DIM,
            regression_dim=6,
            dilations=(1, 2),
        )
    return ResidualVoxelProposalHead(
        input_dim=P1_FEATURE_DIM,
        hidden_dim=_HIDDEN_DIM,
        regression_dim=6,
    )


def _checkpoint_payload(
    contract: _CheckpointContract,
    *,
    schema: str | None = None,
    architecture: str | None = None,
    target_scope: str | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    source = _new_head(contract)
    source_state = {
        name: tensor.detach().clone()
        for name, tensor in source.state_dict().items()
    }
    if contract.architecture == "native_sparse_context_v1":
        model_config = dict(source.model_config())
    else:
        model_config = {
            "input_dim": P1_FEATURE_DIM,
            "hidden_dim": _HIDDEN_DIM,
            "regression_dim": 6,
            "regression_encoding": "center_delta_m_log_size_m",
        }
        # Legacy P1 predates the architecture discriminator. Its absence is
        # valid only because the P1 schema has one historical interpretation.
        if not contract.legacy_metadata:
            model_config["head_architecture"] = contract.architecture
    if architecture is not None:
        model_config["head_architecture"] = architecture

    training_config: dict[str, Any] = {
        "schema": "boxfusion.p1_residual_training.v1",
    }
    # The legacy P1 producer also predates the explicit target-scope field.
    if not contract.legacy_metadata:
        training_config["target_assignment_scope"] = contract.target_scope
    if target_scope is not None:
        training_config["target_assignment_scope"] = target_scope

    payload = {
        "schema": contract.schema if schema is None else schema,
        "feature_names": list(P1_FEATURE_NAMES),
        "model_config": model_config,
        "state_dict": source_state,
        "training_config": training_config,
        "provenance": {
            "train_scene_ids": ["scene0001_00", "scene0002_00"],
            "forbidden_overlap": [],
            "train_scene_list_sha256": "1" * 64,
            "forbidden_scene_list_sha256": "2" * 64,
            "b6_checkpoint_sha256": _B6_SHA256,
        },
    }
    return payload, source_state


def _save_checkpoint(
    tmp_path: Path,
    contract: _CheckpointContract,
    **updates: Any,
) -> tuple[Path, dict[str, torch.Tensor]]:
    payload, source_state = _checkpoint_payload(contract, **updates)
    path = tmp_path / f"{contract.stage.lower()}_head.pt"
    torch.save(payload, path)
    return path, source_state


def _runtime_config(
    checkpoint: Path, contract: _CheckpointContract
) -> ResidualProposalConfig:
    return ResidualProposalConfig(
        enabled=True,
        mode="infer",
        checkpoint=str(checkpoint),
        hidden_dim=_HIDDEN_DIM,
        head_architecture=contract.architecture,
        target_assignment_scope=contract.target_scope,
    ).validated()


@pytest.mark.parametrize("contract", _CONTRACTS, ids=_CONTRACT_IDS)
def test_loads_only_the_matching_p1_family_contract(tmp_path, contract):
    checkpoint, expected_state = _save_checkpoint(tmp_path, contract)

    model, checkpoint_sha256, metadata = load_residual_proposal_head(
        checkpoint,
        expected_config=_runtime_config(checkpoint, contract),
        device="cpu",
        expected_b6_checkpoint_sha256=_B6_SHA256,
    )

    assert type(model) is contract.model_type
    assert model.training is False
    assert checkpoint_sha256 == sha256_file(checkpoint)
    assert metadata["schema"] == contract.schema
    assert set(model.state_dict()) == set(expected_state)
    for name, expected in expected_state.items():
        torch.testing.assert_close(
            model.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("contract", _CONTRACTS, ids=_CONTRACT_IDS)
@pytest.mark.parametrize(
    ("mismatch", "error_pattern"),
    (
        ("schema", "schema mismatch"),
        ("architecture", "head_architecture disagrees"),
        ("target_scope", "target_assignment_scope disagrees"),
    ),
)
def test_contract_metadata_mismatches_fail_closed(
    tmp_path,
    contract,
    mismatch,
    error_pattern,
):
    updates: dict[str, str] = {}
    if mismatch == "schema":
        updates["schema"] = {
            P1_HEAD_SCHEMA: P1R_HEAD_SCHEMA,
            P1R_HEAD_SCHEMA: P1S_HEAD_SCHEMA,
            P1S_HEAD_SCHEMA: P1_HEAD_SCHEMA,
        }[contract.schema]
    elif mismatch == "architecture":
        updates["architecture"] = (
            "per_voxel_mlp"
            if contract.architecture == "native_sparse_context_v1"
            else "native_sparse_context_v1"
        )
    else:
        updates["target_scope"] = (
            "snapshot_inside_only"
            if contract.target_scope == "scene_global"
            else "scene_global"
        )
    checkpoint, _ = _save_checkpoint(tmp_path, contract, **updates)

    with pytest.raises(ValueError, match=error_pattern):
        load_residual_proposal_head(
            checkpoint,
            expected_config=_runtime_config(checkpoint, contract),
            device="cpu",
            expected_b6_checkpoint_sha256=_B6_SHA256,
        )


@pytest.mark.parametrize("contract", _CONTRACTS, ids=_CONTRACT_IDS)
def test_loading_does_not_perturb_the_callers_cpu_rng(tmp_path, contract):
    checkpoint, _ = _save_checkpoint(tmp_path, contract)
    torch.manual_seed(20260730)
    state_before = torch.get_rng_state().clone()

    load_residual_proposal_head(
        checkpoint,
        expected_config=_runtime_config(checkpoint, contract),
        device="cpu",
        expected_b6_checkpoint_sha256=_B6_SHA256,
    )

    torch.testing.assert_close(
        torch.get_rng_state(),
        state_before,
        rtol=0.0,
        atol=0.0,
    )

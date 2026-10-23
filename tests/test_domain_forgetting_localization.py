"""Tests for per-module drift and rollback (tiny two-block BERT, offline)."""

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from fabricpc.experiments.domain_forgetting.localization import (
    MultiDomainEvaluator,
    compute_drift,
    compute_rollback,
    load_run,
    module_groups,
    module_keys,
    module_names,
    norm_matched_positions,
    random_rollback_flat,
    rollback_flat,
    rollback_positions,
    rollback_transitions,
    squared_updates,
)
from fabricpc.experiments.domain_forgetting.runner import evaluate_domains
from fabricpc.experiments.domain_forgetting.validation import evaluate_fixed_set
from tests.domain_forgetting_fixtures import build_tiny_run

BERT_MINI_MODULES = [
    "token_embeddings",
    "position_embeddings",
    *(f"L{i}_{part}" for i in range(4) for part in ("mha", "mlp1", "mlp2")),
    "mlm_transform",
    "mlm_head",
]


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    config = build_tiny_run(tmp_path_factory.mktemp("localization"))
    return load_run(config.output_dir)


def test_modules_partition_every_parameter_exactly_once(context):
    flat = context.snapshots[0]
    modules = module_names(context.structure, flat)
    assert modules == [
        "token_embeddings",
        "position_embeddings",
        *(f"L{i}_{part}" for i in range(2) for part in ("mha", "mlp1", "mlp2")),
        "mlm_transform",
        "mlm_head",
    ]
    keys = [key for module in modules for key in module_keys(flat, module)]
    assert sorted(keys) == sorted(flat)
    assert len(keys) == len(set(keys))
    for name in context.structure.nodes:
        if name.endswith("_residual") or name.startswith("dropout_mask"):
            assert name not in modules
    groups = module_groups(modules)
    assert groups == {
        "embeddings": ["token_embeddings", "position_embeddings"],
        "blocks_0-0": ["L0_mha", "L0_mlp1", "L0_mlp2"],
        "blocks_1-1": ["L1_mha", "L1_mlp1", "L1_mlp2"],
        "head": ["mlm_transform", "mlm_head"],
    }
    members = [m for group in groups.values() for m in group]
    assert sorted(members) == sorted(modules)


def test_bert_mini_groups_follow_depth():
    groups = module_groups(BERT_MINI_MODULES)
    assert list(groups) == ["embeddings", "blocks_0-1", "blocks_2-3", "head"]
    assert [len(v) for v in groups.values()] == [2, 6, 6, 2]
    assert len(BERT_MINI_MODULES) == 16


def test_rollback_changes_only_the_module(context):
    target, reference = context.snapshots[3], context.snapshots[1]
    keys = module_keys(target, "L0_mlp1")
    rolled = rollback_flat(target, reference, keys)
    for key, value in rolled.items():
        expected = reference[key] if key in keys else target[key]
        assert value.tobytes() == expected.tobytes(), key
    assert any(target[k].tobytes() != reference[k].tobytes() for k in keys)
    same = rollback_flat(target, target, keys)
    assert all(same[k].tobytes() == target[k].tobytes() for k in target)


def test_random_control_replaces_exactly_the_requested_count(context):
    target = context.snapshots[3]
    reference = {key: value + 1.0 for key, value in target.items()}
    count = int(target["L0_mlp1/weights/W_ff1"].size)
    rolled = random_rollback_flat(target, reference, count, np.random.default_rng(0))
    changed = sum(int((rolled[k] != target[k]).sum()) for k in target)
    assert changed == count
    for key in target:
        mask = rolled[key] != target[key]
        np.testing.assert_array_equal(rolled[key][mask], reference[key][mask])
    again = random_rollback_flat(target, reference, count, np.random.default_rng(0))
    assert all(again[k].tobytes() == rolled[k].tobytes() for k in target)
    other = random_rollback_flat(target, reference, count, np.random.default_rng(1))
    assert any(other[k].tobytes() != rolled[k].tobytes() for k in target)
    with pytest.raises(ValueError, match="count"):
        random_rollback_flat(target, reference, 10**9, np.random.default_rng(0))


def test_drift_matches_hand_computed_norms(context):
    drift = compute_drift(context.snapshots, context.structure, context.domains)
    assert [t["name"] for t in drift["transitions"]] == ["S0->S1", "S1->S2", "S2->S3"]
    s0, s1, s2 = context.snapshots[:3]
    keys = module_keys(s0, "mlm_head")
    flat = lambda snap: np.concatenate(
        [snap[k].astype(np.float64).ravel() for k in keys]
    )
    entry = drift["transitions"][1]["modules"]["mlm_head"]
    delta = flat(s2) - flat(s1)
    assert entry["update_norm"] == pytest.approx(np.linalg.norm(delta))
    assert entry["relative_update_norm"] == pytest.approx(
        np.linalg.norm(delta) / np.linalg.norm(flat(s1))
    )
    cosine = flat(s1) @ flat(s2) / (np.linalg.norm(flat(s1)) * np.linalg.norm(flat(s2)))
    assert entry["cosine"] == pytest.approx(cosine)
    assert entry["cumulative_norm"] == pytest.approx(
        np.linalg.norm(flat(s2) - flat(s0))
    )
    first = drift["transitions"][0]["modules"]["mlm_head"]
    assert first["cumulative_norm"] == pytest.approx(first["update_norm"])
    assert set(drift["transitions"][0]["groups"]) == set(drift["groups"])
    assert drift["transitions"][0]["model"]["update_norm"] > 0
    assert sum(drift["num_scalars"].values()) == sum(v.size for v in s0.values())


def test_gathered_evaluator_matches_the_full_head_evaluation(context):
    params = context.params(2)
    for batch_size in (3, 8):
        evaluator = MultiDomainEvaluator(
            context.structure, context.valid_sets, batch_size
        )
        batched = evaluator(params)
        for name in context.domains:
            reference = evaluate_fixed_set(
                params, context.structure, context.valid_sets[name], batch_size=4
            )
            assert batched[name]["num_masked_tokens"] == (
                reference["num_masked_tokens"]
            )
            assert batched[name]["masked_ce"] == pytest.approx(
                reference["masked_ce"], rel=1e-6
            )
            assert batched[name]["masked_accuracy"] == pytest.approx(
                reference["masked_accuracy"], abs=1e-12
            )
    assert evaluate_domains(params, context.structure, context.valid_sets) == (
        MultiDomainEvaluator(context.structure, context.valid_sets)(params)
    )
    matrix = json.loads((context.run_dir / "masked_ce_matrix.json").read_text())
    for index, name in enumerate(context.domains):
        assert batched[name]["masked_ce"] == pytest.approx(
            matrix["matrix"][2][index], rel=1e-6
        )


def test_norm_matched_control_reaches_the_module_squared_update(context):
    target, reference = context.snapshots[3], context.snapshots[1]
    squares = squared_updates(target, reference)
    keys = module_keys(target, "L0_mlp1")
    budget = float(
        sum(np.sum((reference[k].astype(np.float64) - target[k]) ** 2) for k in keys)
    )
    positions = norm_matched_positions(squares, budget, np.random.default_rng(0))
    assert squares[positions].sum() >= budget * (1 - 1e-9)
    assert squares[positions[:-1]].sum() < budget
    assert len(set(positions.tolist())) == len(positions)
    again = norm_matched_positions(squares, budget, np.random.default_rng(0))
    np.testing.assert_array_equal(positions, again)
    rolled = rollback_positions(target, reference, positions)
    assert squared_updates(rolled, reference).sum() == pytest.approx(
        squares.sum() - squares[positions].sum()
    )
    assert len(norm_matched_positions(squares, 0.0, np.random.default_rng(0))) == 0


def test_rollback_reports_recovery_cost_and_random_control(context):
    result = compute_rollback(context, random_draws=2, seed=0, eval_batch_size=8)
    assert rollback_transitions(3) == [(3, 1), (3, 2), (2, 1)]
    assert [t["name"] for t in result["transitions"]] == ["S3<-S1", "S3<-S2", "S2<-S1"]
    entry = result["transitions"][0]
    assert entry["reference_domain"] == "news"
    assert entry["current_domain"] == "wiki"
    assert set(entry["modules"]) == set(result["modules"])
    assert set(entry["groups"]) == set(result["groups"])
    head = entry["modules"]["mlm_head"]
    evaluator = MultiDomainEvaluator(context.structure, context.valid_sets, 8)
    rolled = evaluator(
        context.params(
            rollback_flat(
                context.snapshots[3],
                context.snapshots[1],
                module_keys(context.snapshots[0], "mlm_head"),
            )
        )
    )
    for name in context.domains:
        assert head["recovery"][name] == pytest.approx(
            entry["baseline"][name]["masked_ce"] - rolled[name]["masked_ce"]
        )
    assert head["cost"] == pytest.approx(
        rolled["wiki"]["masked_ce"] - entry["baseline"]["wiki"]["masked_ce"]
    )
    assert len(head["random"]["draws"]) == 2
    assert head["random"]["recovery_std"]["news"] >= 0
    matched = head["random_norm_matched"]
    assert len(matched["draws"]) == 2
    for draw in matched["draws"]:
        assert draw["squared_update"] >= head["squared_update"] * (1 - 1e-9)
        assert draw["num_scalars"] > 0
    assert set(head["recovery_minus_random_norm_matched"]) == set(context.domains)
    assert (
        result["provenance"]["stage_params_sha256"]
        == json.loads((context.run_dir / "summary.json").read_text())[
            "stage_params_sha256"
        ]
    )
    assert result["provenance"]["valid_sha256"] == context.run_config["valid_sha256"]
    again = compute_rollback(context, random_draws=2, seed=0, eval_batch_size=8)
    assert again["transitions"][0]["modules"]["mlm_head"]["random"] == head["random"]
    assert again["transitions"][0]["modules"]["mlm_head"]["random_norm_matched"] == (
        matched
    )
    assert Path(context.run_dir).exists()

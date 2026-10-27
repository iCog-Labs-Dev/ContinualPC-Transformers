"""Tests for the forgetting predictors, rank correlation, bootstrap and analysis CLI."""

import json
import math
import os
import shutil

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from fabricpc.experiments.domain_forgetting.correlations import (
    FISHER_LABEL,
    bootstrap_ci,
    compute_correlations,
    compute_predictors,
    paired_bootstrap_difference,
    permutation_p_value,
    run_predictor_table,
    spearman,
)
from fabricpc.experiments.domain_forgetting.localization import load_run
from fabricpc.training.train_autoregressive import compute_loss
from fabricpc.training.train_backprop import compute_forward_pass
from tests.domain_forgetting_fixtures import build_tiny_run


def test_spearman_known_orderings_and_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    x, y = [1, 2, 2, 3], [1, 2, 3, 4]
    rx, ry = np.array([1, 2.5, 2.5, 4]), np.array([1, 2, 3, 4])
    expected = np.corrcoef(rx, ry)[0, 1]
    assert spearman(x, y) == pytest.approx(expected)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    with pytest.raises(ValueError):
        spearman([1, 2], [1, 2, 3])


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    x = rng.normal(size=16)
    y = x + 0.5 * rng.normal(size=16)
    first = bootstrap_ci(x, y, resamples=500, seed=3)
    assert first == bootstrap_ci(x, y, resamples=500, seed=3)
    assert first != bootstrap_ci(x, y, resamples=500, seed=4)
    assert first["low"] <= spearman(x, y) <= first["high"]
    assert 0 < first["num_valid"] <= 500
    perfect = bootstrap_ci(np.arange(16), np.arange(16), resamples=200, seed=0)
    assert perfect["low"] == pytest.approx(1.0)


def test_paired_bootstrap_uses_the_same_resampled_modules():
    rng = np.random.default_rng(0)
    y = rng.normal(size=16)
    x = y + 0.1 * rng.normal(size=16)
    z = rng.normal(size=16)
    result = paired_bootstrap_difference(x, z, y, resamples=400, seed=1)
    assert result == paired_bootstrap_difference(x, z, y, resamples=400, seed=1)
    assert result["estimate"] == pytest.approx(spearman(x, y) - spearman(z, y))
    assert result["low"] <= result["estimate"] <= result["high"]
    assert result["low"] > 0
    assert result["fraction_positive"] > 0.95
    same = paired_bootstrap_difference(x, x, y, resamples=100, seed=0)
    assert same["low"] == same["high"] == 0.0


def test_permutation_p_value_is_seeded_and_calibrated():
    rng = np.random.default_rng(0)
    y = rng.normal(size=16)
    strong = permutation_p_value(y, y, permutations=2000, seed=0)
    assert strong == pytest.approx(1 / 2001)
    noise = rng.normal(size=16)
    p = permutation_p_value(noise, y, permutations=2000, seed=0)
    assert p == permutation_p_value(noise, y, permutations=2000, seed=0)
    exact = stats.spearmanr(noise, y).pvalue
    assert abs(p - exact) < 0.05
    assert math.isnan(permutation_p_value([1, 1, 1], [1, 2, 3]))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return build_tiny_run(tmp_path_factory.mktemp("correlations"))


def test_predictors_match_direct_gradients(run):
    context = load_run(run.output_dir)
    result = compute_predictors(context, num_windows=8, batch_size=4)
    assert [t["name"] for t in result["transitions"]] == ["S3<-S1", "S3<-S2", "S2<-S1"]
    entry = result["transitions"][0]
    assert (entry["old_domain"], entry["new_domain"]) == ("news", "reviews")
    predictors = entry["predictors"]
    assert set(predictors) == {
        "update_norm",
        "relative_update_norm",
        "grad_norm",
        "fisher",
        "param_count",
    }
    structure = context.structure
    params = context.params(1)

    def mean_ce(p, arrays, index):
        batch = {
            "x": jnp.asarray(arrays["masked_input_ids"][index]),
            "y": jnp.asarray(arrays["labels"][index]),
            "loss_mask": jnp.asarray(arrays["loss_mask"][index]),
            "attention_mask": jnp.asarray(arrays["attention_mask"][index]),
        }
        state = compute_forward_pass(p, structure, batch, jax.random.PRNGKey(0))
        return compute_loss(
            state, batch["y"], "mlm_head", "cross_entropy", loss_mask=batch["loss_mask"]
        )

    def module_norm(grads, module):
        return float(
            np.sqrt(
                sum(
                    np.sum(np.asarray(g) ** 2)
                    for g in jax.tree_util.tree_leaves(grads.nodes[module])
                )
            )
        )

    reviews = context.valid_sets["reviews"]
    grads = jax.grad(mean_ce)(params, reviews, slice(0, 8))
    for module in ("mlm_head", "L0_mha", "token_embeddings"):
        assert predictors["grad_norm"][module] == pytest.approx(
            module_norm(grads, module), rel=1e-4
        )
    news = context.valid_sets["news"]
    squares = []
    for index in range(8):
        if news["loss_mask"][index].sum() == 0:
            continue
        g = jax.grad(mean_ce)(params, news, slice(index, index + 1))
        squares.append(module_norm(g, "L1_mlp2") ** 2)
    assert entry["fisher_windows"] == len(squares)
    assert predictors["fisher"]["L1_mlp2"] == pytest.approx(np.mean(squares), rel=1e-4)
    assert result["labels"]["fisher"] == FISHER_LABEL
    assert "empirical Fisher (per-window mean CE, true labels)" in (
        result["definition"]["fisher"]
    )
    assert predictors["param_count"]["mlm_head"] == sum(
        context.snapshots[0][k].size
        for k in context.snapshots[0]
        if k.startswith("mlm_head/")
    )


def test_analysis_cli_all_and_multi_seed_correlations(run, tmp_path):
    from examples import domain_forgetting_analysis as cli

    common = [
        "--run_dir",
        run.output_dir,
        "--random_draws",
        "1",
        "--map_windows",
        "8",
        "--map_batch_size",
        "4",
        "--pc_infer_steps",
        "3",
        "--pc_eta_infer",
        "0.001",
        "--predictor_windows",
        "8",
        "--predictor_batch_size",
        "4",
        "--bootstrap",
        "100",
        "--eval_batch_size",
        "8",
        "--quiet",
    ]
    results = cli.main(["all", *common])
    for name in ("drift", "rollback", "error_maps", "predictors", "correlations"):
        assert os.path.exists(os.path.join(run.output_dir, f"{name}.json")), name
    assert os.path.exists(os.path.join(run.output_dir, "correlations.png"))
    corr = results["correlations"]
    per_run = corr["runs"][run.output_dir]
    assert set(per_run) == {"S3<-S1", "S3<-S2", "S2<-S1"}
    predictors = per_run["S3<-S1"]["predictors"]
    assert {
        "update_norm",
        "fisher",
        "spike_only_onset",
        "spike_only_retrospective",
        "standard_pc_onset",
        "standard_pc_retrospective",
    } <= set(predictors)
    assert {
        "spike_only_onset_per_dim",
        "standard_pc_retrospective_per_dim",
    } <= set(predictors)
    for name, value in predictors.items():
        rho = value["rho"]
        assert math.isnan(rho) or -1.0 <= rho <= 1.0
        p_value = value["permutation_p"]
        assert math.isnan(p_value) or 1 / 2001 <= p_value <= 1.0
        if name.startswith(("spike_only_", "standard_pc_")):
            assert "vs_update_norm" in value
        else:
            assert "vs_update_norm" not in value
    assert per_run["S3<-S1"]["num_modules"] == 10
    agreement = per_run["S3<-S1"]["map_agreement"]
    assert set(agreement) == {
        "onset",
        "onset_per_dim",
        "retrospective",
        "retrospective_per_dim",
    }
    table = run_predictor_table(run.output_dir)
    maps = json.loads(open(os.path.join(run.output_dir, "error_maps.json")).read())
    onset = next(
        m
        for m in maps["maps"]
        if (m["setting"], m["kind"], m["stage"]) == ("spike_only", "onset", 1)
    )
    assert table["S3<-S1"]["predictors"]["spike_only_onset"] == {
        m: v["energy_sum_ce"] for m, v in onset["modules"].items()
    }
    assert corr["provenance"][run.output_dir]["stage_params_sha256"] == (
        json.loads(open(os.path.join(run.output_dir, "summary.json")).read())[
            "stage_params_sha256"
        ]
    )
    for name in ("drift", "rollback", "error_maps", "predictors"):
        saved = json.loads(open(os.path.join(run.output_dir, f"{name}.json")).read())
        assert saved["provenance"]["stage_params_sha256"] == (
            corr["provenance"][run.output_dir]["stage_params_sha256"]
        ), name

    copy = tmp_path / "seed_copy"
    shutil.copytree(run.output_dir, copy)
    both = compute_correlations([run.output_dir, str(copy)], resamples=50, seed=0)
    entry = both["aggregate"]["S3<-S1"]["update_norm"]
    assert entry["num_runs"] == 2
    assert entry["rhos"][0] == entry["rhos"][1]
    assert entry["sd"] == pytest.approx(0.0)
    saved = json.loads(open(os.path.join(run.output_dir, "correlations.json")).read())
    assert saved["aggregate"]["S3<-S1"]["update_norm"]["num_runs"] == 1


def test_correlations_refuse_analyses_of_other_stage_snapshots(run, tmp_path):
    from examples import domain_forgetting_analysis as cli

    pristine = tmp_path / "pristine"
    shutil.copytree(run.output_dir, pristine)
    cli.main(
        [
            "rollback",
            "--run_dir",
            str(pristine),
            "--random_draws",
            "1",
            "--eval_batch_size",
            "8",
            "--quiet",
        ]
    )
    run_predictor_table(pristine)

    def tampered(name, edit):
        copy = tmp_path / name
        shutil.copytree(pristine, copy)
        target = copy / edit[0]
        payload = json.loads(target.read_text())
        edit[1](payload)
        target.write_text(json.dumps(payload))
        return copy

    stage = tampered(
        "stage",
        (
            "checkpoint_manifest.json",
            lambda m: m["stages"][2].update(params_sha256="0" * 64),
        ),
    )
    with pytest.raises(ValueError, match="stale"):
        run_predictor_table(stage)
    valid = tampered(
        "valid",
        ("run_config.json", lambda c: c["valid_sha256"].update(news="0" * 64)),
    )
    with pytest.raises(ValueError, match="stale"):
        compute_correlations([str(valid)], resamples=10, permutations=10)
    missing = tampered("missing", ("rollback.json", lambda r: r.pop("provenance")))
    with pytest.raises(ValueError, match="provenance"):
        run_predictor_table(missing)

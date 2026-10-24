"""Tests for the architecture baseline of the energy-recovery analysis."""

import itertools
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from fabricpc.experiments.domain_forgetting import architecture_baseline as ab

ORDERS = list(itertools.permutations(("news", "reviews", "wiki")))
TYPES = ("mha", "mlp1", "mlp2")
MODULES = [f"L{i}_{t}" for i in range(4) for t in TYPES]
EXTRA = ["token_embeddings", "mlm_head"]


def make_table(recovery, energy, perdim=None, update=None, modules=MODULES):
    """Return a ``run_predictor_table``-shaped table, same values per transition."""
    perdim = energy if perdim is None else perdim
    update = energy if update is None else update
    every = list(modules) + EXTRA

    def full(values):
        mapping = dict(zip(modules, map(float, values)))
        return {m: mapping.get(m, 1.0) for m in every}

    return {
        name: {
            "old_domain": "news",
            "recovery": full(recovery),
            "predictors": {
                "spike_only_onset": full(energy),
                "spike_only_onset_per_dim": full(perdim),
                "update_norm": full(update),
                "fisher": full(energy),
            },
        }
        for name in ab.TRANSITIONS
    }


def run(order, table, name):
    return {"run_dir": name, "order": order, "table": table}


def arch_runs():
    """Recovery is type-major then depth; energy is random noise."""
    rng = np.random.default_rng(0)
    recovery = [100 * TYPES.index(m.split("_")[1]) + int(m[1]) for m in MODULES]
    return [
        run(order, make_table(recovery, rng.uniform(1, 10, 12)), f"o{i}")
        for i, order in enumerate(ORDERS)
    ]


def energy_runs():
    """Recovery is type plus a within-type energy order unrelated to depth."""
    rng = np.random.default_rng(1)
    runs = []
    for i, order in enumerate(ORDERS):
        log_energy = np.zeros(12)
        for t in range(3):
            log_energy[[3 * d + t for d in range(4)]] = rng.permutation(4)
        recovery = [
            100 * TYPES.index(m.split("_")[1]) + log_energy[k]
            for k, m in enumerate(MODULES)
        ]
        runs.append(run(order, make_table(recovery, np.exp(log_energy)), f"o{i}"))
    return runs


def test_arch_only_signal_is_not_supported():
    result = ab.analyse(arch_runs(), permutations=50)
    assert result["config"]["held_out_orders"] == [
        ab.order_key(o) for o in ORDERS if o not in ab.DEVELOPMENT_ORDERS
    ]
    for key in result["primary"]["held_out_orders"]:
        entry = result["looo"][key][ab.PRIMARY_TRANSITION]
        assert entry["arch"]["rho"] == pytest.approx(1.0)
        assert entry["arch+energy"]["rho"] == pytest.approx(1.0)
        np.testing.assert_allclose(
            entry["arch"]["prediction"],
            [4 * t + d + 1 for d in range(4) for t in range(3)],
            atol=1e-8,
        )
    assert result["primary"]["status"] == "not supported"
    assert result["primary"]["supported"] is False
    assert result["primary"]["num_orders"] == 4


def test_energy_beyond_type_is_supported():
    result = ab.analyse(energy_runs(), permutations=200)
    primary = result["primary"]
    assert primary["status"] == "supported"
    assert primary["positive"] == 4 and primary["mean"] > 0.05
    for key, delta in primary["deltas"].items():
        entry = result["looo"][key][ab.PRIMARY_TRANSITION]
        assert entry["arch+energy"]["rho"] == pytest.approx(1.0)
        assert delta == pytest.approx(1.0 - entry["arch"]["rho"])
    every = result["comparisons"]["all"][ab.PRIMARY_TRANSITION]["arch+energy_vs_arch"]
    assert every["num_orders"] == 6
    secondary = result["secondary"][ab.order_key(ORDERS[0])][ab.PRIMARY_TRANSITION]
    assert secondary["energy"]["seed_mean"]["partial"] == pytest.approx(1.0)
    assert secondary["energy"]["seed_mean"]["p"] < 0.05
    assert len(secondary["energy"]["per_seed"]) == 1


def test_explicit_held_out_and_development():
    runs = energy_runs()
    result = ab.analyse(runs, development=ORDERS[:5], permutations=10)
    assert result["primary"]["held_out_orders"] == [ab.order_key(ORDERS[5])]
    with pytest.raises(ValueError, match="no runs"):
        ab.analyse(runs[:3], held_out=[ORDERS[5]], permutations=10)


def test_no_held_out_orders_is_not_evaluated():
    runs = [r for r in arch_runs() if r["order"] in ab.DEVELOPMENT_ORDERS]
    result = ab.analyse(runs, permutations=10)
    assert result["primary"]["status"] == ab.NOT_EVALUATED
    assert result["primary"]["supported"] is None
    assert (
        result["comparisons"]["all"][ab.PRIMARY_TRANSITION]["arch+energy_vs_arch"][
            "num_orders"
        ]
        == 2
    )


def test_seed_averaging_and_grouping_by_order():
    forward, reverse = ab.FORWARD, ab.REVERSE
    base = np.arange(1.0, 13.0)
    runs = [
        run(forward, make_table(base, base), "b"),
        run(reverse, make_table(base[::-1], base), "a"),
        run(forward, make_table(base + 2, 3 * base), "c"),
    ]
    orders = ab.group_orders(runs)
    assert set(orders) == {forward, reverse}
    data = orders[forward]
    assert data.run_dirs == ["b", "c"] and data.modules == MODULES
    np.testing.assert_allclose(data.recovery["S3<-S1"], base + 1)
    np.testing.assert_allclose(data.features["S3<-S1"]["energy"], 2 * base)
    np.testing.assert_allclose(data.per_seed[1]["recovery"]["S3<-S1"], base + 2)
    assert len(orders[reverse].per_seed) == 1


def test_permute_within_type_preserves_types():
    types = [m.split("_")[1] for m in MODULES]
    rng = np.random.default_rng(3)
    moved = False
    for _ in range(20):
        index = ab.permute_within(types, rng)
        assert sorted(index) == list(range(12))
        assert [types[i] for i in index] == types
        moved |= list(index) != list(range(12))
    assert moved


def test_ols_known_answer():
    x = np.arange(5.0)[:, None]
    coefficients = ab.ols_fit(ab.with_intercept(x), 2.0 + 3.0 * x[:, 0])
    np.testing.assert_allclose(coefficients, [2.0, 3.0])


def test_guards():
    base = np.arange(1.0, 13.0)
    bad = base.copy()
    bad[4] = 0.0
    runs = [run(o, make_table(base, base), str(o)) for o in ab.DEVELOPMENT_ORDERS]
    runs[1] = run(ab.REVERSE, make_table(base, bad), "bad")
    with pytest.raises(ValueError, match="log needs positive"):
        ab.analyse(runs, permutations=10)
    short = run(ab.FORWARD, make_table(base[:11], base[:11], modules=MODULES[:11]), "s")
    with pytest.raises(ValueError, match="11 block modules"):
        ab.group_orders([short])
    other = [m.replace("L3", "L4") for m in MODULES]
    with pytest.raises(ValueError, match="differ"):
        ab.group_orders(
            [
                run(ab.FORWARD, make_table(base, base), "f"),
                run(ab.REVERSE, make_table(base, base, modules=other), "r"),
            ]
        )
    with pytest.raises(ValueError, match="differ"):
        ab.group_orders(
            [
                run(ab.FORWARD, make_table(base, base), "f0"),
                run(ab.FORWARD, make_table(base, base, modules=other), "f1"),
            ]
        )
    missing = make_table(base, base)
    del missing["S3<-S1"]["predictors"]["update_norm"]
    with pytest.raises(ValueError, match="update_norm is missing"):
        ab.group_orders([run(ab.FORWARD, missing, "m")])
    with pytest.raises(ValueError, match="at least two orders"):
        ab.analyse([runs[0]], permutations=10)
    with pytest.raises(ValueError, match="unknown module type"):
        ab.module_type("L0_attn")


def test_cli_writes_json(tmp_path, monkeypatch, capsys):
    from examples import domain_forgetting_analysis as cli

    tables = {}
    run_dirs = []
    for index, entry in enumerate(energy_runs()):
        path = tmp_path / f"run{index}"
        path.mkdir()
        config = {"config": {"domains": list(entry["order"]), "seed": 0}}
        (path / "run_config.json").write_text(json.dumps(config))
        tables[str(path)] = entry["table"]
        run_dirs.append(str(path))
    monkeypatch.setattr(ab, "run_predictor_table", lambda path: tables[str(path)])
    output = tmp_path / "out"
    cli.main(
        [
            "architecture",
            "--run_dirs",
            *run_dirs,
            "--development_orders",
            "news,reviews,wiki",
            "wiki,reviews,news",
            "--output_dir",
            str(output),
            "--permutations",
            "20",
            "--seed",
            "1",
        ]
    )
    written = json.loads((output / "architecture_baseline.json").read_text())
    assert written["primary"]["status"] == "supported"
    assert written["config"]["permutations"] == 20 and written["config"]["seed"] == 1
    assert "git_commit" in written and written["run_dirs"] == run_dirs
    assert "supported" in capsys.readouterr().out

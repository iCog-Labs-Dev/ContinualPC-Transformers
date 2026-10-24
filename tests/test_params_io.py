"""Tests for saving and loading GraphParams with fabricpc.utils.params_io."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.utils import load_params, read_params_metadata, save_params
from fabricpc.utils.params_io import params_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_KWARGS = dict(
    depth=1, embed_dim=16, num_heads=2, mlp_dim=32, seq_len=8, vocab_size=20
)


def _structure(**overrides):
    return create_encoder_mlm_model(
        **{**MODEL_KWARGS, **overrides},
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=2, max_norm=5.0),
    )


def _assert_params_equal(left: GraphParams, right: GraphParams) -> None:
    assert list(left.nodes) == list(right.nodes)
    for name in left.nodes:
        for collection in ("weights", "biases"):
            a, b = getattr(left.nodes[name], collection), getattr(
                right.nodes[name], collection
            )
            assert set(a) == set(b), (name, collection)
            for key in a:
                assert a[key].dtype == b[key].dtype
                np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]))


def test_round_trip_is_exact_and_follows_graph_order(tmp_path):
    structure = _structure()
    params = initialize_params(structure, jax.random.PRNGKey(0))

    archive, sidecar = save_params(tmp_path / "model.npz", params)
    loaded, metadata = load_params(tmp_path / "model.npz", structure)

    assert (archive, sidecar) == params_paths(tmp_path / "model")
    assert metadata == {}
    assert list(loaded.nodes) == list(structure.nodes)
    _assert_params_equal(loaded, params)
    assert not list(tmp_path.glob(".*.tmp"))


def test_metadata_round_trips_and_is_readable_alone(tmp_path):
    params = initialize_params(_structure(), jax.random.PRNGKey(0))
    metadata = {
        "vocab": {"[PAD]": 0, "cat": 5},
        "tokenizer_json": None,
        "config": {"architecture": "encoder", "seq_len": 8, "dropout_rate": 0.1},
    }

    save_params(tmp_path / "model", params, metadata)

    assert read_params_metadata(tmp_path / "model.json") == metadata
    assert load_params(tmp_path / "model", _structure())[1] == metadata
    sidecar = json.loads((tmp_path / "model.json").read_text())
    assert (sidecar["format"], sidecar["version"]) == ("fabricpc-params", 1)
    assert sidecar["schema"]["L0_mha"]["weights"].keys() == set(
        params.nodes["L0_mha"].weights
    )


def test_metadata_must_be_json(tmp_path):
    params = initialize_params(_structure(), jax.random.PRNGKey(0))

    with pytest.raises(TypeError):
        save_params(tmp_path / "model", params, {"key": jnp.ones(2)})


def test_missing_parameterless_node_is_tolerated(tmp_path):
    structure = _structure()
    params = initialize_params(structure, jax.random.PRNGKey(0))
    assert params.nodes["mlm_loss_mask"] == NodeParams(weights={}, biases={})
    trimmed = GraphParams(
        nodes={k: v for k, v in params.nodes.items() if k != "mlm_loss_mask"}
    )

    save_params(tmp_path / "model", trimmed)
    loaded, _ = load_params(tmp_path / "model", structure)

    assert list(loaded.nodes) == list(structure.nodes)
    assert loaded.nodes["mlm_loss_mask"] == NodeParams(weights={}, biases={})
    _assert_params_equal(loaded, params)


def test_structure_mismatch_raises_with_every_problem(tmp_path):
    structure = _structure()
    params = initialize_params(structure, jax.random.PRNGKey(0))
    nodes = dict(params.nodes)
    del nodes["L0_mlp1"]
    mha = nodes["L0_mha"]
    nodes["L0_mha"] = NodeParams(
        weights={k: v for k, v in mha.weights.items() if k != "W_q"},
        biases={**mha.biases, "b_extra": jnp.zeros(16)},
    )
    nodes["stray"] = NodeParams(weights={"w": jnp.zeros(2)}, biases={})
    save_params(tmp_path / "model", GraphParams(nodes=nodes))

    with pytest.raises(ValueError) as error:
        load_params(tmp_path / "model", structure)

    message = str(error.value)
    assert "nodes not in the graph: ['stray']" in message
    assert "missing node 'L0_mlp1'" in message
    assert "L0_mha.weights: missing ['W_q']" in message
    assert "L0_mha.biases: unexpected ['b_extra']" in message


def test_shape_mismatch_raises(tmp_path):
    params = initialize_params(_structure(), jax.random.PRNGKey(0))
    save_params(tmp_path / "model", params)

    with pytest.raises(ValueError, match=r"W_out: shape \(16, 20\), expected"):
        load_params(tmp_path / "model", _structure(vocab_size=21))
    with pytest.raises(ValueError, match="missing node 'L1_mha'"):
        load_params(tmp_path / "model", _structure(depth=2))


def test_rejects_other_formats_and_mismatched_archive(tmp_path):
    params = initialize_params(_structure(), jax.random.PRNGKey(0))
    save_params(tmp_path / "a", params)
    save_params(tmp_path / "b", initialize_params(_structure(), jax.random.PRNGKey(1)))

    (tmp_path / "a.npz").write_bytes((tmp_path / "b.npz").read_bytes())
    with pytest.raises(ValueError, match="digest"):
        load_params(tmp_path / "a", _structure())

    sidecar = json.loads((tmp_path / "b.json").read_text())
    (tmp_path / "b.json").write_text(json.dumps({**sidecar, "version": 99}))
    with pytest.raises(ValueError, match="unsupported fabricpc-params version 99"):
        load_params(tmp_path / "b", _structure())

    (tmp_path / "c.json").write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError, match="not a fabricpc-params file"):
        load_params(tmp_path / "c", _structure())
    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "missing", _structure())


def test_loads_in_a_fresh_process(tmp_path):
    structure = _structure()
    params = initialize_params(structure, jax.random.PRNGKey(7))
    save_params(tmp_path / "model", params, {"config": MODEL_KWARGS})
    script = textwrap.dedent(f"""
        import numpy as np
        from fabricpc.core.inference import InferenceSGDNormClip
        from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
        from fabricpc.utils import load_params, read_params_metadata

        path = {str(tmp_path / "model")!r}
        config = read_params_metadata(path)["config"]
        structure = create_encoder_mlm_model(
            **config,
            inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=2, max_norm=5.0),
        )
        params, _ = load_params(path, structure)
        np.savez(
            {str(tmp_path / "reloaded.npz")!r},
            **{{
                f"{{node}}/{{collection}}/{{name}}": np.asarray(value)
                for node, entry in params.nodes.items()
                for collection in ("weights", "biases")
                for name, value in getattr(entry, collection).items()
            }},
        )
        """)
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": os.pathsep.join(
            p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH")) if p
        ),
    }

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    reloaded = np.load(tmp_path / "reloaded.npz")
    expected = {
        f"{node}/{collection}/{name}": np.asarray(value)
        for node, entry in params.nodes.items()
        for collection in ("weights", "biases")
        for name, value in getattr(entry, collection).items()
    }
    assert set(reloaded.files) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(reloaded[key], value)

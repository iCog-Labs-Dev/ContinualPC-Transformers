"""Tests for the post-LN BERT encoder option and the Hugging Face BERT converter."""

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.encoder_transformer import (
    AddPositionEmbeddingNode,
    create_encoder_mlm_model,
)
from fabricpc.nodes.transformer_v2 import LnMlp1Node, MhaResidualNode, Mlp2ResidualNode
from fabricpc.training.train_autoregressive import (
    get_graph_param_gradient_autoregressive,
)
from fabricpc.training.train_backprop import (
    compute_forward_pass,
    compute_loss_autoregressive,
)
from fabricpc.utils import load_params, read_params_metadata

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
TINY_HF_CONFIG = {
    "vocab_size": 50,
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 32,
    "max_position_embeddings": 32,
}
DEMO_SMALL = [
    "--depth",
    "1",
    "--embed_dim",
    "16",
    "--num_heads",
    "2",
    "--mlp_dim",
    "32",
]


def _load_example(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", EXAMPLES / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def convert():
    return _load_example("convert_hf_bert")


@pytest.fixture(scope="module")
def demo():
    return _load_example("bert_demo")


def _synthetic_state_dict(
    rng: np.random.Generator,
    *,
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    intermediate_size: int,
    max_position_embeddings: int,
    **_unused,
) -> dict:
    def rand(*shape: int) -> np.ndarray:
        return rng.normal(size=shape).astype(np.float32)

    h, ff = hidden_size, intermediate_size
    word = rand(vocab_size, h)
    bias = rand(vocab_size)
    state = {
        "bert.embeddings.word_embeddings.weight": word,
        "bert.embeddings.position_embeddings.weight": rand(max_position_embeddings, h),
        "bert.embeddings.token_type_embeddings.weight": rand(2, h),
        "bert.embeddings.LayerNorm.weight": rand(h),
        "bert.embeddings.LayerNorm.bias": rand(h),
        "cls.predictions.bias": bias,
        "cls.predictions.transform.dense.weight": rand(h, h),
        "cls.predictions.transform.dense.bias": rand(h),
        "cls.predictions.transform.LayerNorm.weight": rand(h),
        "cls.predictions.transform.LayerNorm.bias": rand(h),
        "cls.predictions.decoder.weight": word,
        "cls.predictions.decoder.bias": bias,
    }
    for i in range(num_hidden_layers):
        prefix = f"bert.encoder.layer.{i}"
        for name in ("query", "key", "value"):
            state[f"{prefix}.attention.self.{name}.weight"] = rand(h, h)
            state[f"{prefix}.attention.self.{name}.bias"] = rand(h)
        state.update(
            {
                f"{prefix}.attention.output.dense.weight": rand(h, h),
                f"{prefix}.attention.output.dense.bias": rand(h),
                f"{prefix}.attention.output.LayerNorm.weight": rand(h),
                f"{prefix}.attention.output.LayerNorm.bias": rand(h),
                f"{prefix}.intermediate.dense.weight": rand(ff, h),
                f"{prefix}.intermediate.dense.bias": rand(ff),
                f"{prefix}.output.dense.weight": rand(h, ff),
                f"{prefix}.output.dense.bias": rand(h),
                f"{prefix}.output.LayerNorm.weight": rand(h),
                f"{prefix}.output.LayerNorm.bias": rand(h),
            }
        )
    return state


def _batch(ids: np.ndarray, attention_mask: np.ndarray, loss_mask=None) -> dict:
    return {
        "x": jnp.asarray(ids, dtype=jnp.int32),
        "y": jnp.asarray(ids, dtype=jnp.int32),
        "attention_mask": jnp.asarray(attention_mask, dtype=jnp.float32),
        "loss_mask": jnp.asarray(
            attention_mask if loss_mask is None else loss_mask, dtype=jnp.float32
        ),
    }


def test_converted_bert_logits_match_hugging_face(convert):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.BertConfig(**TINY_HF_CONFIG)
    model = transformers.BertForMaskedLM(config).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.1 * torch.randn_like(param))
    state_dict = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    seq_len = 8
    converted = convert.convert_bert_state_dict(
        state_dict, config=config.to_dict(), seq_len=seq_len
    )
    structure = convert.build_bert_structure(
        convert.bert_model_config(config.to_dict(), seq_len)
    )
    params = convert.graph_params_for(converted, structure)
    ids = np.array([[2, 5, 7, 9, 11, 3, 0, 0], [2, 13, 17, 19, 23, 29, 31, 3]])
    attention_mask = np.array([[1, 1, 1, 1, 1, 1, 0, 0], [1] * 8])

    with torch.no_grad():
        expected = model(
            input_ids=torch.tensor(ids), attention_mask=torch.tensor(attention_mask)
        ).logits.numpy()
    state = compute_forward_pass(
        params, structure, _batch(ids, attention_mask), jax.random.PRNGKey(0)
    )
    head = params.nodes["mlm_head"]
    hidden = state.nodes["mlm_transform"].z_mu
    logits = np.asarray(hidden @ head.weights["W_out"] + head.biases["b_out"])
    probs = np.asarray(state.nodes["mlm_head"].z_mu)

    valid = attention_mask.astype(bool)
    assert np.abs(logits - expected)[valid].max() < 1e-5
    expected_log_probs = torch.log_softmax(torch.tensor(expected), dim=-1).numpy()
    assert np.abs(np.log(probs) - expected_log_probs)[valid].max() < 1e-5


def test_convert_maps_and_transposes_synthetic_state_dict(convert):
    config = {**TINY_HF_CONFIG, "hidden_size": 8, "intermediate_size": 12}
    config["num_attention_heads"] = 2
    state = _synthetic_state_dict(np.random.default_rng(0), **config)

    params = convert.convert_bert_state_dict(state, config=config, seq_len=5)

    layer = "bert.encoder.layer.1"
    np.testing.assert_array_equal(
        params.nodes["L1_mlp1"].weights["W_ff1"],
        state[f"{layer}.intermediate.dense.weight"].T,
    )
    assert params.nodes["L1_mlp1"].weights["W_ff1"].shape == (8, 12)
    np.testing.assert_array_equal(
        params.nodes["L1_mlp2"].weights["W_ff2"],
        state[f"{layer}.output.dense.weight"].T,
    )
    np.testing.assert_array_equal(
        params.nodes["L1_mha"].weights["W_q"],
        state[f"{layer}.attention.self.query.weight"].T,
    )
    np.testing.assert_array_equal(
        params.nodes["L1_mha"].biases["ln_beta"],
        state[f"{layer}.attention.output.LayerNorm.bias"],
    )
    np.testing.assert_array_equal(
        params.nodes["mlm_transform"].weights["W_t"],
        state["cls.predictions.transform.dense.weight"].T,
    )
    np.testing.assert_array_equal(
        params.nodes["mlm_head"].weights["W_out"],
        state["bert.embeddings.word_embeddings.weight"].T,
    )
    np.testing.assert_allclose(
        params.nodes["position_embeddings"].weights["position_embeddings"],
        state["bert.embeddings.position_embeddings.weight"][:5]
        + state["bert.embeddings.token_type_embeddings.weight"][0],
    )
    structure = convert.build_bert_structure(convert.bert_model_config(config, 5))
    convert.graph_params_for(params, structure)


def test_convert_rejects_missing_unknown_and_untied_keys(convert):
    state = _synthetic_state_dict(np.random.default_rng(0), **TINY_HF_CONFIG)
    convert.convert_bert_state_dict(
        {**state, "bert.pooler.dense.weight": np.zeros((16, 16))},
        config=TINY_HF_CONFIG,
        seq_len=8,
    )

    missing = dict(state)
    del missing["bert.encoder.layer.1.output.LayerNorm.bias"]
    with pytest.raises(ValueError, match="missing state dict keys"):
        convert.convert_bert_state_dict(missing, config=TINY_HF_CONFIG, seq_len=8)
    with pytest.raises(ValueError, match="unmapped state dict keys"):
        convert.convert_bert_state_dict(
            {**state, "bert.encoder.layer.2.output.dense.bias": np.zeros(16)},
            config=TINY_HF_CONFIG,
            seq_len=8,
        )
    untied = {**state, "cls.predictions.decoder.bias": np.ones(50, np.float32)}
    with pytest.raises(ValueError, match="decoder.bias"):
        convert.convert_bert_state_dict(untied, config=TINY_HF_CONFIG, seq_len=8)
    with pytest.raises(ValueError, match="seq_len"):
        convert.convert_bert_state_dict(state, config=TINY_HF_CONFIG, seq_len=33)


def test_graph_params_for_rejects_params_of_another_graph(convert):
    state = _synthetic_state_dict(np.random.default_rng(0), **TINY_HF_CONFIG)
    params = convert.convert_bert_state_dict(state, config=TINY_HF_CONFIG, seq_len=8)
    other = convert.build_bert_structure(
        convert.bert_model_config({**TINY_HF_CONFIG, "num_hidden_layers": 1}, 8)
    )

    with pytest.raises(ValueError, match="do not match"):
        convert.graph_params_for(params, other)


def test_check_loading_info_rejects_weights_hugging_face_would_initialise(convert):
    complete = {
        "missing_keys": [],
        "unexpected_keys": ["bert.pooler.dense.weight", "cls.seq_relationship.bias"],
        "mismatched_keys": set(),
        "error_msgs": [],
    }
    convert.check_loading_info(complete)
    convert.check_loading_info({**complete, "unexpected_keys": ["pooler.dense.bias"]})

    with pytest.raises(ValueError, match="missing_keys"):
        convert.check_loading_info(
            {**complete, "missing_keys": ["cls.predictions.bias"]}
        )
    with pytest.raises(ValueError, match="unexpected_keys"):
        convert.check_loading_info(
            {**complete, "unexpected_keys": ["bert.encoder.extra"]}
        )
    with pytest.raises(ValueError, match="mismatched_keys"):
        convert.check_loading_info(
            {**complete, "mismatched_keys": {("bert.embeddings.x", (1,), (2,))}}
        )


def _write_vocab(convert, path: Path, words) -> None:
    path.write_text("\n".join([*convert.BERT_SPECIAL_TOKENS, *words]) + "\n")


def test_wordpiece_tokenizer_matches_bert_tokenizer_fast(convert, tmp_path):
    transformers = pytest.importorskip("transformers")
    vocab_path = tmp_path / "vocab.txt"
    words = ["the", "cafe", "un", "##aff", "##able", "is", "a", "dog", ",", ".", "!"]
    _write_vocab(convert, vocab_path, [*words, "run", "##ning", "中", "'", "s"])
    sentences = [
        "The café is unaffable!",
        "a DOG, running.",
        "the dog's [MASK] is unknownword",
        "  multiple   spaces\tand\nnewlines 中文 ",
    ]

    ours = convert.build_wordpiece_tokenizer(vocab_path)
    reference = transformers.BertTokenizerFast.from_pretrained(str(tmp_path))

    for sentence in sentences:
        expected = reference(sentence, add_special_tokens=False)["input_ids"]
        assert ours.encode(sentence).ids == expected, sentence
    assert ours.get_vocab() == reference.get_vocab()


def _tiny_mlm(**kwargs):
    return create_encoder_mlm_model(
        depth=2,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        inference=InferenceSGDNormClip(eta_infer=0.01, infer_steps=3, max_norm=5.0),
        use_attention_mask=True,
        dropout_rate=kwargs.pop("dropout_rate", 0.1),
        **kwargs,
    )


def _param_layout(params):
    return {
        name: {
            "weights": {k: v.shape for k, v in node.weights.items()},
            "biases": {k: v.shape for k, v in node.biases.items()},
        }
        for name, node in params.nodes.items()
    }


def _tiny_batch() -> dict:
    ids = np.array([[1, 5, 4, 7, 2, 0], [1, 9, 11, 4, 13, 2]])
    attention_mask = np.array([[1, 1, 1, 1, 1, 0], [1] * 6])
    loss_mask = np.array([[0, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]])
    return _batch(ids, attention_mask, loss_mask)


def test_default_encoder_is_unchanged():
    default = _tiny_mlm()
    explicit = _tiny_mlm(architecture="encoder", residual_identity=False)
    key = jax.random.PRNGKey(3)
    params = initialize_params(default, key)

    assert list(default.nodes) == list(explicit.nodes)
    assert set(default.edges) == set(explicit.edges)
    new_options = {"layer_norm", "norm_position", "pre_ln", "exact_gelu"}
    new_options |= {"post_ln", "ln_eps"}
    for name, node in default.nodes.items():
        config = dict(node.node_info.node_config)
        assert config == dict(explicit.nodes[name].node_info.node_config)
        assert not new_options & set(config)
    assert not any(name.endswith("_residual") for name in default.nodes)
    assert "mlm_transform" not in default.nodes
    assert _param_layout(params) == _param_layout(initialize_params(explicit, key))
    assert _param_layout(params)["L0_mlp1"] == {
        "weights": {"ln_gamma": (16,), "W_ff1": (16, 32)},
        "biases": {"ln_beta": (16,), "b_ff1": (32,)},
    }
    assert _param_layout(params)["L0_mlp2"] == {
        "weights": {"W_ff2": (32, 16)},
        "biases": {"b_ff2": (16,)},
    }
    assert _param_layout(params)["position_embeddings"] == {
        "weights": {"position_embeddings": (6, 16)},
        "biases": {},
    }
    assert "final_layernorm" in params.nodes
    assert all(
        default.nodes[name].node_info.scaling_config is not None
        for name in ("L0_mha", "L0_mlp1", "final_layernorm")
    )
    left = compute_forward_pass(params, default, _tiny_batch(), key)
    right = compute_forward_pass(params, explicit, _tiny_batch(), key)
    np.testing.assert_array_equal(
        left.nodes["mlm_head"].z_mu, right.nodes["mlm_head"].z_mu
    )


def test_node_options_at_defaults_leave_config_unchanged():
    common = {"shape": (6, 16), "name": "n", "embed_dim": 16}
    assert dict(
        MhaResidualNode(
            **common, num_heads=4, norm_position="pre", ln_eps=1e-5
        )._extra_config
    ) == dict(MhaResidualNode(**common, num_heads=4)._extra_config)
    assert dict(
        LnMlp1Node(
            **common, ff_dim=32, pre_ln=True, exact_gelu=False, ln_eps=1e-5
        )._extra_config
    ) == dict(LnMlp1Node(**common, ff_dim=32)._extra_config)
    assert dict(
        Mlp2ResidualNode(**common, ff_dim=32, post_ln=False, ln_eps=1e-5)._extra_config
    ) == dict(Mlp2ResidualNode(**common, ff_dim=32)._extra_config)
    assert dict(
        AddPositionEmbeddingNode(
            **common, seq_len=6, layer_norm=False, ln_eps=1e-5
        )._extra_config
    ) == dict(AddPositionEmbeddingNode(**common, seq_len=6)._extra_config)
    with pytest.raises(ValueError, match="norm_position"):
        MhaResidualNode(**common, num_heads=4, norm_position="middle")


def test_bert_builder_layout_and_argument_checks():
    structure = _tiny_mlm(architecture="bert")
    params = initialize_params(structure, jax.random.PRNGKey(0))

    assert "final_layernorm" not in structure.nodes
    assert {
        "L1_mlp2->mlm_transform:in",
        "mlm_transform->mlm_head:in",
        "L0_mha->L0_mlp2:residual",
    } <= set(structure.edges)
    assert not any(name.endswith("_residual") for name in structure.nodes)
    assert all(
        node.node_info.scaling_config is None for node in structure.nodes.values()
    )
    assert set(params.nodes["L0_mlp1"].weights) == {"W_ff1"}
    assert set(params.nodes["L0_mlp2"].weights) == {"W_ff2", "ln_gamma"}
    assert set(params.nodes["position_embeddings"].biases) == {"ln_beta"}
    assert structure.nodes["L0_mha"].node_info.node_config["ln_eps"] == 1e-12
    assert structure.nodes["mlm_transform"].node_info.node_config["ln_eps"] == 1e-12
    custom = _tiny_mlm(architecture="bert", ln_eps=1e-6)
    assert custom.nodes["L1_mlp2"].node_info.node_config["ln_eps"] == 1e-6
    with pytest.raises(ValueError, match="architecture"):
        _tiny_mlm(architecture="roberta")
    with pytest.raises(ValueError, match="ln_eps"):
        _tiny_mlm(ln_eps=1e-6)


def test_bert_residual_identity_is_optional_and_bert_only():
    with_identity = _tiny_mlm(architecture="bert", residual_identity=True)
    direct = _tiny_mlm(architecture="bert")
    key = jax.random.PRNGKey(0)

    assert {"L0_residual", "L1_residual"} <= set(with_identity.nodes)
    assert {"L0_mha->L0_residual:in", "L0_residual->L0_mlp2:residual"} <= set(
        with_identity.edges
    )
    assert "L0_mha->L0_mlp2:residual" not in with_identity.edges
    with_params = initialize_params(with_identity, key)
    assert with_params.nodes["L0_residual"] == NodeParams(weights={}, biases={})
    direct_params = initialize_params(direct, key)
    assert {
        name: layout
        for name, layout in _param_layout(with_params).items()
        if not name.endswith("_residual")
    } == _param_layout(direct_params)
    shared = GraphParams(
        nodes={
            name: direct_params.nodes.get(name, node)
            for name, node in with_params.nodes.items()
        }
    )
    batch = _tiny_batch()
    np.testing.assert_array_equal(
        compute_forward_pass(direct_params, direct, batch, key).nodes["mlm_head"].z_mu,
        compute_forward_pass(shared, with_identity, batch, key).nodes["mlm_head"].z_mu,
    )
    with pytest.raises(ValueError, match="residual_identity"):
        _tiny_mlm(residual_identity=True)


def test_bert_pc_head_gradient_sums_what_backprop_averages():
    structure = create_encoder_mlm_model(
        depth=2,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        inference=InferenceSGDNormClip(eta_infer=0.0, infer_steps=2, max_norm=5.0),
        use_attention_mask=True,
        architecture="bert",
    )
    key = jax.random.PRNGKey(0)
    params = initialize_params(structure, key)
    batch = _tiny_batch()

    pc_grads, *_ = get_graph_param_gradient_autoregressive(
        params, batch, structure, key
    )
    bp_grads = jax.grad(
        lambda p: compute_loss_autoregressive(p, structure, batch, key)[0]
    )(params)

    masked = float(jnp.sum(batch["loss_mask"]))
    for collection, name in (("weights", "W_out"), ("biases", "b_out")):
        np.testing.assert_allclose(
            getattr(pc_grads.nodes["mlm_head"], collection)[name] / masked,
            getattr(bp_grads.nodes["mlm_head"], collection)[name],
            rtol=1e-4,
            atol=1e-7,
        )
    hidden = jax.tree_util.tree_leaves(pc_grads.nodes["mlm_transform"])
    assert all(float(jnp.max(jnp.abs(leaf))) == 0.0 for leaf in hidden)


@pytest.fixture
def converted_bert(convert, demo, tmp_path):
    """Params converted from a one-layer synthetic BERT with a tiny-corpus vocab."""
    pytest.importorskip("tokenizers")
    vocab_path = tmp_path / "vocab.txt"
    words = sorted({w for text in demo.TINY_TRAIN_TEXTS for w in demo._words(text)})
    _write_vocab(convert, vocab_path, words)
    tokenizer = convert.build_wordpiece_tokenizer(vocab_path)
    config = {
        **TINY_HF_CONFIG,
        "vocab_size": tokenizer.get_vocab_size(),
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
    }
    state = _synthetic_state_dict(np.random.default_rng(0), **config)
    state = {key: 0.1 * value for key, value in state.items()}
    model_config = convert.bert_model_config(config, seq_len=16)
    params = convert.graph_params_for(
        convert.convert_bert_state_dict(state, config=config, seq_len=16),
        convert.build_bert_structure(model_config),
    )
    archive = convert.write_bert_params(
        tmp_path / "bert.npz", params, tokenizer, model_config
    )
    assert (tmp_path / "bert.tokenizer.json").read_text() == tokenizer.to_str()
    return archive, params


def test_written_bert_params_load_with_metadata(convert, converted_bert):
    archive, params = converted_bert
    metadata = read_params_metadata(archive)
    structure = convert.build_bert_structure(metadata["config"])

    loaded, loaded_metadata = load_params(archive, structure)

    assert metadata["config"]["architecture"] == "bert"
    assert metadata["config"]["ln_eps"] == 1e-12
    assert loaded_metadata["vocab"]["[MASK]"] == 4
    for name, node in params.nodes.items():
        for key, value in {**node.weights, **node.biases}.items():
            stored = {**loaded.nodes[name].weights, **loaded.nodes[name].biases}
            np.testing.assert_array_equal(stored[key], value)


@pytest.mark.parametrize("mode", ["backprop", "pc"])
def test_bert_demo_trains_one_step_from_converted_params(
    demo, converted_bert, monkeypatch, tmp_path, mode
):
    archive, params = converted_bert
    real = demo.TRAINERS[mode]
    started = []

    def spy(start, *args, **kwargs):
        started.append(start)
        return real(start, *args, **kwargs)

    monkeypatch.setitem(demo.TRAINERS, mode, spy)
    out = tmp_path / "trained.npz"

    metrics = demo.main(
        DEMO_SMALL
        + ["--mode", mode, "--num_epochs", "0.2", "--architecture", "bert"]
        + ["--init_params", str(archive), "--tokenizer", "checkpoint"]
        + ["--save_params", str(out)]
    )

    assert np.isfinite(metrics["masked_loss"])
    (start,) = started
    for name, node in params.nodes.items():
        for key, value in node.weights.items():
            np.testing.assert_array_equal(start.nodes[name].weights[key], value)
    saved = read_params_metadata(out)
    assert saved["config"]["architecture"] == "bert"
    assert saved["config"]["ln_eps"] == 1e-12
    assert saved["config"]["residual_identity"] is False
    assert saved["tokenizer_json"] == read_params_metadata(archive)["tokenizer_json"]
    structure = demo.build_model(
        demo.parse_args(DEMO_SMALL + ["--architecture", "bert"]),
        vocab_size=len(saved["vocab"]),
    )
    trained, _ = load_params(out, structure)
    leaves = jax.tree_util.tree_leaves(trained)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_bert_demo_rejects_architecture_and_ln_eps_mismatch(
    demo, converted_bert, tmp_path
):
    archive, _ = converted_bert
    argv = DEMO_SMALL + ["--num_epochs", "0.2", "--init_params", str(archive)]
    argv += ["--tokenizer", "checkpoint"]

    with pytest.raises(ValueError, match="architecture 'bert' does not match"):
        demo.main(argv)

    sidecar = Path(archive).with_suffix(".json")
    data = json.loads(sidecar.read_text())
    data["metadata"]["config"]["ln_eps"] = 1e-6
    sidecar.write_text(json.dumps(data))
    bert_argv = argv + ["--architecture", "bert"]
    assert demo.resolve_ln_eps(demo.parse_args(bert_argv)) == 1e-6
    with pytest.raises(
        ValueError, match="ln_eps 1e-06 does not match this run's ln_eps 1e-12"
    ):
        demo.main(bert_argv + ["--ln_eps", "1e-12"])
    with pytest.raises(
        ValueError, match="seq_len 16 does not match this run's seq_len 8"
    ):
        demo.main(bert_argv + ["--seq_len", "8"])


def test_bert_demo_flag_checks(demo, tmp_path):
    with pytest.raises(SystemExit):
        demo.parse_args(["--tokenizer", "checkpoint"])
    with pytest.raises(SystemExit):
        demo.parse_args(["--ln_eps", "1e-6"])
    args = demo.parse_args(DEMO_SMALL + ["--architecture", "bert"])
    assert demo.resolve_ln_eps(args) == 1e-12
    assert "mlm_transform" in demo.build_model(args, vocab_size=30).nodes


def test_bert_demo_residual_identity_flag(demo, converted_bert, tmp_path):
    archive, _ = converted_bert
    with pytest.raises(SystemExit):
        demo.parse_args(["--residual_identity"])
    args = demo.parse_args(
        DEMO_SMALL + ["--architecture", "bert", "--residual_identity"]
    )
    assert "L0_residual" in demo.build_model(args, vocab_size=30).nodes
    out = tmp_path / "trained.npz"

    demo.main(
        DEMO_SMALL
        + ["--num_epochs", "0.2", "--architecture", "bert", "--residual_identity"]
        + ["--init_params", str(archive), "--tokenizer", "checkpoint"]
        + ["--save_params", str(out)]
    )

    assert read_params_metadata(out)["config"]["residual_identity"] is True

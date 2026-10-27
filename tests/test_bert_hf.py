"""Tests for the post-LN BERT graph and the Hugging Face BERT converter."""

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.bert_demo import (
    TINY_MLM_CORPUS,
    _inherit_checkpoint_ln_eps,
    _param_shapes,
    load_checkpoint,
    parse_args,
    run_pretrain,
    summarize_init_diagnostics,
)
from examples.convert_hf_bert import (
    BERT_SPECIAL_TOKENS,
    bert_model_config,
    build_bert_structure,
    build_wordpiece_tokenizer,
    check_loading_info,
    convert_bert_state_dict,
    graph_params_for,
    write_bert_checkpoint,
)
from fabricpc.core.inference import InferenceSGDNormClip, InferenceSpikingPrecision
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.encoder_transformer import (
    AddPositionEmbeddingNode,
    copy_encoder_params,
    create_encoder_mlm_model,
    create_encoder_sentiment_classifier,
)
from fabricpc.nodes.transformer_v2 import LnMlp1Node, MhaResidualNode, Mlp2ResidualNode
from fabricpc.training.train_backprop import (
    _loss_and_grad_backprop_mlm_no_update,
    compute_forward_pass,
)
from fabricpc.training.train_mlm import (
    get_graph_param_gradient_pcn_mlm,
    mean_pc_gradients,
)
from tests.test_bert_pc_mlm import (
    _cosines,
    _max_free_depth,
    _relative_errors,
    _two_sample_batch,
)

TINY_HF_CONFIG = {
    "vocab_size": 50,
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 32,
    "max_position_embeddings": 32,
}


def _synthetic_state_dict(
    rng: np.random.Generator,
    *,
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    intermediate_size: int,
    max_position_embeddings: int,
    **_unused,
) -> dict[str, np.ndarray]:
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


def _hf_logits(model, ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    torch = pytest.importorskip("torch")
    with torch.no_grad():
        return model(
            input_ids=torch.tensor(ids), attention_mask=torch.tensor(attention_mask)
        ).logits.numpy()


def _our_logits(params, structure, ids: np.ndarray, attention_mask: np.ndarray):
    batch = {
        "x": jnp.asarray(ids, dtype=jnp.int32),
        "attention_mask": jnp.asarray(attention_mask, dtype=jnp.float32),
        "loss_mask": jnp.asarray(attention_mask, dtype=jnp.float32),
    }
    state = compute_forward_pass(params, structure, batch, jax.random.PRNGKey(0))
    head = params.nodes["mlm_head"]
    hidden = state.nodes["mlm_transform"].z_mu
    logits = hidden @ head.weights["W_out"] + head.biases["b_out"]
    return np.asarray(logits), np.asarray(state.nodes["mlm_head"].z_mu)


def test_converted_bert_logits_match_hugging_face():
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
    converted = convert_bert_state_dict(
        state_dict, config=config.to_dict(), seq_len=seq_len
    )
    structure = build_bert_structure(bert_model_config(config.to_dict(), seq_len))
    params = graph_params_for(converted, structure)
    ids = np.array([[2, 5, 7, 9, 11, 3, 0, 0], [2, 13, 17, 19, 23, 29, 31, 3]])
    attention_mask = np.array([[1, 1, 1, 1, 1, 1, 0, 0], [1] * 8])

    expected = _hf_logits(model, ids, attention_mask)
    logits, probs = _our_logits(params, structure, ids, attention_mask)

    valid = attention_mask.astype(bool)
    assert np.abs(logits - expected)[valid].max() < 1e-5
    expected_log_probs = torch.log_softmax(torch.tensor(expected), dim=-1).numpy()
    assert np.abs(np.log(probs) - expected_log_probs)[valid].max() < 1e-5


def test_convert_maps_and_transposes_synthetic_state_dict():
    config = {**TINY_HF_CONFIG, "hidden_size": 8, "intermediate_size": 12}
    config["num_attention_heads"] = 2
    state = _synthetic_state_dict(np.random.default_rng(0), **config)

    params = convert_bert_state_dict(state, config=config, seq_len=5)

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
    structure = build_bert_structure(bert_model_config(config, 5))
    graph_params_for(params, structure)


def test_convert_rejects_missing_unknown_and_untied_keys():
    state = _synthetic_state_dict(np.random.default_rng(0), **TINY_HF_CONFIG)
    convert_bert_state_dict(
        {**state, "bert.pooler.dense.weight": np.zeros((16, 16))},
        config=TINY_HF_CONFIG,
        seq_len=8,
    )

    missing = dict(state)
    del missing["bert.encoder.layer.1.output.LayerNorm.bias"]
    with pytest.raises(ValueError, match="missing state dict keys"):
        convert_bert_state_dict(missing, config=TINY_HF_CONFIG, seq_len=8)
    with pytest.raises(ValueError, match="unmapped state dict keys"):
        convert_bert_state_dict(
            {**state, "bert.encoder.layer.2.output.dense.bias": np.zeros(16)},
            config=TINY_HF_CONFIG,
            seq_len=8,
        )
    untied = {**state, "cls.predictions.decoder.bias": np.ones(50, np.float32)}
    with pytest.raises(ValueError, match="decoder.bias"):
        convert_bert_state_dict(untied, config=TINY_HF_CONFIG, seq_len=8)
    with pytest.raises(ValueError, match="seq_len"):
        convert_bert_state_dict(state, config=TINY_HF_CONFIG, seq_len=33)


def _write_vocab(path, words) -> None:
    path.write_text("\n".join([*BERT_SPECIAL_TOKENS, *words]) + "\n")


def test_wordpiece_tokenizer_matches_bert_tokenizer_fast(tmp_path):
    transformers = pytest.importorskip("transformers")
    vocab_path = tmp_path / "vocab.txt"
    words = ["the", "cafe", "un", "##aff", "##able", "is", "a", "dog", ",", ".", "!"]
    _write_vocab(vocab_path, [*words, "run", "##ning", "中", "'", "s"])
    sentences = [
        "The café is unaffable!",
        "a DOG, running.",
        "the dog's [MASK] is unknownword",
        "  multiple   spaces\tand\nnewlines 中文 ",
    ]

    ours = build_wordpiece_tokenizer(vocab_path)
    reference = transformers.BertTokenizerFast.from_pretrained(str(tmp_path))

    for sentence in sentences:
        expected = reference(sentence, add_special_tokens=False)["input_ids"]
        assert ours.encode(sentence).ids == expected, sentence
    assert ours.get_vocab() == reference.get_vocab()


def _spiking_bert(infer_steps: int, dropout_rate: float):
    return create_encoder_mlm_model(
        depth=2,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        inference=InferenceSpikingPrecision(
            eta_infer=0.0, infer_steps=infer_steps, max_norm=5.0
        ),
        use_attention_mask=True,
        dropout_rate=dropout_rate,
        architecture="bert",
    )


def test_spiking_zil_matches_backprop_on_bert_graph_with_dropout():
    with jax.enable_x64(True):
        structure = _spiking_bert(_max_free_depth(_spiking_bert(1, 0.3)), 0.3)
        key = jax.random.PRNGKey(0)
        params = initialize_params(structure, key)
        batch = _two_sample_batch()
        pc_grads, pc_grads_metrics, *_ = get_graph_param_gradient_pcn_mlm(
            params, batch, structure, key, forward_updates="initial"
        )
        pc_grads = mean_pc_gradients(pc_grads, pc_grads_metrics["num_masked_tokens"])
        _loss, bp_grads, _count = _loss_and_grad_backprop_mlm_no_update(
            params, batch, structure, key
        )
        assert jax.tree_util.tree_leaves(pc_grads)[0].dtype == jnp.float64
        errors = _relative_errors(pc_grads, bp_grads)
        cosines = _cosines(pc_grads, bp_grads)

    assert "mlm_transform" in errors
    assert all(error < 1e-6 for error in errors.values()), errors
    assert all(cosine > 1.0 - 1e-10 for cosine in cosines.values()), cosines


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
        dropout_rate=0.1,
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


def test_default_encoder_is_unchanged():
    default = _tiny_mlm()
    explicit = _tiny_mlm(architecture="encoder")
    key = jax.random.PRNGKey(3)
    params = initialize_params(default, key)
    batch = _two_sample_batch()

    assert list(default.nodes) == list(explicit.nodes)
    for name, node in default.nodes.items():
        config = dict(node.node_info.node_config)
        assert config == dict(explicit.nodes[name].node_info.node_config)
        assert not {"layer_norm", "norm_position", "pre_ln", "exact_gelu"} & set(config)
        assert not {"post_ln", "ln_eps"} & set(config)
    assert _param_layout(params) == _param_layout(initialize_params(explicit, key))
    assert _param_layout(params)["L0_mlp1"] == {
        "weights": {"ln_gamma": (16,), "W_ff1": (16, 32)},
        "biases": {"ln_beta": (16,), "b_ff1": (32,)},
    }
    assert "final_layernorm" in params.nodes
    assert all(
        default.nodes[name].node_info.scaling_config is not None
        for name in ("L0_mha", "L0_mlp1", "final_layernorm")
    )
    left = compute_forward_pass(params, default, batch, key)
    right = compute_forward_pass(params, explicit, batch, key)
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
        LnMlp1Node(**common, ff_dim=32, pre_ln=True, exact_gelu=False)._extra_config
    ) == dict(LnMlp1Node(**common, ff_dim=32)._extra_config)
    assert dict(
        Mlp2ResidualNode(**common, ff_dim=32, post_ln=False)._extra_config
    ) == dict(Mlp2ResidualNode(**common, ff_dim=32)._extra_config)
    assert dict(
        AddPositionEmbeddingNode(**common, seq_len=6, layer_norm=False)._extra_config
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
        "L0_mha->L0_residual:in",
        "L0_residual->L0_mlp2:residual",
    } <= set(structure.edges)
    assert "L0_mha->L0_mlp2:residual" not in structure.edges
    assert params.nodes["L0_residual"].weights == {}
    assert all(
        node.node_info.scaling_config is None for node in structure.nodes.values()
    )
    assert set(params.nodes["L0_mlp1"].weights) == {"W_ff1"}
    assert set(params.nodes["L0_mlp2"].weights) == {"W_ff2", "ln_gamma"}
    assert set(params.nodes["position_embeddings"].biases) == {"ln_beta"}
    assert structure.nodes["L0_mha"].node_info.node_config["ln_eps"] == 1e-12
    with pytest.raises(ValueError, match="architecture"):
        _tiny_mlm(architecture="roberta")
    with pytest.raises(ValueError, match="ln_eps"):
        _tiny_mlm(ln_eps=1e-6)


def test_copy_encoder_params_from_bert_mlm_to_bert_classifier():
    source = initialize_params(_tiny_mlm(architecture="bert"), jax.random.PRNGKey(0))
    classifier = create_encoder_sentiment_classifier(
        depth=2,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        num_classes=2,
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=5.0),
        architecture="bert",
    )
    target = initialize_params(classifier, jax.random.PRNGKey(1))

    copied = copy_encoder_params(source, target)

    np.testing.assert_array_equal(
        copied.nodes["L1_mlp2"].weights["ln_gamma"],
        source.nodes["L1_mlp2"].weights["ln_gamma"],
    )
    assert "mlm_transform" not in copied.nodes


def _tokenized_bert_fixture(tmp_path, seq_len: int = 6):
    pytest.importorskip("tokenizers")
    vocab_path = tmp_path / "vocab.txt"
    words = sorted({word for text in TINY_MLM_CORPUS for word in text.split()})
    _write_vocab(vocab_path, words)
    tokenizer = build_wordpiece_tokenizer(vocab_path)
    sep = tokenizer.token_to_id("[SEP]")
    ids = [i for text in TINY_MLM_CORPUS for i in [*tokenizer.encode(text).ids, sep]]
    token_path = tmp_path / "tokens.npy"
    np.save(token_path, np.asarray(ids, dtype=np.int32))
    tokenizer_path = tmp_path / "bert.tokenizer.json"

    config = {
        **TINY_HF_CONFIG,
        "vocab_size": tokenizer.get_vocab_size(),
        "num_hidden_layers": 1,
    }
    state = _synthetic_state_dict(np.random.default_rng(0), **config)
    state = {key: 0.1 * value for key, value in state.items()}
    model_config = bert_model_config(config, seq_len)
    params = graph_params_for(
        convert_bert_state_dict(state, config=config, seq_len=seq_len),
        build_bert_structure(model_config),
    )
    checkpoint = tmp_path / "bert.json"
    write_bert_checkpoint(checkpoint, params, tokenizer, model_config)
    assert tokenizer_path.exists()
    return checkpoint, token_path, tokenizer_path


def _bert_pretrain_argv(tmp_path, token_path, tokenizer_path, *extra):
    return [
        "--stage",
        "pretrain",
        "--data_source",
        "tokenized",
        "--token_path",
        str(token_path),
        "--tokenizer_path",
        str(tokenizer_path),
        "--seq_len",
        "6",
        "--embed_dim",
        "16",
        "--num_heads",
        "4",
        "--mlp_dim",
        "32",
        "--depth",
        "1",
        "--batch_size",
        "2",
        "--max_steps",
        "1",
        "--pc_eta_infer",
        "0.01",
        "--validation_interval",
        "0",
        "--log_interval",
        "0",
        *extra,
    ]


@pytest.mark.parametrize("mode", ["backprop", "pc"])
def test_bert_pretrain_runs_from_converted_checkpoint(tmp_path, capsys, mode):
    checkpoint, token_path, tokenizer_path = _tokenized_bert_fixture(tmp_path)
    expected = summarize_init_diagnostics(load_checkpoint(str(checkpoint))["params"])
    out_path = tmp_path / "out.json"

    run_pretrain(
        parse_args(
            _bert_pretrain_argv(
                tmp_path,
                token_path,
                tokenizer_path,
                "--mode",
                mode,
                "--architecture",
                "bert",
                "--init_checkpoint",
                str(checkpoint),
                "--checkpoint",
                str(out_path),
            )
        )
    )

    out = capsys.readouterr().out
    assert f"param_std={expected['param_std']:.6f}" in out
    assert "done stage=pretrain step=1" in out
    saved = load_checkpoint(str(out_path))
    assert saved["config"]["architecture"] == "bert"
    leaves = jax.tree_util.tree_leaves(saved["params"])
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_init_checkpoint_rejects_architecture_and_seq_len_mismatch(tmp_path):
    checkpoint, token_path, tokenizer_path = _tokenized_bert_fixture(tmp_path)
    argv = _bert_pretrain_argv(
        tmp_path, token_path, tokenizer_path, "--init_checkpoint", str(checkpoint)
    )

    with pytest.raises(ValueError, match="architecture 'bert' does not match"):
        run_pretrain(parse_args(argv))

    argv = [*argv, "--architecture", "bert"]
    argv[argv.index("--seq_len") + 1] = "5"
    with pytest.raises(ValueError, match="seq_len 6 does not match"):
        run_pretrain(parse_args(argv))


def test_check_loading_info_rejects_weights_hugging_face_would_initialise():
    complete = {
        "missing_keys": [],
        "unexpected_keys": ["bert.pooler.dense.weight", "cls.seq_relationship.bias"],
        "mismatched_keys": set(),
        "error_msgs": [],
    }
    check_loading_info(complete)
    check_loading_info({**complete, "unexpected_keys": ["pooler.dense.bias"]})

    with pytest.raises(ValueError, match="missing_keys"):
        check_loading_info({**complete, "missing_keys": ["cls.predictions.bias"]})
    with pytest.raises(ValueError, match="unexpected_keys"):
        check_loading_info({**complete, "unexpected_keys": ["bert.encoder.extra"]})
    with pytest.raises(ValueError, match="mismatched_keys"):
        check_loading_info(
            {**complete, "mismatched_keys": {("bert.embeddings.x", (1,), (2,))}}
        )


def test_bert_residual_identity_is_optional_and_bert_only():
    kwargs = dict(
        depth=2,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=20,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=2, max_norm=5.0),
        use_attention_mask=True,
    )
    with_identity = create_encoder_mlm_model(**kwargs, architecture="bert")
    direct = create_encoder_mlm_model(
        **kwargs, architecture="bert", residual_identity=False
    )

    assert {"L0_residual", "L1_residual"} <= set(with_identity.nodes)
    assert not any(name.endswith("_residual") for name in direct.nodes)
    key = jax.random.PRNGKey(0)
    assert _param_shapes(initialize_params(direct, key)) == _param_shapes(
        initialize_params(with_identity, key)
    )
    with pytest.raises(ValueError, match="residual_identity"):
        create_encoder_mlm_model(**kwargs, residual_identity=True)


def test_bert_ln_eps_is_inherited_from_checkpoint_and_mismatch_rejected(tmp_path):
    checkpoint, token_path, tokenizer_path = _tokenized_bert_fixture(tmp_path)
    data = json.loads(checkpoint.read_text())
    data["config"]["ln_eps"] = 1e-6
    checkpoint.write_text(json.dumps(data))
    argv = _bert_pretrain_argv(
        tmp_path,
        token_path,
        tokenizer_path,
        "--architecture",
        "bert",
        "--init_checkpoint",
        str(checkpoint),
    )

    args = parse_args(argv)
    _inherit_checkpoint_ln_eps(args)
    assert args.ln_eps == 1e-6

    with pytest.raises(ValueError, match="ln_eps 1e-06 does not match"):
        run_pretrain(parse_args([*argv, "--ln_eps", "1e-12"]))

"""Smoke tests for examples/bert_demo.py (MLM pretraining demo)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DEMO_PATH = Path(__file__).resolve().parents[1] / "examples" / "bert_demo.py"
SMALL = ["--depth", "1", "--embed_dim", "16", "--num_heads", "2", "--mlp_dim", "32"]


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("bert_demo_under_test", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_corpus_pretraining_reports_validation_metrics(demo, capsys):
    metrics = demo.main(SMALL + ["--num_epochs", "2"])

    lines = capsys.readouterr().out.splitlines()
    epoch_lines = [line for line in lines if line.startswith("epoch ")]
    assert len(epoch_lines) == 2
    assert all("train_batch_mean_loss=" in line for line in epoch_lines)
    assert "epoch 2/2" in epoch_lines[-1]
    assert not any("Perplexity" in line for line in lines)

    assert metrics["num_masked_tokens"] > 0
    assert 0.0 <= metrics["masked_accuracy"] <= 1.0
    assert metrics["masked_loss"] > 0.0


def test_bpe_pretraining_from_text_files(demo, tmp_path):
    pytest.importorskip("tokenizers")
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"
    train_file.write_text("\n".join(demo.TINY_TRAIN_TEXTS))
    val_file.write_text("\n".join(demo.TINY_VAL_TEXTS))

    metrics = demo.main(
        SMALL
        + ["--num_epochs", "1", "--tokenizer", "bpe"]
        + ["--vocab_size", "80"]
        + ["--train_file", str(train_file), "--val_file", str(val_file)]
    )

    assert metrics["num_masked_tokens"] > 0


def test_train_and_val_files_must_be_given_together(demo, tmp_path):
    with pytest.raises(SystemExit):
        demo.parse_args(["--train_file", str(tmp_path / "train.txt")])


def test_word_tokenizer_maps_unseen_validation_words_to_unk(demo):
    train_ids, val_ids, vocab = demo.word_tokenize(["the cat sat"], ["the dog sat"])

    assert "dog" not in vocab
    assert vocab["[UNK]"] in val_ids.tolist()
    assert train_ids[-1] == vocab["[SEP]"]


def test_too_little_text_fails_clearly(demo):
    args = demo.parse_args(SMALL)
    train_ids, _, vocab = demo.word_tokenize(["a short line"], ["a short line"])

    with pytest.raises(ValueError, match="no full batch"):
        demo.make_loader(train_ids, vocab, args, seed=0, shuffle=False)


@pytest.mark.parametrize("warmup_fraction", [-0.1, 1.0])
def test_optimizer_rejects_invalid_warmup_fraction(demo, warmup_fraction):
    args = SimpleNamespace(
        lr=1e-3, weight_decay=0.0, warmup_fraction=warmup_fraction, grad_clip=1.0
    )

    with pytest.raises(ValueError, match="warmup_fraction"):
        demo.build_optimizer(args, total_steps=10)


def test_optimizer_warms_up_then_decays(demo):
    import jax.numpy as jnp
    import optax

    args = SimpleNamespace(lr=1.0, weight_decay=0.0, warmup_fraction=0.5, grad_clip=0)
    optimizer = demo.build_optimizer(args, total_steps=4)
    params = {"w": jnp.zeros(())}
    state = optimizer.init(params)
    steps = []
    for _ in range(4):
        updates, state = optimizer.update({"w": jnp.ones(())}, state, params)
        steps.append(float(-updates["w"]))

    assert steps[0] == pytest.approx(0.0, abs=1e-6)
    assert steps[1] < steps[2]
    assert steps[3] < steps[2]


@pytest.mark.parametrize(
    "num_epochs, expected", [(2, 10), (1.5, 7), (1.7, 9), (0.2, 1)]
)
def test_training_steps_match_the_loop_truncation(demo, num_epochs, expected):
    assert demo.count_training_steps(num_epochs, batches_per_epoch=5) == expected


@pytest.mark.parametrize("num_epochs", [1.1, 0, -1])
def test_rejects_epochs_the_loop_cannot_run(demo, num_epochs):
    with pytest.raises(ValueError, match="num_epochs"):
        demo.count_training_steps(num_epochs, batches_per_epoch=3)


def test_pc_mode_pretraining_reports_validation_metrics(demo, capsys):
    metrics = demo.main(SMALL + ["--mode", "pc", "--num_epochs", "1"])

    assert metrics["num_masked_tokens"] > 0
    out = capsys.readouterr().out
    assert "epoch 1/1 train_pc_inferred_loss=" in out
    assert "train_masked_loss=" not in out


def test_pc_flags_configure_inference_and_output_energy(demo):
    args = demo.parse_args(
        SMALL
        + ["--mode", "pc", "--pc_infer_steps", "7", "--pc_eta_infer", "0.02"]
        + ["--pc_max_norm", "3.0", "--pc_output_energy_scale", "2.5"]
    )

    structure = demo.build_model(args, vocab_size=30)

    inference = structure.config["inference"].config
    assert inference["infer_steps"] == 7
    assert inference["eta_infer"] == pytest.approx(0.02)
    assert inference["max_norm"] == pytest.approx(3.0)
    head_config = structure.nodes["mlm_head"].node_info.node_config
    assert head_config["output_energy_scale"] == pytest.approx(2.5)
    assert demo.TRAINERS[args.mode] is demo.train_autoregressive


def test_ab_comparison_pretrains_both_modes_per_trial(demo):
    pytest.importorskip("scipy")

    results = demo.main(SMALL + ["--ab_trials", "2", "--num_epochs", "1"])

    assert (results.arm_a_name, results.arm_b_name) == ("backprop", "pc")
    assert results.metric == "masked_loss"
    assert len(results.arm_a_trials) == len(results.arm_b_trials) == 2
    values_a = [trial.metric_value for trial in results.arm_a_trials]
    values_b = [trial.metric_value for trial in results.arm_b_trials]
    assert all(np.isfinite(v) and v > 0 for v in values_a + values_b)
    assert values_a != values_b


def test_ab_comparison_fails_before_training_without_scipy(demo, monkeypatch):
    real_find_spec = demo.importlib.util.find_spec
    monkeypatch.setattr(
        demo.importlib.util,
        "find_spec",
        lambda name: None if name == "scipy" else real_find_spec(name),
    )

    with pytest.raises(SystemExit):
        demo.parse_args(["--ab_trials", "2"])


@pytest.mark.parametrize("infer_steps, warns", [(2, True), (10, False)])
def test_warns_when_pc_inference_cannot_reach_early_layers(
    demo, capsys, infer_steps, warns
):
    args = demo.parse_args(
        SMALL + ["--mode", "pc", "--pc_infer_steps", str(infer_steps)]
    )
    train_ids, val_ids, vocab = demo.load_token_streams(args)
    loader, _ = demo.make_loaders(args, train_ids, val_ids, vocab, seed=0)
    structure = demo.build_model(args, vocab_size=len(vocab))
    key = demo.jax.random.PRNGKey(0)
    params = demo.initialize_params(structure, key)
    state = loader.state_dict()

    unreached = demo.warn_unreached_pc_nodes(structure, params, loader, key)

    assert ("token_embeddings" in unreached) is warns
    assert ("warning:" in capsys.readouterr().err) is warns
    assert loader.state_dict() == state

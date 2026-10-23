"""Tests for fixed, pre-masked MLM validation sets."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.experiments.domain_forgetting.validation import (
    VALID_ARRAY_NAMES,
    FixedMlmBatches,
    evaluate_fixed_set,
    load_validation_arrays,
    materialize_validation_set,
    pack_windows,
    save_validation_arrays,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.training.train_backprop import compute_forward_pass
from fabricpc.utils.data.dataloader import MlmTextBatcher

VOCAB = 24
SPECIAL = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}


def _docs(seed: int = 0, count: int = 30) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(5, VOCAB, size=int(rng.integers(2, 9))) for _ in range(count)]


def _windows(max_windows: int | None = 12) -> np.ndarray:
    return pack_windows(
        _docs(),
        seq_len=6,
        cls_token_id=SPECIAL["[CLS]"],
        sep_token_id=SPECIAL["[SEP]"],
        max_windows=max_windows,
    )


def _materialize(windows: np.ndarray) -> dict[str, np.ndarray]:
    return materialize_validation_set(
        windows,
        mask_token_id=SPECIAL["[MASK]"],
        vocab_size=VOCAB,
        special_token_ids=set(SPECIAL.values()),
        mask_prob=0.3,
    )


def _tiny_bert(dropout_rate: float = 0.0):
    return create_encoder_mlm_model(
        depth=1,
        embed_dim=16,
        num_heads=4,
        mlp_dim=32,
        seq_len=6,
        vocab_size=VOCAB,
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=5.0),
        use_attention_mask=True,
        dropout_rate=dropout_rate,
        architecture="bert",
    )


def test_pack_windows_matches_mlm_text_batcher_layout():
    docs = _docs()
    stream = np.concatenate(
        [
            x
            for i, doc in enumerate(docs)
            for x in ([np.array([3])] if i else []) + [doc]
        ]
    )
    windows = _windows(max_windows=None)
    assert windows.dtype == np.int32
    assert windows.shape == (len(stream) // 5, 6)
    batcher = MlmTextBatcher(
        token_ids=stream,
        seq_len=6,
        batch_size=1,
        pad_token_id=SPECIAL["[PAD]"],
        mask_token_id=SPECIAL["[MASK]"],
        vocab_size=VOCAB,
        special_token_ids=set(SPECIAL.values()),
        cls_token_id=SPECIAL["[CLS]"],
        mask_prob=0.0,
        shuffle=False,
    )
    rows = [batch["x"][0] for batch in batcher][: len(windows)]
    np.testing.assert_array_equal(np.stack(rows), windows)
    capped = _windows(max_windows=4)
    np.testing.assert_array_equal(capped, windows[:4])


def test_pack_windows_rejects_bad_arguments():
    with pytest.raises(ValueError, match="seq_len"):
        pack_windows([], seq_len=1, cls_token_id=2, sep_token_id=3)
    with pytest.raises(ValueError, match="max_windows"):
        pack_windows([], seq_len=6, cls_token_id=2, sep_token_id=3, max_windows=0)
    assert pack_windows([], seq_len=6, cls_token_id=2, sep_token_id=3).shape == (0, 6)


def test_materialized_masks_are_fixed_and_bert_style():
    windows = _windows()
    first, second = _materialize(windows), _materialize(windows)
    for name in VALID_ARRAY_NAMES:
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_array_equal(first["input_ids"], windows)
    np.testing.assert_array_equal(first["labels"], windows)
    assert first["attention_mask"].all()
    assert not first["loss_mask"][:, 0].any()
    assert first["loss_mask"].any()
    unmasked = ~first["loss_mask"]
    np.testing.assert_array_equal(
        first["masked_input_ids"][unmasked], windows[unmasked]
    )
    other = materialize_validation_set(
        windows,
        mask_token_id=SPECIAL["[MASK]"],
        vocab_size=VOCAB,
        special_token_ids=set(SPECIAL.values()),
        mask_prob=0.3,
        seed=1,
    )
    assert not np.array_equal(other["loss_mask"], first["loss_mask"])


def test_save_load_round_trip_checks_sha_and_caps_windows(tmp_path):
    arrays = _materialize(_windows())
    sha_one = save_validation_arrays(tmp_path / "one.npz", arrays)
    sha_two = save_validation_arrays(tmp_path / "two.npz", _materialize(_windows()))
    assert sha_one == sha_two
    loaded = load_validation_arrays(tmp_path / "one.npz", expected_sha256=sha_one)
    for name in VALID_ARRAY_NAMES:
        np.testing.assert_array_equal(loaded[name], arrays[name])
    capped = load_validation_arrays(tmp_path / "one.npz", max_windows=3)
    assert all(len(value) == 3 for value in capped.values())
    with pytest.raises(ValueError, match="SHA-256"):
        load_validation_arrays(tmp_path / "one.npz", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="only"):
        load_validation_arrays(tmp_path / "one.npz", max_windows=13)
    with pytest.raises(ValueError, match="missing"):
        save_validation_arrays(tmp_path / "bad.npz", {"input_ids": arrays["input_ids"]})


def test_fixed_batches_yield_the_stored_arrays_in_order():
    arrays = _materialize(_windows())
    batches = FixedMlmBatches(arrays, batch_size=5)
    assert len(batches) == 3
    first = list(batches)
    second = list(batches)
    assert [len(b["x"]) for b in first] == [5, 5, 2]
    for key, name in (
        ("x", "masked_input_ids"),
        ("y", "labels"),
        ("loss_mask", "loss_mask"),
        ("attention_mask", "attention_mask"),
    ):
        np.testing.assert_array_equal(
            np.concatenate([b[key] for b in first]), arrays[name]
        )
        np.testing.assert_array_equal(
            np.concatenate([b[key] for b in second]), arrays[name]
        )
    with pytest.raises(ValueError, match="batch_size"):
        FixedMlmBatches(arrays, batch_size=0)


def _manual_metrics(params, structure, arrays):
    batch = {
        "x": jnp.asarray(arrays["masked_input_ids"]),
        "attention_mask": jnp.asarray(arrays["attention_mask"]),
        "loss_mask": jnp.asarray(arrays["loss_mask"]),
    }
    state = compute_forward_pass(params, structure, batch, jax.random.PRNGKey(5))
    probs = np.asarray(state.nodes["mlm_head"].z_mu, dtype=np.float64)
    mask = arrays["loss_mask"]
    labels = arrays["labels"]
    picked = np.take_along_axis(probs, labels[..., None], axis=-1)[..., 0]
    ce = float(-np.log(picked[mask] + 1e-10).mean())
    acc = float((probs.argmax(-1) == labels)[mask].mean())
    return ce, acc, int(mask.sum())


def test_evaluation_sums_over_masked_tokens_and_is_deterministic():
    arrays = _materialize(_windows())
    structure = _tiny_bert()
    params = initialize_params(structure, jax.random.PRNGKey(0))
    first = evaluate_fixed_set(params, structure, arrays, batch_size=5)
    second = evaluate_fixed_set(params, structure, arrays, batch_size=5)
    assert first == second
    whole = evaluate_fixed_set(params, structure, arrays, batch_size=64)
    ce, acc, count = _manual_metrics(params, structure, arrays)
    for result in (first, whole):
        assert result["num_masked_tokens"] == count
        assert result["masked_ce"] == pytest.approx(ce, rel=1e-5)
        assert result["masked_accuracy"] == pytest.approx(acc)
        assert result["masked_perplexity"] == pytest.approx(np.exp(result["masked_ce"]))


def test_evaluation_turns_dropout_off():
    arrays = _materialize(_windows())
    plain = _tiny_bert(0.0)
    dropout = _tiny_bert(0.5)
    params = initialize_params(plain, jax.random.PRNGKey(0))
    assert evaluate_fixed_set(params, plain, arrays) == evaluate_fixed_set(
        params, dropout, arrays
    )


def test_evaluation_rejects_sets_without_masked_tokens():
    arrays = dict(_materialize(_windows()))
    arrays["loss_mask"] = np.zeros_like(arrays["loss_mask"])
    structure = _tiny_bert()
    params = initialize_params(structure, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="no masked tokens"):
        evaluate_fixed_set(params, structure, arrays)

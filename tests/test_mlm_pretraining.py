"""End-to-end masked-LM pretraining with the existing backprop training loop."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.training import evaluate_mlm, train_backprop_autoregressive
from fabricpc.utils.data import MlmTextBatcher

SEQ_LEN = 8
VOCAB_SIZE = 24
SPECIALS = {0, 1, 2, 3}


def build_model(rng_key):
    structure = create_encoder_mlm_model(
        depth=1,
        embed_dim=16,
        num_heads=2,
        mlp_dim=32,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=2, max_norm=5.0),
    )
    return initialize_params(structure, rng_key), structure


def make_loader(seed=0, num_tokens=96, mask_prob=0.3):
    pattern = np.tile(np.arange(4, VOCAB_SIZE, dtype=np.int32), 8)[:num_tokens]
    return MlmTextBatcher(
        token_ids=pattern,
        seq_len=SEQ_LEN,
        batch_size=4,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=VOCAB_SIZE,
        special_token_ids=SPECIALS,
        cls_token_id=1,
        seed=seed,
        mask_prob=mask_prob,
    )


class _ListLoader:
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def test_backprop_mlm_training_reduces_masked_loss(rng_key):
    params, structure = build_model(rng_key)
    val_loader = make_loader(seed=1)
    before = evaluate_mlm(params, structure, val_loader, rng_key)["masked_loss"]

    trained, _, _ = train_backprop_autoregressive(
        params,
        structure,
        make_loader(),
        optax.adam(1e-2),
        {"num_epochs": 10},
        rng_key,
        verbose=False,
    )

    after = evaluate_mlm(trained, structure, val_loader, rng_key)["masked_loss"]
    assert after < 0.8 * before


def test_epoch_callback_can_run_mlm_validation(rng_key):
    params, structure = build_model(rng_key)
    val_loader = make_loader(seed=1)

    def validate(epoch_idx, epoch_params, structure, config, key, **_):
        return evaluate_mlm(epoch_params, structure, val_loader, key)

    _, _, epoch_results = train_backprop_autoregressive(
        params,
        structure,
        make_loader(),
        optax.adam(1e-2),
        {"num_epochs": 2},
        rng_key,
        verbose=False,
        epoch_callback=validate,
    )

    assert len(epoch_results) == 2
    assert all(result["num_masked_tokens"] > 0 for result in epoch_results)


def _fixed_batch(loss_mask):
    tokens = jnp.arange(4, 4 + SEQ_LEN, dtype=jnp.int32)[None, :]
    return {
        "x": tokens,
        "y": tokens,
        "loss_mask": jnp.asarray(loss_mask, dtype=jnp.float32)[None, :],
        "attention_mask": jnp.ones((1, SEQ_LEN), dtype=jnp.float32),
    }


class TestEvaluateMlm:
    def test_weights_batches_by_masked_token_count(self, rng_key):
        params, structure = build_model(rng_key)
        one = _fixed_batch([1, 0, 0, 0, 0, 0, 0, 0])
        three = _fixed_batch([0, 1, 1, 1, 0, 0, 0, 0])

        combined = evaluate_mlm(params, structure, _ListLoader([one, three]), rng_key)
        first = evaluate_mlm(params, structure, _ListLoader([one]), rng_key)
        second = evaluate_mlm(params, structure, _ListLoader([three]), rng_key)

        expected = (first["masked_loss"] * 1 + second["masked_loss"] * 3) / 4
        assert combined["num_masked_tokens"] == 4
        assert combined["masked_loss"] == pytest.approx(expected, rel=1e-5)

    def test_restores_loader_state(self, rng_key):
        params, structure = build_model(rng_key)
        loader = make_loader(seed=3)

        first = evaluate_mlm(params, structure, loader, rng_key)
        second = evaluate_mlm(params, structure, loader, rng_key)

        assert loader.state_dict() == {"epoch": 0}
        assert first == second

    def test_rejects_batches_without_loss_mask(self, rng_key):
        params, structure = build_model(rng_key)
        batch = {
            k: v for k, v in _fixed_batch([1] * SEQ_LEN).items() if k != "loss_mask"
        }

        with pytest.raises(ValueError, match="loss_mask"):
            evaluate_mlm(params, structure, _ListLoader([batch]), rng_key)

    def test_rejects_loader_with_no_masked_tokens(self, rng_key):
        params, structure = build_model(rng_key)
        empty = _fixed_batch([0] * SEQ_LEN)

        with pytest.raises(ValueError, match="no masked tokens"):
            evaluate_mlm(params, structure, _ListLoader([empty]), rng_key)

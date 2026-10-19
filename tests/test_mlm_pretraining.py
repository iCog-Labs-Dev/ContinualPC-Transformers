"""End-to-end masked-LM pretraining with the existing backprop and PC loops."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.training import (
    evaluate_autoregressive,
    evaluate_mlm,
    train_autoregressive,
    train_backprop_autoregressive,
)
from fabricpc.training.train import replicate_opt_state, replicate_params, shard_batch
from fabricpc.training.train_autoregressive import (
    compute_loss,
    create_pmap_train_step_autoregressive,
    get_graph_param_gradient_autoregressive,
)
from fabricpc.training.train_backprop import train_step_backprop_autoregressive
from fabricpc.utils.data import MlmTextBatcher

train_backprop = importlib.import_module("fabricpc.training.train_backprop")

SEQ_LEN = 8
VOCAB_SIZE = 24
SPECIALS = {0, 1, 2, 3}


def build_model(rng_key, infer_steps=2):
    structure = create_encoder_mlm_model(
        depth=1,
        embed_dim=16,
        num_heads=2,
        mlp_dim=32,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        inference=InferenceSGDNormClip(
            eta_infer=0.1, infer_steps=infer_steps, max_norm=5.0
        ),
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


class TestEvaluatorCache:
    def test_alternating_graphs_reuse_their_evaluators(self, rng_key, monkeypatch):
        monkeypatch.setattr(train_backprop, "_masked_lm_eval_cache", {})
        params_a, structure_a = build_model(rng_key)
        params_b, structure_b = build_model(jax.random.PRNGKey(1))
        loader = _ListLoader([_fixed_batch([1, 0, 1, 0, 0, 0, 0, 0])])

        evaluate_mlm(params_a, structure_a, loader, rng_key)
        first = train_backprop._jitted_masked_lm_batch_totals(structure_a)
        evaluate_mlm(params_b, structure_b, loader, rng_key)
        evaluate_mlm(params_a, structure_a, loader, rng_key)

        assert train_backprop._jitted_masked_lm_batch_totals(structure_a) is first
        assert train_backprop._jitted_masked_lm_batch_totals(structure_b) is not first

    def test_least_recently_used_evaluator_is_evicted(self, rng_key, monkeypatch):
        monkeypatch.setattr(train_backprop, "_masked_lm_eval_cache", {})
        monkeypatch.setattr(train_backprop, "_MASKED_LM_EVAL_CACHE_SIZE", 2)
        structures = [build_model(rng_key)[1] for _ in range(3)]

        first = train_backprop._jitted_masked_lm_batch_totals(structures[0])
        train_backprop._jitted_masked_lm_batch_totals(structures[1])
        train_backprop._jitted_masked_lm_batch_totals(structures[0])
        train_backprop._jitted_masked_lm_batch_totals(structures[2])

        cache = train_backprop._masked_lm_eval_cache
        assert id(structures[1]) not in cache
        assert train_backprop._jitted_masked_lm_batch_totals(structures[0]) is first

    def test_oldest_evaluator_is_evicted(self, rng_key, monkeypatch):
        monkeypatch.setattr(train_backprop, "_masked_lm_eval_cache", {})
        monkeypatch.setattr(train_backprop, "_MASKED_LM_EVAL_CACHE_SIZE", 2)
        structures = [build_model(rng_key)[1] for _ in range(3)]

        fns = [train_backprop._jitted_masked_lm_batch_totals(s) for s in structures]

        cache = train_backprop._masked_lm_eval_cache
        assert len(cache) == 2
        assert id(structures[0]) not in cache
        assert train_backprop._jitted_masked_lm_batch_totals(structures[2]) is fns[2]


def _training_batches():
    batches = list(make_loader(mask_prob=0.3))
    return [{k: jnp.asarray(v) for k, v in batch.items()} for batch in batches]


class TestLoopTrainStep:
    @pytest.mark.parametrize("use_pmap", [False, True])
    def test_compiled_step_returns_no_predictions(self, rng_key, use_pmap):
        params, structure = build_model(rng_key)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        batch = _training_batches()[0]
        key = rng_key
        if use_pmap:
            params = replicate_params(params, 1)
            opt_state = replicate_opt_state(opt_state, 1)
            batch = shard_batch({k: v for k, v in batch.items() if k != "loss_mask"}, 1)
            key = rng_key[None, :]

        step = train_backprop._create_loop_train_step(
            structure, optimizer, True, use_pmap
        )
        out = jax.eval_shape(step, params, opt_state, batch, key)

        assert len(out) == 3
        assert out[2].size == 1
        assert not any(
            leaf.shape[-2:] == (SEQ_LEN, VOCAB_SIZE)
            for leaf in jax.tree_util.tree_leaves(out)
            if leaf.ndim >= 3
        )

    def test_loop_losses_match_direct_steps(self, rng_key):
        params, structure = build_model(rng_key)
        optimizer = optax.adam(1e-3)
        batches = _training_batches()

        _, losses, _ = train_backprop_autoregressive(
            params,
            structure,
            _ListLoader(batches),
            optimizer,
            {"num_epochs": 2},
            rng_key,
            verbose=False,
        )

        step = jax.jit(
            lambda p, o, b, k: train_step_backprop_autoregressive(
                p, o, b, structure, optimizer, k
            )
        )
        opt_state, key, expected = optimizer.init(params), rng_key, []
        for _ in range(2):
            epoch_key, key = jax.random.split(key)
            batch_keys = jax.random.split(epoch_key, len(batches))
            epoch = []
            for batch, batch_key in zip(batches, batch_keys):
                params, opt_state, loss, _ = step(params, opt_state, batch, batch_key)
                epoch.append(float(loss))
            expected.append(epoch)

        assert losses == expected


def test_pc_mlm_training_updates_encoder_and_reduces_masked_loss(rng_key):
    params, structure = build_model(rng_key, infer_steps=8)
    val_loader = make_loader(seed=1)
    before = evaluate_mlm(params, structure, val_loader, rng_key)["masked_loss"]

    trained, _, _ = train_autoregressive(
        params,
        structure,
        make_loader(),
        optax.adam(1e-2),
        {"num_epochs": 10},
        rng_key,
        verbose=False,
    )

    after = evaluate_mlm(trained, structure, val_loader, rng_key)["masked_loss"]
    assert after < before
    for node in ("token_embeddings", "L0_mha", "L0_mlp1", "L0_mlp2"):
        for name, initial in params.nodes[node].weights.items():
            final = trained.nodes[node].weights[name]
            assert jnp.all(jnp.isfinite(final))
            assert not jnp.allclose(final, initial), f"{node}.{name} never updated"


class TestPcMlmGradient:
    def test_gradients_ignore_unmasked_targets(self, rng_key):
        params, structure = build_model(rng_key)
        batch = next(iter(make_loader()))
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        unmasked = batch["loss_mask"] == 0
        changed = {**batch, "y": jnp.where(unmasked, (batch["y"] + 1) % 24, batch["y"])}

        grads, _, _, _ = get_graph_param_gradient_autoregressive(
            params, batch, structure, rng_key
        )
        changed_grads, _, _, _ = get_graph_param_gradient_autoregressive(
            params, changed, structure, rng_key
        )

        assert all(
            jnp.allclose(a, b, atol=1e-6)
            for a, b in zip(
                jax.tree_util.tree_leaves(grads),
                jax.tree_util.tree_leaves(changed_grads),
            )
        )

    def test_reported_cross_entropy_counts_masked_tokens_only(self, rng_key):
        params, structure = build_model(rng_key)
        batch = {k: jnp.asarray(v) for k, v in next(iter(make_loader())).items()}

        _, _, metric, final_state = get_graph_param_gradient_autoregressive(
            params, batch, structure, rng_key
        )

        probs = final_state.nodes["mlm_head"].z_mu
        one_hot = jax.nn.one_hot(batch["y"], VOCAB_SIZE)
        per_token = -jnp.sum(one_hot * jnp.log(probs + 1e-10), axis=-1)
        mask = batch["loss_mask"]
        expected = jnp.sum(per_token * mask) / jnp.sum(mask)
        assert metric == pytest.approx(float(expected), rel=1e-5)


class TestPcStepInputs:
    def test_supplied_dropout_mask_is_kept(self, rng_key):
        structure = create_encoder_mlm_model(
            depth=1,
            embed_dim=16,
            num_heads=2,
            mlp_dim=32,
            seq_len=SEQ_LEN,
            vocab_size=VOCAB_SIZE,
            dropout_rate=0.5,
            inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=2, max_norm=5.0),
        )
        params = initialize_params(structure, rng_key)
        batch = {k: jnp.asarray(v) for k, v in next(iter(make_loader())).items()}
        task = sorted(t for t in structure.task_map if t.startswith("dropout_"))[0]
        zeros = jnp.zeros((4, SEQ_LEN, 16))

        _, _, _, final_state = get_graph_param_gradient_autoregressive(
            params, {**batch, task: zeros}, structure, rng_key
        )

        node = structure.task_map[task]
        assert jnp.array_equal(final_state.nodes[node].z_latent, zeros)

    def test_warns_when_batch_entry_matches_no_task(self, rng_key):
        params, structure = build_model(rng_key)
        batch = {k: jnp.asarray(v) for k, v in next(iter(make_loader())).items()}
        batch["segment_ids"] = jnp.zeros((4, SEQ_LEN))

        with pytest.warns(UserWarning, match="segment_ids"):
            get_graph_param_gradient_autoregressive(params, batch, structure, rng_key)


class TestPcLoopRejectsUnsupportedMaskedPaths:
    def test_training_rejects_masked_batches_on_pmap(self, rng_key):
        params, structure = build_model(rng_key)

        with pytest.raises(NotImplementedError, match="CUDA_VISIBLE_DEVICES"):
            train_autoregressive(
                params,
                structure,
                make_loader(),
                optax.adam(1e-2),
                {"num_epochs": 1},
                rng_key,
                verbose=False,
                pmap_single_device=True,
            )

    def test_pmap_step_factory_rejects_masked_batches(self, rng_key):
        params, structure = build_model(rng_key)
        optimizer = optax.adam(1e-2)
        step = create_pmap_train_step_autoregressive(structure, optimizer)
        batch = {k: jnp.asarray(v) for k, v in next(iter(make_loader())).items()}

        with pytest.raises(NotImplementedError, match="loss_mask"):
            step(
                replicate_params(params, 1),
                replicate_opt_state(optimizer.init(params), 1),
                shard_batch(batch, 1),
                rng_key[None],
            )

    @pytest.mark.parametrize("pmap_single_device", [False, True])
    def test_evaluator_points_to_evaluate_mlm(self, rng_key, pmap_single_device):
        params, structure = build_model(rng_key)

        with pytest.raises(ValueError, match="evaluate_mlm"):
            evaluate_autoregressive(
                params,
                structure,
                make_loader(),
                {"use_causal_mask": False},
                rng_key,
                pmap_single_device=pmap_single_device,
            )

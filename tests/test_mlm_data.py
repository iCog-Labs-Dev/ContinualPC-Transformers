"""Tests for masked-language-model masking, batching and tokenization."""

import numpy as np
import pytest

from fabricpc.utils.data import (
    MLM_SPECIAL_TOKENS,
    MlmTextBatcher,
    apply_mlm_mask,
    tokenize_texts_bpe,
)

SPECIALS = {0, 1, 2, 3}  # pad, cls, sep, mask


def make_batcher(token_ids, **overrides):
    kwargs = dict(
        token_ids=np.asarray(token_ids, dtype=np.int32),
        seq_len=5,
        batch_size=2,
        pad_token_id=0,
        mask_token_id=3,
        vocab_size=32,
        special_token_ids=SPECIALS,
        seed=0,
        shuffle=False,
    )
    kwargs.update(overrides)
    return MlmTextBatcher(**kwargs)


class TestApplyMlmMask:
    def test_special_tokens_are_never_selected(self):
        tokens = np.array([[1, 5, 6, 2, 0]], dtype=np.int32)

        _, _, loss_mask = apply_mlm_mask(
            tokens,
            mask_token_id=3,
            vocab_size=10,
            special_token_ids=SPECIALS,
            seed=0,
            mask_prob=1.0,
        )

        assert loss_mask.tolist() == [[False, True, True, False, False]]

    def test_random_replacements_exclude_special_tokens(self):
        tokens = np.tile(np.array([[4, 5, 6, 7]], dtype=np.int32), (20, 1))

        masked, targets, loss_mask = apply_mlm_mask(
            tokens,
            mask_token_id=3,
            vocab_size=8,
            special_token_ids=SPECIALS,
            seed=1,
            mask_prob=1.0,
        )

        random_replacements = loss_mask & (masked != targets) & (masked != 3)
        assert np.any(random_replacements)
        assert not np.any(np.isin(masked[random_replacements], list(SPECIALS)))

    def test_targets_are_the_original_tokens(self):
        tokens = np.array([[4, 5, 6, 7]], dtype=np.int32)

        _, targets, _ = apply_mlm_mask(
            tokens,
            mask_token_id=3,
            vocab_size=8,
            special_token_ids=SPECIALS,
            seed=0,
            mask_prob=1.0,
        )

        assert np.array_equal(targets, tokens)


class TestMlmTextBatcher:
    def test_masks_are_float32_for_jax_clamps(self):
        batch = next(iter(make_batcher(np.arange(10, 20), mask_prob=0.5)))

        assert batch["loss_mask"].dtype == np.float32
        assert batch["attention_mask"].dtype == np.float32
        assert batch["x"].dtype == np.int32 and batch["y"].dtype == np.int32

    def test_prefixes_cls_and_uses_stream_stride(self):
        batcher = make_batcher(np.arange(10, 20), cls_token_id=1, mask_prob=1.0)

        batch = next(iter(batcher))

        assert np.array_equal(batch["y"][:, 0], [1, 1])
        assert not np.any(batch["loss_mask"][:, 0])
        assert np.all(batch["attention_mask"][:, 0] == 1.0)
        assert np.array_equal(batch["y"][0, 1:], [10, 11, 12, 13])
        assert np.array_equal(batch["y"][1, 1:], [14, 15, 16, 17])

    def test_padding_is_excluded_from_attention(self):
        batcher = make_batcher(np.arange(10, 17), stride=5, mask_prob=1.0)

        batch = next(iter(batcher))

        assert batch["attention_mask"][1].tolist() == [1, 1, 0, 0, 0]
        assert not np.any(batch["loss_mask"][1, 2:])

    def test_drops_incomplete_final_batch(self):
        batcher = make_batcher(np.arange(10, 25), mask_prob=0.5)

        assert len(batcher) == 1
        assert len(list(batcher)) == 1

    def test_every_batch_has_a_masked_position(self):
        # mask_prob is tiny, so most first draws select nothing and get redrawn.
        batcher = make_batcher(np.arange(10, 50), batch_size=1, mask_prob=0.02)

        batches = list(batcher)

        assert len(batches) == 8
        assert all(batch["loss_mask"].sum() >= 1 for batch in batches)

    def test_tiny_mask_prob_still_masks_every_batch(self):
        # 1e-9 never samples a mask, so the forced fallback must kick in.
        batcher = make_batcher(np.arange(10, 30), batch_size=1, mask_prob=1e-9)

        batches = list(batcher)

        assert all(batch["loss_mask"].sum() == 1 for batch in batches)
        for batch in batches:
            position = batch["loss_mask"].astype(bool)
            assert batch["x"][position] == 3 and batch["y"][position] >= 10

    def test_windows_of_only_special_tokens_are_skipped(self):
        # 10..14 fills one window; the trailing [SEP] alone would be unmaskable.
        batcher = make_batcher(np.array([10, 11, 12, 13, 14, 2]), batch_size=1)

        assert len(batcher) == 1
        assert np.array_equal(next(iter(batcher))["y"][0], [10, 11, 12, 13, 14])

    def test_stream_of_only_special_tokens_has_no_batches(self):
        assert len(make_batcher(np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]))) == 0

    @pytest.mark.parametrize(
        "overrides",
        [{"pad_token_id": 9}, {"mask_token_id": 9}, {"cls_token_id": 9}],
    )
    def test_rejects_special_ids_missing_from_special_set(self, overrides):
        with pytest.raises(ValueError, match="must be in special_token_ids"):
            make_batcher(np.arange(10, 20), **overrides)

    def test_attention_follows_position_not_token_value(self):
        # Pad id 0 inside real text stays visible; only the padded tail is hidden.
        batcher = make_batcher(np.array([10, 0, 11, 12, 13, 14, 15]), stride=5)

        batch = next(iter(batcher))

        assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 1]
        assert batch["attention_mask"][1].tolist() == [1, 1, 0, 0, 0]

    def test_nearby_seeds_do_not_share_epoch_streams(self):
        tokens = np.arange(10, 30)
        seed_one = make_batcher(tokens, seed=1, mask_prob=0.5)
        seed_zero = make_batcher(tokens, seed=0, mask_prob=0.5)
        list(seed_zero)  # advance to epoch 1

        first = next(iter(seed_one))
        second = next(iter(seed_zero))

        assert not np.array_equal(first["loss_mask"], second["loss_mask"])

    @pytest.mark.parametrize("mask_prob", [0.0, 1.5])
    def test_rejects_mask_prob_outside_unit_interval(self, mask_prob):
        with pytest.raises(ValueError, match="mask_prob"):
            make_batcher(np.arange(10, 20), mask_prob=mask_prob)

    def test_state_dict_replays_the_same_epoch(self):
        batcher = make_batcher(np.arange(10, 30), shuffle=True, mask_prob=0.5)
        state = batcher.state_dict()

        first = list(batcher)
        batcher.load_state_dict(state)
        replay = list(batcher)
        next_epoch = list(batcher)

        for a, b in zip(first, replay):
            assert all(np.array_equal(a[key], b[key]) for key in a)
        assert any(
            not np.array_equal(a["loss_mask"], c["loss_mask"])
            for a, c in zip(first, next_epoch)
        )


class TestTokenizeTextsBpe:
    def test_trains_tokenizer_and_separates_texts(self):
        pytest.importorskip("tokenizers")
        texts = ["the cat sat on the mat", "a dog ran in the park"] * 4

        token_ids, tokenizer = tokenize_texts_bpe(texts, vocab_size=60)

        sep_id = tokenizer.token_to_id("[SEP]")
        assert token_ids.dtype == np.int32
        assert int(np.sum(token_ids == sep_id)) == len(texts)
        for token in MLM_SPECIAL_TOKENS:
            assert tokenizer.token_to_id(token) is not None

    def test_reuses_a_given_tokenizer(self):
        pytest.importorskip("tokenizers")
        _, tokenizer = tokenize_texts_bpe(["the cat sat on the mat"] * 4, vocab_size=40)

        token_ids, same = tokenize_texts_bpe(
            ["the mat sat"], vocab_size=40, tokenizer=tokenizer
        )

        assert same is tokenizer
        assert token_ids[-1] == tokenizer.token_to_id("[SEP]")

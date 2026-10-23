"""Tests for the domain-forgetting data pipeline (offline, synthetic text)."""

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from examples.convert_hf_bert import BERT_SPECIAL_TOKENS, build_wordpiece_tokenizer
from fabricpc.experiments.domain_forgetting.data import (
    DOMAIN_REGISTRY,
    WIKITEXT_OVERLAP_NOTE,
    TokenStats,
    build_domain_caches,
    build_train_stream,
    check_token_range,
    deduplicate_domains,
    detokenize_wikitext,
    is_wikitext_title,
    iter_tokenized,
    load_domain_cache,
    normalize_text,
    prepare_documents,
    regroup_wikitext_articles,
    shuffle_documents,
    strip_imdb_markup,
    validate_domain_names,
)

DOMAIN_WORDS = {
    "news": ["market", "stocks", "bank", "trade", "rates", "shares", "growth"],
    "reviews": ["film", "movie", "actor", "plot", "scene", "great", "boring"],
    "wiki": ["river", "city", "born", "war", "century", "king", "north"],
}
COMMON_WORDS = ["the", "a", "of", "and", "is", "in", "was"]


def tiny_tokenizer(tmp_path):
    """Return a WordPiece tokenizer over the synthetic domain words."""
    pytest.importorskip("tokenizers")
    words = sorted({*COMMON_WORDS, *(w for ws in DOMAIN_WORDS.values() for w in ws)})
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join([*BERT_SPECIAL_TOKENS, *words]) + "\n")
    return build_wordpiece_tokenizer(vocab_path)


def _sentence(rng: np.random.Generator, domain: str, length: int) -> str:
    pool = DOMAIN_WORDS[domain] + COMMON_WORDS
    return " ".join(pool[int(i)] for i in rng.integers(0, len(pool), size=length))


def synthetic_domain_rows(
    seed: int = 0, train_docs: int = 40, valid_docs: int = 30
) -> dict[str, dict[str, list[str]]]:
    """Raw rows for news (plain), reviews (IMDB markup) and wiki (WikiText lines)."""
    rng = np.random.default_rng(seed)
    rows: dict[str, dict[str, list[str]]] = {}
    for name in ("news", "reviews"):
        sep = " <br /><br /> " if name == "reviews" else "  "
        rows[name] = {
            split: [
                _sentence(rng, name, 6) + sep + _sentence(rng, name, 6)
                for _ in range(count)
            ]
            for split, count in (("train", train_docs), ("valid", valid_docs))
        }
    rows["wiki"] = {}
    for split, count in (("train", train_docs), ("valid", valid_docs)):
        lines: list[str] = []
        for index in range(count):
            lines += [
                "",
                f" = {_sentence(rng, 'wiki', 2)} {index} = \n",
                "",
                f" {_sentence(rng, 'wiki', 8)} \n",
                "",
                " = = section = = \n",
                "",
                f" {_sentence(rng, 'wiki', 4)} \n",
            ]
        rows["wiki"][split] = lines
    return rows


def test_normalize_text_nfkc_collapses_whitespace_and_strips():
    assert normalize_text("  ﬁne Ａ  b\t\nc  ") == "fine A b c"
    assert normalize_text(" \n\t ") == ""


def test_imdb_tag_removal():
    assert strip_imdb_markup("a<br />b<br/>c<BR>d") == "a\nb\nc\nd"
    docs = prepare_documents(DOMAIN_REGISTRY["reviews"], ["good<br /><br />film", ""])
    assert docs == ["good film"]


def test_wikitext_title_detection_and_regrouping():
    assert is_wikitext_title(" = Valkyria Chronicles = \n", "", "")
    assert is_wikitext_title(" = X = \n", None, "")
    assert not is_wikitext_title(" = = Gameplay = = \n", "", "")
    assert not is_wikitext_title(" = = = Plot = = = \n", "", "")
    assert not is_wikitext_title(" plain = text = here \n", "", "")
    assert not is_wikitext_title(" = Title = \n", " body \n", "")
    assert not is_wikitext_title(" = Title = \n", "", " body \n")
    lines = [
        " preamble \n",
        "",
        " = First = \n",
        "",
        " body one \n",
        "",
        " = = Section = = \n",
        "",
        " body two \n",
        "",
        " = Second = \n",
        "",
        " body three \n",
    ]
    assert regroup_wikitext_articles(lines) == [
        " preamble \n",
        " = First = \n\n body one \n\n = = Section = = \n\n body two \n",
        " = Second = \n\n body three \n",
    ]
    docs = prepare_documents(DOMAIN_REGISTRY["wiki"], lines)
    assert docs == [
        "preamble",
        "= First = body one = = Section = = body two",
        "= Second = body three",
    ]


def test_wikitext_body_lines_with_equals_are_not_titles():
    lines = [
        "",
        " = Wayne Gretzky = \n",
        "",
        " GP = games played ; G = goals ; A = assists ; pts = points ; pim = \n",
        " = assists ; pts = points ; pim = \n",
        "",
        " = x + y = \n",
        " the equation above holds . \n",
        "",
        " = = Statistics = = \n",
        "",
        " = E = mc ^ 2 = \n",
        " more text \n",
        "",
        " = Next Article = \n",
        "",
        " body \n",
    ]
    articles = regroup_wikitext_articles(lines)
    assert len(articles) == 2
    assert articles[0].startswith(" = Wayne Gretzky = ")
    assert " = assists ; pts = points ; pim = " in articles[0]
    assert " = x + y = " in articles[0]
    assert " = E = mc ^ 2 = " in articles[0]
    assert articles[1] == " = Next Article = \n\n body \n"


def test_wikitext_number_separators_are_detokenised():
    lines = [
        "",
        " = Title = \n",
        "",
        " a 1 @,@ 000 km long , 3 @.@ 5 m wide , state @-@ of @-@ the @-@ art \n",
        "@-@ edge and edge @,@\n",
    ]
    (doc,) = prepare_documents(DOMAIN_REGISTRY["wiki"], lines)
    assert "@" not in doc
    assert "1,000 km" in doc
    assert "3.5 m" in doc
    assert "state-of-the-art" in doc
    assert detokenize_wikitext(" 2 @-@ 3 ") == " 2-3 "
    notes = " ".join(DOMAIN_REGISTRY["wiki"].notes)
    assert "@-@" in notes and "BertPreTokenizer" in notes


def test_deduplicate_within_cross_and_valid_vs_train():
    train = {"news": ["a", "b", "a", "c"], "reviews": ["b", "d", "d"]}
    valid = {"news": ["c", "e", "e"], "reviews": ["a", "d", "f"]}
    train_out, valid_out, report = deduplicate_domains(train, valid)
    assert train_out == {"news": ["a", "b", "c"], "reviews": ["d"]}
    assert valid_out == {"news": ["e"], "reviews": ["f"]}
    assert report["train_within_domain_duplicates"] == {"news": 1, "reviews": 1}
    assert report["train_cross_domain_duplicates"] == {"news": 0, "reviews": 1}
    assert report["valid_within_domain_duplicates"] == {"news": 1, "reviews": 0}
    assert report["valid_train_overlap_removed"] == {"news": 1, "reviews": 2}
    assert report["train_kept"] == {"news": 3, "reviews": 1}
    assert report["valid_kept"] == {"news": 1, "reviews": 1}


def test_shuffle_is_deterministic_permutation():
    docs = [f"doc{i}" for i in range(50)]
    first = shuffle_documents(docs, 3)
    assert first == shuffle_documents(docs, 3)
    assert first != shuffle_documents(docs, 4)
    assert first != docs
    assert sorted(first) == sorted(docs)


def test_token_range_rejection():
    check_token_range(np.array([0, 4, 9]), 10)
    check_token_range(np.array([], dtype=np.int64), 10)
    with pytest.raises(ValueError, match="outside"):
        check_token_range(np.array([0, 10]), 10)
    with pytest.raises(ValueError, match="outside"):
        check_token_range(np.array([-1, 3]), 10)
    with pytest.raises(ValueError, match="outside"):
        build_train_stream(
            [np.array([1, 2]), np.array([3, 12])],
            tokens_per_domain=4,
            sep_token_id=9,
            vocab_size=10,
        )


def test_train_stream_has_exact_budget_and_stops_reading():
    consumed = []

    def docs():
        for index in range(100):
            consumed.append(index)
            yield np.array([index % 7 + 1] * 3)

    stream = build_train_stream(
        docs(), tokens_per_domain=10, sep_token_id=9, vocab_size=10
    )
    assert stream.dtype == np.uint16
    assert stream.tolist() == [1, 1, 1, 9, 2, 2, 2, 9, 3, 3]
    assert consumed == [0, 1, 2]
    with pytest.raises(ValueError, match="fewer than"):
        build_train_stream(
            [np.array([1, 2])], tokens_per_domain=5, sep_token_id=9, vocab_size=10
        )
    with pytest.raises(ValueError, match="uint16"):
        build_train_stream(
            [np.array([1])], tokens_per_domain=1, sep_token_id=0, vocab_size=70000
        )


def test_iter_tokenized_counts_words_tokens_and_unknowns(tmp_path):
    tokenizer = tiny_tokenizer(tmp_path)
    stats = TokenStats()
    ids = list(
        iter_tokenized(
            tokenizer,
            ["the market", "the zebra of"],
            vocab_size=tokenizer.get_vocab_size(),
            unk_token_id=tokenizer.token_to_id("[UNK]"),
            stats=stats,
            chunk_size=1,
        )
    )
    assert [len(x) for x in ids] == [2, 3]
    assert stats.summary() == {
        "documents": 2,
        "words": 5,
        "tokens": 5,
        "unk_tokens": 1,
        "tokens_per_word": 1.0,
        "unk_rate": 0.2,
    }


def test_validate_domain_names():
    assert validate_domain_names(["wiki", " news "]) == ["wiki", "news"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_domain_names(["news", "news"])
    with pytest.raises(ValueError, match="unknown"):
        validate_domain_names(["news", "poems"])
    with pytest.raises(ValueError, match="at least one"):
        validate_domain_names([])


def build_tiny_caches(tmp_path, tokenizer, **kwargs):
    """Build caches for the three synthetic domains and return the manifest."""
    options = {
        "tokens_per_domain": 60,
        "seq_len": 6,
        "num_valid_windows": 8,
        "mask_prob": 0.3,
        **kwargs,
    }
    return build_domain_caches(
        synthetic_domain_rows(),
        tokenizer_json=tokenizer.to_str(),
        out_dir=tmp_path / "data",
        **options,
    )


def test_build_domain_caches_writes_verified_caches(tmp_path):
    tokenizer = tiny_tokenizer(tmp_path)
    manifest = build_tiny_caches(tmp_path, tokenizer)
    data_dir = tmp_path / "data"
    on_disk = json.loads((data_dir / "dataset_manifest.json").read_text())
    assert on_disk["domains"] == ["news", "reviews", "wiki"]
    assert manifest["overlap_notes"] == [WIKITEXT_OVERLAP_NOTE]
    assert "later-listed domain" in on_disk["dedup_policy"]
    assert "--domains order" in on_disk["dedup_policy"]
    wiki_manifest = on_disk["per_domain"]["wiki"]
    assert "@-@" in wiki_manifest["detokenization"]
    assert on_disk["per_domain"]["news"]["detokenization"] is None
    assert manifest["num_valid_windows"] == 8
    assert not manifest["valid_windows_reduced"]
    for name in ("news", "reviews", "wiki"):
        assert sorted(p.name for p in (data_dir / name).iterdir()) == [
            "manifest.json",
            "train.npy",
            "valid_windows.npz",
        ]
        cache = load_domain_cache(
            data_dir, name, tokenizer_json=tokenizer.to_str(), seq_len=6
        )
        assert cache.train.dtype == np.uint16
        assert len(cache.train) == 60
        assert int(cache.train.max()) < tokenizer.get_vocab_size()
        assert cache.valid["input_ids"].shape == (8, 6)
        assert (cache.valid["input_ids"][:, 0] == tokenizer.token_to_id("[CLS]")).all()
        assert cache.manifest["train"]["token_stats"]["unk_rate"] < 0.5
    prefix = load_domain_cache(
        data_dir,
        "news",
        tokenizer_json=tokenizer.to_str(),
        seq_len=6,
        tokens_per_domain=25,
        max_valid_windows=4,
    )
    assert len(prefix.train) == 25
    assert len(prefix.valid["input_ids"]) == 4


def test_build_domain_caches_is_deterministic_and_seeded(tmp_path):
    tokenizer = tiny_tokenizer(tmp_path)
    first = build_tiny_caches(tmp_path / "one", tokenizer)
    second = build_tiny_caches(tmp_path / "two", tokenizer)
    other = build_tiny_caches(tmp_path / "three", tokenizer, data_seed=1)
    for name in ("news", "reviews", "wiki"):
        a, b, c = (m["per_domain"][name] for m in (first, second, other))
        assert a["train"]["array_sha256"] == b["train"]["array_sha256"]
        assert a["valid"]["arrays_sha256"] == b["valid"]["arrays_sha256"]
        assert a["train"]["array_sha256"] != c["train"]["array_sha256"]
        assert a["valid"]["arrays_sha256"] == c["valid"]["arrays_sha256"]


def test_build_domain_caches_uses_minimum_valid_windows(tmp_path):
    tokenizer = tiny_tokenizer(tmp_path)
    manifest = build_tiny_caches(tmp_path, tokenizer, num_valid_windows=10_000)
    available = manifest["available_valid_windows"]
    assert manifest["num_valid_windows"] == min(available.values())
    assert manifest["valid_windows_reduced"]


def test_load_domain_cache_rejects_tampering_and_mismatches(tmp_path):
    tokenizer = tiny_tokenizer(tmp_path)
    build_tiny_caches(tmp_path, tokenizer)
    data_dir = tmp_path / "data"
    tokenizer_json = tokenizer.to_str()
    with pytest.raises(ValueError, match="seq_len"):
        load_domain_cache(data_dir, "news", tokenizer_json=tokenizer_json, seq_len=8)
    with pytest.raises(ValueError, match="different tokenizer"):
        load_domain_cache(
            data_dir, "news", tokenizer_json=tokenizer_json + " ", seq_len=6
        )
    with pytest.raises(ValueError, match="requested"):
        load_domain_cache(
            data_dir,
            "news",
            tokenizer_json=tokenizer_json,
            seq_len=6,
            tokens_per_domain=61,
        )
    train = np.load(data_dir / "news" / "train.npy")
    train[0] = (int(train[0]) + 1) % tokenizer.get_vocab_size()
    np.save(data_dir / "news" / "train.npy", train)
    with pytest.raises(ValueError, match="SHA-256"):
        load_domain_cache(data_dir, "news", tokenizer_json=tokenizer_json, seq_len=6)
    valid = dict(np.load(data_dir / "reviews" / "valid_windows.npz"))
    valid["labels"] = valid["labels"].copy()
    valid["labels"][0, 1] += 1
    np.savez(data_dir / "reviews" / "valid_windows.npz", **valid)
    with pytest.raises(ValueError, match="SHA-256"):
        load_domain_cache(data_dir, "reviews", tokenizer_json=tokenizer_json, seq_len=6)

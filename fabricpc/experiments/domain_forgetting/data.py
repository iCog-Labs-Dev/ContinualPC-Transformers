"""Pinned domain corpora, normalization, deduplication and token caches."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from fabricpc.experiments.domain_forgetting.artifacts import (
    arrays_sha256,
    atomic_save_npy,
    atomic_write_json,
    file_sha256,
    sha256_text,
)
from fabricpc.experiments.domain_forgetting.validation import (
    DEFAULT_NUM_VALID_WINDOWS,
    VALID_MASK_SEED,
    VALID_SHUFFLE_SEED,
    load_validation_arrays,
    materialize_validation_set,
    pack_windows,
    save_validation_arrays,
)

BERT_SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
CACHE_FORMAT = "fabricpc-domain-forgetting-cache"
CACHE_VERSION = 1
TOKENIZE_CHUNK = 1024

WIKITEXT_DETOKENISATION_NOTE = (
    "WikiText number separators detokenised before NFKC: ' @-@ ' -> '-', "
    "' @,@ ' -> ',', ' @.@ ' -> '.' (a missing space at a line edge is also "
    "accepted); spaces around other punctuation are left as they are, since "
    "BERT's BertPreTokenizer splits punctuation into separate tokens anyway"
)
WIKITEXT_TITLE_NOTE = (
    "lines regrouped into articles at raw ' = Title = ' lines (a single '=' "
    "level, matched after stripping only the trailing newline) whose "
    "previous and next raw lines are empty; '= = Section = =' lines never "
    "start an article"
)
DEDUP_POLICY = (
    "exact duplicates by SHA-256 of the normalized document: within a domain "
    "the first occurrence is kept; a train document already present in the "
    "train set of a domain listed earlier in the builder's --domains order "
    "is dropped from the later-listed domain; validation duplicates within "
    "a domain and validation documents present in any domain's train set "
    "are dropped"
)
WIKITEXT_OVERLAP_NOTE = (
    "WikiText-103 is drawn from English Wikipedia, which is part of BERT's "
    "pretraining data (Wikipedia + BooksCorpus); domain C likely overlaps the "
    "pretraining distribution. AG News and IMDB are not known overlaps."
)


@dataclass(frozen=True)
class DomainSpec:
    """A pinned Hugging Face dataset used as one training domain."""

    name: str
    dataset_id: str
    config: str | None
    revision: str
    text_field: str
    train_splits: tuple[str, ...]
    valid_split: str
    kind: str = "plain"
    notes: tuple[str, ...] = field(default_factory=tuple)


DOMAIN_REGISTRY: dict[str, DomainSpec] = {
    "news": DomainSpec(
        name="news",
        dataset_id="fancyzhx/ag_news",
        config=None,
        revision="eb185aade064a813bc0b7f42de02595523103ca4",
        text_field="text",
        train_splits=("train",),
        valid_split="test",
        notes=("labels discarded",),
    ),
    "reviews": DomainSpec(
        name="reviews",
        dataset_id="stanfordnlp/imdb",
        config="plain_text",
        revision="e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
        text_field="text",
        train_splits=("train", "unsupervised"),
        valid_split="test",
        kind="imdb",
        notes=("labels discarded", "<br /> replaced by a newline"),
    ),
    "wiki": DomainSpec(
        name="wiki",
        dataset_id="Salesforce/wikitext",
        config="wikitext-103-raw-v1",
        revision="b08601e04326c79dfdd32d625aee71d232d685c3",
        text_field="text",
        train_splits=("train",),
        valid_split="validation",
        kind="wikitext",
        notes=(
            WIKITEXT_TITLE_NOTE,
            "empty lines dropped after regrouping",
            WIKITEXT_DETOKENISATION_NOTE,
            WIKITEXT_OVERLAP_NOTE,
        ),
    ),
}

_BR_TAG = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_WIKITEXT_TITLE = re.compile(r" = [^=](?:.*[^=])? = ")
_WIKITEXT_SEPARATOR = re.compile(r" ?@([-,.])@ ?")


def validate_domain_names(names: Sequence[str]) -> list[str]:
    """Return ``names`` after rejecting empty, duplicate or unknown domains."""
    names = [name.strip() for name in names if name.strip()]
    if not names:
        raise ValueError("at least one domain is required")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate domains: {duplicates}")
    unknown = [name for name in names if name not in DOMAIN_REGISTRY]
    if unknown:
        raise ValueError(f"unknown domains {unknown}; known: {sorted(DOMAIN_REGISTRY)}")
    return names


def normalize_text(text: str) -> str:
    """Apply Unicode NFKC, collapse every whitespace run to a space and strip."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def strip_imdb_markup(text: str) -> str:
    """Replace IMDB ``<br />`` tags with newlines."""
    return _BR_TAG.sub("\n", text)


def is_wikitext_title(
    line: str, previous: str | None = "", following: str | None = ""
) -> bool:
    """Return whether a raw WikiText line is a top-level article title.

    The line, with only its trailing newline stripped, must be
    ``" = Title = "`` with a single ``=`` level, and the raw lines before
    and after it must be empty (``None`` stands for the start or end of the
    split). Section headers (``" = = Section = = "``) are never titles.
    """
    if not _WIKITEXT_TITLE.fullmatch(line.rstrip("\n")):
        return False
    return all(
        neighbour is None or not neighbour.strip()
        for neighbour in (previous, following)
    )


def regroup_wikitext_articles(lines: Iterable[str]) -> list[str]:
    """Group raw WikiText lines into articles starting at each title line.

    Titles are detected on the raw lines (empty lines included) and empty
    lines are dropped afterwards; section headers stay in their article.
    Lines before the first title form their own document.
    """
    raw = list(lines)
    articles: list[list[str]] = []
    current: list[str] = []
    for index, line in enumerate(raw):
        previous = raw[index - 1] if index > 0 else None
        following = raw[index + 1] if index + 1 < len(raw) else None
        if is_wikitext_title(line, previous, following) and current:
            articles.append(current)
            current = []
        if line.strip():
            current.append(line)
    if current:
        articles.append(current)
    return ["\n".join(article) for article in articles]


def detokenize_wikitext(text: str) -> str:
    """Replace the ``@-@``, ``@,@`` and ``@.@`` separators with ``-``, ``,``, ``.``."""
    return _WIKITEXT_SEPARATOR.sub(r"\1", text)


def prepare_documents(spec: DomainSpec, rows: Sequence[str]) -> list[str]:
    """Return the normalized, non-empty documents of one split."""
    if spec.kind == "wikitext":
        raw = [detokenize_wikitext(text) for text in regroup_wikitext_articles(rows)]
    elif spec.kind == "imdb":
        raw = [strip_imdb_markup(row) for row in rows]
    elif spec.kind == "plain":
        raw = list(rows)
    else:
        raise ValueError(f"unknown domain kind {spec.kind!r}")
    documents = [normalize_text(text) for text in raw]
    return [text for text in documents if text]


def document_hash(text: str) -> str:
    """Return the SHA-256 of a normalized document."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_domains(
    train: Mapping[str, Sequence[str]],
    valid: Mapping[str, Sequence[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, Any]]:
    """Drop exact duplicates by document SHA-256.

    In order: duplicates within each domain's train set (first kept), train
    documents already present in an earlier domain's train set (domains in
    ``train`` order), duplicates within each validation set, and validation
    documents whose hash appears in any domain's train set.
    """
    seen_train: dict[str, str] = {}
    train_out: dict[str, list[str]] = {}
    report: dict[str, dict[str, int]] = {
        "train_input": {},
        "train_within_domain_duplicates": {},
        "train_cross_domain_duplicates": {},
        "train_kept": {},
        "valid_input": {},
        "valid_within_domain_duplicates": {},
        "valid_train_overlap_removed": {},
        "valid_kept": {},
    }
    for name, docs in train.items():
        own: set[str] = set()
        kept: list[str] = []
        within = cross = 0
        for doc in docs:
            digest = document_hash(doc)
            if digest in own:
                within += 1
                continue
            own.add(digest)
            if digest in seen_train:
                cross += 1
                continue
            kept.append(doc)
        for digest in own:
            seen_train.setdefault(digest, name)
        train_out[name] = kept
        report["train_input"][name] = len(docs)
        report["train_within_domain_duplicates"][name] = within
        report["train_cross_domain_duplicates"][name] = cross
        report["train_kept"][name] = len(kept)

    valid_out: dict[str, list[str]] = {}
    for name, docs in valid.items():
        own = set()
        kept = []
        within = overlap = 0
        for doc in docs:
            digest = document_hash(doc)
            if digest in own:
                within += 1
                continue
            own.add(digest)
            if digest in seen_train:
                overlap += 1
                continue
            kept.append(doc)
        valid_out[name] = kept
        report["valid_input"][name] = len(docs)
        report["valid_within_domain_duplicates"][name] = within
        report["valid_train_overlap_removed"][name] = overlap
        report["valid_kept"][name] = len(kept)
    return train_out, valid_out, report


def shuffle_documents(docs: Sequence[str], seed: int) -> list[str]:
    """Return ``docs`` in a permutation fixed by ``seed``."""
    order = np.random.default_rng(seed).permutation(len(docs))
    return [docs[int(index)] for index in order]


def check_token_range(ids: np.ndarray, vocab_size: int) -> None:
    """Raise unless every id lies in ``[0, vocab_size)``."""
    ids = np.asarray(ids)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= vocab_size):
        raise ValueError(
            f"token ids outside [0, {vocab_size}): "
            f"min {int(ids.min())}, max {int(ids.max())}"
        )


@dataclass
class TokenStats:
    """Running fragmentation and ``[UNK]`` counts of tokenized documents."""

    documents: int = 0
    words: int = 0
    tokens: int = 0
    unk_tokens: int = 0

    def summary(self) -> dict[str, float | int]:
        return {
            **asdict(self),
            "tokens_per_word": self.tokens / self.words if self.words else 0.0,
            "unk_rate": self.unk_tokens / self.tokens if self.tokens else 0.0,
        }


def iter_tokenized(
    tokenizer: Any,
    docs: Sequence[str],
    *,
    vocab_size: int,
    unk_token_id: int,
    stats: TokenStats,
    chunk_size: int = TOKENIZE_CHUNK,
) -> Iterator[np.ndarray]:
    """Tokenize documents lazily in order, checking ids and updating ``stats``."""
    for start in range(0, len(docs), chunk_size):
        chunk = list(docs[start : start + chunk_size])
        encodings = tokenizer.encode_batch(chunk, add_special_tokens=False)
        for text, encoding in zip(chunk, encodings):
            ids = np.asarray(encoding.ids, dtype=np.int64)
            check_token_range(ids, vocab_size)
            stats.documents += 1
            stats.words += len(text.split())
            stats.tokens += int(ids.size)
            stats.unk_tokens += int(np.sum(ids == unk_token_id))
            yield ids


def build_train_stream(
    token_docs: Iterable[np.ndarray],
    *,
    tokens_per_domain: int,
    sep_token_id: int,
    vocab_size: int,
) -> np.ndarray:
    """Join documents with ``[SEP]`` and cut to exactly ``tokens_per_domain``.

    Documents are consumed only until the budget is met. The stream holds no
    padding, so every token counts toward the budget. Returns ``uint16``.
    """
    if tokens_per_domain < 1:
        raise ValueError("tokens_per_domain must be positive")
    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("vocabulary does not fit in uint16")
    pieces: list[np.ndarray] = []
    total = 0
    for index, doc in enumerate(token_docs):
        if index > 0:
            pieces.append(np.array([sep_token_id], dtype=np.int64))
            total += 1
        doc = np.asarray(doc, dtype=np.int64)
        check_token_range(doc, vocab_size)
        pieces.append(doc)
        total += len(doc)
        if total >= tokens_per_domain:
            break
    if total < tokens_per_domain:
        raise ValueError(
            f"domain has only {total} tokens, fewer than the "
            f"{tokens_per_domain} requested"
        )
    return np.concatenate(pieces)[:tokens_per_domain].astype(np.uint16)


def special_token_ids(tokenizer: Any) -> dict[str, int]:
    """Return the ids of the BERT special tokens, failing if any is missing."""
    ids = {token: tokenizer.token_to_id(token) for token in BERT_SPECIAL_TOKENS}
    missing = [token for token, idx in ids.items() if idx is None]
    if missing:
        raise ValueError(f"tokenizer is missing special tokens {missing}")
    return {token: int(idx) for token, idx in ids.items()}


def load_hf_split_texts(spec: DomainSpec, split: str) -> list[str]:
    """Download one pinned split and return its text field (needs network)."""
    from datasets import load_dataset

    dataset = load_dataset(
        spec.dataset_id, spec.config, revision=spec.revision, split=split
    )
    return [str(text) for text in dataset[spec.text_field]]


def load_hf_domain_rows(spec: DomainSpec) -> dict[str, list[str]]:
    """Return ``{"train": rows, "valid": rows}`` for a pinned domain."""
    train_rows: list[str] = []
    for split in spec.train_splits:
        train_rows.extend(load_hf_split_texts(spec, split))
    return {"train": train_rows, "valid": load_hf_split_texts(spec, spec.valid_split)}


def build_domain_caches(
    rows: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    tokenizer_json: str,
    out_dir: str | Path,
    tokens_per_domain: int,
    seq_len: int,
    data_seed: int = 0,
    num_valid_windows: int = DEFAULT_NUM_VALID_WINDOWS,
    mask_prob: float = 0.15,
    specs: Mapping[str, DomainSpec] = DOMAIN_REGISTRY,
) -> dict[str, Any]:
    """Build ``<out_dir>/<domain>/{train.npy, valid_windows.npz, manifest.json}``.

    ``rows`` maps each domain to its raw ``train`` and ``valid`` rows, so the
    whole pipeline runs on in-memory text. Writes ``dataset_manifest.json``
    and returns it. Validation sets keep ``num_valid_windows`` windows, or
    the minimum available across domains when any has fewer.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(tokenizer_json)
    vocab_size = int(tokenizer.get_vocab_size())
    special = special_token_ids(tokenizer)
    names = list(rows)
    prepared_train = {
        name: prepare_documents(specs[name], rows[name]["train"]) for name in names
    }
    prepared_valid = {
        name: prepare_documents(specs[name], rows[name]["valid"]) for name in names
    }
    train_docs, valid_docs, dedup = deduplicate_domains(prepared_train, prepared_valid)

    out_path = Path(out_dir)
    train_stats: dict[str, TokenStats] = {}
    train_info: dict[str, dict[str, Any]] = {}
    for name in names:
        stats = TokenStats()
        stream = build_train_stream(
            iter_tokenized(
                tokenizer,
                shuffle_documents(train_docs[name], data_seed),
                vocab_size=vocab_size,
                unk_token_id=special["[UNK]"],
                stats=stats,
            ),
            tokens_per_domain=tokens_per_domain,
            sep_token_id=special["[SEP]"],
            vocab_size=vocab_size,
        )
        train_path = atomic_save_npy(out_path / name / "train.npy", stream)
        train_stats[name] = stats
        train_info[name] = {
            "file": f"{name}/train.npy",
            "dtype": str(stream.dtype),
            "tokens": int(stream.size),
            "file_sha256": file_sha256(train_path),
            "array_sha256": arrays_sha256({"train": stream}),
        }

    windows: dict[str, np.ndarray] = {}
    valid_stats: dict[str, TokenStats] = {}
    for name in names:
        stats = TokenStats()
        windows[name] = pack_windows(
            iter_tokenized(
                tokenizer,
                shuffle_documents(valid_docs[name], VALID_SHUFFLE_SEED),
                vocab_size=vocab_size,
                unk_token_id=special["[UNK]"],
                stats=stats,
            ),
            seq_len=seq_len,
            cls_token_id=special["[CLS]"],
            sep_token_id=special["[SEP]"],
            max_windows=num_valid_windows,
        )
        valid_stats[name] = stats
    available = {name: int(len(value)) for name, value in windows.items()}
    used_windows = min(available.values())
    if used_windows < 1:
        raise ValueError(f"a domain has no full validation window: {available}")

    tokenizer_sha = sha256_text(tokenizer_json)
    manifests: dict[str, dict[str, Any]] = {}
    for name in names:
        arrays = materialize_validation_set(
            windows[name][:used_windows],
            mask_token_id=special["[MASK]"],
            vocab_size=vocab_size,
            special_token_ids=set(special.values()),
            seed=VALID_MASK_SEED,
            mask_prob=mask_prob,
        )
        valid_sha = save_validation_arrays(
            out_path / name / "valid_windows.npz", arrays
        )
        spec = specs[name]
        manifest = {
            "format": CACHE_FORMAT,
            "version": CACHE_VERSION,
            "domain": name,
            "source": {
                "dataset_id": spec.dataset_id,
                "config": spec.config,
                "revision": spec.revision,
                "text_field": spec.text_field,
                "train_splits": list(spec.train_splits),
                "valid_split": spec.valid_split,
                "raw_train_rows": len(rows[name]["train"]),
                "raw_valid_rows": len(rows[name]["valid"]),
            },
            "notes": list(spec.notes),
            "normalization": "NFKC, whitespace collapsed, stripped",
            "detokenization": (
                WIKITEXT_DETOKENISATION_NOTE if spec.kind == "wikitext" else None
            ),
            "documents": {
                "train_prepared": len(prepared_train[name]),
                "valid_prepared": len(prepared_valid[name]),
                "train_kept": len(train_docs[name]),
                "valid_kept": len(valid_docs[name]),
            },
            "tokenizer_sha256": tokenizer_sha,
            "vocab_size": vocab_size,
            "special_token_ids": special,
            "seq_len": int(seq_len),
            "train": {
                **train_info[name],
                "data_seed": int(data_seed),
                "tokens_per_domain": int(tokens_per_domain),
                "token_stats": train_stats[name].summary(),
            },
            "valid": {
                "file": f"{name}/valid_windows.npz",
                "arrays_sha256": valid_sha,
                "num_windows": int(used_windows),
                "available_windows": available[name],
                "requested_windows": int(num_valid_windows),
                "shuffle_seed": VALID_SHUFFLE_SEED,
                "mask_seed": VALID_MASK_SEED,
                "mask_prob": float(mask_prob),
                "masked_tokens": int(arrays["loss_mask"].sum()),
                "token_stats": valid_stats[name].summary(),
            },
        }
        atomic_write_json(out_path / name / "manifest.json", manifest)
        manifests[name] = manifest

    dataset_manifest = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "domains": names,
        "tokenizer_sha256": tokenizer_sha,
        "vocab_size": vocab_size,
        "seq_len": int(seq_len),
        "tokens_per_domain": int(tokens_per_domain),
        "data_seed": int(data_seed),
        "num_valid_windows": int(used_windows),
        "requested_valid_windows": int(num_valid_windows),
        "available_valid_windows": available,
        "valid_windows_reduced": used_windows < num_valid_windows,
        "dedup": dedup,
        "dedup_policy": DEDUP_POLICY,
        "overlap_notes": [WIKITEXT_OVERLAP_NOTE] if "wiki" in names else [],
        "per_domain": manifests,
    }
    atomic_write_json(out_path / "dataset_manifest.json", dataset_manifest)
    return dataset_manifest


@dataclass
class DomainCache:
    """A verified domain cache: training stream, validation arrays, manifest."""

    name: str
    train: np.ndarray
    valid: dict[str, np.ndarray]
    manifest: dict[str, Any]


def load_domain_cache(
    data_dir: str | Path,
    name: str,
    *,
    tokenizer_json: str,
    seq_len: int,
    tokens_per_domain: int | None = None,
    max_valid_windows: int | None = None,
) -> DomainCache:
    """Load and checksum-verify one domain cache.

    The cache must have been built with this tokenizer and ``seq_len``. The
    first ``tokens_per_domain`` stream tokens and ``max_valid_windows``
    validation windows are kept when given.
    """
    domain_dir = Path(data_dir) / name
    manifest = json.loads((domain_dir / "manifest.json").read_text())
    if manifest.get("format") != CACHE_FORMAT:
        raise ValueError(f"{domain_dir}: not a domain-forgetting cache")
    if manifest["tokenizer_sha256"] != sha256_text(tokenizer_json):
        raise ValueError(f"{domain_dir}: cache was built with a different tokenizer")
    if int(manifest["seq_len"]) != int(seq_len):
        raise ValueError(
            f"{domain_dir}: cache seq_len {manifest['seq_len']} != model {seq_len}"
        )
    train_path = domain_dir / "train.npy"
    if file_sha256(train_path) != manifest["train"]["file_sha256"]:
        raise ValueError(f"{train_path}: SHA-256 does not match the manifest")
    train = np.load(train_path, allow_pickle=False)
    if tokens_per_domain is not None:
        if tokens_per_domain > len(train):
            raise ValueError(
                f"{train_path}: {len(train)} tokens cached, "
                f"{tokens_per_domain} requested"
            )
        train = train[:tokens_per_domain]
    valid = load_validation_arrays(
        domain_dir / "valid_windows.npz",
        expected_sha256=manifest["valid"]["arrays_sha256"],
        max_windows=max_valid_windows,
    )
    return DomainCache(name=name, train=train, valid=valid, manifest=manifest)

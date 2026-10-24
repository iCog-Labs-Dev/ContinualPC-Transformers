#!/usr/bin/env python3
"""Convert a Hugging Face ``BertForMaskedLM`` into FabricPC BERT params.

The output is a ``fabricpc.utils.save_params`` file pair (``.npz`` plus
``.json``) for the ``create_encoder_mlm_model(architecture="bert")`` graph,
with the vocab, WordPiece ``tokenizer_json`` and model config as metadata. It
loads with ``bert_demo.py --architecture bert --init_params <out>
--tokenizer checkpoint``. A ``<stem>.tokenizer.json`` is written next to it.

Token type 0 is folded into the position table, the pooler and NSP head are
dropped, and the decoder weight is copied untied from the word embeddings.
Loading the Hugging Face model needs the optional ``hf`` extra (torch and
transformers); ``convert_bert_state_dict`` works on numpy arrays alone.

Usage:
    PYTHONPATH=. python examples/convert_hf_bert.py \\
        --model google/bert_uncased_L-4_H-256_A-4 --seq_len 128 \\
        --out checkpoints/bert_mini_128.npz
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.types import GraphParams, GraphStructure, NodeParams
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.utils.params_io import save_params

BERT_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
UNUSED_CHECKPOINT_KEY_PATTERNS = (
    r"(bert\.)?pooler\..*",
    r"cls\.seq_relationship\..*",
)
IGNORED_KEY_PATTERNS = (
    r"bert\.pooler\..*",
    r"cls\.seq_relationship\..*",
    r"bert\.embeddings\.position_ids",
    r"bert\.embeddings\.token_type_ids",
)


def _param_shapes(params: GraphParams) -> dict[tuple[str, str, str], tuple]:
    return {
        (node_name, collection, name): tuple(value.shape)
        for node_name, node in params.nodes.items()
        for collection in ("weights", "biases")
        for name, value in getattr(node, collection).items()
    }


def _transposed(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value).T)


def _bert_key_map(depth: int) -> dict[str, tuple[str, str, str, bool]]:
    """Return ``hf_key -> (node, collection, param, transpose)`` for direct copies."""
    mapping = {
        "bert.embeddings.word_embeddings.weight": (
            "token_embeddings",
            "weights",
            "embeddings",
            False,
        ),
        "bert.embeddings.LayerNorm.weight": (
            "position_embeddings",
            "weights",
            "ln_gamma",
            False,
        ),
        "bert.embeddings.LayerNorm.bias": (
            "position_embeddings",
            "biases",
            "ln_beta",
            False,
        ),
        "cls.predictions.transform.dense.weight": (
            "mlm_transform",
            "weights",
            "W_t",
            True,
        ),
        "cls.predictions.transform.dense.bias": (
            "mlm_transform",
            "biases",
            "b_t",
            False,
        ),
        "cls.predictions.transform.LayerNorm.weight": (
            "mlm_transform",
            "weights",
            "ln_gamma",
            False,
        ),
        "cls.predictions.transform.LayerNorm.bias": (
            "mlm_transform",
            "biases",
            "ln_beta",
            False,
        ),
        "cls.predictions.decoder.weight": ("mlm_head", "weights", "W_out", True),
        "cls.predictions.bias": ("mlm_head", "biases", "b_out", False),
    }
    for i in range(depth):
        prefix = f"bert.encoder.layer.{i}"
        mha, mlp1, mlp2 = f"L{i}_mha", f"L{i}_mlp1", f"L{i}_mlp2"
        for hf_name, ours in (("query", "q"), ("key", "k"), ("value", "v")):
            mapping[f"{prefix}.attention.self.{hf_name}.weight"] = (
                mha,
                "weights",
                f"W_{ours}",
                True,
            )
            mapping[f"{prefix}.attention.self.{hf_name}.bias"] = (
                mha,
                "biases",
                f"b_{ours}",
                False,
            )
        mapping.update(
            {
                f"{prefix}.attention.output.dense.weight": (
                    mha,
                    "weights",
                    "W_o",
                    True,
                ),
                f"{prefix}.attention.output.dense.bias": (mha, "biases", "b_o", False),
                f"{prefix}.attention.output.LayerNorm.weight": (
                    mha,
                    "weights",
                    "ln_gamma",
                    False,
                ),
                f"{prefix}.attention.output.LayerNorm.bias": (
                    mha,
                    "biases",
                    "ln_beta",
                    False,
                ),
                f"{prefix}.intermediate.dense.weight": (mlp1, "weights", "W_ff1", True),
                f"{prefix}.intermediate.dense.bias": (mlp1, "biases", "b_ff1", False),
                f"{prefix}.output.dense.weight": (mlp2, "weights", "W_ff2", True),
                f"{prefix}.output.dense.bias": (mlp2, "biases", "b_ff2", False),
                f"{prefix}.output.LayerNorm.weight": (
                    mlp2,
                    "weights",
                    "ln_gamma",
                    False,
                ),
                f"{prefix}.output.LayerNorm.bias": (mlp2, "biases", "ln_beta", False),
            }
        )
    return mapping


def _check_bert_config(config: Mapping[str, Any], seq_len: int) -> None:
    if config.get("hidden_act", "gelu") != "gelu":
        raise ValueError(f"only erf GELU is supported, got {config['hidden_act']!r}")
    if config.get("position_embedding_type", "absolute") != "absolute":
        raise ValueError("only absolute position embeddings are supported")
    if not 0 < seq_len <= int(config["max_position_embeddings"]):
        raise ValueError(
            f"seq_len must be in [1, {config['max_position_embeddings']}], got {seq_len}"
        )


def convert_bert_state_dict(
    state_dict: Mapping[str, np.ndarray],
    *,
    config: Mapping[str, Any],
    seq_len: int,
) -> GraphParams:
    """Map a ``BertForMaskedLM`` state dict to FabricPC ``architecture="bert"`` params.

    ``config`` is the Hugging Face config as a dict (``num_hidden_layers``,
    ``max_position_embeddings``, ...). Torch ``Linear`` weights are ``(out, in)``
    and are transposed. The position table is the first ``seq_len`` rows plus
    token type 0. A tied ``cls.predictions.decoder.bias`` must equal
    ``cls.predictions.bias``. Pooler, NSP and buffer keys are ignored; any
    other unmapped key or any missing key raises ``ValueError``. Only nodes
    with parameters are returned.
    """
    _check_bert_config(config, seq_len)
    depth = int(config["num_hidden_layers"])
    mapping = _bert_key_map(depth)
    special = {
        "bert.embeddings.position_embeddings.weight",
        "bert.embeddings.token_type_embeddings.weight",
        "cls.predictions.decoder.bias",
    }
    ignored = [
        key
        for key in state_dict
        if any(re.fullmatch(pattern, key) for pattern in IGNORED_KEY_PATTERNS)
    ]
    unknown = sorted(set(state_dict) - set(mapping) - special - set(ignored))
    if unknown:
        raise ValueError(f"unmapped state dict keys: {unknown}")
    required = set(mapping) | (special - {"cls.predictions.decoder.bias"})
    missing = sorted(required - set(state_dict))
    if missing:
        raise ValueError(f"missing state dict keys: {missing}")

    nodes: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    def put(node: str, collection: str, name: str, value: np.ndarray) -> None:
        entry = nodes.setdefault(node, {"weights": {}, "biases": {}})
        entry[collection][name] = np.asarray(value, dtype=np.float32)

    for key, (node, collection, name, transpose) in mapping.items():
        value = np.asarray(state_dict[key])
        put(node, collection, name, _transposed(value) if transpose else value)

    positions = np.asarray(state_dict["bert.embeddings.position_embeddings.weight"])
    token_type = np.asarray(state_dict["bert.embeddings.token_type_embeddings.weight"])
    put(
        "position_embeddings",
        "weights",
        "position_embeddings",
        positions[:seq_len] + token_type[0][None, :],
    )
    if "cls.predictions.decoder.bias" in state_dict and not np.array_equal(
        state_dict["cls.predictions.decoder.bias"], state_dict["cls.predictions.bias"]
    ):
        raise ValueError(
            "cls.predictions.decoder.bias differs from cls.predictions.bias"
        )
    return GraphParams(
        nodes={
            node: NodeParams(
                weights={k: jnp.asarray(v) for k, v in entry["weights"].items()},
                biases={k: jnp.asarray(v) for k, v in entry["biases"].items()},
            )
            for node, entry in nodes.items()
        }
    )


def bert_model_config(config: Mapping[str, Any], seq_len: int) -> dict[str, Any]:
    """Return the FabricPC model config for a Hugging Face BERT config."""
    return {
        "architecture": "bert",
        "depth": int(config["num_hidden_layers"]),
        "embed_dim": int(config["hidden_size"]),
        "num_heads": int(config["num_attention_heads"]),
        "mlp_dim": int(config["intermediate_size"]),
        "seq_len": int(seq_len),
        "vocab_size": int(config["vocab_size"]),
        "ln_eps": float(config.get("layer_norm_eps", 1e-12)),
    }


def build_bert_structure(
    model_config: Mapping[str, Any],
    dropout_rate: float = 0.0,
    residual_identity: bool = False,
) -> GraphStructure:
    """Build the FabricPC ``architecture="bert"`` MLM graph for ``model_config``.

    ``residual_identity`` adds parameterless nodes only, so the converted
    params load into the graph either way.
    """
    return create_encoder_mlm_model(
        depth=model_config["depth"],
        embed_dim=model_config["embed_dim"],
        num_heads=model_config["num_heads"],
        mlp_dim=model_config["mlp_dim"],
        seq_len=model_config["seq_len"],
        vocab_size=model_config["vocab_size"],
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=5.0),
        use_attention_mask=True,
        dropout_rate=dropout_rate,
        architecture="bert",
        ln_eps=model_config["ln_eps"],
        residual_identity=residual_identity,
    )


def graph_params_for(converted: GraphParams, structure: GraphStructure) -> GraphParams:
    """Return params for every node of ``structure`` with ``converted`` filled in.

    Raises ``ValueError`` unless ``converted`` holds exactly the graph's
    parameters with the graph's shapes.
    """
    fresh = initialize_params(structure, jax.random.PRNGKey(0))
    params = GraphParams(
        nodes={
            name: converted.nodes.get(name, node) for name, node in fresh.nodes.items()
        }
    )
    if set(converted.nodes) - set(fresh.nodes) or _param_shapes(
        params
    ) != _param_shapes(fresh):
        raise ValueError("converted parameters do not match the FabricPC BERT graph")
    return params


def build_wordpiece_tokenizer(vocab_path: str | Path, lowercase: bool = True):
    """Build a ``tokenizers.Tokenizer`` matching ``BertTokenizerFast`` for ``vocab.txt``.

    No post-processor is set: the FabricPC loaders insert ``[CLS]`` and
    ``[SEP]`` themselves.
    """
    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import WordPiece

    tokenizer = Tokenizer(WordPiece.from_file(str(vocab_path), unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=None,
        lowercase=lowercase,
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    missing = [t for t in BERT_SPECIAL_TOKENS if tokenizer.token_to_id(t) is None]
    if missing:
        raise ValueError(f"vocab is missing special tokens {missing}")
    tokenizer.add_special_tokens(BERT_SPECIAL_TOKENS)
    return tokenizer


def write_bert_params(
    out: str | Path,
    params: GraphParams,
    tokenizer,
    model_config: Mapping[str, Any],
) -> Path:
    """Write ``params`` with ``save_params`` and ``<stem>.tokenizer.json`` beside it.

    The metadata holds the vocab, ``tokenizer_json`` and ``model_config``.
    Returns the ``.npz`` path.
    """
    tokenizer_json = tokenizer.to_str()
    vocab = {str(token): int(idx) for token, idx in tokenizer.get_vocab().items()}
    if len(vocab) != model_config["vocab_size"]:
        raise ValueError(
            f"tokenizer has {len(vocab)} tokens, model has {model_config['vocab_size']}"
        )
    archive, _ = save_params(
        out,
        params,
        {
            "vocab": vocab,
            "tokenizer_json": tokenizer_json,
            "config": dict(model_config),
        },
    )
    archive.with_name(f"{archive.stem}.tokenizer.json").write_text(tokenizer_json)
    return archive


def _resolve_vocab_file(model: str, revision: str | None) -> Path:
    local = Path(model) / "vocab.txt"
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=model, filename="vocab.txt", revision=revision))


def _resolve_source_revision(model: str, revision: str | None) -> str | None:
    """Return the Hub commit sha the weights came from, or None for a local dir."""
    if Path(model).is_dir():
        return revision
    from huggingface_hub import HfApi

    return HfApi().model_info(model, revision=revision).sha


def _resolve_lowercase(model: str, revision: str | None) -> bool:
    """Return ``do_lower_case`` from the repo's ``tokenizer_config.json``, else True.

    True is ``BertTokenizerFast``'s default when the repo has no tokenizer config.
    """
    local = Path(model) / "tokenizer_config.json"
    if Path(model).is_dir():
        path = local if local.exists() else None
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        try:
            path = Path(
                hf_hub_download(
                    repo_id=model, filename="tokenizer_config.json", revision=revision
                )
            )
        except EntryNotFoundError:
            path = None
    if path is None:
        return True
    return bool(json.loads(path.read_text()).get("do_lower_case", True))


def check_loading_info(loading_info: Mapping[str, Any]) -> None:
    """Raise unless ``from_pretrained`` loaded every MLM weight from the checkpoint.

    Hugging Face fills weights missing from a checkpoint with a fresh random
    initialisation, so a head-less or renamed checkpoint would otherwise convert
    silently. Only pooler and next-sentence weights may be left unused.
    """
    problems = {
        name: sorted(loading_info.get(name) or [])
        for name in ("missing_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(name)
    }
    unexpected = sorted(
        key
        for key in loading_info.get("unexpected_keys") or []
        if not any(re.fullmatch(p, key) for p in UNUSED_CHECKPOINT_KEY_PATTERNS)
    )
    if unexpected:
        problems["unexpected_keys"] = unexpected
    if problems:
        raise ValueError(
            f"checkpoint does not hold a complete BertForMaskedLM: {problems}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="HF model id or local dir")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument(
        "--out", required=True, help=".npz path; a .json sidecar is written beside it"
    )
    parser.add_argument(
        "--lowercase",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: do_lower_case from the repo's tokenizer_config.json, else True",
    )
    args = parser.parse_args(argv)

    import torch
    from transformers import BertForMaskedLM

    model, loading_info = BertForMaskedLM.from_pretrained(
        args.model, revision=args.revision, output_loading_info=True
    )
    if model.config.model_type != "bert":
        raise ValueError(f"expected a BERT model, got {model.config.model_type!r}")
    check_loading_info(loading_info)
    model = model.to(torch.device("cpu")).eval()
    hf_config = model.config.to_dict()
    state_dict = {
        key: value.detach().cpu().numpy() for key, value in model.state_dict().items()
    }
    ignored = sorted(
        key
        for key in state_dict
        if any(re.fullmatch(pattern, key) for pattern in IGNORED_KEY_PATTERNS)
    )
    converted = convert_bert_state_dict(
        state_dict, config=hf_config, seq_len=args.seq_len
    )
    model_config = {
        **bert_model_config(hf_config, args.seq_len),
        "source_model": args.model,
        "source_revision": _resolve_source_revision(args.model, args.revision),
    }
    params = graph_params_for(converted, build_bert_structure(model_config))
    lowercase = args.lowercase
    if lowercase is None:
        lowercase = _resolve_lowercase(args.model, args.revision)
    tokenizer = build_wordpiece_tokenizer(
        _resolve_vocab_file(args.model, args.revision), lowercase=lowercase
    )
    archive = write_bert_params(args.out, params, tokenizer, model_config)
    print(f"ignored_keys={ignored}")
    print(f"params={archive}")


if __name__ == "__main__":
    main()

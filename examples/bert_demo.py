"""
BERT-style encoder: masked-language-model pretraining with backprop or PC.

Builds a bidirectional encoder (``create_encoder_mlm_model``), batches text
with ``MlmTextBatcher`` (BERT 80/10/10 masking, padding masks) and trains it
with the standard loops: ``train_backprop_autoregressive`` for ``--mode
backprop`` or ``train_autoregressive`` (predictive coding) for ``--mode pc``.
Validation reports feedforward masked-token loss and accuracy each epoch,
comparable across modes. The training loss is the backprop masked loss, or in
PC mode ``train_pc_inferred_loss``: the masked loss after inference has moved
the latents toward the targets, which reads lower than a feedforward loss.

Without arguments it runs on a small built-in corpus with a word-level
vocabulary (no extra dependencies). Pass text files (one document per line)
to pretrain on your own data; ``--tokenizer bpe`` needs the ``text`` extra
(``pip install -e '.[text]'``). ``--save_params`` writes the trained weights
with ``fabricpc.utils.save_params``; ``--init_params`` starts training from
such a file instead of a fresh initialisation. ``--architecture bert`` builds
the Hugging Face BERT layout; with ``--tokenizer checkpoint`` it trains from a
file written by ``examples/convert_hf_bert.py`` using that file's tokenizer.

Usage:
    PYTHONPATH=. python examples/bert_demo.py
    PYTHONPATH=. python examples/bert_demo.py --mode pc --pc_infer_steps 10
    PYTHONPATH=. python examples/bert_demo.py --ab_trials 5
    PYTHONPATH=. python examples/bert_demo.py --train_file train.txt \\
        --val_file val.txt --tokenizer bpe --vocab_size 4000 --seq_len 64
    PYTHONPATH=. python examples/bert_demo.py --save_params runs/mlm.npz
    PYTHONPATH=. python examples/bert_demo.py --init_params runs/mlm.npz
    PYTHONPATH=. python examples/bert_demo.py --architecture bert \\
        --init_params checkpoints/bert_mini_128.npz --tokenizer checkpoint \\
        --seq_len 128 --depth 4 --embed_dim 256 --num_heads 4 --mlp_dim 1024
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import functools
import importlib.util
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.experiments.ab_experiment import ABExperiment, ABResults, ExperimentArm
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.encoder_transformer import BERT_LN_EPS, create_encoder_mlm_model
from fabricpc.training import (
    count_training_batches,
    evaluate_mlm,
    train_autoregressive,
    train_backprop_autoregressive,
)
from fabricpc.training.train_autoregressive import (
    get_graph_param_gradient_autoregressive,
)
from fabricpc.training.train_backprop import compute_loss_autoregressive
from fabricpc.utils.data import MLM_SPECIAL_TOKENS, MlmTextBatcher, tokenize_texts_bpe
from fabricpc.utils.params_io import load_params, read_params_metadata, save_params

TINY_TRAIN_TEXTS = [
    "the cat sat on the warm mat by the door",
    "a small dog ran across the green park",
    "the bird sang a song in the tall tree",
    "the dog and the cat slept on the mat",
    "a child read a book under the tall tree",
    "the sun rose over the quiet green park",
    "the cat watched the bird in the tree",
    "a dog barked at the door of the house",
] * 4
TINY_VAL_TEXTS = [
    "the bird sat on the mat by the tree",
    "a cat ran across the park to the door",
    "the dog read a song under the warm sun",
    "a child watched the dog in the green park",
] * 4


def read_texts(path: Path) -> List[str]:
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    texts = [text for text in texts if text]
    if not texts:
        raise ValueError(f"{path} contains no non-empty lines")
    return texts


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_tokenize(
    train_texts: Sequence[str], val_texts: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Word-level vocabulary from the training texts; unknown words -> [UNK].

    Words are lowercased runs of ASCII letters, digits and apostrophes; any
    other character is dropped, so use ``--tokenizer bpe`` for non-ASCII text.
    """
    vocab = {token: idx for idx, token in enumerate(MLM_SPECIAL_TOKENS)}
    for word in sorted({w for text in train_texts for w in _words(text)}):
        vocab.setdefault(word, len(vocab))

    def encode(texts: Sequence[str]) -> np.ndarray:
        ids: List[int] = []
        for text in texts:
            ids.extend(vocab.get(word, vocab["[UNK]"]) for word in _words(text))
            ids.append(vocab["[SEP]"])
        return np.asarray(ids, dtype=np.int32)

    return encode(train_texts), encode(val_texts), vocab


def bpe_tokenize(
    train_texts: Sequence[str], val_texts: Sequence[str], vocab_size: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], str]:
    """BPE token ids, vocabulary and the trained tokenizer's JSON."""
    train_ids, tokenizer = tokenize_texts_bpe(train_texts, vocab_size=vocab_size)
    val_ids, _ = tokenize_texts_bpe(
        val_texts, vocab_size=vocab_size, tokenizer=tokenizer
    )
    return train_ids, val_ids, tokenizer.get_vocab(), tokenizer.to_str()


def checkpoint_tokenize(
    train_texts: Sequence[str], val_texts: Sequence[str], path: Path
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], str]:
    """Token ids from the tokenizer saved with the params at ``path``."""
    tokenizer_json = read_params_metadata(path).get("tokenizer_json")
    if tokenizer_json is None:
        raise ValueError(f"{path} holds no tokenizer_json for --tokenizer checkpoint")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(tokenizer_json)
    train_ids, _ = tokenize_texts_bpe(train_texts, vocab_size=0, tokenizer=tokenizer)
    val_ids, _ = tokenize_texts_bpe(val_texts, vocab_size=0, tokenizer=tokenizer)
    return train_ids, val_ids, tokenizer.get_vocab(), tokenizer_json


def make_loader(
    token_ids: np.ndarray, vocab: Dict[str, int], args, *, seed: int, shuffle: bool
) -> MlmTextBatcher:
    loader = MlmTextBatcher(
        token_ids=token_ids,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        pad_token_id=vocab["[PAD]"],
        mask_token_id=vocab["[MASK]"],
        vocab_size=len(vocab),
        special_token_ids={vocab[token] for token in MLM_SPECIAL_TOKENS},
        cls_token_id=vocab["[CLS]"],
        seed=seed,
        mask_prob=args.mask_prob,
        shuffle=shuffle,
    )
    if len(loader) == 0:
        raise ValueError(
            f"{len(token_ids)} tokens give no full batch of {args.batch_size} "
            f"windows at seq_len={args.seq_len}; use more text or smaller sizes"
        )
    return loader


def count_training_steps(num_epochs: float, batches_per_epoch: int) -> int:
    """Optimizer steps train_backprop_autoregressive runs for ``num_epochs``.

    Raises if ``num_epochs`` is not positive or its fractional final epoch
    would run no batch, which the training loop rejects.
    """
    if num_epochs <= 0:
        raise ValueError("--num_epochs must be positive")
    fraction = num_epochs - math.floor(num_epochs)
    if fraction > 0 and round(fraction * batches_per_epoch) == 0:
        raise ValueError(
            f"--num_epochs {num_epochs} leaves no batch in its final partial "
            f"epoch ({batches_per_epoch} batches per epoch); use a whole number "
            "or a larger fraction"
        )
    return count_training_batches(num_epochs, batches_per_epoch)


def build_optimizer(
    args, total_steps: int, grad_scale: float = 1.0
) -> optax.GradientTransformation:
    """AdamW with linear warmup, cosine decay and global-norm clipping.

    Gradients are multiplied by ``grad_scale`` before clipping.
    """
    if not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=int(args.warmup_fraction * total_steps),
        decay_steps=total_steps,
    )
    adamw = optax.adamw(schedule, weight_decay=args.weight_decay)
    if args.grad_clip <= 0:
        return optax.chain(optax.scale(grad_scale), adamw)
    return optax.chain(
        optax.scale(grad_scale), optax.clip_by_global_norm(args.grad_clip), adamw
    )


def expected_masked_tokens(loader, max_batches: int = 50) -> float:
    """Mean ``loss_mask`` count over the loader's first batches.

    The loader's state is restored so training sees the same batches.
    """
    state = loader.state_dict()
    counts = []
    for batch in loader:
        counts.append(float(np.sum(batch["loss_mask"])))
        if len(counts) == max_batches:
            break
    loader.load_state_dict(state)
    return float(np.mean(counts))


def pc_grad_scale(loader) -> float:
    """Scale that approximately matches PC weight gradients to backprop's per token.

    PC weight gradients sum over masked tokens while the backprop loss averages
    them. Dividing by the mean masked count per batch (estimated from the
    loader's first batches) makes ``--grad_clip`` act on comparable norms in
    both modes. Per-batch counts vary, so the match is approximate.
    """
    return 1.0 / expected_masked_tokens(loader)


TRAINERS = {
    mode: functools.partial(trainer, use_pmap=False)
    for mode, trainer in (
        ("backprop", train_backprop_autoregressive),
        ("pc", train_autoregressive),
    )
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT-style MLM pretraining")
    parser.add_argument("--mode", choices=sorted(TRAINERS), default="backprop")
    parser.add_argument("--train_file", type=Path, default=None)
    parser.add_argument("--val_file", type=Path, default=None)
    parser.add_argument(
        "--tokenizer",
        choices=["word", "bpe", "checkpoint"],
        default="word",
        help="word: ASCII words, no extra deps; bpe: needs the text extra; "
        "checkpoint: the tokenizer saved with --init_params",
    )
    parser.add_argument("--vocab_size", type=int, default=4000, help="BPE only")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument(
        "--architecture",
        choices=["encoder", "bert"],
        default="encoder",
        help="encoder: pre-LN with a final LayerNorm; bert: post-LN Hugging Face "
        "BERT layout, e.g. for params from convert_hf_bert.py",
    )
    parser.add_argument(
        "--ln_eps",
        type=float,
        default=None,
        help="LayerNorm epsilon for --architecture bert (default: the "
        "--init_params file's, else 1e-12)",
    )
    parser.add_argument(
        "--residual_identity",
        action="store_true",
        help="--architecture bert only: route each block's attention residual "
        "sum to its second feed-forward node through an identity node",
    )
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--num_epochs", type=float, default=10)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument(
        "--warmup_fraction", type=float, default=0.1, help="of total steps"
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="global-norm clip; in PC mode gradients are first divided by the "
        "mean masked-token count per batch, so the clip approximately matches "
        "backprop's per-token scale",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pc_infer_steps", type=int, default=40, help="PC only")
    parser.add_argument("--pc_eta_infer", type=float, default=0.1, help="PC only")
    parser.add_argument("--pc_max_norm", type=float, default=5.0, help="PC only")
    parser.add_argument(
        "--pc_output_energy_scale", type=float, default=1.0, help="PC only"
    )
    parser.add_argument(
        "--ab_trials",
        type=int,
        default=0,
        help="Compare backprop vs PC over this many paired seeds instead of "
        "training --mode (needs scipy, from the experiments extra)",
    )
    parser.add_argument(
        "--ab_metric", choices=["masked_loss", "masked_accuracy"], default="masked_loss"
    )
    parser.add_argument(
        "--save_params",
        type=Path,
        default=None,
        help="Write the trained params to this .npz (plus a .json sidecar)",
    )
    parser.add_argument(
        "--init_params",
        type=Path,
        default=None,
        help="Start training from params saved with --save_params; the vocab, "
        "--seq_len and architecture must match",
    )
    args = parser.parse_args(argv)
    if (args.train_file is None) != (args.val_file is None):
        parser.error("--train_file and --val_file must be given together")
    if args.ab_trials > 0 and (args.save_params or args.init_params):
        parser.error("--save_params and --init_params are not used with --ab_trials")
    if args.tokenizer == "checkpoint" and args.init_params is None:
        parser.error("--tokenizer checkpoint needs --init_params")
    if args.architecture != "bert" and (
        args.ln_eps is not None or args.residual_identity
    ):
        parser.error("--ln_eps and --residual_identity need --architecture bert")
    if args.ab_trials > 0 and importlib.util.find_spec("scipy") is None:
        parser.error("--ab_trials needs scipy: pip install -e '.[experiments]'")
    return args


def build_model(args, vocab_size: int):
    """Encoder MLM graph; the PC inference settings only matter for ``--mode pc``."""
    return create_encoder_mlm_model(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        dropout_rate=args.dropout_rate,
        pc_output_energy_scale=args.pc_output_energy_scale,
        architecture=args.architecture,
        ln_eps=args.ln_eps,
        residual_identity=args.residual_identity,
        inference=InferenceSGDNormClip(
            eta_infer=args.pc_eta_infer,
            infer_steps=args.pc_infer_steps,
            max_norm=args.pc_max_norm,
        ),
    )


def load_token_streams(
    args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Optional[str]]:
    """Tokenize the training and validation texts into token-id streams.

    Returns the train and validation ids, the vocabulary and the tokenizer's
    JSON (``None`` for the word tokenizer, whose vocabulary is enough).
    """
    if args.train_file is None:
        train_texts, val_texts = TINY_TRAIN_TEXTS, TINY_VAL_TEXTS
    else:
        train_texts, val_texts = read_texts(args.train_file), read_texts(args.val_file)
    if args.tokenizer == "bpe":
        return bpe_tokenize(train_texts, val_texts, args.vocab_size)
    if args.tokenizer == "checkpoint":
        return checkpoint_tokenize(train_texts, val_texts, args.init_params)
    return (*word_tokenize(train_texts, val_texts), None)


def resolve_ln_eps(args) -> Optional[float]:
    """LayerNorm epsilon for ``--architecture bert``; ``None`` for the encoder.

    An unset ``--ln_eps`` takes the value recorded with ``--init_params``, else
    BERT's ``1e-12``.
    """
    if args.architecture != "bert":
        return None
    if args.ln_eps is not None:
        return float(args.ln_eps)
    if args.init_params is not None:
        saved = read_params_metadata(args.init_params).get("config", {}).get("ln_eps")
        if saved is not None:
            return float(saved)
    return BERT_LN_EPS


def model_config(args, vocab_size: int) -> Dict[str, Any]:
    """Model settings saved with ``--save_params`` and checked by ``--init_params``."""
    architecture = {"architecture": args.architecture}
    if args.architecture == "bert":
        architecture["ln_eps"] = (
            BERT_LN_EPS if args.ln_eps is None else float(args.ln_eps)
        )
        architecture["residual_identity"] = bool(args.residual_identity)
    return {
        **architecture,
        "depth": args.depth,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "mlp_dim": args.mlp_dim,
        "seq_len": args.seq_len,
        "vocab_size": vocab_size,
        "dropout_rate": args.dropout_rate,
    }


def load_init_params(
    path: Path, structure: GraphStructure, vocab: Dict[str, int], args
) -> GraphParams:
    """Params saved by ``--save_params`` to start training from.

    The file's architecture (``"encoder"`` when it records none), ``seq_len``,
    ``ln_eps`` (when both record one) and vocabulary must match the current run
    before its parameters are checked against ``structure``.
    """
    metadata = read_params_metadata(path)
    config = metadata.get("config", {})
    expected = model_config(args, len(vocab))
    for key in ("architecture", "seq_len", "ln_eps"):
        saved = config.get(key, "encoder" if key == "architecture" else None)
        if saved is not None and key in expected and saved != expected[key]:
            raise ValueError(
                f"{path}: saved {key} {saved!r} does not match this run's "
                f"{key} {expected[key]!r}"
            )
    saved_vocab = metadata.get("vocab")
    if saved_vocab is not None and {
        str(token): int(idx) for token, idx in saved_vocab.items()
    } != dict(vocab):
        raise ValueError(f"{path}: saved vocab does not match the training vocab")
    return load_params(path, structure)[0]


def make_loaders(args, train_ids, val_ids, vocab, seed: int):
    return (
        make_loader(train_ids, vocab, args, seed=seed, shuffle=True),
        make_loader(val_ids, vocab, args, seed=seed + 1, shuffle=False),
    )


def print_data_summary(train_ids, val_ids, vocab, batches_per_epoch: int) -> None:
    print(
        f"vocab={len(vocab)} train_tokens={len(train_ids)} "
        f"val_tokens={len(val_ids)} batches_per_epoch={batches_per_epoch}"
    )


NEGLIGIBLE_GRADIENT_RATIO = 1e-3
MISALIGNED_GRADIENT_COSINE = 0.5


def _tree_norm(tree) -> float:
    return float(
        jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(tree)))
    )


def _tree_dot(a, b) -> float:
    return float(
        sum(
            jnp.sum(x * y)
            for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
        )
    )


def warn_weak_pc_gradients(structure, params, loader, rng_key) -> Dict[str, List[str]]:
    """Warn about nodes whose PC gradient on the first batch falls short of backprop's.

    Each node's PC weight gradient, divided by the batch's masked-token count,
    is compared with the backprop gradient of the masked loss. A node is
    ``negligible`` when that norm ratio is below ``NEGLIGIBLE_GRADIENT_RATIO``
    and ``misaligned`` when the cosine is below ``MISALIGNED_GRADIENT_COSINE``.
    How far the output error reaches depends on ``--pc_infer_steps`` times
    ``--pc_eta_infer`` and on float32 precision, not on step count alone. The
    loader's state is restored so training sees the same batches.
    """
    state = loader.state_dict()
    batch = {k: jnp.asarray(v) for k, v in next(iter(loader)).items()}
    loader.load_state_dict(state)
    pc_grads, _, _, _ = get_graph_param_gradient_autoregressive(
        params, batch, structure, rng_key
    )
    bp_grads = jax.grad(
        lambda p: compute_loss_autoregressive(p, structure, batch, rng_key)[0]
    )(params)
    masked = float(jnp.sum(batch["loss_mask"]))

    weak: Dict[str, List[str]] = {"negligible": [], "misaligned": []}
    for name, pc_node in pc_grads.nodes.items():
        bp_norm = _tree_norm(bp_grads.nodes[name])
        if not jax.tree_util.tree_leaves(pc_node) or bp_norm == 0.0:
            continue
        pc_norm = _tree_norm(pc_node) / masked
        if pc_norm / bp_norm < NEGLIGIBLE_GRADIENT_RATIO:
            weak["negligible"].append(name)
        elif (
            _tree_dot(pc_node, bp_grads.nodes[name]) / masked / (pc_norm * bp_norm)
            < MISALIGNED_GRADIENT_COSINE
        ):
            weak["misaligned"].append(name)

    config = structure.config["inference"].config
    for kind, names in weak.items():
        if names:
            print(
                f"warning: with --pc_infer_steps {config['infer_steps']} and "
                f"--pc_eta_infer {config['eta_infer']} these nodes get a {kind} "
                f"PC gradient relative to backprop: {', '.join(names)}",
                file=sys.stderr,
            )
    return weak


def pretrain(
    args, train_ids, val_ids, vocab, tokenizer_json: Optional[str] = None
) -> Dict[str, float]:
    """Pretrain with ``--mode`` and return the final validation metrics.

    Starts from ``--init_params`` when given and writes the trained params to
    ``--save_params`` with the vocab, ``tokenizer_json`` and model config.
    """
    train_loader, val_loader = make_loaders(args, train_ids, val_ids, vocab, args.seed)
    print_data_summary(train_ids, val_ids, vocab, len(train_loader))
    structure = build_model(args, vocab_size=len(vocab))
    init_key, train_key = jax.random.split(jax.random.PRNGKey(args.seed))
    params = initialize_params(structure, init_key)
    if args.init_params is not None:
        params = load_init_params(args.init_params, structure, vocab, args)
    if args.mode == "pc":
        warn_weak_pc_gradients(structure, params, train_loader, init_key)
    total_steps = count_training_steps(args.num_epochs, len(train_loader))
    num_epochs_to_run = math.ceil(args.num_epochs)
    train_label = (
        "train_batch_mean_loss" if args.mode == "backprop" else "train_pc_inferred_loss"
    )

    def validate(epoch_idx, epoch_params, structure, config, key, ce_loss, **_):
        metrics = evaluate_mlm(epoch_params, structure, val_loader, key)
        print(
            f"epoch {epoch_idx + 1}/{num_epochs_to_run} "
            f"{train_label}={ce_loss:.4f} "
            f"val_masked_loss={metrics['masked_loss']:.4f} "
            f"val_masked_acc={metrics['masked_accuracy']:.2%} "
            f"val_tokens={metrics['num_masked_tokens']}"
        )
        return metrics

    grad_scale = pc_grad_scale(train_loader) if args.mode == "pc" else 1.0
    trained_params, _, epoch_metrics = TRAINERS[args.mode](
        params,
        structure,
        train_loader,
        build_optimizer(args, total_steps, grad_scale),
        {"num_epochs": args.num_epochs},
        train_key,
        verbose=False,
        epoch_callback=validate,
    )
    if args.save_params is not None:
        archive, _ = save_params(
            args.save_params,
            trained_params,
            {
                "vocab": dict(vocab),
                "tokenizer_json": tokenizer_json,
                "config": model_config(args, len(vocab)),
            },
        )
        print(f"saved params to {archive}")
    return epoch_metrics[-1]


def compare_modes(args, train_ids, val_ids, vocab) -> ABResults:
    """Paired backprop-vs-PC pretraining over ``--ab_trials`` seeds.

    Every trial pretrains both arms from a fresh initialisation on the same
    loaders, then scores them with ``evaluate_mlm`` on ``--ab_metric``. Both
    arms share the same settings; the PC arm's gradients are divided by the
    mean masked-token count per batch before clipping, so clipping is only
    approximately matched (see ``pc_grad_scale``). Each PC trial checks its
    own initialisation with ``warn_weak_pc_gradients`` before training.
    ``--mode`` is not used.
    """
    batches_per_epoch = len(make_loaders(args, train_ids, val_ids, vocab, 0)[0])
    print_data_summary(train_ids, val_ids, vocab, batches_per_epoch)
    total_steps = count_training_steps(args.num_epochs, batches_per_epoch)

    def model_factory(key):
        structure = build_model(args, vocab_size=len(vocab))
        return initialize_params(structure, key), structure

    train_loader, _ = make_loaders(args, train_ids, val_ids, vocab, args.seed)
    optimizers = {
        "backprop": build_optimizer(args, total_steps),
        "pc": build_optimizer(args, total_steps, pc_grad_scale(train_loader)),
    }

    def eval_fn(params, structure, loader, config, key):
        return evaluate_mlm(params, structure, loader, key)

    def pc_train_fn(params, structure, loader, optimizer, config, key, **kwargs):
        warn_weak_pc_gradients(structure, params, loader, key)
        return TRAINERS["pc"](
            params, structure, loader, optimizer, config, key, **kwargs
        )

    train_fns = {"backprop": TRAINERS["backprop"], "pc": pc_train_fn}

    arm_a, arm_b = (
        ExperimentArm(
            name=mode,
            model_factory=model_factory,
            train_fn=train_fns[mode],
            eval_fn=eval_fn,
            optimizer=optimizers[mode],
            train_config={"num_epochs": args.num_epochs},
        )
        for mode in ("backprop", "pc")
    )
    results = ABExperiment(
        arm_a=arm_a,
        arm_b=arm_b,
        metric=args.ab_metric,
        data_loader_factory=lambda seed: make_loaders(
            args, train_ids, val_ids, vocab, seed
        ),
        n_trials=args.ab_trials,
        seed_offset=args.seed,
    ).run()
    direction = "lower" if args.ab_metric == "masked_loss" else "higher"
    print(f"{args.ab_metric}: {direction} is better; Diff = backprop - pc")
    results.print_summary()
    return results


def main(argv: Optional[Sequence[str]] = None):
    """Pretrain (or compare modes with ``--ab_trials``) and return the results."""
    args = parse_args(argv)
    args.ln_eps = resolve_ln_eps(args)
    train_ids, val_ids, vocab, tokenizer_json = load_token_streams(args)
    if args.ab_trials > 0:
        return compare_modes(args, train_ids, val_ids, vocab)
    return pretrain(args, train_ids, val_ids, vocab, tokenizer_json)


if __name__ == "__main__":
    main()

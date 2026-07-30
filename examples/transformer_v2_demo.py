"""
FabricPC Transformer V2 Execution Script

Trains the decomposed PC Transformer model on WikiText-103 (BPE tokenization)
or Tiny Shakespeare (Character-level tokenization), evaluates its performance,
and generates sample text using temperature sampling.

Architecture (each transformer block decomposed into separate PC nodes):

    input ──→ Embedding ──→ MhaResidual_0 ──→ LnMlp1_0 ──→ Mlp2Residual_0 ──→ ... ──→ VocabProjection
                                  ↑                              ↑
                             mask │                       (skip from MhaResidual)

    Each block: MhaResidual(in + mask) ──→ LnMlp1 ──→ Mlp2Residual(in + skip)

Benchmark Results (WikiText-103, 2M-token slice, BPE tokenizer):
------------------------------------------------------------------
Configuration:
  embed_dim=64, num_heads=8, mlp_dim=256, depth=2, seq_len=128,
  batch_size=16, lr=1.35e-4, weight_init_std=0.01296, dropout_rate=0.175
  PC-only: infer_steps=10, eta_infer=0.01381

Empirical Performance (Predictive Coding vs Backprop Transformer):
  Final train perplexity: 196.35 vs 68.57
  Final test perplexity:  240.08 vs 106.07
  Test accuracy:          12.71% vs 21.45%
  Test CE loss:           5.4810 vs 4.6641
  Training time:          15,686 s vs 3,189 s

Usage:
    PYTHONPATH=. python examples/transformer_v2_demo.py --tokenizer bpe
    PYTHONPATH=. python examples/transformer_v2_demo.py --mode backprop --tokenizer bpe
    PYTHONPATH=. python examples/transformer_v2_demo.py --tokenizer char --num_epochs 5
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import jax
import jax.numpy as jnp
from fabricpc.graph_initialization import initialize_params
from fabricpc.training import (
    train_autoregressive,
    evaluate_autoregressive,
    train_backprop_autoregressive,
    evaluate_backprop_autoregressive,
    generate_autoregressive,
)
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.utils.data import CharDataLoader, BpeDataLoader
import optax
import time
import os
import json
from diagnostics.plot_history import plot_single

# Tuned on WikiText-103 (2M-token slice), BPE tokenizer (vocab_size=8000).
BPE_DEFAULTS = {
    "embed_dim": 64,
    "num_heads": 8,
    "mlp_dim": 256,
    "depth": 2,
    "seq_len": 128,
    "batch_size": 16,
    "num_epochs": 5,
    "infer_steps": 10,
    "lr": 0.0001345863556572305,
    "eta_infer": 0.013807634865926306,
    "weight_init_std": 0.012956684480545929,
    "dropout_rate": 0.175,
}

# Tuned on Tiny Shakespeare, char tokenizer (two-phase search, val PPL 12.22).
CHAR_DEFAULTS = {
    "embed_dim": 64,
    "num_heads": 8,
    "mlp_dim": 256,
    "depth": 2,
    "seq_len": 128,
    "batch_size": 16,
    "num_epochs": 5,
    "infer_steps": 12,
    "lr": 0.00012108621644524519,
    "eta_infer": 0.0174852165627398,
    "weight_init_std": 0.015166293102182283,
    "dropout_rate": 0.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decomposed PC Transformer on WikiText-103 or Tiny Shakespeare"
    )
    parser.add_argument(
        "--mode",
        choices=["pc", "backprop"],
        default="pc",
        help="Training mode: predictive coding or backpropagation (default: pc)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Number of transformer layers (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=None,
        help="Embedding dimension (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=None,
        help="Number of attention heads (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--mlp_dim",
        type=int,
        default=None,
        help="MLP hidden dimension (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Sequence length (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (per-device for PC, total for backprop; default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=None,
        help="PC inference steps (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--eta_infer",
        type=float,
        default=None,
        help="PC inference learning rate (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--weight_init_std",
        type=float,
        default=None,
        help="Weight init std (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=None,
        help="Residual-branch dropout rate for transformer v2 (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap training sequences (e.g. 2000000) to bound a large BPE corpus",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tokenizer",
        choices=["char", "bpe"],
        default="char",
        help="Tokenizer to use (char: Tiny Shakespeare, bpe: WikiText-103)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-batch training energy"
    )
    return parser.parse_args(), parser


def main(args=None):
    if args is None:
        args, _ = parse_args()

    use_pc = args.mode == "pc"

    # --- Resolve tuned defaults for any arg left unset (None sentinel) ---
    # An explicit CLI value overrides; otherwise we fill from the tokenizer's
    # tuned constants. Done before batch_size so the tuned batch_size applies.
    use_bpe = args.tokenizer == "bpe"
    defaults = BPE_DEFAULTS if use_bpe else CHAR_DEFAULTS
    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    # --- Batch size ---
    if use_pc:
        n_devices = jax.device_count()
        batch_size = args.batch_size * n_devices
        print(f"PC mode: {n_devices} device(s), total batch_size={batch_size}")
    else:
        batch_size = args.batch_size
        print(f"Backprop mode: single device, batch_size={batch_size}")

    # --- Data ---
    if use_bpe:
        train_loader = BpeDataLoader(
            "train",
            seq_len=args.seq_len,
            batch_size=batch_size,
            shuffle=True,
            seed=args.seed,
            max_samples=args.max_train_samples,
        )
        test_loader = BpeDataLoader(
            "test", seq_len=args.seq_len, batch_size=batch_size, shuffle=False
        )
    else:
        train_loader = CharDataLoader(
            "train",
            seq_len=args.seq_len,
            batch_size=batch_size,
            shuffle=True,
            seed=args.seed,
        )
        test_loader = CharDataLoader(
            "test", seq_len=args.seq_len, batch_size=batch_size, shuffle=False
        )

    vocab_size = train_loader.vocab_size
    char_to_ix = train_loader.token_to_idx if use_bpe else train_loader.char_to_idx

    # --- Model ---
    structure = create_deep_transformer(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        inference=InferenceSGDNormClip(
            eta_infer=args.eta_infer,
            infer_steps=args.infer_steps,
            max_norm=5.0,
            latent_decay=0.0,
        ),
        weight_init={"type": "normal", "std": args.weight_init_std},
        dropout_rate=args.dropout_rate,
    )

    # --- Train & Evaluate ---
    master_rng_key = jax.random.PRNGKey(args.seed)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    params = initialize_params(structure, graph_key)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {n_params:,}")

    train_config = {
        "num_epochs": args.num_epochs,
        "use_causal_mask": True,
    }
    steps_per_epoch = train_loader.num_sequences // batch_size
    schedule = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=args.num_epochs * steps_per_epoch,
        alpha=0.1,
    )
    optimizer = optax.adam(schedule)

    print(f"Vocab Size: {vocab_size}")
    start = time.time()

    def iter_callback(epoch_idx, batch_idx, energy):
        if args.verbose and (batch_idx + 1) % 50 == 0:
            print(
                f"Epoch {epoch_idx + 1} | Batch {batch_idx + 1} | Energy: {energy:.4f}"
            )
        return energy

    # Per-epoch metric history tracking
    train_energy_hist = []
    train_ppl_hist = []
    test_ppl_hist = []
    test_metrics_hist = []

    eval_fn = evaluate_autoregressive if use_pc else evaluate_backprop_autoregressive

    def epoch_callback(
        epoch_idx, cb_params, cb_structure, cb_config, cb_rng, *, energy=None, ce_loss=None
    ):
        if energy is not None:
            train_energy_hist.append(float(energy))
        if ce_loss is not None:
            train_ppl_hist.append(float(jnp.exp(ce_loss)))
        m = eval_fn(cb_params, cb_structure, test_loader, train_config, cb_rng)
        test_metrics_hist.append(m)
        test_ppl_hist.append(float(m["perplexity"]))
        print(f"  [epoch {epoch_idx + 1}] test perplexity: {m['perplexity']:.2f}")
        return m["perplexity"]

    if use_pc:
        trained_params, _, _ = train_autoregressive(
            params,
            structure,
            train_loader,
            optimizer,
            train_config,
            train_key,
            verbose=True,
            iter_callback=iter_callback,
            epoch_callback=epoch_callback,
        )
    else:
        trained_params, _, _ = train_backprop_autoregressive(
            params,
            structure,
            train_loader,
            optimizer,
            train_config,
            train_key,
            verbose=True,
            epoch_callback=epoch_callback,
        )

    print(f"Training completed in {time.time() - start:.1f}s")

    # --- Evaluate ---
    eval_start = time.time()
    metrics = (
        test_metrics_hist[-1] 
        if test_metrics_hist 
        else eval_fn(trained_params, structure, test_loader, train_config, eval_key)
    )
    print(f"Evaluation completed in {time.time() - eval_start:.1f}s")

    print(f"Test Accuracy:   {metrics['accuracy'] * 100:.2f}%")
    print(f"Test CE Loss:    {metrics['loss']:.4f}")
    print(f"Test Perplexity: {metrics['perplexity']:.2f}")

    # Save metrics history & plot curves
    plot_dir = "runs/plots"
    os.makedirs(plot_dir, exist_ok=True)
    tag = f"{args.mode}_{args.tokenizer}"
    epochs = list(range(1, len(test_ppl_hist) + 1))
    history_dict = {
        "epochs": epochs,
        "train_energy": train_energy_hist,
        "train_ppl": train_ppl_hist,
        "test_ppl": test_ppl_hist,
    }
    history_file = os.path.join(plot_dir, f"history_{tag}.json")
    with open(history_file, "w") as fh:
        json.dump(history_dict, fh, indent=2)

    plot_single(tag, history_dict, plot_dir)
    print(f"Saved plots and history to {plot_dir}/ (tag: {tag})")

    # --- Text Generation ---
    gen_key = jax.random.PRNGKey(99)
    prompt_text = "ROMEO: "
    if use_bpe:
        seed_indices = train_loader._tok.encode(prompt_text).ids
    else:
        seed_indices = [char_to_ix.get(c, 0) for c in prompt_text]

    current_indices = ([0] * (args.seq_len - len(seed_indices)) + seed_indices)[
        -args.seq_len :
    ]
    prompt = jnp.array(current_indices, dtype=jnp.int32)

    print("--- Generating ---")
    generated = generate_autoregressive(
        trained_params,
        structure,
        prompt,
        max_new_tokens=200,
        rng_key=gen_key,
        temperature=0.8,
    )

    # Decode - skip the prompt padding, decode only from where prompt starts
    generated_ids = generated[args.seq_len :]
    print(prompt_text + train_loader.decode(generated_ids))


if __name__ == "__main__":
    main()

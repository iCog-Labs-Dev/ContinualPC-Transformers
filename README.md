# ContinualPC-Transformers

Continual learning research built on the FabricPC library and our PC-Transformer model. This repository is maintained by the iCog Labs NAC team.

## Lineage - why this is a clone

This repository is a copy of [trueagi-io/FabricPC](https://github.com/trueagi-io/FabricPC), started from commit [`<316367c6aa4895b0bf0f789ad1c2f58ce70f4242>`]. It is an **independent copy, not a fork**: we diverge freely and do not push changes back automatically.

We cloned rather than depending on FabricPC as an external package because our continual learning experiments require occasional changes to the library internals (for example, node-local learning rule hooks) that cannot be composed from the public building blocks alone. Keeping a copy lets us make those changes directly while still using FabricPC as a library for everything else.

The original FabricPC license is retained in `LICENSE`. Where a change we make here is broadly useful, we aim to contribute it back upstream rather than only keeping it in this repo.

## What this repository contains
- **`fabricpc/`** — the FabricPC library (predictive-coding nodes, energies, graph assembly, inference, and learning rules), plus our PC-Transformer v2 model (`fabricpc/nodes/transformer_v2.py`). We treat this as a dependency and change it only when necessary.
- **`<continualpct>/`** — our continual learning framework: domain-split data loaders, the sequential (A→B→C) evaluation harness, forgetting/transfer metrics (ACC, BWT, FWT), and the CL mechanisms under study (self-replay from a frozen teacher; node-local prediction error gated plasticity).
- **`experiments/`** — run scripts and configs for the experiments.
- **`diagnostics/`** — analysis tools (n-gram context-coverage, corpus sizing, history plotting).

## The experiments

We study whether predictive coding offers advantages for continual language modeling over standard backpropagation.Building on PC-Transformer v2 (a character/BPE-level PC transformer), the planned arc is:
1. **Inference analysis** — whether extra inference time settling recovers generalization under domain shift (a PC-native mechanism unavailable to backprop).
2. **CL benchmark** — sequential training across different domains, measuring forgetting and transfer.
3. **Self-replay** — anti-forgetting via replay from a frozen previous model.
4. **PE-gated plasticity** — the PC-native mechanism: per-node prediction error modulating that node's learning rate. This is the primary novel contribution.

## Installation

Clone the repository and install in editable mode. The command below pulls FabricPC, all optional dependencies, and a version-matched JAX backend - pick the line for your hardware:

```bash
# GPU, CUDA 12:
pip install -U -e ".[all,cuda12]"

# GPU, CUDA 13 (needs NVIDIA driver >= 580):
pip install -U -e ".[all,cuda13]"

# CPU only (the base `jax` dependency is the CPU build):
pip install -U -e ".[all]"
```

For a minimal install (core library only, no demos, utils, or dataloaders),
omit `[all]`:

```bash
pip install -e .
```

**Why `-U`?** `jax[cuda12]` installs a coupled set of packages (`jax`, `jaxlib`, `jax-cuda12-plugin`, `jax-cuda12-pjrt`) whose versions must match.
FabricPC's base dependencies install the CPU `jax`/`jaxlib` first, so without `-U` pip leaves them in place while pulling the newest CUDA plugin — a mismatch that makes JAX fail at import or at the first GPU operation. `-U` upgrades the whole set together.

### Verify the install

```python
import fabricpc
import jax
print(jax.devices())  # should list your GPU(s) or CPU
```

### Optional dependency groups

`all` bundles every group except the hardware backend; combine it with one
backend extra (`cuda12`, `cuda13`, or `cpu`).

| Group | Contents | Install with |
|-------|----------|--------------|
| `dev` | pytest, hypothesis, black, mypy, pre-commit | `pip install -e ".[dev]"` |
| `tfds` | TensorFlow Datasets for MNIST/CIFAR loaders | `pip install -e ".[tfds]"` |
| `experiments` | SciPy for statistical analysis | `pip install -e ".[experiments]"` |
| `viz` | Plotly, Aim, Pandas for dashboarding | `pip install -e ".[viz]"` |
| `cuda12` | JAX CUDA 12 backend | `pip install -U -e ".[cuda12]"` |
| `cuda13` | JAX CUDA 13 backend (driver >= 580) | `pip install -U -e ".[cuda13]"` |


## Quickstart

Train the PC-Transformer on character-level Tiny Shakespeare:

```bash
PYTHONPATH=. python examples/transformer_v2_demo.py --tokenizer char --mode pc --seed 0
```

Train the matched backpropagation control on the same architecture:

```bash
PYTHONPATH=. python examples/transformer_v2_demo.py --tokenizer char --mode backprop --seed 0
```

Each run writes to `runs/`:
- `runs/logs/` — per-epoch training and test perplexity
- `runs/metrics/metrics_<mode>_<tokenizer>_seed<seed>.json` — final metrics
- `runs/checkpoints/ckpt_<mode>_<tokenizer>_seed<seed>.pkl` — saved params + config
- `runs/plots/` — learning-curve figures and history JSON

Run a multi-seed comparison (five seeds each):

```bash
for s in 0 1 2 3 4; do
  PYTHONPATH=. python examples/transformer_v2_demo.py --tokenizer char --mode pc --seed $s
  PYTHONPATH=. python examples/transformer_v2_demo.py --tokenizer char --mode backprop --seed $s
done
```

Aggregate the per-seed metrics into mean ± std:

```bash
python diagnostics/aggregate_seeds.py   
```
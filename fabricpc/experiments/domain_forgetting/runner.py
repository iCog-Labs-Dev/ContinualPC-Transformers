"""Sequential MLM domain-forgetting runner for backprop and spike-only PC.

Stage S0 is the loaded checkpoint; stage ``s`` trains on the ``s``-th domain
from stage ``s - 1``'s parameters with a fresh optimizer and schedule. After
every stage all domains are evaluated on their fixed validation sets.

In PC mode the spike-only schedule (``eta_infer`` 0, ``spike_eta`` 1,
longest depth, initial forward updates) makes PC weight gradients equal
``pc_output_energy_scale`` times the backprop gradients, so the global-norm
clip is ``grad_clip * pc_output_energy_scale`` and Adam's ``eps`` is
``adam_eps * pc_output_energy_scale``. Adam is invariant to a common
gradient scale once ``eps`` scales with it, so the PC update equals
backprop's at any scale.

Each update averages its micro-batch gradients weighted by their masked-token
counts (``Σ_k g_k n_k / Σ_k n_k``), so it equals the gradient of the mean
masked CE over the whole batch.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.bert_demo import load_checkpoint, load_init_params
from examples.bert_forgetting_benchmark import (
    _reject_symlink_components,
    _reject_symlink_tree,
    ensure_output_dir,
)
from fabricpc.core.depth import node_output_depths
from fabricpc.core.inference import InferenceSGDNormClip, InferenceSpikingPrecision
from fabricpc.core.types import GraphParams, GraphStructure, NodeParams
from fabricpc.experiments.domain_forgetting.artifacts import (
    atomic_save_npz,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_npz,
    sha256_text,
)
from fabricpc.experiments.domain_forgetting.data import (
    DomainCache,
    load_domain_cache,
    special_token_ids,
    validate_domain_names,
)
from fabricpc.experiments.domain_forgetting.metrics import (
    forgetting_matrices,
    pc_bp_agreement,
    summarize_matrices,
)
from fabricpc.experiments.domain_forgetting.validation import MultiDomainEvaluator
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import (
    dumps_strict,
    params_hash,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes.encoder_transformer import create_encoder_mlm_model
from fabricpc.training.optimizers import build_warmup_cosine_adamw
from fabricpc.training.train import _convert_batch
from fabricpc.training.train_backprop import (
    _apply_optimizer_update_when_masked,
    _loss_and_grad_backprop_mlm_no_update,
)
from fabricpc.training.train_mlm import create_jit_gradient_step_pcn_mlm
from fabricpc.utils.data.dataloader import MlmTextBatcher

BERT_MINI_SHAPE = {
    "depth": 4,
    "embed_dim": 256,
    "num_heads": 4,
    "mlp_dim": 1024,
    "seq_len": 128,
    "vocab_size": 30522,
}
MODES = ("backprop", "pc")
METRICS = ("masked_ce", "masked_perplexity", "masked_accuracy")
MATRIX_FILES = {
    "masked_ce": "masked_ce_matrix.json",
    "masked_perplexity": "masked_perplexity_matrix.json",
    "masked_accuracy": "masked_accuracy_matrix.json",
}
RESUME_IGNORED_KEYS = ("log_interval", "output_dir")
COMPARE_IGNORED_KEYS = (
    "mode",
    "output_dir",
    "log_interval",
    "init_perturbation",
    "init_perturbation_seed",
)
ANALYSIS_FILES = (
    "drift.json",
    "rollback.json",
    "error_maps.json",
    "predictors.json",
    "correlations.json",
    "correlations.png",
)
RUN_ARTIFACT_FILES = (
    "run_config.json",
    "checkpoint_manifest.json",
    "dataset_manifest.json",
    "tokenizer.json",
    "evaluations.jsonl",
    *MATRIX_FILES.values(),
    "forgetting_matrices.json",
    "summary.json",
    "exposure.json",
    "runtime.json",
    "pc_bp_agreement.json",
    *ANALYSIS_FILES,
)
RUN_ARTIFACT_PATTERNS = ("stage_*.npz", "training_log_stage_*.jsonl")


@dataclass(frozen=True)
class RunConfig:
    """Every setting that determines a run's results."""

    checkpoint: str
    data_dir: str
    domains: tuple[str, ...]
    output_dir: str
    mode: str = "backprop"
    tokens_per_domain: int = 5_000_000
    lr: float = 1e-4
    batch_size: int = 64
    micro_batch_size: int | None = None
    warmup_fraction: float = 0.1
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 2.0
    adam_eps: float = 1e-8
    dropout_rate: float = 0.1
    mask_prob: float = 0.15
    seed: int = 0
    eval_batch_size: int = 64
    max_valid_windows: int | None = None
    expected_shape: dict[str, int] | None = field(
        default_factory=lambda: dict(BERT_MINI_SHAPE)
    )
    pc_inference: str = "spiking"
    pc_eta_infer: float = 0.0
    pc_spike_eta: float = 1.0
    pc_spike_depth: str = "longest"
    pc_forward_updates: str = "initial"
    pc_infer_steps: int | None = None
    pc_max_norm: float = 5.0
    pc_output_energy_scale: float = 1.0
    init_perturbation: float = 0.0
    init_perturbation_seed: int = 0
    log_interval: int = 10

    def validate(self) -> None:
        validate_domain_names(self.domains)
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        for name in ("tokens_per_domain", "batch_size", "eval_batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        micro = self.micro_batch_size
        if micro is not None and (micro < 1 or self.batch_size % micro):
            raise ValueError("micro_batch_size must be positive and divide batch_size")
        if not 0.0 <= self.warmup_fraction <= 1.0:
            raise ValueError("warmup_fraction must be in [0, 1]")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if not (math.isfinite(self.init_perturbation) and self.init_perturbation >= 0):
            raise ValueError("init_perturbation must be finite and non-negative")
        if self.mode == "pc":
            if self.pc_inference != "spiking":
                raise ValueError("only --pc_inference spiking is supported")
            if self.pc_output_energy_scale <= 0.0:
                raise ValueError("pc_output_energy_scale must be positive")

    @property
    def clip_norm(self) -> float:
        """The global-norm clip, scaled by the PC output-energy scale in PC mode."""
        if self.mode == "pc":
            return self.grad_clip * self.pc_output_energy_scale
        return self.grad_clip

    @property
    def optimizer_eps(self) -> float:
        """Adam's ``eps``, scaled by the PC output-energy scale in PC mode."""
        if self.mode == "pc":
            return self.adam_eps * self.pc_output_energy_scale
        return self.adam_eps

    def identity(self) -> dict[str, Any]:
        """The settings that must match for a run to be resumed."""
        values = asdict(self)
        for key in RESUME_IGNORED_KEYS:
            values.pop(key)
        values["domains"] = list(self.domains)
        return values


@dataclass
class BaseModel:
    """The validated source checkpoint and the graph built for it."""

    structure: GraphStructure
    params: GraphParams
    vocab: dict[str, int]
    tokenizer_json: str
    model_config: dict[str, Any]
    special_ids: dict[str, int]
    infer_steps: int | None
    source: dict[str, Any]


def max_free_depth(structure: GraphStructure, mode: str = "longest") -> int:
    """Return the deepest output depth over nodes with inputs."""
    depths = node_output_depths(structure, mode)
    return max(
        depth
        for name, depth in depths.items()
        if structure.nodes[name].node_info.in_degree > 0
    )


def build_structure(
    model_config: Mapping[str, Any], config: RunConfig, inference: Any
) -> GraphStructure:
    """Build the BERT MLM graph of a checkpoint config for this run."""
    return create_encoder_mlm_model(
        depth=int(model_config["depth"]),
        embed_dim=int(model_config["embed_dim"]),
        num_heads=int(model_config["num_heads"]),
        mlp_dim=int(model_config["mlp_dim"]),
        seq_len=int(model_config["seq_len"]),
        vocab_size=int(model_config["vocab_size"]),
        inference=inference,
        use_attention_mask=True,
        dropout_rate=config.dropout_rate,
        pc_output_energy_scale=(
            config.pc_output_energy_scale if config.mode == "pc" else 1.0
        ),
        architecture="bert",
        ln_eps=model_config.get("ln_eps"),
        residual_identity=True,
    )


def _spiking_inference(config: RunConfig, infer_steps: int) -> Any:
    return InferenceSpikingPrecision(
        eta_infer=config.pc_eta_infer,
        infer_steps=infer_steps,
        max_norm=config.pc_max_norm,
        spike_eta=config.pc_spike_eta,
        depth_mode=config.pc_spike_depth,
    )


def load_base_model(config: RunConfig) -> BaseModel:
    """Validate the source checkpoint and load its parameters into memory.

    Checks ``architecture == "bert"``, the expected shape, that the tokenizer
    vocabulary equals the checkpoint vocabulary, and (through
    ``load_init_params``) architecture, ``seq_len``, ``ln_eps`` and every
    parameter shape. In PC mode ``pc_infer_steps`` defaults to the longest
    free depth and may not be smaller.
    """
    from tokenizers import Tokenizer

    path = Path(config.checkpoint)
    checkpoint = load_checkpoint(str(path))
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model_config = dict(checkpoint["config"])
    if model_config.get("architecture") != "bert":
        raise ValueError(f"{path}: checkpoint architecture must be 'bert'")
    if config.expected_shape is not None:
        mismatched = {
            key: (model_config.get(key), value)
            for key, value in config.expected_shape.items()
            if model_config.get(key) != value
        }
        if mismatched:
            raise ValueError(f"{path}: shape (found, expected) {mismatched}")
    tokenizer_json = checkpoint["tokenizer_json"]
    if not tokenizer_json:
        raise ValueError(f"{path}: checkpoint has no tokenizer_json")
    tokenizer = Tokenizer.from_str(tokenizer_json)
    vocab = {str(k): int(v) for k, v in checkpoint["vocab"].items()}
    if tokenizer.get_vocab() != vocab or len(vocab) != model_config["vocab_size"]:
        raise ValueError(f"{path}: tokenizer vocab does not match the checkpoint")

    infer_steps = None
    if config.mode == "pc":
        probe = build_structure(model_config, config, _spiking_inference(config, 1))
        needed = max_free_depth(probe, config.pc_spike_depth)
        infer_steps = config.pc_infer_steps or needed
        if infer_steps < needed:
            raise ValueError(
                f"pc_infer_steps {infer_steps} < longest free depth {needed}; "
                "the deepest nodes would never spike"
            )
        inference = _spiking_inference(config, infer_steps)
    else:
        inference = InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=5.0)
    structure = build_structure(model_config, config, inference)
    params = load_init_params(
        str(path),
        initialize_params(structure, jax.random.PRNGKey(0)),
        vocab,
        architecture="bert",
        seq_len=int(model_config["seq_len"]),
        ln_eps=model_config.get("ln_eps"),
    )
    tensor_path = path.with_name(json.loads(path.read_text())["tensor_file"])
    return BaseModel(
        structure=structure,
        params=params,
        vocab=vocab,
        tokenizer_json=tokenizer_json,
        model_config=model_config,
        special_ids=special_token_ids(tokenizer),
        infer_steps=infer_steps,
        source={
            "checkpoint": str(path),
            "checkpoint_sha256": file_sha256(path),
            "tensor_file": str(tensor_path),
            "tensor_sha256": file_sha256(tensor_path),
            "tokenizer_sha256": sha256_text(tokenizer_json),
            "params_sha256": params_hash(params),
            "model_config": model_config,
        },
    )


def num_stage_updates(tokens_per_domain: int, batch_size: int, seq_len: int) -> int:
    """``ceil(tokens_per_domain / (batch_size * (seq_len - 1)))``."""
    return math.ceil(tokens_per_domain / (batch_size * (seq_len - 1)))


def build_stage_optimizer(
    config: RunConfig, num_updates: int
) -> tuple[optax.GradientTransformation, dict[str, Any]]:
    """AdamW with linear warmup then cosine decay to ``min_lr_ratio * lr``.

    Mirrors ``build_optimizer`` in ``examples/bert_demo.py`` (global-norm clip
    chained before ``build_warmup_cosine_adamw``) and adds the decay floor.
    """
    warmup_steps = min(num_updates, int(round(config.warmup_fraction * num_updates)))
    optimizer = build_warmup_cosine_adamw(
        learning_rate=config.lr,
        total_steps=num_updates,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        min_learning_rate=config.min_lr_ratio * config.lr,
        eps=config.optimizer_eps,
    )
    if config.clip_norm > 0:
        optimizer = optax.chain(optax.clip_by_global_norm(config.clip_norm), optimizer)
    return optimizer, {
        "type": "warmup_cosine_adamw",
        "lr": config.lr,
        "min_lr": config.min_lr_ratio * config.lr,
        "warmup_steps": warmup_steps,
        "total_steps": num_updates,
        "weight_decay": config.weight_decay,
        "clip_norm": config.clip_norm,
        "adam_eps": config.optimizer_eps,
    }


def flatten_params(params: GraphParams) -> dict[str, np.ndarray]:
    """Return ``{"node/weights|biases/name": array}`` for every leaf."""
    return {
        f"{node_name}/{collection}/{name}": np.asarray(value)
        for node_name, node in params.nodes.items()
        for collection in ("weights", "biases")
        for name, value in getattr(node, collection).items()
    }


def params_from_flat(
    flat: Mapping[str, np.ndarray], structure: GraphStructure
) -> GraphParams:
    """Rebuild graph params in the structure's node order from flat leaves."""
    nodes: dict[str, dict[str, dict[str, Any]]] = {
        name: {"weights": {}, "biases": {}} for name in structure.nodes
    }
    for key, value in flat.items():
        node_name, collection, name = key.split("/", 2)
        if node_name not in nodes or collection not in ("weights", "biases"):
            raise ValueError(f"unexpected parameter {key!r}")
        nodes[node_name][collection][name] = jnp.asarray(value)
    return GraphParams(
        nodes={
            name: NodeParams(weights=entry["weights"], biases=entry["biases"])
            for name, entry in nodes.items()
        }
    )


def perturb_params(params: GraphParams, relative: float, seed: int) -> GraphParams:
    """Return ``θ · (1 + relative · ε)`` with ``ε ~ N(0, 1)`` per scalar.

    Noise is drawn in float64 by ``numpy.random.default_rng(seed)`` over the
    leaves in sorted ``flatten_params`` key order and cast back to each
    leaf's dtype. ``relative`` 0 returns ``params`` unchanged.
    """
    if relative == 0.0:
        return params
    rng = np.random.default_rng(seed)
    flat = flatten_params(params)
    perturbed = {}
    for key in sorted(flat):
        value = np.asarray(flat[key])
        noise = rng.standard_normal(value.shape)
        perturbed[key] = (value.astype(np.float64) * (1.0 + relative * noise)).astype(
            value.dtype
        )
    return GraphParams(
        nodes={
            name: NodeParams(
                weights={
                    k: jnp.asarray(perturbed[f"{name}/weights/{k}"])
                    for k in node.weights
                },
                biases={
                    k: jnp.asarray(perturbed[f"{name}/biases/{k}"]) for k in node.biases
                },
            )
            for name, node in params.nodes.items()
        }
    )


def save_stage_snapshot(path: str | Path, params: GraphParams) -> dict[str, str]:
    """Atomically save stage params; return the file and parameter SHA-256."""
    atomic_save_npz(path, flatten_params(params))
    return {"file_sha256": file_sha256(path), "params_sha256": params_hash(params)}


def load_stage_snapshot(path: str | Path, structure: GraphStructure) -> GraphParams:
    """Load a ``stage_<s>.npz`` snapshot as graph params."""
    return params_from_flat(load_npz(path), structure)


GradientFn = Callable[
    [GraphParams, dict[str, jax.Array], jax.Array],
    tuple[GraphParams, jax.Array, jax.Array, dict[str, jax.Array]],
]


def make_gradient_fn(
    mode: str, structure: GraphStructure, forward_updates: str = "initial"
) -> GradientFn:
    """Return a jitted ``(params, batch, key) -> (grads, loss_sum, count, extras)``.

    ``loss_sum`` is the masked CE summed over the batch's masked tokens.
    Both modes draw dropout masks from ``key`` the same way.
    """
    if mode == "backprop":

        @jax.jit
        def backprop(params, batch, key):
            loss, grads, count = _loss_and_grad_backprop_mlm_no_update(
                params, batch, structure, key
            )
            return grads, loss * count, count, {}

        return backprop
    pc_step = create_jit_gradient_step_pcn_mlm(structure, forward_updates)

    def pc(params, batch, key):
        grads, loss_sum, count, rises, energy = pc_step(params, batch, key)
        return (
            grads,
            loss_sum,
            count,
            {
                "pc_energy_rises": rises,
                "pc_total_energy": energy,
            },
        )

    return pc


def accumulate_gradients(
    gradient_fn: GradientFn,
    params: GraphParams,
    micro_batches: Sequence[Mapping[str, jax.Array]],
    keys: Sequence[jax.Array],
) -> tuple[GraphParams, float, float, list[dict[str, Any]]]:
    """Return the token-weighted mean gradient of one update's micro-batches.

    ``gradient_fn`` returns each micro-batch's mean-CE gradient ``g_k`` over
    its ``n_k`` masked tokens; the result is ``Σ_k g_k n_k / max(Σ_k n_k, 1)``
    with the summed masked CE, the total count and the per-micro-batch
    extras.
    """
    total = None
    loss_sum = 0.0
    count = 0.0
    extras = []
    for batch, key in zip(micro_batches, keys, strict=True):
        grads, batch_loss, batch_count, batch_extras = gradient_fn(params, batch, key)
        weighted = jax.tree_util.tree_map(lambda g: g * batch_count, grads)
        total = (
            weighted
            if total is None
            else jax.tree_util.tree_map(jnp.add, total, weighted)
        )
        loss_sum += float(batch_loss)
        count += float(batch_count)
        extras.append(batch_extras)
    if total is None:
        raise ValueError("an update needs at least one micro-batch")
    denominator = max(count, 1.0)
    return (
        jax.tree_util.tree_map(lambda g: g / denominator, total),
        loss_sum,
        count,
        extras,
    )


def train_stage(
    params: GraphParams,
    structure: GraphStructure,
    stream: np.ndarray,
    *,
    config: RunConfig,
    stage: int,
    special_ids: Mapping[str, int],
    gradient_fn: GradientFn,
    verbose: bool = True,
) -> tuple[GraphParams, dict[str, Any], list[dict[str, Any]]]:
    """Train one domain from ``params`` with a fresh optimizer and schedule.

    Runs ``num_stage_updates`` updates of ``batch_size`` sequences, each the
    masked-token-weighted mean of ``batch_size / micro_batch_size``
    micro-batch gradients (``accumulate_gradients``); the batcher reshuffles
    and restarts when an epoch ends. Batches, MLM masks
    and dropout keys depend only on ``seed`` and ``stage``, so both modes see
    the same ones. Returns the params, the exposure record and the log.
    """
    seq_len = int(structure.nodes[structure.task_map["y"]].node_info.shape[0])
    vocab_size = int(structure.nodes[structure.task_map["y"]].node_info.shape[-1])
    micro = config.micro_batch_size or config.batch_size
    accumulation = config.batch_size // micro
    num_updates = num_stage_updates(
        config.tokens_per_domain, config.batch_size, seq_len
    )
    batcher = MlmTextBatcher(
        token_ids=np.asarray(stream, dtype=np.int32),
        seq_len=seq_len,
        batch_size=micro,
        pad_token_id=special_ids["[PAD]"],
        mask_token_id=special_ids["[MASK]"],
        vocab_size=vocab_size,
        special_token_ids=set(special_ids.values()),
        cls_token_id=special_ids["[CLS]"],
        seed=config.seed + stage,
        mask_prob=config.mask_prob,
        shuffle=True,
        token_unit="subword",
    )
    if len(batcher) == 0:
        raise ValueError("domain stream is too short for one micro-batch")
    optimizer, optimizer_info = build_stage_optimizer(config, num_updates)
    opt_state = optimizer.init(params)
    update_fn = jax.jit(
        lambda p, o, g, c: _apply_optimizer_update_when_masked(
            params=p, opt_state=o, grads=g, optimizer=optimizer, masked_count=c
        )
    )
    rng_key = jax.random.fold_in(jax.random.PRNGKey(config.seed), stage)
    counts = {
        "stream_tokens": 0,
        "non_padding_tokens": 0,
        "masked_tokens": 0,
        "token_slots": 0,
        "micro_batches": 0,
    }
    log: list[dict[str, Any]] = []
    pending_batches: list[dict[str, jax.Array]] = []
    pending_keys: list[jax.Array] = []
    updates = 0
    while updates < num_updates:
        for batch_data in batcher:
            attention = np.asarray(batch_data["attention_mask"])
            counts["stream_tokens"] += int(attention[:, 1:].sum())
            counts["non_padding_tokens"] += int(attention.sum())
            counts["masked_tokens"] += int(np.asarray(batch_data["loss_mask"]).sum())
            counts["token_slots"] += int(attention.size)
            counts["micro_batches"] += 1
            rng_key, step_key = jax.random.split(rng_key)
            pending_batches.append(_convert_batch(batch_data))
            pending_keys.append(step_key)
            if len(pending_batches) < accumulation:
                continue
            pending_grads, pending_loss, pending_count, pending_extras = (
                accumulate_gradients(gradient_fn, params, pending_batches, pending_keys)
            )
            params, opt_state = update_fn(
                params, opt_state, pending_grads, jnp.asarray(pending_count)
            )
            updates += 1
            entry: dict[str, Any] = {
                "stage": stage,
                "update": updates,
                "masked_loss": pending_loss / max(pending_count, 1.0),
                "masked_tokens": int(pending_count),
            }
            if config.mode == "pc":
                entry["pc_total_energy"] = float(
                    np.mean([float(e["pc_total_energy"]) for e in pending_extras])
                )
                entry["pc_energy_rises"] = int(
                    sum(int(e["pc_energy_rises"]) for e in pending_extras)
                )
            log.append(entry)
            if verbose and config.log_interval and updates % config.log_interval == 0:
                print(
                    f"stage={stage} update={updates}/{num_updates} "
                    + " ".join(
                        f"{k}={v:.4g}"
                        for k, v in entry.items()
                        if k not in ("stage", "update")
                    )
                )
            pending_batches = []
            pending_keys = []
            if updates >= num_updates:
                break
    tokens_per_update = config.batch_size * (seq_len - 1)
    exposure = {
        "updates": updates,
        "gradient_accumulation_steps": accumulation,
        "micro_batch_size": micro,
        "tokens_per_domain": config.tokens_per_domain,
        "max_stream_tokens_per_update": tokens_per_update,
        **counts,
        "exposure_minus_budget": counts["stream_tokens"] - config.tokens_per_domain,
        "within_one_update": abs(counts["stream_tokens"] - config.tokens_per_domain)
        <= tokens_per_update,
        "epochs_started": int(batcher.state_dict()["epoch"]),
        "optimizer": optimizer_info,
    }
    return params, exposure, log


def evaluate_domains(
    params: GraphParams,
    structure: GraphStructure,
    valid_sets: Mapping[str, Mapping[str, np.ndarray]],
    *,
    batch_size: int = 64,
    evaluator: MultiDomainEvaluator | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate ``params`` on every domain's fixed validation set.

    Uses ``evaluator`` when given, else builds a ``MultiDomainEvaluator``
    (masked positions gathered before the vocabulary projection).
    """
    if evaluator is None:
        evaluator = MultiDomainEvaluator(structure, valid_sets, batch_size)
    return evaluator(params)


def prepare_output_dir(
    output_dir: str | Path, *, resume: bool, overwrite: bool
) -> Path:
    """Refuse a non-empty directory unless resuming or overwriting; reject symlinks."""
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    path = Path(output_dir)
    if resume:
        _reject_symlink_components(path)
        path.mkdir(parents=True, exist_ok=True)
    else:
        path = ensure_output_dir(path, overwrite=overwrite)
    _reject_symlink_tree(path)
    if overwrite:
        remove_run_artifacts(path)
    return path


def remove_run_artifacts(path: Path) -> list[str]:
    """Delete the run and analysis files a run directory can hold; return them.

    Only regular files named in ``RUN_ARTIFACT_FILES`` or matching
    ``RUN_ARTIFACT_PATTERNS`` are removed; symlinks are refused.
    """
    candidates = [path / name for name in RUN_ARTIFACT_FILES]
    for pattern in RUN_ARTIFACT_PATTERNS:
        candidates.extend(sorted(path.glob(pattern)))
    removed = []
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError(f"Refusing to delete symlink: {candidate}")
        if candidate.is_file():
            candidate.unlink()
            removed.append(candidate.name)
    return removed


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_text(
        path, "".join(dumps_strict(r, sort_keys=True) + "\n" for r in records)
    )


def _stage_is_complete(
    output: Path,
    entry: Mapping[str, Any] | None,
    *,
    stage: int,
    domain: str | None,
    parent_sha: str | None,
    evaluations: Mapping[tuple[int, str], Mapping[str, Any]],
    domains: Sequence[str],
    valid_sha: Mapping[str, str],
    structure: GraphStructure,
) -> GraphParams | None:
    """Return the stage's params when its snapshot and evaluations verify."""
    if entry is None or entry.get("domain") != domain:
        return None
    if entry.get("parent_params_sha256") != parent_sha:
        return None
    path = output / f"stage_{stage}.npz"
    if not path.exists() or file_sha256(path) != entry["file_sha256"]:
        return None
    params = load_stage_snapshot(path, structure)
    if params_hash(params) != entry["params_sha256"]:
        return None
    for name in domains:
        record = evaluations.get((stage, name))
        if (
            record is None
            or record.get("params_sha256") != entry["params_sha256"]
            or record.get("valid_sha256") != valid_sha[name]
        ):
            return None
    return params


def _matrix(
    records: Mapping[tuple[int, str], Mapping[str, Any]],
    domains: Sequence[str],
    metric: str,
) -> list[list[float]]:
    return [
        [float(records[(stage, name)][metric]) for name in domains]
        for stage in range(len(domains) + 1)
    ]


def run_domain_forgetting(
    config: RunConfig,
    *,
    resume: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run S0 -> S1 ... -> S_n, evaluating every domain after every stage.

    Writes ``run_config.json``, ``checkpoint_manifest.json``,
    ``dataset_manifest.json``, ``tokenizer.json``, ``evaluations.jsonl``, the
    CE, perplexity and accuracy matrices, ``forgetting_matrices.json``,
    ``summary.json``, ``exposure.json``, ``runtime.json``,
    ``training_log_stage_<s>.jsonl`` and ``stage_<s>.npz``. A stage commits
    when its ``checkpoint_manifest.json`` entry is written. With ``resume``,
    committed stages whose snapshot and evaluations verify are skipped; the
    first stage that does not verify, and every later one, is retrained from
    the previous snapshot. With ``overwrite``, every known run and analysis
    artifact is deleted first (``remove_run_artifacts``). With
    ``init_perturbation`` > 0, S0 is the checkpoint perturbed by
    ``perturb_params``.
    """
    config.validate()
    domains = list(config.domains)
    started = time.time()
    output = prepare_output_dir(config.output_dir, resume=resume, overwrite=overwrite)
    base = load_base_model(config)
    seq_len = int(base.model_config["seq_len"])
    caches: dict[str, DomainCache] = {
        name: load_domain_cache(
            config.data_dir,
            name,
            tokenizer_json=base.tokenizer_json,
            seq_len=seq_len,
            tokens_per_domain=config.tokens_per_domain,
            max_valid_windows=config.max_valid_windows,
        )
        for name in domains
    }
    window_counts = {name: len(c.valid["input_ids"]) for name, c in caches.items()}
    if len(set(window_counts.values())) != 1:
        raise ValueError(f"validation sets differ in size: {window_counts}")
    valid_sets = {name: cache.valid for name, cache in caches.items()}
    valid_sha = {
        name: cache.manifest["valid"]["arrays_sha256"] for name, cache in caches.items()
    }
    initial_params = perturb_params(
        base.params, config.init_perturbation, config.init_perturbation_seed
    )
    initial_params_sha = params_hash(initial_params)

    run_config = {
        "config": config.identity(),
        "mode": config.mode,
        "clip_norm": config.clip_norm,
        "adam_eps": config.optimizer_eps,
        "init_perturbation": {
            "relative": config.init_perturbation,
            "seed": config.init_perturbation_seed,
            "rule": "theta * (1 + relative * N(0, 1)) per scalar",
            "params_sha256": initial_params_sha,
        },
        "pc": (
            {
                "inference": config.pc_inference,
                "eta_infer": config.pc_eta_infer,
                "spike_eta": config.pc_spike_eta,
                "spike_depth": config.pc_spike_depth,
                "forward_updates": config.pc_forward_updates,
                "infer_steps": base.infer_steps,
                "max_norm": config.pc_max_norm,
                "output_energy_scale": config.pc_output_energy_scale,
            }
            if config.mode == "pc"
            else None
        ),
        "source": base.source,
        "valid_sha256": valid_sha,
        "num_valid_windows": next(iter(window_counts.values())),
        "train_array_sha256": {
            name: cache.manifest["train"]["array_sha256"]
            for name, cache in caches.items()
        },
        "updates_per_stage": num_stage_updates(
            config.tokens_per_domain, config.batch_size, seq_len
        ),
    }
    run_config_path = output / "run_config.json"
    previous_config = _read_json(run_config_path) if resume else None
    if previous_config is not None and json.loads(
        dumps_strict(previous_config, sort_keys=True)
    ) != json.loads(dumps_strict(run_config, sort_keys=True)):
        raise ValueError(f"{run_config_path}: cannot resume with a different setup")
    atomic_write_json(run_config_path, run_config)
    atomic_write_text(output / "tokenizer.json", base.tokenizer_json)
    dataset_manifest = Path(config.data_dir) / "dataset_manifest.json"
    if dataset_manifest.exists():
        destination = output / "dataset_manifest.json"
        if destination.is_symlink():
            raise ValueError(f"Refusing to write through symlink: {destination}")
        shutil.copyfile(dataset_manifest, destination)

    manifest_path = output / "checkpoint_manifest.json"
    previous_manifest = (_read_json(manifest_path) if resume else None) or {}
    previous_stages = {
        int(entry["stage"]): entry for entry in previous_manifest.get("stages", [])
    }
    previous_evals = {
        (int(r["stage"]), r["domain"]): r
        for r in (_read_jsonl(output / "evaluations.jsonl") if resume else [])
    }
    previous_runtime = (_read_json(output / "runtime.json") if resume else None) or {}
    previous_exposure = (_read_json(output / "exposure.json") if resume else None) or {}

    gradient_fn = make_gradient_fn(
        config.mode, base.structure, config.pc_forward_updates
    )
    params = initial_params
    evaluator = MultiDomainEvaluator(base.structure, valid_sets, config.eval_batch_size)
    stages: list[dict[str, Any]] = []
    evaluations: dict[tuple[int, str], dict[str, Any]] = {}
    exposure: dict[str, Any] = {}
    stage_runtime: dict[str, Any] = {}
    reusing = resume
    parent_sha: str | None = None
    for stage in range(len(domains) + 1):
        domain = None if stage == 0 else domains[stage - 1]
        stage_start = time.time()
        reused = None
        if reusing:
            reused = _stage_is_complete(
                output,
                previous_stages.get(stage),
                stage=stage,
                domain=domain,
                parent_sha=parent_sha,
                evaluations=previous_evals,
                domains=domains,
                valid_sha=valid_sha,
                structure=base.structure,
            )
            if stage == 0 and reused is not None:
                if params_hash(reused) != initial_params_sha:
                    reused = None
            reusing = reused is not None
        if reused is not None:
            params = reused
            entry = previous_stages[stage]
            stages.append(entry)
            for name in domains:
                evaluations[(stage, name)] = dict(previous_evals[(stage, name)])
            if domain is not None:
                exposure[domain] = previous_exposure.get(domain)
            stage_runtime[str(stage)] = {
                **(previous_runtime.get("stages", {}).get(str(stage)) or {}),
                "reused": True,
            }
            parent_sha = entry["params_sha256"]
            if verbose:
                print(f"stage={stage} reused params_sha256={parent_sha[:12]}")
            continue

        initial_sha = params_hash(params)
        train_seconds = 0.0
        if domain is not None:
            params, stage_exposure, log = train_stage(
                params,
                base.structure,
                caches[domain].train,
                config=config,
                stage=stage,
                special_ids=base.special_ids,
                gradient_fn=gradient_fn,
                verbose=verbose,
            )
            train_seconds = time.time() - stage_start
            exposure[domain] = stage_exposure
            _write_jsonl(output / f"training_log_stage_{stage}.jsonl", log)
        snapshot = save_stage_snapshot(output / f"stage_{stage}.npz", params)
        eval_start = time.time()
        results = evaluate_domains(
            params, base.structure, valid_sets, evaluator=evaluator
        )
        for index, name in enumerate(domains):
            evaluations[(stage, name)] = {
                "stage": stage,
                "after_domain": domain,
                "domain": name,
                "domain_index": index,
                **results[name],
                "params_sha256": snapshot["params_sha256"],
                "valid_sha256": valid_sha[name],
            }
            if verbose:
                print(
                    f"stage={stage} eval domain={name} "
                    f"ce={results[name]['masked_ce']:.4f} "
                    f"acc={results[name]['masked_accuracy']:.4f}"
                )
        _write_jsonl(
            output / "evaluations.jsonl",
            [evaluations[key] for key in sorted(evaluations)],
        )
        entry = {
            "stage": stage,
            "domain": domain,
            "file": f"stage_{stage}.npz",
            "initial_params_sha256": initial_sha,
            "parent_params_sha256": parent_sha,
            **snapshot,
            "exposure": exposure.get(domain) if domain is not None else None,
        }
        stages.append(entry)
        stage_runtime[str(stage)] = {
            "train_seconds": train_seconds,
            "eval_seconds": time.time() - eval_start,
            "reused": False,
        }
        atomic_write_json(output / "exposure.json", exposure)
        atomic_write_json(
            manifest_path,
            {
                "source": base.source,
                "initial_params_sha256": initial_params_sha,
                "domains": domains,
                "stages": stages,
            },
        )
        parent_sha = snapshot["params_sha256"]

    matrices = {metric: _matrix(evaluations, domains, metric) for metric in METRICS}
    for metric, filename in MATRIX_FILES.items():
        atomic_write_json(
            output / filename,
            {
                "metric": metric,
                "domains": domains,
                "stages": [f"S{s}" for s in range(len(domains) + 1)],
                "rows": "stage",
                "columns": "domain in training order",
                "matrix": matrices[metric],
            },
        )
    ce, ppl, acc = (matrices[m] for m in METRICS)
    atomic_write_json(
        output / "forgetting_matrices.json",
        {"domains": domains, **forgetting_matrices(ce, ppl, acc)},
    )
    summary = {
        **summarize_matrices(domains, ce, ppl, acc),
        "mode": config.mode,
        "seed": config.seed,
        "tokens_per_domain": config.tokens_per_domain,
        "num_valid_windows": run_config["num_valid_windows"],
        "stage_params_sha256": [entry["params_sha256"] for entry in stages],
        "exposure_within_one_update": all(
            bool(item and item["within_one_update"]) for item in exposure.values()
        ),
    }
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "exposure.json", exposure)
    atomic_write_json(
        output / "runtime.json",
        {
            "device": str(jax.devices()[0]),
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "x64": jnp.asarray(1.0).dtype == jnp.float64,
            "wall_seconds": time.time() - started,
            "stages": stage_runtime,
        },
    )
    return {
        "matrices": matrices,
        "summary": summary,
        "stages": stages,
        "exposure": exposure,
        "output_dir": str(output),
    }


def _comparable_config(run_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a run's config without the keys a comparison may vary."""
    values = dict(run_config["config"])
    for key in list(values):
        if key in COMPARE_IGNORED_KEYS or key.startswith("pc_"):
            values.pop(key)
    return json.loads(dumps_strict(values, sort_keys=True))


def check_comparable_runs(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> list[str]:
    """Return why two ``run_config.json`` payloads cannot be compared (empty if fine).

    The configs must be equal except for ``mode``, ``pc_*``, ``output_dir``,
    ``log_interval`` and ``init_perturbation`` (with its seed), with equal
    validation and train-array SHA-256s, the same source parameters and the
    same domain order.
    """
    problems = []
    a, b = _comparable_config(left), _comparable_config(right)
    differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if differing:
        problems.append(f"configs differ in {differing}")
    if list(left["config"]["domains"]) != list(right["config"]["domains"]):
        problems.append("domain orders differ")
    for key in ("valid_sha256", "train_array_sha256"):
        if left.get(key) != right.get(key):
            problems.append(f"{key} differ")
    if left["source"]["params_sha256"] != right["source"]["params_sha256"]:
        problems.append("source checkpoints differ")
    return problems


def compare_runs(run_dir: str | Path, reference_dir: str | Path) -> dict[str, Any]:
    """Compare a run with a reference run of the same setup.

    Used for PC vs BP and for BP vs a BP run from a perturbed initialisation
    (the float32 noise floor). Raises ``ValueError`` unless
    ``check_comparable_runs`` passes.
    """
    run_path, reference_path = Path(run_dir), Path(reference_dir)
    run_config = json.loads((run_path / "run_config.json").read_text())
    reference_config = json.loads((reference_path / "run_config.json").read_text())
    problems = check_comparable_runs(run_config, reference_config)
    if problems:
        raise ValueError(
            f"{run_path} and {reference_path} are not comparable: "
            + "; ".join(problems)
        )
    ce_run = json.loads((run_path / MATRIX_FILES["masked_ce"]).read_text())["matrix"]
    ce_ref = json.loads((reference_path / MATRIX_FILES["masked_ce"]).read_text())[
        "matrix"
    ]
    stages = range(len(ce_ref))
    return {
        "modes": [run_config["mode"], reference_config["mode"]],
        "init_perturbation": [
            run_config["config"].get("init_perturbation", 0.0),
            reference_config["config"].get("init_perturbation", 0.0),
        ],
        **pc_bp_agreement(
            ce_run,
            ce_ref,
            [load_npz(run_path / f"stage_{s}.npz") for s in stages],
            [load_npz(reference_path / f"stage_{s}.npz") for s in stages],
        ),
    }


def compare_pc_bp_runs(pc_dir: str | Path, bp_dir: str | Path) -> dict[str, Any]:
    """Compare a PC run directory with a BP run directory (``compare_runs``)."""
    return compare_runs(pc_dir, bp_dir)

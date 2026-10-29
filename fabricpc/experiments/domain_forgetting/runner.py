"""Sequential MLM domain-forgetting runner for backprop, spike-only and standard PC.

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

Standard PC (``pc_inference="standard"``) runs ``pc_infer_steps`` steps of
norm-clipped SGD inference at rate ``pc_eta_infer`` and takes the weight
update from the final inference state (``pc_forward_updates="none"``). Its
gradient is not equal to backprop's, so the scaled clip and ``eps`` are kept
as conventions that match spike-only PC, not as an equivalence; the scale
relation to backprop is only approximate. ePC (``pc_inference="epc"``)
runs ``pc_infer_steps`` steps of error-parameterized inference
(``EPCInference``) at rate ``pc_eta_infer`` with the same weight rule and
energy statistics; its regime (backprop-like to PC equilibrium) is read from
the ε-spectrum (``pc_regime_interval``, or the ``probe`` command).

Each update is the gradient of the mean masked CE over the whole batch: the
backprop micro-batch gradients are weighted by their masked-token counts
(``Σ_k g_k n_k / Σ_k n_k``); PC micro-batch gradients are token sums, since
the MLM head energy is ``s · Σ_masked CE`` (a fixed per-token precision), and
are added and divided once by ``Σ_k n_k``.

With ``training="joint"`` the run is the multi-task upper bound instead: one
stage trains on the concatenated domain streams for ``len(domains) *
tokens_per_domain`` tokens under one schedule, with windows shuffled across
domains, so each domain is seen for its full budget within one update.

With ``gate`` other than ``"none"`` every stage from 2 on multiplies each
module's AdamW update by a fixed per-module multiplier (``gates.GatePlan``),
chained after the optimizer so that a multiplier of 0 freezes the module;
frozen modules' gradients are also zeroed before the global-norm clip.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

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
from fabricpc.core.inference import (
    InferenceSGDNormClip,
    InferenceSGDNormClipDecay,
    InferenceSpikingPrecision,
    gather_inputs,
)
from fabricpc.core.epsilon_spectrum import EpsilonSpectrum, make_epsilon_spectrum
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.scaling import scale_inputs
from fabricpc.core.types import GraphParams, GraphState, GraphStructure, NodeParams
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
from fabricpc.training.plasticity_gates import scale_module_updates
from fabricpc.training.train import _convert_batch
from fabricpc.training.train_backprop import (
    _apply_optimizer_update_when_masked,
    _loss_and_grad_backprop_mlm_no_update,
)
from fabricpc.training.train_mlm import (
    _prepare_pcn_mlm_batch,
    create_jit_gradient_step_pcn_mlm,
    create_pc_bp_gradient_comparison,
    get_graph_param_gradient_pcn_mlm,
    validate_epc_forward_updates,
)
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
TRAINING = ("sequential", "joint")
JOINT_STAGE = "joint"
GATES = ("none", "map", "random", "inverse", "type", "oracle", "freeze")
GATE_SETTINGS = {
    "none": (),
    "map": ("gate_beta", "gate_map_windows"),
    "inverse": ("gate_beta", "gate_map_windows"),
    "type": ("gate_beta", "gate_map_windows"),
    "random": ("gate_beta", "gate_map_windows", "gate_seed"),
    "oracle": ("gate_beta", "gate_source"),
    "freeze": ("gate_modules",),
}
PC_INFERENCE = ("spiking", "standard", "epc")
ITERATIVE_PC = ("standard", "epc")
DEFAULT_FORWARD_UPDATES = {"spiking": "initial", "standard": "none", "epc": "none"}
HEAD_ENERGY = "sum_masked_ce"
METRICS = ("masked_ce", "masked_perplexity", "masked_accuracy")
MATRIX_FILES = {
    "masked_ce": "masked_ce_matrix.json",
    "masked_perplexity": "masked_perplexity_matrix.json",
    "masked_accuracy": "masked_accuracy_matrix.json",
}
REGIME_DEFAULTS: dict[str, Any] = {
    "pc_regime_interval": 0,
    "pc_regime_eta": None,
    "pc_regime_steps": None,
    "pc_regime_iters": 20,
    "pc_regime_final": False,
}
ETA_CONTROL = ("none", "energy")
CONTROL_SETTINGS = (
    "pc_energy_target",
    "pc_eta_gain",
    "pc_eta_min",
    "pc_eta_max",
    "pc_eta_backoff",
)
ETA_DROP_EMA = 0.7
ETA_STEP_CLIP = (0.5, 2.0)
RESUME_IGNORED_KEYS = (
    "log_interval",
    "output_dir",
    "pc_alignment_batches",
    *REGIME_DEFAULTS,
)
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
    "alignment.jsonl",
    "regime.jsonl",
    "gates.jsonl",
    *ANALYSIS_FILES,
)
RUN_ARTIFACT_PATTERNS = ("stage_*.npz", "training_log_stage_*.jsonl")


@dataclass(frozen=True)
class RunConfig:
    """Every setting that determines a run's results.

    The regime probe (``pc_regime_*``) only reads the params. Each probe runs
    ``pc_regime_iters`` Lanczos steps, one Hessian-vector product (a forward
    and a double backward pass through the ε-energy) each, about as much as
    tens of training micro-batch gradients; ``pc_regime_final`` adds
    ``pc_regime_steps`` ePC steps and a second spectrum at the relaxed errors.

    ``pc_eta_control="energy"`` (ePC only) adapts the ePC rate after every
    update so the measured relative energy drop tracks ``pc_energy_target``
    (``control_eta``), within ``eta_bounds``; the rate restarts at
    ``pc_eta_infer`` at every stage.
    """

    checkpoint: str
    data_dir: str
    domains: tuple[str, ...]
    output_dir: str
    mode: str = "backprop"
    training: str = "sequential"
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
    pc_alignment_batches: int = 0
    pc_regime_interval: int = 0
    pc_regime_eta: float | None = None
    pc_regime_steps: int | None = None
    pc_regime_iters: int = 20
    pc_regime_final: bool = False
    pc_eta_control: str = "none"
    pc_energy_target: float = 0.17
    pc_eta_gain: float = 0.5
    pc_eta_min: float | None = None
    pc_eta_max: float | None = None
    pc_eta_backoff: float = 0.5
    init_perturbation: float = 0.0
    init_perturbation_seed: int = 0
    gate: str = "none"
    gate_beta: float = 1.0
    gate_modules: tuple[str, ...] = ()
    gate_source: tuple[str, ...] = ()
    gate_map_windows: int = 256
    gate_seed: int = 0
    log_interval: int = 10

    def validate(self) -> None:
        validate_domain_names(self.domains)
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.training not in TRAINING:
            raise ValueError(
                f"training must be one of {TRAINING}, got {self.training!r}"
            )
        if self.training == "joint":
            if len(self.domains) < 2:
                raise ValueError("joint training needs at least two domains")
            if self.pc_alignment_batches:
                raise ValueError("pc_alignment_batches needs training 'sequential'")
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
        if self.pc_alignment_batches < 0:
            raise ValueError("pc_alignment_batches must be non-negative")
        if self.mode != "pc" and self.pc_alignment_batches:
            raise ValueError("pc_alignment_batches needs mode 'pc'")
        self._validate_regime()
        if self.mode == "pc":
            if self.pc_inference not in PC_INFERENCE:
                raise ValueError(
                    f"pc_inference must be one of {PC_INFERENCE}, "
                    f"got {self.pc_inference!r}"
                )
            if self.pc_output_energy_scale <= 0.0:
                raise ValueError("pc_output_energy_scale must be positive")
            if self.pc_inference in ITERATIVE_PC:
                self._validate_iterative()
        self._validate_eta_control()
        self._validate_gate()

    def _validate_eta_control(self) -> None:
        """The rate controller's settings apply only to ePC runs that enable it."""
        if self.pc_eta_control not in ETA_CONTROL:
            raise ValueError(
                f"pc_eta_control must be one of {ETA_CONTROL}, "
                f"got {self.pc_eta_control!r}"
            )
        if self.pc_eta_control == "none":
            unused = [
                name
                for name, default in CONTROL_DEFAULTS.items()
                if getattr(self, name) != default
            ]
            if unused:
                raise ValueError(f"{', '.join(unused)} need pc_eta_control 'energy'")
            return
        if self.mode != "pc" or self.pc_inference != "epc":
            raise ValueError("pc_eta_control 'energy' needs mode 'pc' and ePC")
        if not 0.0 < self.pc_energy_target < 1.0:
            raise ValueError("pc_energy_target must be in (0, 1)")
        if not (math.isfinite(self.pc_eta_gain) and self.pc_eta_gain > 0.0):
            raise ValueError("pc_eta_gain must be finite and positive")
        if not 0.0 < self.pc_eta_backoff <= 1.0:
            raise ValueError("pc_eta_backoff must be in (0, 1]")
        low, high = self.eta_bounds
        if not all(math.isfinite(v) and v > 0.0 for v in (low, high)):
            raise ValueError("pc_eta_min and pc_eta_max must be finite and positive")
        if low > high:
            raise ValueError("pc_eta_min must not exceed pc_eta_max")
        if not low <= self.pc_eta_infer <= high:
            raise ValueError("pc_eta_infer must lie in [pc_eta_min, pc_eta_max]")

    @property
    def eta_bounds(self) -> tuple[float, float]:
        """The controller's (min, max) rate, defaulting to pc_eta_infer / 100 and
        pc_eta_infer."""
        low = self.pc_eta_infer / 100 if self.pc_eta_min is None else self.pc_eta_min
        high = self.pc_eta_infer if self.pc_eta_max is None else self.pc_eta_max
        return float(low), float(high)

    def _validate_regime(self) -> None:
        """The regime probe's settings apply only when its interval is set.

        Any mode may probe (backprop is the control). ``pc_regime_eta`` and
        ``pc_regime_steps`` default to the run's ePC settings and are
        required otherwise.
        """
        if self.pc_regime_interval < 0:
            raise ValueError("pc_regime_interval must be non-negative")
        if self.pc_regime_interval == 0:
            unused = [
                name
                for name, default in REGIME_DEFAULTS.items()
                if getattr(self, name) != default
            ]
            if unused:
                raise ValueError(f"{', '.join(unused)} need pc_regime_interval > 0")
            return
        eta, steps = self.regime_setting
        if eta is None or steps is None:
            raise ValueError(
                "pc_regime_interval needs pc_regime_eta and pc_regime_steps "
                "unless the run is ePC"
            )
        if not (math.isfinite(eta) and eta > 0):
            raise ValueError("pc_regime_eta must be finite and positive")
        if steps < 1:
            raise ValueError("pc_regime_steps must be >= 1")
        if self.pc_regime_iters < 1:
            raise ValueError("pc_regime_iters must be >= 1")

    @property
    def regime_setting(self) -> tuple[float | None, int | None]:
        """The (eta, steps) the regime probe judges, defaulting to the run's ePC."""
        epc = self.mode == "pc" and self.pc_inference == "epc"
        eta = self.pc_regime_eta
        steps = self.pc_regime_steps
        if epc:
            eta = self.pc_eta_infer if eta is None else eta
            steps = self.pc_infer_steps if steps is None else steps
        return eta, steps

    def _validate_gate(self) -> None:
        """Each gate takes only its own settings, and gates need sequential runs.

        Freeze module and group names are resolved against the graph when the
        run starts (``gates.GatePlan``).
        """
        if self.gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {self.gate!r}")
        used = GATE_SETTINGS[self.gate]
        unused = [
            name
            for name, default in GATE_DEFAULTS.items()
            if name not in used and _setting(getattr(self, name)) != default
        ]
        if unused:
            raise ValueError(f"{', '.join(unused)} do not apply to gate {self.gate!r}")
        if self.gate == "none":
            return
        if self.training != "sequential":
            raise ValueError("gates need training 'sequential'")
        if "gate_beta" in used and not (
            math.isfinite(self.gate_beta) and self.gate_beta > 0
        ):
            raise ValueError("gate_beta must be finite and positive")
        if "gate_map_windows" in used and self.gate_map_windows < 1:
            raise ValueError("gate_map_windows must be positive")
        if self.gate_seed < 0:
            raise ValueError("gate_seed must be non-negative")
        if self.gate == "freeze":
            if not self.gate_modules:
                raise ValueError("gate 'freeze' needs gate_modules")
            if len(set(self.gate_modules)) != len(self.gate_modules):
                raise ValueError("gate_modules has duplicates")
        if self.gate == "oracle":
            if not self.gate_source:
                raise ValueError("gate 'oracle' needs gate_source run directories")
            if len(set(self.gate_source)) != len(self.gate_source):
                raise ValueError("gate_source has duplicates")

    def _validate_iterative(self) -> None:
        """Standard PC and ePC need explicit inference settings, no spike flags."""
        name = "standard PC" if self.pc_inference == "standard" else "ePC"
        if not self.pc_eta_infer > 0.0:
            raise ValueError(f"{name} needs pc_eta_infer > 0")
        if self.pc_infer_steps is None or self.pc_infer_steps < 1:
            raise ValueError(
                f"{name} needs pc_infer_steps >= 1 (there is no depth default)"
            )
        if self.pc_forward_updates != "none":
            raise ValueError(
                f"{name} needs pc_forward_updates 'none' (weight update from "
                f"the final inference state), got {self.pc_forward_updates!r}"
            )
        default_max_norm = type(self).__dataclass_fields__["pc_max_norm"].default
        if self.pc_inference == "epc" and self.pc_max_norm != default_max_norm:
            raise ValueError(
                "pc_max_norm does not apply to ePC (no latent-gradient clip)"
            )
        spike_flags = [
            name
            for name, value, default in (
                ("pc_spike_eta", self.pc_spike_eta, 1.0),
                ("pc_spike_depth", self.pc_spike_depth, "longest"),
            )
            if value != default
        ]
        if spike_flags:
            raise ValueError(
                f"{', '.join(spike_flags)} only apply to spiking PC, "
                f"not {self.pc_inference}"
            )

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

    @property
    def stage_domains(self) -> tuple[str, ...]:
        """The domain trained at each stage ``1..n``; joint training has one stage."""
        return (JOINT_STAGE,) if self.training == "joint" else tuple(self.domains)

    @property
    def stage_tokens(self) -> int:
        """Stream tokens per trained stage."""
        if self.training == "joint":
            return self.tokens_per_domain * len(self.domains)
        return self.tokens_per_domain

    def identity(self) -> dict[str, Any]:
        """The settings that must match for a run to be resumed.

        ``training`` is left out for sequential runs, every gate setting
        for ungated runs and every rate-controller setting for uncontrolled
        runs, so their identity is unchanged from before joint training,
        gates and the controller existed.
        """
        values = asdict(self)
        for key in RESUME_IGNORED_KEYS:
            values.pop(key)
        if values["training"] == "sequential":
            values.pop("training")
        if values["gate"] == "none":
            for key in ("gate", *GATE_DEFAULTS):
                values.pop(key)
        else:
            values["gate_modules"] = list(self.gate_modules)
            values["gate_source"] = list(self.gate_source)
        if values["pc_eta_control"] == "none":
            for key in ("pc_eta_control", *CONTROL_DEFAULTS):
                values.pop(key)
        values["domains"] = list(self.domains)
        return values


GATE_DEFAULTS: dict[str, Any] = {
    f.name: f.default for f in fields(RunConfig) if f.name.startswith("gate_")
}
CONTROL_DEFAULTS: dict[str, Any] = {
    f.name: f.default for f in fields(RunConfig) if f.name in CONTROL_SETTINGS
}


def _setting(value: Any) -> Any:
    """Return lists as tuples so settings compare equal to their defaults."""
    return tuple(value) if isinstance(value, list) else value


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
    """Spike-only PC inference with ``infer_steps`` steps."""
    return InferenceSpikingPrecision(
        eta_infer=config.pc_eta_infer,
        infer_steps=infer_steps,
        max_norm=config.pc_max_norm,
        spike_eta=config.pc_spike_eta,
        depth_mode=config.pc_spike_depth,
    )


def _standard_inference(config: RunConfig) -> Any:
    """Standard PC inference that also counts per-sample energy rises.

    ``rise_decay=1`` keeps every update identical to ``InferenceSGDNormClip``;
    the guarded step only adds the rise count used to detect divergence.
    """
    return InferenceSGDNormClipDecay(
        eta_infer=config.pc_eta_infer,
        infer_steps=config.pc_infer_steps,
        max_norm=config.pc_max_norm,
        rise_decay=1.0,
    )


def build_inference(
    config: RunConfig, model_config: Mapping[str, Any]
) -> tuple[Any, int | None]:
    """Return the run's inference object and its PC inference step count.

    Spike-only PC defaults ``pc_infer_steps`` to the longest free depth and
    rejects fewer steps; standard PC and ePC use ``pc_infer_steps`` as given.
    Backprop gets a placeholder inference object and ``None``.
    """
    if config.mode != "pc":
        return InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=5.0), None
    if config.pc_inference == "standard":
        return _standard_inference(config), int(config.pc_infer_steps)
    if config.pc_inference == "epc":
        return (
            EPCInference(
                eta_infer=config.pc_eta_infer,
                infer_steps=int(config.pc_infer_steps),
                latent_decay=0.0,
            ),
            int(config.pc_infer_steps),
        )
    probe = build_structure(model_config, config, _spiking_inference(config, 1))
    needed = max_free_depth(probe, config.pc_spike_depth)
    infer_steps = config.pc_infer_steps or needed
    if infer_steps < needed:
        raise ValueError(
            f"pc_infer_steps {infer_steps} < longest free depth {needed}; "
            "the deepest nodes would never spike"
        )
    return _spiking_inference(config, infer_steps), infer_steps


def load_base_model(config: RunConfig) -> BaseModel:
    """Validate the source checkpoint and load its parameters into memory.

    Checks ``architecture == "bert"``, the expected shape, that the tokenizer
    vocabulary equals the checkpoint vocabulary, and (through
    ``load_init_params``) architecture, ``seq_len``, ``ln_eps`` and every
    parameter shape. The inference object comes from ``build_inference``.
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

    inference, infer_steps = build_inference(config, model_config)
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
    config: RunConfig,
    num_updates: int,
    multipliers: Mapping[str, float] | None = None,
) -> tuple[optax.GradientTransformation, dict[str, Any]]:
    """AdamW with linear warmup then cosine decay to ``min_lr_ratio * lr``.

    Mirrors ``build_optimizer`` in ``examples/bert_demo.py`` (global-norm clip
    chained before ``build_warmup_cosine_adamw``) and adds the decay floor.
    ``multipliers`` chains ``scale_module_updates`` after AdamW, so it scales
    each module's final update, weight decay included; modules with
    multiplier 0 also get zero gradients before the clip, so the clip's
    global norm covers only the trainable modules.
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
    info: dict[str, Any] = {
        "type": "warmup_cosine_adamw",
        "lr": config.lr,
        "min_lr": config.min_lr_ratio * config.lr,
        "warmup_steps": warmup_steps,
        "total_steps": num_updates,
        "weight_decay": config.weight_decay,
        "clip_norm": config.clip_norm,
        "adam_eps": config.optimizer_eps,
    }
    if multipliers is not None:
        frozen = {name: 0.0 for name, value in multipliers.items() if value == 0.0}
        optimizer = optax.chain(optimizer, scale_module_updates(multipliers))
        if frozen:
            optimizer = optax.chain(scale_module_updates(frozen), optimizer)
        info["module_multipliers"] = {k: float(v) for k, v in multipliers.items()}
    return optimizer, info


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


class GradientFn(Protocol):
    """``(params, batch, key, eta_infer=None) -> (grads, loss_sum, count, extras)``.

    ``eta_infer`` overrides the ePC rate for one call; only ePC accepts it.
    """

    def __call__(
        self,
        params: GraphParams,
        batch: dict[str, jax.Array],
        key: jax.Array,
        eta_infer: jax.Array | None = None,
    ) -> tuple[GraphParams, jax.Array, jax.Array, dict[str, jax.Array]]: ...


def _reject_eta_infer(eta_infer: jax.Array | None) -> None:
    """Raise unless ``eta_infer`` is None (the gradient function is not ePC)."""
    if eta_infer is not None:
        raise ValueError("eta_infer overrides only the ePC rate")


def state_total_energy(
    params: GraphParams, state: GraphState, structure: GraphStructure
) -> jax.Array:
    """Batch-mean total energy of ``state`` over every node with inputs.

    Each node's prediction is recomputed from the current latents of its
    inputs, so the energy is that of ``state`` itself.
    """
    total = jnp.array(0.0)
    batch_size = None
    for name in structure.node_order:
        node_info = structure.nodes[name].node_info
        if node_info.in_degree == 0:
            continue
        inputs = scale_inputs(
            gather_inputs(node_info, structure, state), node_info.scaling_config
        )
        _, projected = node_info.node_class.forward(
            params.nodes[name], inputs, state.nodes[name], node_info
        )
        batch_size = projected.energy.shape[0]
        total = total + jnp.sum(projected.energy)
    return total / (batch_size or 1)


def make_gradient_fn(
    mode: str,
    structure: GraphStructure,
    forward_updates: str = "initial",
    pc_inference: str = "spiking",
) -> GradientFn:
    """Return a jitted ``(params, batch, key) -> (grads, loss_sum, count, extras)``.

    Backprop ``grads`` are the gradient of the batch's mean masked CE; PC
    ``grads`` are summed over its masked tokens (the head energy is
    ``s · Σ_masked CE``), so ``accumulate_gradients`` needs ``summed=True``.
    ``loss_sum`` is the masked CE summed over the batch's masked tokens.
    Both modes draw dropout masks from ``key`` the same way. Standard PC and
    ePC extras add ``pc_energy_start`` (the target-clamped feedforward state)
    and ``pc_energy_end`` (the final inference state the weight update uses),
    both batch means of the total energy recomputed at the state's latents.
    ``pc_energy_rises`` counts per-sample energy increases between
    consecutive inference steps, summed over samples; ePC also reports
    ``pc_energy_total_rises``, the steps whose batch-total energy rose. The
    energies are per-sample sums under the head energy ``s · Σ_masked CE``, in
    units about ``N`` (masked tokens per micro-batch) times those of the
    earlier mean-CE head. ``pc_total_energy`` is the batch mean of the
    energies stored in the final state: under standard PC they were computed
    before the last latent update, so it is the energy of step ``T - 1``, not
    ``pc_energy_end``; ePC's final state is re-derived at its final errors, so
    there it equals ``pc_energy_end``.

    The ePC function takes an optional traced ``eta_infer`` that replaces the
    configured rate, so changing it between calls does not recompile.
    """
    if mode == "backprop":

        @jax.jit
        def backprop(params, batch, key, eta_infer=None):
            _reject_eta_infer(eta_infer)
            loss, grads, count = _loss_and_grad_backprop_mlm_no_update(
                params, batch, structure, key
            )
            return grads, loss * count, count, {}

        return backprop
    validate_epc_forward_updates(structure, forward_updates)
    if pc_inference in ITERATIVE_PC:

        @jax.jit
        def standard(params, batch, key, eta_infer=None):
            grads, metrics, initial, _free, final = get_graph_param_gradient_pcn_mlm(
                params, batch, structure, key, forward_updates, eta_infer
            )
            count = metrics["num_masked_tokens"]
            return (
                grads,
                metrics["masked_loss"] * count,
                count,
                {
                    "pc_energy_rises": metrics["pc_energy_rises"],
                    "pc_total_energy": metrics["pc_total_energy"],
                    "pc_energy_start": state_total_energy(params, initial, structure),
                    "pc_energy_end": state_total_energy(params, final, structure),
                    **{
                        key: metrics[key]
                        for key in ("pc_energy_total_rises",)
                        if key in metrics
                    },
                },
            )

        return standard
    pc_step = create_jit_gradient_step_pcn_mlm(structure, forward_updates)

    def pc(params, batch, key, eta_infer=None):
        _reject_eta_infer(eta_infer)
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
    *,
    summed: bool = False,
    eta_infer: jax.Array | None = None,
) -> tuple[GraphParams, float, float, list[dict[str, Any]]]:
    """Return the token-weighted mean gradient of one update's micro-batches.

    With ``summed`` false ``gradient_fn`` returns each micro-batch's mean-CE
    gradient ``g_k`` over its ``n_k`` masked tokens (backprop) and the result
    is ``Σ_k g_k n_k / max(Σ_k n_k, 1)``. With ``summed`` true it returns the
    token-summed gradient ``G_k`` (PC, whose head energy is ``s · Σ_masked
    CE``) and the result is ``Σ_k G_k / max(Σ_k n_k, 1)``: the normalisation
    happens once, over the whole update. Also returns the summed masked CE,
    the total count and the per-micro-batch extras. ``eta_infer``, when given,
    is passed to every micro-batch, so one update uses one ePC rate.
    """
    rate = {} if eta_infer is None else {"eta_infer": eta_infer}
    total = None
    loss_sum = 0.0
    count = 0.0
    extras = []
    for batch, key in zip(micro_batches, keys, strict=True):
        grads, batch_loss, batch_count, batch_extras = gradient_fn(
            params, batch, key, **rate
        )
        weighted = (
            grads
            if summed
            else jax.tree_util.tree_map(lambda g: g * batch_count, grads)
        )
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


class RegimeProbe(Protocol):
    """``(params, batch, key, eta_infer=None) -> (spectrum, final_spectrum)``."""

    def __call__(
        self,
        params: GraphParams,
        batch: dict[str, jax.Array],
        key: jax.Array,
        eta_infer: jax.Array | None = None,
    ) -> tuple[EpsilonSpectrum, EpsilonSpectrum | None]: ...


def make_regime_probe(
    structure: GraphStructure,
    iters: int = 20,
    final: tuple[float, int] | None = None,
) -> RegimeProbe:
    """Return a jitted ``(params, batch, key, eta_infer=None) -> (spectrum,
    final_spectrum)``.

    ``spectrum`` is the ε-energy Hessian's ``EpsilonSpectrum`` (on device)
    at the target-clamped feedforward state of ``batch`` (ε = 0), with
    ``iters`` Lanczos steps; dropout masks come from ``key`` when
    ``structure`` has dropout. With ``final = (eta, steps)``,
    ``final_spectrum`` is the spectrum at the errors reached by ``steps``
    ePC steps at rate ``eta`` from there (else None): the regime verdict is
    a statement about ε = 0, and this checks it along the relaxation.
    ``eta_infer``, when given, replaces ``eta`` for that relaxation without
    recompiling.
    """
    spectrum_fn = make_epsilon_spectrum(structure, iters)
    relaxed = None
    if final is not None:
        eta, steps = final
        relaxed = structure._replace(
            config={
                **structure.config,
                "inference": EPCInference(eta_infer=eta, infer_steps=int(steps)),
            }
        )

    @jax.jit
    def probe(params, batch, key, eta_infer=None):
        _shared, _metrics, _free, initial, clamps = _prepare_pcn_mlm_batch(
            params, batch, structure, key
        )
        spectrum = spectrum_fn(params, initial, clamps, key)
        if relaxed is None:
            return spectrum, None
        state = EPCInference.run_inference(
            params, initial, clamps, relaxed, eta_infer=eta_infer
        )
        return spectrum, spectrum_fn(params, state, clamps, key)

    return probe


def spectrum_scalars(spectrum: EpsilonSpectrum) -> dict[str, Any]:
    """The JSON scalars of one ``EpsilonSpectrum``."""
    spectrum = spectrum.host()
    return {
        "lambda_max": float(spectrum.lambda_max),
        "lambda_min": float(spectrum.lambda_min),
        "negative_weight": float(spectrum.negative_weight),
        "gradient_norm": float(spectrum.gradient_norm),
        "lanczos_k": int(spectrum.k),
        "random_start": bool(spectrum.random_start),
    }


def regime_summary(
    spectrum: EpsilonSpectrum,
    eta: float,
    steps: int,
    final_spectrum: EpsilonSpectrum | None = None,
) -> dict[str, Any]:
    """The spectrum's scalars and ``EPCInference(eta, steps).regime`` verdict.

    The verdict reads the quadratic model at the spectrum's point (ε = 0 for
    ``make_regime_probe``), so it can mispredict on an indefinite spectrum or
    when the curvature grows as ε relaxes. With ``final_spectrum`` (the
    spectrum at the relaxed errors) the summary adds ``lambda_max_final``,
    ``lambda_min_final``, ``eta_lambda_max_final`` and ``unstable_final``
    (``eta · lambda_max_final > 2``).

    ``f_valid`` is true when f̄ is finite, in [0, 1] and ``eta · lambda_max
    <= 2``. Otherwise ``f_weighted`` is None, and unless no positive mode
    carries gradient weight ``band`` is "invalid", ``f_max`` is None and
    ``regime`` names the cause with ``eta · lambda_max`` only.
    """
    host = spectrum.host()
    regime = EPCInference(eta_infer=eta, infer_steps=steps).regime(host)
    f_weighted = regime.f_weighted
    f_valid = (
        math.isfinite(f_weighted)
        and 0.0 <= f_weighted <= 1.0
        and regime.eta_lambda_max <= 2.0
    )
    invalid = not f_valid and not (math.isnan(f_weighted) and not regime.unstable)
    verdict = str(regime)
    if invalid:
        cause = "unstable" if regime.unstable else "relaxed fraction outside [0, 1]"
        verdict = f"invalid: {cause}, eta*lambda_max = {regime.eta_lambda_max:.3g}"
    summary = {
        **spectrum_scalars(host),
        "f_weighted": f_weighted if f_valid else None,
        "f_valid": bool(f_valid),
        "f_max": None if invalid else regime.f_max,
        "band": "invalid" if invalid else regime.band,
        "unstable": bool(regime.unstable),
        "regime": verdict,
    }
    if final_spectrum is not None:
        final = spectrum_scalars(final_spectrum)
        summary["lambda_max_final"] = final["lambda_max"]
        summary["lambda_min_final"] = final["lambda_min"]
        summary["eta_lambda_max_final"] = eta * final["lambda_max"]
        summary["unstable_final"] = eta * final["lambda_max"] > 2.0
    return summary


def module_weight_norms(
    params: GraphParams, structure: GraphStructure
) -> dict[str, float]:
    """Global L2 norm of each parameterised node's parameters, in graph order."""
    norms = {}
    for name in structure.node_order:
        leaves = jax.tree_util.tree_leaves(params.nodes[name])
        if leaves:
            norms[name] = float(optax.tree.norm(leaves))
    return norms


class AlignmentFn(Protocol):
    """``(params, batch, key, eta_infer=None) -> {module: {pc_norm, bp_norm,
    cosine}}``; ``eta_infer`` overrides the ePC rate."""

    def __call__(
        self,
        params: GraphParams,
        batch: dict[str, jax.Array],
        key: jax.Array,
        eta_infer: jax.Array | None = None,
    ) -> dict[str, dict[str, jax.Array]]: ...


def make_alignment_fn(
    structure: GraphStructure, forward_updates: str = "none"
) -> AlignmentFn:
    """Return the jitted per-module PC vs backprop gradient comparison; its
    optional traced ``eta_infer`` overrides the ePC rate without recompiling."""
    return create_pc_bp_gradient_comparison(structure, forward_updates)


def fixed_mlm_batches(
    stream: np.ndarray,
    structure: GraphStructure,
    special_ids: Mapping[str, int],
    *,
    num_batches: int,
    batch_size: int,
    mask_prob: float,
    seed: int,
) -> tuple[list[dict[str, jax.Array]], list[jax.Array]]:
    """Return the first ``num_batches`` unshuffled MLM batches and their keys.

    Masks come from ``seed`` and the dropout keys from ``PRNGKey(seed)``,
    so the batches are the same on every call.
    """
    seq_len = int(structure.nodes[structure.task_map["y"]].node_info.shape[0])
    vocab_size = int(structure.nodes[structure.task_map["y"]].node_info.shape[-1])
    batcher = MlmTextBatcher(
        token_ids=np.asarray(stream, dtype=np.int32),
        seq_len=seq_len,
        batch_size=batch_size,
        pad_token_id=special_ids["[PAD]"],
        mask_token_id=special_ids["[MASK]"],
        vocab_size=vocab_size,
        special_token_ids=set(special_ids.values()),
        cls_token_id=special_ids["[CLS]"],
        seed=seed,
        mask_prob=mask_prob,
        shuffle=False,
        token_unit="subword",
    )
    batches = []
    for batch_data in batcher:
        if len(batches) >= num_batches:
            break
        batches.append(_convert_batch(batch_data))
    if len(batches) < num_batches:
        raise ValueError(
            f"domain stream has {len(batches)} batches, {num_batches} requested"
        )
    keys = list(jax.random.split(jax.random.PRNGKey(seed), num_batches))
    return batches, keys


def gradient_alignment(
    alignment_fn: AlignmentFn,
    params: GraphParams,
    batches: Sequence[Mapping[str, jax.Array]],
    keys: Sequence[jax.Array],
    structure: GraphStructure,
    eta_infer: jax.Array | None = None,
) -> dict[str, Any]:
    """Average the per-module PC vs backprop comparison over ``batches``.

    ``eta_infer``, when given, is the ePC rate of every comparison.

    A module is one parameterised graph node, in graph order. Per module:
    the mean and minimum cosine, the mean norms and ``norm_ratio`` (mean of
    ``pc_norm / bp_norm`` over batches with a non-zero BP gradient, else
    ``None``); ``rel_norm_ratio`` divides it by the ``mlm_head`` ratio.
    PC gradients carry the output-energy scale ``s`` and backprop gradients do
    not, so ``norm_ratio`` includes ``s``; ``rel_norm_ratio`` cancels it.
    ``mean_cosine`` includes the head, whose cosine is always about 1;
    ``mean_cosine_hidden`` leaves it out. ``module_order`` lists the modules
    in graph order.
    """
    rate = {} if eta_infer is None else {"eta_infer": eta_infer}
    per_batch = [
        jax.tree_util.tree_map(float, alignment_fn(params, batch, key, **rate))
        for batch, key in zip(batches, keys, strict=True)
    ]
    if not per_batch:
        raise ValueError("gradient alignment needs at least one batch")
    modules: dict[str, dict[str, Any]] = {}
    for name in [n for n in structure.node_order if n in per_batch[0]]:
        pc = np.array([b[name]["pc_norm"] for b in per_batch])
        bp = np.array([b[name]["bp_norm"] for b in per_batch])
        cosine = np.array([b[name]["cosine"] for b in per_batch])
        nonzero = bp > 0
        modules[name] = {
            "cosine": float(cosine.mean()),
            "cosine_min": float(cosine.min()),
            "pc_norm": float(pc.mean()),
            "bp_norm": float(bp.mean()),
            "norm_ratio": (
                float(np.mean(pc[nonzero] / bp[nonzero])) if nonzero.any() else None
            ),
        }
    head = modules.get(structure.task_map["y"], {}).get("norm_ratio")
    for row in modules.values():
        row["rel_norm_ratio"] = (
            row["norm_ratio"] / head if head and row["norm_ratio"] is not None else None
        )
    cosines = [row["cosine"] for row in modules.values()]
    hidden = [
        row["cosine"]
        for name, row in modules.items()
        if name != structure.task_map["y"]
    ]
    return {
        "num_batches": len(per_batch),
        "mean_cosine": float(np.mean(cosines)),
        "mean_cosine_hidden": float(np.mean(hidden)) if hidden else None,
        "min_cosine": float(np.min(cosines)),
        "module_order": list(modules),
        "modules": modules,
    }


def control_eta(
    eta: float,
    smoothed: float | None,
    drop: float,
    increased: bool,
    *,
    target: float,
    gain: float,
    eta_min: float,
    eta_max: float,
    backoff: float,
) -> tuple[float, float, bool]:
    """One step of the ePC rate controller.

    Returns ``(eta_next, smoothed_next, backed_off)``. An update whose
    inference ended above its start energy (``increased``) or whose ``drop``
    is not finite backs off: it enters the average as a drop of 1, the
    deepest possible, and caps the step factor at ``backoff``. The drop is
    smoothed as d̃ ← 0.7·d̃ + 0.3·d, seeded with the first d; the step factor
    is r = clip((target / max(d̃, 1e-6))^gain, 0.5, 2.0) and
    eta_next = clip(eta·r, eta_min, eta_max). Averaging a divergence as a
    full drop keeps the following updates from raising the rate straight
    back to where it diverged.
    """
    backed_off = increased or not math.isfinite(drop)
    if backed_off:
        drop = 1.0
    smoothed = (
        drop
        if smoothed is None
        else ETA_DROP_EMA * smoothed + (1.0 - ETA_DROP_EMA) * drop
    )
    low, high = ETA_STEP_CLIP
    factor = min(max((target / max(smoothed, 1e-6)) ** gain, low), high)
    if backed_off:
        factor = min(factor, backoff)
    return min(max(eta * factor, eta_min), eta_max), smoothed, backed_off


def train_stage(
    params: GraphParams,
    structure: GraphStructure,
    stream: np.ndarray,
    *,
    config: RunConfig,
    stage: int,
    special_ids: Mapping[str, int],
    gradient_fn: GradientFn,
    multipliers: Mapping[str, float] | None = None,
    regime_fn: Callable[[int, int, GraphParams, float | None], None] | None = None,
    verbose: bool = True,
) -> tuple[GraphParams, dict[str, Any], list[dict[str, Any]]]:
    """Train one domain from ``params`` with a fresh optimizer and schedule.

    Runs ``num_stage_updates`` updates of ``batch_size`` sequences, each the
    masked-token-weighted mean of ``batch_size / micro_batch_size``
    micro-batch gradients (``accumulate_gradients``); the batcher reshuffles
    and restarts when an epoch ends. Batches, MLM masks
    and dropout keys depend only on ``seed`` and ``stage``, so both modes see
    the same ones. ``multipliers`` scales each module's update
    (``build_stage_optimizer``). Each log entry holds ``grad_norm``, the
    global norm of the update's gradient as the clip sees it (frozen modules
    zeroed), and ``clipped``, whether the clip rescaled it. ``regime_fn(update,
    num_updates, params, eta)`` is called before the first update and after
    every update with the ePC rate the next update uses, or after the last
    update the rate it used (None without the rate controller); it must
    only read ``params``. ``pc_energy_rel_decrease`` is NaN (null in the log)
    when either energy is non-finite or the start energy is not positive.
    With ``pc_eta_control="energy"`` the rate starts at ``pc_eta_infer``,
    each log entry adds ``eta_infer`` (the rate it used), ``eta_next``,
    ``eta_backoff`` and ``pc_energy_rel_decrease_ema`` (``control_eta``), and
    the exposure record adds the stage's rate range, mean and last used
    value under ``eta_infer``.
    Returns the params, the exposure record and the log.
    """
    seq_len = int(structure.nodes[structure.task_map["y"]].node_info.shape[0])
    vocab_size = int(structure.nodes[structure.task_map["y"]].node_info.shape[-1])
    micro = config.micro_batch_size or config.batch_size
    accumulation = config.batch_size // micro
    num_updates = num_stage_updates(config.stage_tokens, config.batch_size, seq_len)
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
    optimizer, optimizer_info = build_stage_optimizer(config, num_updates, multipliers)
    opt_state = optimizer.init(params)
    update_fn = jax.jit(
        lambda p, o, g, c: _apply_optimizer_update_when_masked(
            params=p, opt_state=o, grads=g, optimizer=optimizer, masked_count=c
        )
    )
    frozen = {name for name, value in (multipliers or {}).items() if value == 0.0}
    clip_norm_fn = jax.jit(
        lambda g: optax.tree.norm(
            {name: node for name, node in g.nodes.items() if name not in frozen}
        )
    )
    controlled = config.pc_eta_control == "energy"
    eta = config.pc_eta_infer
    eta_min, eta_max = config.eta_bounds
    smoothed: float | None = None
    etas_used: list[float] = []
    if regime_fn is not None:
        regime_fn(0, num_updates, params, eta if controlled else None)
    rng_key = jax.random.fold_in(jax.random.PRNGKey(config.seed), stage)
    counts = {
        "stream_tokens": 0,
        "non_padding_tokens": 0,
        "masked_tokens": 0,
        "token_slots": 0,
        "micro_batches": 0,
    }
    log: list[dict[str, Any]] = []
    energy_warned = False
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
                accumulate_gradients(
                    gradient_fn,
                    params,
                    pending_batches,
                    pending_keys,
                    summed=config.mode == "pc",
                    eta_infer=jnp.asarray(eta) if controlled else None,
                )
            )
            grad_norm = float(clip_norm_fn(pending_grads))
            params, opt_state = update_fn(
                params, opt_state, pending_grads, jnp.asarray(pending_count)
            )
            updates += 1
            entry: dict[str, Any] = {
                "stage": stage,
                "update": updates,
                "masked_loss": pending_loss / max(pending_count, 1.0),
                "masked_tokens": int(pending_count),
                "grad_norm": grad_norm,
                "clipped": bool(
                    config.clip_norm > 0
                    and pending_count > 0
                    and grad_norm > config.clip_norm
                ),
            }
            if config.mode == "pc":
                entry["pc_total_energy"] = float(
                    np.mean([float(e["pc_total_energy"]) for e in pending_extras])
                )
                entry["pc_energy_rises"] = int(
                    sum(int(e["pc_energy_rises"]) for e in pending_extras)
                )
                if "pc_energy_total_rises" in pending_extras[0]:
                    entry["pc_energy_total_rises"] = int(
                        sum(int(e["pc_energy_total_rises"]) for e in pending_extras)
                    )
                if config.pc_inference in ITERATIVE_PC:
                    start = float(
                        np.mean([float(e["pc_energy_start"]) for e in pending_extras])
                    )
                    end = float(
                        np.mean([float(e["pc_energy_end"]) for e in pending_extras])
                    )
                    entry["pc_energy_start"] = start
                    entry["pc_energy_end"] = end
                    entry["pc_energy_rel_decrease"] = (
                        (start - end) / start
                        if math.isfinite(start) and math.isfinite(end) and start > 0
                        else math.nan
                    )
                    entry["pc_energy_increased"] = end > start
                    if end > start and not energy_warned:
                        energy_warned = True
                        print(
                            f"WARNING stage={stage} update={updates}: "
                            f"{config.pc_inference} PC inference raised the energy "
                            f"({start:.4g} -> {end:.4g}); inference is diverging, "
                            "lower --pc_eta_infer"
                        )
            if controlled:
                entry["eta_infer"] = eta
                etas_used.append(eta)
                eta, smoothed, backed_off = control_eta(
                    eta,
                    smoothed,
                    entry["pc_energy_rel_decrease"],
                    entry["pc_energy_increased"],
                    target=config.pc_energy_target,
                    gain=config.pc_eta_gain,
                    eta_min=eta_min,
                    eta_max=eta_max,
                    backoff=config.pc_eta_backoff,
                )
                entry["eta_next"] = eta
                entry["eta_backoff"] = backed_off
                entry["pc_energy_rel_decrease_ema"] = smoothed
            log.append(entry)
            if regime_fn is not None:
                regime_fn(
                    updates,
                    num_updates,
                    params,
                    (
                        (etas_used[-1] if updates == num_updates else eta)
                        if controlled
                        else None
                    ),
                )
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
        "stage_tokens": config.stage_tokens,
        "max_stream_tokens_per_update": tokens_per_update,
        **counts,
        "exposure_minus_budget": counts["stream_tokens"] - config.stage_tokens,
        "within_one_update": abs(counts["stream_tokens"] - config.stage_tokens)
        <= tokens_per_update,
        "epochs_started": int(batcher.state_dict()["epoch"]),
        "optimizer": optimizer_info,
    }
    if controlled:
        exposure["eta_infer"] = {
            "min": min(etas_used),
            "max": max(etas_used),
            "mean": float(np.mean(etas_used)),
            "last": etas_used[-1],
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


def _gate_record_matches(
    records: Sequence[Mapping[str, Any]],
    *,
    stage: int,
    gate: str,
    parent_sha: str | None,
    exposure: Mapping[str, Any] | None,
) -> bool:
    """True when ``stage`` has one gate record that matches what it trained with.

    The record must start from ``parent_sha`` and its multipliers must be the
    ``module_multipliers`` of the stage's optimizer (absent when ungated).
    """
    found = [r for r in records if int(r["stage"]) == stage]
    if len(found) != 1 or exposure is None:
        return False
    record = found[0]
    if record.get("gate") != gate or record.get("params_sha256") != parent_sha:
        return False
    used = (exposure.get("optimizer") or {}).get("module_multipliers")
    return used == (record["multipliers"] if record.get("gated") else None)


def _matrix(
    records: Mapping[tuple[int, str], Mapping[str, Any]],
    domains: Sequence[str],
    metric: str,
    num_stages: int,
) -> list[list[float]]:
    return [
        [float(records[(stage, name)][metric]) for name in domains]
        for stage in range(num_stages)
    ]


def summarize_joint(
    domains: Sequence[str], ce: Any, ppl: Any, acc: Any
) -> dict[str, Any]:
    """Per-domain scores of a joint run: S0, the jointly trained S1, and the gain."""
    ce_array, ppl_array, acc_array = (
        np.asarray(m, dtype=np.float64) for m in (ce, ppl, acc)
    )
    if ce_array.shape != (2, len(domains)):
        raise ValueError("a joint run has stages S0 and S1 for every domain")
    return {
        "domains": list(domains),
        "initial_ce": dict(zip(domains, ce_array[0].tolist())),
        "final_ce": dict(zip(domains, ce_array[1].tolist())),
        "final_perplexity": dict(zip(domains, ppl_array[1].tolist())),
        "final_accuracy": dict(zip(domains, acc_array[1].tolist())),
        "learning_gain_ce": dict(zip(domains, (ce_array[0] - ce_array[1]).tolist())),
        "sign_conventions": {
            "learning_gain_ce": "CE[S0, d] - CE[S1, d]; positive = learned",
        },
    }


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
    ``perturb_params``. With ``pc_alignment_batches`` N > 0, every trained
    stage appends the PC vs backprop ``gradient_alignment`` of its start and
    end params on N fixed batches of its domain to ``alignment.jsonl``.
    With ``training="joint"`` there is one trained stage, named ``joint``, on
    all domains mixed; ``forgetting_matrices.json`` is not written and the
    summary holds ``summarize_joint``'s per-domain scores. With a ``gate``,
    every trained stage appends its module multipliers (``GatePlan.record``)
    to ``gates.jsonl`` and the run config holds ``GatePlan.describe()``.
    With ``pc_regime_interval`` k > 0, every trained stage appends to
    ``regime.jsonl`` the ε-spectrum (``make_regime_probe``), the
    ``EPCInference(pc_regime_eta, pc_regime_steps).regime`` verdict and the
    per-module weight norms at its start, every k updates and its end, on
    one fixed dropout-free micro-batch of its domain (seed ``seed + stage``).
    With ``pc_regime_final`` each record also holds the spectrum after
    ``pc_regime_steps`` ePC steps (``regime_summary``). The probe only reads
    the params, so the run's results do not depend on it. The regime
    settings are resume-ignored: reused stages keep their records and
    retrained stages are probed with the current settings, so a resumed
    ``regime.jsonl`` can mix settings across stages; every record carries
    its own (interval, final, eta, steps, Lanczos iterations). With
    ``pc_eta_control="energy"`` and no explicit ``pc_regime_eta``, each
    record judges the controller's current rate and marks
    ``eta_controlled``; ``run_config.json`` holds the controller under
    ``pc.eta_control``.
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
    gate_plan = None
    if config.gate != "none":
        from fabricpc.experiments.domain_forgetting.gates import GatePlan

        gate_plan = GatePlan(
            config, base, {name: cache.train for name, cache in caches.items()}
        )

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
                "head_energy": HEAD_ENERGY,
                **(
                    {
                        "eta_control": {
                            "rule": config.pc_eta_control,
                            "energy_target": config.pc_energy_target,
                            "gain": config.pc_eta_gain,
                            "eta_min": config.eta_bounds[0],
                            "eta_max": config.eta_bounds[1],
                            "backoff": config.pc_eta_backoff,
                            "drop_ema": ETA_DROP_EMA,
                            "step_clip": list(ETA_STEP_CLIP),
                            "stage_reset": True,
                        }
                    }
                    if config.pc_eta_control != "none"
                    else {}
                ),
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
            config.stage_tokens, config.batch_size, seq_len
        ),
    }
    if gate_plan is not None:
        run_config["gate"] = gate_plan.describe()
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
    previous_alignment = _read_jsonl(output / "alignment.jsonl") if resume else []
    previous_gates = _read_jsonl(output / "gates.jsonl") if resume else []
    previous_regime = _read_jsonl(output / "regime.jsonl") if resume else []
    regime_records: list[dict[str, Any]] = []
    regime_probe = None
    if config.pc_regime_interval:
        regime_structure = build_structure(
            base.model_config,
            replace(config, dropout_rate=0.0),
            base.structure.config["inference"],
        )
        regime_eta, regime_steps = config.regime_setting
        regime_probe = make_regime_probe(
            regime_structure,
            config.pc_regime_iters,
            (regime_eta, regime_steps) if config.pc_regime_final else None,
        )

    def stage_regime_fn(stage: int, domain: str, stream: np.ndarray):
        """Return the stage's ``train_stage`` regime callback and its timer."""
        batch_size = config.micro_batch_size or config.batch_size
        batches, keys = fixed_mlm_batches(
            stream,
            regime_structure,
            base.special_ids,
            num_batches=1,
            batch_size=batch_size,
            mask_prob=config.mask_prob,
            seed=config.seed + stage,
        )
        seconds = [0.0]

        def record(
            update: int,
            num_updates: int,
            stage_params: GraphParams,
            eta_infer: float | None = None,
        ) -> None:
            interval = config.pc_regime_interval
            if update not in (0, num_updates) and update % interval:
                return
            started = time.time()
            controlled = eta_infer is not None and config.pc_regime_eta is None
            eta = eta_infer if controlled else regime_eta
            spectrum, final_spectrum = regime_probe(
                stage_params,
                batches[0],
                keys[0],
                jnp.asarray(eta) if controlled else None,
            )
            when = (
                "start"
                if update == 0
                else "end" if update == num_updates else "interval"
            )
            regime_records.append(
                {
                    "stage": stage,
                    "domain": domain,
                    "update": update,
                    "num_updates": num_updates,
                    "when": when,
                    "params_sha256": params_hash(stage_params),
                    "regime_interval": config.pc_regime_interval,
                    "regime_final": config.pc_regime_final,
                    "eta_infer": eta,
                    **({"eta_controlled": True} if controlled else {}),
                    "infer_steps": regime_steps,
                    "lanczos_iters": config.pc_regime_iters,
                    "batch_size": batch_size,
                    "seed": config.seed + stage,
                    **regime_summary(spectrum, eta, regime_steps, final_spectrum),
                    "weight_norms": module_weight_norms(stage_params, base.structure),
                }
            )
            seconds[0] += time.time() - started
            if verbose:
                last = regime_records[-1]
                print(
                    f"stage={stage} update={update} regime "
                    f"lambda_max={last['lambda_max']:.4g} "
                    f"lambda_min={last['lambda_min']:.4g}: {last['regime']}"
                )

        return record, seconds

    gate_records: list[dict[str, Any]] = []

    gradient_fn = make_gradient_fn(
        config.mode, base.structure, config.pc_forward_updates, config.pc_inference
    )
    alignment_fn = (
        make_alignment_fn(base.structure, config.pc_forward_updates)
        if config.pc_alignment_batches
        else None
    )
    alignment_records: list[dict[str, Any]] = []

    def record_alignment(
        stage: int,
        domain: str,
        when: str,
        stage_params: GraphParams,
        eta_infer: float | None = None,
    ) -> float:
        """Append one alignment record and return the seconds it took.

        ``eta_infer`` is the controller's rate to compare at (None: the
        configured ``pc_eta_infer``); the record holds the rate used.
        """
        started = time.time()
        batches, keys = fixed_mlm_batches(
            caches[domain].train,
            base.structure,
            base.special_ids,
            num_batches=config.pc_alignment_batches,
            batch_size=config.micro_batch_size or config.batch_size,
            mask_prob=config.mask_prob,
            seed=config.seed + stage,
        )
        alignment_records.append(
            {
                "stage": stage,
                "domain": domain,
                "when": when,
                "params_sha256": params_hash(stage_params),
                "pc_inference": config.pc_inference,
                "output_energy_scale": config.pc_output_energy_scale,
                "head_energy": HEAD_ENERGY,
                "batch_size": config.micro_batch_size or config.batch_size,
                "seed": config.seed + stage,
                "eta_infer": config.pc_eta_infer if eta_infer is None else eta_infer,
                **({"eta_controlled": True} if eta_infer is not None else {}),
                **gradient_alignment(
                    alignment_fn,
                    stage_params,
                    batches,
                    keys,
                    base.structure,
                    None if eta_infer is None else jnp.asarray(eta_infer),
                ),
            }
        )
        if verbose:
            last = alignment_records[-1]
            print(
                f"stage={stage} alignment {when} "
                f"mean_cosine={last['mean_cosine']:.4f} "
                f"min_cosine={last['min_cosine']:.4f}"
            )
        return time.time() - started

    params = initial_params
    evaluator = MultiDomainEvaluator(base.structure, valid_sets, config.eval_batch_size)
    stages: list[dict[str, Any]] = []
    evaluations: dict[tuple[int, str], dict[str, Any]] = {}
    exposure: dict[str, Any] = {}
    stage_runtime: dict[str, Any] = {}
    reusing = resume
    parent_sha: str | None = None
    stage_domains = list(config.stage_domains)
    num_stages = len(stage_domains) + 1
    for stage in range(num_stages):
        domain = None if stage == 0 else stage_domains[stage - 1]
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
            if reused is not None and gate_plan is not None and domain is not None:
                if not _gate_record_matches(
                    previous_gates,
                    stage=stage,
                    gate=config.gate,
                    parent_sha=parent_sha,
                    exposure=previous_exposure.get(domain),
                ):
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
            alignment_records.extend(
                r for r in previous_alignment if int(r["stage"]) == stage
            )
            gate_records.extend(r for r in previous_gates if int(r["stage"]) == stage)
            regime_records.extend(
                r for r in previous_regime if int(r["stage"]) == stage
            )
            if verbose:
                print(f"stage={stage} reused params_sha256={parent_sha[:12]}")
            continue

        initial_sha = params_hash(params)
        train_seconds = 0.0
        alignment_seconds = 0.0
        multipliers = None
        if domain is not None:
            if gate_plan is not None:
                gate_record = gate_plan.record(
                    stage, domain, params, caches[domain].train
                )
                gate_records.append(gate_record)
                _write_jsonl(output / "gates.jsonl", gate_records)
                if gate_record["gated"]:
                    multipliers = gate_record["multipliers"]
                    if verbose:
                        print(
                            f"stage={stage} gate={config.gate} "
                            + " ".join(f"{m}={v:.3g}" for m, v in multipliers.items())
                        )
            if alignment_fn is not None:
                alignment_seconds = record_alignment(stage, domain, "start", params)
            stream = (
                np.concatenate([caches[name].train for name in domains])
                if domain == JOINT_STAGE
                else caches[domain].train
            )
            regime_fn, regime_seconds = (
                stage_regime_fn(stage, domain, stream)
                if regime_probe is not None
                else (None, [0.0])
            )
            train_start = time.time()
            params, stage_exposure, log = train_stage(
                params,
                base.structure,
                stream,
                config=config,
                stage=stage,
                special_ids=base.special_ids,
                gradient_fn=gradient_fn,
                multipliers=multipliers,
                regime_fn=regime_fn,
                verbose=verbose,
            )
            train_seconds = time.time() - train_start - regime_seconds[0]
            if regime_probe is not None:
                _write_jsonl(output / "regime.jsonl", regime_records)
            exposure[domain] = stage_exposure
            _write_jsonl(output / f"training_log_stage_{stage}.jsonl", log)
            if alignment_fn is not None:
                alignment_seconds += record_alignment(
                    stage,
                    domain,
                    "end",
                    params,
                    stage_exposure.get("eta_infer", {}).get("last"),
                )
                _write_jsonl(output / "alignment.jsonl", alignment_records)
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
        if alignment_fn is not None:
            stage_runtime[str(stage)]["alignment_seconds"] = alignment_seconds
        if regime_probe is not None and domain is not None:
            stage_runtime[str(stage)]["regime_seconds"] = regime_seconds[0]
        if gate_plan is not None and domain is not None:
            stage_runtime[str(stage)]["gate_seconds"] = gate_records[-1]["seconds"]
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

    matrices = {
        metric: _matrix(evaluations, domains, metric, num_stages) for metric in METRICS
    }
    for metric, filename in MATRIX_FILES.items():
        atomic_write_json(
            output / filename,
            {
                "metric": metric,
                "domains": domains,
                "stages": [f"S{s}" for s in range(num_stages)],
                "rows": "stage",
                "columns": "domain in training order",
                "matrix": matrices[metric],
            },
        )
    ce, ppl, acc = (matrices[m] for m in METRICS)
    if config.training == "joint":
        scores = summarize_joint(domains, ce, ppl, acc)
    else:
        atomic_write_json(
            output / "forgetting_matrices.json",
            {"domains": domains, **forgetting_matrices(ce, ppl, acc)},
        )
        scores = summarize_matrices(domains, ce, ppl, acc)
    summary = {
        **scores,
        "training": config.training,
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

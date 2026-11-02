"""Local prediction-error maps of stage snapshots.

PC inference is run on a snapshot with the inputs, attention mask and loss
mask clamped, dropout off and the output clamped to the one-hot labels of
every position (only masked positions carry energy). For each hidden node
``m`` and window ``b`` two per-sample energies are recorded:

- ``final_energy`` = ``½‖z_m − μ_m‖²`` at the end of inference, with ``μ_m``
  recomputed from the final latents (the graph's Gaussian node energy);
- ``forward_error_energy`` = ``½‖z_m(T) − μ_m(0)‖²``, the error against the
  feedforward prediction that the forward-update weight rule uses.

Under spike-only inference (``eta_infer`` 0, one unit step per node at its
longest depth, T = the longest free depth) ``z_m(T) − μ_m(0) = −s ∂C/∂z_m``
exactly, with ``C`` the batch's summed masked CE and ``s`` the output-energy
scale, so the raw ``forward_error_energy`` is BP's backward signal
``½ s²‖∂C/∂z_m‖²``. ``final_energy`` differs from it at nodes whose inputs
spiked after they did. Under standard PC the map is ``final_energy``.

**Convention.** ``error_maps.json`` records ``head_energy`` =
``"sum_masked_ce"``. Files without it predate this head energy, which was
``s · Σ_masked CE_b / N``. The spike-only ``energy``, ``energy_sum_ce`` and
``forward_error_energy`` (and ``*_sum_ce``) values are the same numbers
under both conventions; the spike-only ``final_energy`` and
``output_ce_energy``, every ``standard_pc`` field, and ``energy_trace``
(raw, about ``N`` times larger) are not comparable across them.

**Normalisation.** The MLM head energy of window ``b`` is
``s · Σ_masked CE_b`` (a fixed per-token precision). Maps report each raw
energy divided by ``N²``, with ``N`` the masked-token count of the whole
batch, which in the spike-only setting is ``½ s²‖∂L/∂z_m‖²`` for ``L`` the
batch's mean masked CE, and divided by ``s²`` (``*_sum_ce``), which in the
spike-only setting is ``½‖∂(Σ_masked CE_b)/∂z_m‖²`` and does not depend on
the other windows of the batch. The head module's value is
``½ s²‖(t − p) ⊙ mask‖² / N²``, BP's signal ``½‖s ∂L/∂logits‖²`` at its
logits; the head's own CE energy divided by ``N`` is reported as
``output_ce_energy``.
Maps are means over windows with at least one masked token. Each module
also gets ``energy_sum_ce_per_dim``, ``energy_sum_ce`` over the node's
output width (e.g. 1,024 for ``mlp1`` and 256 for ``mlp2`` in BERT-Mini).

Under standard PC every map records ``energy_trace``: the raw total energy
(summed over nodes with inputs, mean over the map's windows) at each
inference step ``0..T``, the convergence check of the iterative solve.

**Learner-native maps.** The ``learner`` setting runs the run's own learner
(``learner_map_structure``): ePC with its ``pc_eta_infer`` and
``pc_infer_steps`` (the energy controller's starting rate η₀, at which every
stage restarts), standard PC with its own inference, spike-only PC with its
own schedule, and backprop runs with the spike-only schedule, whose errors
are BP's signal. Its primary energy is the error of the run's weight rule
(``learner_energy_field``): ``forward_error_energy`` with forward updates
(spike-only PC, backprop), ``final_energy`` otherwise (ePC, where it is
``½‖ε_m‖²`` at the relaxed errors, and standard PC). Module grouping and
normalisation are the other settings'. One ePC step from ε = 0 gives
``ε_m = −η s ∂C/∂z_m``, so its ``energy_sum_ce`` is ``η²`` times the
spike-only map; after ``T`` steps at small ``ηλ_max`` it approaches
``(ηT)²`` times it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.inference import (
    InferenceBase,
    InferenceSGDNormClip,
    InferenceSpikingPrecision,
    gather_inputs,
)
from fabricpc.core.scaling import scale_inputs
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.experiments.domain_forgetting.localization import (
    RunContext,
    module_names,
    rollback_transitions,
    run_provenance,
    stage_name,
)
from fabricpc.experiments.domain_forgetting.runner import (
    GATE_MAP_SOURCES,
    HEAD_ENERGY,
    RunConfig,
    build_inference,
    build_structure,
    max_free_depth,
)
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training.train_autoregressive import compute_loss, create_dropout_clamps

SPIKE_ONLY = "spike_only"
STANDARD_PC = "standard_pc"
LEARNER = "learner"
SETTINGS = (SPIKE_ONLY, STANDARD_PC)
MAP_SETTINGS = (*SETTINGS, LEARNER)
MAP_SOURCES = GATE_MAP_SOURCES
PRIMARY_ENERGY = {SPIKE_ONLY: "forward_error_energy", STANDARD_PC: "final_energy"}
ENERGY_FIELDS = ("final_energy", "forward_error_energy")


@dataclass(frozen=True)
class ErrorMapConfig:
    """Error-map settings.

    ``pc_eta_infer`` has no default and is required with the ``standard_pc``
    setting: under the head energy ``s · Σ_masked CE`` the latent gradients
    grow with the masked tokens per batch, so the rate must be calibrated
    for ``batch_size`` (e.g. with the runner's ``probe``).
    """

    num_windows: int = 256
    batch_size: int = 8
    pc_infer_steps: int = 120
    pc_eta_infer: float | None = None
    pc_max_norm: float = 5.0
    output_energy_scale: float = 1.0
    settings: tuple[str, ...] = SETTINGS

    def validate(self) -> None:
        if self.num_windows < 1 or self.batch_size < 1:
            raise ValueError("num_windows and batch_size must be positive")
        if self.pc_infer_steps < 1:
            raise ValueError("pc_infer_steps must be positive")
        if STANDARD_PC in self.settings and not (
            self.pc_eta_infer is not None and self.pc_eta_infer > 0
        ):
            raise ValueError(
                "the standard_pc setting needs an explicit pc_eta_infer > 0, "
                "calibrated for the map batch size"
            )
        if self.output_energy_scale <= 0:
            raise ValueError("output_energy_scale must be positive")
        unknown = set(self.settings) - set(MAP_SETTINGS)
        if unknown or not self.settings:
            raise ValueError(f"settings must be a non-empty subset of {MAP_SETTINGS}")


def spike_only_inference(infer_steps: int, max_norm: float = 5.0) -> Any:
    """The spike-only schedule: rate 0 off-spike, a unit step at longest depth."""
    return InferenceSpikingPrecision(
        eta_infer=0.0,
        infer_steps=infer_steps,
        max_norm=max_norm,
        spike_eta=1.0,
        depth_mode="longest",
    )


def build_map_structure(
    context: RunContext, setting: str, config: ErrorMapConfig
) -> GraphStructure:
    """Build the run's graph without dropout and with the setting's inference."""
    run_config = replace(
        context.config,
        mode="pc",
        dropout_rate=0.0,
        pc_output_energy_scale=config.output_energy_scale,
    )
    if setting == SPIKE_ONLY:
        probe = build_structure(
            context.model_config, run_config, spike_only_inference(1)
        )
        inference = spike_only_inference(
            max_free_depth(probe, "longest"), config.pc_max_norm
        )
    elif setting == STANDARD_PC:
        inference = InferenceSGDNormClip(
            eta_infer=config.pc_eta_infer,
            infer_steps=config.pc_infer_steps,
            max_norm=config.pc_max_norm,
        )
    else:
        raise ValueError(f"setting must be one of {SETTINGS}, got {setting!r}")
    return build_structure(context.model_config, run_config, inference)


def learner_energy_field(config: RunConfig) -> str:
    """The ``node_energies`` field of the error in the run's weight rule.

    ``forward_error_energy`` (``z_T − μ_0``) for forward updates and for
    backprop runs (spike-only errors), ``final_energy`` (``z_T − μ_T``) for
    ``pc_forward_updates="none"``.
    """
    if config.mode == "pc" and config.pc_forward_updates == "none":
        return "final_energy"
    return "forward_error_energy"


def learner_map_structure(
    model_config: Mapping[str, Any], config: RunConfig
) -> GraphStructure:
    """Build the run's graph without dropout and with its learner's inference.

    PC runs keep their inference (``build_inference``; ePC at its starting
    rate ``pc_eta_infer``) and output-energy scale; backprop runs get the
    spike-only schedule at scale 1.
    """
    if config.mode == "pc":
        run = replace(config, dropout_rate=0.0)
    else:
        run = replace(
            config,
            mode="pc",
            pc_inference="spiking",
            pc_eta_infer=0.0,
            pc_spike_eta=1.0,
            pc_spike_depth="longest",
            pc_forward_updates="initial",
            pc_infer_steps=None,
            pc_output_energy_scale=1.0,
            dropout_rate=0.0,
        )
    inference, _ = build_inference(run, model_config)
    return build_structure(model_config, run, inference)


def output_energy_scale(structure: GraphStructure) -> float:
    """The output node's energy scale ``s`` (1 when unset)."""
    output = structure.task_map["y"]
    config = structure.nodes[output].node_info.node_config
    return float(config.get("output_energy_scale", 1.0))


def hidden_nodes(structure: GraphStructure) -> list[str]:
    """Return the nodes with inputs that are not the output, in graph order."""
    output = structure.task_map["y"]
    return [
        name
        for name in structure.node_order
        if structure.nodes[name].node_info.in_degree > 0 and name != output
    ]


def _clamps(
    structure: GraphStructure, batch: Mapping[str, jax.Array]
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Return the input clamps and the input clamps plus one-hot targets."""
    batch_size = batch["x"].shape[0]
    output = structure.task_map["y"]
    vocab_size = structure.nodes[output].node_info.shape[-1]
    float_dtype = jnp.result_type(float)
    inputs = {
        structure.task_map["x"]: batch["x"],
        structure.task_map["loss_mask"]: batch["loss_mask"].astype(float_dtype),
    }
    if "attention_mask" in structure.task_map:
        inputs[structure.task_map["attention_mask"]] = batch["attention_mask"].astype(
            float_dtype
        )
    inputs.update(
        create_dropout_clamps(
            structure, batch_size, jax.random.PRNGKey(0), training=False
        )
    )
    targets = jax.nn.one_hot(batch["y"], vocab_size, dtype=float_dtype)
    return inputs, {**inputs, output: targets}


def _half_square(values: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum(values**2, axis=tuple(range(1, values.ndim)))


def _total_energy(state: GraphState, structure: GraphStructure) -> jax.Array:
    """Batch-summed energy of every node with inputs."""
    return sum(
        jnp.sum(state.nodes[name].energy)
        for name, node in structure.nodes.items()
        if node.node_info.in_degree > 0
    )


def _traced_inference(
    params: GraphParams,
    initial: GraphState,
    clamps: Mapping[str, jax.Array],
    structure: GraphStructure,
) -> tuple[GraphState, jax.Array]:
    """Run the base inference loop step by step; return the energy per step.

    Each step's energy is that of the state entering it, measured by the
    step's own forward phase, so the trace covers states ``0..T-1``.
    """
    inference = structure.config["inference"]
    cls = type(inference)
    config = inference.config

    def body(state, _):
        state = cls.zero_grads(params, state, clamps, structure)
        state = cls.forward_value_and_grad(params, state, clamps, structure)
        energy = _total_energy(state, structure)
        state = cls.update_latents(params, state, clamps, structure, config)
        return state, energy

    return jax.lax.scan(body, initial, None, length=int(config["infer_steps"]))


def node_energies(
    params: GraphParams,
    structure: GraphStructure,
    batch: Mapping[str, jax.Array],
    *,
    trace: bool = False,
) -> dict[str, Any]:
    """Run target-clamped PC inference on one batch; return per-sample energies.

    Returns raw energies under the head energy ``s · Σ_masked CE``:
    ``{"final_energy": {node: (B,)}, "forward_error_energy": {node: (B,)},
    "head_energy": {field: (B,)}, "output_ce_energy": (B,),
    "masked_tokens": (B,), "batch_masked_tokens": ()}``. ``head_energy`` is
    the head's ``½ s²‖(t − p) ⊙ mask‖²`` at the final state
    (``final_energy``) and at the feedforward state
    (``forward_error_energy``). With ``trace`` (base-class inference loops
    only, e.g. standard PC) it also returns ``energy_trace`` ``(T + 1,)``,
    the batch-summed raw total energy entering each step and after the last.
    ``aggregate_map`` applies the map normalisation.
    """
    _inputs, clamps = _clamps(structure, batch)
    batch_size = batch["x"].shape[0]
    output = structure.task_map["y"]
    inference_cls = type(structure.config["inference"])
    initial = initialize_graph_state(
        structure, batch_size, jax.random.PRNGKey(0), clamps=clamps, params=params
    )
    if trace:
        if inference_cls.run_inference is not InferenceBase.run_inference:
            raise ValueError("energy traces need the base inference loop")
        final, energies = _traced_inference(params, initial, clamps, structure)
    else:
        final = inference_cls.run_inference(params, initial, clamps, structure)
    refreshed = InferenceBase.forward_value_and_grad(
        params,
        InferenceBase.zero_grads(params, final, clamps, structure),
        clamps,
        structure,
    )
    nodes = hidden_nodes(structure)
    mask = batch["loss_mask"].astype(jnp.result_type(float))
    scale = output_energy_scale(structure)

    def head(error: jax.Array) -> jax.Array:
        return scale**2 * _half_square(error)

    extra = {}
    if trace:
        extra["energy_trace"] = jnp.concatenate(
            [energies, _total_energy(refreshed, structure)[None]]
        )
    return {
        **extra,
        "final_energy": {n: refreshed.nodes[n].energy for n in nodes},
        "forward_error_energy": {
            n: _half_square(final.nodes[n].z_latent - initial.nodes[n].z_mu)
            for n in nodes
        },
        "head_energy": {
            "final_energy": head(refreshed.nodes[output].error),
            "forward_error_energy": head(initial.nodes[output].error),
        },
        "output_ce_energy": refreshed.nodes[output].energy,
        "masked_tokens": jnp.sum(mask, axis=-1),
        "batch_masked_tokens": jnp.sum(mask),
    }


def _perturbed_state(
    perturbations: Mapping[str, jax.Array],
    params: GraphParams,
    structure: GraphStructure,
    initial: GraphState,
    clamps: Mapping[str, jax.Array],
) -> GraphState:
    """Feedforward pass with ``perturbations[m]`` added to each free node output."""
    state = initial
    for name in structure.node_order:
        info = structure.nodes[name].node_info
        if info.in_degree == 0:
            continue
        inputs = scale_inputs(
            gather_inputs(info, structure, state), info.scaling_config
        )
        _, projected = info.node_class.forward(
            params.nodes[name], inputs, state.nodes[name], info
        )
        if name in clamps:
            state = update_node_in_state(state, name, z_mu=projected.z_mu)
            continue
        latent = projected.z_mu
        if name in perturbations:
            latent = latent + perturbations[name]
        state = update_node_in_state(state, name, z_mu=projected.z_mu, z_latent=latent)
    return state


def bp_signal_energies(
    params: GraphParams,
    structure: GraphStructure,
    batch: Mapping[str, jax.Array],
    output_energy_scale: float = 1.0,
) -> dict[str, jax.Array]:
    """Return ``½ s²‖∂L/∂z_m‖²`` per sample for every hidden node ``m``.

    ``L`` is the backprop MLM loss (``compute_loss``: mean masked CE over the
    batch) of a dropout-free feedforward pass, differentiated through
    additive perturbations of each node's output at zero.
    """
    inputs, _ = _clamps(structure, batch)
    batch_size = batch["x"].shape[0]
    output = structure.task_map["y"]
    initial = initialize_graph_state(
        structure, batch_size, jax.random.PRNGKey(0), clamps=inputs, params=params
    )
    nodes = hidden_nodes(structure)
    zeros = {n: jnp.zeros_like(initial.nodes[n].z_latent) for n in nodes}

    def loss(perturbations):
        state = _perturbed_state(perturbations, params, structure, initial, inputs)
        return compute_loss(
            state,
            batch["y"],
            output,
            loss_type="cross_entropy",
            loss_mask=batch["loss_mask"],
        )

    grads = jax.grad(loss)(zeros)
    return {n: output_energy_scale**2 * _half_square(grads[n]) for n in nodes}


def learner_signal_energies(
    params: GraphParams,
    structure: GraphStructure,
    batch: Mapping[str, jax.Array],
    field: str,
) -> dict[str, jax.Array]:
    """Return the learner's ``½‖e_m‖² / (s² N²)`` per sample and hidden node.

    ``e_m`` is the ``field`` error of ``node_energies`` after the structure's
    inference, ``s`` the output-energy scale and ``N`` the batch's masked
    tokens, so the units are ``bp_signal_energies``' at output scale 1, which
    it equals under the spike-only schedule.
    """
    result = node_energies(params, structure, batch)
    scale = output_energy_scale(structure)
    count = jnp.maximum(result["batch_masked_tokens"], 1.0)
    return {n: v / (scale**2 * count**2) for n, v in result[field].items()}


def map_specs(domains: Sequence[str]) -> list[dict[str, Any]]:
    """Return the (kind, domain, stage) maps to compute.

    Onset: domain ``s + 1`` at ``S_s`` (the new domain before training on
    it). Retrospective: domain ``r`` at ``S_t`` for every rollback transition
    ``S_t <- S_r`` (for three domains A at S2 and S3, and B at S3).
    """
    specs = [
        {"kind": "onset", "domain": domains[stage], "stage": stage}
        for stage in range(1, len(domains))
    ]
    specs.extend(
        {"kind": "retrospective", "domain": domains[reference - 1], "stage": target}
        for target, reference in sorted(
            rollback_transitions(len(domains)), key=lambda pair: (pair[1], pair[0])
        )
    )
    return specs


def window_batches(
    arrays: Mapping[str, np.ndarray], num_windows: int, batch_size: int
) -> list[dict[str, jax.Array]]:
    """Return the first ``num_windows`` fixed windows as model batches."""
    count = min(num_windows, len(arrays["input_ids"]))
    return [
        {
            "x": jnp.asarray(arrays["masked_input_ids"][start : start + batch_size]),
            "y": jnp.asarray(arrays["labels"][start : start + batch_size]),
            "loss_mask": jnp.asarray(arrays["loss_mask"][start : start + batch_size]),
            "attention_mask": jnp.asarray(
                arrays["attention_mask"][start : start + batch_size]
            ),
        }
        for start in range(0, count, batch_size)
    ]


def output_dims(structure: GraphStructure, modules: Sequence[str]) -> dict[str, int]:
    """Return each module node's output width (last axis of its shape)."""
    return {m: int(structure.nodes[m].node_info.shape[-1]) for m in modules}


def energy_trace_summary(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean total energy per window at every inference step over ``batches``."""
    windows = int(sum(len(np.asarray(b["masked_tokens"])) for b in batches))
    total = np.sum([np.asarray(b["energy_trace"], np.float64) for b in batches], 0)
    values = total / max(windows, 1)
    last = float(values[-1])
    previous = float(values[-2]) if len(values) > 1 else last
    return {
        "steps": list(range(len(values))),
        "total_energy": values.tolist(),
        "first": float(values[0]),
        "last": last,
        "last_step_relative_change": (
            (last - previous) / abs(previous) if previous else 0.0
        ),
        "num_windows": windows,
    }


def aggregate_map(
    batches: Sequence[Mapping[str, Any]], modules: Sequence[str], scale: float
) -> dict[str, Any]:
    """Average per-sample energies over windows with a masked token.

    ``batches`` are ``node_energies`` outputs. Modules get
    ``final_energy`` and ``forward_error_energy`` (raw energy / ``N²``) and
    their ``*_sum_ce`` variants (raw energy / ``s²``); the head module gets
    its logit signal in both fields and ``output_ce_energy`` is the raw head
    energy / ``N``. Other hidden nodes (``L{i}_residual``) are reported under
    ``residuals``.
    """
    per_node: dict[str, dict[str, list[np.ndarray]]] = {}
    output_ce: list[np.ndarray] = []
    windows = 0
    for result in batches:
        valid = np.asarray(result["masked_tokens"]) > 0
        count = max(float(result["batch_masked_tokens"]), 1.0)
        windows += int(valid.sum())
        output_ce.append(
            np.asarray(result["output_ce_energy"], np.float64)[valid] / count
        )
        fields = {
            field: {
                **result[field],
                "__head__": result["head_energy"][field],
            }
            for field in ENERGY_FIELDS
        }
        for field, values in fields.items():
            for node, value in values.items():
                selected = np.asarray(value, np.float64)[valid]
                entry = per_node.setdefault(node, {})
                entry.setdefault(field, []).append(selected / count**2)
                entry.setdefault(f"{field}_sum_ce", []).append(selected / scale**2)
    if windows == 0:
        raise ValueError("no window has a masked token")
    means = {
        node: {field: float(np.concatenate(v).mean()) for field, v in fields.items()}
        for node, fields in per_node.items()
    }
    head = [m for m in modules if m not in means]
    if len(head) != 1:
        raise ValueError(f"expected exactly one output module, got {head}")
    module_values = {m: means[m] if m in means else means["__head__"] for m in modules}
    residuals = {
        n: v for n, v in means.items() if n not in module_values and n != "__head__"
    }
    return {
        "num_windows": windows,
        "modules": module_values,
        "residuals": residuals,
        "output_ce_energy": float(np.concatenate(output_ce).mean()),
    }


def map_setting(
    context: RunContext, setting: str, config: ErrorMapConfig
) -> tuple[GraphStructure, str, float, dict[str, Any]]:
    """Return a setting's graph, primary energy, output scale and its record."""
    if setting == LEARNER:
        structure = learner_map_structure(context.model_config, context.config)
        primary = learner_energy_field(context.config)
        scale = output_energy_scale(structure)
    else:
        structure = build_map_structure(context, setting, config)
        primary = PRIMARY_ENERGY[setting]
        scale = config.output_energy_scale
    inference = structure.config["inference"]
    record = {
        "inference": type(inference).__name__,
        **{k: v for k, v in dict(inference.config).items()},
        "primary_energy": primary,
    }
    if setting == LEARNER:
        record.update(
            mode=context.config.mode,
            pc_inference=(
                context.config.pc_inference if context.config.mode == "pc" else None
            ),
            output_energy_scale=scale,
        )
    return structure, primary, scale, record


def _shared_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k != "settings"}


def missing_settings(
    existing: Mapping[str, Any],
    context: RunContext,
    config: ErrorMapConfig,
) -> tuple[str, ...]:
    """Return the settings of ``config`` that ``existing`` lacks.

    Raises ``ValueError`` when ``existing`` was computed from other stages,
    windows or head energy, or holds one of the settings with another
    inference record, since merging would change it.
    """
    if existing.get("provenance") != run_provenance(context):
        raise ValueError("existing error maps were computed from other stages")
    if existing.get("head_energy") != HEAD_ENERGY:
        raise ValueError("existing error maps use another head energy")
    requested = _shared_fields(json.loads(json.dumps(asdict(config))))
    if _shared_fields(existing.get("config", {})) != requested:
        raise ValueError(
            f"existing error maps use config {_shared_fields(existing['config'])}, "
            f"not {requested}"
        )
    missing = []
    for setting in config.settings:
        if setting not in existing.get("settings", {}):
            missing.append(setting)
            continue
        record = json.loads(json.dumps(map_setting(context, setting, config)[3]))
        if existing["settings"][setting] != record:
            raise ValueError(
                f"existing error maps hold setting {setting!r} as "
                f"{existing['settings'][setting]}, not {record}"
            )
    return tuple(missing)


def merge_error_maps(
    existing: Mapping[str, Any], new: Mapping[str, Any]
) -> dict[str, Any]:
    """Add ``new``'s settings to ``existing``, refusing to change any of its keys.

    Both must share provenance, head energy and every config field but
    ``settings``; ``new`` must not repeat a setting of ``existing``.
    """
    new = json.loads(json.dumps(new))
    for key in ("provenance", "head_energy", "domains", "modules"):
        if existing.get(key) != new.get(key):
            raise ValueError(f"error maps differ in {key}; refusing to merge")
    if _shared_fields(existing["config"]) != _shared_fields(new["config"]):
        raise ValueError("error maps differ in their config; refusing to merge")
    repeated = sorted(set(existing["settings"]) & set(new["settings"]))
    if repeated:
        raise ValueError(f"settings {repeated} already exist; refusing to change them")
    merged = dict(existing)
    merged["settings"] = {**existing["settings"], **new["settings"]}
    merged["config"] = {
        **existing["config"],
        "settings": [*existing["config"]["settings"], *new["config"]["settings"]],
    }
    merged["maps"] = [*existing["maps"], *new["maps"]]
    return merged


def compute_error_maps(
    context: RunContext,
    config: ErrorMapConfig = ErrorMapConfig(),
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute every map of ``map_specs`` in every setting of ``config``."""
    config.validate()
    modules = module_names(context.structure, context.snapshots[0])
    settings: dict[str, Any] = {}
    maps = []
    for setting in config.settings:
        structure, primary, scale, settings[setting] = map_setting(
            context, setting, config
        )
        trace = setting == STANDARD_PC
        step = jax.jit(
            lambda p, b, s=structure, t=trace: node_energies(p, s, b, trace=t)
        )
        dims = output_dims(structure, modules)
        for spec in map_specs(context.domains):
            params = context.params(spec["stage"])
            batches = window_batches(
                context.valid_sets[spec["domain"]],
                config.num_windows,
                config.batch_size,
            )
            results = [jax.device_get(step(params, batch)) for batch in batches]
            aggregated = aggregate_map(results, modules, scale)
            for module, values in aggregated["modules"].items():
                values["energy"] = values[primary]
                values["energy_sum_ce"] = values[f"{primary}_sum_ce"]
                values["output_dim"] = dims[module]
                values["energy_sum_ce_per_dim"] = values["energy_sum_ce"] / dims[module]
            if trace:
                aggregated["energy_trace"] = energy_trace_summary(results)
            maps.append(
                {
                    **spec,
                    "snapshot": stage_name(spec["stage"]),
                    "setting": setting,
                    **aggregated,
                }
            )
            if verbose:
                print(
                    f"{setting} {spec['kind']} {spec['domain']}@S{spec['stage']} "
                    f"windows={aggregated['num_windows']}"
                )
    return {
        "domains": list(context.domains),
        "modules": modules,
        "provenance": run_provenance(context),
        "config": asdict(config),
        "head_energy": HEAD_ENERGY,
        "settings": settings,
        "normalisation": (
            "energy = per-sample 1/2||.||^2 / N^2 averaged over windows with a "
            "masked token, with the head energy s * sum(masked CE) and N the "
            "batch's masked-token count; *_sum_ce = raw energy / s^2 is "
            "batch-composition invariant in the spike-only setting"
        ),
        "maps": maps,
    }

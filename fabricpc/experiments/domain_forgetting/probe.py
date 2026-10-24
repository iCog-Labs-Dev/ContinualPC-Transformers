"""Per-module PC vs backprop gradient alignment probe.

Compares PC weight gradients with backprop gradients of the masked CE on a
few fixed training batches of one domain, for one or several PC inference
settings, from a checkpoint or from a stage snapshot of a finished run.
A module is one parameterised graph node, as in ``localization.py``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from fabricpc.experiments.domain_forgetting.artifacts import file_sha256
from fabricpc.experiments.domain_forgetting.data import load_domain_cache
from fabricpc.experiments.domain_forgetting.runner import (
    DEFAULT_FORWARD_UPDATES,
    RunConfig,
    build_inference,
    build_structure,
    fixed_mlm_batches,
    gradient_alignment,
    load_base_model,
    load_stage_snapshot,
    make_alignment_fn,
)
from fabricpc.experiments.forgetting_benchmark.sequential_trainer import params_hash

Setting = tuple[float, int | None]


def parse_grid(text: str) -> list[tuple[float, int]]:
    """Parse ``"eta:steps,eta:steps,..."`` into ``[(eta, steps), ...]``."""
    settings = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"grid entry {item!r} is not eta:steps")
        try:
            eta, steps = float(parts[0]), int(parts[1])
        except ValueError as error:
            raise ValueError(f"grid entry {item!r} is not eta:steps") from error
        if steps < 1:
            raise ValueError(f"grid entry {item!r} needs steps >= 1")
        settings.append((eta, steps))
    if not settings:
        raise ValueError("grid has no eta:steps entries")
    return settings


def _stage_params(run_dir: Path, stage: int, structure: Any) -> tuple[Any, str]:
    """Load a verified ``stage_<s>.npz`` of a run; return params and their hash."""
    manifest = json.loads((run_dir / "checkpoint_manifest.json").read_text())
    entries = {int(entry["stage"]): entry for entry in manifest["stages"]}
    if stage not in entries:
        raise ValueError(f"{run_dir}: no committed stage {stage}")
    entry = entries[stage]
    path = run_dir / entry["file"]
    if file_sha256(path) != entry["file_sha256"]:
        raise ValueError(f"{path}: SHA-256 does not match the manifest")
    params = load_stage_snapshot(path, structure)
    sha = params_hash(params)
    if sha != entry["params_sha256"]:
        raise ValueError(f"{path}: params SHA-256 does not match the manifest")
    return params, sha


def probe_alignment(
    *,
    domain: str,
    checkpoint: str | None = None,
    data_dir: str | None = None,
    run_dir: str | Path | None = None,
    stage: int | None = None,
    num_batches: int = 4,
    batch_size: int = 8,
    pc_inference: str = "spiking",
    settings: Sequence[Setting] = ((0.0, None),),
    max_norm: float | None = None,
    output_energy_scale: float | None = None,
    forward_updates: str | None = None,
    dropout_rate: float = 0.0,
    mask_prob: float | None = None,
    seed: int = 0,
    expected_shape: dict[str, int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Return per-module PC vs backprop alignment for each ``(eta, steps)`` setting.

    Params are the checkpoint's, or stage ``stage`` of ``run_dir`` (checked
    against its manifest); ``checkpoint`` and ``data_dir`` default to the
    run's, and so do ``max_norm``, ``output_energy_scale`` and ``mask_prob``
    when not given (otherwise 5.0, 1.0 and 0.15). Dropout stays 0 unless
    ``dropout_rate`` is given. Every setting sees the same ``num_batches``
    unshuffled batches of
    ``domain``'s training stream, with masks and dropout keys from ``seed``.
    ``forward_updates`` defaults to ``"initial"`` for spiking and ``"none"``
    for standard PC. Each setting holds ``gradient_alignment``'s result.
    """
    if num_batches < 1 or batch_size < 1:
        raise ValueError("num_batches and batch_size must be positive")
    if (stage is None) != (run_dir is None):
        raise ValueError("stage and run_dir must be given together")
    if not settings:
        raise ValueError("probe needs at least one setting")
    run_path = Path(run_dir) if run_dir is not None else None
    if run_path is not None:
        recorded = json.loads((run_path / "run_config.json").read_text())["config"]
        checkpoint = checkpoint or recorded["checkpoint"]
        data_dir = data_dir or recorded["data_dir"]
        max_norm = recorded["pc_max_norm"] if max_norm is None else max_norm
        if output_energy_scale is None:
            output_energy_scale = recorded["pc_output_energy_scale"]
        mask_prob = recorded["mask_prob"] if mask_prob is None else mask_prob
    max_norm = 5.0 if max_norm is None else max_norm
    output_energy_scale = 1.0 if output_energy_scale is None else output_energy_scale
    mask_prob = 0.15 if mask_prob is None else mask_prob
    if checkpoint is None:
        raise ValueError("probe needs a checkpoint or a run_dir")
    data_dir = data_dir or "data/domain_forgetting"
    forward_updates = forward_updates or DEFAULT_FORWARD_UPDATES.get(
        pc_inference, "initial"
    )
    first_eta, first_steps = settings[0]
    config = RunConfig(
        checkpoint=str(checkpoint),
        data_dir=str(data_dir),
        domains=(domain,),
        output_dir="",
        mode="pc",
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        mask_prob=mask_prob,
        seed=seed,
        expected_shape=expected_shape,
        pc_inference=pc_inference,
        pc_eta_infer=first_eta,
        pc_infer_steps=first_steps,
        pc_forward_updates=forward_updates,
        pc_max_norm=max_norm,
        pc_output_energy_scale=output_energy_scale,
    )
    for eta, steps in settings:
        replace(config, pc_eta_infer=eta, pc_infer_steps=steps).validate()
    base = load_base_model(config)
    params, sha = base.params, base.source["params_sha256"]
    if run_path is not None:
        params, sha = _stage_params(run_path, int(stage), base.structure)
    cache = load_domain_cache(
        config.data_dir,
        domain,
        tokenizer_json=base.tokenizer_json,
        seq_len=int(base.model_config["seq_len"]),
    )
    batches, keys = fixed_mlm_batches(
        cache.train,
        base.structure,
        base.special_ids,
        num_batches=num_batches,
        batch_size=batch_size,
        mask_prob=mask_prob,
        seed=seed,
    )
    results = []
    for eta, steps in settings:
        setting_config = replace(config, pc_eta_infer=eta, pc_infer_steps=steps)
        inference, infer_steps = build_inference(setting_config, base.model_config)
        structure = build_structure(base.model_config, setting_config, inference)
        alignment = gradient_alignment(
            make_alignment_fn(structure, forward_updates),
            params,
            batches,
            keys,
            structure,
        )
        results.append({"eta_infer": eta, "infer_steps": infer_steps, **alignment})
        if verbose:
            print(format_alignment(results[-1]))
    return {
        "checkpoint": str(checkpoint),
        "run_dir": str(run_path) if run_path is not None else None,
        "stage": stage,
        "params_sha256": sha,
        "domain": domain,
        "num_batches": num_batches,
        "batch_size": batch_size,
        "seed": seed,
        "dropout_rate": dropout_rate,
        "mask_prob": mask_prob,
        "pc_inference": pc_inference,
        "forward_updates": forward_updates,
        "max_norm": max_norm,
        "output_energy_scale": output_energy_scale,
        "settings": results,
    }


def _fmt(value: float | None, spec: str) -> str:
    """``value`` formatted with ``spec``, or ``n/a`` when it is None."""
    return "n/a" if value is None else format(value, spec)


def format_alignment(result: dict[str, Any]) -> str:
    """One line per module, then the mean, for one probed setting."""
    prefix = f"eta={result['eta_infer']:g} steps={result['infer_steps']}"
    width = max(len(name) for name in result["modules"])
    lines = [
        f"{prefix} {name:<{width}} cos={row['cosine']:.4f} "
        f"min={row['cosine_min']:.4f} ratio={_fmt(row['norm_ratio'], '.3e')} "
        f"rel={_fmt(row['rel_norm_ratio'], '.3e')}"
        for name, row in result["modules"].items()
    ]
    lines.append(
        f"{prefix} {'mean':<{width}} cos={result['mean_cosine']:.4f} "
        f"hidden={_fmt(result.get('mean_cosine_hidden'), '.4f')} "
        f"min={result['min_cosine']:.4f}"
    )
    return "\n".join(lines)

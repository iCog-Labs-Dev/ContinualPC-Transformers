"""Repeat Stage A with smaller latent rates and explicit frozen-control checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/content/anchored_pc"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    previous = root / "results_20261024T122712Z"
    original = json.loads((previous / "summary.json").read_text())
    manifest = json.loads((root / "bundle_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source/input changed: {name}")

    sys.path.insert(0, str(root / "src"))
    import jax
    from fabricpc.experiments.anchored_pc import (
        DiagnosticConfig,
        render_report,
        run_diagnostic,
    )

    if jax.devices()[0].platform != "gpu":
        raise RuntimeError(
            "Follow-up requires the Colab GPU; refusing silent CPU fallback"
        )
    values = dict(original["config"])
    values.update(rates=(0.001, 0.0001), output_dir=str(args.output))
    for name in ("step_pairs", "scales"):
        values[name] = tuple(values[name])
    config = DiagnosticConfig(**values)
    config.validate()
    serialized = json.loads(json.dumps(asdict(config)))
    changes = {
        key: {"before": original["config"][key], "after": value}
        for key, value in serialized.items()
        if original["config"][key] != value
    }
    if set(changes) != {"rates", "output_dir"}:
        raise ValueError(f"Unexpected experimental changes: {changes}")
    print("Controlled changes:", json.dumps(changes), flush=True)
    result = run_diagnostic(config)
    if result.get("initial_params_sha256") != original["initial_params_sha256"]:
        raise RuntimeError("Checkpoint parameters do not match the first run")
    with np.load(previous / "probe_batches.npz", allow_pickle=False) as old:
        with np.load(args.output / "probe_batches.npz", allow_pickle=False) as new:
            if set(old.files) != set(new.files) or not all(
                np.array_equal(old[name], new[name]) for name in old.files
            ):
                raise RuntimeError(
                    "Training/validation batches or dropout keys changed"
                )
    provenance = {
        "changes": changes,
        "same_params": True,
        "same_batches_and_keys": True,
        "source_manifest": manifest,
        "previous_status": original["status"],
    }
    (args.output / "followup_provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    (args.output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    hardware = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    (args.output / "hardware.txt").write_text(hardware.stdout + hardware.stderr)
    render_report(args.output)
    print(
        "Frozen checkpoint, batch arrays, masks and keys verified identical", flush=True
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "decision",
                    "reason",
                    "elapsed_seconds",
                    "selected_solver",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

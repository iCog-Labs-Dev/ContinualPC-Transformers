"""pcinterp adapter for the current FabricPC Transformer V2 repository.

This file is repository-side integration code.  It does not reimplement any
pcinterp measurement.  It exposes deterministic, ordinary feed-forward states
from a saved Transformer V2 checkpoint and provides the continuation operation
needed by pcinterp activation patching.

All representations use the final sequence position:

    embed, L0_mha, L0_mlp1, L0_mlp2,
    L1_mha, L1_mlp1, L1_mlp2, logits

Internal nodes expose ``z_mu``.  ``logits`` exposes the corrected
``pre_activation`` field, i.e. raw vocabulary logits rather than softmax
probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.core.inference import InferenceSGDNormClip, gather_inputs
from fabricpc.core.scaling import scale_inputs
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.graph_initialization import initialize_graph_state
from fabricpc.nodes.transformer_v2 import create_deep_transformer


GRAPH_NODES: tuple[str, ...] = (
    "embed",
    "L0_mha",
    "L0_mlp1",
    "L0_mlp2",
    "L1_mha",
    "L1_mlp1",
    "L1_mlp2",
    "logits",
)
CHAR_V1_DATASET_SHA256 = (
    "7c0858a950cc59c962f4c4a1cd77c2bf5fb788d4859b587ed1dbeff3072fed9a"
)
CHAR_V1_MANIFEST_SHA256 = (
    "c776d6a43f454b31c9525927eb76268dd3f579aa4e2914c60e6b24d5a393441f"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_frozen_dataset(directory: str | Path) -> Any:
    """Verify and load ``char_v1`` as a pcinterp Dataset."""
    from pcinterp.probes import Split
    from pcinterp.studies.common import Dataset

    directory = Path(directory)
    dataset_path = directory / "dataset.npz"
    manifest_path = directory / "manifest.json"
    observed_manifest_hash = _file_sha256(manifest_path)
    if observed_manifest_hash != CHAR_V1_MANIFEST_SHA256:
        raise RuntimeError(
            "frozen manifest hash mismatch: "
            f"expected {CHAR_V1_MANIFEST_SHA256}, "
            f"observed {observed_manifest_hash}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    observed_file_hash = _file_sha256(dataset_path)
    expected_file_hash = manifest["dataset_file_sha256"]
    if expected_file_hash != CHAR_V1_DATASET_SHA256:
        raise RuntimeError("manifest does not name the frozen char_v1 dataset hash")
    if observed_file_hash != expected_file_hash:
        raise RuntimeError(
            "frozen dataset hash mismatch: "
            f"expected {expected_file_hash}, observed {observed_file_hash}"
        )

    with np.load(dataset_path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}

    expected_arrays = manifest["array_sha256"]
    if set(arrays) != set(expected_arrays):
        raise RuntimeError("dataset arrays do not match the frozen manifest")
    for name, expected in expected_arrays.items():
        observed = _array_sha256(arrays[name])
        if observed != expected:
            raise RuntimeError(
                f"frozen array hash mismatch for {name}: "
                f"expected {expected}, observed {observed}"
            )

    return Dataset(
        inputs=arrays["inputs"],
        split=Split(
            train=arrays["split_train"],
            validation=arrays["split_validation"],
            test=arrays["split_test"],
        ),
        labels={
            "long_range_recurrence": arrays["long_range_recurrence"],
            "strict_induction": arrays["strict_induction"],
        },
    )


class FabricPCTransformerV2Adapter:
    """Read and patch feed-forward states from one Transformer V2 checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        name: str | None = None,
        batch_size: int = 256,
        nodes: Sequence[str] = GRAPH_NODES,
    ) -> None:
        checkpoint = Path(checkpoint)
        with checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        if not isinstance(saved, dict) or not {"params", "config"} <= set(saved):
            raise ValueError("checkpoint must contain 'params' and 'config'")

        # Convert checkpoint NumPy leaves before tracing the forward pass.
        self.params = jax.tree_util.tree_map(
            jnp.asarray,
            saved["params"],
        )
        self.config = dict(saved["config"])
        self._nodes = tuple(nodes)
        self._batch_size = int(batch_size)
        self.name = name or checkpoint.stem

        if self._batch_size < 1:
            raise ValueError("batch_size must be positive")
        missing = set(self._nodes) - set(GRAPH_NODES)
        if missing:
            raise KeyError(f"unsupported graph nodes: {sorted(missing)}")
        if int(self.config["depth"]) != 2:
            raise ValueError("the eight-node paper adapter requires depth=2")

        vocab_size = int(
            self.params.nodes["logits"].biases["b_out"].shape[-1]
        )
        # Inference is deliberately zero-step here.  pcinterp measures the
        # ordinary feed-forward representation learned by each training rule.
        inference = InferenceSGDNormClip(
            eta_infer=float(self.config.get("eta_infer", 0.0483)),
            infer_steps=0,
            latent_decay=0.0,
            max_norm=float(self.config.get("max_norm", 5.0)),
        )
        self.structure = create_deep_transformer(
            depth=int(self.config["depth"]),
            embed_dim=int(self.config["embed_dim"]),
            num_heads=int(self.config["num_heads"]),
            mlp_dim=int(self.config["mlp_dim"]),
            seq_len=int(self.config["seq_len"]),
            vocab_size=vocab_size,
            inference=inference,
            weight_init={
                "type": "normal",
                "std": float(self.config.get("weight_init_std", 0.0228)),
            },
            dropout_rate=float(self.config.get("dropout_rate", 0.0)),
        )

        absent = set(GRAPH_NODES) - set(self.structure.nodes)
        if absent:
            raise RuntimeError(f"reconstructed graph is missing nodes: {sorted(absent)}")
        checkpoint_nodes = set(self.params.nodes)
        missing_parameters = set(GRAPH_NODES) - checkpoint_nodes

        if missing_parameters:
            raise RuntimeError(
                "checkpoint is missing required Transformer parameters: "
                f"{sorted(missing_parameters)}"
            )

        self._activation_fns: dict[tuple[str, ...], Any] = {}
        self._patch_fns: dict[str, Any] = {}

    def node_names(self) -> tuple[str, ...]:
        return self._nodes

    def _validate_inputs(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs)
        expected = int(self.config["seq_len"])
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(
                f"inputs must have shape (n_samples, {expected}); got {values.shape}"
            )
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"token inputs must be integer-valued; got {values.dtype}")
        return values.astype(np.int32, copy=False)

    def _batches(self, inputs: np.ndarray):
        import jax.numpy as jnp

        for start in range(0, len(inputs), self._batch_size):
            yield start, jnp.asarray(inputs[start : start + self._batch_size])

    def _eval_clamps(self, inputs: Any) -> dict[str, Any]:
        import jax.numpy as jnp

        clamps: dict[str, Any] = {
            self.structure.task_map["x"]: inputs,
        }
        for task_name, node_name in self.structure.task_map.items():
            if task_name.startswith("dropout_"):
                node_shape = self.structure.nodes[node_name].node_info.shape
                clamps[node_name] = jnp.ones(
                    (inputs.shape[0], *node_shape), dtype=jnp.float32
                )
        return clamps

    def _feedforward_state(self, inputs: Any) -> Any:
        import jax

        clamps = self._eval_clamps(inputs)
        return initialize_graph_state(
            self.structure,
            inputs.shape[0],
            jax.random.PRNGKey(0),
            clamps=clamps,
            params=self.params,
        )

    @staticmethod
    def _node_value(state: Any, node: str) -> Any:
        if node == "logits":
            # VocabProjectionNode stores its raw scores here after the
            # pre_activation repair; z_mu contains softmax probabilities.
            return state.nodes[node].pre_activation[:, -1, :]
        return state.nodes[node].z_mu[:, -1, :]

    def _activation_fn(self, wanted: tuple[str, ...]):
        import jax

        if wanted not in self._activation_fns:
            def calculate(inputs):
                state = self._feedforward_state(inputs)
                return tuple(self._node_value(state, node) for node in wanted)

            self._activation_fns[wanted] = jax.jit(calculate)
        return self._activation_fns[wanted]

    def activations(
        self,
        inputs: np.ndarray,
        nodes: Sequence[str],
    ) -> dict[str, np.ndarray]:
        import jax

        values = self._validate_inputs(inputs)
        wanted = tuple(nodes)
        unknown = set(wanted) - set(self._nodes)
        if unknown:
            raise KeyError(f"nodes not exposed by this adapter: {sorted(unknown)}")
        if len(set(wanted)) != len(wanted):
            raise ValueError("requested nodes must be unique")
        if not wanted:
            return {}

        compiled = self._activation_fn(wanted)
        collected: dict[str, list[np.ndarray]] = {node: [] for node in wanted}
        for _, piece in self._batches(values):
            output = compiled(piece)
            jax.block_until_ready(output)
            for node, matrix in zip(wanted, output, strict=True):
                collected[node].append(np.asarray(matrix, dtype=np.float32))
        return {
            node: np.concatenate(parts, axis=0)
            for node, parts in collected.items()
        }

    def logits(self, inputs: np.ndarray) -> np.ndarray:
        return self.activations(inputs, ("logits",))["logits"]

    def _continue_from_patch(self, state: Any, node: str, replacement: Any) -> Any:
        import jax.numpy as jnp

        node_state = state.nodes[node]
        patched_latent = node_state.z_latent.at[:, -1, :].set(replacement)
        state = update_node_in_state(state, node, z_latent=patched_latent)

        try:
            start = self.structure.node_order.index(node) + 1
        except ValueError as error:
            raise KeyError(f"node absent from graph order: {node}") from error

        for downstream in self.structure.node_order[start:]:
            graph_node = self.structure.nodes[downstream]
            info = graph_node.node_info
            if info.in_degree == 0:
                continue
            edge_inputs = gather_inputs(info, self.structure, state)
            scaled_inputs = scale_inputs(edge_inputs, info.scaling_config)
            _, projected = info.node_class.forward(
                self.params.nodes[downstream],
                scaled_inputs,
                state.nodes[downstream],
                info,
            )
            projected = projected._replace(
                z_latent=projected.z_mu,
                error=jnp.zeros_like(projected.error),
            )
            state = state._replace(
                nodes={**state.nodes, downstream: projected}
            )
        return state

    def _patch_fn(self, node: str):
        import jax

        if node not in self._patch_fns:
            def calculate(inputs, replacement):
                if node == "logits":
                    return replacement
                state = self._feedforward_state(inputs)
                state = self._continue_from_patch(state, node, replacement)
                return self._node_value(state, "logits")

            self._patch_fns[node] = jax.jit(calculate)
        return self._patch_fns[node]

    def logits_with_patch(
        self,
        inputs: np.ndarray,
        node: str,
        replacement: np.ndarray,
    ) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        values = self._validate_inputs(inputs)
        if node not in self._nodes:
            raise KeyError(f"node not exposed by this adapter: {node}")
        replacement = np.asarray(replacement, dtype=np.float32)
        expected_width = int(self.structure.nodes[node].node_info.shape[-1])
        expected_shape = (len(values), expected_width)
        if replacement.shape != expected_shape:
            raise ValueError(
                f"replacement for {node} must have shape {expected_shape}; "
                f"got {replacement.shape}"
            )

        compiled = self._patch_fn(node)
        out: list[np.ndarray] = []
        for start, piece in self._batches(values):
            stop = start + piece.shape[0]
            result = compiled(piece, jnp.asarray(replacement[start:stop]))
            jax.block_until_ready(result)
            out.append(np.asarray(result, dtype=np.float32))
        return np.concatenate(out, axis=0)


def _check_every_patch(adapter: FabricPCTransformerV2Adapter, inputs: np.ndarray) -> None:
    """Add an all-node patch check to pcinterp's core adapter validation."""
    activations = adapter.activations(inputs, adapter.node_names())
    baseline = adapter.logits(inputs)
    for node in adapter.node_names():
        identity = adapter.logits_with_patch(inputs, node, activations[node])
        identity_error = float(np.max(np.abs(identity - baseline)))
        if not np.allclose(identity, baseline, atol=1.0e-4, rtol=1.0e-5):
            raise AssertionError(
                f"{node}: identity patch changed logits; max_abs={identity_error:.3g}"
            )

        perturbed = activations[node].copy()
        perturbed[:, 0] += 0.5
        changed = adapter.logits_with_patch(inputs, node, perturbed)
        change = float(np.max(np.abs(changed - baseline)))
        if not change > 1.0e-7:
            raise AssertionError(f"{node}: perturbed patch did not propagate")
        print(
            f"PASS {node:<8} identity_max_abs={identity_error:.3g} "
            f"perturbation_max_abs={change:.3g}"
        )


def main() -> None:
    from pcinterp.model import check_adapter

    parser = argparse.ArgumentParser(description="Validate the FabricPC pcinterp adapter")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("artifacts/interpretation/char_v1"),
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()

    dataset = load_frozen_dataset(args.dataset_dir)
    inputs = np.asarray(dataset.inputs[: args.samples])
    adapter = FabricPCTransformerV2Adapter(
        args.checkpoint,
        name=args.name,
        batch_size=args.batch_size,
    )

    report = check_adapter(adapter, inputs)
    print(json.dumps(report, indent=2, default=str))
    _check_every_patch(adapter, inputs)
    print("FabricPC Transformer V2 adapter contract: OK")


if __name__ == "__main__":
    main()
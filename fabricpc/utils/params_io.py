"""Save and load ``GraphParams`` as an ``.npz`` archive plus a JSON sidecar.

``save_params("model.npz", params, metadata)`` writes the arrays to
``model.npz`` and ``model.json``. The sidecar records the file format and
version, a schema mapping every ``(node, collection, name)`` parameter to its
archive key, a SHA-256 digest of the archive and free-form JSON ``metadata``
such as a vocabulary or model config. Nothing is pickled, so loading never
executes code from the file.

``load_params(path, structure)`` checks the file against the parameters a
fresh ``initialize_params(structure, ...)`` would create, so a file only loads
into a graph with the same parameter names and shapes.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.types import GraphParams, GraphStructure, NodeParams
from fabricpc.graph_initialization import initialize_params

PARAMS_FORMAT = "fabricpc-params"
PARAMS_FORMAT_VERSION = 1
_COLLECTIONS = ("weights", "biases")

PathLike = Union[str, os.PathLike]


def params_paths(path: PathLike) -> Tuple[Path, Path]:
    """Return the ``(.npz archive, .json sidecar)`` paths for ``path``.

    Any suffix of ``path`` is replaced, so ``model``, ``model.npz`` and
    ``model.json`` all name the same pair of files.
    """
    path = Path(path)
    return path.with_suffix(".npz"), path.with_suffix(".json")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_params(
    path: PathLike,
    params: GraphParams,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Write ``params`` and JSON ``metadata`` to the file pair for ``path``.

    Each file is written to a temporary file and renamed into place, so an
    interrupted save never leaves a truncated file. The sidecar's digest lets
    ``load_params`` reject an archive that does not belong to it.

    Args:
        path: Destination; its suffix is replaced by ``.npz`` and ``.json``.
        params: Parameters to save.
        metadata: JSON-serialisable data stored with the parameters.

    Returns:
        The ``(archive, sidecar)`` paths written.

    Raises:
        TypeError: If ``metadata`` is not JSON-serialisable.
    """
    archive_path, sidecar_path = params_paths(path)
    arrays: List[np.ndarray] = []
    schema: Dict[str, Dict[str, Dict[str, str]]] = {}
    for node_name, node in params.nodes.items():
        entry: Dict[str, Dict[str, str]] = {}
        for collection in _COLLECTIONS:
            keys: Dict[str, str] = {}
            for name, value in getattr(node, collection).items():
                keys[name] = f"arr_{len(arrays)}"
                arrays.append(np.asarray(value))
            entry[collection] = keys
        schema[node_name] = entry

    buffer = io.BytesIO()
    np.savez(buffer, *arrays)
    archive = buffer.getvalue()
    sidecar = {
        "format": PARAMS_FORMAT,
        "version": PARAMS_FORMAT_VERSION,
        "archive": archive_path.name,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "schema": schema,
        "metadata": dict(metadata or {}),
    }
    sidecar_text = json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False)
    _atomic_write_bytes(archive_path, archive)
    _atomic_write_bytes(sidecar_path, sidecar_text.encode("utf-8"))
    return archive_path, sidecar_path


def read_params_metadata(path: PathLike) -> Dict[str, Any]:
    """Return the ``metadata`` stored with the parameters at ``path``.

    Only the sidecar is read, so a caller can check the metadata (for example
    a vocabulary) before building a graph to load into.
    """
    return _read_sidecar(params_paths(path)[1])["metadata"]


def _read_sidecar(sidecar_path: Path) -> Dict[str, Any]:
    if not sidecar_path.exists():
        raise FileNotFoundError(f"params sidecar not found: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict) or sidecar.get("format") != PARAMS_FORMAT:
        raise ValueError(f"{sidecar_path} is not a {PARAMS_FORMAT} file")
    if sidecar.get("version") != PARAMS_FORMAT_VERSION:
        raise ValueError(
            f"{sidecar_path}: unsupported {PARAMS_FORMAT} version "
            f"{sidecar.get('version')!r}, expected {PARAMS_FORMAT_VERSION}"
        )
    return sidecar


def _expected_shapes(
    structure: GraphStructure, rng_key: jax.Array
) -> Dict[str, Dict[str, Dict[str, Tuple[int, ...]]]]:
    abstract = jax.eval_shape(lambda key: initialize_params(structure, key), rng_key)
    return {
        node_name: {
            collection: {
                name: tuple(value.shape)
                for name, value in getattr(
                    abstract.nodes[node_name], collection
                ).items()
            }
            for collection in _COLLECTIONS
        }
        for node_name in structure.nodes
    }


def load_params(
    path: PathLike,
    structure: GraphStructure,
    rng_key: Optional[jax.Array] = None,
) -> Tuple[GraphParams, Dict[str, Any]]:
    """Load parameters saved by ``save_params`` for ``structure``.

    The file must hold exactly the parameters ``initialize_params(structure,
    rng_key)`` creates, with the same names and shapes. Nodes without
    parameters may be absent from the file and get empty entries. The result
    follows the order of ``structure.nodes``.

    Args:
        path: The archive, the sidecar, or either without a suffix.
        structure: Graph to load the parameters into.
        rng_key: Key for the reference initialisation; only shapes are used.

    Returns:
        ``(params, metadata)``.

    Raises:
        FileNotFoundError: If either file is missing.
        ValueError: If the files are not a matching pair of this format, or
            any node or parameter is missing, unexpected or mis-shaped.
    """
    archive_path, sidecar_path = params_paths(path)
    sidecar = _read_sidecar(sidecar_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"params archive not found: {archive_path}")
    archive = archive_path.read_bytes()
    if hashlib.sha256(archive).hexdigest() != sidecar.get("archive_sha256"):
        raise ValueError(
            f"{archive_path} does not match the digest recorded in {sidecar_path}"
        )
    schema: Dict[str, Dict[str, Dict[str, str]]] = sidecar["schema"]
    expected = _expected_shapes(
        structure, jax.random.PRNGKey(0) if rng_key is None else rng_key
    )

    problems: List[str] = []
    extra_nodes = sorted(set(schema) - set(expected))
    if extra_nodes:
        problems.append(f"nodes not in the graph: {extra_nodes}")
    nodes: Dict[str, NodeParams] = {}
    with np.load(io.BytesIO(archive), allow_pickle=False) as stored:
        for node_name, node_shapes in expected.items():
            if node_name not in schema:
                if any(node_shapes[c] for c in _COLLECTIONS):
                    problems.append(f"missing node {node_name!r}")
                nodes[node_name] = NodeParams(weights={}, biases={})
                continue
            loaded: Dict[str, Dict[str, jnp.ndarray]] = {}
            for collection in _COLLECTIONS:
                keys = schema[node_name].get(collection, {})
                shapes = node_shapes[collection]
                where = f"{node_name}.{collection}"
                missing = sorted(set(shapes) - set(keys))
                extra = sorted(set(keys) - set(shapes))
                if missing:
                    problems.append(f"{where}: missing {missing}")
                if extra:
                    problems.append(f"{where}: unexpected {extra}")
                loaded[collection] = {}
                for name in shapes:
                    if name not in keys:
                        continue
                    if keys[name] not in stored.files:
                        problems.append(f"{where}.{name}: not in the archive")
                        continue
                    value = stored[keys[name]]
                    if tuple(value.shape) != shapes[name]:
                        problems.append(
                            f"{where}.{name}: shape {tuple(value.shape)}, "
                            f"expected {shapes[name]}"
                        )
                        continue
                    loaded[collection][name] = jnp.asarray(value)
            nodes[node_name] = NodeParams(
                weights=loaded["weights"], biases=loaded["biases"]
            )
    if problems:
        raise ValueError(
            f"{archive_path} does not match the graph:\n  " + "\n  ".join(problems)
        )
    return GraphParams(nodes=nodes), sidecar["metadata"]

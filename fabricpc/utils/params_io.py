"""Save and load ``GraphParams`` as an ``.npz`` archive plus a JSON sidecar.

``save_params("model.npz", params, metadata)`` writes the arrays to
``model.npz`` and ``model.json``. The sidecar records the file format and
version, a schema mapping every ``(node, collection, name)`` parameter to its
archive key and dtype, a SHA-256 digest and free-form JSON ``metadata`` such
as a vocabulary or model config. The digest covers the archive bytes and the
schema, not the metadata. Nothing is pickled, so loading never executes code
from the file.

Each file is replaced atomically, but the pair is not: a crash between the
two writes leaves a new archive beside the old sidecar, which the digest
check in ``load_params`` then rejects.

``load_params(path, structure)`` checks the file against the parameters a
fresh ``initialize_params(structure, ...)`` would create, so a file only loads
into a graph with the same parameter names, shapes and dtypes.
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
_SUFFIXES = (".npz", ".json")

PathLike = Union[str, os.PathLike]


def params_paths(path: PathLike) -> Tuple[Path, Path]:
    """Return the ``(.npz archive, .json sidecar)`` paths for ``path``.

    A ``.npz`` or ``.json`` suffix is stripped and any other suffix kept, so
    ``model``, ``model.npz`` and ``model.json`` name the same pair of files
    and ``bert.base`` names ``bert.base.npz`` and ``bert.base.json``.
    """
    path = Path(path)
    if path.suffix in _SUFFIXES:
        path = path.with_suffix("")
    return path.with_name(f"{path.name}.npz"), path.with_name(f"{path.name}.json")


def _digest(archive: bytes, schema: Mapping[str, Any]) -> str:
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(archive + canonical.encode("utf-8")).hexdigest()


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
    ``load_params`` reject an archive that does not belong to it. Dtypes numpy
    cannot store natively, such as ``bfloat16``, are recorded in the schema
    and restored on load.

    Args:
        path: Destination; see ``params_paths``.
        params: Parameters to save.
        metadata: JSON-serialisable data stored with the parameters.

    Returns:
        The ``(archive, sidecar)`` paths written.

    Raises:
        TypeError: If ``metadata`` holds a value JSON cannot represent.
        ValueError: If ``metadata`` holds a NaN or infinite float.
    """
    archive_path, sidecar_path = params_paths(path)
    arrays: List[np.ndarray] = []
    schema: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}
    for node_name, node in params.nodes.items():
        entry: Dict[str, Dict[str, Dict[str, str]]] = {}
        for collection in _COLLECTIONS:
            keys: Dict[str, Dict[str, str]] = {}
            for name, value in getattr(node, collection).items():
                array = np.asarray(value)
                keys[name] = {"key": f"arr_{len(arrays)}", "dtype": array.dtype.name}
                arrays.append(array)
            entry[collection] = keys
        schema[node_name] = entry

    buffer = io.BytesIO()
    np.savez(buffer, *arrays)
    archive = buffer.getvalue()
    sidecar = {
        "format": PARAMS_FORMAT,
        "version": PARAMS_FORMAT_VERSION,
        "archive": archive_path.name,
        "sha256": _digest(archive, schema),
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
    problem = _schema_problem(sidecar)
    if problem:
        raise ValueError(f"{sidecar_path}: malformed sidecar, {problem}")
    return sidecar


def _schema_problem(sidecar: Dict[str, Any]) -> Optional[str]:
    """Describe the first structural problem of a sidecar, or return None."""
    if not isinstance(sidecar.get("sha256"), str):
        return "'sha256' must be a string"
    if not isinstance(sidecar.get("metadata"), dict):
        return "'metadata' must be an object"
    schema = sidecar.get("schema")
    if not isinstance(schema, dict):
        return "'schema' must be an object"
    for node_name, entry in schema.items():
        if not isinstance(entry, dict):
            return f"schema entry {node_name!r} must be an object"
        unknown = sorted(set(entry) - set(_COLLECTIONS))
        if unknown:
            return f"{node_name!r} has unknown collections {unknown}"
        for collection, params in entry.items():
            if not isinstance(params, dict):
                return f"{node_name}.{collection} must be an object"
            for name, spec in params.items():
                if not (
                    isinstance(spec, dict)
                    and isinstance(spec.get("key"), str)
                    and isinstance(spec.get("dtype"), str)
                ):
                    return f"{node_name}.{collection}.{name} needs string 'key' and 'dtype'"
    return None


def _expected_shapes(
    structure: GraphStructure, rng_key: jax.Array
) -> Dict[str, Dict[str, Dict[str, jax.ShapeDtypeStruct]]]:
    abstract = jax.eval_shape(lambda key: initialize_params(structure, key), rng_key)
    return {
        node_name: {
            collection: {
                name: value
                for name, value in getattr(
                    abstract.nodes[node_name], collection
                ).items()
            }
            for collection in _COLLECTIONS
        }
        for node_name in structure.nodes
    }


def _read_array(
    stored: Any, spec: Mapping[str, str], reference: jax.ShapeDtypeStruct
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return ``(array, None)``, or ``(None, problem)`` if it does not fit."""
    if spec["key"] not in stored.files:
        return None, "not in the archive"
    value = stored[spec["key"]]
    try:
        dtype = jnp.dtype(spec["dtype"])
    except TypeError:
        return None, f"unknown dtype {spec['dtype']!r}"
    if value.dtype != dtype:
        if value.dtype.kind != "V" or value.dtype.itemsize != dtype.itemsize:
            return None, f"stored as {value.dtype}, schema says {dtype}"
        value = value.view(dtype)
    if tuple(value.shape) != tuple(reference.shape):
        return None, f"shape {tuple(value.shape)}, expected {tuple(reference.shape)}"
    if value.dtype != reference.dtype:
        return None, f"dtype {value.dtype}, expected {reference.dtype}"
    return value, None


def load_params(
    path: PathLike,
    structure: GraphStructure,
    rng_key: Optional[jax.Array] = None,
) -> Tuple[GraphParams, Dict[str, Any]]:
    """Load parameters saved by ``save_params`` for ``structure``.

    The file must hold exactly the parameters ``initialize_params(structure,
    rng_key)`` creates, with the same names, shapes and dtypes. Nodes without
    parameters may be absent from the file and get empty entries. The result
    follows the order of ``structure.nodes``.

    Args:
        path: The archive, the sidecar, or either without a suffix.
        structure: Graph to load the parameters into.
        rng_key: Key for the reference initialisation; only shapes and dtypes
            are used.

    Returns:
        ``(params, metadata)``.

    Raises:
        FileNotFoundError: If either file is missing.
        ValueError: If the files are not a matching pair of this format, or
            any node or parameter is missing, unexpected, mis-shaped or of
            another dtype.
    """
    archive_path, sidecar_path = params_paths(path)
    sidecar = _read_sidecar(sidecar_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"params archive not found: {archive_path}")
    archive = archive_path.read_bytes()
    schema: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = sidecar["schema"]
    if _digest(archive, schema) != sidecar["sha256"]:
        raise ValueError(
            f"{archive_path} does not match the digest recorded in {sidecar_path}"
        )
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
                specs = schema[node_name].get(collection, {})
                shapes = node_shapes[collection]
                where = f"{node_name}.{collection}"
                missing = sorted(set(shapes) - set(specs))
                extra = sorted(set(specs) - set(shapes))
                if missing:
                    problems.append(f"{where}: missing {missing}")
                if extra:
                    problems.append(f"{where}: unexpected {extra}")
                loaded[collection] = {}
                for name, reference in shapes.items():
                    if name not in specs:
                        continue
                    value, problem = _read_array(stored, specs[name], reference)
                    if problem:
                        problems.append(f"{where}.{name}: {problem}")
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

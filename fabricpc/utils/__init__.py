"""Utility functions and data loading for FabricPC."""

import importlib

from fabricpc.utils.helpers import layernorm
from fabricpc.utils.params_io import load_params, read_params_metadata, save_params

# Submodules
from fabricpc.utils import data


# Lazy import for dashboarding to avoid requiring aim at import time
def __getattr__(name: str):
    if name == "dashboarding":
        return importlib.import_module("fabricpc.utils.dashboarding")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Helpers
    "layernorm",
    # Parameter files
    "save_params",
    "load_params",
    "read_params_metadata",
    # Submodules
    "data",
    "dashboarding",
]

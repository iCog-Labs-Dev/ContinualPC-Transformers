"""Shared offline fixtures for the domain-forgetting analysis tests."""

import numpy as np

from fabricpc.experiments.domain_forgetting.runner import run_domain_forgetting
from tests.test_domain_forgetting_data import build_tiny_caches, tiny_tokenizer
from tests.test_domain_forgetting_runner import tiny_config, write_tiny_bert_checkpoint

TINY_DEPTH = 2


def build_tiny_run(directory, *, dtype=np.float32, depth=TINY_DEPTH, **overrides):
    """Train a tiny two-block BERT on three synthetic domains; return its config."""
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = tiny_tokenizer(directory)
    checkpoint = write_tiny_bert_checkpoint(
        directory, tokenizer, dtype=dtype, depth=depth
    )
    build_tiny_caches(directory, tokenizer)
    config = tiny_config(checkpoint, directory / "data", directory / "run", **overrides)
    run_domain_forgetting(config, verbose=False)
    return config

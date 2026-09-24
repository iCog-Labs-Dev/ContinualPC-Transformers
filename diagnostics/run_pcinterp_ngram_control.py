"""Run fixed character-level controls on the frozen probe dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pcinterp.studies import probe_study

from diagnostics.build_interpretation_dataset import fit_trigram_counts
from diagnostics.pcinterp_adapter import load_frozen_dataset
from fabricpc.utils.data import CharDataLoader


PROBE_BUDGETS = (100, 500, 2_000, 10_000)


class FixedNgramProbeAdapter:
    """Input-only unigram and training-corpus trigram features."""

    name = "fixed_char_ngram"

    def __init__(self, trigram_counts: np.ndarray):
        self.counts = np.asarray(trigram_counts)
        self.vocab_size = int(self.counts.shape[-1])

    def node_names(self) -> tuple[str, ...]:
        return ("current_char", "trigram_distribution")

    def activations(
        self,
        inputs: np.ndarray,
        nodes,
    ) -> dict[str, np.ndarray]:
        inputs = np.asarray(inputs, dtype=np.int32)
        requested = tuple(nodes)

        unknown = set(requested) - set(self.node_names())
        if unknown:
            raise KeyError(f"unknown control features: {sorted(unknown)}")

        previous = inputs[:, -2]
        current = inputs[:, -1]

        eye = np.eye(self.vocab_size, dtype=np.float32)
        current_char = eye[current]

        local_counts = self.counts[previous, current].astype(np.float32)
        trigram_distribution = (local_counts + 1.0) / (
            local_counts.sum(axis=1, keepdims=True) + self.vocab_size
        )

        available = {
            "current_char": current_char,
            "trigram_distribution": trigram_distribution,
        }
        return {node: available[node] for node in requested}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("artifacts/interpretation/char_v1"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runs/pcinterp/probes/fixed_char_ngram"),
    )
    args = parser.parse_args()

    dataset = load_frozen_dataset(args.dataset_dir)

    train_loader = CharDataLoader(
        "train",
        seq_len=dataset.inputs.shape[1],
        batch_size=1,
        shuffle=False,
    )
    counts = fit_trigram_counts(
        np.asarray(train_loader.data, dtype=np.int32),
        train_loader.vocab_size,
    )

    result = probe_study.run(
        FixedNgramProbeAdapter(counts),
        dataset,
        args.output_dir,
        budgets=PROBE_BUDGETS,
        repeats=5,
        seed=0,
        auroc_minimum=0.60,
        selectivity_minimum=0.05,
        positive_minimum=200,
    )

    for row in result["summary"]:
        verdict = "PASS" if row["gate_passed"] else "FAIL"
        print(
            f"{row['node']:22} {row['factor']:22} "
            f"AUROC={row['auroc']:.4f} "
            f"shuffled={row['shuffled_auroc']:.4f} "
            f"selectivity={row['selectivity']:+.4f} "
            f"AUSC={row['ausc']:.4f} gate={verdict}"
        )


if __name__ == "__main__":
    main()
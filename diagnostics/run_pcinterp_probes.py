"""Run the frozen pcinterp probe study for one Transformer V2 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from pcinterp.studies import probe_study

from diagnostics.pcinterp_adapter import (
    FabricPCTransformerV2Adapter,
    load_frozen_dataset,
)


PROBE_BUDGETS = (100, 500, 2_000, 10_000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    dataset = load_frozen_dataset(args.dataset_dir)
    adapter = FabricPCTransformerV2Adapter(
        args.checkpoint,
        name=args.name,
        batch_size=args.batch_size,
    )

    result = probe_study.run(
        adapter,
        dataset,
        args.output_dir,
        budgets=PROBE_BUDGETS,
        repeats=5,
        seed=0,
        auroc_minimum=0.60,
        selectivity_minimum=0.05,
        positive_minimum=200,
    )

    print(f"\nProbe summary: {args.name}")
    for row in result["summary"]:
        verdict = "PASS" if row["gate_passed"] else "FAIL"
        print(
            f"{row['node']:8} {row['factor']:22} "
            f"AUROC={row['auroc']:.4f} "
            f"shuffled={row['shuffled_auroc']:.4f} "
            f"selectivity={row['selectivity']:+.4f} "
            f"AUSC={row['ausc']:.4f} gate={verdict}"
        )

    print("\nOutputs:")
    for output in result["outputs"]:
        print(output)


if __name__ == "__main__":
    main()
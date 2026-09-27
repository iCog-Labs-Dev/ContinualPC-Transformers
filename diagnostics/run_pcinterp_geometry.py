"""Compare final BP, PC-FP, and shared-initialization representation geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

from pcinterp.studies import geometry_study

from diagnostics.pcinterp_adapter import (
    FabricPCTransformerV2Adapter,
    load_frozen_dataset,
)


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
        default=Path("runs/pcinterp/geometry/final_seed0"),
    )
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    dataset = load_frozen_dataset(args.dataset_dir)

    adapters = [
        FabricPCTransformerV2Adapter(
            "runs/checkpoints/ckpt_bp_char_seed0_init.pkl",
            name="shared_init_seed0",
            batch_size=args.batch_size,
        ),
        FabricPCTransformerV2Adapter(
            "runs/checkpoints/ckpt_bp_char_seed0_epoch5.pkl",
            name="bp_seed0_epoch5",
            batch_size=args.batch_size,
        ),
        FabricPCTransformerV2Adapter(
            "runs/checkpoints/ckpt_pc_fp_T20_eta0.3864_char_seed0_epoch5.pkl",
            name="pc_fp_seed0_epoch5",
            batch_size=args.batch_size,
        ),
    ]

    result = geometry_study.run(
        adapters,
        dataset,
        args.output_dir,
        top_eigenvalues=16,
    )

    print("\nCKA:")
    for row in result["cka"]:
        print(
            f"{row['node']:8} "
            f"{row['left']:20} vs {row['right']:20} "
            f"CKA={row['cka']:.4f}"
        )

    print("\nEffective ranks:")
    for row in result["rank"]:
        print(
            f"{row['node']:8} {row['model']:20} "
            f"participation={row['participation_rank']:.2f} "
            f"entropy={row['entropy_rank']:.2f} "
            f"variance={row['total_variance']:.4g}"
        )

    print("\nOutputs:")
    for output in result["outputs"]:
        print(output)


if __name__ == "__main__":
    main()
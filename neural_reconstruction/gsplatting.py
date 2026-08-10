"""Gaussian splat training entry point.

This module used to probe for a `gsplat.train()` function and a `gsplat` console
script, neither of which exists, so it could only ever raise `EnvironmentError`
(MPO-223). It now delegates to the real trainer in `neural_reconstruction.trainer`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from neural_reconstruction.trainer import (  # noqa: E402
    TrainConfig,
    TrainResult,
    train_from_colmap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train(
    images_dir: Path | str,
    colmap_dir: Path | str,
    out_dir: Path | str,
    max_steps: int = 7_000,
    max_image_side: int = 1_600,
    sh_degree: int = 3,
    device: Optional[str] = None,
) -> TrainResult:
    """Train a splat model. Kept as a thin, stable wrapper over the trainer."""
    cfg = TrainConfig(
        max_steps=max_steps, max_image_side=max_image_side, sh_degree=sh_degree
    )
    return train_from_colmap(colmap_dir, images_dir, out_dir, config=cfg, device=device)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train a gaussian splatting model from frames + a COLMAP model."
    )
    # No hardcoded absolute paths (MPO-229) — these are repo-relative defaults.
    p.add_argument("--images", type=Path, default=Path("frames"))
    p.add_argument(
        "--colmap-out", "--colmap_out", dest="colmap_out", type=Path,
        default=Path("colmap"),
        help="COLMAP model dir, or any ancestor of it.",
    )
    p.add_argument("--out", type=Path, default=Path("splat"))
    p.add_argument("--max-steps", type=int, default=7_000)
    p.add_argument("--max-image-side", type=int, default=1_600,
                   help="Downscale training images so they fit in VRAM.")
    p.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3])
    p.add_argument("--device", default=None, help="Defaults to cuda when available.")
    args = p.parse_args()

    result = train(
        images_dir=args.images,
        colmap_dir=args.colmap_out,
        out_dir=args.out,
        max_steps=args.max_steps,
        max_image_side=args.max_image_side,
        sh_degree=args.sh_degree,
        device=args.device,
    )
    logging.info(
        "Training finished: %d steps, final loss %.5f, %d gaussians -> %s",
        result.steps, result.final_loss, result.n_gaussians, result.ply_path,
    )

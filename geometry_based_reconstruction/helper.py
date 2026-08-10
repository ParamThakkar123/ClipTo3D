"""Diagnostic: compare COLMAP-registered image names against a frames directory.

Useful when reconstruction silently drops most frames — usually a filename or
subdirectory mismatch between what COLMAP recorded and what is on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

_workspace_root = Path(__file__).resolve().parents[1]
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from colmap_io import find_model_dir, read_images_text  # noqa: E402
from structure_from_motion.sfm import list_frames  # noqa: E402


def compare(model_dir: Path, frames_dir: Path, n_examples: int = 10) -> None:
    model_dir = find_model_dir(model_dir)
    images = read_images_text(model_dir / "images.txt")
    model_names = sorted(img.name for img in images.values())
    model_basenames = {Path(n).name for n in model_names}

    frames = list_frames(frames_dir)
    frame_names = [p.name for p in frames]
    missing = [f for f in frame_names if f not in model_basenames]

    print(f"Model dir:      {model_dir}")
    print(f"Model entries:  {len(model_names)}")
    print(f"Frames found:   {len(frames)}")
    print(f"Unregistered:   {len(missing)}")
    print(f"Example model entries: {model_names[:n_examples]}")
    print(f"Example frame filenames: {frame_names[:n_examples]}")
    if missing:
        print(f"Example unregistered frames: {missing[:n_examples]}")
        print(
            "\nIf nearly all frames are unregistered, the names differ (full paths, "
            "different extensions, or a nested model directory). If only some are, "
            "COLMAP failed to register those views — usually too little overlap or "
            "motion blur."
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-dir", "--images-txt",
        dest="model_dir", type=Path,
        default=Path("structure_from_motion/colmap_output/sparse/model_txt"),
        help="COLMAP model dir, or any ancestor of it.",
    )
    p.add_argument("--frames", type=Path, default=Path("frames"))
    args = p.parse_args()
    compare(args.model_dir, args.frames)

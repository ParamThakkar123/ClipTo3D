"""Convert a COLMAP text model into a nerf-style `transforms.json` dataset.

Parsing lives in `colmap_io` (see that module for the two bugs this used to
have). What remains here is purely the dataset layout: pose convention, image
placement, and intrinsics emission.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from colmap_io import Camera, find_model_dir, read_model

# COLMAP is +X right, +Y down, +Z forward. The nerf/gsplat `transform_matrix`
# convention is +X right, +Y up, +Z back, so the camera axes are flipped in Y
# and Z. Applied to the camera-to-world matrix, not the world points.
_COLMAP_TO_NERF = np.diag([1.0, -1.0, -1.0, 1.0])


def _intrinsics_dict(cam: Camera) -> Dict[str, Any]:
    fx, fy, cx, cy = cam.intrinsics
    return {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": cam.width,
        "h": cam.height,
        "camera_angle_x": float(2.0 * np.arctan(cam.width / (2.0 * fx))),
        "camera_angle_y": float(2.0 * np.arctan(cam.height / (2.0 * fy))),
    }


def colmap_to_transforms(
    colmap_dir: Path | str,
    image_dir: Path | str,
    output_dir: Path | str,
    copy_images: bool = True,
    nerf_convention: bool = True,
) -> Path:
    """Write `<output_dir>/transforms.json` from the COLMAP model in `colmap_dir`.

    `colmap_dir` may point at the model directory itself or any ancestor of it;
    the actual model is located with `colmap_io.find_model_dir`.

    Intrinsics are emitted globally when the model has a single camera, and
    per-frame otherwise. The previous implementation read intrinsics from a
    loop variable *after* the loop finished, so a multi-camera model silently
    adopted whichever camera was visited last.
    """
    colmap_dir = Path(colmap_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    model_dir = find_model_dir(colmap_dir)
    cameras, images, _ = read_model(model_dir)
    print(f"Reading COLMAP model from {model_dir} ({len(images)} images, {len(cameras)} cameras)")

    images_out = output_dir / "images"
    if copy_images:
        images_out.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    missing = []
    # Sort by filename so the dataset order is deterministic and matches the
    # frame sequence, rather than following COLMAP's internal image ids.
    for img in sorted(images.values(), key=lambda im: im.name):
        src = image_dir / Path(img.name).name
        if not src.exists():
            missing.append(img.name)
            continue

        c2w = img.camera_to_world_matrix()
        if nerf_convention:
            c2w = c2w @ _COLMAP_TO_NERF

        if copy_images:
            shutil.copy2(src, images_out / src.name)
            file_path = f"images/{src.name}"
        else:
            # Relative to output_dir so the dataset stays relocatable together
            # with the frames directory.
            file_path = Path(os.path.relpath(src.resolve(), output_dir.resolve())).as_posix()

        frame: Dict[str, Any] = {
            "image_id": img.id,
            "file_path": file_path,
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": img.id,
        }
        if len(cameras) > 1:
            frame.update(_intrinsics_dict(cameras[img.camera_id]))
        frames.append(frame)

    if missing:
        print(f"WARNING: {len(missing)} model images missing from {image_dir}, e.g. {missing[:3]}")
    if not frames:
        raise RuntimeError(
            f"No model image was found in {image_dir}. The COLMAP model references "
            f"names like {list(images.values())[0].name!r}."
        )

    out: Dict[str, Any] = {}
    if len(cameras) == 1:
        out.update(_intrinsics_dict(next(iter(cameras.values()))))
    else:
        print(f"Multi-camera model ({len(cameras)} cameras): emitting per-frame intrinsics.")
    out["colmap_model_dir"] = str(model_dir)
    out["frames"] = frames

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "transforms.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(frames)} frames to {out_path}")
    return out_path


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert COLMAP outputs to a nerf-style transforms.json dataset."
    )
    parser.add_argument(
        "--colmap_dir",
        type=Path,
        required=True,
        help="COLMAP model dir, or any ancestor of it (nested model dirs are searched).",
    )
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory of source frames.")
    parser.add_argument("--output_dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--no_copy_images",
        dest="copy_images",
        action="store_false",
        help="Reference frames in place instead of copying them into the dataset.",
    )
    parser.add_argument(
        "--colmap_convention",
        dest="nerf_convention",
        action="store_false",
        help="Emit raw COLMAP camera axes instead of the nerf/gsplat convention.",
    )
    args = parser.parse_args(argv)

    colmap_to_transforms(
        args.colmap_dir,
        args.image_dir,
        args.output_dir,
        copy_images=args.copy_images,
        nerf_convention=args.nerf_convention,
    )


if __name__ == "__main__":
    main()

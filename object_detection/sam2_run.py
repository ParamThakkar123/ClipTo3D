"""SAM2 automatic mask generation over extracted frames.

OPT-IN STAGE. This is deliberately not part of the default reconstruction
pipeline (see `pipeline.py`), for two reasons:

1. Nothing downstream consumes `sam2_detections.json` today. It is write-only
   output.
2. It is the most expensive per-frame stage in the repo — the *large* checkpoint
   run through the automatic mask generator on every frame.

Two plausible futures were identified for it (MPO-230): masking dynamic objects
out before SfM so they stop corrupting poses, or segmenting the final cloud so
the viewer can isolate objects. Either would want something cheaper than
whole-frame auto-masking, so the stage is kept working and reachable but is no
longer implied to be part of a normal run.

Install with the extra:  uv sync --extra sam2
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

MAX_SIDE = 1600  # cap on the longest image side, to bound peak memory


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_generator(sam2_config: str, sam2_checkpoint: str, device: torch.device):
    """Import sam2 lazily so this module stays importable without the extra."""
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore[import-not-found]
        from sam2.build_sam import build_sam2  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "The SAM2 stage requires the optional 'sam2' dependency, which is not "
            "part of the default install because nothing downstream consumes its "
            "output yet. Install it with:  uv sync --extra sam2"
        ) from e

    model = build_sam2(sam2_config, sam2_checkpoint, device=device, apply_postprocessing=False)
    return SAM2AutomaticMaskGenerator(model)


def _detections_from_masks(masks_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert SAM2 mask dicts to compact bbox/score/area records."""
    detections: List[Dict[str, Any]] = []
    for i, m in enumerate(masks_list):
        seg = m.get("segmentation")
        if seg is None:
            continue
        seg_arr = np.asarray(seg, dtype=bool)
        if not seg_arr.any():
            continue
        ys, xs = seg_arr.nonzero()
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        score = float(m.get("predicted_iou") or m.get("stability_score") or 0.0)
        detections.append(
            {
                "id": i,
                "bbox": [x1, y1, x2 - x1 + 1, y2 - y1 + 1],  # x, y, w, h
                "score": score,
                "mask_pixels": int(seg_arr.sum()),
            }
        )
    return detections


def run_on_frames(
    frames_dir: str = "frames",
    output_json: str = "sam2_detections.json",
    max_images: Optional[int] = None,
    sam2_config: Optional[str] = None,
    sam2_checkpoint: Optional[str] = None,
    cache_dir: Optional[str] = None,
    save_masks: bool = False,
) -> List[Dict[str, Any]]:
    """Run SAM2 automatic mask generation over every image in `frames_dir`.

    Config/checkpoint resolution order: explicit argument, then the
    SAM2_CONFIG / SAM2_CHECKPOINT environment variables, then the in-repo
    default locations.

    `cache_dir`, when given, is used for torch/HF model caches. It replaces an
    earlier heuristic that scanned every drive letter for free space and picked
    one at runtime, which made the cache location nondeterministic.
    """
    device = _device()
    here = os.path.dirname(__file__)

    sam2_config = (
        sam2_config
        or os.environ.get("SAM2_CONFIG")
        or os.path.join(here, "sam2_config", "sam2_hiera_l.yaml")
    )
    sam2_checkpoint = (
        sam2_checkpoint
        or os.environ.get("SAM2_CHECKPOINT")
        or os.path.join(here, "sam2_checkpoints", "sam2_hiera_large.pt")
    )

    if not os.path.isfile(sam2_checkpoint):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {sam2_checkpoint}. Place one in "
            f"object_detection/sam2_checkpoints/ or set SAM2_CHECKPOINT."
        )
    if not os.path.isfile(sam2_config):
        raise FileNotFoundError(
            f"SAM2 config not found: {sam2_config}. Place one in "
            f"object_detection/sam2_config/ or set SAM2_CONFIG."
        )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", cache_dir)
        os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
        print(f"Using model cache dir: {cache_dir}")

    print(f"Building SAM2 from config={sam2_config} checkpoint={sam2_checkpoint} on {device}")
    generator = _build_generator(sam2_config, sam2_checkpoint, device)

    img_paths: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        img_paths.extend(sorted(glob.glob(os.path.join(frames_dir, ext))))
    img_paths.sort()
    if max_images:
        img_paths = img_paths[:max_images]
    if not img_paths:
        raise FileNotFoundError(f"No images found in {frames_dir}")

    debug_dir = os.path.dirname(output_json) or "."
    results: List[Dict[str, Any]] = []

    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_path}: cannot open image ({e})")
            continue

        w, h = img.size
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / float(max(w, h))
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS
            )
            w, h = img.size

        try:
            masks_list = generator.generate(np.array(img))
        except Exception as e:
            print(f"Error running SAM2 on {img_path}: {e}")
            continue

        detections = _detections_from_masks(masks_list)
        results.append(
            {"image_path": img_path, "width": w, "height": h, "detections": detections}
        )
        print(f"Processed {img_path}: {len(detections)} detections.")

        if save_masks:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            for mi, m in enumerate(masks_list[:3]):
                seg = np.asarray(m.get("segmentation"), dtype=np.uint8) * 255
                out = os.path.join(debug_dir, f"{stem}_mask_{mi}.png")
                Image.fromarray(seg).convert("L").save(out)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved detection results to {output_json}")
    print(
        "NOTE: no pipeline stage currently reads this file. See the module "
        "docstring for context."
    )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run SAM2 automatic mask generation on extracted frames (opt-in stage)."
    )
    p.add_argument("--frames_dir", default="frames")
    p.add_argument("--output_json", default="sam2_detections.json")
    p.add_argument("--max_images", type=int)
    p.add_argument("--sam2_config", help="Path to sam2 yaml config.")
    p.add_argument("--sam2_checkpoint", help="Path to sam2 .pt checkpoint.")
    p.add_argument("--cache_dir", help="Directory for torch/HF model caches.")
    p.add_argument("--save_masks", action="store_true", help="Dump first few masks per frame.")
    args = p.parse_args()

    run_on_frames(
        frames_dir=args.frames_dir,
        output_json=args.output_json,
        max_images=args.max_images,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        cache_dir=args.cache_dir,
        save_masks=args.save_masks,
    )

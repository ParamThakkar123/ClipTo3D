import os
import json
import glob
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image
import torch
import numpy as np
import argparse
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import sys
from dotenv import load_dotenv
import shutil

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _find_mask_tensors(outputs: Any) -> List[torch.Tensor]:
    masks: List[torch.Tensor] = []
    if hasattr(outputs, "items"):
        items = outputs.items()
    else:
        try:
            items = [(k, getattr(outputs, k)) for k in dir(outputs) if not k.startswith("_")]
        except Exception:
            items = []

    for k, v in items:
        if isinstance(v, torch.Tensor) and v.ndim == 4:
            masks.append(v)
    return masks

def masks_to_detections(masks: torch.Tensor, orig_size: Tuple[int, int], threshold: float = 0.5) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    _, n, h, w = masks.shape
    masks_up = F.interpolate(masks, size=orig_size, mode="bilinear", align_corners=False)
    masks_up = masks_up.squeeze(0)
    for i in range(n):
        mask_prob = masks_up[i].cpu()
        bin_mask = (mask_prob >= threshold).numpy().astype("uint8")
        if bin_mask.sum() == 0:
            continue
        ys, xs = bin_mask.nonzero()
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        detection = {
            "id": i,
            "bbox": [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)],  # x, y, w, h
            "score": float(mask_prob.mean().item()),
            "mask_pixels": int(bin_mask.sum()),
        }
        detections.append(detection)
    return detections

def select_disk(min_free_bytes: int = 10 * 1024**3) -> str:
    """
    Return a path on a disk that has at least `min_free_bytes` free.
    Prefer largest free disks; fallback to cwd.
    """
    candidates = []
    try:
        if os.name == "nt":
            # check common drive letters
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    try:
                        total, used, free = shutil.disk_usage(drive)
                        candidates.append((free, drive))
                    except Exception:
                        continue
        else:
            # explicit list of common mount points (use list, not set)
            paths = ["/", os.path.expanduser("~"), "/mnt", "/media"]
            for base in paths:
                # only consider existing directories
                if os.path.isdir(base):
                    try:
                        total, used, free = shutil.disk_usage(base)
                        candidates.append((free, base))
                    except Exception:
                        pass
                    # scan subdirs under /mnt and /media if they exist
                    if base in ("/mnt", "/media"):
                        try:
                            for entry in os.listdir(base):
                                p = os.path.join(base, entry)
                                if os.path.isdir(p):
                                    try:
                                        total, used, free = shutil.disk_usage(p)
                                        candidates.append((free, p))
                                    except Exception:
                                        pass
                        except Exception:
                            pass
    except Exception:
        pass

    # pick largest free that meets threshold
    candidates.sort(reverse=True, key=lambda x: x[0])
    for free, path in candidates:
        if free >= min_free_bytes:
            return path

    # fallback to current working directory
    return os.getcwd()

def run_on_frames(
    frames_dir: str = "frames",
    output_json: str = "sam2_detections.json",
    threshold: float = 0.2,
    max_images: Optional[int] = None,
    save_first: bool = True,
    sam2_config: Optional[str] = None,
    sam2_checkpoint: Optional[str] = None,
    debug: bool = False,
    min_space_gb: int = 10,
) -> List[Dict[str, Any]]:
    """
    Process images in `frames_dir` using the external `sam2` package.
    If sam2_config / sam2_checkpoint are not provided they are taken from:
      - environment variables SAM2_CONFIG / SAM2_CHECKPOINT
      - local workspace defaults: object_detection/sam2_config/* and object_detection/sam2_checkpoints/*
    The function will prefer a disk with at least `min_space_gb` GB free for model building/cache.
    """
    device = _device()

    # prefer args -> env -> workspace defaults
    sam2_config = sam2_config or os.environ.get("SAM2_CONFIG") or os.path.join(os.path.dirname(__file__), "sam2_config", "sam2_hiera_l.yaml")
    sam2_checkpoint = sam2_checkpoint or os.environ.get("SAM2_CHECKPOINT") or os.path.join(os.path.dirname(__file__), "sam2_checkpoints", "sam2_hiera_large.pt")

    if not os.path.isfile(sam2_checkpoint):
        raise RuntimeError(f"SAM2 checkpoint not found: {sam2_checkpoint}. Place a checkpoint in [object_detection/sam2_checkpoints](object_detection/sam2_checkpoints) or set SAM2_CHECKPOINT env var.")
    if not os.path.isfile(sam2_config):
        raise RuntimeError(f"SAM2 config not found: {sam2_config}. Place a config in [object_detection/sam2_config](object_detection/sam2_config) or set SAM2_CONFIG env var.")

    # choose disk for model building / cache
    min_free_bytes = int(min_space_gb) * 1024 ** 3
    chosen_root = select_disk(min_free_bytes=min_free_bytes)
    cache_dir = os.path.join(chosen_root, "cliptoworld_cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        # prefer existing settings but set defaults so heavy IO (extracted weights/caches) go to chosen disk
        os.environ.setdefault("TORCH_HOME", cache_dir)
        os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
        os.environ.setdefault("TMPDIR", cache_dir)
        print(f"Using cache dir {cache_dir} on disk {chosen_root} for model building (min {min_space_gb} GB)")
    except Exception as e:
        print(f"Warning: failed to prepare cache dir {cache_dir}: {e}")

    try:
        print(f"Building SAM2 model from config={sam2_config} checkpoint={sam2_checkpoint}")
        sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device, apply_postprocessing=False)
        sam2_generator = SAM2AutomaticMaskGenerator(sam2_model)
        use_sam2 = True
    except Exception as e:
        raise RuntimeError(f"Failed to build SAM2 model: {e}")
    
    patterns = [os.path.join(frames_dir, "*.png"), os.path.join(frames_dir, "*.jpg"), os.path.join(frames_dir, "*.jpeg")]
    img_paths: List[str] = []
    for p in patterns:
        img_paths.extend(sorted(glob.glob(p)))
    if max_images:
        img_paths = img_paths[:max_images]

    results: List[Dict[str, Any]] = []
    MAX_SIDE = 1600  # max width/height to avoid huge images; tune as needed
    first_saved = False
    first_out_path = os.path.join(
        os.path.dirname(output_json) or ".",
        f"{os.path.splitext(os.path.basename(output_json))[0]}_first_processed.jpg",
    )
    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_path}: cannot open image ({e})")
            continue
        orig_w, orig_h = img.size
        # downscale very large images to avoid MemoryError
        if max(orig_w, orig_h) > MAX_SIDE:
            scale = MAX_SIDE / float(max(orig_w, orig_h))
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(f"Resized {os.path.basename(img_path)}: {orig_w}x{orig_h} -> {new_w}x{new_h}")
            orig_w, orig_h = img.size

        detections: List[Dict[str, Any]] = []
        if use_sam2 and sam2_generator is not None:
            try:
                img_np = np.array(img)
                masks_list = sam2_generator.generate(img_np)
                for i, m in enumerate(masks_list):
                    seg = m.get("segmentation")
                    if seg is None:
                        continue
                    seg_arr = np.asarray(seg, dtype=bool)
                    if seg_arr.sum() == 0:
                        continue
                    ys, xs = seg_arr.nonzero()
                    y1, y2 = int(ys.min()), int(ys.max())
                    x1, x2 = int(xs.min()), int(xs.max())
                    score = float(m.get("predicted_iou") or m.get("stability_score") or 0.0)
                    detections.append({
                        "id": i,
                        "bbox": [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)],
                        "score": score,
                        "mask_pixels": int(seg_arr.sum()),
                    })
                if debug or (save_first and not first_saved):
                    try:
                        debug_dir = os.path.dirname(output_json) or "."
                        for mi, m in enumerate(masks_list[:3]):
                            seg = np.asarray(m.get("segmentation"), dtype=np.uint8) * 255
                            mask_path = os.path.join(debug_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_mask_{mi}.jpg")
                            Image.fromarray(seg).convert("L").save(mask_path, format="JPEG", quality=95)
                            if debug:
                                print(f"DEBUG: saved mask {mi} -> {mask_path}")
                    except Exception as e:
                        print(f"Warning: failed to save debug masks (sam2): {e}")
            except Exception as e:
                print(f"Error running sam2 AutomaticMaskGenerator on {img_path}: {e}")
                continue
        else:
            raise RuntimeError("SAM2 AutomaticMaskGenerator not initialized.")

        results.append({
            "image_path": img_path,
            "width": orig_w,
            "height": orig_h,
            "detections": detections
        })
        print(f"Processed {img_path}: found {len(detections)} detections.")
        if save_first and not first_saved:
            try:
                img.save(first_out_path, format="JPEG", quality=95)
                print(f"Saved first processed image to {first_out_path}")
                first_saved = True
            except Exception as e:
                print(f"Warning: failed to save first processed image: {e}")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Saved detection results to {output_json}")
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run SAM2 object detection on extracted frames.")
    p.add_argument("--frames_dir", default="frames", help="Directory containing extracted frames.")
    p.add_argument("--output_json", default="sam2_detections.json", help="Output JSON file for detections.")
    p.add_argument("--threshold", type=float, default=0.5, help="Mask threshold for detections.")
    p.add_argument("--max_images", type=int, help="Maximum number of images to process.")
    p.add_argument("--no_save_first", action="store_true", help="Do not save the first processed image.")
    p.add_argument("--debug", action="store_true", help="Print debug info and save mask PNGs for the first images.")
    p.add_argument("--sam2_config", help="Path to sam2 yaml config (overrides env/workspace default).")
    p.add_argument("--sam2_checkpoint", help="Path to sam2 checkpoint .pt (overrides env/workspace default).")
    args = p.parse_args()

    run_on_frames(
        frames_dir=args.frames_dir,
        output_json=args.output_json,
        threshold=args.threshold,
        max_images=args.max_images,
        save_first=not args.no_save_first,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        debug=args.debug,
    )
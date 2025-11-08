from pathlib import Path
from PIL import Image
import numpy as np
import torch
import cv2
import sys

DepthAnythingV2 = None
try:
    from .depth_anything_v2.dpt import DepthAnythingV2
except Exception:
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except Exception:
        local_pkg = Path(__file__).resolve().parents[0] / "depth_anything_v2"
        if local_pkg.exists():
            sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
            try:
                from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
            except Exception:
                DepthAnythingV2 = None
        else:
            DepthAnythingV2 = None

def estimate_depths_midas(
        frames_dir: str = "./frames",
        out_dir: str = "./depth_maps",
        model_type: str = "DPT_Large",
        use_cuda: bool = True,
):
    """
    Estimate depth maps for all images in `frames_dir` and save them (PNG + .npy)
    into `out_dir`.

    - frames_dir is relative to this script (recommended: project-root/frames).
    - out_dir will be created if missing (project-root/depth_maps).
    - model_type: "MiDaS" (full) or "MiDaS_small".
    """
    frames_path =  Path(__file__).resolve().parents[1] / Path(frames_dir).relative_to(".")
    out_path = Path(__file__).resolve().parents[1] / Path(out_dir).relative_to(".")

    frames_path = frames_path.resolve()
    out_path = out_path.resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "MiDaS_small":
        transform = midas_transform.small_transform
    else:
        transform = midas_transform.dpt_transform

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    for img_file in sorted(frames_path.iterdir()):
        if not img_file.is_file() or img_file.suffix.lower() not in exts:
            continue

        img = Image.open(img_file).convert("RGB")
        orig_w, orig_h = img.size

        img_np = np.asarray(img)
        transformed = transform(img_np)
        if isinstance(transformed, dict):
            input_batch = transformed.get("image")
        else:
            input_batch = transformed
        input_batch = input_batch.to(device)
        if input_batch.dim() == 3:
            model_input = input_batch.unsqueeze(0)
        else:
            model_input = input_batch
        with torch.no_grad():
            prediction = midas(model_input)
            if prediction.dim() == 3:
                pred = prediction.squeeze(0)
            else:
                pred = prediction.squeeze()
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            depth = pred.cpu().numpy()

        depth_min, depth_max = depth.min(), depth.max()
        if depth_max - depth_min > 1e-6:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = depth * 0.0
        depth_img = (depth_norm * 255).astype(np.uint8)

        png_name = out_path / f"{img_file.stem}_depth.png"
        npy_name = out_path / f"{img_file.stem}_depth.npy"

        Image.fromarray(depth_img).save(png_name)
        np.save(npy_name, depth.astype(np.float32))

    print(f"Depth maps saved to: {out_path}")


def estimate_depths(
        frames_dir: str = "./frames",
        out_dir: str = "./depth_maps",
        model_backend: str = "midas",              
        model_type: str = "DPT_Large",           
        depthanything_ckpt: str | None = None,     
        use_cuda: bool = True,
):
    """
    Wrapper that supports multiple backends. Saves PNG and .npy depth maps to out_dir.
    """
    frames_path =  Path(__file__).resolve().parents[1] / Path(frames_dir).relative_to(".")
    out_path = Path(__file__).resolve().parents[1] / Path(out_dir).relative_to(".")

    frames_path = frames_path.resolve()
    out_path = out_path.resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    backend = model_backend.lower()
    if backend == "midas":
        # reuse existing midas implementation
        return estimate_depths_midas(frames_dir=frames_dir, out_dir=out_dir, model_type=model_type, use_cuda=use_cuda)

    elif backend == "depthanythingv2":
        # instantiate DepthAnythingV2 with default args (user can modify if needed)
        model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])

        if depthanything_ckpt is None:
            raise ValueError("depthanything_ckpt must be provided for depthanythingv2 backend. "
                             "Provide path to .pth checkpoint (e.g. './depth_estimation/depth_anything_ckpt/checkpoints/depth_anything_v2_vitl.pth').")

        # load checkpoint (map to CPU/GPU depending on device)
        ckpt_map = "cpu" if device.type == "cpu" else None
        state = torch.load(str(depthanything_ckpt), map_location=ckpt_map)
        # If checkpoint was saved as a dict with 'state_dict' key, handle it
        if isinstance(state, dict) and "state_dict" in state and not any(k.startswith('module.') for k in state.keys()):
            state = state["state_dict"]
        # Allow for module prefix variations
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        for img_file in sorted(frames_path.iterdir()):
            if not img_file.is_file() or img_file.suffix.lower() not in exts:
                continue

            raw_bgr = cv2.imread(str(img_file))
            if raw_bgr is None:
                print(f"Warning: failed to read {img_file}, skipping.")
                continue

            with torch.no_grad():
                depth = model.infer_image(raw_bgr)

            depth = np.array(depth).squeeze()
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth[:, :, 0]

            orig_h, orig_w = raw_bgr.shape[:2]
            if (depth.shape[0], depth.shape[1]) != (orig_h, orig_w):
                depth = cv2.resize(depth.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            depth_min, depth_max = depth.min(), depth.max()
            if depth_max - depth_min > 1e-6:
                depth_norm = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = depth * 0.0
            depth_img = (depth_norm * 255).astype(np.uint8)

            png_name = out_path / f"{img_file.stem}_depth.png"
            npy_name = out_path / f"{img_file.stem}_depth.npy"

            Image.fromarray(depth_img).save(png_name)
            np.save(npy_name, depth.astype(np.float32))

        print(f"Depth maps saved to: {out_path}")

    else:
        raise ValueError("Unsupported model_backend. Use 'midas' or 'depthanythingv2'.")


if __name__ == "__main__":
    # estimate_depths_midas(frames_dir="./frames", out_dir="./depth_maps", model_type="DPT_Large", use_cuda=True)

    estimate_depths(frames_dir="./frames", out_dir="./depth_maps",
                    model_backend="depthanythingv2",
                    depthanything_ckpt="./depth_estimation/depth_anything_ckpt/checkpoints/depth_anything_v2_vitl.pth",
                    use_cuda=True)
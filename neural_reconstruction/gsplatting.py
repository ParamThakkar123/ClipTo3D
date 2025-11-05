import subprocess
import shutil
import json
from pathlib import Path
import logging
import sys
from typing import Optional

# ensure workspace root is on sys.path so package-style imports work when running the script directly
try:
    from structure_from_motion.sfm import list_frames
except Exception:
    _workspace_root = Path(__file__).resolve().parents[1]
    if str(_workspace_root) not in sys.path:
        sys.path.insert(0, str(_workspace_root))
    from structure_from_motion.sfm import list_frames

logging.basicConfig(level=logging.INFO)


def find_gsplat_bin(name: str = "gsplat") -> Optional[str]:
    p = shutil.which(name)
    if p:
        logging.info(f"Found gsplat binary at {p}")
        return p
    logging.debug("gsplat binary not found in PATH.")
    return None


def create_default_config(out_dir: Path, images_dir: Path, colmap_model_txt: Path, depth_dir: Optional[Path] = None):
    cfg = {
        "images": str(images_dir),
        "colmap_model": str(colmap_model_txt),
        "output_dir": str(out_dir),
        "num_steps": 20000,
        "lr_init": 0.01,
        "batch_size": 4096,
        "depth_maps": str(depth_dir) if depth_dir else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "gsplat_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)
    logging.info(f"Created default gsplat config at {cfg_path}")
    return cfg_path


def run_gsplat_train_py(cfg_path: Path) -> bool:
    """
    Import installed gsplat module and call its Python API. Return True on success.
    """
    try:
        import gsplat  # imported here so script doesn't require gsplat at module import time
    except Exception as e:
        logging.debug("gsplat module import failed: %s", e)
        return False

    try:
        with cfg_path.open() as f:
            cfg = json.load(f)
        if hasattr(gsplat, "train"):
            logging.info("Launching gsplat.train(...) via Python API")
            gsplat.train(cfg)  # type: ignore
            return True
        logging.warning("gsplat module found but no train() entrypoint detected.")
        return False
    except Exception as e:
        logging.error("Error running gsplat Python API: %s", e)
        return False


def run_gsplat_train_cli(gsplat_bin: str, cfg_path: Path) -> None:
    cmd = [gsplat_bin, "train", "--config", str(cfg_path)]
    logging.info("Running gsplat CLI: %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def train_from_colmap(
        images_dir: Path,
        colmap_model_txt: Path,
        out_dir: Path,
        depth_maps_dir: Optional[Path] = None,
):
    images_dir = images_dir.resolve()
    colmap_model_txt = colmap_model_txt.resolve()
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory {images_dir} does not exist.")
    if not colmap_model_txt.exists() or not colmap_model_txt.is_dir():
        raise FileNotFoundError(f"COLMAP model directory {colmap_model_txt} does not exist.")

    out_dir = out_dir.resolve()
    cfg_path = create_default_config(out_dir, images_dir, colmap_model_txt, depth_maps_dir)

    # Prefer Python API (requires gsplat installed in this interpreter)
    if run_gsplat_train_py(cfg_path):
        logging.info(f"gsplat training finished. Results in {out_dir}")
        return

    # Fallback to CLI if available
    gsplat_bin = find_gsplat_bin()
    if gsplat_bin:
        run_gsplat_train_cli(gsplat_bin, cfg_path)
        return

    raise EnvironmentError("gsplat not available (no Python module and no gsplat CLI). Install gsplat in this environment or add gsplat CLI to PATH.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train gaussian splatting model from frames + COLMAP output")
    parser.add_argument("--images", type=Path, default=Path("frames"), help="frames directory (images)")
    parser.add_argument("--colmap-out", type=Path, default=Path(r"E:\ClipToWorld\structure_from_motion\colmap_output"), help="COLMAP output directory (contains sparse/model_txt)")
    parser.add_argument("--out", type=Path, default=Path("gsplat_output"), help="training output dir")
    parser.add_argument("--depth-maps", type=Path, default=Path("depth_maps"), help="optional precomputed depth maps")
    args = parser.parse_args()

    imgs = list_frames(args.images)  # from structure_from_motion.sfm
    logging.info("Found %d images under %s", len(imgs), args.images)

    train_from_colmap(args.images, args.colmap_out, args.out, args.depth_maps if args.depth_maps.exists() else None)
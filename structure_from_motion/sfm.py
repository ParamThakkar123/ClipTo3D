import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def resolve_colmap_bin(colmap_bin: Optional[str] = None) -> str:
    """Locate a usable COLMAP binary."""
    def _is_exec(p: Path): return p.is_file() and os.access(str(p), os.X_OK)
    search = ("colmap", "colmap.exe")

    if colmap_bin:
        p = Path(colmap_bin)
        if p.is_dir():
            for s in search:
                cand = p / s
                if _is_exec(cand): 
                    return str(cand)
        if _is_exec(p): 
            return str(p)
        which = shutil.which(colmap_bin)
        if which and _is_exec(Path(which)): 
            return which
        raise FileNotFoundError(f"COLMAP binary '{colmap_bin}' not found or not executable.")

    for s in search:
        w = shutil.which(s)
        if w and _is_exec(Path(w)): 
            return w
    raise FileNotFoundError("COLMAP binary not found on PATH. Provide --colmap-bin path.")


def _run_cmd(cmd):
    """Run a command with real-time logging."""
    logging.info("▶ %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        logging.info(line.strip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def list_frames(frames_dir: Path) -> List[Path]:
    """List sorted image files."""
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sorted([p for p in Path(frames_dir).iterdir() if p.suffix in exts])


def run_colmap_fast(frames_dir: Path, out_dir: Path, colmap_bin: Optional[str] = None, use_gpu=True):
    """Fast COLMAP pipeline tuned for large video frame sets."""
    cb = resolve_colmap_bin(colmap_bin)
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = out_dir / "database.db"
    sparse = out_dir / "sparse"
    txt = sparse / "model_txt"

    feat_cmd = [
        cb, "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(frames_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
        "--SiftExtraction.max_num_features", "4096",    
        "--SiftExtraction.num_threads", str(os.cpu_count() or 8),
    ]
    _run_cmd(feat_cmd)

    match_cmd = [
        cb, "sequential_matcher",
        "--database_path", str(db),
        "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        "--SiftMatching.num_threads", str(os.cpu_count() or 8),
        "--SiftMatching.max_ratio", "0.7",
        "--SequentialMatching.overlap", "2",           
    ]
    _run_cmd(match_cmd)

    sparse.mkdir(parents=True, exist_ok=True)
    map_cmd = [
        cb, "mapper",
        "--database_path", str(db),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse),
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.tri_min_angle", "2",
        "--Mapper.abs_pose_min_num_inliers", "10",
        "--Mapper.filter_max_reproj_error", "8",
        "--Mapper.ba_global_max_refinements", "1",     
        "--Mapper.ba_local_max_refinements", "1",
        "--Mapper.extract_colors", "0",                  
        "--Mapper.num_threads", str(os.cpu_count() or 8),
    ]
    _run_cmd(map_cmd)

    txt.mkdir(parents=True, exist_ok=True)

    model_dirs = []
    if sparse.exists():
        for p in sorted(sparse.iterdir()):
            if p.is_dir():
                if (p / "cameras.bin").exists() and (p / "images.bin").exists() and (p / "points3D.bin").exists():
                    model_dirs.append(p)
        if not model_dirs and (sparse / "cameras.bin").exists():
            model_dirs = [sparse]

    if not model_dirs:
        contents = sorted([str(p) for p in sparse.iterdir()]) if sparse.exists() else []
        raise RuntimeError(
            f"No COLMAP binary model found in '{sparse}'. "
            f"Expected subfolder(s) with cameras.bin, images.bin, points3D.bin. "
            f"Directory contents: {contents}"
        )
    
    for i, model_dir in enumerate(model_dirs):
        out_txt = txt / (model_dir.name if model_dir is not sparse else f"model_{i}")
        out_txt.mkdir(parents=True, exist_ok=True)
        convert_cmd = [
            cb, "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(out_txt),
            "--output_type", "TXT"
        ]
        _run_cmd(convert_cmd)

    logging.info("✅ Fast COLMAP reconstruction done.")
    return txt


if __name__ == "__main__":
    import argparse
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Fast COLMAP reconstruction for video frames")
    parser.add_argument("--frames", type=Path, default=repo / "frames")
    parser.add_argument("--out", type=Path, default=repo / "structure_from_motion" / "colmap_output_fast")
    parser.add_argument("--colmap-bin", type=str, default=None)
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    args = parser.parse_args()

    imgs = list_frames(args.frames)
    logging.info("Found %d frames", len(imgs))
    if not imgs:
        raise SystemExit("No frames found.")

    txt_model = run_colmap_fast(args.frames, args.out, args.colmap_bin, use_gpu=args.use_gpu)
    logging.info("Model TXT output: %s", txt_model)
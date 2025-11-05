from pathlib import Path
import sys

# try to import helpers from workspace
try:
    from geometry_based_reconstruction.gbr import list_model_image_names
    from structure_from_motion.sfm import list_frames
except Exception:
    _workspace_root = Path(__file__).resolve().parents[1]
    if str(_workspace_root) not in sys.path:
        sys.path.insert(0, str(_workspace_root))
    from geometry_based_reconstruction.gbr import list_model_image_names  # type: ignore
    from structure_from_motion.sfm import list_frames  # type: ignore

def compare(images_txt: Path, frames_dir: Path, n_examples: int = 10):
    model_names = list_model_image_names(images_txt)
    frames = list_frames(frames_dir)
    model_set = set(model_names)
    frame_names = [p.name for p in frames]
    missing = [f for f in frame_names if f not in model_set and f.lower() not in model_set and Path(f).stem not in model_set]
    print(f"Model entries: {len(model_names)}; Frames found: {len(frames)}; Missing matches: {len(missing)}")
    print("Example model entries:", model_names[:n_examples])
    print("Example frame filenames:", frame_names[:n_examples])
    print("Example missing frames:", missing[:n_examples])
    if missing:
        print("\nSuggestions:")
        print(" - Check whether images.txt stores full paths, different extensions, or different stems.")
        print(" - Try using a numeric subfolder inside your model_txt (COLMAP outputs like model_txt/0).")
        print(" - Option: rename frames to match model names or generate a mapping before running reconstruction.")
        print(" - If many frames skip, consider relaxing voxel filtering: lower --voxel-size (e.g. 0.005) or --min-voxel-points 1.")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--images-txt", type=Path, default=Path("structure_from_motion/colmap_output/sparse/model_txt/images.txt"))
    p.add_argument("--frames", type=Path, default=Path("frames"))
    args = p.parse_args()
    compare(args.images_txt, args.frames)
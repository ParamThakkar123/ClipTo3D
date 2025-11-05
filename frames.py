import os
import shutil
import glob
import subprocess
from typing import List, Optional
from medal_clip import download_medal_clip
import argparse

def _ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available in the system PATH. Please install ffmpeg to proceed.")
    
def extract_frames(
        input_path: str,
        out_dir: str = "frames",
        pattern: str = "frame_%06d.jpg",
        fps: Optional[float] = None,
        interval_seconds: Optional[float] = None,
        start: Optional[float] = None,
        duration: Optional[float] = None,
        overwrite: bool = True
) -> List[str]:
    _ensure_ffmpeg_available()

    if fps is not None and interval_seconds is not None:
        raise ValueError("Specify either fps or interval_seconds, not both.")
    
    os.makedirs(out_dir, exist_ok=True)

    # ensure output filenames use JPEG extension (.jpg)
    root, ext = os.path.splitext(pattern)
    if ext.lower() not in (".jpg", ".jpeg"):
        pattern = f"{root}.jpg"
    out_path_template = os.path.join(out_dir, pattern)

    cmd = ["ffmpeg", "-hide_banner"]
    if not overwrite:
        cmd += ["-n"]
    else:
        cmd += ["-y"]

    if start is not None:
        cmd += ["-ss", str(start)]

    cmd += ["-i", input_path]

    if duration is not None:
        cmd += ["-t", str(duration)]

    filters = []
    if fps is not None:
        filters.append(f"fps={float(fps)}")
    elif interval_seconds is not None:
        filters.append(f"fps=1/{float(interval_seconds)}")

    if filters:
        cmd += ["-vf", ",".join(filters)]
    # prefer high-quality JPEG output; lower q is higher quality (1-31)
    cmd += ["-q:v", "2"]
 
    cmd += [out_path_template]
 
    subprocess.run(cmd, check=True)
 
    # report JPEG files (.jpg then .jpeg)
    matched = sorted(glob.glob(os.path.join(out_dir, "*.jpg")) + glob.glob(os.path.join(out_dir, "*.jpeg")))
    return matched

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract frames from a video file using ffmpeg.")
    p.add_argument("input", help="Path to the input video file.")
    p.add_argument("--out_dir", default="frames", help="Output directory for extracted frames.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--fps", type=float, help="Frames per second to extract.")
    group.add_argument("--every", type=float, help="Extract one frame every N seconds.")
    p.add_argument("--start", type=float, help="Start time in seconds.")
    p.add_argument("--duration", type=float, help="Duration in seconds to extract frames.")
    p.add_argument("--no_overwrite", action="store_true", help="Do not overwrite existing files.")
    args = p.parse_args()

    input_arg = args.input
    # Use the provided input path by default; if it's a URL, we'll overwrite this with the downloaded path.
    video_path = input_arg

    if input_arg.startswith("http://") or input_arg.startswith("https://"):
        print("Downloading video from URL...")
        video_path = download_medal_clip(input_arg, out_dir="temp_video")

    try:
        extract_frames(
            video_path,
            out_dir=args.out_dir,
            fps=args.fps,
            interval_seconds=args.every,
            start=args.start,
            duration=args.duration,
            overwrite=not args.no_overwrite
        )
        print(f"Frames extracted to directory: {args.out_dir}")
    except Exception as e:
        print(f"Error: {e}")
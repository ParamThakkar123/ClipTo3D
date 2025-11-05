import os
import shutil
import subprocess
from typing import Optional
from medal_clip import download_medal_clip
import argparse

def _ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available in the system PATH. Install ffmpeg to proceed.")

def extract_audio(
    input_path: str,
    out_dir: str = "audio",
    sample_rate: int = 16000,
    channels: int = 1,
    fmt: str = "wav",
    start: Optional[float] = None,
    duration: Optional[float] = None,
    overwrite: bool = True,
) -> str:
    _ensure_ffmpeg_available()

    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_name = f"{base}_sr{sample_rate}_ch{channels}.{fmt}"
    out_path = os.path.join(out_dir, out_name)

    cmd = ["ffmpeg", "-hide_banner"]
    cmd += ["-y"] if overwrite else ["-n"]

    if start is not None:
        cmd += ["-ss", str(start)]

    cmd += ["-i", input_path]

    if duration is not None:
        cmd += ["-t", str(duration)]

    # Drop video, force sample rate/channels and 16-bit PCM for WAV
    cmd += ["-vn", "-ar", str(sample_rate), "-ac", str(channels)]

    # For WAV use s16 (16-bit PCM); for other formats let ffmpeg choose defaults
    if fmt.lower() in ("wav", "wave"):
        cmd += ["-acodec", "pcm_s16le"]

    cmd += [out_path]

    subprocess.run(cmd, check=True)
    return out_path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract audio from video for ambient analysis.")
    p.add_argument("input", help="Path to input video file or Medal clip URL.")
    p.add_argument("--out_dir", default="audio", help="Output directory.")
    p.add_argument("--sr", type=int, default=16000, help="Sample rate (Hz).")
    p.add_argument("--ch", type=int, default=1, help="Number of channels (1=mono).")
    p.add_argument("--fmt", default="wav", help="Output audio format/extension.")
    p.add_argument("--start", type=float, help="Start time in seconds.")
    p.add_argument("--duration", type=float, help="Duration in seconds.")
    p.add_argument("--no_overwrite", action="store_true", help="Do not overwrite existing files.")
    args = p.parse_args()

    input_arg = args.input
    video_path = input_arg
    if input_arg.startswith("http://") or input_arg.startswith("https://"):
        print("Downloading video from URL...")
        video_path = download_medal_clip(input_arg, out_dir="temp_video")

    try:
        out = extract_audio(
            video_path,
            out_dir=args.out_dir,
            sample_rate=args.sr,
            channels=args.ch,
            fmt=args.fmt,
            start=args.start,
            duration=args.duration,
            overwrite=not args.no_overwrite,
        )
        print(f"Saved audio to: {out}")
    except Exception as e:
        print(f"Error: {e}")
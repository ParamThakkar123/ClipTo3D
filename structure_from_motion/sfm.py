import logging
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Printed when GPU SIFT was explicitly asked for and failed. COLMAP's SiftGPU
# uses CUDA when COLMAP was built with it, and OpenGL otherwise — and the
# OpenGL path needs a display, which is why a headless server used to hit a raw
# subprocess error here (MPO-232).
GPU_SIFT_HINT = (
    "COLMAP's GPU SIFT needs either a CUDA-enabled COLMAP build, or an OpenGL "
    "context (which a headless host does not have). Check `nvidia-smi` inside "
    "the container, confirm COLMAP was built with -DCUDA_ENABLED=ON, or pass "
    "use_gpu=False / --no-colmap-gpu to run SIFT on the CPU."
)


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


@lru_cache(maxsize=16)
def supported_options(colmap_bin: str, command: str) -> frozenset:
    """Dotted option names the installed COLMAP accepts for `command`.

    COLMAP renamed several options between releases — 3.12 moved
    `SiftExtraction.use_gpu` to `FeatureExtraction.use_gpu` and
    `SiftMatching.use_gpu` to `FeatureMatching.use_gpu`. Passing the wrong
    spelling is a hard failure ("unrecognised option"), so the names are
    resolved against the actual binary instead of pinned to one version.
    """
    try:
        proc = subprocess.run(
            [colmap_bin, command, "-h"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    text = (proc.stdout or "") + (proc.stderr or "")
    return frozenset(re.findall(r"--([A-Za-z0-9_]+\.[A-Za-z0-9_]+)", text))


def pick_option(supported: frozenset, *candidates: str) -> str:
    """First candidate the binary understands.

    Falls back to the first candidate when nothing matches, so COLMAP reports
    its own error rather than this silently dropping the flag.
    """
    for name in candidates:
        if name in supported:
            return name
    return candidates[0]


def detect_gpu_sift() -> Tuple[bool, str]:
    """Report whether COLMAP's GPU SIFT is worth attempting, and why.

    `use_gpu` used to default to True and was never checked, so a headless
    server went straight into a raw subprocess failure (MPO-232). This is a
    cheap pre-flight: it establishes that a driver and a device exist. It
    cannot tell whether COLMAP itself was built with CUDA — that is why the
    caller also falls back on failure rather than trusting this alone.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False, "nvidia-smi not on PATH - no NVIDIA driver visible"
    try:
        proc = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"nvidia-smi could not be run ({type(exc).__name__})"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, f"nvidia-smi exited {proc.returncode}: {detail[0] if detail else 'no output'}"
    gpus = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("GPU ")]
    if not gpus:
        return False, "nvidia-smi reported no GPUs"
    return True, f"{len(gpus)} NVIDIA GPU(s) visible"


class ColmapCancelled(RuntimeError):
    """The caller asked to stop while COLMAP was running."""


def _run_cmd(cmd, should_cancel: Optional[Callable[[], bool]] = None) -> List[str]:
    """Run a command with real-time logging, returning its captured output.

    `should_cancel` is polled once per output line. COLMAP is chatty enough
    that this is responsive in practice, and it is cooperative rather than a
    hard kill: the process is terminated between lines, then given a grace
    period before SIGKILL. Previously this blocked on stdout with no way to
    interrupt, so a runaway job ran to completion (MPO-243).
    """
    logging.info("▶ %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines: List[str] = []
    assert proc.stdout is not None  # stdout=PIPE
    for line in proc.stdout:
        line = line.rstrip()
        lines.append(line)
        logging.info(line)
        if should_cancel is not None and should_cancel():
            logging.warning("Cancellation requested; terminating COLMAP.")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise ColmapCancelled(f"cancelled during: {' '.join(cmd[:2])}")
    proc.wait()
    if proc.returncode != 0:
        # The tail matters: COLMAP reports the actual cause (no CUDA device, no
        # GL context) on stderr and then exits non-zero with nothing else.
        tail = "\n".join(lines[-10:]) or "<no output>"
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}\n{tail}")
    return lines


def _run_sift_stage(
    stage: str,
    build_cmd: Callable[[bool], List[str]],
    gpu: bool,
    gpu_explicit: bool,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> bool:
    """Run a SIFT stage, falling back to CPU when the GPU was only a guess.

    Returns the mode actually used, so a later stage does not re-attempt a GPU
    path that has already been shown not to work on this host.
    """
    try:
        _run_cmd(build_cmd(gpu), should_cancel)
        return gpu
    except ColmapCancelled:
        raise
    except RuntimeError as exc:
        if not gpu:
            raise
        if gpu_explicit:
            raise RuntimeError(f"COLMAP {stage} failed with GPU SIFT. {GPU_SIFT_HINT}") from exc
        logging.warning(
            "COLMAP %s failed with GPU SIFT; retrying on CPU. %s\nUnderlying error: %s",
            stage, GPU_SIFT_HINT, exc,
        )
        _run_cmd(build_cmd(False), should_cancel)
        return False


def list_frames(frames_dir: Path) -> List[Path]:
    """List sorted image files."""
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sorted([p for p in Path(frames_dir).iterdir() if p.suffix in exts])


def run_colmap_fast(
    frames_dir: Path,
    out_dir: Path,
    colmap_bin: Optional[str] = None,
    use_gpu: Optional[bool] = None,
    refine_intrinsics: bool = True,
    overlap: int = 10,
    loop_detection: bool = False,
    vocab_tree_path: Optional[str] = None,
    max_num_features: int = 2048,
    should_cancel: Optional[Callable[[], bool]] = None,
):
    """Fast COLMAP pipeline tuned for large video frame sets.

    `use_gpu=None` (the default) detects whether GPU SIFT is available and
    falls back to CPU if the attempt fails anyway. Passing True or False makes
    the choice binding: an explicit True that fails raises with a hint rather
    than silently costing 10x the runtime on CPU.

    `refine_intrinsics` controls bundle-adjustment refinement of focal length and
    distortion. It defaults to True, and turning it off badly distorts the
    reconstruction whenever the true intrinsics are unknown — which is always,
    for video from an arbitrary camera. COLMAP seeds focal length at
    1.2 * max(width, height) and, without refinement, is stuck with that guess.

    Measured on a synthetic scene with known ground truth (14 views, true focal
    500.0 at 640x480, so COLMAP's seed is 768):

        refine=0 -> focal 768.0 (53.6% error), camera-centre RMSE 3.17% of
                    scene span, rotation error 1.85 deg median / 7.0 deg max
        refine=1 -> focal 499.3 (0.1% error),  camera-centre RMSE 0.02% of
                    scene span, rotation error 0.11 deg median

    Matching (MPO-242):

    `overlap` was 2, which is very tight — and tighter still now that keyframe
    selection (MPO-237) deliberately widens the baseline between consecutive
    frames. It now defaults to COLMAP's own default of 10.

    `max_num_features` was 4096, which is high for video frames — they are
    lower-effective-resolution than photos and most of those features are
    redundant across a sequence. Measured on a 28-frame synthetic sequence:
    4096 -> 27.0s, 2048 -> 19.3s (**-29%**) with pose error unchanged
    (0.033% vs 0.031% of scene span). 1024 is faster still (15.1s) but doubles
    the error, so 2048 is the floor worth taking.

    `loop_detection` matches non-adjacent frames via a vocabulary tree so a clip
    that revisits a viewpoint — orbiting an object, panning back across a room —
    closes the loop rather than producing two disconnected reconstructions.

    It defaults to **off**, which is a deliberate departure from MPO-242.
    Measured cost on the same sequence: 19.3s -> 33.9s (**+75%**) for no change
    in pose error. That scene has a trivially connected frame chain and no
    accumulated drift, so it cannot show the upside — the cost is real and
    measured, the benefit is real in principle but unmeasured here. Turn it on
    for orbiting/revisiting captures, where it is exactly the right tool.

    COLMAP downloads the vocabulary tree on first use; `vocab_tree_path` (or
    $COLMAP_VOCAB_TREE, which the worker image sets) points at a local copy so
    enabling it costs nothing extra and works offline.

    Only disable intrinsic refinement when the intrinsics are already known.
    """
    cb = resolve_colmap_bin(colmap_bin)
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = out_dir / "database.db"
    sparse = out_dir / "sparse"
    txt = sparse / "model_txt"

    gpu_explicit = use_gpu is not None
    if gpu_explicit:
        gpu = bool(use_gpu)
        logging.info("GPU SIFT: %s (explicitly requested)", "on" if gpu else "off")
    else:
        gpu, why = detect_gpu_sift()
        logging.info("GPU SIFT: %s (auto-detected: %s)", "on" if gpu else "off", why)

    # Resolved against the installed binary, not hardcoded to one COLMAP release.
    feat_opts = supported_options(cb, "feature_extractor")
    match_opts = supported_options(cb, "sequential_matcher")
    feat_gpu = pick_option(feat_opts, "FeatureExtraction.use_gpu", "SiftExtraction.use_gpu")
    feat_threads = pick_option(feat_opts, "FeatureExtraction.num_threads", "SiftExtraction.num_threads")
    match_gpu = pick_option(match_opts, "FeatureMatching.use_gpu", "SiftMatching.use_gpu")
    match_threads = pick_option(match_opts, "FeatureMatching.num_threads", "SiftMatching.num_threads")
    logging.info("COLMAP option names: %s / %s", feat_gpu, match_gpu)

    def feat_cmd(g: bool) -> List[str]:
        return [
            cb, "feature_extractor",
            "--database_path", str(db),
            "--image_path", str(frames_dir),
            "--ImageReader.single_camera", "1",
            f"--{feat_gpu}", "1" if g else "0",
            "--SiftExtraction.max_num_features", str(max_num_features),
            f"--{feat_threads}", str(os.cpu_count() or 8),
        ]

    # A GPU failure here settles the question for matching too.
    gpu = _run_sift_stage("feature_extractor", feat_cmd, gpu, gpu_explicit, should_cancel)

    def match_cmd(g: bool) -> List[str]:
        cmd = [
            cb, "sequential_matcher",
            "--database_path", str(db),
            f"--{match_gpu}", "1" if g else "0",
            f"--{match_threads}", str(os.cpu_count() or 8),
            "--SiftMatching.max_ratio", "0.7",
            "--SequentialMatching.overlap", str(overlap),
        ]
        if loop_detection and "SequentialMatching.loop_detection" in match_opts:
            cmd += ["--SequentialMatching.loop_detection", "1"]
            tree = vocab_tree_path or os.environ.get("COLMAP_VOCAB_TREE")
            if tree:
                cmd += ["--SequentialMatching.vocab_tree_path", str(tree)]
        elif loop_detection:
            logging.warning(
                "This COLMAP build has no SequentialMatching.loop_detection; "
                "revisited viewpoints will not close the loop."
            )
        return cmd

    _run_sift_stage("sequential_matcher", match_cmd, gpu, gpu_explicit, should_cancel)

    sparse.mkdir(parents=True, exist_ok=True)
    map_cmd = [
        cb, "mapper",
        "--database_path", str(db),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse),
        "--Mapper.ba_refine_focal_length", "1" if refine_intrinsics else "0",
        # Principal point stays fixed at the image centre: it is weakly
        # observable and refining it tends to destabilize the solve.
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "1" if refine_intrinsics else "0",
        "--Mapper.tri_min_angle", "2",
        "--Mapper.abs_pose_min_num_inliers", "10",
        "--Mapper.filter_max_reproj_error", "8",
        "--Mapper.ba_global_max_refinements", "1",     
        "--Mapper.ba_local_max_refinements", "1",
        "--Mapper.extract_colors", "0",                  
        "--Mapper.num_threads", str(os.cpu_count() or 8),
    ]
    _run_cmd(map_cmd, should_cancel)

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
        _run_cmd(convert_cmd, should_cancel)

    logging.info("✅ Fast COLMAP reconstruction done.")
    return txt


if __name__ == "__main__":
    import argparse
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Fast COLMAP reconstruction for video frames")
    parser.add_argument("--frames", type=Path, default=repo / "frames")
    parser.add_argument("--out", type=Path, default=repo / "structure_from_motion" / "colmap_output_fast")
    parser.add_argument("--colmap-bin", type=str, default=None)
    # Tri-state: unset means detect. --gpu makes it binding, so a host that
    # should have GPU SIFT fails loudly instead of quietly running on CPU.
    gpu_grp = parser.add_mutually_exclusive_group()
    gpu_grp.add_argument("--gpu", dest="use_gpu", action="store_true", default=None,
                         help="Require GPU SIFT; fail if unavailable.")
    gpu_grp.add_argument("--no-gpu", dest="use_gpu", action="store_false", default=None,
                         help="Force CPU SIFT.")
    parser.add_argument(
        "--no-refine-intrinsics", dest="refine_intrinsics", action="store_false",
        help="Fix focal length and distortion at COLMAP's initial guess. Only "
             "correct when the true intrinsics are already known.",
    )
    parser.add_argument("--overlap", type=int, default=10,
                        help="Sequential matching window. Interacts with keyframe spacing.")
    parser.add_argument("--loop-detection", action="store_true",
                        help="Vocab-tree loop closure for orbiting/revisiting captures. "
                             "Measured +75%% matching time; off by default (MPO-242).")
    parser.add_argument("--vocab-tree", default=None,
                        help="Local vocabulary tree. Defaults to $COLMAP_VOCAB_TREE, else "
                             "COLMAP downloads one.")
    parser.add_argument("--max-num-features", type=int, default=2048,
                        help="SIFT features per frame.")
    args = parser.parse_args()

    imgs = list_frames(args.frames)
    logging.info("Found %d frames", len(imgs))
    if not imgs:
        raise SystemExit("No frames found.")

    txt_model = run_colmap_fast(
        args.frames, args.out, args.colmap_bin,
        use_gpu=args.use_gpu, refine_intrinsics=args.refine_intrinsics,
        overlap=args.overlap, loop_detection=args.loop_detection,
        vocab_tree_path=args.vocab_tree, max_num_features=args.max_num_features,
    )
    logging.info("Model TXT output: %s", txt_model)
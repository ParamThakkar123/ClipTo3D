"""Run the reconstruction pipeline end to end against one job directory.

Before this, running a clip meant invoking eight separate module CLIs in the
right order and reconciling their disagreeing default paths by hand (MPO-224).
Every stage here reads and writes the shared `JobPaths` layout.

    python pipeline.py clip.mp4 --job runs/my-clip

Stages, in order:

    frames   ffmpeg frame extraction
    keyframe drop redundant and blurry frames (MPO-237)
    depth    monocular relative-depth estimation
    colmap   structure from motion (poses + sparse points)
    dataset  COLMAP -> nerf-style transforms.json
    fuse     depth fusion into a point cloud (needs depth + colmap)
    mesh     TSDF surface extraction, for the AR formats
    splat    gaussian splat training (needs colmap; CUDA only)

`splat` is not in the default set because it needs a CUDA toolchain. The SAM2
stage is deliberately absent entirely — nothing consumes its output (MPO-230);
run `object_detection/sam2_run.py` directly if you want it.

This is the minimal runner that makes the stages compose. The polished
orchestrator — quality presets, console entry points, and sharing one code path
with the job service — is MPO-245.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from job_paths import JobPaths
from job_state import JobCancelled, JobState, fingerprint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")

ALL_STAGES = ["frames", "keyframe", "depth", "colmap", "dataset", "fuse", "mesh", "splat", "export"]
DEFAULT_STAGES = ["frames", "keyframe", "depth", "colmap", "dataset", "fuse", "mesh", "export"]


def _has_files(d: Path, patterns: Sequence[str] = ("*",)) -> bool:
    if not d.is_dir():
        return False
    return any(next(d.glob(p), None) is not None for p in patterns)


def stage_frames(job: JobPaths, video: Path, fps: Optional[float], every: Optional[float]) -> None:
    from frames import extract_frames

    out = extract_frames(
        str(video), out_dir=str(job.frames), fps=fps, interval_seconds=every, overwrite=True
    )
    logger.info("Extracted %d frames to %s", len(out), job.frames)
    if not out:
        raise RuntimeError(f"ffmpeg produced no frames from {video}")


def stage_keyframe(
    job: JobPaths, min_motion: float, blur_percentile: float,
    min_frames: int, max_frames: Optional[int],
) -> None:
    from keyframes import select_and_write

    sel = select_and_write(
        job.frames, job.keyframes,
        min_motion=min_motion,
        blur_percentile=blur_percentile,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    # Reported per job so the reduction is visible rather than implicit.
    logger.info("[keyframe] %s", sel.summary())
    if sel.n_kept == 0:
        raise RuntimeError(f"keyframe selection kept 0 of {sel.n_extracted} frames in {job.frames}")


def working_frames(job: JobPaths) -> Path:
    """Frames the downstream stages should consume.

    The keyframe subset when one has been produced, otherwise the raw
    extraction — so `--stages frames depth ...` without `keyframe` still works.
    """
    if job.keyframes_manifest.is_file() and any(
        p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in job.keyframes.iterdir()
    ):
        return job.keyframes
    return job.frames


def stage_depth(
    job: JobPaths, backend: str, checkpoint: Optional[Path], encoder: str,
    batch_size: int, depth_format: str,
) -> None:
    from depth_estimation.depth import estimate_depths

    estimate_depths(
        frames_dir=working_frames(job),
        out_dir=job.depth,
        model_backend=backend,
        depthanything_ckpt=checkpoint,
        encoder=encoder,
        batch_size=batch_size,
        depth_format=depth_format,
    )


def stage_colmap(
    job: JobPaths, colmap_bin: Optional[str], use_gpu: Optional[bool], refine_intrinsics: bool,
    overlap: int, loop_detection: bool, max_num_features: int,
    should_cancel=None,
) -> None:
    from structure_from_motion.sfm import run_colmap_fast

    txt = run_colmap_fast(
        working_frames(job), job.colmap, colmap_bin,
        use_gpu=use_gpu, refine_intrinsics=refine_intrinsics,
        overlap=overlap, loop_detection=loop_detection,
        max_num_features=max_num_features,
        should_cancel=should_cancel,
    )
    logger.info("COLMAP text model at %s", txt)


def stage_dataset(job: JobPaths, copy_images: bool) -> None:
    from convert_colmap_to_gs import colmap_to_transforms

    colmap_to_transforms(job.colmap, working_frames(job), job.dataset, copy_images=copy_images)


def stage_fuse(
    job: JobPaths, voxel_frac: float, voxel_size, min_views: int, stride: int
) -> None:
    from fusion.fuse import fuse

    fuse(
        colmap_dir=job.colmap,
        frames_dir=working_frames(job),
        depth_dir=job.depth,
        out_ply=job.fused_ply,
        voxel_frac=voxel_frac,
        voxel_size=voxel_size,
        min_views=min_views,
        stride=stride,
    )


def stage_splat(job: JobPaths, max_steps: int, max_image_side: int) -> None:
    from neural_reconstruction.gsplatting import train

    result = train(
        images_dir=working_frames(job),
        colmap_dir=job.colmap,
        out_dir=job.splat,
        max_steps=max_steps,
        max_image_side=max_image_side,
    )
    logger.info(
        "Splat training done: %d gaussians, final loss %.5f -> %s",
        result.n_gaussians, result.final_loss, result.ply_path,
    )


def stage_mesh(job: JobPaths, resolution: int, stride: int) -> None:
    import numpy as np

    from meshing import mesh_from_job

    try:
        import skimage  # noqa: F401
    except ModuleNotFoundError as exc:
        # Loud, not silent: without this the job still "succeeds" with a point
        # cloud and simply never produces a USDZ, which is only noticed when
        # someone goes looking for the AR asset.
        raise ModuleNotFoundError(
            "Surface extraction needs scikit-image, which is not installed: "
            "`uv sync --extra mesh`. Drop `mesh` from --stages to skip it and "
            "produce point-cloud exports only."
        ) from exc

    mesh = mesh_from_job(job.colmap, working_frames(job), job.depth,
                         resolution=resolution, stride=stride)
    if mesh.n_vertices == 0:
        # Not fatal: the point cloud is still a valid result, and the AR
        # formats are simply unavailable for this job.
        logger.warning("[mesh] surface extraction produced no geometry")
        return
    payload = {"vertices": mesh.vertices, "faces": mesh.faces}
    if mesh.vertex_colors is not None:
        payload["colors"] = mesh.vertex_colors
    job.mesh_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(job.mesh_npz, **payload)
    logger.info("[mesh] %s -> %s", mesh.describe(), job.mesh_npz)


def stage_export(job: JobPaths, budget_bytes: int, lods: int) -> None:
    from export import export_all

    results = export_all(job.root, budget_bytes=budget_bytes, lods=lods)
    for r in results:
        logger.info("[export] %s", r.describe())
    over = [r for r in results if not r.within_budget]
    if over:
        # A warning, not a failure: the reconstruction is still valid, and an
        # operator may legitimately want an over-budget desktop export.
        logger.warning(
            "[export] %d export(s) exceed the %.0f MB mobile budget",
            len(over), budget_bytes / 1e6,
        )


def run_pipeline(
    video: Optional[Path],
    job_root: Path,
    stages: Sequence[str] = tuple(DEFAULT_STAGES),
    fps: Optional[float] = 4.0,
    every: Optional[float] = None,
    keyframe_min_motion: float = 0.012,
    keyframe_blur_percentile: float = 20.0,
    keyframe_min_frames: int = 12,
    keyframe_max_frames: Optional[int] = 300,
    depth_backend: str = "depthanythingv2",  # offline-capable, pinned (MPO-233)
    depth_checkpoint: Optional[Path] = None,
    depth_encoder: str = "vitl",
    depth_batch_size: int = 8,      # MPO-241
    depth_format: str = "fp16",     # MPO-240
    colmap_bin: Optional[str] = None,
    colmap_gpu: Optional[bool] = None,  # None = detect (MPO-232)
    colmap_refine_intrinsics: bool = True,
    colmap_overlap: int = 10,            # MPO-242: was effectively 2
    colmap_loop_detection: bool = False,  # MPO-242: opt-in, +75% matching time
    colmap_max_features: int = 2048,
    copy_dataset_images: bool = True,
    voxel_frac: float = 0.004,
    voxel_size: Optional[float] = None,
    min_views: int = 2,
    stride: int = 2,
    splat_steps: int = 7_000,
    splat_max_side: int = 1_600,
    mesh_resolution: int = 192,
    mesh_stride: int = 2,
    export_budget_bytes: int = 20 * 1024 * 1024,
    export_lods: int = 3,
    force: bool = False,
) -> JobPaths:
    """Run `stages` for one job.

    Stages are skipped when their recorded fingerprint (parameters + inputs)
    still matches, so a re-run is a no-op and a job that died mid-COLMAP
    resumes at COLMAP rather than re-extracting frames (MPO-243).
    """
    job = JobPaths(job_root).ensure()
    state = JobState.load(job.root)
    # A sentinel left behind by a previous run must not kill this one.
    state.clear_cancel()
    logger.info("Job layout:\n%s", job.describe())

    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; valid: {ALL_STAGES}")

    if "frames" in stages:
        if video is None:
            raise ValueError("a video path is required to run the 'frames' stage")
        video = Path(video)
        if not video.is_file():
            raise FileNotFoundError(f"video not found: {video}")
        # Keep the source alongside its outputs so a job is self-describing.
        kept = job.input_dir / video.name
        if not kept.exists():
            shutil.copy2(video, kept)

    # Stage parameters, per stage, so changing one knob only invalidates the
    # stages it actually affects (MPO-243).
    params = {
        "frames": {"fps": fps, "every": every},
        "keyframe": {
            "min_motion": keyframe_min_motion, "blur_percentile": keyframe_blur_percentile,
            "min_frames": keyframe_min_frames, "max_frames": keyframe_max_frames,
        },
        "depth": {
            "backend": depth_backend, "encoder": depth_encoder,
            "checkpoint": str(depth_checkpoint) if depth_checkpoint else None,
            "batch_size": depth_batch_size, "format": depth_format,
        },
        "colmap": {
            "gpu": colmap_gpu, "refine_intrinsics": colmap_refine_intrinsics,
            "overlap": colmap_overlap, "loop_detection": colmap_loop_detection,
            "max_features": colmap_max_features,
        },
        "dataset": {"copy_images": copy_dataset_images},
        "fuse": {
            "voxel_frac": voxel_frac, "voxel_size": voxel_size,
            "min_views": min_views, "stride": stride,
        },
        "splat": {"steps": splat_steps, "max_side": splat_max_side},
        "mesh": {"resolution": mesh_resolution, "stride": mesh_stride},
        "export": {"budget": export_budget_bytes, "lods": export_lods},
    }
    # What each stage consumes. `working_frames` is resolved lazily because the
    # keyframe stage may not have run yet on this pass.
    def stage_inputs(stage: str) -> list:
        return {
            "frames": [job.input_dir],
            "keyframe": [job.frames],
            "depth": [working_frames(job)],
            "colmap": [working_frames(job)],
            "dataset": [job.colmap_model_txt, working_frames(job)],
            "fuse": [job.colmap_model_txt, job.depth, working_frames(job)],
            "splat": [job.colmap_model_txt, working_frames(job)],
            "mesh": [job.colmap_model_txt, job.depth, working_frames(job)],
            "export": [job.fused_ply, job.mesh_npz, job.splat],
        }[stage]

    outputs_present = {
        "frames": lambda: _has_files(job.frames, ("*.jpg", "*.jpeg", "*.png")),
        "keyframe": lambda: job.keyframes_manifest.is_file(),
        "depth": lambda: _has_files(job.depth, ("*_depth.npy", "*_depth.png")),
        "colmap": lambda: _has_files(job.colmap_sparse, ("**/cameras.txt", "**/cameras.bin")),
        "dataset": lambda: job.transforms_json.is_file(),
        "fuse": lambda: job.fused_ply.is_file(),
        "splat": lambda: (job.splat / "splat.ply").is_file(),
        "mesh": lambda: job.mesh_npz.is_file(),
        "export": lambda: _has_files(job.export, ("*.glb", "*.splat", "*.usdz")),
    }

    if force:
        state.reset(stages)

    # Fingerprints chain: a stage's identity includes its upstream stages'
    # fingerprints, so changing a depth parameter invalidates fusion even
    # though fusion's own parameters and its inputs' size/mtime are unchanged.
    # Without this, invalidation relies on downstream artifacts happening to
    # differ in size or mtime — which is exactly how a cache silently serves
    # stale results.
    upstream = {
        "frames": [],
        "keyframe": ["frames"],
        "depth": ["keyframe"],
        "colmap": ["keyframe"],
        "dataset": ["colmap"],
        "fuse": ["depth", "colmap"],
        "splat": ["colmap"],
        "mesh": ["depth", "colmap"],
        "export": ["fuse", "mesh", "splat"],
    }
    fingerprints: dict = {}

    for stage in ALL_STAGES:
        if stage not in stages:
            continue

        # Between stages is the natural place to stop: every artifact on disk
        # is complete, so the job can be resumed cleanly.
        if state.cancel_requested():
            state.cancel_stage(stage)
            logger.warning("[%s] cancellation requested; stopping", stage)
            raise JobCancelled(f"cancelled before {stage}")

        deps = {
            dep: fingerprints.get(
                dep,
                # Not run this pass: fall back to what the last run recorded.
                (state.stages[dep].fingerprint if dep in state.stages else None),
            )
            for dep in upstream[stage]
        }
        fp = fingerprint({**params[stage], "__upstream__": deps}, stage_inputs(stage))
        fingerprints[stage] = fp
        if not force and state.is_current(stage, fp) and outputs_present[stage]():
            logger.info("[%s] up to date, skipping", stage)
            state.emit(stage, 1.0, "cached")
            continue
        if not force and outputs_present[stage]() and stage not in state.stages:
            # Outputs exist from a run that predates job state. Adopt them
            # rather than redoing an hour of COLMAP on first upgrade.
            logger.info("[%s] outputs already present, adopting", stage)
            state.start(stage, fp)
            state.finish(stage, "adopted pre-existing outputs")
            continue

        started = time.time()
        state.start(stage, fp)
        logger.info("[%s] starting", stage)
        try:
            if stage == "frames":
                stage_frames(job, Path(video), fps, every)  # type: ignore[arg-type]
            elif stage == "keyframe":
                stage_keyframe(job, keyframe_min_motion, keyframe_blur_percentile,
                               keyframe_min_frames, keyframe_max_frames)
            elif stage == "depth":
                stage_depth(job, depth_backend, depth_checkpoint, depth_encoder,
                            depth_batch_size, depth_format)
            elif stage == "colmap":
                stage_colmap(job, colmap_bin, colmap_gpu, colmap_refine_intrinsics,
                             colmap_overlap, colmap_loop_detection, colmap_max_features,
                             state.canceller())
            elif stage == "dataset":
                stage_dataset(job, copy_dataset_images)
            elif stage == "fuse":
                stage_fuse(job, voxel_frac, voxel_size, min_views, stride)
            elif stage == "splat":
                stage_splat(job, splat_steps, splat_max_side)
            elif stage == "mesh":
                stage_mesh(job, mesh_resolution, mesh_stride)
            elif stage == "export":
                stage_export(job, export_budget_bytes, export_lods)
        except JobCancelled:
            state.cancel_stage(stage)
            raise
        except Exception as exc:
            # Record before re-raising: the whole point is that a resume knows
            # exactly which stage to restart at.
            state.fail(stage, f"{type(exc).__name__}: {exc}")
            raise

        elapsed = time.time() - started
        state.finish(stage, f"{elapsed:.1f}s")
        logger.info("[%s] done in %.1fs", stage, elapsed)

    state.emit("pipeline", 1.0, "complete")
    logger.info("Pipeline complete. Job root: %s", job.root)
    return job


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Run the video -> 3D reconstruction pipeline against a job directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"stages: {', '.join(ALL_STAGES)}   (default: {', '.join(DEFAULT_STAGES)})",
    )
    p.add_argument("video", nargs="?", type=Path, help="Input video (needed for the frames stage).")
    p.add_argument("--job", type=Path, required=True, help="Job directory to create/reuse.")
    p.add_argument("--stages", nargs="+", default=DEFAULT_STAGES, metavar="STAGE")
    p.add_argument("--force", action="store_true", help="Re-run stages even if already complete.")

    g = p.add_argument_group("frames")
    mx = g.add_mutually_exclusive_group()
    mx.add_argument("--fps", type=float, default=4.0)
    mx.add_argument("--every", type=float, help="One frame every N seconds.")

    g = p.add_argument_group("keyframe")
    g.add_argument("--keyframe-min-motion", type=float, default=0.012,
                   help="Median parallax (fraction of image width) needed to keep a frame. "
                        "Raise for a more aggressive cut.")
    g.add_argument("--keyframe-blur-percentile", type=float, default=20.0,
                   help="Sharpness floor as a percentile of this clip's own distribution.")
    g.add_argument("--keyframe-min-frames", type=int, default=12)
    g.add_argument("--keyframe-max-frames", type=int, default=300)

    g = p.add_argument_group("depth")
    g.add_argument("--depth-backend", default="depthanythingv2",
                   choices=["midas", "depthanythingv2"],
                   help="depthanythingv2 runs offline from pinned weights; midas needs torch.hub.")
    g.add_argument("--depth-checkpoint", type=Path,
                   help="Override the pinned Depth-Anything-v2 checkpoint with a local .pth.")
    g.add_argument("--depth-encoder", default="vitl", choices=["vits", "vitb", "vitl"],
                   help="vits is the cheap preview tier; vitl the quality tier.")
    g.add_argument("--depth-batch-size", type=int, default=8)
    g.add_argument("--depth-format", default="fp16", choices=["fp16", "png16", "fp32"],
                   help="fp16 halves depth storage; png16 is ~10x smaller.")

    g = p.add_argument_group("colmap")
    g.add_argument("--colmap-bin", default=None)
    # Unset means detect GPU SIFT availability; --colmap-gpu makes it binding.
    gpu_grp = g.add_mutually_exclusive_group()
    gpu_grp.add_argument("--colmap-gpu", dest="colmap_gpu", action="store_true", default=None,
                         help="Require GPU SIFT; fail if unavailable.")
    gpu_grp.add_argument("--no-colmap-gpu", dest="colmap_gpu", action="store_false", default=None,
                         help="Force CPU SIFT.")
    g.add_argument(
        "--no-refine-intrinsics", dest="colmap_refine_intrinsics", action="store_false",
        help="Fix focal length at COLMAP's initial guess; badly distorts geometry "
             "unless the true intrinsics are known.",
    )
    g.add_argument("--colmap-overlap", type=int, default=10,
                   help="Sequential matching window; interacts with keyframe spacing.")
    g.add_argument("--loop-detection", dest="colmap_loop_detection", action="store_true",
                   help="Vocab-tree loop closure for orbiting captures (+75%% matching time).")
    g.add_argument("--colmap-max-features", type=int, default=2048)
    g.add_argument("--no-copy-dataset-images", dest="copy_dataset_images", action="store_false")

    g = p.add_argument_group("fuse")
    g.add_argument("--voxel-frac", type=float, default=0.004,
                   help="Voxel size as a fraction of sparse-cloud radius (COLMAP scale is arbitrary).")
    g.add_argument("--voxel-size", type=float, default=None,
                   help="Absolute voxel size; overrides --voxel-frac.")
    g.add_argument("--min-views", type=int, default=2)
    g.add_argument("--stride", type=int, default=2)

    g = p.add_argument_group("splat")
    g.add_argument("--splat-steps", type=int, default=7_000)
    g.add_argument("--splat-max-side", type=int, default=1_600)

    args = p.parse_args(argv)

    try:
        run_pipeline(
            video=args.video,
            job_root=args.job,
            stages=args.stages,
            fps=args.fps,
            every=args.every,
            keyframe_min_motion=args.keyframe_min_motion,
            keyframe_blur_percentile=args.keyframe_blur_percentile,
            keyframe_min_frames=args.keyframe_min_frames,
            keyframe_max_frames=args.keyframe_max_frames,
            depth_backend=args.depth_backend,
            depth_checkpoint=args.depth_checkpoint,
            depth_encoder=args.depth_encoder,
            depth_batch_size=args.depth_batch_size,
            depth_format=args.depth_format,
            colmap_bin=args.colmap_bin,
            colmap_gpu=args.colmap_gpu,
            colmap_refine_intrinsics=args.colmap_refine_intrinsics,
            colmap_overlap=args.colmap_overlap,
            colmap_loop_detection=args.colmap_loop_detection,
            colmap_max_features=args.colmap_max_features,
            copy_dataset_images=args.copy_dataset_images,
            voxel_frac=args.voxel_frac,
            voxel_size=args.voxel_size,
            min_views=args.min_views,
            stride=args.stride,
            splat_steps=args.splat_steps,
            splat_max_side=args.splat_max_side,
            force=args.force,
        )
    except Exception as e:
        logger.error("%s: %s", type(e).__name__, e)
        sys.exit(1)


if __name__ == "__main__":
    main()

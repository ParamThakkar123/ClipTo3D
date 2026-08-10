"""`clipto3d` — one command for the whole pipeline (MPO-245).

`main.py` printed "Hello from cliptoworld!" and everything real lived in eight
separate `if __name__ == "__main__"` blocks, each with its own argparse and its
own default paths that disagreed with the others. Running a clip meant issuing
eight commands in the right order and reconciling paths between each.

    clipto3d reconstruct clip.mp4 --out ./result
    clipto3d reconstruct clip.mp4 --out ./result --quality final
    clipto3d status ./result
    clipto3d cancel ./result

The per-module CLIs still exist and are still useful for running one stage in
isolation; they are a thin layer over the same importable stage functions, not
the entry point. This is the code path the GPU worker uses too, so the CLI and
the service cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from job_paths import JobPaths
from job_state import JobCancelled, JobState
from pipeline import ALL_STAGES, DEFAULT_STAGES, run_pipeline


@dataclass(frozen=True)
class Preset:
    """A coherent set of quality/cost choices.

    These knobs interact — a small encoder with a huge keyframe budget is
    incoherent — so they are set together rather than left as independent
    flags a caller has to get right.
    """

    name: str
    description: str
    fps: float
    depth_encoder: str
    depth_format: str
    keyframe_min_motion: float
    keyframe_max_frames: int
    colmap_max_features: int
    splat_steps: int
    stride: int


PRESETS: Dict[str, Preset] = {
    "preview": Preset(
        name="preview",
        description="Fast look at whether a clip reconstructs at all.",
        fps=2.0,
        depth_encoder="vits",
        depth_format="png16",
        keyframe_min_motion=0.02,
        keyframe_max_frames=60,
        colmap_max_features=1024,
        splat_steps=2_000,
        stride=4,
    ),
    "balanced": Preset(
        name="balanced",
        description="Default. Sensible quality without a long wait.",
        fps=4.0,
        depth_encoder="vitb",
        depth_format="fp16",
        keyframe_min_motion=0.012,
        keyframe_max_frames=150,
        colmap_max_features=2048,
        splat_steps=7_000,
        stride=2,
    ),
    "final": Preset(
        name="final",
        description="Best quality this pipeline produces; slowest.",
        fps=6.0,
        depth_encoder="vitl",
        depth_format="fp16",
        keyframe_min_motion=0.008,
        keyframe_max_frames=300,
        colmap_max_features=4096,
        splat_steps=30_000,
        stride=1,
    ),
}
DEFAULT_PRESET = "balanced"


def _cmd_reconstruct(args: argparse.Namespace) -> int:
    preset = PRESETS[args.quality]
    stages: List[str] = list(args.stages) if args.stages else list(DEFAULT_STAGES)
    if args.splat and "splat" not in stages:
        stages.append("splat")

    # Explicit flags beat the preset; the preset beats the bare default.
    def pick(explicit, from_preset):
        return from_preset if explicit is None else explicit

    try:
        job = run_pipeline(
            video=args.video,
            job_root=args.out,
            stages=stages,
            fps=pick(args.fps, preset.fps),
            every=args.every,
            keyframe_min_motion=pick(args.keyframe_min_motion, preset.keyframe_min_motion),
            keyframe_max_frames=pick(args.keyframe_max_frames, preset.keyframe_max_frames),
            depth_encoder=pick(args.depth_encoder, preset.depth_encoder),
            depth_format=pick(args.depth_format, preset.depth_format),
            depth_backend=args.depth_backend,
            colmap_bin=args.colmap_bin,
            colmap_gpu=args.colmap_gpu,
            colmap_loop_detection=args.loop_detection,
            colmap_max_features=pick(args.colmap_max_features, preset.colmap_max_features),
            stride=pick(args.stride, preset.stride),
            splat_steps=pick(args.splat_steps, preset.splat_steps),
            force=args.force,
        )
    except JobCancelled as exc:
        print(f"cancelled: {exc}", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        state = JobState.load(Path(args.out))
        failed = state.progress()["failed"]
        if failed:
            print(f"failed at stage(s): {', '.join(failed)}", file=sys.stderr)
            print(f"re-run the same command to resume from there", file=sys.stderr)
        return 1

    print(f"\nDone. Job root: {job.root}")
    for label, path in (
        ("point cloud", job.fused_ply),
        ("transforms", job.transforms_json),
        ("splat", job.splat / "splat.ply"),
    ):
        if path.is_file():
            print(f"  {label:12} {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.job)
    state = JobState.load(root)
    if not state.stages:
        print(f"no job state at {root}")
        return 1

    prog = state.progress()
    if args.json:
        print(json.dumps({"progress": prog, "events": state.events(limit=args.tail)}, indent=2))
        return 0

    print(f"job: {root}")
    print(f"  {prog['completed']}/{prog['total']} stages complete"
          + (f", running {prog['running']}" if prog["running"] else "")
          + (" [CANCEL REQUESTED]" if prog["cancelled"] else ""))
    for name in ALL_STAGES:
        rec = state.stages.get(name)
        if rec is None:
            continue
        secs = f"{rec.seconds:.1f}s" if rec.seconds else ""
        detail = rec.error or rec.message
        print(f"  {name:9} {rec.status:9} {secs:>8}  {detail}")
    if args.tail:
        print("\nrecent events:")
        for e in state.events(limit=args.tail):
            frac = "" if e.get("fraction") is None else f"{e['fraction'] * 100:5.1f}%"
            print(f"  {e['stage']:9} {frac:>6} {e['message']}")
    return 0


def _cmd_cancel(args: argparse.Namespace) -> int:
    state = JobState.load(Path(args.job))
    state.request_cancel(args.reason)
    print(f"cancellation requested for {args.job}")
    print("The running job stops at its next checkpoint; artifacts already written are kept.")
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    """Drop intermediates a finished job no longer needs (MPO-240).

    Deliberately opt-in rather than automatic after fusion. Depth is by far the
    most expensive artifact to recompute, and resume (MPO-243) uses its presence
    to decide whether the depth stage can be skipped — deleting it by default
    would turn every re-run into a full re-inference.
    """
    job = JobPaths(args.job)
    targets = {
        "depth": job.depth,
        "frames": job.frames,
        "keyframes": job.keyframes,
    }
    chosen = [k for k in ("depth", "frames", "keyframes") if getattr(args, k)]
    if not chosen:
        chosen = ["depth"]

    if not job.fused_ply.is_file() and not (job.splat / "splat.ply").is_file() and not args.force:
        print("refusing to clean: this job has produced no point cloud or splat yet.",
              file=sys.stderr)
        print("Pass --force if you really mean it.", file=sys.stderr)
        return 1

    import shutil as _shutil

    freed = 0
    for name in chosen:
        d = targets[name]
        if not d.is_dir():
            continue
        size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        if args.dry_run:
            print(f"would remove {name:9} {size / 1e6:8.1f} MB  {d}")
        else:
            _shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            print(f"removed {name:9} {size / 1e6:8.1f} MB")
        freed += size
    verb = "would free" if args.dry_run else "freed"
    print(f"{verb} {freed / 1e6:.1f} MB")
    return 0


def _cmd_presets(_args: argparse.Namespace) -> int:
    for p in PRESETS.values():
        marker = " (default)" if p.name == DEFAULT_PRESET else ""
        print(f"{p.name}{marker}: {p.description}")
        print(f"  fps {p.fps}  encoder {p.depth_encoder}  depth {p.depth_format}  "
              f"keyframes<={p.keyframe_max_frames}  features {p.colmap_max_features}  "
              f"splat steps {p.splat_steps}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report this machine's reconstruction capability.

    Shared with the desktop shell (MPO-249), which needs the same answer
    before it can decide whether to run locally or upload — and useful on its
    own for the far more common question of why a stage was skipped.
    """
    import json as _json

    from local_runtime import MODE_SERVICE, as_dict, probe, report

    cap = probe(ffmpeg=args.ffmpeg, colmap=args.colmap)
    print(_json.dumps(as_dict(cap), indent=2) if args.json else report(cap))
    # Non-zero when this machine cannot reconstruct on its own, so a script or
    # an installer can branch on it without parsing the output.
    return 1 if cap.mode() == MODE_SERVICE else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clipto3d",
        description="Reconstruct a 3D scene from a video clip.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("reconstruct", help="Run the pipeline against a clip.")
    r.add_argument("video", type=Path, nargs="?")
    r.add_argument("--out", type=Path, required=True, help="Job directory to create/reuse.")
    r.add_argument("--quality", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    r.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=None)
    r.add_argument("--splat", action="store_true", help="Also train gaussian splats (needs CUDA).")
    r.add_argument("--force", action="store_true", help="Ignore cached stages and redo everything.")
    # All default to None so "unset" is distinguishable from "explicitly set",
    # which is what lets the preset fill only the gaps.
    r.add_argument("--fps", type=float, default=None)
    r.add_argument("--every", type=float, default=None)
    r.add_argument("--depth-backend", default="depthanythingv2",
                   choices=["midas", "depthanythingv2"])
    r.add_argument("--depth-encoder", default=None, choices=["vits", "vitb", "vitl"])
    r.add_argument("--depth-format", default=None, choices=["fp16", "png16", "fp32"])
    r.add_argument("--keyframe-min-motion", type=float, default=None)
    r.add_argument("--keyframe-max-frames", type=int, default=None)
    r.add_argument("--colmap-bin", default=None)
    r.add_argument("--colmap-max-features", type=int, default=None)
    r.add_argument("--stride", type=int, default=None)
    r.add_argument("--splat-steps", type=int, default=None)
    r.add_argument("--loop-detection", action="store_true",
                   help="Vocab-tree loop closure for orbiting captures (+75%% matching time).")
    gpu = r.add_mutually_exclusive_group()
    gpu.add_argument("--colmap-gpu", dest="colmap_gpu", action="store_true", default=None)
    gpu.add_argument("--no-colmap-gpu", dest="colmap_gpu", action="store_false", default=None)
    r.set_defaults(func=_cmd_reconstruct)

    s = sub.add_parser("status", help="Show stage status and progress for a job.")
    s.add_argument("job", type=Path)
    s.add_argument("--json", action="store_true", help="Machine-readable, for a service to poll.")
    s.add_argument("--tail", type=int, default=0, help="Show the last N progress events.")
    s.set_defaults(func=_cmd_status)

    c = sub.add_parser("cancel", help="Ask a running job to stop at its next checkpoint.")
    c.add_argument("job", type=Path)
    c.add_argument("--reason", default="")
    c.set_defaults(func=_cmd_cancel)

    cl = sub.add_parser("clean", help="Delete intermediates from a finished job.")
    cl.add_argument("job", type=Path)
    cl.add_argument("--depth", action="store_true", help="Depth maps (the default, and the largest).")
    cl.add_argument("--frames", action="store_true", help="Extracted frames.")
    cl.add_argument("--keyframes", action="store_true", help="Selected keyframes.")
    cl.add_argument("--dry-run", action="store_true", help="Report what would go, delete nothing.")
    cl.add_argument("--force", action="store_true",
                    help="Clean even though the job produced no output.")
    cl.set_defaults(func=_cmd_clean)

    pr = sub.add_parser("presets", help="List quality presets.")
    pr.set_defaults(func=_cmd_presets)

    dr = sub.add_parser(
        "doctor",
        help="Report what this machine can reconstruct, and where jobs should run.")
    dr.add_argument("--json", action="store_true",
                    help="Machine-readable, for the desktop shell.")
    dr.add_argument("--ffmpeg", help="Path to ffmpeg, if it is not on PATH.")
    dr.add_argument("--colmap", help="Path to COLMAP, if it is not on PATH.")
    dr.set_defaults(func=_cmd_doctor)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reconstruct" and args.video is None and (
        not args.stages or "frames" in args.stages
    ):
        print("a video is required unless you skip the 'frames' stage", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

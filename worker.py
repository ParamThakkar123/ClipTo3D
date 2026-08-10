"""Lease-driven worker pool (MPO-244).

`job_registry` has had `claim()` and `heartbeat()` since the queue was written,
but until now nothing outside its unit tests ever called them. The service
dispatched work through a `ThreadPoolExecutor`, so the lease columns stayed
NULL and two failure modes went unnoticed, both invisible until a restart:

* A job still `queued` when the process died stayed queued forever. Nothing
  polled the table, so a redeploy silently stranded every pending job.
* A job `running` when the process died stayed `running` forever. Its
  `lease_expires` was NULL, and `NULL < now` evaluates to NULL rather than
  true, so `reap_expired()` could not see it either — the one place that was
  supposed to surface the problem was structurally blind to it.

This module is the missing half: worker threads that claim from the table,
heartbeat the lease while they work, and complete the row when they finish. It
is what makes the registry's recovery path reachable rather than theoretical.

The pool owns *status*; the injected `runner` owns *work*. Keeping that split
is what lets the API tests stub reconstruction without also having to
reimplement the state machine.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from job_registry import CANCELLED, DONE, FAILED, JobRegistry

log = logging.getLogger("clipto3d.worker")

# A worker refreshes its lease well inside the deadline, so an ordinary GC
# pause or a slow COLMAP write cannot cost it a job it is actively running.
HEARTBEAT_FRACTION = 0.25
MIN_HEARTBEAT_SECONDS = 5.0

Runner = Callable[[str, Path, str, Dict[str, Any]], None]


class WorkerPool:
    """A fixed set of threads that drain the registry's queue.

    Concurrency is bounded by GPU count rather than CPU: reconstruction is
    GPU-bound, so running more jobs than there are devices makes each one
    slower without finishing any sooner.
    """

    def __init__(
        self,
        registry: JobRegistry,
        root: Path,
        runner: Runner,
        max_workers: int = 1,
        poll_interval: float = 0.1,
        worker_prefix: str = "recon",
    ):
        self.registry = registry
        self.root = Path(root)
        self.runner = runner
        # 0 is meaningful: an API process in a split deployment serves requests
        # and enqueues rows, and dedicated worker containers drain them.
        self.max_workers = max(0, int(max_workers))
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        # Set on submit so a freshly-queued job starts immediately instead of
        # waiting out a poll interval. Lossy with several workers idle, which
        # is why the poll timeout stays as the backstop.
        self._wake = threading.Event()
        self._threads: list[threading.Thread] = []
        self._worker_prefix = worker_prefix
        self._started = False

    # --- lifecycle --------------------------------------------------------

    def start(self) -> "WorkerPool":
        if self._started:
            return self
        self._started = True
        if self.max_workers == 0:
            log.info("no in-process workers; expecting a separate worker process")
            return self
        for i in range(self.max_workers):
            t = threading.Thread(
                target=self._loop,
                args=(f"{self._worker_prefix}-{i}",),
                name=f"{self._worker_prefix}-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
        return self

    def notify(self) -> None:
        """Tell an idle worker there may be work. Never blocks the caller."""
        self._wake.set()

    def shutdown(self, wait: bool = True, timeout: Optional[float] = 30.0) -> None:
        """Stop claiming new work; optionally wait for in-flight jobs.

        In-flight jobs are allowed to finish rather than being killed, because
        a half-written export is worse than a slow shutdown. Anything still
        running when `timeout` lapses keeps its lease, so the next process to
        start reclaims it once that lease expires.
        """
        self._stop.set()
        self._wake.set()
        if not wait:
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        for t in self._threads:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            t.join(timeout=remaining)
        stragglers = [t.name for t in self._threads if t.is_alive()]
        if stragglers:
            log.warning(
                "workers still running at shutdown: %s; their leases will expire "
                "and the jobs will be reclaimed", ", ".join(stragglers),
            )

    # --- the loop ---------------------------------------------------------

    def _loop(self, worker_id: str) -> None:
        while not self._stop.is_set():
            try:
                rec = self.registry.claim(worker_id)
            except Exception:  # a transient SQLite error must not kill a worker
                log.exception("claim failed on %s", worker_id)
                self._wake.wait(self.poll_interval)
                continue

            if rec is None:
                # Wait for a nudge, but wake anyway so an expired lease
                # elsewhere is eventually picked up.
                self._wake.wait(self.poll_interval)
                self._wake.clear()
                continue

            self._execute(worker_id, rec)

    def _execute(self, worker_id: str, rec) -> None:
        from job_state import JobCancelled, JobState

        job_dir = self.root / rec.id
        video = self.root / rec.id / "input" / rec.video_name

        # Cancelled before a worker ever picked it up: honour it without
        # spending a GPU slot proving the pipeline can stop.
        if JobState(root=job_dir).cancel_requested():
            self.registry.complete(rec.id, status=CANCELLED)
            log.info("job %s was cancelled before it started", rec.id)
            return

        stop_beat = self._start_heartbeat(worker_id, rec.id)
        started = time.monotonic()
        status, error = DONE, ""
        try:
            self.runner(rec.id, video, rec.quality, dict(rec.params or {}))
        except JobCancelled:
            status = CANCELLED
        except BaseException as exc:  # noqa: BLE001 - a failed job is data, not a crash
            status, error = FAILED, f"{type(exc).__name__}: {exc}"
            log.exception("job %s failed", rec.id)
        finally:
            stop_beat.set()

        # The job may have been deleted while it ran. Complete() is then a
        # no-op against a missing row, and the directory the runner recreated
        # is an orphan no caller can see — so remove it.
        if self.registry.get(rec.id) is None:
            shutil.rmtree(job_dir, ignore_errors=True)
            log.info("job %s was deleted while running; discarded its output", rec.id)
            return

        self.registry.complete(rec.id, status=status, error=error)
        log.info("job %s finished: %s in %.1fs", rec.id, status, time.monotonic() - started)

    def _start_heartbeat(self, worker_id: str, job_id: str) -> threading.Event:
        """Refresh the lease until the job ends.

        Without this every job over one lease-length looks abandoned and gets
        reclaimed *while it is still running*, which is worse than no recovery
        at all — two workers on one job directory.
        """
        interval = max(MIN_HEARTBEAT_SECONDS,
                       self.registry.lease_seconds * HEARTBEAT_FRACTION)
        done = threading.Event()

        def beat() -> None:
            while not done.wait(interval):
                try:
                    if not self.registry.heartbeat(job_id, worker_id):
                        # Someone else owns it now, or it is already finished.
                        # Stop refreshing rather than fighting over the row.
                        log.warning("lost the lease on job %s", job_id)
                        return
                except Exception:
                    log.exception("heartbeat failed for job %s", job_id)

        threading.Thread(target=beat, name=f"beat-{job_id}", daemon=True).start()
        return done


def pipeline_runner(root: Path, storage=None, registry: Optional[JobRegistry] = None) -> Runner:
    """The real reconstruction, as a `Runner`.

    Lives here rather than in `service.py` so a standalone worker process can
    use the identical code path without importing FastAPI — the worker image
    has COLMAP and CUDA in it and no reason to also carry a web framework.

    Exceptions propagate on purpose: the pool maps `JobCancelled` to
    `cancelled` and anything else to `failed`. Deciding status here as well is
    how a job once ended up marked done by one layer and failed by the other.
    """
    root = Path(root)

    def run(job_id: str, video: Path, quality: str, params: Dict[str, Any]) -> None:
        from cli import PRESETS
        from pipeline import run_pipeline

        preset = PRESETS.get(quality, PRESETS["balanced"])
        job_dir = root / job_id
        run_pipeline(
            video=video,
            job_root=job_dir,
            fps=preset.fps,
            depth_encoder=preset.depth_encoder,
            depth_format=preset.depth_format,
            keyframe_min_motion=preset.keyframe_min_motion,
            keyframe_max_frames=preset.keyframe_max_frames,
            colmap_max_features=preset.colmap_max_features,
            stride=preset.stride,
            **params,
        )
        if storage is not None:
            # Stateless workers: push artifacts somewhere durable, since the
            # local scratch directory is not guaranteed to outlive the job.
            try:
                storage.job(job_id).upload_tree(job_dir / "export")
            except Exception as exc:  # artifacts are on disk; do not fail the job
                log.warning("artifact upload failed for job %s: %s", job_id, exc)
                if registry is not None:
                    registry.update(job_id, error=f"upload warning: {exc}")

    return run


class Reaper:
    """Periodically return unowned `running` jobs to the queue.

    Startup recovery covers the common case — a restart — but a worker can die
    without the process dying with it (an OOM kill of a subprocess, a thread
    that raised past its handler). Without a periodic sweep those jobs wait for
    the next restart to be noticed.
    """

    def __init__(self, registry: JobRegistry, interval: float, on_requeue=None):
        self.registry = registry
        self.interval = interval
        self.on_requeue = on_requeue
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "Reaper":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop, name="reaper", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                requeued = self.registry.requeue_stale()
            except Exception:
                log.exception("reaper sweep failed")
                continue
            if requeued:
                log.warning("requeued %d job(s) whose worker went away: %s",
                            len(requeued), ", ".join(requeued))
                if self.on_requeue is not None:
                    self.on_requeue()


# --- standalone worker process --------------------------------------------
#
# The API and the reconstruction have very different shapes: one is a small
# always-up HTTP process, the other wants a GPU, COLMAP and several gigabytes
# of image. Running them as separate containers against the same jobs
# directory is what the lease queue is for — and it means an OOM during a
# reconstruction cannot take the API down with it.


def main(argv: Optional[list] = None) -> int:
    """Drain the queue in a dedicated process. `python worker.py --help`."""
    import argparse
    import os
    import signal

    from storage import from_uri

    ap = argparse.ArgumentParser(
        description="ClipTo3D reconstruction worker: claims jobs and runs them.")
    ap.add_argument("--jobs-root", default=os.environ.get("CLIPTO3D_JOBS_ROOT", "./runs"),
                    help="shared jobs directory; must be the same one the API writes to")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("CLIPTO3D_WORKERS", "1")),
                    help="concurrent jobs; set this to your GPU count")
    ap.add_argument("--storage", default=os.environ.get("CLIPTO3D_STORAGE", ""),
                    help="artifact storage URI, e.g. s3://bucket/prefix")
    ap.add_argument("--lease-seconds", type=int,
                    default=int(os.environ.get("CLIPTO3D_LEASE_SECONDS", "3600")))
    ap.add_argument("--reaper-seconds", type=float,
                    default=float(os.environ.get("CLIPTO3D_REAPER_SECONDS", "60")))
    ap.add_argument("--log-level", default=os.environ.get("CLIPTO3D_LOG_LEVEL", "INFO"))
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    root = Path(args.jobs_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    registry = JobRegistry(root / "jobs.db", lease_seconds=args.lease_seconds)
    store = from_uri(args.storage) if args.storage else None

    pool = WorkerPool(
        registry=registry,
        root=root,
        runner=pipeline_runner(root, store, registry),
        max_workers=max(1, args.workers),
    )
    reaper = Reaper(registry, interval=args.reaper_seconds, on_requeue=pool.notify)

    # This process may be the only one running, so it owns startup recovery
    # too. Safe alongside a recovering API: a live lease sits in the future
    # and `requeue_stale` leaves it alone.
    stale = registry.requeue_stale()
    if stale:
        log.warning("requeued %d stranded job(s): %s", len(stale), ", ".join(stale))

    stopping = threading.Event()

    def handle_signal(signum, _frame):
        # Stop claiming and let the current job finish. A SIGKILL after the
        # grace period is fine: the lease lapses and the job is reclaimed at
        # the stage it reached, because every stage checkpoints.
        log.info("signal %s received; finishing the current job then exiting", signum)
        stopping.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handle_signal)
        except (ValueError, OSError):  # pragma: no cover - not the main thread
            pass

    log.info("worker starting: root=%s workers=%d", root, pool.max_workers)
    pool.start()
    reaper.start()
    try:
        while not stopping.wait(1.0):
            pass
    finally:
        reaper.stop()
        pool.shutdown(wait=True, timeout=None)
    log.info("worker stopped")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    raise SystemExit(main())

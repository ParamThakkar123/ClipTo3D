"""Durable per-job state: caching, resume, progress and cancellation (MPO-243).

Nothing in the pipeline was resumable or observable. Every re-run redid
everything from frame extraction, a stage that failed 40 minutes into COLMAP
discarded all of it, progress was `print` to stdout that no client could poll,
and a runaway job ran to completion because there was no way to interrupt it.

Four pieces, all backed by files in the job directory so they survive the
process and are readable by a service that did not start it:

``state.json``
    Per-stage status, timing, and the **fingerprint** of the inputs and
    parameters that produced it. A stage whose fingerprint still matches is
    skipped; change a parameter and only the affected stages re-run.

``events.jsonl``
    Append-only progress events (stage, fraction, message). Append-only
    because a reader tailing the file must never see a partial rewrite.

``CANCEL``
    A sentinel file. Cooperative cancellation: stages check it between frames
    or iterations, so a cancelled job stops at a consistent point instead of
    being killed mid-write.

Writes are atomic (temp file + replace) so a crash mid-write cannot leave
unparseable state — which would strand a job far more effectively than the
crash itself.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

STATE_FILE = "state.json"
EVENTS_FILE = "events.jsonl"
CANCEL_FILE = "CANCEL"
LOCK_FILE = ".state.lock"

# Distinguishes concurrent writers' temp files within one process.
_TMP_SEQ = itertools.count()


# --- cross-process locking ------------------------------------------------
#
# Windows will not rename onto a file another handle has open, so a reader
# calling `load()` makes a concurrent `save()` fail with PermissionError.
# Retrying the rename is not enough — under four writers and two readers it
# still lost 4 runs in 6. Readers and writers have to not overlap, and the
# lock has to be visible across processes, because the service polls state
# that a worker process writes.

if os.name == "nt":
    import msvcrt

    def _acquire(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _release(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _acquire(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def _locked(root: Path, timeout: float = 10.0):
    """Hold an exclusive lock on the job's state for the duration of the block.

    Both OS primitives are released automatically if the holder dies, so a
    crashed worker cannot wedge a job. On timeout this proceeds *unlocked*
    rather than raising: degraded serialisation beats failing a job outright.
    """
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(root / LOCK_FILE), os.O_RDWR | os.O_CREAT, 0o644)
    held = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                _acquire(fd)
                held = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.002)
        try:
            yield
        finally:
            if held:
                with contextlib.suppress(OSError):
                    _release(fd)
    finally:
        os.close(fd)

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"


class JobCancelled(RuntimeError):
    """Raised when a cancellation request is observed."""


@dataclass
class StageRecord:
    status: str = PENDING
    fingerprint: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    seconds: Optional[float] = None
    message: str = ""
    error: str = ""

    @property
    def duration(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at


@dataclass
class JobState:
    root: Path
    stages: Dict[str, StageRecord] = field(default_factory=dict)

    # --- construction -----------------------------------------------------

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILE

    @property
    def events_path(self) -> Path:
        return self.root / EVENTS_FILE

    @property
    def cancel_path(self) -> Path:
        return self.root / CANCEL_FILE

    @classmethod
    def load(cls, root: Path | str) -> "JobState":
        root = Path(root)
        state = cls(root=root)
        path = root / STATE_FILE
        if path.is_file():
            try:
                with _locked(root):
                    raw = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return state          # replaced out from under us; nothing yet
            except json.JSONDecodeError:
                # Corrupt state is recoverable: treat as a fresh job rather
                # than refusing to run at all.
                return state
            for name, rec in (raw.get("stages") or {}).items():
                known = {k: v for k, v in rec.items() if k in StageRecord.__annotations__}
                state.stages[name] = StageRecord(**known)
        return state

    # --- persistence ------------------------------------------------------

    def save(self, attempts: int = 5) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "stages": {name: asdict(rec) for name, rec in self.stages.items()},
        }
        # A unique temp name per writer. A fixed one (state.json.tmp) is a real
        # race: two threads writing state at once both open the same file, and
        # os.replace then fails on Windows because the other still holds it.
        tmp = self.state_path.with_name(
            f".state.{os.getpid()}.{threading.get_ident()}.{next(_TMP_SEQ)}.tmp"
        )
        text = json.dumps(payload, indent=2)
        # The lock keeps readers out; the retry covers the case where it timed
        # out and we are writing unlocked, plus antivirus holding the file open.
        with _locked(self.root):
            tmp.write_text(text, encoding="utf-8")
            for attempt in range(attempts):
                try:
                    os.replace(tmp, self.state_path)
                    return
                except PermissionError:
                    if attempt == attempts - 1:
                        tmp.unlink(missing_ok=True)
                        raise
                    time.sleep(0.02 * (attempt + 1))

    def record(self, stage: str) -> StageRecord:
        return self.stages.setdefault(stage, StageRecord())

    # --- caching / resume -------------------------------------------------

    def is_current(self, stage: str, fingerprint: str) -> bool:
        """True when `stage` completed against exactly these inputs."""
        rec = self.stages.get(stage)
        return bool(rec and rec.status == DONE and rec.fingerprint == fingerprint)

    def start(self, stage: str, fingerprint: str) -> None:
        rec = self.record(stage)
        rec.status = RUNNING
        rec.fingerprint = fingerprint
        rec.started_at = time.time()
        rec.finished_at = None
        rec.error = ""
        self.save()
        self.emit(stage, 0.0, "started")

    def finish(self, stage: str, message: str = "") -> None:
        rec = self.record(stage)
        rec.status = DONE
        rec.finished_at = time.time()
        rec.seconds = rec.duration
        rec.message = message
        self.save()
        self.emit(stage, 1.0, message or "done")

    def fail(self, stage: str, error: str) -> None:
        rec = self.record(stage)
        rec.status = FAILED
        rec.finished_at = time.time()
        rec.seconds = rec.duration
        rec.error = error
        self.save()
        self.emit(stage, None, f"failed: {error}", level="error")

    def cancel_stage(self, stage: str) -> None:
        rec = self.record(stage)
        rec.status = CANCELLED
        rec.finished_at = time.time()
        rec.seconds = rec.duration
        self.save()
        self.emit(stage, None, "cancelled", level="warning")

    def reset(self, stages: Iterable[str]) -> None:
        """Force the given stages to re-run."""
        for s in stages:
            self.stages.pop(s, None)
        self.save()

    # --- progress ---------------------------------------------------------

    def emit(
        self,
        stage: str,
        fraction: Optional[float],
        message: str,
        level: str = "info",
    ) -> None:
        """Append one progress event. Never raises — progress must not fail a job."""
        event = {
            "t": time.time(),
            "stage": stage,
            "fraction": None if fraction is None else round(float(fraction), 4),
            "message": message,
            "level": level,
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with open(self.events_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        except OSError:
            pass

    def events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.events_path.is_file():
            return []
        out = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn final line from a crash mid-append
        return out[-limit:] if limit else out

    def progress(self) -> Dict[str, Any]:
        """Snapshot a client can poll."""
        done = sum(1 for r in self.stages.values() if r.status == DONE)
        return {
            "stages": {n: r.status for n, r in self.stages.items()},
            "completed": done,
            "total": len(self.stages),
            "running": next((n for n, r in self.stages.items() if r.status == RUNNING), None),
            "failed": [n for n, r in self.stages.items() if r.status == FAILED],
            "cancelled": self.cancel_requested(),
        }

    # --- cancellation -----------------------------------------------------

    def request_cancel(self, reason: str = "") -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.cancel_path.write_text(reason or "cancel requested", encoding="utf-8")

    def clear_cancel(self) -> None:
        self.cancel_path.unlink(missing_ok=True)

    def cancel_requested(self) -> bool:
        return self.cancel_path.exists()

    def raise_if_cancelled(self, stage: str = "") -> None:
        if self.cancel_requested():
            raise JobCancelled(f"cancelled{' during ' + stage if stage else ''}")

    def canceller(self) -> Callable[[], bool]:
        """A callable long-running stages can poll between units of work."""
        return self.cancel_requested


# --- fingerprinting -------------------------------------------------------

def _hash_update_path(h: "hashlib._Hash", path: Path) -> None:
    """Fold a file's identity into the hash without reading its contents.

    Name, size and nanosecond mtime. Hashing gigabytes of frames to decide
    whether to skip a stage would cost more than the stage itself.

    This is deliberately not the only invalidation signal: stage fingerprints
    also chain their upstream stages' fingerprints, so a rewritten-but-identical
    -looking artifact still invalidates everything downstream of it.
    """
    try:
        st = path.stat()
    except OSError:
        return
    h.update(path.name.encode("utf-8", "replace"))
    h.update(str(st.st_size).encode())
    h.update(str(st.st_mtime_ns).encode())


def fingerprint(
    params: Optional[Dict[str, Any]] = None,
    inputs: Optional[Iterable[Path | str]] = None,
) -> str:
    """Stable digest of a stage's parameters and input files.

    Directories are folded in as their sorted file listing, so adding or
    changing a frame invalidates the stages that consume it.
    """
    h = hashlib.sha256()
    h.update(json.dumps(params or {}, sort_keys=True, default=str).encode("utf-8"))
    for item in sorted(str(p) for p in (inputs or [])):
        p = Path(item)
        h.update(b"\0")
        h.update(item.encode("utf-8", "replace"))
        if p.is_dir():
            for child in sorted(p.iterdir(), key=lambda c: c.name):
                if child.is_file():
                    _hash_update_path(h, child)
        elif p.is_file():
            _hash_update_path(h, p)
    return h.hexdigest()

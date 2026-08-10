"""The lease queue, actually wired to workers (MPO-244).

`job_registry` grew `claim()` and `heartbeat()` when the queue was written, and
`tests/test_job_registry.py` has always covered them — directly, against the
registry. What no test covered was whether anything *called* them, and for a
long time nothing did: the service dispatched through a ThreadPoolExecutor and
the lease columns stayed NULL.

That gap is the subject here. These tests drive the pool the way the service
does and assert the properties the README claims: a restart does not strand
work, a dead worker's job comes back, and a live worker's job does not get
stolen out from under it.
"""

import threading
import time

import pytest

from job_registry import JobRecord, JobRegistry
from worker import Reaper, WorkerPool


def make_job(reg, job_id="j1", owner="owner", status="queued", **kw):
    rec = JobRecord(id=job_id, owner=owner, created_at=time.time(),
                    video_name="clip.mp4", quality="balanced", status=status, **kw)
    reg.add(rec)
    return rec


@pytest.fixture
def reg(tmp_path):
    return JobRegistry(tmp_path / "jobs.db", lease_seconds=60)


def drain(pool, reg, job_id, tries=200):
    for _ in range(tries):
        rec = reg.get(job_id)
        if rec is not None and rec.status in ("done", "failed", "cancelled"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"{job_id} never finished (status={reg.get(job_id)})")


# --- the basic contract ---------------------------------------------------

def test_a_queued_job_gets_claimed_and_completed(tmp_path, reg):
    ran = threading.Event()
    pool = WorkerPool(reg, tmp_path, lambda *a: ran.set(), max_workers=1).start()
    try:
        make_job(reg)
        pool.notify()
        assert drain(pool, reg, "j1").status == "done"
        assert ran.is_set()
    finally:
        pool.shutdown()


def test_a_failing_job_is_recorded_not_swallowed(tmp_path, reg):
    def boom(*_a):
        raise RuntimeError("colmap fell over")

    pool = WorkerPool(reg, tmp_path, boom, max_workers=1).start()
    try:
        make_job(reg)
        pool.notify()
        rec = drain(pool, reg, "j1")
        assert rec.status == "failed"
        assert "colmap fell over" in rec.error
    finally:
        pool.shutdown()


def test_a_cancelled_job_is_not_marked_failed(tmp_path, reg):
    """Cancellation is a request, not a fault; conflating them loses the
    distinction between 'you stopped it' and 'it broke'."""
    from job_state import JobCancelled

    def cancel(*_a):
        raise JobCancelled("stopped at checkpoint")

    pool = WorkerPool(reg, tmp_path, cancel, max_workers=1).start()
    try:
        make_job(reg)
        pool.notify()
        assert drain(pool, reg, "j1").status == "cancelled"
    finally:
        pool.shutdown()


def test_a_job_cancelled_before_it_starts_never_runs(tmp_path, reg):
    """A cancel that lands while the job is still queued should not cost a
    GPU slot proving the pipeline can stop."""
    from job_state import JobState

    ran = threading.Event()
    (tmp_path / "j1").mkdir()
    JobState(root=tmp_path / "j1").request_cancel("changed my mind")

    pool = WorkerPool(reg, tmp_path, lambda *a: ran.set(), max_workers=1).start()
    try:
        make_job(reg)
        pool.notify()
        assert drain(pool, reg, "j1").status == "cancelled"
        assert not ran.is_set(), "the runner was invoked for a cancelled job"
    finally:
        pool.shutdown()


# --- the failures that used to be invisible -------------------------------

def test_a_job_stranded_by_a_dead_process_is_requeued(reg):
    """The bug this module exists for.

    A process that died mid-job left the row `running` with a NULL lease.
    Nothing polled the table, and `NULL < now` is NULL rather than true, so
    the row was invisible to every recovery path and sat there forever.
    """
    make_job(reg, status="running")            # exactly what the old code left
    assert reg.get("j1").lease_expires is None

    assert reg.requeue_stale() == ["j1"]
    assert reg.get("j1").status == "queued"


def test_a_lapsed_lease_is_requeued_but_a_live_one_is_not(reg):
    """Recovery must not steal a job from a worker that is still working."""
    make_job(reg, "live")
    make_job(reg, "dead")
    reg.claim("worker-live")                   # takes the oldest: "live"
    reg.claim("worker-dead")

    # The dead worker's lease lapses; the live one keeps heartbeating.
    reg.update("dead", lease_expires=time.time() - 1)

    assert reg.requeue_stale() == ["dead"]
    assert reg.get("live").status == "running", "stole a job from a live worker"
    assert reg.get("dead").status == "queued"


def test_reap_expired_sees_a_null_lease(reg):
    """/health reported 0 expired leases while jobs were permanently stuck,
    because a NULL lease never satisfied `lease_expires < now`."""
    make_job(reg, status="running")
    assert reg.reap_expired() == ["j1"]


def test_a_restarted_pool_picks_up_stranded_work(tmp_path, reg):
    """End to end: strand a job, start a fresh pool, watch it recover."""
    make_job(reg, status="running")            # the previous process died here

    ran = threading.Event()
    pool = WorkerPool(reg, tmp_path, lambda *a: ran.set(), max_workers=1).start()
    try:
        reg.requeue_stale()                    # what the API does at startup
        pool.notify()
        assert drain(pool, reg, "j1").status == "done"
        assert ran.is_set()
    finally:
        pool.shutdown()


def test_the_heartbeat_keeps_a_long_job_from_being_reclaimed(tmp_path, tmp_path_factory):
    """Without a heartbeat, any job outliving one lease is reclaimed while it
    is still running — two workers in one job directory, which is worse than
    no recovery at all."""
    # A short lease so the test does not have to run for an hour.
    reg = JobRegistry(tmp_path / "jobs.db", lease_seconds=1)
    release = threading.Event()
    started = threading.Event()

    def slow(*_a):
        started.set()
        release.wait(timeout=10)

    import worker as worker_mod
    pool = WorkerPool(reg, tmp_path, slow, max_workers=1)
    # Beat faster than the 1s lease. The constant floor exists so production
    # does not beat every 5ms; the test needs to opt under it.
    original = worker_mod.MIN_HEARTBEAT_SECONDS
    worker_mod.MIN_HEARTBEAT_SECONDS = 0.1
    try:
        pool.start()
        make_job(reg)
        pool.notify()
        assert started.wait(timeout=5)

        # Well past the lease. A heartbeat is the only thing keeping it alive.
        time.sleep(1.5)
        assert reg.requeue_stale() == [], "a running job was reclaimed mid-flight"
        assert reg.get("j1").status == "running"
    finally:
        worker_mod.MIN_HEARTBEAT_SECONDS = original
        release.set()
        pool.shutdown()


def test_the_reaper_requeues_without_a_restart(tmp_path, reg):
    """A worker can die without the process dying with it; that should not
    wait for the next deploy to be noticed."""
    make_job(reg, status="running")
    woken = threading.Event()
    reaper = Reaper(reg, interval=0.05, on_requeue=woken.set).start()
    try:
        for _ in range(100):
            if reg.get("j1").status == "queued":
                break
            time.sleep(0.02)
        assert reg.get("j1").status == "queued"
        assert woken.is_set(), "the pool was never told there is work again"
    finally:
        reaper.stop()


# --- concurrency ----------------------------------------------------------

def test_two_workers_never_run_the_same_job(tmp_path, reg):
    concurrent = []
    lock = threading.Lock()
    live = {"n": 0}

    def track(*_a):
        with lock:
            live["n"] += 1
            concurrent.append(live["n"])
        time.sleep(0.05)
        with lock:
            live["n"] -= 1

    pool = WorkerPool(reg, tmp_path, track, max_workers=4).start()
    try:
        for i in range(8):
            make_job(reg, f"j{i}")
        pool.notify()
        for i in range(8):
            drain(pool, reg, f"j{i}")
        assert max(concurrent) <= 4, "more jobs ran at once than there are workers"
        assert len(concurrent) == 8, "a job ran twice or not at all"
    finally:
        pool.shutdown()


def test_api_only_mode_starts_no_workers(tmp_path, reg):
    """A split deployment has the API enqueue and worker containers drain."""
    ran = threading.Event()
    pool = WorkerPool(reg, tmp_path, lambda *a: ran.set(), max_workers=0).start()
    try:
        make_job(reg)
        pool.notify()
        time.sleep(0.3)
        assert reg.get("j1").status == "queued", "an API-only process claimed a job"
        assert not ran.is_set()
    finally:
        pool.shutdown()

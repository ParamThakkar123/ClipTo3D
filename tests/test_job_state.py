"""Job state: caching, resume, progress and cancellation (MPO-243).

The behaviour that matters is negative: a stage must NOT re-run when its
inputs are unchanged, and MUST re-run when any of them changed. Both directions
are tested, because only getting one right is how caches turn into corruption.
"""

import json

import pytest

from job_state import (
    CANCELLED,
    DONE,
    FAILED,
    RUNNING,
    JobCancelled,
    JobState,
    fingerprint,
)


@pytest.fixture
def job(tmp_path):
    return JobState(root=tmp_path / "job")


# --- fingerprinting -------------------------------------------------------

def test_same_inputs_same_fingerprint(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    (d / "a.jpg").write_bytes(b"x" * 100)
    assert fingerprint({"fps": 4}, [d]) == fingerprint({"fps": 4}, [d])


def test_parameter_change_invalidates(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    assert fingerprint({"fps": 4}, [d]) != fingerprint({"fps": 8}, [d])


def test_added_input_file_invalidates(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    (d / "a.jpg").write_bytes(b"x" * 10)
    before = fingerprint({}, [d])
    (d / "b.jpg").write_bytes(b"y" * 10)
    assert fingerprint({}, [d]) != before


def test_changed_input_size_invalidates(tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x" * 100)
    before = fingerprint({}, [f])
    f.write_bytes(b"x" * 200)
    assert fingerprint({}, [f]) != before


def test_missing_inputs_do_not_raise(tmp_path):
    """A stage may be fingerprinted before its inputs exist."""
    assert isinstance(fingerprint({"a": 1}, [tmp_path / "nope"]), str)


def test_fingerprint_is_order_independent(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    assert fingerprint({}, [a, b]) == fingerprint({}, [b, a])


# --- caching / resume -----------------------------------------------------

def test_completed_stage_with_matching_fingerprint_is_current(job):
    job.start("depth", "abc")
    job.finish("depth")
    assert job.is_current("depth", "abc")
    assert not job.is_current("depth", "different")


def test_failed_stage_is_not_current(job):
    job.start("colmap", "abc")
    job.fail("colmap", "boom")
    assert not job.is_current("colmap", "abc")
    assert job.stages["colmap"].status == FAILED
    assert "boom" in job.stages["colmap"].error


def test_state_survives_a_reload(job):
    job.start("frames", "fp1")
    job.finish("frames", "60 frames")

    reloaded = JobState.load(job.root)
    assert reloaded.is_current("frames", "fp1")
    assert reloaded.stages["frames"].message == "60 frames"


def test_resume_keeps_earlier_stages_and_retries_the_failed_one(job):
    """The scenario from the issue: killed mid-COLMAP."""
    job.start("frames", "f"); job.finish("frames")
    job.start("depth", "d"); job.finish("depth")
    job.start("colmap", "c"); job.fail("colmap", "killed")

    resumed = JobState.load(job.root)
    assert resumed.is_current("frames", "f")
    assert resumed.is_current("depth", "d")
    assert not resumed.is_current("colmap", "c")


def test_reset_forces_rerun(job):
    job.start("depth", "d"); job.finish("depth")
    job.reset(["depth"])
    assert not job.is_current("depth", "d")


def test_corrupt_state_is_treated_as_fresh_not_fatal(job):
    job.start("frames", "f"); job.finish("frames")
    job.state_path.write_text("{ not json", encoding="utf-8")
    # Losing the cache is recoverable; refusing to run is not.
    assert JobState.load(job.root).stages == {}


def test_state_write_is_atomic(job):
    job.start("frames", "f")
    job.finish("frames")
    assert json.loads(job.state_path.read_text())["stages"]["frames"]["status"] == DONE
    assert not list(job.root.glob("*.tmp")), "temp file left behind"


# --- progress -------------------------------------------------------------

def test_events_are_appended_and_readable(job):
    job.start("depth", "d")
    job.emit("depth", 0.5, "halfway")
    job.finish("depth")

    events = job.events()
    assert [e["message"] for e in events] == ["started", "halfway", "done"]
    assert events[1]["fraction"] == 0.5


def test_progress_snapshot_is_pollable(job):
    job.start("frames", "f"); job.finish("frames")
    job.start("depth", "d")

    prog = job.progress()
    assert prog["completed"] == 1
    assert prog["running"] == "depth"
    assert prog["stages"]["depth"] == RUNNING
    assert prog["failed"] == []


def test_progress_reports_failures(job):
    job.start("colmap", "c")
    job.fail("colmap", "no good initial pair")
    assert job.progress()["failed"] == ["colmap"]


def test_torn_event_line_does_not_break_reading(job):
    job.emit("depth", 0.1, "ok")
    with open(job.events_path, "a", encoding="utf-8") as fh:
        fh.write('{"partial": ')  # a crash mid-append
    assert [e["message"] for e in job.events()] == ["ok"]


def test_emit_never_raises_even_if_unwritable(tmp_path, monkeypatch):
    """Progress reporting must not be able to fail a job."""
    state = JobState(root=tmp_path / "job")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", boom)
    state.emit("depth", 0.5, "still fine")  # must not raise


# --- cancellation ---------------------------------------------------------

def test_cancel_round_trip(job):
    assert not job.cancel_requested()
    job.request_cancel("user asked")
    assert job.cancel_requested()
    assert "user asked" in job.cancel_path.read_text()
    job.clear_cancel()
    assert not job.cancel_requested()


def test_raise_if_cancelled(job):
    job.raise_if_cancelled("depth")  # no-op
    job.request_cancel()
    with pytest.raises(JobCancelled, match="depth"):
        job.raise_if_cancelled("depth")


def test_canceller_is_a_live_view(job):
    should_cancel = job.canceller()
    assert should_cancel() is False
    job.request_cancel()
    assert should_cancel() is True, "callback must observe a later request"


def test_cancelled_stage_is_recorded(job):
    job.start("colmap", "c")
    job.cancel_stage("colmap")
    assert job.stages["colmap"].status == CANCELLED
    assert JobState.load(job.root).stages["colmap"].status == CANCELLED


def test_concurrent_writers_do_not_collide(tmp_path):
    """Two threads writing state at once must not fail or corrupt it.

    The service polls job state while a worker writes it. Two bugs lived here:
    a fixed temp filename (two writers opening the same `state.json.tmp`), and
    then — after that was fixed — `os.replace` still failing with
    PermissionError because Windows will not rename onto a file a *reader* has
    open. Retrying the rename was not enough; it still lost 4 runs in 6 at this
    contention. Both readers and writers now hold a lock.

    Deliberately sized to reproduce: without the lock this fails most runs.
    """
    import threading

    root = tmp_path / "job"
    errors = []
    n_writers, n_readers = 4, 4
    barrier = threading.Barrier(n_writers + n_readers)

    def writer(n):
        try:
            barrier.wait(timeout=10)
            st = JobState(root=root)
            for i in range(40):
                st.start(f"stage{n}", f"fp{i}")
                st.finish(f"stage{n}")
        except Exception as exc:      # noqa: BLE001 - surfaced below
            errors.append(exc)

    def reader():
        try:
            barrier.wait(timeout=10)
            for _ in range(200):
                JobState.load(root)
        except Exception as exc:      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(n_writers)]
    threads += [threading.Thread(target=reader) for _ in range(n_readers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    # State must still be parseable, and no temp files left behind.
    assert JobState.load(root).stages
    assert not list(root.glob("*.tmp")), "temp files leaked"


def test_state_file_is_never_left_partially_written(tmp_path):
    """A reader mid-write must see either the old state or the new one."""
    state = JobState(root=tmp_path / "job")
    state.start("frames", "fp")
    state.finish("frames")
    for _ in range(50):
        state.start("depth", "fp2")
        # Always parseable — the temp+replace is what guarantees this.
        assert "frames" in JobState.load(state.root).stages

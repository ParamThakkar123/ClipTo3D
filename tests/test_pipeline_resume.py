"""Caching, resume and cancellation through the real orchestrator (MPO-243).

Uses stubbed stage functions rather than real COLMAP/depth, so the thing under
test is the orchestration logic — which stages run, which are skipped, and what
happens after a failure or a cancel.
"""

import pytest

import pipeline
from job_state import JobCancelled, JobState


@pytest.fixture
def stub_stages(monkeypatch, tmp_path):
    """Replace every stage with a recorder that creates its expected output."""
    calls = []

    def make(name):
        def run(job, *a, **k):
            calls.append(name)
            if name == "frames":
                job.frames.mkdir(parents=True, exist_ok=True)
                (job.frames / "frame_0001.jpg").write_bytes(b"jpg")
            elif name == "keyframe":
                job.keyframes.mkdir(parents=True, exist_ok=True)
                (job.keyframes / "frame_0001.jpg").write_bytes(b"jpg")
                job.keyframes_manifest.write_text('{"n_kept": 1}')
            elif name == "depth":
                job.depth.mkdir(parents=True, exist_ok=True)
                (job.depth / "frame_0001_depth.npy").write_bytes(b"npy")
            elif name == "colmap":
                (job.colmap_sparse / "0").mkdir(parents=True, exist_ok=True)
                (job.colmap_sparse / "0" / "cameras.txt").write_text("# cameras")
            elif name == "dataset":
                job.dataset.mkdir(parents=True, exist_ok=True)
                job.transforms_json.write_text("{}")
            elif name == "fuse":
                job.cloud.mkdir(parents=True, exist_ok=True)
                job.fused_ply.write_bytes(b"ply")
            elif name == "mesh":
                job.cloud.mkdir(parents=True, exist_ok=True)
                job.mesh_npz.write_bytes(b"npz")
            elif name == "export":
                job.export.mkdir(parents=True, exist_ok=True)
                (job.export / "cloud.glb").write_bytes(b"glb")
        return run

    for name in ("frames", "keyframe", "depth", "colmap", "dataset", "fuse", "mesh", "export"):
        monkeypatch.setattr(pipeline, f"stage_{name}", make(name))
    return calls


@pytest.fixture
def video(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"fake video")
    return v


def test_first_run_executes_every_stage(tmp_path, stub_stages, video):
    pipeline.run_pipeline(video=video, job_root=tmp_path / "job")
    assert stub_stages == ["frames", "keyframe", "depth", "colmap", "dataset", "fuse",
                       "mesh", "export"]


def test_second_run_is_a_no_op(tmp_path, stub_stages, video):
    job_root = tmp_path / "job"
    pipeline.run_pipeline(video=video, job_root=job_root)
    stub_stages.clear()
    pipeline.run_pipeline(video=video, job_root=job_root)
    assert stub_stages == [], "unchanged inputs must not re-run anything"


def test_force_redoes_everything(tmp_path, stub_stages, video):
    job_root = tmp_path / "job"
    pipeline.run_pipeline(video=video, job_root=job_root)
    stub_stages.clear()
    pipeline.run_pipeline(video=video, job_root=job_root, force=True)
    assert len(stub_stages) == 8


def test_changing_a_stage_parameter_reruns_only_what_it_affects(tmp_path, stub_stages, video):
    job_root = tmp_path / "job"
    pipeline.run_pipeline(video=video, job_root=job_root)
    stub_stages.clear()

    # A depth-only knob: COLMAP does not consume depth, so it must not re-run.
    pipeline.run_pipeline(video=video, job_root=job_root, depth_encoder="vits")
    assert "depth" in stub_stages
    assert "colmap" not in stub_stages
    assert "frames" not in stub_stages


def test_fuse_reruns_when_depth_changes(tmp_path, stub_stages, video):
    """Fusion consumes depth, so a depth change must invalidate it."""
    job_root = tmp_path / "job"
    pipeline.run_pipeline(video=video, job_root=job_root)
    stub_stages.clear()
    pipeline.run_pipeline(video=video, job_root=job_root, depth_encoder="vits")
    assert "fuse" in stub_stages


def test_failure_is_recorded_and_the_run_resumes_there(tmp_path, stub_stages, video, monkeypatch):
    """The issue's scenario: killed mid-COLMAP."""
    job_root = tmp_path / "job"

    def exploding_colmap(job, *a, **k):
        stub_stages.append("colmap")
        raise RuntimeError("mapper died")

    monkeypatch.setattr(pipeline, "stage_colmap", exploding_colmap)
    with pytest.raises(RuntimeError, match="mapper died"):
        pipeline.run_pipeline(video=video, job_root=job_root)

    assert stub_stages == ["frames", "keyframe", "depth", "colmap"]
    state = JobState.load(job_root)
    assert state.progress()["failed"] == ["colmap"]

    # Repair COLMAP and resume: frames and depth must not be redone.
    stub_stages.clear()

    def fixed_colmap(job, *a, **k):
        stub_stages.append("colmap")
        (job.colmap_sparse / "0").mkdir(parents=True, exist_ok=True)
        (job.colmap_sparse / "0" / "cameras.txt").write_text("# cameras")

    monkeypatch.setattr(pipeline, "stage_colmap", fixed_colmap)
    pipeline.run_pipeline(video=video, job_root=job_root)

    assert "frames" not in stub_stages, "resume re-extracted frames"
    assert "depth" not in stub_stages, "resume re-ran depth"
    assert stub_stages == ["colmap", "dataset", "fuse", "mesh", "export"]


def test_cancellation_stops_between_stages(tmp_path, stub_stages, video, monkeypatch):
    job_root = tmp_path / "job"

    def depth_then_cancel(job, *a, **k):
        stub_stages.append("depth")
        (job.depth).mkdir(parents=True, exist_ok=True)
        (job.depth / "frame_0001_depth.npy").write_bytes(b"npy")
        JobState(root=job.root).request_cancel("user")

    monkeypatch.setattr(pipeline, "stage_depth", depth_then_cancel)
    with pytest.raises(JobCancelled):
        pipeline.run_pipeline(video=video, job_root=job_root)

    assert "colmap" not in stub_stages, "kept going after cancellation"
    state = JobState.load(job_root)
    assert state.stages["colmap"].status == "cancelled"
    # Completed work is retained so the job can be resumed.
    assert state.is_current("frames", state.stages["frames"].fingerprint)


def test_a_stale_cancel_sentinel_does_not_kill_the_next_run(tmp_path, stub_stages, video):
    job_root = tmp_path / "job"
    JobState(root=job_root).request_cancel("old")
    pipeline.run_pipeline(video=video, job_root=job_root)
    assert len(stub_stages) == 8


def test_progress_events_are_written(tmp_path, stub_stages, video):
    job_root = tmp_path / "job"
    pipeline.run_pipeline(video=video, job_root=job_root)
    events = JobState.load(job_root).events()
    stages_seen = {e["stage"] for e in events}
    assert {"frames", "depth", "colmap", "fuse"} <= stages_seen
    assert any(e["message"] == "complete" for e in events)


def test_preexisting_outputs_are_adopted_not_recomputed(tmp_path, stub_stages, video):
    """Upgrading an old job must not redo an hour of COLMAP."""
    job_root = tmp_path / "job"
    job = pipeline.JobPaths(job_root).ensure()
    # Simulate a job from before job state existed.
    (job.frames / "frame_0001.jpg").write_bytes(b"jpg")

    pipeline.run_pipeline(video=video, job_root=job_root)
    assert "frames" not in stub_stages, "re-extracted frames that were already there"

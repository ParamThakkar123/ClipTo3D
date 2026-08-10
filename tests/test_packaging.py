"""The project must stay buildable as a package (MPO-245).

`package = true` means `uv sync` builds a wheel, so anything the build backend
needs has to be present wherever that happens — including inside the worker
image, whose dependency layer copies only a few files. A missing one fails
every fresh image build while working fine locally, which is the worst place
for this to surface.
"""

import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
DOCKERFILE = REPO / "docker" / "Dockerfile"


def test_declared_readme_exists():
    readme = PYPROJECT["project"].get("readme")
    if readme:
        assert (REPO / readme).is_file(), f"pyproject declares readme={readme!r}, which is missing"


def test_dockerfile_dependency_layer_copies_everything_the_build_needs():
    """The exact regression: hatchling failed on `Readme file does not exist`
    because the dependency layer copied pyproject.toml and uv.lock only."""
    dockerfile = (REPO / "docker" / "Dockerfile").read_text(encoding="utf-8")
    dep_copy = next(
        (ln for ln in dockerfile.splitlines()
         if ln.startswith("COPY ") and "pyproject.toml" in ln),
        None,
    )
    assert dep_copy is not None, "no dependency-layer COPY found"

    needed = {"pyproject.toml", "uv.lock"}
    readme = PYPROJECT["project"].get("readme")
    if readme:
        needed.add(readme)
    missing = {f for f in needed if f not in dep_copy}
    assert not missing, f"the image's dependency layer does not copy {sorted(missing)}"


def test_console_script_target_is_importable():
    script = PYPROJECT["project"]["scripts"]["clipto3d"]
    module, _, func = script.partition(":")
    mod = __import__(module)
    assert callable(getattr(mod, func)), f"{script} is not callable"


def test_wheel_includes_every_top_level_stage_package():
    """A package omitted here imports locally (repo root on sys.path) and is
    absent from the installed wheel."""
    include = PYPROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]["include"]
    for pkg in ("depth_estimation", "fusion", "neural_reconstruction",
                "point_clouds", "structure_from_motion"):
        if (REPO / pkg).is_dir():
            assert any(pkg in entry for entry in include), f"{pkg}/ missing from the wheel"


def test_worker_image_installs_every_extra_the_default_pipeline_needs():
    """The mesh stage is in DEFAULT_STAGES, so the image must be able to run it.

    Found the hard way: the image installed only `cuda` and `splat`, so the
    mesh stage was skipped and no USDZ was ever produced — while the job still
    reported success with a point cloud.
    """
    import pipeline

    dockerfile = (REPO / "docker" / "Dockerfile").read_text(encoding="utf-8")
    sync_line = next(
        (ln for ln in dockerfile.splitlines() if "uv sync --frozen" in ln), None)
    assert sync_line is not None, "no uv sync line in the Dockerfile"

    # Which extras each default stage requires to do its job.
    needed_by_stage = {"mesh": "mesh", "export": "ar"}
    for stage, extra in needed_by_stage.items():
        if stage in pipeline.DEFAULT_STAGES:
            assert f"--extra {extra}" in sync_line, (
                f"stage {stage!r} is in DEFAULT_STAGES but the image does not "
                f"install --extra {extra}"
            )


def test_mesh_stage_reports_a_missing_extra_rather_than_skipping(monkeypatch, tmp_path):
    """Silently producing no AR assets is worse than failing."""
    import builtins

    import pipeline
    from job_paths import JobPaths

    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "skimage":
            raise ModuleNotFoundError("No module named 'skimage'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    job = JobPaths(tmp_path / "job").ensure()
    with pytest.raises(ModuleNotFoundError, match="--extra mesh"):
        pipeline.stage_mesh(job, resolution=32, stride=2)


def test_worker_image_pins_a_python_with_hashed_cuda_wheels():
    """PyTorch's CUDA index does not publish hashes for every wheel.

    `torch==2.13.0+cu130` carries sha256 fragments for cp311 and cp312 only;
    cp313 and cp314 have none, so uv.lock cannot record them and
    `uv sync --frozen --extra cuda` fails with a hash mismatch on a newer
    interpreter. Left unpinned, the image's Python follows uv's default and the
    build breaks the day that default moves.
    """
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "ARG PYTHON_VERSION=3.12" in text, "the CUDA-safe Python pin is gone"
    assert "UV_PYTHON=${PYTHON_VERSION}" in text, "the pin is declared but not used"


def test_cuda_wheels_in_the_lock_are_hashed_for_the_pinned_python():
    """Guards the pin against the lock changing underneath it.

    If a future relock loses hashes for cp312 too, this fails here rather than
    twenty minutes into a container build.
    """
    import re

    lock = (REPO / "uv.lock").read_text(encoding="utf-8")
    m = re.search(r'name = "torch"\nversion = "([^"]*cu\d+)"(.*?)(?=\n\[\[package\]\])',
                  lock, re.S)
    if m is None:
        pytest.skip("no CUDA torch in the lock")

    linux312 = [ln for ln in m.group(2).splitlines()
                if "cp312" in ln and "linux" in ln and ".whl" in ln]
    assert linux312, "no cp312 linux wheel for CUDA torch; the Dockerfile pin is wrong"
    assert all("hash = " in ln for ln in linux312), (
        "CUDA torch wheels for the pinned Python lost their hashes; "
        "uv sync --frozen --extra cuda will fail")

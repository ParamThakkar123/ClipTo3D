"""The base install must stay headless (MPO-234).

Every way to look at a result used to be a blocking desktop window, and the GL
stack sat in the base dependency set — so a container or CI install pulled in
matplotlib/pyglet/PyOpenGL it could never use.

These tests pin both halves of "done": no GL packages in the base dependency
set, and no module the pipeline touches importing one.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Anything that needs a display or a GL context.
GL_PACKAGES = {"matplotlib", "pyglet", "pyopengl", "pyopengl-accelerate", "trimesh"}

# Modules the pipeline actually drives. fusion.view is deliberately excluded:
# it is the one dev-only preview, and it lives behind the `viewer` extra.
PIPELINE_MODULES = [
    "pipeline",
    "job_paths",
    "frames",
    "audio",
    "colmap_io",
    "pointcloud_io",
    "depth_scale",
    "checkpoints",
    "convert_colmap_to_gs",
    "fusion.fuse",
    "point_clouds.point_clouds_file",
    "structure_from_motion.sfm",
]


def base_dependencies():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["dependencies"]


def requirement_name(spec: str) -> str:
    for sep in (">=", "==", "<=", "~=", ">", "<", "[", ";", " "):
        spec = spec.split(sep)[0]
    return spec.strip().lower()


def test_base_dependencies_have_no_gl_packages():
    offenders = {requirement_name(d) for d in base_dependencies()} & GL_PACKAGES
    assert not offenders, f"GL packages must live in the viewer extra, not base: {offenders}"


def test_viewer_extra_still_provides_them():
    """They were moved, not deleted — the debug preview must still be installable."""
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    viewer = {requirement_name(d) for d in data["project"]["optional-dependencies"]["viewer"]}
    assert GL_PACKAGES <= viewer, f"viewer extra is missing {GL_PACKAGES - viewer}"


@pytest.mark.parametrize("module", PIPELINE_MODULES)
def test_pipeline_module_imports_no_gl(module):
    """Import in a subprocess and assert nothing GL-related landed in sys.modules.

    A subprocess is the point: an in-process check would pass simply because
    another test already imported the module.
    """
    code = (
        "import sys, importlib\n"
        f"importlib.import_module({module!r})\n"
        "gl = {'matplotlib', 'pyglet', 'OpenGL', 'trimesh'}\n"
        "hit = sorted(gl & {m.split('.')[0] for m in sys.modules})\n"
        "print(','.join(hit))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True
    )
    assert proc.returncode == 0, f"{module} failed to import:\n{proc.stderr}"
    assert proc.stdout.strip() == "", f"{module} pulled in GL modules: {proc.stdout.strip()}"


def test_no_blocking_window_calls_outside_the_debug_previewer():
    """`.show()` / `plt.show()` must not reappear in pipeline code."""
    offenders = []
    for path in REPO.rglob("*.py"):
        rel = path.relative_to(REPO)
        parts = set(rel.parts)
        if ".venv" in parts or "tests" in parts or "depth_anything_v2" in parts:
            continue
        if rel.as_posix() == "fusion/view.py":  # the one sanctioned preview
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "plt.show(" in stripped or ".show()" in stripped:
                offenders.append(f"{rel.as_posix()}:{lineno}: {stripped}")
    assert not offenders, "blocking window calls outside fusion/view.py:\n" + "\n".join(offenders)

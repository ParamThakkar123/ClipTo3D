"""End-to-end integration test against real COLMAP output.

Slow (runs an actual reconstruction) and requires the `colmap` binary, so it is
skipped unless COLMAP is on PATH and `CLIPTO3D_RUN_INTEGRATION=1` is set:

    CLIPTO3D_RUN_INTEGRATION=1 uv run pytest tests/test_integration_colmap.py -v

What it establishes that the unit tests cannot:

* `colmap_io` parses genuine COLMAP output, not just hand-written fixtures.
  Notably COLMAP's default camera model is SIMPLE_RADIAL, the exact model the
  old `gbr.intrinsics_from_camera` misread (MPO-228).
* Recovered camera poses match ground truth up to a similarity transform, which
  would fail outright if the quaternion column order were wrong.
* The depth-scale fit (MPO-225) recovers consistent geometry from per-frame
  affine-scrambled disparity, and the fused cloud lands on the true surfaces.
* The stages compose through `JobPaths` without manual path fixing (MPO-224).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_scene import (  # noqa: E402
    build_scene,
    default_planes,
    umeyama_similarity,
    write_synthetic_disparity,
)

pytestmark = [
    pytest.mark.skipif(
        shutil.which("colmap") is None, reason="COLMAP binary not on PATH"
    ),
    pytest.mark.skipif(
        os.environ.get("CLIPTO3D_RUN_INTEGRATION") != "1",
        reason="set CLIPTO3D_RUN_INTEGRATION=1 to run (slow)",
    ),
]

N_VIEWS = 14


@pytest.fixture(scope="module")
def reconstruction(tmp_path_factory):
    """Render the scene, run COLMAP once, and return everything downstream needs."""
    from job_paths import JobPaths
    from structure_from_motion.sfm import run_colmap_fast

    root = tmp_path_factory.mktemp("integ")
    job = JobPaths(root).ensure()

    scene = build_scene(job.frames, n_views=N_VIEWS, with_depth=True)
    affine = write_synthetic_disparity(scene, job.depth, seed=1)
    run_colmap_fast(job.frames, job.colmap, None, use_gpu=True)

    return {"job": job, "scene": scene, "affine": affine}


def _paired_poses(images, scene):
    idx = {n: i for i, n in enumerate(scene["names"])}
    rec_c, gt_c, rec_R, gt_R = [], [], [], []
    for im in images.values():
        j = idx.get(Path(im.name).name)
        if j is None:
            continue
        rec_c.append(im.camera_center)
        gt_c.append(scene["centres"][j])
        rec_R.append(im.R)
        gt_R.append(scene["rotations"][j])
    return np.array(rec_c), np.array(gt_c), rec_R, gt_R


class TestRealColmapParsing:
    def test_most_views_register(self, reconstruction):
        from colmap_io import find_model_dir, read_model

        cams, imgs, pts = read_model(find_model_dir(reconstruction["job"].colmap))
        assert len(imgs) >= N_VIEWS - 2, f"only {len(imgs)}/{N_VIEWS} registered"
        assert len(pts) > 500
        assert len(cams) == 1

    def test_camera_model_is_parsed(self, reconstruction):
        """COLMAP defaults to SIMPLE_RADIAL, the model the old reader got wrong."""
        from colmap_io import find_model_dir, read_model

        cams, _, _ = read_model(find_model_dir(reconstruction["job"].colmap))
        cam = next(iter(cams.values()))
        fx, fy, cx, cy = cam.intrinsics
        assert cam.width == 640 and cam.height == 480
        assert isinstance(cam.width, int)
        # SIMPLE_* models are isotropic, and the principal point sits near centre.
        assert fx == pytest.approx(fy)
        assert cx == pytest.approx(320.0, abs=40.0)
        assert cy == pytest.approx(240.0, abs=40.0)

    def test_observations_link_to_points(self, reconstruction):
        from colmap_io import find_model_dir, read_model

        _, imgs, pts = read_model(find_model_dir(reconstruction["job"].colmap))
        img = next(iter(imgs.values()))
        assert len(img.xys) > 100
        linked = [p for p in img.point3D_ids if p != -1]
        assert len(linked) > 50
        assert all(int(p) in pts for p in linked)


class TestPoseRecovery:
    def test_camera_centres_match_ground_truth(self, reconstruction):
        """Fails loudly if the quaternion column order is wrong."""
        from colmap_io import find_model_dir, read_model

        _, imgs, _ = read_model(find_model_dir(reconstruction["job"].colmap))
        rec_c, gt_c, _, _ = _paired_poses(imgs, reconstruction["scene"])

        _s, _R, _T, rmse = umeyama_similarity(rec_c, gt_c)
        span = float(np.linalg.norm(gt_c.max(axis=0) - gt_c.min(axis=0)))
        assert rmse / span < 0.05, f"centre rmse {rmse:.4f} is {100*rmse/span:.1f}% of span"

    def test_rotations_match_ground_truth(self, reconstruction):
        from colmap_io import find_model_dir, read_model

        _, imgs, _ = read_model(find_model_dir(reconstruction["job"].colmap))
        rec_c, gt_c, rec_R, gt_R = _paired_poses(imgs, reconstruction["scene"])
        _s, R_align, _T, _rmse = umeyama_similarity(rec_c, gt_c)

        errs = []
        for Rr, Rg in zip(rec_R, gt_R):
            R_err = Rg @ (Rr @ R_align.T).T
            errs.append(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))))
        # A scrambled quaternion gives tens of degrees of error.
        assert np.median(errs) < 10.0, f"median rotation error {np.median(errs):.2f} deg"


class TestDatasetExport:
    def test_transforms_json_is_well_formed(self, reconstruction):
        import json

        from convert_colmap_to_gs import colmap_to_transforms

        job = reconstruction["job"]
        out = colmap_to_transforms(job.colmap, job.frames, job.dataset)
        d = json.loads(out.read_text())

        assert isinstance(d["w"], int) and isinstance(d["h"], int)
        assert d["w"] == 640 and d["h"] == 480
        assert d["fl_x"] > 0
        assert len(d["frames"]) >= N_VIEWS - 2
        for fr in d["frames"]:
            m = np.array(fr["transform_matrix"])
            assert m.shape == (4, 4)
            np.testing.assert_allclose(m[3], [0, 0, 0, 1], atol=1e-9)
            # Upper-left block stays a rotation under the axis flip.
            assert np.isclose(abs(np.linalg.det(m[:3, :3])), 1.0, atol=1e-6)
        assert (job.dataset / "images").is_dir()


class TestDepthScaleAndFusion:
    """The MPO-225 payload: per-frame affine disparity -> one consistent cloud."""

    def test_scale_fit_recovers_per_frame_affine(self, reconstruction):
        from colmap_io import find_model_dir, read_model
        from depth_scale import fit_and_convert

        job = reconstruction["job"]
        _cams, imgs, pts = read_model(find_model_dir(job.colmap))
        scene = reconstruction["scene"]
        idx = {n: i for i, n in enumerate(scene["names"])}

        errors, fitted = [], 0
        for im in imgs.values():
            j = idx.get(Path(im.name).name)
            if j is None:
                continue
            disp = np.load(job.depth / f"{Path(im.name).stem}_depth.npy").astype(np.float64)
            metric, fit = fit_and_convert(disp, im, pts, min_points=20)
            if metric is None:
                continue
            fitted += 1
            gt_z = scene["depths"][j]
            # COLMAP's world scale is arbitrary, so compare depth *ratios* by
            # normalizing each map by its own median over shared valid pixels.
            valid = np.isfinite(gt_z) & np.isfinite(metric)
            if valid.sum() < 1000:
                continue
            a = metric[valid] / np.median(metric[valid])
            b = gt_z[valid] / np.median(gt_z[valid])
            errors.append(float(np.median(np.abs(a - b))))

        assert fitted >= N_VIEWS - 4, f"only {fitted} frames fitted"
        assert np.median(errors) < 0.06, f"median normalized depth error {np.median(errors):.4f}"

    def test_fused_cloud_lands_on_the_true_surfaces(self, reconstruction):
        from fusion.fuse import fuse
        from pointcloud_io import read_ply

        job = reconstruction["job"]
        out = fuse(
            colmap_dir=job.colmap, frames_dir=job.frames, depth_dir=job.depth,
            out_ply=job.fused_ply, voxel_frac=0.004, min_views=2, stride=4,
        )
        pts, cols = read_ply(out)
        assert len(pts) > 2_000
        assert cols is not None and len(cols) == len(pts)
        assert np.isfinite(pts).all()

        # Bring the cloud into world coordinates via the camera-centre similarity,
        # then measure distance to the nearest ground-truth plane.
        from colmap_io import find_model_dir, read_model

        _, imgs, _ = read_model(find_model_dir(job.colmap))
        rec_c, gt_c, _, _ = _paired_poses(imgs, reconstruction["scene"])
        s, R, T, _ = umeyama_similarity(rec_c, gt_c)
        world = (s * (R @ pts.T.astype(np.float64)).T) + T

        dists = []
        for plane in default_planes():
            c = plane.corners
            n = np.cross(c[1] - c[0], c[3] - c[0])
            n = n / np.linalg.norm(n)
            dists.append(np.abs((world - c[0]) @ n))
        nearest = np.min(np.stack(dists, axis=1), axis=1)

        scene_span = float(np.linalg.norm(gt_c.max(axis=0) - gt_c.min(axis=0)))
        # Most points should sit close to a real surface; the tail is depth-model
        # error at plane boundaries, which is expected.
        assert np.median(nearest) < 0.05 * scene_span, (
            f"median distance to nearest true plane {np.median(nearest):.4f} "
            f"(scene span {scene_span:.3f})"
        )
        assert float((nearest < 0.1 * scene_span).mean()) > 0.7

    def test_voxel_size_controls_output_size(self, reconstruction):
        """Point count must follow voxel size (i.e. scene extent), not frame count.

        Voxel sizes are given as fractions of the sparse-cloud radius. Absolute
        sizes are meaningless across reconstructions: COLMAP solved this
        true-radius-1.0 scene at radius 9.6, so a nominal 0.05 voxel was 0.5% of
        the scene and merged almost nothing.
        """
        from fusion.fuse import fuse
        from pointcloud_io import read_ply

        job = reconstruction["job"]
        coarse, _ = read_ply(
            fuse(colmap_dir=job.colmap, frames_dir=job.frames, depth_dir=job.depth,
                 out_ply=job.cloud / "coarse.ply", voxel_frac=0.05, min_views=1, stride=6)
        )
        fine, _ = read_ply(
            fuse(colmap_dir=job.colmap, frames_dir=job.frames, depth_dir=job.depth,
                 out_ply=job.cloud / "fine.ply", voxel_frac=0.005, min_views=1, stride=6)
        )
        assert len(coarse) > 0 and len(fine) > 0
        # A 10x smaller voxel must yield substantially more points.
        assert len(fine) > 3 * len(coarse), f"coarse={len(coarse)} fine={len(fine)}"

    def test_absolute_voxel_size_overrides_the_fraction(self, reconstruction):
        from fusion.fuse import fuse
        from pointcloud_io import read_ply

        job = reconstruction["job"]
        a, _ = read_ply(
            fuse(colmap_dir=job.colmap, frames_dir=job.frames, depth_dir=job.depth,
                 out_ply=job.cloud / "abs.ply", voxel_frac=0.5, voxel_size=0.05,
                 min_views=1, stride=8)
        )
        b, _ = read_ply(
            fuse(colmap_dir=job.colmap, frames_dir=job.frames, depth_dir=job.depth,
                 out_ply=job.cloud / "frac.ply", voxel_frac=0.5, min_views=1, stride=8)
        )
        # The absolute 0.05 is far finer than 0.5 * radius, so it keeps more points.
        assert len(a) > len(b)


class TestPipelineWiring:
    def test_stages_compose_through_job_paths(self, reconstruction):
        """MPO-224: dataset + fuse run off the shared layout with no path fixing."""
        from pipeline import run_pipeline

        job = reconstruction["job"]
        run_pipeline(
            video=None, job_root=job.root, stages=["dataset", "fuse"],
            voxel_size=0.02, min_views=1, stride=6, force=True,
        )
        assert job.transforms_json.is_file()
        assert job.fused_ply.is_file()

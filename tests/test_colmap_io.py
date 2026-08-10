import numpy as np
import pytest

from colmap_io import (
    Camera,
    ColmapFormatError,
    find_model_dir,
    images_by_name,
    qvec2rotmat,
    read_cameras_text,
    read_images_text,
    read_points3D_text,
    sole_camera,
)

CAMERAS_TXT = """# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 PINHOLE 1920 1080 1000.0 1001.0 960.0 540.0
"""

# Image 2 has an EMPTY POINTS2D line, which is what COLMAP writes for an image
# with no triangulated points. The previous parser filtered blank lines before
# pairing records two-at-a-time, so everything after image 2 was shifted by one.
IMAGES_TXT_WITH_EMPTY_TRACK = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 frame_000001.jpg
100.0 200.0 5 150.0 250.0 -1
2 1.0 0.0 0.0 0.0 1.0 2.0 3.0 1 frame_000002.jpg

3 1.0 0.0 0.0 0.0 4.0 5.0 6.0 1 frame_000003.jpg
10.0 20.0 7
"""

POINTS3D_TXT = """# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
5 1.0 2.0 3.0 255 128 0 0.5 1 0 3 1
7 -1.0 0.0 4.0 10 20 30 0.25 3 0
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


class TestImagePairing:
    """Regression coverage for the POINTS2D desync (MPO-228)."""

    def test_empty_points2d_line_does_not_desync_later_images(self, tmp_path):
        images = read_images_text(_write(tmp_path, "images.txt", IMAGES_TXT_WITH_EMPTY_TRACK))

        assert sorted(images) == [1, 2, 3]

        # The image after the empty-track one must keep its own pose.
        assert images[3].name == "frame_000003.jpg"
        np.testing.assert_allclose(images[3].tvec, [4.0, 5.0, 6.0])

        # And the empty-track image itself parses with no observations.
        assert images[2].name == "frame_000002.jpg"
        np.testing.assert_allclose(images[2].tvec, [1.0, 2.0, 3.0])
        assert images[2].xys.shape == (0, 2)
        assert images[2].point3D_ids.shape == (0,)

    def test_observations_parsed(self, tmp_path):
        images = read_images_text(_write(tmp_path, "images.txt", IMAGES_TXT_WITH_EMPTY_TRACK))
        np.testing.assert_allclose(images[1].xys, [[100.0, 200.0], [150.0, 250.0]])
        np.testing.assert_array_equal(images[1].point3D_ids, [5, -1])

    def test_name_with_spaces_is_preserved(self, tmp_path):
        txt = (
            "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 my frame 01.jpg\n"
            "1.0 2.0 3\n"
        )
        images = read_images_text(_write(tmp_path, "images.txt", txt))
        assert images[1].name == "my frame 01.jpg"

    def test_truncated_pose_line_raises(self, tmp_path):
        txt = "1 1.0 0.0 0.0\n\n"
        with pytest.raises(ColmapFormatError):
            read_images_text(_write(tmp_path, "images.txt", txt))

    def test_bad_points2d_arity_raises(self, tmp_path):
        txt = "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 a.jpg\n1.0 2.0\n"
        with pytest.raises(ColmapFormatError):
            read_images_text(_write(tmp_path, "images.txt", txt))


class TestQuaternion:
    def test_identity(self):
        np.testing.assert_allclose(qvec2rotmat([1, 0, 0, 0]), np.eye(3))

    def test_ninety_degrees_about_z_uses_wxyz_order(self):
        """Guards the gbr.py bug: reading QW QX QY QZ as qx qy qz qw."""
        s = np.sqrt(0.5)
        R = qvec2rotmat([s, 0.0, 0.0, s])  # w, x, y, z
        np.testing.assert_allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
        # x axis maps to y axis
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_is_orthonormal(self):
        R = qvec2rotmat([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)

    def test_unnormalized_input_is_normalized(self):
        np.testing.assert_allclose(qvec2rotmat([2, 0, 0, 0]), np.eye(3))

    def test_degenerate_quaternion_falls_back_to_identity(self):
        np.testing.assert_allclose(qvec2rotmat([0, 0, 0, 0]), np.eye(3))


class TestIntrinsics:
    def test_pinhole(self):
        cam = Camera(1, "PINHOLE", 1920, 1080, np.array([1000.0, 1001.0, 960.0, 540.0]))
        assert cam.intrinsics == (1000.0, 1001.0, 960.0, 540.0)

    def test_simple_pinhole(self):
        cam = Camera(1, "SIMPLE_PINHOLE", 640, 480, np.array([500.0, 320.0, 240.0]))
        assert cam.intrinsics == (500.0, 500.0, 320.0, 240.0)

    def test_simple_radial_reads_f_cx_cy_not_fx_fy_cx(self):
        """SIMPLE_RADIAL is `f, cx, cy, k`. The old gbr reader treated it as
        `fx, fy, cx, cy`, yielding fy=cx and cy=k."""
        cam = Camera(1, "SIMPLE_RADIAL", 640, 480, np.array([500.0, 320.0, 240.0, 0.01]))
        assert cam.intrinsics == (500.0, 500.0, 320.0, 240.0)

    def test_opencv(self):
        cam = Camera(
            1, "OPENCV", 640, 480, np.array([500.0, 501.0, 320.0, 240.0, 0.1, 0.2, 0.3, 0.4])
        )
        assert cam.intrinsics == (500.0, 501.0, 320.0, 240.0)

    def test_unknown_model_raises_rather_than_guessing(self):
        cam = Camera(1, "MADE_UP_MODEL", 640, 480, np.array([1.0, 2.0, 3.0, 4.0]))
        with pytest.raises(ColmapFormatError, match="unsupported COLMAP camera model"):
            _ = cam.intrinsics

    def test_too_few_params_raises(self):
        cam = Camera(1, "PINHOLE", 640, 480, np.array([500.0, 501.0]))
        with pytest.raises(ColmapFormatError):
            _ = cam.intrinsics


class TestCameras:
    def test_width_height_are_ints(self, tmp_path):
        cams = read_cameras_text(_write(tmp_path, "cameras.txt", CAMERAS_TXT))
        cam = cams[1]
        assert isinstance(cam.width, int) and isinstance(cam.height, int)
        assert (cam.width, cam.height) == (1920, 1080)

    def test_empty_file_raises(self, tmp_path):
        with pytest.raises(ColmapFormatError):
            read_cameras_text(_write(tmp_path, "cameras.txt", "# only a comment\n"))

    def test_sole_camera_rejects_multi_camera_models(self):
        cams = {
            1: Camera(1, "PINHOLE", 10, 10, np.array([1.0, 1.0, 5.0, 5.0])),
            2: Camera(2, "PINHOLE", 10, 10, np.array([2.0, 2.0, 5.0, 5.0])),
        }
        with pytest.raises(ColmapFormatError, match="single-camera"):
            sole_camera(cams)

    def test_sole_camera_returns_the_one_camera(self):
        cam = Camera(7, "PINHOLE", 10, 10, np.array([1.0, 1.0, 5.0, 5.0]))
        assert sole_camera({7: cam}) is cam


class TestPoints3D:
    def test_parses_xyz_rgb_and_track(self, tmp_path):
        pts = read_points3D_text(_write(tmp_path, "points3D.txt", POINTS3D_TXT))
        assert sorted(pts) == [5, 7]
        np.testing.assert_allclose(pts[5].xyz, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(pts[5].rgb, [255, 128, 0])
        assert pts[5].error == 0.5
        np.testing.assert_array_equal(pts[5].image_ids, [1, 3])
        np.testing.assert_array_equal(pts[5].point2D_idxs, [0, 1])
        np.testing.assert_array_equal(pts[7].image_ids, [3])

    def test_missing_file_is_not_required(self, tmp_path):
        pts = read_points3D_text(_write(tmp_path, "points3D.txt", "# empty\n"))
        assert pts == {}


class TestGeometry:
    def test_world_to_camera_matches_camera_center(self, tmp_path):
        images = read_images_text(_write(tmp_path, "images.txt", IMAGES_TXT_WITH_EMPTY_TRACK))
        img = images[3]
        # A point at the camera center maps to the origin of the camera frame.
        cam_pt = img.world_to_camera(img.camera_center.reshape(1, 3))
        np.testing.assert_allclose(cam_pt[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_camera_to_world_inverts_world_to_camera(self, tmp_path):
        images = read_images_text(_write(tmp_path, "images.txt", IMAGES_TXT_WITH_EMPTY_TRACK))
        img = images[3]
        world = np.array([[1.0, -2.0, 7.0], [0.0, 0.0, 1.0], [-5.0, 3.0, 0.5]])
        np.testing.assert_allclose(img.camera_to_world(img.world_to_camera(world)), world, atol=1e-12)

    def test_camera_to_world_agrees_with_c2w_matrix(self, tmp_path):
        images = read_images_text(_write(tmp_path, "images.txt", IMAGES_TXT_WITH_EMPTY_TRACK))
        img = images[3]
        cam = np.array([[0.5, 1.5, 2.5], [-1.0, 0.0, 3.0]])
        via_matrix = (
            img.camera_to_world_matrix() @ np.hstack([cam, np.ones((len(cam), 1))]).T
        ).T[:, :3]
        np.testing.assert_allclose(img.camera_to_world(cam), via_matrix, atol=1e-12)

    def test_c2w_inverts_world_to_camera(self, tmp_path):
        images = read_images_text(_write(tmp_path, "images.txt", IMAGES_TXT_WITH_EMPTY_TRACK))
        img = images[3]
        world = np.array([[1.0, -2.0, 7.0], [0.0, 0.0, 1.0]])
        cam = img.world_to_camera(world)
        c2w = img.camera_to_world_matrix()
        homog = np.hstack([cam, np.ones((len(cam), 1))])
        back = (c2w @ homog.T).T[:, :3]
        np.testing.assert_allclose(back, world, atol=1e-12)


class TestFindModelDir:
    def test_prefers_the_model_with_most_points(self, tmp_path):
        small = tmp_path / "model_txt" / "0"
        big = tmp_path / "model_txt" / "1"
        for d, pts in ((small, POINTS3D_TXT), (big, POINTS3D_TXT + "9 0 0 0 1 1 1 0.1 1 0\n")):
            d.mkdir(parents=True)
            (d / "cameras.txt").write_text(CAMERAS_TXT, encoding="utf-8")
            (d / "images.txt").write_text(IMAGES_TXT_WITH_EMPTY_TRACK, encoding="utf-8")
            (d / "points3D.txt").write_text(pts, encoding="utf-8")

        assert find_model_dir(tmp_path) == big

    def test_raises_when_no_model_present(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_model_dir(tmp_path)


def test_images_by_name_indexes_basenames(tmp_path):
    images = read_images_text(_write(tmp_path, "images.txt", IMAGES_TXT_WITH_EMPTY_TRACK))
    by_name = images_by_name(images)
    assert set(by_name) == {"frame_000001.jpg", "frame_000002.jpg", "frame_000003.jpg"}
    assert by_name["frame_000003.jpg"].id == 3

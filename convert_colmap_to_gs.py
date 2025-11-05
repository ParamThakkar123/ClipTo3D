import json
import numpy as np
from pathlib import Path
import shutil


def read_cameras_txt(path):
    """Read cameras.txt from COLMAP."""
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            elems = line.strip().split()
            camera_id = int(elems[0])
            model = elems[1]
            width = float(elems[2])
            height = float(elems[3])
            params = list(map(float, elems[4:]))
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_images_txt(path):
    """Read images.txt from COLMAP."""
    images = {}
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    for i in range(0, len(lines), 2):
        elems = lines[i].split()
        image_id = int(elems[0])
        qvec = np.array(list(map(float, elems[1:5])))
        tvec = np.array(list(map(float, elems[5:8])))
        camera_id = int(elems[8])
        name = elems[9]
        images[name] = {
            "id": image_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
        }
    return images


def qvec_to_rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    q0, q1, q2, q3 = qvec
    return np.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1*q2 - q0*q3),     2 * (q1*q3 + q0*q2)],
        [2 * (q1*q2 + q0*q3),     1 - 2 * (q1**2 + q3**2), 2 * (q2*q3 - q0*q1)],
        [2 * (q1*q3 - q0*q2),     2 * (q2*q3 + q0*q1),     1 - 2 * (q1**2 + q2**2)],
    ])


def colmap_to_transforms(colmap_dir, image_dir, output_dir):
    """Convert COLMAP model to Gaussian Splatting format."""
    colmap_dir = Path(colmap_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    # Create directories
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    # Load COLMAP data
    cameras = read_cameras_txt(colmap_dir / "cameras.txt")
    images = read_images_txt(colmap_dir / "images.txt")

    frames = []

    for name, data in images.items():
        cam = cameras[data["camera_id"]]

        # Handle intrinsics depending on camera model
        if cam["model"] in ["PINHOLE", "SIMPLE_PINHOLE"]:
            fx = fy = cam["params"][0]
            cx = cam["params"][1] if len(cam["params"]) > 1 else cam["width"] / 2
            cy = cam["params"][2] if len(cam["params"]) > 2 else cam["height"] / 2
        elif cam["model"] in ["OPENCV", "OPENCV_FISHEYE"]:
            fx, fy, cx, cy = cam["params"][:4]
        else:
            fx = cam["params"][0]
            fy = cam["params"][1] if len(cam["params"]) > 1 else fx
            cx = cam["params"][2] if len(cam["params"]) > 2 else cam["width"] / 2
            cy = cam["params"][3] if len(cam["params"]) > 3 else cam["height"] / 2

        # Build extrinsics
        R = qvec_to_rotmat(data["qvec"])
        t = data["tvec"]
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t

        frame = {
            "image_id": data["id"],
            "file_path": f"images/{name}",
            "transform_matrix": c2w.tolist()
        }
        frames.append(frame)

        # Copy image
        src = image_dir / name
        dst = output_dir / "images" / name
        if src.exists():
            shutil.copy(src, dst)
        else:
            print(f"⚠️ Missing image: {src}")

    # Compute camera angle in radians
    camera_angle_x = 2 * np.arctan(cam["width"] / (2 * fx))

    # Write JSON
    out_data = {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": cam["width"],
        "h": cam["height"],
        "camera_angle_x": camera_angle_x,
        "frames": frames,
    }

    with open(output_dir / "transforms.json", "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"✅ Wrote {len(frames)} frames to {output_dir / 'transforms.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert COLMAP outputs to Gaussian Splatting dataset format."
    )
    parser.add_argument("--colmap_dir", type=str, required=True, help="Path to COLMAP sparse model directory.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to original images.")
    parser.add_argument("--output_dir", type=str, default="gs_dataset", help="Output dataset directory.")
    args = parser.parse_args()

    colmap_to_transforms(args.colmap_dir, args.image_dir, args.output_dir)
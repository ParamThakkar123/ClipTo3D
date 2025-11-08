import os
import json
import numpy as np 
import imageio
from tqdm import tqdm

TRANS = "gsplat_output/transforms.json"
OUT_PLY = "fused_cloud.ply"
IMG_ROOT = "gsplat_output"   
DEPTH_DIR = "depth_maps"   
MAX_PER_FRAME = 50000        
GLOBAL_MAX_DEPTH = 5.0      

def load_transforms(path):
    with open(path, 'r') as f:
        return json.load(f)

def matrix_ok(m):
    return isinstance(m, list) and len(m) == 4 and all(isinstance(r, list) and len(r) == 4 for r in m)

def write_ply(filename, pts, cols):
    """Write colored point cloud to PLY (binary little endian)."""
    pts = np.asarray(pts, dtype=np.float32)
    cols = np.clip(np.asarray(cols) * 255.0, 0, 255).astype(np.uint8)
    n = len(pts)
    with open(filename, 'wb') as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, cols):
            f.write(p.tobytes())
            f.write(c.tobytes())

def main():
    j = load_transforms(TRANS)
    fx, fy, cx, cy = j.get("fl_x"), j.get("fl_y"), j.get("cx"), j.get("cy")
    w, h = int(j.get("w")), int(j.get("h"))

    pts_all, cols_all = [], []

    for f in tqdm(j["frames"], desc="Processing frames"):
        fp = os.path.join(IMG_ROOT, f["file_path"])
        if not matrix_ok(f.get("transform_matrix", None)):
            continue

        T = np.array(f["transform_matrix"], dtype=np.float32)

        base = os.path.splitext(os.path.basename(f["file_path"]))[0]
        depth_path = os.path.join(DEPTH_DIR, base + "_depth.npy")
        if not os.path.exists(depth_path):
            print(f"⚠️ Missing depth map for {base}, skipping.")
            continue

        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != (h, w):
            print(f"⚠️ Skipping {base}: depth size mismatch ({depth.shape})")
            continue

        img = imageio.imread(fp)[..., :3].astype(np.float32) / 255.0

        us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        Z = depth
        valid = (Z > 0) & np.isfinite(Z)
        valid_idx = np.flatnonzero(valid.ravel())
        if valid_idx.size == 0:
            continue

        if valid_idx.size > MAX_PER_FRAME:
            sel = np.random.choice(valid_idx, MAX_PER_FRAME, replace=False)
        else:
            sel = valid_idx

        us_r = us.ravel()[sel]
        vs_r = vs.ravel()[sel]
        z_r = Z.ravel()[sel].astype(np.float32)

        if np.max(z_r) > 0:
            z_r = (z_r / np.max(z_r)) * GLOBAL_MAX_DEPTH
        else:
            continue

        x_r = (us_r - cx) * z_r / fx
        y_r = (vs_r - cy) * z_r / fy
        pts_cam = np.stack([x_r, y_r, z_r, np.ones(len(sel), dtype=np.float32)], axis=0)

        pts_world = (T @ pts_cam).T[:, :3]

        cols = img.reshape(-1, 3)[sel]

        mask = np.isfinite(pts_world).all(axis=1) & (np.abs(pts_world) < 100).all(axis=1)
        pts_world = pts_world[mask]
        cols = cols[mask]

        if len(pts_world) == 0:
            continue

        pts_all.append(pts_world)
        cols_all.append(cols)

    if not pts_all:
        print("❌ No valid points reconstructed.")
        return

    P = np.vstack(pts_all).astype(np.float32)
    C = np.vstack(cols_all).astype(np.float32)

    print(f"✅ Reconstructed {len(P):,} points. Writing to {OUT_PLY}...")
    write_ply(OUT_PLY, P, C)
    print("🎉 Done! Output saved:", OUT_PLY)


if __name__ == "__main__":
    main()
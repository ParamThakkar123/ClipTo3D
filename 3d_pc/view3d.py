import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
import trimesh

PLY = "fused_cloud.ply"
MAX_POINTS = 50000  # downsample for plotting

def load_ply(path):
    ply = PlyData.read(path)
    v = ply['vertex'].data
    pts = np.vstack([v['x'], v['y'], v['z']]).T
    try:
        cols = np.vstack([v['red'], v['green'], v['blue']]).T / 255.0
    except Exception:
        cols = None
    return pts, cols

def main():
    pts, cols = load_ply(PLY)
    n = len(pts)
    if n == 0:
        print("no points in", PLY)
        return
    if n > MAX_POINTS:
        idx = np.random.choice(n, MAX_POINTS, replace=False)
        pts = pts[idx]
        cols = cols[idx] if cols is not None else None
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection='3d')
    c = cols if cols is not None else 'k'
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=c, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Fused cloud preview: " + PLY)
    plt.show()
    mesh = trimesh.load("fused_cloud.ply", process=False)
    print("vertices:", len(mesh.vertices))
    # trimesh shows pointcloud in an interactive window (requires pyglet)
    mesh.show()

if __name__ == "__main__":
    main()
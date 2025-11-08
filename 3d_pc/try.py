import trimesh
mesh = trimesh.load("E:\\ClipToWorld\\fused_cloud.ply", process=False)
print("vertices:", len(mesh.vertices))
try:
    mesh.show()
except Exception as e:
    print("interactive viewer failed:", e)
    print("Install pyglet and PyOpenGL to enable mesh.show():")
    print("    pip install pyglet PyOpenGL PyOpenGL_accelerate")
    mesh.export("fused_cloud_export.ply")
    print("Exported fused_cloud_export.ply for external viewing.")
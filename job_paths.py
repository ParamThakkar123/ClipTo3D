"""One artifact layout for a reconstruction job (MPO-224).

Previously every stage invented its own default directories, and they did not
agree: `convert_colmap_to_gs.py` wrote `gs_dataset/` while the fusion stage
read a hardcoded `gsplat_output/transforms.json` that nothing ever produced.
Running the pipeline meant reconciling paths by hand between eight commands.

A `JobPaths` is just a root plus named subdirectories. Stages still accept
explicit paths — this is what the CLI and orchestrator use to fill them in
consistently, and it is the seam a storage backend would slot into later
(MPO-236).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobPaths:
    """Filesystem layout for a single reconstruction job.

        <root>/
          input/     source video
          frames/    extracted frames
          keyframes/ the subset worth reconstructing (+ keyframes.json)
          depth/     depth maps (+ depth_meta.json)
          colmap/    database.db, sparse/, sparse/model_txt/
          dataset/   transforms.json (+ images/)
          splat/     gaussian splat training output
          cloud/     fused point cloud + extracted mesh
          export/    web/mobile formats
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def keyframes(self) -> Path:
        return self.root / "keyframes"

    @property
    def keyframes_manifest(self) -> Path:
        return self.keyframes / "keyframes.json"

    @property
    def depth(self) -> Path:
        return self.root / "depth"

    @property
    def colmap(self) -> Path:
        return self.root / "colmap"

    @property
    def colmap_sparse(self) -> Path:
        return self.colmap / "sparse"

    @property
    def colmap_model_txt(self) -> Path:
        return self.colmap_sparse / "model_txt"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def transforms_json(self) -> Path:
        return self.dataset / "transforms.json"

    @property
    def splat(self) -> Path:
        return self.root / "splat"

    @property
    def cloud(self) -> Path:
        return self.root / "cloud"

    @property
    def fused_ply(self) -> Path:
        return self.cloud / "fused_cloud.ply"

    @property
    def mesh_npz(self) -> Path:
        """Extracted surface. A mesh, not a cloud, is what AR needs (MPO-248)."""
        return self.cloud / "mesh.npz"

    @property
    def export(self) -> Path:
        return self.root / "export"

    def all_dirs(self) -> list[Path]:
        return [
            self.input_dir, self.frames, self.keyframes, self.depth, self.colmap,
            self.dataset, self.splat, self.cloud, self.export,
        ]

    def ensure(self) -> "JobPaths":
        """Create every directory in the layout. Returns self for chaining."""
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)
        return self

    def describe(self) -> str:
        lines = [f"job root: {self.root}"]
        for d in self.all_dirs():
            marker = "" if d.exists() else "  (missing)"
            lines.append(f"  {d.name}/{marker}")
        return "\n".join(lines)

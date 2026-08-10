# syntax=docker/dockerfile:1.7
#
# Reconstruction worker image (MPO-232).
#
# The pipeline assumes two native binaries on PATH that are not pip-installable
# and do not exist on a bare host: ffmpeg (frames.py, audio.py) and COLMAP
# (sfm.py). COLMAP's GPU SIFT additionally needs either a CUDA-enabled build or
# an OpenGL context — and a headless server has no GL context, so this image
# builds COLMAP with CUDA and without GUI/OpenGL entirely.
#
#   docker build -f docker/Dockerfile -t clipto3d-worker .
#   docker run --gpus all -v "$PWD/runs:/runs" clipto3d-worker /runs/clip.mp4 --job /runs/job1
#
# CUDA 13 matches the +cu130 torch wheels pinned by the `cuda` extra (MPO-231).

ARG CUDA_VERSION=13.0.1
ARG UBUNTU_VERSION=24.04

# CUDA 13 dropped Maxwell/Pascal/Volta, so sm_75 (Turing) is the floor.
# Narrow this to your fleet's actual SM to cut build time substantially.
ARG CUDA_ARCHS="75;80;86;89;90"
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# Declared here, before the first FROM, because an ARG used in a FROM line must
# be in the global scope — one declared after a FROM belongs to that stage and
# expands to empty in the next one.
ARG RUNTIME_FLAVOR=runtime


###############################################################################
# Stage 1 — build COLMAP with CUDA SIFT, no GUI, no OpenGL.
###############################################################################
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS colmap-builder

ARG CUDA_ARCHS
# Last 3.x. COLMAP 4.x additionally FetchContents ONNX, which this pipeline has
# no use for. sfm.py only drives feature_extractor / sequential_matcher /
# mapper / model_converter, all long-stable.
ARG COLMAP_REF=3.13.0
# Empty = use every core. Set it (e.g. --build-arg BUILD_JOBS=4) on a machine
# with limited RAM: ninja at full width on 349 C++/CUDA targets can push the
# builder VM into OOM, which surfaces as buildkit dying with `rpc error: EOF`.
ARG BUILD_JOBS=""

ENV DEBIAN_FRONTEND=noninteractive

# No qt5: GUI_ENABLED=OFF means the Qt UI is never built.
#
# The GL dev packages are required even though OPENGL_ENABLED=OFF, because
# COLMAP's cmake/FindDependencies.cmake calls find_package(OpenGL REQUIRED) and
# find_package(Glew REQUIRED) unconditionally — the option gates usage, not
# discovery, and configure fails outright without them.
#
# Verified on the built binary: it still links libGL, libGLX, libGLEW,
# libGLdispatch and libX11. That is fine and is why the runtime stage installs
# the matching sonames — linking a library is not the same as opening a
# display. What makes this headless is that COLMAP is built *with CUDA*
# (`colmap -h` reports "with CUDA"), so SIFT uses CUDA kernels and never asks
# for a GL context. Without a CUDA build, SiftGPU would fall back to OpenGL and
# would genuinely need a display.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libsqlite3-dev \
        libceres-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libgl1-mesa-dev \
        libglew-dev \
        # BLAS/LAPACK for the faiss that COLMAP FetchContents. OpenBLAS rather
        # than libmkl-full-dev, which is several GB for no benefit here.
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${COLMAP_REF}" https://github.com/colmap/colmap.git /src/colmap

RUN cmake -S /src/colmap -B /src/colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/colmap \
        -DCUDA_ENABLED=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DGUI_ENABLED=OFF \
        -DOPENGL_ENABLED=OFF \
        -DTESTS_ENABLED=OFF \
        # Only enables the Delaunay mesher, which nothing here drives.
        -DCGAL_ENABLED=OFF \
    && cmake --build /src/colmap/build --target install ${BUILD_JOBS:+--parallel ${BUILD_JOBS}} \
    && rm -rf /src/colmap


###############################################################################
# Stage 2 — Python environment, with gsplat's CUDA kernels compiled ahead of
# time so the runtime host needs no toolchain for the common path.
###############################################################################
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS python-builder

ARG TORCH_CUDA_ARCH_LIST
# Pinned, not left to uv's default. PyTorch's CUDA index publishes sha256
# fragments for its cp311/cp312 wheels but not for cp313+, so `uv sync
# --frozen --extra cuda` fails on a newer interpreter with a hash mismatch
# against wheels the lock could not record hashes for. Pinning here also stops
# the image's Python drifting whenever uv's default moves.
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    UV_LINK_MODE=copy \
    UV_PYTHON=${PYTHON_VERSION} \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    TORCH_EXTENSIONS_DIR=/opt/torch_extensions

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependency layer, cached independently of the source tree. --frozen means the
# committed uv.lock is authoritative: the image cannot silently drift from it.
#
# README.md comes along because pyproject declares `readme = "README.md"`, and
# the project is now an installable package (MPO-245) — hatchling fails the
# build without it. It changes rarely, so it costs almost no cache churn.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra cuda --extra splat --extra mesh --extra ar

# Warm gsplat's JIT cache. This needs nvcc and TORCH_CUDA_ARCH_LIST so it
# targets the fleet's SMs rather than the build machine's — `docker build` has
# no GPU, so device probing is not available.
#
# BUDGET WARNING: this is the slowest step in the image by a wide margin.
# gsplat's kernels are heavily templated and are compiled once per architecture
# in TORCH_CUDA_ARCH_LIST; the five-arch default was still running after 45
# minutes on a 4-core laptop, with no progress output. Narrow the list to the
# SMs you actually deploy on. It also ignores BUILD_JOBS, which only reaches
# COLMAP's cmake.
#
# REQUIRE_PRECOMPILE=1 (the default) makes a miss here a hard build failure,
# because the runtime stage now ships *without* nvcc — silently continuing
# would produce an image that cannot train splats at all. Pair
# REQUIRE_PRECOMPILE=0 with RUNTIME_FLAVOR=devel for the old JIT fallback.
#
# The directory is created unconditionally so the runtime stage always has
# something to COPY.
ARG REQUIRE_PRECOMPILE=1
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /opt/torch_extensions \
    && if uv run --no-sync python -c "from gsplat.cuda._backend import _C; print('gsplat kernels precompiled')"; then \
           touch /opt/torch_extensions/.precompiled; \
       elif [ "${REQUIRE_PRECOMPILE}" = "1" ]; then \
           echo "FATAL: gsplat kernels did not precompile, and the runtime stage has no nvcc."; \
           echo "Narrow TORCH_CUDA_ARCH_LIST, or build with REQUIRE_PRECOMPILE=0 RUNTIME_FLAVOR=devel."; \
           exit 1; \
       else \
           echo "WARNING: gsplat kernels not precompiled; first job will JIT them."; \
       fi


###############################################################################
# Stage 3 — runtime.
###############################################################################
# `runtime`, not `devel`: with the kernels precompiled in stage 2 there is
# nothing left that needs a CUDA toolchain, and the runtime base is several GB
# smaller. Override with RUNTIME_FLAVOR=devel (plus REQUIRE_PRECOMPILE=0) if
# you need nvcc available for a cold JIT.
FROM nvidia/cuda:${CUDA_VERSION}-${RUNTIME_FLAVOR}-ubuntu${UBUNTU_VERSION} AS runtime

ARG TORCH_CUDA_ARCH_LIST
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    TORCH_EXTENSIONS_DIR=/opt/torch_extensions \
    CLIPTO3D_CHECKPOINT_DIR=/opt/checkpoints \
    PATH="/opt/colmap/bin:/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    # No X server here; make sure nothing tries to open a display.
    QT_QPA_PLATFORM=offscreen

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ninja-build \
        ca-certificates \
        curl \
        libboost-program-options1.83.0 \
        libboost-graph1.83.0 \
        libfreeimage3 \
        libflann1.9 \
        libgoogle-glog0v6t64 \
        libceres4t64 \
        libmetis5 \
        libsqlite3-0 \
        libcurl4t64 \
        libgomp1 \
        # BLAS/LAPACK, pulled in through the faiss that COLMAP FetchContents.
        libopenblas0 \
        # COLMAP is configured against GL even though it does not use it; keep
        # the runtime sonames present so the binary loads. Loading a library is
        # not opening a display — no X server is needed. libopengl0 provides
        # libOpenGL.so.0, which libgl1 alone does not.
        libgl1 \
        libopengl0 \
        libglew2.2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=colmap-builder /opt/colmap /opt/colmap
COPY --from=python-builder /opt/python /opt/python
COPY --from=python-builder /opt/torch_extensions /opt/torch_extensions
COPY --from=python-builder /app/.venv /app/.venv

WORKDIR /app
COPY . /app

# Fail the build rather than a job if the native prerequisites are not actually
# reachable. Neither of these needs a GPU, so both run during `docker build`.
# `set -eux` so a failure names the offending command instead of surfacing as a
# bare exit 127 from whichever link in a && chain broke.
RUN set -eux; \
    echo "PATH=$PATH"; \
    command -v colmap; \
    command -v ffmpeg; \
    command -v python; \
    # Assert the dynamic linker is satisfied BEFORE running colmap. Piping
    # `colmap -h` into head would take head's exit status, which silently
    # passed a build whose colmap could not load libopenblas at all.
    if ldd /opt/colmap/bin/colmap | grep 'not found'; then \
        echo "FATAL: colmap has unresolved shared libraries"; exit 1; \
    fi; \
    colmap -h > /dev/null; \
    ffmpeg -version > /dev/null; \
    python -c "import pipeline, colmap_io, fusion.fuse, keyframes, checkpoints; print('pipeline imports OK')"

# Depth weights, pinned by URL + SHA256 and verified at build time (MPO-233).
# vitl is ~1.3GB; set to depth-anything-v2-vits for a much smaller image, or to
# an empty string to skip and mount a cache at /opt/checkpoints instead.
#
# Only the depthanythingv2 backend is offline-capable — torch.hub supplies the
# MiDaS *architecture*, not just its weights, so that backend still needs
# network on first use. It is not the default any more.
# Vocabulary tree for COLMAP loop closure (MPO-242). Without it, sequential
# matching never links revisited viewpoints and an orbiting clip reconstructs
# as two disconnected models. COLMAP would fetch this at match time; baking it
# in keeps the worker offline-capable and keeps the download off the job's
# critical path.
ARG VOCAB_TREE_URL="https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin"
ARG VOCAB_TREE_SHA256="96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc"
ENV COLMAP_VOCAB_TREE=/opt/colmap/share/vocab_tree.bin
RUN mkdir -p /opt/colmap/share \
    && curl -fsSL -o "$COLMAP_VOCAB_TREE" "${VOCAB_TREE_URL}" \
    && echo "${VOCAB_TREE_SHA256}  $COLMAP_VOCAB_TREE" | sha256sum -c - \
    && ls -l "$COLMAP_VOCAB_TREE"

ARG BAKE_CHECKPOINTS="depth-anything-v2-vitl"
RUN if [ -n "${BAKE_CHECKPOINTS}" ]; then \
        python checkpoints.py --fetch ${BAKE_CHECKPOINTS} && \
        python checkpoints.py --verify ; \
    else \
        echo "No checkpoints baked in; mount a cache at \$CLIPTO3D_CHECKPOINT_DIR." ; \
    fi

ENTRYPOINT ["python", "pipeline.py"]
CMD ["--help"]

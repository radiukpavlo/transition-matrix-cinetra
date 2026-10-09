# syntax=docker/dockerfile:1
# CMR-TRANS-AUDIT execution image (docs/EXECUTION_ENVIRONMENT_SPEC.md 1.0.0; Decisions D3, D22, D23; ADR-0003).
# Both base images are pinned by digest (the pins of the CXR-TRANS-AUDIT image, confirmed on 2026-09-25).
# Re-confirm a pin with: docker buildx imagetools inspect ubuntu:24.04 (or ghcr.io/astral-sh/uv:0.12.18)
ARG UBUNTU_IMAGE=ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.18@sha256:3adc3706091ce7c2fe595e669628caedd6d951551b92b258b7e7dbe06d9440bc

FROM ${UV_IMAGE} AS uv

# Stage base: Ubuntu, tools, uv and the uv-managed CPython (also runs `uv lock`, Compose service `lock`).
FROM ${UBUNTU_IMAGE} AS base
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    TZ=Etc/UTC \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/cache/uv \
    PYTHONPATH=/workspace/src \
    MPLBACKEND=Agg
ENV PATH=/opt/venv/bin:$PATH
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs unzip pigz pv jq rsync time \
 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
# Git reads the Windows checkout through the bind mount (env spec section 7.1, measured 2026-09-28):
# - core.autocrlf=true mirrors Git for Windows. 1,042 tracked text files, including the 532 hash-locked
#   historical files, hold CRLF bytes while the index holds LF; with false, every one of them reads as modified.
# - core.checkStat=minimal and core.trustctime=false: the Windows index stores no inode, uid or Linux ctime, so
#   the default comparison re-hashed all 1.4 GB of tracked files (49 s per status); this compares mtime and size (3.6 s).
RUN uv python install 3.12 \
 && git config --system --add safe.directory /workspace \
 && git config --system core.autocrlf true \
 && git config --system core.checkStat minimal \
 && git config --system core.trustctime false
# From here on the interpreter is fixed: a .python-version the image lacks fails the build instead of downloading.
ENV UV_PYTHON_DOWNLOADS=never
WORKDIR /workspace

# Stage env (default, CPU): locked /opt/venv with the runtime, dev and clinical dependencies; immutable at run time.
FROM base AS env
COPY pyproject.toml uv.lock .python-version /opt/project/
RUN --mount=type=cache,id=tmmri-uv,target=/cache/uv \
    cd /opt/project && uv sync --locked --no-install-project --group clinical

# Stage env-gpu (optional, Decision D20): env plus torch 2.14.0+cu126 and torchvision 0.29.0+cu126.
FROM env AS env-gpu
ENV TORCH_HOME=/cache/torch
RUN --mount=type=cache,id=tmmri-uv,target=/cache/uv \
    cd /opt/project && uv sync --locked --no-install-project --group clinical --group gpu

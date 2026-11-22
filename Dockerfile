# syntax=docker/dockerfile:1
#
# 图纸智析 (北京市大学生"人工智能+"创新大赛 · 赛题 4) — offline inference image.
# One 12-field JSON per input PNG, written next to the input's own name.
#
#   docker build -t cadvec-infer .
#   docker run --rm -v "$PWD/in:/in:ro" -v "$PWD/out:/out" cadvec-infer \
#       --inputs /in --out /out --device cpu --verify
#
# No --mode: the default is `blend` (documented in infer.py), the submitted readout.
# `--mode simple|wide` work too; `--mode v3` needs selector artefacts this repo does
# not ship and falls back to `simple` with a note in <out>/infer_report.json.
#
# `--out` must be empty (the delivery CLI writes into a fresh directory), and the
# container needs write access to it.
#
# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
# This image is CPU-only on purpose: python:3.12-slim has no CUDA userland.  To run on a
# GPU, either
#   (a) swap the base image and keep the same instructions:
#         FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime     # torch + CUDA preinstalled
#       and build with
#         --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 \
#         --build-arg TORCH_VERSION=2.14.1+cu130 \
#         --build-arg TV_VERSION=0.29.1+cu130
#       (drop the python3 install below; that image already has python + torch), or
#   (b) start from nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04 and apt-get install
#       python3.12 python3.12-venv before the pip steps.
# Then run with `docker run --gpus all ... --device cuda:0`.
#
# A GPU run also needs a CUDA-enabled *torchvision*: with the +cpu wheel every cuda
# forward dies inside the detector's post-processing ("torchvision::nms has no CUDA
# backend").  infer.py probes torchvision::nms / roi_align on CUDA before running and
# refuses with a --tv-overlay hint instead of a traceback, so a mis-built image fails
# fast and legibly.
#
# ---------------------------------------------------------------------------
# OFFLINE BUILD
# ---------------------------------------------------------------------------
# Nothing here reaches the network at run time, but pip does at build time.  On an
# air-gapped builder, download the wheels on a connected machine:
#
#   pip download -d wheels --index-url https://download.pytorch.org/whl/cpu \
#       torch==2.14.1 torchvision==0.29.2
#   pip download -d wheels numpy==2.5.4 pillow==13.0.0 scikit-learn==1.9.1 \
#       scipy==1.18.1 joblib==1.6.0 threadpoolctl==3.8.0 opencv-python-headless==4.14.0.94 \
#       tqdm==4.70.1 psutil==7.2.2
#
# copy that directory to ./wheels next to this Dockerfile, add the
# `COPY ["./wheels/", "/wheels/"]` line that the pip step below mentions, and build with
#
#   docker build --build-arg PIP_EXTRA="--no-index --find-links=/wheels" -t cadvec-infer .
#
# (`--network none` is then safe).  The checkpoint is a separate 193 MB blob: bake it in
# with the default WEIGHTS arg, or leave it on the host and mount it
# (`-v $PWD/weights:/app/weights:ro`).
# ---------------------------------------------------------------------------

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=4

# opencv-python-headless needs libglib/libgomp only; no GUI libs, no apt network needed
# if the base image already carries them (it does not, hence these two).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG PYPI_INDEX=https://pypi.org/simple
ARG PIP_EXTRA=""
# Version tags follow the index: CPU wheels are 2.14.1 / 0.29.2, the cu130 pair measured
# here is 2.14.1+cu130 / 0.29.1+cu130 (see requirements.txt — the CUDA torchvision is a
# different artefact, not just a different build of the same version).
ARG TORCH_VERSION=2.14.1
ARG TV_VERSION=0.29.2

# torch/torchvision first: same versions as the verified venv's pip freeze, but from the
# index so the CUDA vs CPU flavour stays a build choice.  See requirements.txt for the
# full measured freeze.
RUN pip install --no-cache-dir ${PIP_EXTRA} --index-url "${TORCH_INDEX}" \
        torch==${TORCH_VERSION} \
        torchvision==${TV_VERSION} \
 && pip install --no-cache-dir ${PIP_EXTRA} --index-url "${PYPI_INDEX}" \
        numpy==2.5.4 \
        pillow==13.0.0 \
        scikit-learn==1.9.1 \
        scipy==1.18.1 \
        joblib==1.6.0 \
        threadpoolctl==3.8.0 \
        opencv-python-headless==4.14.0.94 \
        tqdm==4.70.1 \
        psutil==7.2.2

# Optional wheelhouse for the offline build.  It is *not* copied here: this repository
# ships no wheels/ directory, and COPY of a path that is absent from the build context
# fails the build.  For an air-gapped builder, put the downloaded wheels in ./wheels,
# add `COPY ["./wheels/", "/wheels/"]` right after this comment, and build with
# `--build-arg PIP_EXTRA="--no-index --find-links=/wheels"`.  Online builds need neither.
RUN mkdir -p /wheels

# The code: the stdlib-only cadvec package, the delivery CLI and its helpers, and the
# entry point.  results/, evidence/ and data/ are deliberately not copied (see .dockerignore).
#
# targets/ is NOT optional: the default run is
#   delivery/cad_mrcnn_delivery.py -> evaluation/instance_eval/batch.py
#   -> model/emit.py -> model/contract.py, which loads <repo>/targets/targets.py by
#   explicit path and then imports src/cadvec/recognition_raster_data.py through it.
# Without targets/ the container dies with TargetError("targets contract not found").
# pyproject.toml is not needed: nothing is pip-installed from the source tree and no
# module reads package metadata at inference time.
COPY infer.py /app/infer.py
COPY delivery/ /app/delivery/
COPY model/ /app/model/
COPY src/ /app/src/
COPY evaluation/ /app/evaluation/
COPY targets/ /app/targets/
COPY requirements.txt /app/requirements.txt

# The weights directory, whole: the detector + 112^2 fine-mask checkpoint
# (weights/fine_mask112_u550.pt, 193 MB) *and* the stage-2 blend selector
# (weights/blend_sameclass_20261119.pkl, ~306 MB).  Both are needed: infer.py's default
# --mode is blend, which refuses to start without the .pkl.  Keep the directory copy —
# copying a single named .pt would leave the default run unable to start.  To keep the
# image small, mount instead: `-v $PWD/weights:/app/weights:ro`.
COPY ["weights/", "/app/weights/"]

# Run it.  WORKDIR is /app and infer.py resolves --checkpoint / --blend-model relative
# to its own file, so the defaults hold for any caller working directory:
#
#   docker run --rm -v <png-dir>:/in:ro -v <out-dir>:/out cadvec-infer \
#       --inputs /in --out /out --device cpu --verify
#
# <out-dir> must be empty or absent; the CLI writes into a fresh directory.  --verify
# runs delivery/verify_delivery.py over the output: /out/infer_report.json gets
# `verifier: {exit, ok, errors}` and the stdout summary line gets `verifier_ok`.

ENTRYPOINT ["python", "infer.py"]
CMD ["--help"]

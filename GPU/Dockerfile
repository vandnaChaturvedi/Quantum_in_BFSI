# ============================================================
# Dockerfile — Quantum Portfolio Optimization (7x H100 80 GB)
# Base  : CUDA 12.1 + cuDNN 8 on Ubuntu 22.04
# Python: 3.10 (conda)
# GPU   : cuStateVec GPU via qiskit-aer-gpu  (CUDA 12 build)
# ============================================================

# ── Stage 1: base CUDA image ─────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 ca-certificates curl git \
        libglib2.0-0 libxext6 libsm6 libxrender1 \
        build-essential libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Miniconda + conda environment ───────────────────
FROM base AS conda-env

RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh \
    -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda update -n base -c defaults conda -y \
    && conda clean -afy

# Copy environment spec and create environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy \
    && find ${CONDA_DIR} -name "*.pyc" -delete

# ── Stage 3: runtime image ───────────────────────────────────
FROM conda-env AS runtime

# NVIDIA runtime flags — expose all 7 H100s
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# Activate qpo_gpu conda env for every RUN / CMD
SHELL ["conda", "run", "-n", "qpo_gpu", "/bin/bash", "-c"]

# Verify GPU stack is importable
RUN python -c "
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler
sim = AerSimulator()
print('Aer methods:', sim.available_methods())
print('GPU stack import OK')
"

# ── Workspace ────────────────────────────────────────────────
WORKDIR /workspace

# Copy source files (code + data)
COPY . /workspace/

# Output directory for plots / results
RUN mkdir -p /workspace/outputs

# ── Entrypoint ───────────────────────────────────────────────
# Default: verify GPU and print device list
# Override CMD when running a specific script:
#   docker run ... qpo-gpu 1_qiskit-gpu.py
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "qpo_gpu", "python"]
CMD ["--version"]

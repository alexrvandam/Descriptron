# syntax=docker/dockerfile:1.7
# =============================================================================
# Descriptron — every environment in one image
# =============================================================================
# Four conda environments that cannot co-exist in one Python installation live
# side by side here, because the conflict is about co-installation, not about
# sharing a disk. In particular metric3d needs torch 2.0.1+cu117, which cannot
# be installed alongside the 2.4/2.5 the rest of the system uses — a container
# is the one place it can be offered at all.
#
#   measure_env     py3.9.20   torch 2.5.0            analysis, BioRAG, Florence-2
#   samm            py3.12.4   torch 2.4.0+cu121      SAM2, SAM2-PAL, DINOLand, torchvision detectors
#   detectron2_env  py3.9.20   torch 2.4.1+cu121      Mask R-CNN (the accurate backend)
#   metric3d        py3.9.2    torch 2.0.1+cu117      depth from defocus (opt-in, see ARG below)
#
# The environments keep the names they have on a native install, so every
# `conda run -n <env> …` call in the GUI works unchanged inside the container.
#
# CUDA: the base supplies the toolchain needed to COMPILE detectron2. Each torch
# wheel ships its own CUDA runtime, which is why one cu121 base can host cu117
# and cu124 environments at the same time. Only the host driver has to be new
# enough (>=525 for cu121).
#
# SIZE: roughly 25 GB with three environments, 31 GB with metric3d. That is the
# honest cost of shipping four torch stacks; see docker/README.md for how to
# build a smaller single-purpose image instead.
#
#   docker build -f docker/Dockerfile -t descriptron:latest .
#   docker build -f docker/Dockerfile --build-arg INCLUDE_METRIC3D=true -t descriptron:depth .
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# opt-in extras. metric3d is off by default: it is not validated for release and
# adds ~6 GB. gpt4 (the v1 OpenAI description scripts) is superseded by BioRAG
# and is deliberately not offered at all.
ARG INCLUDE_METRIC3D=false
ARG INCLUDE_DETECTRON2=true
ARG DETECTRON2_COMMIT=c69939a
# SAM2 pinned to the commit this system was developed against. Upstream main now
# requires torch>=2.5.1, so cloning it silently dragged samm from the validated
# torch 2.4.0+cu121 up to 2.14.0+cu130 — a different environment wearing the same
# name. 7e1596c (2024-08-14) requires torch>=2.3.1 and is what the GUI, SAM2-PAL
# and every published result were built on.
ARG SAM2_COMMIT=7e1596c
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DESCRIPTRON_HOME=/opt/descriptron \
    MPLBACKEND=Agg

# --- system packages -------------------------------------------------------
# libgl1/libglib2.0 are needed by opencv; tk by the GUI; git for the two
# dependencies that are not on PyPI (SAM2 and detectron2)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        tk python3-tk libgtk-3-0 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# --- conda (Miniforge, not Miniconda) --------------------------------------
# Miniforge defaults to conda-forge and never touches repo.anaconda.com. Two
# reasons, and the second is the one that matters for a published image:
#   1. Miniconda now refuses non-interactive installs until Anaconda's Terms of
#      Service are accepted for pkgs/main and pkgs/r — the build fails with
#      CondaToSNonInteractiveError before installing anything.
#   2. Anaconda's default channels carry licensing conditions for larger
#      organisations. An image we distribute should not quietly put that on a
#      museum or university. conda-forge is community-maintained with no such
#      condition.
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p "$CONDA_DIR" \
    && rm /tmp/miniforge.sh \
    && conda config --system --remove-key channels 2>/dev/null || true \
    && conda config --system --add channels conda-forge \
    && conda config --system --set channel_priority strict \
    && conda config --set always_yes true --set changeps1 false \
    && conda clean -afy

# Requirements exported from the working installation. The `@ file:///home/conda/...`
# entries a conda export leaves behind refer to the build machine and must go, and
# torch/torchvision are installed separately so each environment gets the right
# CUDA build rather than whatever PyPI serves.
# Two requirement sources, and the order matters:
#   docker/requirements-*.txt  — DERIVED from what the programs actually import.
#     Authoritative. The exported environments/*_requirements.txt are pip-freeze
#     files, so every conda-installed package (numpy included) is missing from
#     them, and samm's export dates from 2024 and lacks cv2, scipy, rembg and
#     transformers — an image built from it cannot run SAM2-PAL v21 or DINOLand.
#   environments/*_requirements.txt — the historical pins, applied afterwards as
#     a best-effort supplement so version-sensitive extras are not lost.
COPY docker/requirements-measure_env.txt docker/requirements-samm.txt \
     docker/requirements-detectron2_env.txt \
     docker/constraints-measure_env.txt docker/constraints-samm.txt \
     docker/constraints-detectron2_env.txt /tmp/req/
COPY environments/ /tmp/environments/
RUN set -eux; \
    for f in /tmp/environments/*_requirements.txt; do \
        grep -vE '@ file://|^torch(vision|audio)?[=<>]|^torch$|^-e |^#' "$f" \
            > "${f}.clean" || true; \
    done

# --- measure_env : analysis, BioRAG, Florence-2 ----------------------------
RUN conda create -n measure_env python=3.9.20 -y \
    && conda run -n measure_env pip install --no-cache-dir \
         torch==2.5.0 torchvision==0.20.0 \
    && PIP_CONSTRAINT=/tmp/req/constraints-measure_env.txt \
       conda run -n measure_env pip install --no-cache-dir \
         -r /tmp/req/requirements-measure_env.txt \
    && (PIP_CONSTRAINT=/tmp/req/constraints-measure_env.txt \
        conda run -n measure_env pip install --no-cache-dir \
         -r /tmp/environments/measure_env_requirements.txt.clean || \
        echo "historical pins partly unsatisfiable; the derived set above is installed") \
    && conda clean -afy

# --- samm : SAM2, SAM2-PAL, DINOLand, torchvision detectors ----------------
RUN conda create -n samm python=3.12.4 -y \
    && conda run -n samm pip install --no-cache-dir \
         --index-url https://download.pytorch.org/whl/cu121 \
         torch==2.4.0 torchvision==0.19.0 \
    && PIP_CONSTRAINT=/tmp/req/constraints-samm.txt \
       conda run -n samm pip install --no-cache-dir \
         -r /tmp/req/requirements-samm.txt \
    && (PIP_CONSTRAINT=/tmp/req/constraints-samm.txt \
        conda run -n samm pip install --no-cache-dir \
         -r /tmp/environments/samm_requirements.txt.clean || \
        echo "historical pins partly unsatisfiable; the derived set above is installed") \
    && conda clean -afy

# --- detectron2_env : Mask R-CNN -------------------------------------------
# The only component that must be compiled. Pinned to the commit this system was
# developed against; TORCH_CUDA_ARCH_LIST is set explicitly because nvcc cannot
# detect a GPU at build time and would otherwise emit code for none.
#
# TWO flags are needed here, and neither is optional:
#
#   --no-build-isolation : detectron2's setup.py imports torch at build time to
#     locate the CUDA toolchain. A PEP 517 build runs in a fresh environment
#     where torch is absent, so it dies with "ModuleNotFoundError: No module
#     named 'torch'" before the compiler is ever reached.
#
#   NOT -e (non-editable) : with `-e` and no pyproject.toml, pip falls back to
#     `setup.py develop`, and setuptools >= 64 implements that by RE-INVOKING pip
#     as `pip install -e . --use-pep517 --no-deps`. That inner call does not
#     inherit --no-build-isolation, so the isolated build comes back and fails in
#     exactly the same way. A container has no reason to install it editable.
RUN if [ "$INCLUDE_DETECTRON2" = "true" ]; then set -eux; \
      conda create -n detectron2_env python=3.9.20 -y; \
      conda run -n detectron2_env pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 torchvision==0.19.1; \
      PIP_CONSTRAINT=/tmp/req/constraints-detectron2_env.txt \
        conda run -n detectron2_env pip install --no-cache-dir \
        -r /tmp/req/requirements-detectron2_env.txt; \
      # historical pins, best effort: this file asks for opencv-python==4.10.0,
      # a version that does not exist (4.10.0.82/84 do), so it is allowed to fail
      conda run -n detectron2_env pip install --no-cache-dir \
        -r /tmp/environments/detectron2_env_requirements.txt.clean || true; \
      git clone https://github.com/facebookresearch/detectron2 /opt/detectron2; \
      git -C /opt/detectron2 checkout "$DETECTRON2_COMMIT"; \
      TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" FORCE_CUDA=1 \
        conda run -n detectron2_env pip install --no-cache-dir \
          --no-build-isolation /opt/detectron2; \
      conda clean -afy; \
    fi

# --- metric3d : depth from defocus (opt-in) --------------------------------
RUN if [ "$INCLUDE_METRIC3D" = "true" ]; then set -eux; \
      conda create -n metric3d python=3.9 -y; \
      conda run -n metric3d pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu117 \
        torch==2.0.1 torchvision==0.15.2; \
      if [ -f /tmp/environments/metric3d_env_requirements.txt ]; then \
        grep -vE '@ file://|^torch(vision|audio)?[=<>]|^torch$' \
          /tmp/environments/metric3d_env_requirements.txt > /tmp/m3d.txt || true; \
        conda run -n metric3d pip install --no-cache-dir -r /tmp/m3d.txt; \
      fi; \
      conda clean -afy; \
    fi

# --- SAM2 ------------------------------------------------------------------
# Not on PyPI, so it cannot be a declared dependency of any published package —
# it is installed here from source instead, which is exactly what this image is for.
RUN git clone https://github.com/facebookresearch/segment-anything-2 /opt/sam2 \
    && git -C /opt/sam2 checkout "$SAM2_COMMIT" \
    && PIP_CONSTRAINT=/tmp/req/constraints-samm.txt \
       conda run -n samm pip install --no-cache-dir --no-build-isolation /opt/sam2 \
    && conda clean -afy

# --- the Descriptron code --------------------------------------------------
COPY segment-anything-2/gui ${DESCRIPTRON_HOME}/gui
COPY docker/descriptron /usr/local/bin/descriptron
RUN chmod +x /usr/local/bin/descriptron

# Model weights are NOT baked in: they are large, they version independently of
# the code, and a cached volume means an image rebuild does not re-download them.
VOLUME ["/data", "/weights"]
ENV TORCH_HOME=/weights/torch \
    HF_HOME=/weights/huggingface \
    DESCRIPTRON_WEIGHTS=/weights
# DINOLand pulls facebook/dinov3-vitb16-pretrain-lvd1689m through transformers.
# DINOv3 is GATED on HuggingFace: accept the terms on the model page once, then
# pass a token, or the first landmark run fails with a 401 rather than a hint:
#   docker run -e HF_TOKEN=hf_... ...

# The GUI launches DINOLand with `conda run -n $DINOLAND_ENV` and defaults to an
# environment called "biorag" (descriptron-v2-v73.py:15459) which does not exist
# here or on the development machine. samm has torch and transformers, which is
# all DINOLand needs, so point it there rather than create an empty alias.
ENV DINOLAND_ENV=samm

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/descriptron"]
CMD ["help"]

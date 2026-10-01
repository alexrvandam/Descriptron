#!/usr/bin/env bash
# =============================================================================
# build_sam3_env.sh - optional SAM 3 environment for Descriptron (no existing env is touched)
# =============================================================================
# SAM 3 needs Python 3.12 and PyTorch >= 2.7, so it gets its own conda env named `sam3`;
# the GUI finds it next to its own env, or through `conda run -n sam3`.
#
#   bash environments/build_sam3_env.sh
#
# Overrides: CONDA (conda executable), SAM3_ENV (env name, default sam3),
#            SAM3_SRC (where the SAM 3 source is cloned, default $HOME/sam3),
#            TORCH_INDEX (PyTorch wheel index, default CUDA 12.8).
# The checkpoints are gated: request access at https://huggingface.co/facebook/sam3, then run
#   conda run -n sam3 hf auth login
# once. The first SAM 3 run downloads the checkpoint (3.3 GB) into the Hugging Face cache.
# =============================================================================
set -euo pipefail
CONDA="${CONDA:-conda}"
ENV="${SAM3_ENV:-sam3}"
SRC="${SAM3_SRC:-$HOME/sam3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
COMMIT=2345a4ad109ac29c569da749c91d84f10dc08c40      # the SAM 3 commit Descriptron 2.5.0 was tested with
echo "== create env $ENV $(date +%T)"
"$CONDA" create -y -n "$ENV" -c conda-forge --override-channels python=3.12 pip
PY="$("$CONDA" run -n "$ENV" python -c 'import sys; print(sys.executable)')"
echo "== torch $(date +%T)"
"$PY" -m pip install --no-cache-dir torch==2.10.0 torchvision --index-url "$TORCH_INDEX"
echo "== sam3 source $(date +%T)"
[ -d "$SRC/.git" ] || git clone https://github.com/facebookresearch/sam3 "$SRC"
git -C "$SRC" checkout -q "$COMMIT"
"$PY" -m pip install --no-cache-dir -e "$SRC"
"$PY" -m pip install --no-cache-dir huggingface_hub pycocotools opencv-python-headless einops psutil "setuptools<81"   # sam3 imports pkg_resources, einops, psutil
echo "== check $(date +%T)"
"$PY" -c "import torch, sam3; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('sam3 import ok', sam3.__file__)"
echo "== DONE $(date +%T)"

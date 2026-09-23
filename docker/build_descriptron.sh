#!/usr/bin/env bash
# =============================================================================
# build_descriptron.sh — build (or resume) the descriptron:latest image
# =============================================================================
# SAFE TO RE-RUN. Docker caches completed layers, so a build interrupted by a
# suspended laptop, a GSP freeze or a dropped connection resumes at the step it
# died on rather than starting over. Just run this again.
#
#   ./build_descriptron.sh                 # build, then verify
#   ./build_descriptron.sh --with-metric3d # also build the depth env (+6 GB)
#   ./build_descriptron.sh --verify-only   # skip the build, just check the image
#
# Expect ~1 hour and ~25 GB the first time, most of it downloading three torch
# stacks and compiling Detectron2 (step 9 — the step most likely to fail).
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TAG=descriptron:latest
M3D=false
VERIFY_ONLY=false
for a in "$@"; do
  case "$a" in
    --with-metric3d) M3D=true ;;
    --verify-only)   VERIFY_ONLY=true ;;
    *) echo "unknown option: $a"; exit 1 ;;
  esac
done
LOG="$ROOT/docker/build_$(date +%Y%m%d_%H%M).log"

if [ "$VERIFY_ONLY" = false ]; then
  echo "building $TAG (metric3d=$M3D)"
  echo "log: $LOG"
  echo "this resumes from cache if a previous attempt died — nothing is wasted"
  docker build -f docker/Dockerfile \
    --build-arg INCLUDE_METRIC3D=$M3D \
    --build-arg INCLUDE_DETECTRON2=true \
    -t "$TAG" . 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo
    echo "BUILD FAILED (exit $rc). The failing step is the last '#NN ERROR' in:"
    echo "  $LOG"
    echo "Re-run this script once the cause is fixed; completed layers are cached."
    exit "$rc"
  fi
fi

# --- verification: a built image is not the same as a working one -----------
echo
echo "=== verifying $TAG ==="
fail=0
check() {  # check <description> <command...>
  local desc="$1"; shift
  if out=$(docker run --rm "$TAG" "$@" 2>&1); then
    printf '  ok    %-42s %s\n' "$desc" "$(echo "$out" | tr '\n' ' ' | cut -c1-40)"
  else
    printf '  FAIL  %-42s %s\n' "$desc" "$(echo "$out" | tr '\n' ' ' | cut -c1-60)"; fail=1
  fi
}
checksh() {  # checksh <description> <shell snippet>
  local desc="$1"; shift
  if out=$(docker run --rm --entrypoint sh "$TAG" -c "$1" 2>&1); then
    printf '  ok    %-42s %s\n' "$desc" "$(echo "$out" | tr '\n' ' ' | cut -c1-40)"
  else
    printf '  FAIL  %-42s %s\n' "$desc" "$(echo "$out" | tr '\n' ' ' | cut -c1-60)"; fail=1
  fi
}

check    "entrypoint responds"            help
check    "environments present"           envs
checksh  "measure_env imports numpy+pandas" \
         'conda run -n measure_env python -c "import numpy,pandas,cv2,pycocotools;print(numpy.__version__)"'
# assert the VERSION, not just that it imports: the first build produced a
# working image whose samm env had silently moved to torch 2.14.0+cu130
checksh  "samm torch is the pinned 2.4.0" \
         'conda run -n samm python -c "import torch;assert torch.__version__.startswith(\"2.4.0\"), torch.__version__;print(torch.__version__)"'
checksh  "detectron2_env torch is 2.4.1" \
         'conda run -n detectron2_env python -c "import torch;assert torch.__version__.startswith(\"2.4.1\"), torch.__version__;print(torch.__version__)"'
checksh  "measure_env torch is 2.5.0" \
         'conda run -n measure_env python -c "import torch;assert torch.__version__.startswith(\"2.5.0\"), torch.__version__;print(torch.__version__)"'
checksh  "samm has rembg (SAM2-PAL v21)"  'conda run -n samm python -c "import rembg;print(\"rembg ok\")"'
checksh  "SAM2 importable"                'conda run -n samm python -c "import sam2;print(\"sam2 ok\")"'
checksh  "SAM2 is the pinned commit"      'git -C /opt/sam2 rev-parse --short HEAD'
checksh  "detectron2 importable"          'conda run -n detectron2_env python -c "import detectron2;print(detectron2.__version__)"'
checksh  "detectron2 compiled ext loads"  'conda run -n detectron2_env python -c "from detectron2 import _C;print(\"_C ok\")"'
checksh  "SAM2-PAL v21 present"           'ls /opt/descriptron/gui/sam2_pal_batch_v21.py'
checksh  "DINOLand present"               'ls /opt/descriptron/gui/dinov3_landmark_transfer_v51.py'
checksh  "torchvision trainer present"    'ls /opt/descriptron/gui/torchvision_det/tv_train_v1.py'
checksh  "BioRAG programs present"        'ls /opt/descriptron/gui/measure/biorag_*.py | wc -l'
checksh  "DINOLAND_ENV points at a real env" \
         'test -d /opt/conda/envs/$DINOLAND_ENV && echo "$DINOLAND_ENV ok"'

echo
docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' | grep "^  descriptron:latest" || true
if [ "$fail" -eq 0 ]; then
  echo "  ALL CHECKS PASSED"
  echo
  echo "GPU test (needs nvidia-container-toolkit):"
  echo "  docker run --rm --gpus all $TAG shell samm -c 'python -c \"import torch;print(torch.cuda.is_available())\"'"
else
  echo "  SOME CHECKS FAILED — see above. The image exists but is not usable as built."
  exit 1
fi

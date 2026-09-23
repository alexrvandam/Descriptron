#!/usr/bin/env bash
# shellcheck disable=SC2016
# =============================================================================
# build_release_tree.sh — assemble the publishable repository
# =============================================================================
# The working copy lives inside a clone of facebookresearch/segment-anything-2,
# and git will not track files inside another git repository. So the public repo
# is assembled here instead: code only, no data, no weights, no outputs.
#
# Safe and repeatable — it only reads from the working tree. Re-run it after
# changing anything and commit the difference.
#
#   ./build_release_tree.sh [destination]      default: ~/Desktop/Descriptron-v2
# =============================================================================
set -euo pipefail
SRC="/home/localuser/Desktop/Descriptron"
GUI="$SRC/segment-anything-2/gui"
DST="${1:-$HOME/Desktop/Descriptron-v2}"

# Empty the destination WITHOUT touching .git — the earlier `rm -rf "$DST"` would
# have destroyed the repository, its remote and its tags on the second run.
mkdir -p "$DST"
find "$DST" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
mkdir -p "$DST"/{docs,docker,environments,descriptron/measure,descriptron/torchvision_det,descriptron/detectron2}

# The working tree holds every version ever written — 968 .py in measure/ alone,
# of which 67 are the live pipeline. Publishing all of them makes the repository
# unreadable and makes "51,107 lines" mean something different from what the
# paper counts. Default: the live programs, named by the script inventory that
# the supplementary figure is built from. Pass --all to publish everything.
INVENTORY="${INVENTORY:-/home/localuser/Desktop/Descriptron/docs/FigS1_script_inventory.tsv}"
ALL="${ALL:-0}"
LIVE_LIST=""
if [ "$ALL" = "0" ] && [ -f "$INVENTORY" ]; then
  # column 1 is "(left out)" for programs the inventory deliberately retires — the
  # outline miner, its figures, and the superseded instrument comparison. They are
  # kept in the working tree to reproduce old tables, and must not be published.
  LIVE_LIST="$(awk -F'\t' 'NR>1 && $2 != "" && $1 != "(left out)" {print $2}' "$INVENTORY")"
fi

is_live() {  # is_live <basename>
  [ -z "$LIVE_LIST" ] && return 0
  printf '%s\n' "$LIVE_LIST" | grep -qxF "$1"
}

copy_py() {  # copy_py <from> <to>  — .py/.sh only, no backups, live versions only
  local n=0 skipped=0
  while IFS= read -r f; do
    b="$(basename "$f")"
    case "$b" in *.bak*|*"(Copy)"*) continue;; esac
    if is_live "$b"; then cp "$f" "$2"/ && n=$((n+1)); else skipped=$((skipped+1)); fi
  done < <(find "$1" -maxdepth 1 \( -name '*.py' -o -name '*.sh' \))
  [ "$skipped" -gt 0 ] && echo "  $2: $n published, $skipped superseded versions left out"
  return 0
}

# --- the programs ----------------------------------------------------------
# measure/ first: three analysis programs also have a stale copy at the gui/ root,
# and publishing both leaves a reader unable to tell which one runs
copy_py "$GUI/measure" "$DST/descriptron/measure"
copy_py "$GUI" "$DST/descriptron"
for f in "$DST/descriptron"/*.py; do
  b="$(basename "$f")"
  if [ -f "$DST/descriptron/measure/$b" ]; then
    rm -f "$f"
    echo "  dropped duplicate at descriptron/: $b (kept measure/$b)"
  fi
done
# these two directories are shipped whole: torchvision_det is new in v2 and every
# file in it is live; from detectron2/ only the two scripts the GUI calls, plus the
# fixed-budget copy used for the backend comparison
LIVE_LIST="" copy_py "$GUI/torchvision_det" "$DST/descriptron/torchvision_det"
for f in detectron2_training_and_filterV10_and_kpts-17.py \
         detectron2_predict_and_filterV10_and_kptsV2.py \
         detectron2_training_sweep_v1.py; do
  [ -f "$GUI/detectron2/$f" ] && cp "$GUI/detectron2/$f" "$DST/descriptron/detectron2/"
done
mkdir -p "$DST/descriptron/torchvision_det/tests" "$DST/descriptron/measure/tests"
cp "$GUI/torchvision_det/tests/"*.py "$DST/descriptron/torchvision_det/tests/" 2>/dev/null || true
find "$GUI/measure/tests" -maxdepth 1 -name 'test_*.py' ! -name '*.bak*' \
     -exec cp {} "$DST/descriptron/measure/tests/" \; 2>/dev/null || true

# --- data that is part of the code, not of a dataset -----------------------
cp -r "$GUI/measure/biorag_prompts" "$DST/descriptron/measure/" 2>/dev/null || true
cp "$GUI/marmot.jpg" "$DST/descriptron/" 2>/dev/null || true
[ -d "$GUI/icons" ] && cp -r "$GUI/icons" "$DST/descriptron/" || true
cp "$SRC/segment-anything-2/gui/torchvision_det/README.md" \
   "$DST/descriptron/torchvision_det/" 2>/dev/null || true

# --- packaging, container, environments, docs ------------------------------
cp -r "$SRC/packages" "$DST/packages"
# descriptron-mcp is published here, but its master copy lives in its own folder
# (with its own .venv); bring in the current version on every rebuild
MCP_SRC="${MCP_SRC:-$HOME/Desktop/descriptron-mcp}"
rm -rf "$DST/packages/descriptron-mcp"
if [ -d "$MCP_SRC" ]; then
  mkdir -p "$DST/packages/descriptron-mcp"
  rsync -a --exclude '.venv/' --exclude 'dist/' --exclude '__pycache__/' --exclude '*.egg-info/' \
        --exclude '.pytest_cache/' --exclude '.git/' --exclude '*.txt' --exclude 'copy_into_descriptron_repo.sh' \
        "$MCP_SRC/" "$DST/packages/descriptron-mcp/"
fi
rm -rf "$DST"/packages/*/dist "$DST"/packages/*/src/*/tools "$DST"/packages/*/src/*/data
find "$DST/packages" -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true
cp "$SRC/docker/"{Dockerfile,descriptron,README.md,build_descriptron.sh,Dockerfile.mcp,build_mcp_layer.sh} "$DST/docker/"
cp "$SRC/docker/"requirements-*.txt "$SRC/docker/"constraints-*.txt "$DST/docker/"
cp "$SRC/environments/"*.yml "$SRC/environments/"*.txt "$DST/environments/" 2>/dev/null || true
cp "$SRC/README.md" "$DST/" 2>/dev/null || true
# licence, attribution and citation: the release must never go out without them
for f in LICENSE NOTICE CITATION.cff; do
  cp "$SRC/$f" "$DST/" || { echo "missing $SRC/$f — refusing to build a release without it" >&2; exit 1; }
done
# each distributable package carries them too (hatchling puts LICENSE*/NOTICE* in the wheel)
for p in "$DST"/packages/*/; do cp "$SRC/LICENSE" "$SRC/NOTICE" "$p"; done
cp "$SRC/docs/"* "$DST/docs/" 2>/dev/null || true
cp "$SRC/.dockerignore" "$DST/" 2>/dev/null || true
cp "$0" "$DST/" 2>/dev/null || true

# --- a .gitignore that can actually work, since nothing here is nested -----
cat > "$DST/.gitignore" <<'EOF'
__pycache__/
*.pyc
*.bak*
*.egg-info/
dist/
build/
.ipynb_checkpoints/

# weights and data never belong in the repository
*.pth
*.pt
*.ckpt
*.onnx
*.safetensors
checkpoints/
outputs/
*.log
EOF

echo
echo "assembled: $DST"
echo "  programs : $(find "$DST/descriptron" -name '*.py' | wc -l) .py"
echo "  size     : $(du -sh "$DST" | cut -f1)"

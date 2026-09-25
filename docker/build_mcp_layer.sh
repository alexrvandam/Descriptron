#!/usr/bin/env bash
# =============================================================================
# build_mcp_layer.sh — build the image with the MCP server on top of a release
# =============================================================================
#   docker/build_mcp_layer.sh [TAG] [BASE]
#     TAG   default ghcr.io/alexrvandam/descriptron:2.1.0
#     BASE  default ghcr.io/alexrvandam/descriptron:2.0.0
#
# Paths can be overridden: PACKAGES_DIR (built wheels in */dist), MCP_SRC.
# Only the files the layer needs are sent to Docker, never the whole tree.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:-ghcr.io/alexrvandam/descriptron:2.1.0}"
BASE="${2:-ghcr.io/alexrvandam/descriptron:2.0.0}"
PACKAGES_DIR="${PACKAGES_DIR:-$HERE/../packages}"
MCP_SRC="${MCP_SRC:-$HOME/Desktop/descriptron-mcp}"
GUI_DIR="${GUI_DIR:-$HERE/../segment-anything-2/gui}"
# programs changed since the base image, relative to gui/ (space-separated)
OVERLAY="${OVERLAY:-measure/biosyslit_rag_retrieval_v2.py measure/run_full_pipeline_v2.py measure/biorag_ontology_annotator_v2.py measure/biorag_key_feature_filter_v2.py measure/descriptron_credentials.py measure/biorag_llm_backend.py sam2_pal_batch_v21.py dinov3_landmark_transfer_v52.py descriptron-v2-v73.py descriptron-v2-v74.py measure/descriptron_convert.py measure/descriptron_coco_tools.py measure/descriptron_centerline.py measure/descriptron_joints.py measure/descriptron_metadata.py measure/descriptron_shape_stats.py measure/landmark_gpa_V2.py measure/semi_landmark_and_kpts_procrustesV42_GPA.py descriptron_video_track.py detectron2/detectron2_training_and_filterV10_and_kpts-18.py coco_combiner_V13.py coco_converter_v24.py remove_images_from_coco.py measure/build_species_treatment_docx.py measure/zenodo_upload.py}"

CTX="$(mktemp -d)"; trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/wheels" "$CTX/descriptron-mcp" "$CTX/overlay"
for rel in $OVERLAY; do
  mkdir -p "$CTX/overlay/$(dirname "$rel")"; cp "$GUI_DIR/$rel" "$CTX/overlay/$rel"; echo "  overlay: $rel"
done
for pkg in descriptron-core descriptron-vision; do
  whl=$(ls -t "$PACKAGES_DIR/$pkg/dist/"*.whl 2>/dev/null | head -1)
  [ -n "$whl" ] || { echo "no wheel in $PACKAGES_DIR/$pkg/dist — build it first" >&2; exit 1; }
  cp "$whl" "$CTX/wheels/"; echo "  wheel: $(basename "$whl")"
done
rsync -a --exclude '.venv/' --exclude 'dist/' --exclude '__pycache__/' --exclude '*.egg-info/' \
      --exclude '.pytest_cache/' --exclude '.git/' --exclude '*.txt' --exclude 'tests/' \
      "$MCP_SRC/" "$CTX/descriptron-mcp/"
cp "$HERE/descriptron" "$HERE/Dockerfile.mcp" "$CTX/"

docker build --build-arg BASE="$BASE" -f "$CTX/Dockerfile.mcp" -t "$TAG" "$CTX"
echo
echo "built $TAG on $BASE"
docker run --rm "$TAG" mcp --check

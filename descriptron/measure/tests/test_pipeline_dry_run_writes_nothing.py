"""run_full_pipeline_v2 --dry_run must not write any file into output_base: no pipeline_log.txt, no
pipeline_config.yaml, no logs/, no draft taxon profile, no trait input tables (2026-10-03: a dry run rewrote a
finished run's config and log and re-created trait_stats/ in it)."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def test_dry_run_writes_nothing(tmp_path):
    imgs = tmp_path / "images"; imgs.mkdir()
    coco = {"images": [{"id": 1, "file_name": "a_sp1_1.tif", "width": 10, "height": 10}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[0, 0, 5, 0, 5, 5]],
                             "area": 12.5, "bbox": [0, 0, 5, 5], "iscrowd": 0}],
            "categories": [{"id": 1, "name": "wing"}]}
    (tmp_path / "coco.json").write_text(json.dumps(coco))
    (tmp_path / "groups.csv").write_text("filename,group_label\na_sp1_1.tif,sp1\n")
    (tmp_path / "tree.nwk").write_text("(sp1:1,sp2:1);")
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, str(HERE / "run_full_pipeline_v2.py"), "--coco_json", str(tmp_path / "coco.json"),
                        "--image_dir", str(imgs), "--group_labels", str(tmp_path / "groups.csv"), "--output_base", str(out),
                        "--tree", str(tmp_path / "tree.nwk"), "--llm_backend", "none", "--dry_run"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    # no FILE may be written (steps still create their empty output folders in a dry run)
    files = [p for p in out.rglob("*") if p.is_file()] if out.exists() else []
    assert not files, f"dry run wrote files into output_base: {sorted(str(p.relative_to(out)) for p in files)}"


def test_dry_run_with_coded_states(tmp_path):
    """--coded_states_coco: the measured matrix goes to compiled_key_tier_measured and the coded states are added
    into compiled_key_tier (both commands planned, nothing written)"""
    imgs = tmp_path / "images"; imgs.mkdir()
    coco = {"images": [{"id": 1, "file_name": "a_sp1_1.tif", "width": 10, "height": 10}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[0, 0, 5, 0, 5, 5]],
                             "area": 12.5, "bbox": [0, 0, 5, 5], "iscrowd": 0, "attributes": {"setae": "dense"}}],
            "categories": [{"id": 1, "name": "wing"}]}
    (tmp_path / "coco.json").write_text(json.dumps(coco))
    (tmp_path / "groups.csv").write_text("filename,group_label\na_sp1_1.tif,sp1\n")
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, str(HERE / "run_full_pipeline_v2.py"), "--coco_json", str(tmp_path / "coco.json"),
                        "--image_dir", str(imgs), "--group_labels", str(tmp_path / "groups.csv"), "--output_base", str(out),
                        "--llm_backend", "none", "--dry_run", "--only_steps", "key_matrix",
                        "--coded_states_coco", str(tmp_path / "coco.json")],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    log = r.stdout + r.stderr
    i, j = log.find("compiled_key_tier_measured"), log.find("biorag_coded_states_from_coco_v1.py")
    assert i != -1 and j != -1 and i < j, log[-3000:]
    files = [p for p in out.rglob("*") if p.is_file()] if out.exists() else []
    assert not files, sorted(str(p.relative_to(out)) for p in files)

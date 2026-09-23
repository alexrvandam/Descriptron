"""Tests that need no Descriptron data: discovery, run/jobs/cancel, readers, path guard."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from descriptron_mcp import catalogue, data_tools, jobs


@pytest.fixture()
def toolbox(tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    (tools / "sub").mkdir(parents=True)
    (tools / "echo_v1.py").write_text(
        '"""echo_v1.py — print the arguments and write a file"""\n'
        "import sys, pathlib\n"
        "if __name__ == '__main__':\n"
        "    import helper_lib\n"                     # sibling import by plain name must work
        "    out = pathlib.Path(sys.argv[sys.argv.index('--output_dir') + 1])\n"
        "    out.mkdir(parents=True, exist_ok=True)\n"
        "    (out / 'result.txt').write_text(helper_lib.VALUE)\n"
        "    print('ARGS', sys.argv[1:])\n")
    (tools / "helper_lib.py").write_text('"""a library"""\nVALUE = "42"\n')
    (tools / "sub" / "sleepy_v1.py").write_text(
        '"""Sleep for a while."""\nimport time\nif __name__ == "__main__":\n'
        "    print('start', flush=True)\n    time.sleep(60)\n")
    (tools / "fail_v1.py").write_text('"""Always fails."""\nimport sys\nif __name__ == "__main__":\n'
                                      "    sys.exit(3)\n")
    monkeypatch.setenv("DESCRIPTRON_MCP_TOOL_DIRS", str(tools))
    monkeypatch.setenv("DESCRIPTRON_MCP_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("DESCRIPTRON_PYTHON", sys.executable)
    return tmp_path


def test_discovery(toolbox):
    cat = catalogue.discover()
    assert {"echo_v1", "sleepy_v1", "fail_v1"} <= set(cat)
    assert cat["echo_v1"].summary == "print the arguments and write a file"
    assert cat["helper_lib"].runnable is False
    assert catalogue.resolve(cat, "sleepy").name == "sleepy_v1"
    with pytest.raises(ValueError):
        catalogue.resolve(cat, "nope")


def test_run_now_reports_output_and_files(toolbox):
    prog = catalogue.resolve(catalogue.discover(), "echo_v1")
    out = toolbox / "out"
    r = jobs.run_now(prog, ["--output_dir", str(out), "x y"], toolbox, 60, 20)
    assert r["ok"] and "ARGS" in r["stdout_tail"] and "'x y'" in r["stdout_tail"]
    assert r["files_written"]["files"] == [str(out / "result.txt")]
    assert (out / "result.txt").read_text() == "42"


def test_failure_and_timeout(toolbox):
    cat = catalogue.discover()
    assert jobs.run_now(cat["fail_v1"], [], toolbox, 60, 5)["exit_code"] == 3
    r = jobs.run_now(cat["sleepy_v1"], [], toolbox, 1, 5)
    assert r["exit_code"] is None and "start_job" in r["note"]


def test_job_lifecycle(toolbox):
    cat = catalogue.discover()
    j = jobs.start(cat["echo_v1"], ["--output_dir", str(toolbox / "o2")], toolbox)
    for _ in range(100):
        s = jobs.status(j["job_id"])
        if s["state"] != "running":
            break
        time.sleep(0.1)
    assert s["state"] == "finished" and s["exit_code"] == 0
    assert s["files_written"]["count"] == 1
    assert jobs.all_jobs()[0]["job_id"] == j["job_id"]

    k = jobs.start(cat["sleepy_v1"], [], toolbox)
    for _ in range(50):
        if "start" in jobs.log(k["job_id"]):
            break
        time.sleep(0.1)
    assert jobs.status(k["job_id"])["state"] == "running"
    assert jobs.cancel(k["job_id"])["state"] == "cancelled"

    f = jobs.start(cat["fail_v1"], [], toolbox)
    for _ in range(100):
        if jobs.status(f["job_id"])["state"] != "running":
            break
        time.sleep(0.1)
    assert jobs.status(f["job_id"])["state"] == "failed"
    with pytest.raises(ValueError):
        jobs.status("../etc")


def test_readers(tmp_path):
    t = tmp_path / "t.tsv"
    t.write_text("species\tfeature\tv\na\tx\t1\nb\tx\t2\na\ty\t3\n")
    r = data_tools.read_table(str(t), where={"species": "a"}, columns=["v"])
    assert r["matching_rows"] == 2 and r["rows"] == [{"v": "1"}, {"v": "3"}]
    with pytest.raises(ValueError):
        data_tools.read_table(str(t), columns=["missing"])

    coco = tmp_path / "c.json"
    coco.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.tif", "width": 10, "height": 10},
                   {"id": 2, "file_name": "b.tif", "width": 0, "height": 0}],
        "categories": [{"id": 1, "name": "wing"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[0, 0, 1, 1, 1, 0]]}]}))
    s = data_tools.coco_summary(str(coco))
    assert s["annotations_per_category"] == {"wing": 1}
    assert s["images_with_zero_dimensions"]["count"] == 1
    assert s["images_without_annotations"]["examples"] == ["b.tif"]

    from PIL import Image
    Image.new("I;16", (400, 200), 30000).save(tmp_path / "ct.tif")
    data, info = data_tools.view_image(str(tmp_path / "ct.tif"), 100)
    assert data[:2] == b"\xff\xd8" and info["shown_size"] == [100, 50]


def test_evidence(tmp_path):
    (tmp_path / "species_feature_summary.csv").write_text(
        "species,feature_id,n,min,max,mean,sd,median,tier,unit,category,column\n"
        "a,wing.length_mm,3,1.0,1.2,1.1,0,1.1,key,mm,wing,length_mm\n"
        "b,wing.length_mm,2,0.8,0.9,0.85,0,0.85,key,mm,wing,length_mm\n"
        "a,wing.area_mm2,3,2,3,2.5,0,2.5,description,mm2,wing,area_mm2\n")
    e = data_tools.species_evidence(str(tmp_path), "a", tier="key")
    assert e["features"] == 1
    row = e["rows"][0]
    assert (row["all_species_min"], row["all_species_max"], row["n_species"]) == (0.8, 1.2, 2)
    with pytest.raises(ValueError, match="species codes"):
        data_tools.species_evidence(str(tmp_path), "zz")


def test_allowed_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("DESCRIPTRON_MCP_ALLOWED_ROOTS", str(tmp_path / "ok"))
    (tmp_path / "ok").mkdir()
    data_tools.check_path(str(tmp_path / "ok" / "x"))
    with pytest.raises(PermissionError):
        data_tools.check_path(str(tmp_path / "ok" / ".." / "secret"))

"""Every pipeline step leaves a provenance record, and the record can be checked.

run_full_pipeline_v2._run_step stamps <log_dir>/<step>.provenance.json (and a copy in the step's
output folder): the step script and its SHA-256, the inputs hashed BEFORE the run, a manifest of
input folders, and a SHA-256 of every file the step wrote. biorag_provenance_v1 --verify re-checks
it; which() finds the step that wrote a given result file.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import biorag_provenance_v1 as prov          # noqa: E402
import run_full_pipeline_v2 as pipe           # noqa: E402

STEP = """import argparse, pathlib, sys
ap = argparse.ArgumentParser(); ap.add_argument("--json"); ap.add_argument("--image_dir")
ap.add_argument("--output_dir"); ap.add_argument("--fail", action="store_true"); a = ap.parse_args()
(pathlib.Path(a.output_dir) / "result.csv").write_text(open(a.json).read() + "done\\n")
sys.exit(3 if a.fail else 0)
"""


def _setup(tmp_path):
    (tmp_path / "imgs").mkdir(); (tmp_path / "out").mkdir(); (tmp_path / "logs").mkdir()
    (tmp_path / "imgs" / "a.png").write_text("x")
    (tmp_path / "in.csv").write_text("a,b\n")
    (tmp_path / "step.py").write_text(STEP)
    return [sys.executable, str(tmp_path / "step.py"), "--json", str(tmp_path / "in.csv"),
            "--image_dir", str(tmp_path / "imgs"), "--output_dir", str(tmp_path / "out")]


def test_step_is_stamped_and_verifies(tmp_path):
    cmd = _setup(tmp_path)
    assert pipe._run_step(cmd, "stepT", tmp_path / "logs")
    rec_path = tmp_path / "logs" / "stepT.provenance.json"
    rec = json.loads(rec_path.read_text())
    assert rec["script"] == "step.py"
    assert rec["script_sha256"] == prov.sha256(tmp_path / "step.py")
    assert str(tmp_path / "in.csv") in rec["inputs"]
    assert str(tmp_path / "imgs") in rec["extra"]["input_folders"]
    assert str(tmp_path / "out") not in rec["extra"]["input_folders"]      # output folder is not an input
    assert str(tmp_path / "out" / "result.csv") in rec["outputs"]
    assert (tmp_path / "out" / "provenance_stepT.json").is_file()
    assert prov.verify(rec_path)["ok"]


def test_changed_input_or_output_fails_verify(tmp_path):
    cmd = _setup(tmp_path)
    pipe._run_step(cmd, "stepT", tmp_path / "logs")
    rec_path = tmp_path / "logs" / "stepT.provenance.json"
    (tmp_path / "out" / "result.csv").write_text("edited\n")
    assert not prov.verify(rec_path)["ok"]


def test_which_finds_the_step_and_rejects_edited_file(tmp_path):
    cmd = _setup(tmp_path)
    pipe._run_step(cmd, "stepT", tmp_path / "logs")
    hits = prov.which(tmp_path / "out" / "result.csv", tmp_path)
    assert hits and all(h["step"] == "stepT" for h in hits)
    (tmp_path / "out" / "result.csv").write_text("edited\n")
    assert prov.which(tmp_path / "out" / "result.csv", tmp_path) == []


def test_failed_step_still_recorded_and_dry_run_not(tmp_path):
    cmd = _setup(tmp_path)
    assert not pipe._run_step(cmd + ["--fail"], "stepF", tmp_path / "logs")
    assert json.loads((tmp_path / "logs" / "stepF.provenance.json").read_text())["extra"]["exit_code"] == 3
    assert pipe._run_step(cmd, "stepD", tmp_path / "logs", dry_run=True)
    assert not (tmp_path / "logs" / "stepD.provenance.json").exists()


def test_provenance_never_breaks_a_step(tmp_path, monkeypatch):
    cmd = _setup(tmp_path)
    monkeypatch.setattr(pipe, "_prov_import", lambda: (_ for _ in ()).throw(ImportError("gone")))
    assert pipe._run_step(cmd, "stepX", tmp_path / "logs")                 # the step still runs and succeeds
    assert (tmp_path / "out" / "result.csv").is_file()


def test_pipeline_bookkeeping_is_not_a_step_output(tmp_path):
    """A step writing into the output base must not claim the pipeline's running log, the step
    logs or other provenance records, which keep changing after it and would fail --verify."""
    cmd = _setup(tmp_path)
    logs = tmp_path / "out" / "logs"; logs.mkdir()
    (tmp_path / "out" / "pipeline_log.txt").write_text("running\n")
    (logs / "other.provenance.json").write_text("{}")
    pipe._run_step(cmd, "stepB", logs)
    rec = json.loads((logs / "stepB.provenance.json").read_text())
    assert list(rec["outputs"]) == [str(tmp_path / "out" / "result.csv")]
    (tmp_path / "out" / "pipeline_log.txt").write_text("running\nmore\n")
    assert prov.verify(logs / "stepB.provenance.json")["ok"]

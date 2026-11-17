"""Tests for descriptron_phylorag_dialog_v1 (pure functions; no display, no model, no network): the command lines
the dialog builds, the checks it makes before a step, and that the commands it builds run with phylorag_v1.py.
Run: python test_brush/test_phylorag_dialog.py"""
import json, os, subprocess, sys, tempfile
GUI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, GUI)
import descriptron_phylorag_dialog_v1 as d

ok = 0
def t(cond, name):
    global ok
    assert cond, name; ok += 1

sp = d.script_path(GUI)
t(sp is not None and sp.name == "phylorag_v1.py", "program found beside the GUI")
t(d.python_for({"DESCRIPTRON_PHYLORAG_PYTHON": "/x/py"}) == "/x/py" and d.python_for({}) == sys.executable, "python")

tmp = tempfile.mkdtemp()
P = lambda n: os.path.join(tmp, n)
open(P("m.nex"), "w").write("#NEXUS\nBEGIN DATA; DIMENSIONS NTAX=1 NCHAR=2; FORMAT SYMBOLS=\"01\";\n"
                           "CHARSTATELABELS 1 'Leaf blade shape' / lanceolate ovate, 2 'Petal indumentum' / glabrous "
                           "pubescent;\nMATRIX\n'Floria rubra' 1?\n;\nEND;\n")
open(P("taxa.tsv"), "w").write("taxon\ttip_label\nFloria alba\tFa_1\nFloria rubra\t\n")
json.dump([{"id": "1", "title": "Floria alba Smith, 1900", "text": "Floria alba Smith, 1900. Description. Leaf blade "
            "lanceolate. Petals glabrous. " * 3}], open(P("src.json"), "w"))
v = {"nexus": P("m.nex"), "chars": P("chars.json"), "taxa": P("taxa.tsv"), "sources": P("src.json"),
     "out_dir": P("run"), "keywords": "Scolopendromorpha, Geophilomorpha", "names": True, "pdf_dir": "", "text_dir": "",
     "group": "orchid", "context": "", "backend": "claude-code", "model": "", "near": "0.9", "max_sources": "3",
     "include_figures": True, "prefill": True, "review": "", "base_nexus": "", "molecular": "a.fasta; b.nwk",
     "add_missing": True, "include_unreviewed": False, "export_dir": "", "labels": "underscore", "overwrite": False}

c = d.build_cmd("init", v, sp, "PY")
t(c[:4] == ["PY", "-u", str(sp), "init"] and c[c.index("--from-nexus") + 1] == v["nexus"] and "--overwrite" not in c, "init")
c = d.build_cmd("fetch", v, sp, "PY")
t(c[c.index("--keywords") + 1:c.index("--keywords") + 3] == ["Scolopendromorpha", "Geophilomorpha"], "keywords split")
t("--names" in c and c[c.index("--taxa") + 1] == v["taxa"] and "--pdf-dir" not in c, "fetch names, no pdf")
c = d.build_cmd("dry_run", v, sp, "PY")
t(c[-1] == "--dry-run" and c[c.index("--model") + 1] == d.DEFAULT_MODEL and c[c.index("--group") + 1] == "orchid", "dry run")
t("--include-figures" in c and c[c.index("--prefill") + 1] == "clean" and c[c.index("--max-sources") + 1] == "3", "options")
c = d.build_cmd("extract", v, sp, "PY")
t("--dry-run" not in c and c[c.index("--llm-backend") + 1] == "claude-code", "extract")
t(d.review_path(v) == os.path.join(v["out_dir"], "review_sheet.tsv"), "review sheet defaults to the output folder")
c = d.build_cmd("export", v, sp, "PY")
t(c[c.index("--molecular") + 1:c.index("--molecular") + 3] == ["a.fasta", "b.nwk"] and "--add-missing-taxa" in c, "export")
t(c[c.index("--out-dir") + 1] == os.path.join(v["out_dir"], "export") and c[c.index("--labels") + 1] == "underscore",
  "export folder + labels")
c = d.build_cmd("audit", v, sp, "PY")
t(c[c.index("--out-dir") + 1] == os.path.join(v["out_dir"], "reaudit"), "audit folder")

# checks
t(d.check("init", dict(v, nexus="")) .startswith("Choose the published matrix"), "init needs a matrix")
t(d.check("fetch", dict(v, keywords="", names=False)).startswith("Give BioSysLit"), "fetch needs something to fetch")
t(d.check("fetch", dict(v, names=True, taxa="")) == "Searching by taxon name needs the taxa file.", "names need taxa")
t("not found" in d.check("dry_run", v), "characters file must exist")       # chars.json is written by init below

def run(cmd):
    r = subprocess.run([sys.executable if x == "PY" else x for x in cmd], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout

run(d.build_cmd("init", v, sp, "PY"))
t(json.load(open(v["chars"]))[0]["description"] == "Leaf blade shape", "init ran")
t(d.check("dry_run", v) is None, "dry run can start")
t(d.check("export", v).startswith("review sheet not found"), "export needs the sheet")
out = run(d.build_cmd("dry_run", v, sp, "PY"))
t("Floria alba: 1 sources (1 own)" in out and "Floria rubra: no source names this taxon" in out, "dry run ran")
t(d.check("dry_run", dict(v, near="2")).startswith("Near-verbatim threshold must be between"), "near range")
print(f"{ok} checks passed")

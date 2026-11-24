"""Tests for descriptron_ratios_dialog_v1 (pure functions; no display): program and presets found, the command lines the dialog
builds, the checks it makes, and that a built command runs the program end to end on a synthetic COCO file with a known answer.
Run: <python of the GUI env> test_brush/test_ratios_dialog.py"""
import csv, json, os, subprocess, sys, tempfile
GUI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, GUI)
import descriptron_ratios_dialog_v1 as d

ok = 0
def t(cond, name):
    global ok
    assert cond, name; ok += 1

sp = d.script_path(GUI)
t(sp is not None and sp.name == "descriptron_landmark_ratios_v2.py", "program found beside the GUI")
P = d.presets(GUI)
t("ant_head_frontal_v1" in P and os.path.isfile(P["ant_head_frontal_v1"]), "ant head preset listed")
t(d.python_for({"DESCRIPTRON_RATIOS_PYTHON": "/x/py"}) == "/x/py" and d.python_for({}) == sys.executable, "python")

tmp = tempfile.mkdtemp()
J = os.path.join(tmp, "c.json")
images, anns = [], []
for i in range(12):
    sp_ = "SPA" if i < 6 else "SPB"
    w = 6.0 if sp_ == "SPA" else 9.0                      # width/length = 0.6 vs 0.9 exactly
    pts = [[0, 0], [0, 10], [-w / 2, 4], [w / 2, 4]]
    images.append({"id": i + 1, "file_name": f"{sp_}_x{i}.png", "height": 60, "width": 60})
    anns.append({"id": i + 1, "image_id": i + 1, "category_id": 1, "keypoints": [v for p in pts for v in (p[0] + 30, p[1] + 20, 2)]})
json.dump({"images": images, "annotations": anns, "categories": [{"id": 1, "name": "line_keypoints"}]}, open(J, "w"))
v = {"json": J, "out_dir": os.path.join(tmp, "out"), "config": "", "measures": "L=1-2\nW=3-4^1-2\n# comment\n", "ratios": "WI=W/L*100",
     "keypoint_category": "", "group_labels": "", "group_regex": r"^([^_]+)_", "px_per_mm": "", "size_measure": "", "character": "",
     "positive": "", "loo_on": "ratios", "permutations": "19", "assume_same_scale": False}
c = d.build_cmd(v, sp, "PY")
t(c[:3] == ["PY", "-u", str(sp)] and c[c.index("--json") + 1] == J, "json")
t([c[i + 1] for i, x in enumerate(c) if x == "--measure"] == ["L=1-2", "W=3-4^1-2"], "measures split by line, comment dropped")
t(c[c.index("--ratio") + 1] == "WI=W/L*100" and c[c.index("--permutations") + 1] == "19", "ratio + permutations")
t("--config" not in c and "--assume_same_scale" not in c and "--size_measure" not in c, "empty options left out")

t(d.check(v) == "", "valid values pass")
t("COCO" in d.check(dict(v, json="")), "COCO needed")
t("ratio config" in d.check(dict(v, measures="", config="")), "a definition needed")
t("NAME=" in d.check(dict(v, measures="L 1-2")), "malformed measure caught")
t("not found" in d.check(dict(v, config="/no/such.tsv")), "missing config caught")
t("real sizes" in d.check(dict(v, size_measure="L")), "allometry needs a scale")
t(d.check(dict(v, size_measure="L", assume_same_scale=True)) == "", "allometry with same-scale ok")
t("character" in d.check(dict(v, positive="x")), "positive needs a character")
t("whole number" in d.check(dict(v, permutations="many")), "permutations checked")

r = subprocess.run(d.build_cmd(v, sp, sys.executable), capture_output=True, text=True, timeout=300)
t(r.returncode == 0, "program runs: " + r.stderr[-500:])
rows = list(csv.DictReader(open(os.path.join(v["out_dir"], "specimen_measurements.tsv")), delimiter="\t"))
wi = {r_["group"]: float(r_["WI"]) for r_ in rows}
t(abs(wi["SPA"] - 60) < 1e-6 and abs(wi["SPB"] - 90) < 1e-6, "known ratios 60 and 90")
t(os.path.exists(os.path.join(v["out_dir"], "summary.txt")) and os.path.exists(os.path.join(v["out_dir"], "pairwise_dunn.tsv")), "outputs")

vp = dict(v, measures="", ratios="", config=P["ant_head_frontal_v1"], out_dir=os.path.join(tmp, "out2"))
c = d.build_cmd(vp, sp, "PY")
t(c[c.index("--config") + 1] == P["ant_head_frontal_v1"] and "--measure" not in c, "preset config used")
print(f"{ok} checks passed")

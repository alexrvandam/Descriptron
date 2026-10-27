"""Tests for descriptron_dinomask_dialog_v1 (pure functions; no display, no GPU).
Run: python test_brush/test_dinomask_dialog.py"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import descriptron_dinomask_dialog_v1 as dm

ok = 0
def t(cond, name):
    global ok
    assert cond, name; ok += 1

tmp = tempfile.mkdtemp()
for f in ("a.tif", "b.tif"):
    open(os.path.join(tmp, f), "w").close()
coco = {"images": [{"id": 1, "file_name": "a.tif"}, {"id": 2, "file_name": "b.tif"}],
        "categories": [{"id": 1, "name": "head"}, {"id": 2, "name": "line_keypoints"}, {"id": 3, "name": "Trash"}, {"id": 4, "name": "gaster"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[0, 0, 5, 0, 5, 5]]},
                        {"id": 2, "image_id": 1, "category_id": 2, "keypoints": [1, 1, 2], "segmentation": []},
                        {"id": 3, "image_id": 2, "category_id": 3, "segmentation": [[0, 0, 5, 0, 5, 5]]},
                        {"id": 4, "image_id": 2, "category_id": 4, "segmentation": [[0, 0, 5, 0, 5, 5]]}]}
cp = os.path.join(tmp, "train.json"); json.dump(coco, open(cp, "w"))
files, parts, n = dm.coco_summary(cp)
t(files == ["a.tif", "b.tif"], "files"); t(parts == ["gaster", "head"], "parts exclude keypoints/Trash"); t(n == 2, "mask count")
t(dm.images_root(cp, files) == tmp, "images root = COCO folder when files are there")
t(dm.images_root(cp, ["/x/a.tif", "/y/b.tif"]) == "/", "absolute names -> /")
t(dm.count_found(tmp, files) == 2, "found")
sd = dm.script_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
t(sd is not None and (sd / "dinomask_predict_rot4_v1.py").is_file(), "script dir")
base = dict(coco=cp, images=tmp, out=os.path.join(tmp, "o"), model="", size="0", epochs="30", augs=["rot180", "hflip", "bogus"],
            partial=True, parts=parts, predict_dir=tmp, test_list=["a.tif"], run_dir="", fg_border=True, extra="--seed 3")
c = dm.build_cmd(dict(base, mode="train_predict"), sd, "PY")
t(c[:3] == ["PY", "-u", str(sd / "dinomask_train_predict_v4.py")], "train script")
t(c[c.index("--model") + 1] == dm.MODEL_VITL, "ViT-L default when model empty")
t(c[c.index("--aug") + 1] == "rot180,hflip", "augs filtered")
t(c[c.index("--partial_bg") + 1] == "gaster,head", "partial_bg vocabulary")
t("--predict_dir" in c and "--test_list" not in c and c[-2:] == ["--seed", "3"], "predict mode + extra")
t(c[c.index("--fg_mask") + 1] == "border", "fg border"); t(c[c.index("--size") + 1] == "0", "native size")
t(c[c.index("--species_regex") + 1] == "^$" and c[c.index("--min_count") + 1] == "1", "recipe flags")
c = dm.build_cmd(dict(base, mode="holdout", partial=False), sd, "PY")
t(c[c.index("--test_list") + 1] == "a.tif" and "--predict_dir" not in c and "--partial_bg" not in c and "--fg_mask" not in c, "holdout")
c = dm.build_cmd(dict(base, mode="predict_run", run_dir="/r", size="0"), sd, "PY")
t(c[2].endswith("dinomask_predict_rot4_v1.py") and c[c.index("--run") + 1] == "/r" and c[c.index("--size") + 1] == "2048", "rot4, size 0 -> 2048")
t(dm.check(dict(base, mode="train_predict", n_found=2)) == "", "valid")
t("output" in dm.check(dict(base, mode="train_predict", out="")), "no out")
t("not found" not in dm.check(dict(base, mode="holdout", n_found=1)) and dm.check(dict(base, mode="holdout", test_list=[])) != "", "holdout needs picks")
t("None of the training images" in dm.check(dict(base, mode="train_predict", n_found=0)), "images missing")
t("head.pt" in dm.check(dict(base, mode="predict_run", run_dir=tmp)), "run dir needs head.pt")
t(dm.dinov3_python("/x/python", {"DESCRIPTRON_DINOV3_PYTHON": "/my/py"})[0] == "/my/py", "env override")
py, why = dm.dinov3_python()
t(isinstance(py, str) and py, "python chosen: " + why)
t(dm._version_ok("4.56.0") and dm._version_ok("5.7.0") and not dm._version_ok("4.52.4"), "version check")
os.makedirs(os.path.join(tmp, "o"), exist_ok=True)
json.dump({"classes": ["background", "head"], "ious": {"head": [0.8, 10]}}, open(os.path.join(tmp, "o", "results.json"), "w"))
t("0.80" in dm.summarize(os.path.join(tmp, "o"), "holdout"), "summary")
# 2.8.3: hypercolumn points probe
kpc = {"images": [{"id": "a.tif", "file_name": "a.tif"}, {"id": "b.tif", "file_name": "b.tif"}],
       "categories": [{"id": 1, "name": "keypoints"}],
       "annotations": [{"id": 1, "image_id": "a.tif", "category_id": 1, "keypoints": [10, 10, 2, 0, 0, 0, 30, 30, 2],
                        "keypoints_clicked": [10, 10, 2, 20, 20, 0, 30, 30, 2]},
                       {"id": 2, "image_id": "b.tif", "category_id": 1, "keypoints": [5, 5, 2]}]}
kp_path = os.path.join(tmp, "clicks.json"); json.dump(kpc, open(kp_path, "w"))
names, npos, nneg = dm.points_summary(kp_path)
t(names == ["a.tif", "b.tif"] and npos == 3 and nneg == 1, "points summary: positives, not-this negatives from keypoints_clicked")
kdir = os.path.join(tmp, "kp"); os.makedirs(kdir, exist_ok=True); json.dump(kpc, open(os.path.join(kdir, "a_keypoints.json"), "w"))
t(dm.points_summary(kdir)[1] == 3, "points summary of a folder of <image>_keypoints.json")
t(dm.points_summary(cp)[1] == 1, "mask COCO: only its keypoint annotation counts")
pb = dict(base, coco=kp_path, name="scale seta", predict_dir=tmp, size="2048", extra="--ablate")
c = dm.build_cmd(dict(pb, mode="points_probe"), sd, "PY")
t(c[2].endswith("dinomask_hypercolumn_probe_v1.py") and c[c.index("--points") + 1] == kp_path, "probe script + clicks")
t(c[c.index("--name") + 1] == "scale_seta" and c[c.index("--model") + 1] == dm.MODEL_VITL, "name (spaces -> _) + ViT-L")
t(c[c.index("--predict_dir") + 1] == tmp and c[-1] == "--ablate" and "--aug" not in c, "probe predict dir + extra, no mask flags")
c = dm.build_cmd(dict(pb, mode="points_probe", predict_dir=""), sd, "PY")
t("--predict_dir" not in c, "probe: train only when no folder")
c = dm.build_cmd(dict(pb, mode="probe_run", run_dir="/p"), sd, "PY")
t(c[c.index("--run") + 1] == "/p" and "--points" not in c and "--predict_dir" in c, "probe_run")
t(dm.check(dict(pb, mode="points_probe", n_pos=20, n_found=2)) == "", "probe valid")
t("at least 5" in dm.check(dict(pb, mode="points_probe", n_pos=3, n_found=2)), "probe needs clicks")
t("None of the clicked images" in dm.check(dict(pb, mode="points_probe", n_pos=20, n_found=0)), "probe images missing")
t("probe.joblib" in dm.check(dict(pb, mode="probe_run", run_dir=tmp)), "probe_run needs probe.joblib")
po = os.path.join(tmp, "po"); os.makedirs(po, exist_ok=True)
json.dump({"name": "scale_seta", "positive_clicks": 25, "negative_clicks": 4, "validation": "leave-one-image-out",
           "results": {"hypercolumn L6+L12+L18+L23": {"auc": 0.95, "ap": 0.8, "ap_chance": 0.06}}}, open(os.path.join(po, "results.json"), "w"))
json.dump({"a.tif": {"total": 40}, "b.tif": {"total": 2}}, open(os.path.join(po, "counts.json"), "w"))
sm = dm.summarize(po, "points_probe")
t("AUC 0.950" in sm and "42 detections" in sm and "predictions_keypoints.json" in sm, "probe summary")
print(f"{ok}/{ok} passed  (python chosen here: {py} [{why}])")

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
print(f"{ok}/{ok} passed  (python chosen here: {py} [{why}])")

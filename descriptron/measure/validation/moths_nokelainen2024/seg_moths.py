"""Segment spread moths in museum photographs: Florence-2 phrase grounding ('a moth') -> box -> SAM2 box prompt ->
largest component, holes filled, antennae/legs removed by a morphological opening. Writes a cropped RGBA PNG per
image (alpha = mask) and a QC row. Rejects detections that do not look like a spread moth."""
import argparse, csv, sys
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("--images", nargs="+"); ap.add_argument("--list"); ap.add_argument("--out", required=True)
ap.add_argument("--sam2_ckpt", required=True); ap.add_argument("--sam2_cfg", default="sam2.1_hiera_l.yaml")
a = ap.parse_args()
from transformers import AutoProcessor, AutoModelForCausalLM
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
dev = "cuda"
proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
flo = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float16).to(dev).eval()
sam = SAM2ImagePredictor(build_sam2(a.sam2_cfg, a.sam2_ckpt, device=dev))
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
files = a.images or [l.strip() for l in open(a.list) if l.strip()]
qc = open(out / "qc.csv", "a", newline=""); w = csv.writer(qc)
def box_of(img):
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    small = img.copy(); small.thumbnail((1024, 1024)); s = img.width / small.width
    inp = proc(text=task + "a moth", images=small, return_tensors="pt").to(dev, torch.float16)
    with torch.no_grad():
        g = flo.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"], max_new_tokens=256, num_beams=3)
    r = proc.post_process_generation(proc.batch_decode(g, skip_special_tokens=False)[0], task=task, image_size=small.size)[task]
    bb = [np.array(b) * s for b in r["bboxes"]]
    return max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if bb else None
for f in files:
    f = Path(f); stem = f.stem
    try:
        img = Image.open(f).convert("RGB")
    except Exception as e:
        w.writerow([stem, "unreadable", str(e)]); continue
    b = box_of(img)
    if b is None:
        w.writerow([stem, "no_detection", ""]); qc.flush(); continue
    arr = np.array(img); H, W = arr.shape[:2]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam.set_image(arr); m, sc, _ = sam.predict(box=b[None], multimask_output=False)
    m = m[0].astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        w.writerow([stem, "empty_mask", ""]); qc.flush(); continue
    m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); m = np.zeros_like(m); cv2.drawContours(m, cnts, -1, 1, -1)
    x, y, bw, bh = cv2.boundingRect(m)
    r = max(3, int(round(0.012 * bw)))                     # opening removes antennae and legs (thin structures)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m); m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    x, y, bw, bh = cv2.boundingRect(m); area = int(m.sum())
    hull = cv2.convexHull(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]); solidity = area / cv2.contourArea(hull)
    touches = x <= 2 or y <= 2 or x + bw >= W - 2 or y + bh >= H - 2
    ok = (1.15 <= bw / bh <= 3.5) and (0.55 <= solidity <= 0.97) and not touches and bw > 250 and float(sc[0]) > 0.8
    status = "ok" if ok else "rejected"
    pad = int(0.03 * bw); x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + bw + pad), min(H, y + bh + pad)
    rgba = np.dstack([arr[y0:y1, x0:x1], 255 * m[y0:y1, x0:x1]])
    (out / status).mkdir(exist_ok=True); Image.fromarray(rgba).save(out / status / f"{stem}.png")
    w.writerow([stem, status, f"aspect={bw/bh:.2f};solidity={solidity:.2f};score={float(sc[0]):.2f};touch={touches};width={bw}"]); qc.flush()
    print(stem, status, f"{bw/bh:.2f} {solidity:.2f} {float(sc[0]):.2f}", flush=True)

#!/usr/bin/env python3
"""
extract_foundation_embeddings_v1.py - image embeddings from biodiversity foundation models, per specimen
=====================================================================================================

Turns every image of a reference set into an embedding with one of

  bioclip     BioCLIP (Imageomics, ViT-B/16, trained on TreeOfLife-10M)            hf-hub:imageomics/bioclip
  bioclip2    BioCLIP 2 (Imageomics, ViT-L/14, TreeOfLife-200M)                    hf-hub:imageomics/bioclip-2
  clibd5m     CLIBD / BIOSCAN-CLIP image encoder trained on BIOSCAN-5M with DNA and text
              (bioscan-ml/clibd, ckpt/bioscan_clip/ver_1_0/bioscan_5m/image_dna_text_4gpu/best.pth):
              timm ViT-B/16 + LoRA (r=4) on every block's qkv + a 768-d head, rebuilt here and loaded STRICTLY

and writes one row per specimen, so the embedding can be scored with exactly the machinery Descriptron's
character matrix uses (biorag_congruence_compare_v1.py --extra_continuous NAME=TABLE.tsv).

Specimens are identified with the PIPELINE'S OWN rule: species code from the group-labels file, specimen number
from the taxon profile's `specimen_id.number_regex` (biorag_feature_policy.specimen_id). Each image is embedded
whole (as photographed), L2-normalised, and averaged per specimen and view (the structure named in the file name),
so a specimen's row holds one block of columns per view: "<view>.<model>_<k>". Missing views are left empty.

  python extract_foundation_embeddings_v1.py --model bioclip \
      --image_dir <images> --group_labels group_labels.csv --taxon_profile diaphorina.yaml \
      --out_dir <out>
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))          # gui/measure
import biorag_feature_policy as pol                  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
CLIBD_REPO = "bioscan-ml/clibd"
CLIBD_CKPT = "ckpt/bioscan_clip/ver_1_0/bioscan_5m/image_dna_text_4gpu/best.pth"


# ── CLIBD image encoder (same module names as bioscanclip.model.image_encoder, so the checkpoint loads strictly) ──
class _LoRA_qkv_timm(nn.Module):
    def __init__(self, qkv, a_q, b_q, a_v, b_v):
        super().__init__()
        self.qkv, self.linear_a_q, self.linear_b_q, self.linear_a_v, self.linear_b_v = qkv, a_q, b_q, a_v, b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += self.linear_b_q(self.linear_a_q(x))
        qkv[:, :, -self.dim:] += self.linear_b_v(self.linear_a_v(x))
        return qkv


class CLIBDImageEncoder(nn.Module):
    def __init__(self, vit, r=4, num_classes=768):
        super().__init__()
        for blk in vit.blocks:
            q = blk.attn.qkv; d = q.in_features
            blk.attn.qkv = _LoRA_qkv_timm(q, nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False),
                                          nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False))
        self.base_image_encoder = vit
        self.base_image_encoder.reset_classifier(num_classes=num_classes)

    def forward(self, x):
        return self.base_image_encoder(x)


def load_model(name, device):
    if name in ("bioclip", "bioclip2"):
        import open_clip
        hub = {"bioclip": "hf-hub:imageomics/bioclip", "bioclip2": "hf-hub:imageomics/bioclip-2"}[name]
        model, _, preprocess = open_clip.create_model_and_transforms(hub)
        model.eval().to(device)
        return (lambda x: model.encode_image(x)), preprocess, {"source": hub}
    if name == "clibd5m":
        import timm
        from huggingface_hub import hf_hub_download
        from torchvision import transforms
        path = hf_hub_download(CLIBD_REPO, CLIBD_CKPT)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        enc_sd = {k[len("image_encoder."):]: v for k, v in sd.items() if k.startswith("image_encoder.")}
        # the released 5M checkpoint names the ViT "lora_vit." (an older version of the class); same tensors
        enc_sd = {("base_image_encoder." + k[len("lora_vit."):]) if k.startswith("lora_vit.") else k: v
                  for k, v in enc_sd.items()}
        if not enc_sd:
            raise SystemExit(f"no image_encoder.* weights in {path}; keys look like {list(sd)[:5]}")
        enc = CLIBDImageEncoder(timm.create_model("vit_base_patch16_224", pretrained=False))
        enc.load_state_dict(enc_sd, strict=True)          # every tensor must match, none left over
        enc.eval().to(device)
        pre = transforms.Compose([transforms.Resize(256, antialias=True), transforms.CenterCrop(224), transforms.ToTensor(),
                                  transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                       (0.26862954, 0.26130258, 0.27577711))])
        return enc, pre, {"source": f"{CLIBD_REPO}/{CLIBD_CKPT}", "loaded_tensors": len(enc_sd)}
    raise SystemExit(f"unknown model {name}")


def _rgb(path, background="white"):
    """RGB for the model; a transparent background is composited onto `background`, never revealed"""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA"); bg = Image.new("RGBA", im.size, background)
        return Image.alpha_composite(bg, im).convert("RGB")
    return im.convert("RGB")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["bioclip", "bioclip2", "clibd5m"])
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--group_labels", required=True, help="CSV: filename,group_label (the pipeline's species codes)")
    ap.add_argument("--taxon_profile", required=True, help="taxon profile YAML (specimen_id.number_regex)")
    ap.add_argument("--view_regex", default=r"(forewing|head|rostrum|metaleg|te+rminalia)",
                    help="regex whose first group names the view/structure in the file name")
    ap.add_argument("--alpha_background", default="white",
                    help="images with transparency (cutouts) are composited onto this colour; the pixels under a "
                         "transparent background still hold the original photograph, which plain RGB conversion "
                         "would show the model (default white)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)

    profile = pol.load_taxon_profile(a.taxon_profile)
    gl = pd.read_csv(a.group_labels)
    gl.columns = [c.strip() for c in gl.columns]
    fcol = "filename" if "filename" in gl.columns else gl.columns[0]
    lcol = "group_label" if "group_label" in gl.columns else gl.columns[1]
    rows = []
    for fn, sp in zip(gl[fcol], gl[lcol]):
        p = Path(a.image_dir) / str(fn)
        if not p.is_file():
            print(f"  ! missing image: {fn}"); continue
        m = re.search(a.view_regex, str(fn), flags=re.I)
        view = re.sub(r"te+rminalia", "terminalia", m.group(1).lower()) if m else "other"
        rows.append({"image": str(fn), "species": str(sp).strip(), "view": view,
                     "specimen_id": pol.specimen_id(str(fn), str(sp).strip(), profile)})
    meta = pd.DataFrame(rows)
    print(f"{len(meta)} images, {meta.specimen_id.nunique()} specimens, {meta.species.nunique()} species; "
          f"views {meta.view.value_counts().to_dict()}")

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc, pre, info = load_model(a.model, device)
    t0 = time.time(); embs = []
    with torch.inference_mode():
        for i in range(0, len(meta), a.batch):
            x = torch.stack([pre(_rgb(Path(a.image_dir) / f, a.alpha_background)) for f in meta.image[i:i + a.batch]])
            e = enc(x.to(device)).float()
            embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
            print(f"  {min(i + a.batch, len(meta))}/{len(meta)}", flush=True)
    E = np.concatenate(embs)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.model}_image_embeddings.npz", embeddings=E, images=meta.image.values,
                        specimen_id=meta.specimen_id.values, view=meta.view.values, species=meta.species.values)
    dim = E.shape[1]
    cols = [f"{a.model}_{k}" for k in range(dim)]
    df = pd.concat([meta[["specimen_id", "view"]], pd.DataFrame(E, columns=cols)], axis=1)
    per = df.groupby(["specimen_id", "view"])[cols].mean()
    wide = per.unstack("view")
    wide.columns = [f"{v}.{c}" for c, v in wide.columns]
    wide = wide[sorted(wide.columns, key=lambda c: (c.split(".")[0], int(c.rsplit("_", 1)[1])))]
    wide.index.name = "specimen_id"
    wide.to_csv(out / f"{a.model}.tsv", sep="\t")
    meta.to_csv(out / f"{a.model}_images.tsv", sep="\t", index=False)
    rep = {"model": a.model, **info, "alpha_background": a.alpha_background, "images": int(len(meta)), "specimens": int(wide.shape[0]), "dim": int(dim),
           "views": sorted(meta.view.unique().tolist()), "columns": int(wide.shape[1]), "device": device,
           "seconds": round(time.time() - t0, 1), "torch": torch.__version__}
    json.dump(rep, open(out / f"{a.model}_report.json", "w"), indent=2)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()

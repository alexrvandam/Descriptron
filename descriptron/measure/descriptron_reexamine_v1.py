#!/usr/bin/env python3
"""
descriptron_reexamine_v1.py - which specimens should the taxonomist look at again, and why
==========================================================================================

Brings together, for EVERY specimen, the evidence the pipeline already has that a specimen may not belong where it
is filed, and writes it as one ranked list, a readable report and a zoomable map:

  matrix       the character matrix names it (its own record withheld) as another species; what each character set
               calls it, and how many sets agree with its label           calibration/matrix_identification_distances.tsv
  key          the key, with the specimen's record withheld, names it as another species   key*/identification_test.tsv
  values       measurements far from its species (robust z)                compiled_key_tier/outlier_flags.tsv
  outlines     outlines the annotation screen flagged                       annotation_screen/annotation_screen.tsv
  hypothesis   OPTIONAL: an outside species hypothesis (DNA, field notes...) tested by descriptron_trait_stats;
               specimens it calls 're-examine'                              trait_stats*/specimen_flags.csv

  Two separate questions:
  identity     is the specimen filed under the wrong species?  (matrix, key, outside hypothesis)
               re-examine  the matrix names it as another species AND the key or an outside hypothesis agrees, or
                           most character sets disagree with its label
               check       the matrix alone names it elsewhere (or a key at least as reliable as the matrix does)
               fits        none
  annotation   is an outline or a measurement wrong?  (flagged outlines; a value >= --fold times, or <= 1/--fold of,
               its species' median) - reported on its own; it never counts towards an identity flag
  status       the identity status, or "check annotation" when only the annotation is in question
  The matrix leads identity because it is the instrument measured to name best on this material.
A flag is a reason to look again, never a determination.

Outputs (out_dir):
  specimens_to_reexamine.tsv   every specimen, ranked, with the reasons in words and the evidence in columns
  species_summary.tsv          per species: specimens, named back, confused with, flagged specimens
  reexamine_report.md          the list a taxonomist reads
  reexamine_map.html           one self-contained file (no internet needed): species placed by morphological
                               similarity, links = confusions, zoom in to see specimens; search, click for details.
                               Scales to thousands of species (layout computed here, drawn on a canvas).

  python descriptron_reexamine_v1.py --calibration_dir M/calibration --key_loo M/key_v2/identification_test.tsv \
      --outlier_flags M/compiled_key_tier/outlier_flags.tsv --annotation_screen M/annotation_screen/annotation_screen.tsv \
      --group_labels group_labels.csv --taxon_profile profile.yaml --out_dir M/reexamine
"""
import argparse
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                 # noqa: E402

VERSION = "1.0"


# ── matrix verdict ───────────────────────────────────────────────────────────────────────────────
def matrix_verdict(dist: pd.DataFrame, sets=None):
    """per specimen: combined name (mean rank over sets, as biorag_congruence_compare_v1.identify), per-set names,
    sets agreeing with the label, rank of its own species; plus a specimen x species score in [0, 1] for the map."""
    if sets:
        dist = dist[dist["set"].isin(sets)]
    dist = dist.dropna(subset=["d"]).copy()
    dist["rank"] = dist.groupby(["specimen_id", "set"])["d"].rank(method="average")
    n = dist.groupby(["specimen_id", "set"])["candidate_species"].transform("count")
    dist["u"] = np.where(n > 1, (dist["rank"] - 1) / (n - 1).clip(lower=1), 0.0)
    comb = dist.groupby(["specimen_id", "species", "candidate_species"])["rank"].mean().reset_index()
    rows = {}
    for (sid, sp), g in comb.groupby(["specimen_id", "species"]):
        g = g.sort_values("rank")
        named = g.iloc[0]["candidate_species"]
        own = g[g["candidate_species"] == sp]
        own_pos = int((g["rank"] < float(own["rank"].iloc[0])).sum()) + 1 if len(own) else None
        second = g.iloc[1]["rank"] if len(g) > 1 else np.nan
        rows[sid] = {"specimen_id": sid, "species": sp, "matrix_named": named, "matrix_own_rank": own_pos,
                     "matrix_margin": round(float(second - g.iloc[0]["rank"]), 3) if second == second else None}
    per_set = dist.loc[dist.groupby(["specimen_id", "set"])["d"].idxmin(), ["specimen_id", "set", "candidate_species"]]
    for sid, g in per_set.groupby("specimen_id"):
        r = rows.get(sid)
        if r is None:
            continue
        r["sets_scored"] = len(g)
        r["sets_agreeing"] = int((g["candidate_species"] == r["species"]).sum())
        r["per_set"] = "; ".join(f"{s}: {c}" for s, c in zip(g["set"], g["candidate_species"]))
    score = dist.groupby(["specimen_id", "candidate_species"])["u"].mean().unstack()
    return pd.DataFrame(rows.values()).set_index("specimen_id"), score


# ── other evidence, mapped to specimen ids with the pipeline's own rule ─────────────────────────────
def load_key(path):
    """key name per specimen (record withheld) and the key's own accuracy on this material"""
    if not path or not Path(path).exists():
        return {}, None
    k = pd.read_csv(path, sep="\t")
    if "loo" not in k.columns:
        return {}, None
    acc = float((k["loo"].astype(str) == k["species"].astype(str)).mean()) if "species" in k.columns else None
    return {r["specimen_id"]: str(r["loo"]) for _, r in k.iterrows()}, acc


def load_outliers(path, profile, image_species):
    out = defaultdict(list)
    if not path or not Path(path).exists():
        return out
    o = pd.read_csv(path, sep="\t")
    for _, r in o.iterrows():
        sp = str(r.get("species", "")) or image_species.get(str(r["image_base"]), "")
        sid = pol.specimen_id(str(r["image_base"]), sp, profile)
        v, med = float(r["value"]), float(r["species_median"])
        ratio = v / med if med else float("inf")
        out[sid].append((f"{r.get('category', '')} {str(r.get('feature', '')).replace('meas_', '')} "
                         f"{v:.3g} vs species median {med:.3g} ({ratio:.2g}x)", ratio))
    return out


def load_annotation_screen(path, profile, image_species):
    out = defaultdict(list)
    if not path or not Path(path).exists():
        return out
    a = pd.read_csv(path, sep="\t")
    if "flag" not in a.columns:
        return out
    a = a[a["flag"].astype(str).str.lower() == "true"]
    for _, r in a.iterrows():
        img = str(r["image"])
        sp = image_species.get(img)
        if not sp:
            continue
        why = [w for w, c in (("misses the specimen", "fails_coverage"), ("outside its structure", "fails_containment"))
               if str(r.get(c, "")).lower() == "true"]
        out[pol.specimen_id(img, sp, profile)].append(f"{r.get('category', '')} outline {' and '.join(why) or 'flagged'}")
    return out


def load_hypothesis(patterns, profile):
    out = defaultdict(list)
    for pat in patterns or []:
        for f in glob.glob(pat):
            d = pd.read_csv(f)
            struct = Path(f).parent.name
            for _, r in d[d["status"] == "re-examine"].iterrows():
                sid = pol.specimen_id(str(r["specimen_id"]), str(r["assigned_species"]), profile)
                out[sid].append(f"{struct}: named {r.get('named_elsewhere_as', '')}")
    return out


# ── assemble ──────────────────────────────────────────────────────────────────────────────────────
def assemble(mv, key, outl, ann, hyp, key_independent=True, fold=2.0):
    """key_independent=False: the key named fewer specimens back than the matrix on this material, so a key
    disagreement is recorded but counts as evidence only when the matrix also disagrees."""
    rows = []
    for sid, r in mv.iterrows():
        sp = r["species"]; reasons, kinds = [], []
        if r["matrix_named"] != sp:
            kinds.append("matrix")
            reasons.append(f"matrix names it {r['matrix_named']} ({r.get('per_set', '')}; "
                           f"{r.get('sets_agreeing', 0)} of {r.get('sets_scored', 0)} sets agree with {sp})")
        kn = key.get(sid)
        if kn and kn not in ("unresolved", "nan", "None", sp):
            if key_independent or "matrix" in kinds:
                kinds.append("key"); reasons.append(f"key names it {kn}")
            else:
                reasons.append(f"(key names it {kn}; not counted: the key is less reliable than the matrix here)")
        vals = list(dict.fromkeys(v for v, _ in outl.get(sid, [])))   # the same value read from two images counts once
        big = sorted({v for v, r in outl.get(sid, []) if r >= fold or r <= 1 / fold})
        annot = []
        if big:
            annot.append(f"{len(big)} measurement(s) at least {fold:g}x off the {sp} median: " + "; ".join(big[:3])
                         + (" ..." if len(big) > 3 else ""))
        outs = list(dict.fromkeys(ann.get(sid, [])))
        if outs:
            annot.append(f"{len(outs)} outline(s) flagged: " + "; ".join(outs[:3]) + (" ..." if len(outs) > 3 else ""))
        if hyp.get(sid):
            kinds.append("hypothesis"); reasons.append("outside hypothesis test: " + "; ".join(hyp[sid][:3]))
        corro = [k for k in kinds if k in ("key", "hypothesis")]            # identity evidence only
        n_sc = 0 if pd.isna(r.get("sets_scored")) else int(r["sets_scored"])
        n_ag = 0 if pd.isna(r.get("sets_agreeing")) else int(r["sets_agreeing"])
        weak = r["matrix_named"] != sp and n_sc > 0 and n_ag < n_sc / 2
        if "matrix" in kinds and (corro or weak):
            identity = "re-examine"
        elif "matrix" in kinds or (key_independent and "key" in kinds):
            identity = "check"
        else:
            identity = "fits"
        status = identity if identity != "fits" else ("check annotation" if annot else "fits")
        rows.append({"specimen_id": sid, "species": sp, "status": status, "identity": identity,
                     "annotation": " | ".join(annot), "evidence": ", ".join(kinds),
                     "n_evidence": len(kinds), "reasons": " | ".join(reasons + ([f"annotation: {a_}" for a_ in annot])),
                     "matrix_named": r["matrix_named"], "matrix_own_rank": r["matrix_own_rank"],
                     "matrix_margin": r["matrix_margin"], "sets_agreeing": r.get("sets_agreeing"),
                     "sets_scored": r.get("sets_scored"), "per_set": r.get("per_set", ""), "key_named": key.get(sid, ""),
                     "n_outlying_values": len(big), "n_flagged_outlines": len(outs),
                     "hypothesis_flags": len(hyp.get(sid, []))})
    df = pd.DataFrame(rows)
    order = {"re-examine": 0, "check": 1, "check annotation": 2, "fits": 3}
    df["_o"] = df["status"].map(order)
    df = df.sort_values(["_o", "n_evidence", "sets_agreeing", "specimen_id"], ascending=[True, False, True, True]).drop(columns="_o")
    return df


def species_table(df):
    rows = []
    for sp, g in df.groupby("species"):
        conf = Counter(g.loc[g["matrix_named"] != sp, "matrix_named"])
        rows.append({"species": sp, "specimens": len(g), "named_back": int((g["matrix_named"] == sp).sum()),
                     "re_examine": int((g["status"] == "re-examine").sum()), "check": int((g["status"] == "check").sum()),
                     "check_annotation": int((g["annotation"] != "").sum()),
                     "confused_with": "; ".join(f"{k} ({v})" for k, v in conf.most_common())})
    return pd.DataFrame(rows).sort_values(["re_examine", "check", "species"], ascending=[False, False, True])


# ── layout for the map ────────────────────────────────────────────────────────────────────────────
def layout(score: pd.DataFrame, species_of: dict, seed=0):
    """species x species dissimilarity = mean (over a species' specimens) of their rank-score to the other species,
    symmetrised; classical MDS to 2-D. O(S^2) memory: fine for thousands of species."""
    sp = sorted(s for s in set(species_of.values()) | set(score.columns) if s in score.columns)
    rows = score.index.intersection(list(species_of))
    lab = pd.Series([species_of[i] for i in rows], index=rows)
    G = score.loc[rows, sp].groupby(lab.values).mean()          # one grouped mean, not one lookup per species
    M = G.reindex(index=sp).to_numpy(dtype=float)
    M = np.where(np.isnan(M), np.nanmean(M), M)
    D = (M + M.T) / 2.0
    np.fill_diagonal(D, 0.0)
    n = len(sp)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    X = V[:, idx] * np.sqrt(np.clip(w[idx], 1e-12, None))
    X = (X - X.mean(0)) / (np.abs(X).max() or 1.0)
    return sp, X, D


# ── the map ───────────────────────────────────────────────────────────────────────────────────────
HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Specimens to re-examine</title>
<style>
:root{--bg:#f6f8fa;--panel:#ffffff;--ink:#1f2328;--muted:#57606a;--line:#d0d7de;--fits:#0969da;--check:#bf8700;--re:#cf222e;--ann:#8250df;--edge:#8c959f}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--panel:#161b22;--ink:#e6edf3;--muted:#8d96a0;--line:#30363d;--fits:#4493f8;--check:#d29922;--re:#f85149;--ann:#a371f7;--edge:#6e7681}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink);display:flex;height:100vh;overflow:hidden}
#side{width:min(380px,40vw);border-right:1px solid var(--line);background:var(--panel);display:flex;flex-direction:column}
#side header{padding:12px 14px;border-bottom:1px solid var(--line)}h1{font-size:16px;margin:0 0 4px}
.sub{color:var(--muted);font-size:12px}#q{width:100%;margin-top:8px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
#ctrl{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;font-size:12px;color:var(--muted)}
#detail{overflow:auto;padding:10px 14px;flex:1}#detail h2{font-size:15px;margin:6px 0}
.sp{font-weight:600}.pill{display:inline-block;padding:0 6px;border-radius:9px;font-size:11px;color:#fff}
.re{background:var(--re)}.check{background:var(--check)}.fits{background:var(--fits)}.ann{background:var(--ann)}
.row{padding:6px 0;border-bottom:1px solid var(--line);cursor:pointer}.row:hover{background:var(--bg)}.why{color:var(--muted);font-size:12px}
#main{flex:1;position:relative}canvas{width:100%;height:100%;display:block;cursor:grab}
#legend{position:absolute;right:12px;bottom:12px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px}
#legend div{display:flex;align-items:center;gap:6px}.dot{width:10px;height:10px;border-radius:50%}
#tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:12px;display:none}
@media (max-width:700px){body{flex-direction:column}#side{width:100%;height:42vh;border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body>
<div id="side"><header><h1>Specimens to re-examine</h1><div class="sub" id="stats"></div>
<input id="q" placeholder="search species or specimen..." autocomplete="off">
<div id="ctrl"><label><input type="checkbox" id="onlyflag"> only species with flagged specimens</label><label><input type="checkbox" id="edges" checked> confusion links</label><label><input type="checkbox" id="nbr"> nearest neighbours</label></div></header>
<div id="detail"></div></div>
<div id="main"><canvas id="c"></canvas><div id="tip"></div>
<div id="legend"><div><span class="dot" style="background:var(--re)"></span>has specimens to re-examine</div><div><span class="dot" style="background:var(--check)"></span>has specimens to check</div><div><span class="dot" style="background:var(--fits)"></span>all specimens fit</div><div><span class="dot" style="background:var(--ann)"></span>specimen: annotation to check</div><div style="margin-top:4px;color:var(--muted)">scroll = zoom, drag = pan, click = details<br>zoom in to see specimens<br>numbers: species = named back / specimens;<br>specimen = character sets agreeing with its label</div></div></div>
<script>
const DATA=__DATA__;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const C=document.getElementById('c'),X=C.getContext('2d'),tip=document.getElementById('tip');
let W,H,dpr,view={x:0,y:0,k:1},sel=null,hover=null,query='';
const sp=DATA.species, byName={}; sp.forEach(s=>byName[s.id]=s);
const spec=DATA.specimens; const specBy={}; spec.forEach(p=>{(specBy[p.species]=specBy[p.species]||[]).push(p)});
document.getElementById('stats').textContent=`${DATA.n_species} species, ${DATA.n_specimens} specimens: ${DATA.n_re} to re-examine, ${DATA.n_check} to check (identity); ${DATA.n_ann} with an annotation to check. A flag is a reason to look again, never a determination.`;
function resize(){dpr=window.devicePixelRatio||1;W=C.clientWidth;H=C.clientHeight;C.width=W*dpr;C.height=H*dpr;X.setTransform(dpr,0,0,dpr,0,0);fit();draw()}
function fit(){view.k=Math.min(W,H)*0.42;view.x=W/2;view.y=H/2}
const sx=x=>view.x+x*view.k, sy=y=>view.y-y*view.k;
function colour(s){return s.re>0?css('--re'):s.check>0?css('--check'):css('--fits')}
function visible(s){if(document.getElementById('onlyflag').checked&&!(s.re||s.check))return false;return true}
function rad(s){const base=DATA.n_species>500?2.2:3.5;return Math.max(3,Math.min(18,base*Math.sqrt(s.n)))*Math.min(2.2,Math.max(1,view.k/400))}
function specPos(p){const s=byName[p.species];const list=specBy[p.species];const i=list.indexOf(p);const a=2*Math.PI*i/list.length;
  const r=(rad(s)+10)/view.k; let x=s.x+r*Math.cos(a), y=s.y+r*Math.sin(a);
  if(p.status!=='fits'&&byName[p.named]){const t=byName[p.named];x+= (t.x-s.x)*0.12;y+=(t.y-s.y)*0.12}
  return [x,y]}
function draw(){X.clearRect(0,0,W,H);X.fillStyle=css('--bg');X.fillRect(0,0,W,H);
  const showSpec=view.k>DATA.spec_zoom, showLab=view.k>DATA.label_zoom;
  if(document.getElementById('nbr').checked){X.strokeStyle=css('--line');X.lineWidth=1;
    for(const e of DATA.nbr){const a=byName[e[0]],b=byName[e[1]];if(!visible(a)||!visible(b))continue;X.beginPath();X.moveTo(sx(a.x),sy(a.y));X.lineTo(sx(b.x),sy(b.y));X.stroke()}}
  if(document.getElementById('edges').checked){
    for(const e of DATA.conf){const a=byName[e[0]],b=byName[e[1]];if(!visible(a)&&!visible(b))continue;
      X.strokeStyle=css('--edge');X.globalAlpha=0.75;X.lineWidth=Math.min(6,1+e[2]);X.beginPath();X.moveTo(sx(a.x),sy(a.y));X.lineTo(sx(b.x),sy(b.y));X.stroke();X.globalAlpha=1;
      if(view.k>DATA.label_zoom*2.5){X.fillStyle=css('--muted');X.font='11px system-ui';X.fillText(`${e[2]} confused`,(sx(a.x)+sx(b.x))/2+4,(sy(a.y)+sy(b.y))/2-4)}}}
  for(const s of sp){if(!visible(s))continue;const r=rad(s),x=sx(s.x),y=sy(s.y);if(x<-40||y<-40||x>W+40||y>H+40)continue;
    X.beginPath();X.arc(x,y,r,0,2*Math.PI);X.fillStyle=colour(s);X.globalAlpha=(query&&!s.match)?0.15:0.9;X.fill();X.globalAlpha=1;
    if(sel&&sel.type==='sp'&&sel.id===s.id){X.lineWidth=3;X.strokeStyle=css('--ink');X.stroke()}
    if(showLab||s.match||(sel&&sel.id===s.id)){X.fillStyle=css('--ink');X.font='12px system-ui';X.fillText(view.k>DATA.label_zoom*1.6?`${s.id} ${s.named_back}/${s.n}`:s.id,x+r+3,y+4)}}
  if(showSpec){for(const p of spec){const s=byName[p.species];if(!visible(s))continue;const [px,py]=specPos(p);const x=sx(px),y=sy(py);
      if(x<-10||y<-10||x>W+10||y>H+10)continue;X.beginPath();X.arc(x,y,p.status==='fits'?2.5:4,0,2*Math.PI);
      X.fillStyle=p.status==='re-examine'?css('--re'):p.status==='check'?css('--check'):p.status==='check annotation'?css('--ann'):css('--fits');X.fill();
      if(view.k>DATA.spec_zoom*1.8&&p.status==='re-examine'){X.fillStyle=css('--muted');X.font='10px system-ui';X.fillText(`${p.id} ${p.support||''}`,x+5,y+3)}}}
}
function pick(mx,my){let best=null,bd=1e9;
  if(view.k>DATA.spec_zoom){for(const p of spec){if(!visible(byName[p.species]))continue;const [px,py]=specPos(p);const d=Math.hypot(sx(px)-mx,sy(py)-my);if(d<7&&d<bd){bd=d;best={type:'spec',id:p.id,o:p}}}}
  for(const s of sp){if(!visible(s))continue;const d=Math.hypot(sx(s.x)-mx,sy(s.y)-my);if(d<rad(s)+3&&d<bd){bd=d;best={type:'sp',id:s.id,o:s}}}
  return best}
const esc=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function pill(st){return `<span class="pill ${st==='re-examine'?'re':st==='check annotation'?'ann':st}">${st}</span>`}
function showSpecies(s){const list=(specBy[s.id]||[]).slice().sort((a,b)=>({'re-examine':0,check:1,'check annotation':2,fits:3}[a.status]-{'re-examine':0,check:1,'check annotation':2,fits:3}[b.status]));
  document.getElementById('detail').innerHTML=`<h2>${esc(s.id)}</h2><div class="sub">${s.n} specimens; ${s.named_back} named back by the matrix; ${s.re} to re-examine, ${s.check} to check</div>`+
  (s.confused?`<div class="why" style="margin:6px 0">confused with: ${esc(s.confused)}</div>`:'')+
  list.map(p=>`<div class="row" data-id="${esc(p.id)}"><span class="sp">${esc(p.id)}</span> ${pill(p.status)}<div class="why">${esc(p.why||'no evidence against its label')}</div></div>`).join('');
  document.querySelectorAll('#detail .row').forEach(r=>r.onclick=()=>{const p=spec.find(q=>q.id===r.dataset.id);showSpecimen(p)})}
function showSpecimen(p){sel={type:'spec',id:p.id};document.getElementById('detail').innerHTML=`<h2>${esc(p.id)}</h2>${pill(p.status)}<div class="sub" style="margin-top:6px">filed as <b>${esc(p.species)}</b>; matrix names it <b>${esc(p.named)}</b>${p.key?`; key: ${esc(p.key)}`:''}</div><div class="sub">character sets: ${esc(p.per_set||'-')}</div><div class="why" style="margin-top:8px">${esc(p.why||'no evidence against its label')}</div><div class="row" id="back">&larr; ${esc(p.species)}</div>`;
  document.getElementById('back').onclick=()=>{sel={type:'sp',id:p.species};showSpecies(byName[p.species]);draw()};draw()}
function overview(){const top=spec.filter(p=>p.status==='re-examine');document.getElementById('detail').innerHTML=`<h2>To re-examine (${top.length})</h2>`+
  top.map(p=>`<div class="row" data-id="${esc(p.id)}"><span class="sp">${esc(p.id)}</span> <span class="sub">${esc(p.species)} &rarr; ${esc(p.named)}</span><div class="why">${esc(p.why)}</div></div>`).join('');
  document.querySelectorAll('#detail .row').forEach(r=>r.onclick=()=>{const p=spec.find(q=>q.id===r.dataset.id);focus(byName[p.species],2.2);showSpecimen(p)})}
function focus(s,zf){view.k=Math.max(view.k,DATA.spec_zoom*zf);view.x=W/2-s.x*view.k;view.y=H/2+s.y*view.k;draw()}
let drag=null;C.onmousedown=e=>{drag={x:e.offsetX,y:e.offsetY,vx:view.x,vy:view.y,moved:false};C.style.cursor='grabbing'};
window.onmouseup=e=>{if(drag&&!drag.moved){const h=pick(e.offsetX,e.offsetY);if(h){sel=h;h.type==='sp'?showSpecies(h.o):showSpecimen(h.o)}else{sel=null;overview()}draw()}drag=null;C.style.cursor='grab'};
C.onmousemove=e=>{if(drag){const dx=e.offsetX-drag.x,dy=e.offsetY-drag.y;if(Math.abs(dx)+Math.abs(dy)>3)drag.moved=true;view.x=drag.vx+dx;view.y=drag.vy+dy;draw();return}
  const h=pick(e.offsetX,e.offsetY);if(h){tip.style.display='block';tip.style.left=(e.offsetX+12)+'px';tip.style.top=(e.offsetY+12)+'px';
  tip.textContent=h.type==='sp'?`${h.o.id}: ${h.o.named_back}/${h.o.n} named back, ${h.o.re} re-examine, ${h.o.check} check`:`${h.o.id} (${h.o.status}) -> ${h.o.named}; sets agreeing ${h.o.support||'-'}`}else tip.style.display='none'};
C.onwheel=e=>{e.preventDefault();const f=Math.exp(-e.deltaY*0.0015);const mx=e.offsetX,my=e.offsetY;view.x=mx-(mx-view.x)*f;view.y=my-(my-view.y)*f;view.k*=f;draw()};
document.getElementById('q').oninput=e=>{query=e.target.value.trim().toLowerCase();sp.forEach(s=>s.match=!!query&&(s.id.toLowerCase().includes(query)||(specBy[s.id]||[]).some(p=>p.id.toLowerCase().includes(query))));
  const hits=sp.filter(s=>s.match);if(hits.length===1){focus(hits[0],1.5);showSpecies(hits[0])}draw()};
['onlyflag','edges','nbr'].forEach(i=>document.getElementById(i).onchange=draw);
window.onresize=resize;resize();overview();
// deep link: reexamine_map.html#<species or specimen>
function fromHash(){const h=decodeURIComponent(location.hash.slice(1));if(!h)return;const s=byName[h];if(s){focus(s,2.2);sel={type:'sp',id:s.id};showSpecies(s);draw();return}const p=spec.find(q=>q.id===h);if(p){focus(byName[p.species],2.2);showSpecimen(p)}}
window.onhashchange=fromHash;fromHash();
</script></body></html>"""


def write_map(path, df, sp_tab, sp_ids, X, D, k_nbr=2):
    pos = {s: X[i] for i, s in enumerate(sp_ids)}
    st = sp_tab.set_index("species")
    species = []
    for s in sp_ids:
        if s not in st.index:
            continue
        r = st.loc[s]
        species.append({"id": s, "x": round(float(pos[s][0]), 5), "y": round(float(pos[s][1]), 5), "n": int(r["specimens"]),
                        "named_back": int(r["named_back"]), "re": int(r["re_examine"]), "check": int(r["check"]),
                        "confused": r["confused_with"]})
    conf = Counter()
    for _, r in df.iterrows():
        if r["matrix_named"] != r["species"] and r["matrix_named"] in pos and r["species"] in pos:
            conf[tuple(sorted((r["species"], r["matrix_named"])))] += 1
    nbr = set()
    for i, s in enumerate(sp_ids):
        for j in np.argsort(D[i])[1:k_nbr + 1]:
            nbr.add(tuple(sorted((s, sp_ids[j]))))
    specimens = [{"id": r["specimen_id"], "species": r["species"], "status": r["status"], "named": r["matrix_named"],
                  "support": (f"{int(r['sets_agreeing'])}/{int(r['sets_scored'])}" if pd.notna(r.get("sets_scored")) else ""),
                  "key": r["key_named"] if r["key_named"] not in ("", "nan") else "", "per_set": r["per_set"], "why": r["reasons"]}
                 for _, r in df.iterrows()]
    n = len(species)
    data = {"species": species, "specimens": specimens, "conf": [[a, b, c] for (a, b), c in conf.items()],
            "nbr": [list(e) for e in nbr], "n_species": n, "n_specimens": len(specimens),
            "n_re": int((df["status"] == "re-examine").sum()), "n_check": int((df["status"] == "check").sum()),
            "n_ann": int((df["annotation"] != "").sum()),
            # zoom levels at which labels and specimens appear: later for larger reference sets
            "label_zoom": 250 if n <= 60 else 250 * math.sqrt(n / 60),
            "spec_zoom": 600 if n <= 60 else 600 * math.sqrt(n / 60)}
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    Path(path).write_text(HTML.replace("__DATA__", blob), encoding="utf-8")


def write_report(path, df, sp_tab, inputs):
    re_ = df[df["status"] == "re-examine"]; ch = df[df["status"] == "check"]
    L = ["# Specimens to re-examine", "",
         f"{len(df)} specimens of {df['species'].nunique()} species: **{len(re_)} to re-examine**, {len(ch)} to check, "
         f"{(df['status'] == 'fits').sum()} fit their label.", "",
         "A flag is a reason to look at the specimen again, never a determination. *re-examine* = the character matrix "
         "names it as another species and another kind of evidence agrees, or most character sets disagree with its "
         "label; *check* = the matrix alone names it elsewhere. The key and an outside hypothesis only corroborate a "
         "matrix flag. Annotation problems (a flagged outline, or a value at least twice or at most half its species' "
         "median) are listed separately: they are about the outline or the measurement, not the name.", "",
         "Evidence used: " + ", ".join(f"{k} ({v})" for k, v in inputs.items()), "",
         "## To re-examine", "", "| specimen | filed as | matrix names it | sets agreeing | evidence | why |", "|---|---|---|---|---|---|"]
    for _, r in re_.iterrows():
        L.append(f"| {r['specimen_id']} | {r['species']} | {r['matrix_named']} | {r['sets_agreeing']}/{r['sets_scored']} | "
                 f"{r['evidence']} | {r['reasons'].replace('|', '/')} |")
    an = df[df["annotation"] != ""]
    L += ["", f"## Annotation to check ({len(an)}): an outline or a measurement may be wrong (not a question of identity)", "",
          "| specimen | species | identity | what |", "|---|---|---|---|"]
    for _, r in an.iterrows():
        L.append(f"| {r['specimen_id']} | {r['species']} | {r['identity']} | {r['annotation'].replace('|', '/')} |")
    L += ["", "## To check (identity)", "", "| specimen | filed as | evidence | why |", "|---|---|---|---|"]
    for _, r in ch.iterrows():
        L.append(f"| {r['specimen_id']} | {r['species']} | {r['evidence']} | {r['reasons'].replace('|', '/')} |")
    L += ["", "## Species", "", "| species | specimens | named back | re-examine | check | confused with |", "|---|---|---|---|---|---|"]
    for _, r in sp_tab.iterrows():
        L.append(f"| {r['species']} | {r['specimens']} | {r['named_back']} | {r['re_examine']} | {r['check']} | {r['confused_with']} |")
    L += ["", "Open `reexamine_map.html` in a browser for the zoomable map (no internet needed)."]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration_dir", required=True, help="output of biorag_calibrate_v1.py (matrix_identification_distances.tsv)")
    ap.add_argument("--key_loo", default=None, help="key identification_test.tsv (leave one specimen out)")
    ap.add_argument("--outlier_flags", default=None, help="compiled_key_tier/outlier_flags.tsv")
    ap.add_argument("--annotation_screen", default=None, help="annotation_screen/annotation_screen.tsv")
    ap.add_argument("--hypothesis_flags", nargs="*", default=[], help="glob(s) of descriptron_trait_stats specimen_flags.csv "
                    "from a run with an OUTSIDE species hypothesis (DNA, field notes ...)")
    ap.add_argument("--group_labels", default=None, help="filename,group_label (maps flagged images to specimens)")
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--sets", nargs="*", default=None, help="character sets to use (default: all in the distances file)")
    ap.add_argument("--fold", type=float, default=2.0,
                    help="annotation check: a measurement at least this many times, or at most 1/this, of its species' median")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    profile = pol.load_taxon_profile(a.taxon_profile)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    dist = pd.read_csv(Path(a.calibration_dir) / "matrix_identification_distances.tsv", sep="\t")
    image_species = {}
    if a.group_labels and Path(a.group_labels).exists():
        g = pd.read_csv(a.group_labels)
        image_species = dict(zip(g.iloc[:, 0].astype(str), g.iloc[:, 1].astype(str).str.strip()))
    mv, score = matrix_verdict(dist, a.sets)
    key, key_acc = load_key(a.key_loo)
    mat_acc = float((mv["matrix_named"] == mv["species"]).mean())
    key_independent = key_acc is not None and key_acc >= mat_acc
    outl = load_outliers(a.outlier_flags, profile, image_species)
    ann = load_annotation_screen(a.annotation_screen, profile, image_species)
    hyp = load_hypothesis(a.hypothesis_flags, profile)
    df = assemble(mv, key, outl, ann, hyp, key_independent, a.fold)
    sp_tab = species_table(df)
    df.to_csv(out / "specimens_to_reexamine.tsv", sep="\t", index=False)
    sp_tab.to_csv(out / "species_summary.tsv", sep="\t", index=False)
    inputs = {"matrix": f"{len(mv)} specimens, sets {sorted(dist['set'].unique().tolist())}",
              "key": (f"{len(key)} specimens; names {key_acc:.0%} back vs matrix {mat_acc:.0%} -> "
                      + ("counted on its own" if key_independent else "counted only where the matrix also disagrees"))
                     if key else "not given",
              "values": f"{sum(len(v) for v in outl.values())} outlying values" if a.outlier_flags else "not given",
              "outlines": f"{sum(len(v) for v in ann.values())} flagged outlines" if a.annotation_screen else "not given",
              "hypothesis": f"{sum(len(v) for v in hyp.values())} flags" if a.hypothesis_flags else "not given"}
    write_report(out / "reexamine_report.md", df, sp_tab, inputs)
    sp_ids, X, D = layout(score, dict(zip(mv.index, mv["species"])))
    write_map(out / "reexamine_map.html", df, sp_tab, sp_ids, X, D)
    summ = {"version": VERSION, "specimens": len(df), "species": int(df["species"].nunique()),
            "re_examine": int((df["status"] == "re-examine").sum()), "check": int((df["status"] == "check").sum()),
            "fits": int((df["status"] == "fits").sum()), "annotation_to_check": int((df["annotation"] != "").sum()),
            "matrix_named_back": int((mv["matrix_named"] == mv["species"]).sum()),
            "inputs": inputs}
    json.dump(summ, open(out / "reexamine_summary.json", "w"), indent=2)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()

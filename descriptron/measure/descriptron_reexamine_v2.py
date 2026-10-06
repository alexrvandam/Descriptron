#!/usr/bin/env python3
"""
descriptron_reexamine_v2.py - which specimens should the taxonomist look at again, and why (v2: a better map)
==============================================================================================================

v2 = v1 with the SAME analysis (identity/annotation rule, specimens_to_reexamine.tsv, species_summary.tsv, report,
summary numbers) and a new reexamine_map.html:
  - a line from each specimen the matrix names as another species to that species; width/opacity = how decisive
    the call was (how much further its own species ranked than the named one); key calls as a second layer
  - each species linked to its --k_nbr (default 3) nearest species in the matrix distance used for the layout
    (distance on hover)
  - evidence layers switched by checkboxes: matrix calls, key calls, DNA (with --dna_table), annotation flags,
    nearest species; status filter (re-examine / check / check annotation / fits); search box with a hit list
  - click a specimen: card with id, label, matrix name and support, reasons, DNA verdict
  - layout: classical MDS as in v1, then a light repulsion pass so species (each with its specimens on a ring) do
    not overlap; labels placed greedily so they do not pile up; works at phone width (touch: drag + pinch)
  - optional --dna_table (dna_vs_morphology_v1 specimens_dna_vs_matrix.tsv) and --barcode_gap (default: the
    barcode_gap.tsv beside the DNA table). DNA is shown on the map only: it never changes a status.

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
      --group_labels group_labels.csv --taxon_profile profile.yaml --out_dir M/reexamine \
      [--dna_table DNA/specimens_dna_vs_matrix.tsv]
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

VERSION = "1.0"          # the analysis: identical to descriptron_reexamine_v1
MAP_VERSION = "2.0"


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


# ── v2: optional DNA layer ────────────────────────────────────────────────────────────────────────
def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else round(v + 0.0, 4)          # + 0.0 turns -0.0 into 0.0


def _truth(v):
    t = str(v).strip().lower()
    return True if t == "true" else False if t == "false" else None


def load_dna(path, gap_path=None):
    """specimens_dna_vs_matrix.tsv (dna_vs_morphology_v1) -> {specimen_id: [sequence verdicts]} and, from
    barcode_gap.tsv (default: beside it), {species: gap row}. Shown on the map only; never changes a status."""
    if not path or not Path(path).exists():
        return {}, {}
    d = pd.read_csv(path, sep="\t")
    per = defaultdict(list)
    for _, r in d.iterrows():
        if pd.isna(r.get("specimen_id")) or str(r["specimen_id"]).strip() in ("", "nan"):
            continue                                      # a sequence not tied to a specimen cannot be placed
        per[str(r["specimen_id"])].append({"seq": str(r.get("sequence", "")), "species": str(r.get("species", "")),
                                           "nearest": "" if pd.isna(r.get("dna_nearest")) else str(r.get("dna_nearest")),
                                           "k2p": _num(r.get("dna_nearest_k2p")), "agrees": _truth(r.get("dna_agrees_with_label"))})
    gaps = {}
    gp = Path(gap_path) if gap_path else Path(path).parent / "barcode_gap.tsv"
    if gp.exists():
        g = pd.read_csv(gp, sep="\t")
        for _, r in g.iterrows():
            gaps[str(r["species"])] = {"n": None if pd.isna(r.get("sequences")) else int(r["sequences"]),
                                       "max_intra": _num(r.get("max_intra")), "min_inter": _num(r.get("min_inter")),
                                       "nearest": "" if pd.isna(r.get("nearest_species")) else str(r["nearest_species"]),
                                       "gap": _num(r.get("gap"))}
    return dict(per), gaps


def dna_verdict(seqs):
    """one verdict per specimen over its sequences: disagrees if any sequence's nearest is another species"""
    a = [q["agrees"] for q in seqs if q["agrees"] is not None]
    if not a:
        return "no verdict"
    return "agrees" if all(a) else "disagrees"


# ── v2: layout on top of v1's MDS ──────────────────────────────────────────────────────────────────
def declutter(X, n_spec, fill=None, pad=0.45, iters=None, max_species=20000):
    """give each species a disc (radius ~ sqrt(specimens)) sized so the discs fill ~`fill` of the MDS extent, then
    push overlapping discs apart while a weak spring holds every species near its MDS position. Only pairs within
    reach are visited (KD-tree), so thousands of species take seconds. Returns the new positions (rescaled to
    [-1, 1] like v1) and the disc radii in the same units. Above max_species only the radii are returned."""
    P = np.asarray(X, dtype=float).copy()
    n = np.maximum(np.asarray(n_spec, dtype=float), 1.0)
    S = len(P)
    fill = fill if fill is not None else 0.22 * min(1.0, math.sqrt(300.0 / max(S, 1)))   # sparser for big sets
    iters = iters if iters is not None else min(2500, 400 + S // 2)
    ext = np.ptp(P, axis=0) if S > 1 else np.array([1.0, 1.0])
    area = max(float(ext[0] * ext[1]), 0.25)
    k = math.sqrt(fill * area / (math.pi * n.sum()))
    r = k * np.sqrt(n)
    if S < 2 or S > max_species:
        return P, r
    try:
        from scipy.spatial import cKDTree
    except ImportError:                                  # dense fallback, fine for a few hundred species
        cKDTree = None
    X0 = P.copy()
    gap = pad * float(np.median(r))
    reach = 2 * float(r.max()) + gap
    P += np.random.default_rng(0).normal(scale=1e-4, size=P.shape)   # separates species MDS put on one point
    for it in range(iters):
        if cKDTree is not None:
            pr = cKDTree(P).query_pairs(reach, output_type="ndarray")
        else:
            iu = np.triu_indices(S, 1)
            pr = np.c_[iu]
        if len(pr) == 0:
            break
        i, j = pr[:, 0], pr[:, 1]
        diff = P[i] - P[j]
        dist = np.maximum(np.sqrt((diff ** 2).sum(1)), 1e-9)
        over = np.clip(r[i] + r[j] + gap - dist, 0.0, None)
        if over.max() < 1e-4 * k:
            break
        f = diff / dist[:, None] * (over * 0.5)[:, None]
        push = np.zeros_like(P)
        np.add.at(push, i, f)
        np.add.at(push, j, -f)
        P += 0.5 * push + 0.01 * max(0.0, 1.0 - it / (0.75 * iters)) * (X0 - P)   # spring fades out at the end
    c = P.mean(0)
    s = float(np.abs(P - c).max()) or 1.0
    return (P - c) / s, r / s


def specimen_positions(df, sp_ids, P, r):
    """specimens on a ring around their species (a sunflower disc above 18); the specimens the matrix names
    elsewhere take the slots facing the species they are named as."""
    idx = {s: i for i, s in enumerate(sp_ids)}
    out = {}
    for sp, g in df.groupby("species", sort=True):
        if sp not in idx:
            continue
        c, rr, m = P[idx[sp]], r[idx[sp]], len(g)
        if m <= 18:
            ang = -math.pi / 2 + 2 * math.pi * np.arange(m) / m
            slots = np.c_[np.cos(ang), np.sin(ang)] * rr
        else:
            kk = np.arange(m)[::-1]                       # outer slots first
            rad = rr * np.sqrt((kk + 0.5) / m)
            ang = kk * math.pi * (3 - math.sqrt(5))
            slots = np.c_[np.cos(ang), np.sin(ang)] * rad[:, None]
        free = list(range(m))
        rows = sorted(g.itertuples(index=False), key=lambda t: (t.matrix_named == sp or t.matrix_named not in idx,
                                                                 str(t.specimen_id)))
        for t in rows:
            if t.matrix_named != sp and t.matrix_named in idx:
                v = P[idx[t.matrix_named]] - c
                want = math.atan2(v[1], v[0])
                j = min(free, key=lambda f: abs((math.atan2(slots[f][1], slots[f][0]) - want + math.pi) % (2 * math.pi) - math.pi))
            else:
                j = free[0]
            free.remove(j)
            out[t.specimen_id] = (float(c[0] + slots[j][0]), float(c[1] + slots[j][1]))
    return out


def call_strength(score, sid, own, named):
    """how decisive a 'named as another species' call is: the specimen's mean rank-score (0 = nearest, 1 =
    furthest, as in matrix_verdict) to its own species minus that to the named species, in [0, 1]"""
    try:
        u_own, u_named = float(score.at[sid, own]), float(score.at[sid, named])
    except (KeyError, ValueError):
        return None
    if u_own != u_own or u_named != u_named:
        return None
    return round(max(0.0, min(1.0, u_own - u_named)), 3)


HTML_V2 = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Specimens to re-examine</title>
<style>
:root{--bg:#f6f8fa;--panel:#ffffff;--ink:#1f2328;--muted:#57606a;--line:#d0d7de;--fits:#0969da;--check:#bf8700;--re:#cf222e;--ann:#8250df;--key:#1b7c83;--dnaok:#1a7f37;--dnabad:#bc4c00;--halo:rgba(9,105,218,.07);--nbr:#8c959f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d1117;--panel:#161b22;--ink:#e6edf3;--muted:#8d96a0;--line:#30363d;--fits:#4493f8;--check:#d29922;--re:#f85149;--ann:#a371f7;--key:#39c5cf;--dnaok:#3fb950;--dnabad:#f0883e;--halo:rgba(68,147,248,.09);--nbr:#6e7681}}
:root[data-theme="dark"]{--bg:#0d1117;--panel:#161b22;--ink:#e6edf3;--muted:#8d96a0;--line:#30363d;--fits:#4493f8;--check:#d29922;--re:#f85149;--ann:#a371f7;--key:#39c5cf;--dnaok:#3fb950;--dnabad:#f0883e;--halo:rgba(68,147,248,.09);--nbr:#6e7681}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink);display:flex;height:100vh;height:100dvh;overflow:hidden}
#side{width:min(390px,38vw);border-right:1px solid var(--line);background:var(--panel);display:flex;flex-direction:column;min-height:0}
#side header{padding:12px 14px 8px;border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0 0 4px}.sub{color:var(--muted);font-size:12px}
#qwrap{position:relative;margin-top:8px}
#q{width:100%;padding:7px 9px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);font:inherit}
#hits{position:absolute;left:0;right:0;top:100%;z-index:5;background:var(--panel);border:1px solid var(--line);border-radius:6px;margin-top:2px;max-height:260px;overflow:auto;display:none;box-shadow:0 4px 14px rgba(0,0,0,.15)}
#hits div{padding:5px 9px;cursor:pointer;font-size:13px}#hits div:hover,#hits div.on{background:var(--bg)}
details{margin-top:8px}summary{cursor:pointer;font-size:12px;color:var(--muted)}
.grp{display:flex;flex-wrap:wrap;gap:4px 12px;margin-top:6px;font-size:12px}
.grp b{width:100%;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:600}
.grp label{display:inline-flex;align-items:center;gap:4px;cursor:pointer;white-space:nowrap}
.grp label.off{opacity:.45;cursor:not-allowed}
#detail{overflow:auto;padding:10px 14px 18px;flex:1;min-height:0}#detail h2{font-size:15px;margin:6px 0 4px;word-break:break-all}
.pill{display:inline-block;padding:0 7px;border-radius:9px;font-size:11px;color:#fff;vertical-align:1px}
.re{background:var(--re)}.check{background:var(--check)}.fits{background:var(--fits)}.ann{background:var(--ann)}
.row{padding:6px 2px;border-bottom:1px solid var(--line);cursor:pointer}.row:hover{background:var(--bg)}
.why{color:var(--muted);font-size:12px}.sp{font-weight:600}
a.go{color:var(--fits);cursor:pointer;text-decoration:underline;text-underline-offset:2px}
table.kv{border-collapse:collapse;margin:6px 0;font-size:13px;width:100%}table.kv td{padding:3px 6px 3px 0;vertical-align:top;border-bottom:1px solid var(--line)}
table.kv td:first-child{color:var(--muted);white-space:nowrap;width:1%}
ul.reasons{margin:4px 0 0;padding-left:18px;font-size:13px}ul.reasons li{margin:3px 0}
.box{border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin:8px 0;font-size:13px}
.ok{color:var(--dnaok);font-weight:600}.bad{color:var(--dnabad);font-weight:600}
#main{flex:1;position:relative;min-width:0;min-height:0}
canvas{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
#tools{position:absolute;left:10px;top:10px;display:flex;gap:6px}
#tools button{font:12px system-ui,sans-serif;padding:5px 9px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);cursor:pointer}
#legend{position:absolute;right:10px;bottom:10px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:11.5px;line-height:1.3;max-width:250px;opacity:.96;max-height:calc(100% - 60px);overflow:auto}
#legend.hide{display:none}
#legend .li{display:flex;align-items:center;gap:7px;margin:2px 0}#legend canvas{width:30px;height:14px;flex:none;cursor:default}
#legend h3{font-size:11px;margin:6px 0 2px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
#tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:12px;display:none;max-width:320px;z-index:4}
@media (max-width:700px){body{flex-direction:column}#main{order:-1;flex:none;height:55vh;height:55dvh;border-bottom:1px solid var(--line)}
#side{width:100%;flex:1;border-right:0}#legend{max-width:calc(100% - 20px);font-size:11px}h1{font-size:15px}}
</style></head><body>
<div id="side"><header><h1>Specimens to re-examine</h1><div class="sub" id="stats"></div>
<div id="qwrap"><input id="q" placeholder="search species or specimen id..." autocomplete="off" aria-label="search species or specimen"><div id="hits"></div></div>
<details id="opts" open><summary>filters and evidence layers</summary>
<div class="grp" id="statusgrp"><b>show specimens</b></div>
<div class="grp" id="layergrp"><b>evidence layers</b>
<label><input type="checkbox" id="L_matrix" checked> matrix calls</label>
<label><input type="checkbox" id="L_key"> key calls</label>
<label id="L_dna_l"><input type="checkbox" id="L_dna" checked> DNA (COI)</label>
<label><input type="checkbox" id="L_ann" checked> annotation flags</label>
<label><input type="checkbox" id="L_nbr" checked> nearest species</label></div>
</details></header>
<div id="detail"></div></div>
<div id="main"><canvas id="c" aria-label="map of species and specimens"></canvas><div id="tip"></div>
<div id="tools"><button id="fitb" title="show everything">fit</button><button id="legb">legend</button></div>
<div id="legend"></div></div>
<script>
const DATA=__DATA__;
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let COL={};function readColours(){for(const k of ['bg','panel','ink','muted','line','fits','check','re','ann','key','dnaok','dnabad','halo','nbr'])COL[k]=css('--'+k)}
readColours();
const C=$('c'),X=C.getContext('2d'),tip=$('tip');
let W=0,H=0,view={x:0,y:0,k:1},sel=null,query='',HL=[],HP=[];
const sp=DATA.species,byName={};sp.forEach(s=>byName[s.id]=s);
const spec=DATA.specimens,specById={},specBy={};spec.forEach(p=>{specById[p.id]=p;(specBy[p.species]=specBy[p.species]||[]).push(p)});
const nbrOf={};DATA.nbr.forEach(e=>{(nbrOf[e[0]]=nbrOf[e[0]]||[]).push([e[1],e[2]]);(nbrOf[e[1]]=nbrOf[e[1]]||[]).push([e[0],e[2]])});
Object.values(nbrOf).forEach(l=>l.sort((a,b)=>a[1]-b[1]));
const ORDER={'re-examine':0,'check':1,'check annotation':2,'fits':3};
const STATUS=[['re-examine','re',DATA.n_re],['check','check',DATA.n_check],['check annotation','ann',DATA.n_check_ann],['fits','fits',DATA.n_fits]];
const SCOL=st=>st==='re-examine'?COL.re:st==='check'?COL.check:st==='check annotation'?COL.ann:COL.fits;
const esc=t=>String(t==null?'':t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pill=st=>`<span class="pill ${st==='re-examine'?'re':st==='check annotation'?'ann':st}">${st}</span>`;
// controls
$('statusgrp').insertAdjacentHTML('beforeend',STATUS.map(([s,c,n])=>`<label><input type="checkbox" data-status="${s}" checked> <span class="pill ${c}">${s}</span> ${n}</label>`).join(''));
if(!DATA.has_dna){$('L_dna').checked=false;$('L_dna').disabled=true;$('L_dna_l').classList.add('off');$('L_dna_l').title='no --dna_table given'}
let ST={};const readST=()=>document.querySelectorAll('[data-status]').forEach(c=>ST[c.dataset.status]=c.checked);readST();
const on=id=>{const e=$(id);return e.checked&&!e.disabled};
$('stats').textContent=`${DATA.n_species} species, ${DATA.n_specimens} specimens: ${DATA.n_re} to re-examine, ${DATA.n_check} to check (identity); ${DATA.n_ann} with an annotation to check`+(DATA.has_dna?`; COI for ${DATA.n_dna} specimens (${DATA.n_dna_bad} nearest to another species)`:'')+'. A flag is a reason to look again, never a determination.';
// geometry
const sx=x=>view.x+x*view.k,sy=y=>view.y-y*view.k;
const shown=p=>ST[p.status];
const spActive=s=>(specBy[s.id]||[]).some(shown);
const spCol=s=>s.re>0?COL.re:s.check>0?COL.check:COL.fits;
const discR=s=>Math.max(4,Math.min(26,s.r*0.42*view.k));
const showSpec=s=>s.r*view.k>=7;
const dotR=p=>(p.status==='fits'?3.2:4.4)*Math.min(1.7,Math.max(1,view.k/700));
function bounds(){let a=1e9,b=-1e9,c=1e9,d=-1e9;for(const s of sp){a=Math.min(a,s.x-s.r);b=Math.max(b,s.x+s.r);c=Math.min(c,s.y-s.r);d=Math.max(d,s.y+s.r)}return [a,b,c,d]}
function fit(){const [a,b,c,d]=bounds();const m=34,lg=$('legend');const rm=(!lg.classList.contains('hide')&&W>760)?lg.offsetWidth+12:0;const Wf=W-rm;view.k=Math.min((Wf-2*m)/Math.max(b-a,1e-3),(H-2*m-30)/Math.max(d-c,1e-3));view.x=Wf/2-(a+b)/2*view.k;view.y=H/2+15+(c+d)/2*view.k}
function resize(){const dpr=window.devicePixelRatio||1;W=C.clientWidth;H=C.clientHeight;C.width=W*dpr;C.height=H*dpr;X.setTransform(dpr,0,0,dpr,0,0);fit();draw()}
function line(x1,y1,x2,y2,col,w,alpha,dash,arrow,shorten){const dx=x2-x1,dy=y2-y1,L=Math.hypot(dx,dy);if(L<2)return;
  const ux=dx/L,uy=dy/L,ex=x2-ux*(shorten||0),ey=y2-uy*(shorten||0);
  X.save();X.globalAlpha=alpha;X.strokeStyle=col;X.fillStyle=col;X.lineWidth=w;X.setLineDash(dash||[]);X.beginPath();X.moveTo(x1,y1);X.lineTo(ex,ey);X.stroke();
  if(arrow){X.setLineDash([]);const h=5+w*1.6;X.beginPath();X.moveTo(ex,ey);X.lineTo(ex-ux*h-uy*h*0.55,ey-uy*h+ux*h*0.55);X.lineTo(ex-ux*h+uy*h*0.55,ey-uy*h-ux*h*0.55);X.closePath();X.fill()}
  X.restore()}
function dim(on_){return on_?1:0.18}
function matchP(p){return !query||p.id.toLowerCase().includes(query)||p.species.toLowerCase().includes(query)}
function matchS(s){return !query||s.id.toLowerCase().includes(query)||(specBy[s.id]||[]).some(p=>p.id.toLowerCase().includes(query))}
// labels: greedy, highest priority first, four candidate spots, skipped when they would overlap
function placeLabels(list){list.sort((a,b)=>b.prio-a.prio);const placed=[];
  for(const L of list){X.font=L.font;const w=X.measureText(L.text).width,h=L.size;
    const c=(L.inner!=null?[[L.x-w/2,L.y+L.inner+h+1],[L.x-w/2,L.y-L.inner-4]]:[]).concat([[L.x+L.r+3,L.y+h*0.35],[L.x-L.r-3-w,L.y+h*0.35],[L.x-w/2,L.y-L.r-4],[L.x-w/2,L.y+L.r+h]]);
    for(const [tx,ty] of c){const R=[tx-1,ty-h,tx+w+1,ty+3];if(R[0]<0||R[2]>W||R[1]<0||R[3]>H)continue;
      if(placed.some(q=>!(R[2]<q[0]||R[0]>q[2]||R[3]<q[1]||R[1]>q[3]))&&!L.force)continue;
      placed.push(R);X.save();X.globalAlpha=L.alpha;X.lineJoin='round';X.lineWidth=3;X.strokeStyle=COL.bg;X.strokeText(L.text,tx,ty);X.fillStyle=L.col;X.fillText(L.text,tx,ty);X.restore();break}}}
function draw(){X.clearRect(0,0,W,H);X.fillStyle=COL.bg;X.fillRect(0,0,W,H);HL=[];HP=[];const labels=[];
  const selSp=sel&&sel.type==='sp'?sel.id:null,selP=sel&&sel.type==='spec'?specById[sel.id]:null;
  // species halos and rings
  for(const s of sp){const x=sx(s.x),y=sy(s.y),R=s.r*view.k;if(x<-R-40||y<-R-40||x>W+R+40||y>H+R+40)continue;
    X.globalAlpha=dim(spActive(s)&&matchS(s));X.beginPath();X.arc(x,y,R+7,0,2*Math.PI);X.fillStyle=COL.halo;X.fill();X.globalAlpha=1}
  // nearest species
  if(on('L_nbr')){for(const e of DATA.nbr){const a=byName[e[0]],b=byName[e[1]];const hi=selSp&&(e[0]===selSp||e[1]===selSp);
      if(selSp&&!hi&&sp.length>60)continue;if(!hi&&sp.length>200&&view.k<DATA.label_zoom)continue;const x1=sx(a.x),y1=sy(a.y),x2=sx(b.x),y2=sy(b.y);
      line(x1,y1,x2,y2,hi?COL.ink:COL.nbr,hi?1.6:1,hi?0.9:0.55*Math.min(dim(spActive(a)&&matchS(a)),dim(spActive(b)&&matchS(b)))*(selSp?0.6:1),[4,4]);
      HL.push({x1,y1,x2,y2,t:`nearest species: ${e[0]} - ${e[1]}, matrix distance ${e[2].toFixed(3)} (rank ${e[3]} of the nearer)`});
      if(hi||view.k>DATA.nbr_label_zoom)labels.push({text:e[2].toFixed(2),x:(x1+x2)/2,y:(y1+y2)/2,r:0,font:'10px system-ui',size:10,col:COL.muted,prio:hi?50:1,alpha:1})}}
  // evidence lines from specimens
  for(const p of spec){const s=byName[p.species];if(!s)continue;const isSel=selP===p;if(!shown(p)&&!isSel)continue;if(!showSpec(s)&&!isSel&&!(selSp===p.species))continue;
    const x=sx(p.x),y=sy(p.y),a=isSel?1:dim(matchP(p));
    if(p.key&&byName[p.key]&&p.key!==p.species&&(on('L_key')||isSel)){const t=byName[p.key];const x2=sx(t.x),y2=sy(t.y);
      line(x,y,x2,y2,COL.key,isSel?2.2:1.4,(isSel||p.status==='re-examine'||p.status==='check'?0.85:0.4)*a,[6,4],true,discR(t)+2);HL.push({x1:x,y1:y,x2,y2,t:`${p.id}: the key names it ${p.key}`})}
    if(p.dna&&p.dna.verdict==='disagrees'&&(on('L_dna')||isSel)){for(const n of p.dna.nearest_other){const t=byName[n];if(!t)continue;const x2=sx(t.x),y2=sy(t.y);
      line(x,y,x2,y2,COL.dnabad,isSel?2.2:1.5,0.85*a,[2,3],true,discR(t)+2);HL.push({x1:x,y1:y,x2,y2,t:`${p.id}: COI nearest is ${n}`})}}
    if(p.named!==p.species&&byName[p.named]&&(on('L_matrix')||isSel)){const t=byName[p.named];const g=p.strength==null?0.5:p.strength;const x2=sx(t.x),y2=sy(t.y);
      line(x,y,x2,y2,SCOL(p.status==='check annotation'||p.status==='fits'?'check':p.status),(1+4*g)*(isSel?1.3:1),(0.3+0.65*g)*a,null,true,discR(t)+2);
      HL.push({x1:x,y1:y,x2,y2,t:`${p.id} (${p.species}): the matrix names it ${p.named}; ${p.support||'-'} sets agree with its label; decisiveness ${p.strength==null?'-':p.strength.toFixed(2)}`})}}
  // species discs
  for(const s of sp){const x=sx(s.x),y=sy(s.y),r=discR(s);if(x<-60||y<-60||x>W+60||y>H+60)continue;const act=spActive(s),m=matchS(s);
    X.globalAlpha=0.92*dim(act&&m);X.beginPath();X.arc(x,y,r,0,2*Math.PI);X.fillStyle=spCol(s);X.fill();X.globalAlpha=1;
    if(selSp===s.id||(query&&m)){X.lineWidth=selSp===s.id?3:2;X.strokeStyle=COL.ink;X.beginPath();X.arc(x,y,r+2,0,2*Math.PI);X.stroke()}
    HP.push({x,y,r:r+3,o:{type:'sp',id:s.id,o:s}});
    if(view.k>DATA.label_zoom||s.re||s.check||selSp===s.id||(query&&m)){
      labels.push({text:view.k>DATA.label_zoom*1.6?`${s.id} ${s.named_back}/${s.n}`:s.id,x,y,r:Math.max(r,showSpec(s)?s.r*view.k+6:r),inner:showSpec(s)&&s.r*view.k>r+16?r:null,font:(s.re||s.check?'600 ':'')+'12px system-ui',size:12,col:COL.ink,
        prio:(selSp===s.id?1000:0)+(query&&m?500:0)+(s.re?200:s.check?100:0)+s.n,alpha:dim(act&&m),force:selSp===s.id}) }}
  // specimens
  for(const p of spec){const s=byName[p.species];if(!s)continue;const isSel=selP===p;if(!showSpec(s)&&!isSel)continue;if(!shown(p)&&!isSel)continue;
    const x=sx(p.x),y=sy(p.y);if(x<-10||y<-10||x>W+10||y>H+10)continue;const r=dotR(p),a=isSel?1:dim(matchP(p));
    X.globalAlpha=a;X.beginPath();X.arc(x,y,r,0,2*Math.PI);X.fillStyle=SCOL(p.status);X.fill();
    if(on('L_ann')&&p.annotation&&p.status!=='check annotation'){X.lineWidth=2;X.strokeStyle=COL.ann;X.beginPath();X.arc(x,y,r+3,0,2*Math.PI);X.stroke()}
    if(on('L_dna')&&p.dna){const bx=x+r+1,by=y-r-6,q=5;if(p.dna.verdict==='agrees'){X.fillStyle=COL.dnaok;X.fillRect(bx,by,q,q)}
      else if(p.dna.verdict==='disagrees'){X.strokeStyle=COL.dnabad;X.lineWidth=2;X.beginPath();X.moveTo(bx-1,by-1);X.lineTo(bx+q+1,by+q+1);X.moveTo(bx+q+1,by-1);X.lineTo(bx-1,by+q+1);X.stroke()}}
    if(isSel){X.lineWidth=2.5;X.strokeStyle=COL.ink;X.beginPath();X.arc(x,y,r+6,0,2*Math.PI);X.stroke()}
    X.globalAlpha=1;HP.push({x,y,r:Math.max(r+3,8),o:{type:'spec',id:p.id,o:p},spec:1});
    if(isSel||((p.status==='re-examine'||p.status==='check')&&view.k>DATA.label_zoom*1.3)||(p.status!=='fits'&&view.k>DATA.label_zoom*4)||(query&&p.id.toLowerCase().includes(query)))
      labels.push({text:p.id,x,y,r:r+2,font:'10px system-ui',size:10,col:COL.muted,prio:isSel?900:(query?400:0)+(3-ORDER[p.status])*10,alpha:a,force:isSel})}
  placeLabels(labels)}
function segD(px,py,l){const dx=l.x2-l.x1,dy=l.y2-l.y1,L2=dx*dx+dy*dy||1;let t=((px-l.x1)*dx+(py-l.y1)*dy)/L2;t=Math.max(0,Math.min(1,t));return Math.hypot(px-l.x1-t*dx,py-l.y1-t*dy)}
function pick(mx,my){let best=null,bd=1e9;for(const h of HP){const d=Math.hypot(h.x-mx,h.y-my)-(h.spec?2:0);if(d<h.r&&d<bd){bd=d;best=h.o}}return best}
function pickLine(mx,my){let best=null,bd=5;for(const l of HL){const d=segD(mx,my,l);if(d<bd){bd=d;best=l}}return best}
// cards
const go=(id,txt)=>(byName[id]||specById[id])?`<a class="go" data-go="${esc(id)}">${esc(txt||id)}</a>`:esc(txt||id);
function wire(){document.querySelectorAll('#detail [data-go]').forEach(a=>a.onclick=e=>{e.stopPropagation();openId(a.dataset.go)});
  document.querySelectorAll('#detail .row[data-id]').forEach(r=>r.onclick=()=>openId(r.dataset.id))}
function dnaBlock(p){if(!DATA.has_dna)return '';if(!p.dna)return '<div class="box"><b>DNA (COI)</b>: no sequence for this specimen</div>';const d=p.dna;
  const v=d.verdict==='agrees'?`<span class="ok">agrees with its label</span>`:d.verdict==='disagrees'?`<span class="bad">nearest sequence is another species</span>`:'no verdict';
  return `<div class="box"><b>DNA (COI)</b>: ${v}<div class="why">`+d.seqs.map(q=>`${esc(q.seq)}: nearest ${q.nearest?go(q.nearest):'-'}${q.k2p==null?'':`, K2P ${q.k2p.toFixed(4)}`}`).join('<br>')+`</div></div>`}
function gapLine(s){const g=DATA.gaps[s.id];if(!g)return DATA.has_dna?'<div class="why">no barcode-gap row for this species</div>':'';
  return `<div class="box"><b>COI barcode gap</b>: ${g.gap==null?'not computable (one sequence)':g.gap.toFixed(4)+(g.gap<=0?' <span class="bad">(no gap)</span>':'')}<div class="why">${g.n||0} sequence(s); max within ${g.max_intra==null?'-':g.max_intra.toFixed(4)}, min to other species ${g.min_inter==null?'-':g.min_inter.toFixed(4)}${g.nearest?' ('+go(g.nearest)+')':''}</div></div>`}
function showSpecimen(p){sel={type:'spec',id:p.id};const reasons=(p.why||'').split(' | ').filter(Boolean);
  $('detail').innerHTML=`<h2>${esc(p.id)}</h2>${pill(p.status)}<table class="kv">
<tr><td>filed as</td><td>${go(p.species)}</td></tr>
<tr><td>matrix names it</td><td>${go(p.named)}${p.named===p.species?' (its label)':''}</td></tr>
<tr><td>support</td><td>${p.support?`${p.support} character sets agree with its label`:'-'}${p.own_rank?`; its own species ranks #${p.own_rank}`:''}${p.margin!=null?`; margin to runner-up ${p.margin}`:''}</td></tr>
${p.named!==p.species&&p.strength!=null?`<tr><td>decisiveness</td><td>${p.strength.toFixed(2)} <span class="why">(0 = close call, 1 = its own species is the furthest)</span></td></tr>`:''}
<tr><td>per set</td><td class="why">${esc(p.per_set||'-')}</td></tr>
<tr><td>key names it</td><td>${p.key?go(p.key):'-'}</td></tr></table>
<b>Reasons</b>${reasons.length?`<ul class="reasons">${reasons.map(r=>`<li>${esc(r)}</li>`).join('')}</ul>`:'<div class="why">no evidence against its label</div>'}
${dnaBlock(p)}<div class="row" data-go="${esc(p.species)}">&larr; all ${esc(p.species)} specimens</div>`;wire();draw()}
function showSpecies(s){sel={type:'sp',id:s.id};const list=(specBy[s.id]||[]).slice().sort((a,b)=>ORDER[a.status]-ORDER[b.status]||a.id.localeCompare(b.id));
  const nb=(nbrOf[s.id]||[]).slice(0,DATA.k_nbr+2);const only=DATA.dna_only[s.id]||[];
  $('detail').innerHTML=`<h2>${esc(s.id)}</h2><div class="sub">${s.n} specimens; ${s.named_back} named back by the matrix; ${s.re} to re-examine, ${s.check} to check</div>`+
  (s.confused?`<div class="why" style="margin:6px 0">confused with: ${esc(s.confused)}</div>`:'')+
  (nb.length?`<div class="why" style="margin:6px 0">nearest species (matrix distance): ${nb.map(([n,d])=>`${go(n)} ${d.toFixed(3)}`).join(', ')}</div>`:'')+gapLine(s)+
  list.map(p=>`<div class="row" data-id="${esc(p.id)}"><span class="sp">${esc(p.id)}</span> ${pill(p.status)}${p.named!==p.species?` <span class="sub">&rarr; ${esc(p.named)}</span>`:''}${p.dna&&p.dna.verdict==='disagrees'?' <span class="bad">COI</span>':''}<div class="why">${esc(p.why||'no evidence against its label')}</div></div>`).join('')+
  (only.length?`<div class="why" style="margin-top:8px">COI only (no morphology record): ${only.map(esc).join(', ')}</div>`:'');wire();draw()}
function overview(){sel=null;const grp=(st,t)=>{const l=spec.filter(p=>p.status===st&&ST[st]);return l.length?`<h2>${t} (${l.length})</h2>`+l.map(p=>`<div class="row" data-id="${esc(p.id)}"><span class="sp">${esc(p.id)}</span> <span class="sub">${esc(p.species)}${p.named!==p.species?' &rarr; '+esc(p.named):''}</span>${p.dna&&p.dna.verdict==='disagrees'?' <span class="bad">COI</span>':''}<div class="why">${esc(p.why)}</div></div>`).join(''):''};
  $('detail').innerHTML=(grp('re-examine','To re-examine')+grp('check','To check (identity)')+grp('check annotation','Annotation to check'))||'<div class="why">no flagged specimens with the current filter</div>';wire();draw()}
function focusOn(x,y,minR){view.k=Math.max(view.k,minR);view.x=W/2-x*view.k;view.y=H/2+y*view.k}
const zoomFor=r=>Math.min(W,H)/(Math.max(r,1e-3)*12);
function openId(id){const s=byName[id];if(s){focusOn(s.x,s.y,zoomFor(s.r));showSpecies(s);return true}
  const p=specById[id];if(p){const t=byName[p.species];const pts=[p.species,p.named,p.key,...(p.dna?p.dna.nearest_other:[])].filter(n=>byName[n]).map(n=>byName[n]);
    if(pts.length>1){fitBox(pts)}else focusOn(p.x,p.y,zoomFor(t?t.r:0.05));showSpecimen(p);return true}return false}
function fitBox(list){let a=1e9,b=-1e9,c=1e9,d=-1e9;for(const s of list){a=Math.min(a,s.x-s.r);b=Math.max(b,s.x+s.r);c=Math.min(c,s.y-s.r);d=Math.max(d,s.y+s.r)}
  const lg=$('legend'),rm=(!lg.classList.contains('hide')&&W>760)?lg.offsetWidth+12:0,Wf=W-rm,m=50;
  view.k=Math.min((Wf-2*m)/Math.max(b-a,1e-3),(H-2*m)/Math.max(d-c,1e-3),zoomFor(list[0].r));view.x=Wf/2-(a+b)/2*view.k;view.y=H/2+(c+d)/2*view.k}
// search
const hitsEl=$('hits');let hitList=[],hitOn=0;
function renderHits(){if(!query){hitsEl.style.display='none';return}
  hitList=[...sp.filter(s=>s.id.toLowerCase().includes(query)).map(s=>({id:s.id,t:`${s.id}`,k:'species'})),...spec.filter(p=>p.id.toLowerCase().includes(query)).map(p=>({id:p.id,t:p.id,k:p.status}))].slice(0,30);
  hitsEl.innerHTML=hitList.length?hitList.map((h,i)=>`<div data-i="${i}" class="${i===hitOn?'on':''}">${esc(h.t)} <span class="sub">${esc(h.k)}</span></div>`).join(''):'<div class="sub">no match</div>';hitsEl.style.display='block';
  hitsEl.querySelectorAll('[data-i]').forEach(d=>d.onmousedown=e=>{e.preventDefault();choose(+d.dataset.i)})}
function choose(i){const h=hitList[i];if(!h)return;hitsEl.style.display='none';$('q').blur();openId(h.id)}
$('q').oninput=e=>{query=e.target.value.trim().toLowerCase();hitOn=0;renderHits();draw()};
$('q').onkeydown=e=>{if(e.key==='Enter'){choose(hitOn)}else if(e.key==='ArrowDown'){hitOn=Math.min(hitList.length-1,hitOn+1);renderHits();e.preventDefault()}else if(e.key==='ArrowUp'){hitOn=Math.max(0,hitOn-1);renderHits();e.preventDefault()}else if(e.key==='Escape'){e.target.value='';query='';renderHits();draw()}};
$('q').onblur=()=>setTimeout(()=>hitsEl.style.display='none',150);$('q').onfocus=renderHits;
document.querySelectorAll('[data-status]').forEach(c=>c.onchange=()=>{readST();if(!sel)overview();draw()});
['L_matrix','L_key','L_dna','L_ann','L_nbr'].forEach(i=>$(i).onchange=draw);
// pointer: drag to pan, wheel or pinch to zoom, tap to select
const pts=new Map();let drag=null,pinch=null;
C.onpointerdown=e=>{C.setPointerCapture(e.pointerId);pts.set(e.pointerId,[e.offsetX,e.offsetY]);
  if(pts.size===1)drag={x:e.offsetX,y:e.offsetY,vx:view.x,vy:view.y,moved:false};
  else if(pts.size===2){const [a,b]=[...pts.values()];pinch={d:Math.hypot(a[0]-b[0],a[1]-b[1]),k:view.k,vx:view.x,vy:view.y,cx:(a[0]+b[0])/2,cy:(a[1]+b[1])/2};drag=null}};
C.onpointermove=e=>{if(pts.has(e.pointerId))pts.set(e.pointerId,[e.offsetX,e.offsetY]);
  if(pinch&&pts.size===2){const [a,b]=[...pts.values()];const f=Math.hypot(a[0]-b[0],a[1]-b[1])/pinch.d;view.k=pinch.k*f;view.x=pinch.cx-(pinch.cx-pinch.vx)*f;view.y=pinch.cy-(pinch.cy-pinch.vy)*f;draw();return}
  if(drag){const dx=e.offsetX-drag.x,dy=e.offsetY-drag.y;if(Math.abs(dx)+Math.abs(dy)>4)drag.moved=true;if(drag.moved){view.x=drag.vx+dx;view.y=drag.vy+dy;C.style.cursor='grabbing';tip.style.display='none';draw()}return}
  const h=pick(e.offsetX,e.offsetY);let t='';
  if(h)t=h.type==='sp'?`${h.o.id}: ${h.o.named_back}/${h.o.n} named back, ${h.o.re} re-examine, ${h.o.check} check`:`${h.o.id} (${h.o.status}) - matrix: ${h.o.named}; sets agreeing ${h.o.support||'-'}`+(h.o.dna?`; COI ${h.o.dna.verdict}`:'');
  else{const l=pickLine(e.offsetX,e.offsetY);if(l)t=l.t}
  if(t){tip.style.display='block';tip.textContent=t;const tx=Math.min(e.offsetX+12,W-tip.offsetWidth-4);tip.style.left=Math.max(4,tx)+'px';tip.style.top=(e.offsetY+14)+'px';C.style.cursor=h?'pointer':'grab'}else{tip.style.display='none';C.style.cursor='grab'}};
C.onpointerup=C.onpointercancel=e=>{pts.delete(e.pointerId);if(pinch){if(pts.size<2)pinch=null;drag=null;return}
  if(drag&&!drag.moved&&e.type==='pointerup'){const h=pick(e.offsetX,e.offsetY);if(h){h.type==='sp'?showSpecies(h.o):showSpecimen(h.o)}else overview()}drag=null;C.style.cursor='grab'};
C.onpointerleave=()=>{tip.style.display='none'};
C.onwheel=e=>{e.preventDefault();const f=Math.exp(-e.deltaY*0.0015);const mx=e.offsetX,my=e.offsetY;view.x=mx-(mx-view.x)*f;view.y=my-(my-view.y)*f;view.k*=f;draw()};
$('fitb').onclick=()=>{fit();draw()};
$('legb').onclick=()=>{$('legend').classList.toggle('hide');fit();draw()};
// legend with drawn samples
function legend(){const L=$('legend');const it=(id,t)=>`<div class="li"><canvas id="lg_${id}" width="60" height="28"></canvas><span>${t}</span></div>`;
  L.innerHTML='<h3>specimens (dots)</h3>'+it('re','re-examine')+it('ck','check (matrix alone)')+it('an','check annotation only')+it('fi','fits its label')+
  '<h3>species</h3>'+it('sp','species (colour = most serious status); its specimens on the ring')+
  '<h3>evidence layers</h3>'+it('mx','matrix names it as that species; thick = decisive, faint = close call')+it('ky','key names it as that species')+(DATA.has_dna?it('dok','COI nearest = its own species')+it('dbd','COI nearest = another species'):'')+
  it('ar','annotation flag on a specimen with an identity flag or that fits')+it('nb',`${DATA.k_nbr} nearest species (hover: distance)`)+
  '<div class="sub" style="margin-top:6px">Layout: MDS of matrix distances, nudged apart so discs do not overlap; read distances from the dashed lines. Drag = pan, scroll/pinch = zoom, click = details.</div>';
  const g=id=>{const c=$('lg_'+id),x=c.getContext('2d');x.scale(2,2);return x};
  const dot=(id,col,ring)=>{const x=g(id);x.fillStyle=col;x.beginPath();x.arc(15,7,4,0,7);x.fill();if(ring){x.strokeStyle=COL.ann;x.lineWidth=2;x.beginPath();x.arc(15,7,6.5,0,7);x.stroke()}};
  dot('re',COL.re);dot('ck',COL.check);dot('an',COL.ann);dot('fi',COL.fits);dot('ar',COL.fits,true);
  {const x=g('sp');x.fillStyle=COL.halo;x.beginPath();x.arc(15,7,7,0,7);x.fill();x.fillStyle=COL.fits;x.beginPath();x.arc(15,7,3.5,0,7);x.fill()}
  const ln=(id,col,w,dash,arrow)=>{const x=g(id);x.strokeStyle=col;x.fillStyle=col;x.lineWidth=w;x.setLineDash(dash);x.beginPath();x.moveTo(2,7);x.lineTo(arrow?22:28,7);x.stroke();if(arrow){x.setLineDash([]);x.beginPath();x.moveTo(28,7);x.lineTo(21,3);x.lineTo(21,11);x.closePath();x.fill()}};
  {const x=g('mx');x.strokeStyle=COL.re;x.globalAlpha=0.35;x.lineWidth=1;x.beginPath();x.moveTo(2,3);x.lineTo(28,3);x.stroke();x.globalAlpha=0.95;x.lineWidth=4;x.beginPath();x.moveTo(2,10);x.lineTo(28,10);x.stroke()}
  ln('ky',COL.key,1.5,[6,4],true);ln('nb',COL.nbr,1,[4,4],false);
  if(DATA.has_dna){{const x=g('dok');x.fillStyle=COL.fits;x.beginPath();x.arc(12,8,3.5,0,7);x.fill();x.fillStyle=COL.dnaok;x.fillRect(17,1,5,5)}
    {const x=g('dbd');x.fillStyle=COL.re;x.beginPath();x.arc(7,8,3.5,0,7);x.fill();x.strokeStyle=COL.dnabad;x.lineWidth=2;x.beginPath();x.moveTo(11,1);x.lineTo(16,6);x.moveTo(16,1);x.lineTo(11,6);x.stroke();x.lineWidth=1.5;x.setLineDash([2,3]);x.beginPath();x.moveTo(12,9);x.lineTo(29,9);x.stroke()}}}
legend();
if(window.innerWidth<=700){$('opts').open=false;$('legend').classList.add('hide')}
if(window.matchMedia)window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{readColours();legend();draw()});
window.onresize=resize;resize();overview();
// deep link: reexamine_map.html#<species or specimen>
function fromHash(){const h=decodeURIComponent(location.hash.slice(1));if(h)openId(h)}
window.onhashchange=fromHash;fromHash();
</script></body></html>"""


def write_map_v2(path, df, sp_tab, sp_ids, X, D, score, dna=None, gaps=None, k_nbr=3):
    """the v2 map: same species and specimens as v1's write_map, plus specimen positions, call lines, nearest
    species with distances and the optional DNA layer"""
    dna, gaps = dna or {}, gaps or {}
    st = sp_tab.set_index("species")
    keep = [i for i, s in enumerate(sp_ids) if s in st.index]
    ids = [sp_ids[i] for i in keep]
    Xk, Dk = np.asarray(X)[keep], np.asarray(D)[np.ix_(keep, keep)]
    P, r = declutter(Xk, [int(st.loc[s, "specimens"]) for s in ids])
    spos = specimen_positions(df, ids, P, r)
    species = []
    for i, s in enumerate(ids):
        row = st.loc[s]
        species.append({"id": s, "x": round(float(P[i][0]), 5), "y": round(float(P[i][1]), 5), "r": round(float(r[i]), 5),
                        "mds_x": round(float(Xk[i][0]), 5), "mds_y": round(float(Xk[i][1]), 5), "n": int(row["specimens"]),
                        "named_back": int(row["named_back"]), "re": int(row["re_examine"]), "check": int(row["check"]),
                        "confused": row["confused_with"]})
    nbr = {}
    for i, s in enumerate(ids):
        for rank, j in enumerate([j for j in np.argsort(Dk[i]) if j != i][:k_nbr], 1):
            e = tuple(sorted((s, ids[j])))
            if e not in nbr or rank < nbr[e][1]:
                nbr[e] = (round(float(Dk[i, j]), 4), rank)
    known = set(ids)
    specimens, n_dna, n_bad = [], 0, 0
    for _, r_ in df.iterrows():
        sid, spn = r_["specimen_id"], r_["species"]
        x, y = spos.get(sid, (0.0, 0.0))
        rec = {"id": sid, "species": spn, "status": r_["status"], "named": r_["matrix_named"],
               "support": (f"{int(r_['sets_agreeing'])}/{int(r_['sets_scored'])}" if pd.notna(r_.get("sets_scored")) else ""),
               "key": r_["key_named"] if r_["key_named"] not in ("", "nan") and pd.notna(r_["key_named"]) else "",
               "per_set": r_["per_set"] if pd.notna(r_["per_set"]) else "", "why": r_["reasons"],
               "annotation": r_["annotation"] if pd.notna(r_["annotation"]) else "",
               "own_rank": None if pd.isna(r_["matrix_own_rank"]) else int(r_["matrix_own_rank"]),
               "margin": None if pd.isna(r_["matrix_margin"]) else float(r_["matrix_margin"]),
               "strength": call_strength(score, sid, spn, r_["matrix_named"]) if r_["matrix_named"] != spn else None,
               "x": round(x, 5), "y": round(y, 5)}
        if sid in dna:
            v = dna_verdict(dna[sid])
            rec["dna"] = {"verdict": v, "seqs": dna[sid],
                          "nearest_other": sorted({q["nearest"] for q in dna[sid] if q["agrees"] is False and q["nearest"] in known})}
            n_dna += 1
            n_bad += v == "disagrees"
        specimens.append(rec)
    in_df = set(df["specimen_id"])
    dna_only = defaultdict(list)
    for sid, seqs in dna.items():
        if sid not in in_df:
            dna_only[seqs[0]["species"]].append(sid)
    n = len(species)
    data = {"species": species, "specimens": specimens,
            "nbr": [[a, b, d, k] for (a, b), (d, k) in sorted(nbr.items())], "k_nbr": k_nbr,
            "n_species": n, "n_specimens": len(specimens),
            "n_re": int((df["status"] == "re-examine").sum()), "n_check": int((df["status"] == "check").sum()),
            "n_check_ann": int((df["status"] == "check annotation").sum()), "n_fits": int((df["status"] == "fits").sum()),
            "n_ann": int((df["annotation"] != "").sum()),
            "has_dna": bool(dna), "n_dna": n_dna, "n_dna_bad": int(n_bad), "gaps": gaps,
            "dna_only": {k: sorted(v) for k, v in dna_only.items()}, "map_version": MAP_VERSION,
            # zoom (pixels per layout unit) above which every species label / neighbour distance is written
            "label_zoom": 250 if n <= 60 else 250 * math.sqrt(n / 60),
            "nbr_label_zoom": 900 if n <= 60 else 900 * math.sqrt(n / 60)}
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    Path(path).write_text(HTML_V2.replace("__DATA__", blob), encoding="utf-8")
    return data


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
    ap.add_argument("--dna_table", default=None, help="v2, optional map layer: specimens_dna_vs_matrix.tsv from "
                    "dna_vs_morphology_v1 (specimen_id, dna_nearest, dna_nearest_k2p, dna_agrees_with_label); never "
                    "changes a status")
    ap.add_argument("--barcode_gap", default=None, help="v2: barcode_gap.tsv (default: beside --dna_table)")
    ap.add_argument("--k_nbr", type=int, default=3, help="v2 map: link each species to this many nearest species")
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
    dna, gaps = load_dna(a.dna_table, a.barcode_gap)
    mapd = write_map_v2(out / "reexamine_map.html", df, sp_tab, sp_ids, X, D, score, dna, gaps, a.k_nbr)
    summ = {"version": VERSION, "specimens": len(df), "species": int(df["species"].nunique()),
            "re_examine": int((df["status"] == "re-examine").sum()), "check": int((df["status"] == "check").sum()),
            "fits": int((df["status"] == "fits").sum()), "annotation_to_check": int((df["annotation"] != "").sum()),
            "matrix_named_back": int((mv["matrix_named"] == mv["species"]).sum()),
            "inputs": inputs}
    summ["map_version"] = MAP_VERSION
    if a.dna_table:
        summ["dna"] = (f"{mapd['n_dna']} specimens with COI, {mapd['n_dna_bad']} nearest to another species "
                       f"(map layer only)") if dna else f"not found: {a.dna_table}"
    json.dump(summ, open(out / "reexamine_summary.json", "w"), indent=2)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()

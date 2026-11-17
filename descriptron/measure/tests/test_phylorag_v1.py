"""phylorag_v1 (PhyloRAG): every step on a small invented plant genus (Floria), with model replies replayed from
files, so no model and no network are needed. The checks on real data (the centipede run: source selection, the
134 second-extraction states, the cross-check against the hand-made no-proxy decisions, and a live extraction)
are described in the program's README section and are run outside the test suite."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import phylorag_v1 as pr          # noqa: E402

PROG = str(HERE / "phylorag_v1.py")
FILLER = "Collected in wet forest on limestone. " * 15          # puts the synonymy > 400 characters before the description
CHARS = [
    {"id": 1, "description": "Leaf blade shape", "states": {"0": "lanceolate", "1": "ovate"}},
    {"id": 2, "description": "Petal indumentum", "states": {"0": "glabrous", "1": "pubescent"}},
    {"id": 3, "description": "Number of stamens", "type": "meristic"},
    {"id": 4, "description": "Plant height", "type": "continuous", "unit": "cm"},
    {"id": 5, "description": "Sepal keel", "states": {"0": "absent", "1": "present"}, "ordered": True},
    {"id": 6, "description": "Petal colour", "states": {"0": "white", "1": "red"}},
]
SOURCES = [
    {"id": "101", "title": "Floria alba Smith, 1900", "doi": "10.0/alba", "kind": "biosyslit",
     "text": "Floria alba Smith, 1900. Floria rubra Jones, 1890: 12 (misidentification). Material examined. "
             + FILLER + "Description. Leaf blade lanceolate. Petals white, glabrous. Stamens 12. "
             "Plant height 25&ndash;30 cm. Unlike F . rubra, the sepals are keeled."},
    {"id": "102", "title": "Floria rubra Jones, 1890", "doi": "10.0/rubra", "kind": "biosyslit",
     "text": "Floria rubra Jones, 1890. Description. Leaf blade ovate. Petals red, pubescent. Stamens 10. "
             "Plant height 40 cm. Sepals not keeled."},
    {"id": "103", "title": "Fig. 2 in A revision of Floria alba and allies", "doi": "", "kind": "biosyslit",
     "text": "Fig. 2. Floria rubra, habit. Leaf blade ovate, petals red; scale bar 1 cm. Photograph by the authors."},
    {"id": "104", "title": "A revision of Floria, with notes on Floria alba", "doi": "", "kind": "biosyslit",
     "text": "Smith, A. (2020): A revision of Floria, with notes on Floria alba. Journal of Plant Names 1: 1-20, "
             "DOI: 10.0/rev"},
    {"id": "pdf:paper.pdf", "title": "Notes on two Floria species", "doi": "", "kind": "pdf", "file": "/x/paper.pdf",
     "text": "Floria rubra. Leaves ovate; petals pubescent. Floria alba. Leaves lanceolate; flowering in May."},
]
ALBA = [
    {"id": 1, "state": "0", "evidence": "Leaf blade lanceolate."},
    {"id": 2, "state": "0", "evidence": "Petals white, glabrous."},
    {"id": 3, "value": "12", "evidence": "Stamens 12."},
    {"id": 4, "value": "25-30", "evidence": "Plant height 25–30 cm."},
    {"id": 5, "state": "1", "evidence": "Unlike F. rubra, the sepals are keeled."},
    {"id": 6, "state": "2", "evidence": "Petals white, glabrous."},                    # state not allowed
    {"id": 99, "state": "0", "evidence": "Stamens 12."},                                # unknown character
    {"id": 2, "state": "1", "evidence": "Leaves ovate; petals pubescent."},            # from the PDF, after F. rubra
    {"id": 3, "value": "14", "evidence": "Stamens 12."},                                # number not in the quote
    {"id": 1, "state": "0", "evidence": "The leaf blade is narrowly lanceolate and long."},   # paraphrase
]
RUBRA = [
    {"id": 1, "state": "1", "evidence": "Leaf blade ovate."},
    {"id": 6, "state": "1", "evidence": "Petals red, pubescent."},
    {"id": 2, "state": "0,1", "evidence": "Petals red, pubescent."},                   # two states (variation)
]
BASE_NEX = """#NEXUS
BEGIN TAXA;
\tDIMENSIONS NTAX=2;
\tTAXLABELS 'Floria rubra' 'Floria viridis' ;
END;
BEGIN CHARACTERS;
\tDIMENSIONS NCHAR=4;
\tFORMAT DATATYPE=STANDARD GAP=- MISSING=? SYMBOLS="0 1";
\tCHARLABELS
\t\t[1] 'Leaf blade shape'
\t\t[2] 'Petal indumentum'
\t\t[3] 'Sepal keel'
\t\t[4] 'Petal colour'
\t;
\tSTATELABELS
\t\t1 'lanceolate' 'ovate' ,
\t\t2 'glabrous' 'pubescent' ,
\t\t3 'absent' 'present' ,
\t\t4 'white' 'red'
\t;
\tMATRIX
\t'Floria rubra'     0(0 1)?1
\t'Floria viridis'   1??-
\t;
END;
BEGIN ASSUMPTIONS;
TYPESET * default = ORD: 3 ;
ENDBLOCK;
"""


def run(*args, ok=True):
    r = subprocess.run([sys.executable, "-I", PROG, *map(str, args)], capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


def tsv(p):
    return list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))


@pytest.fixture
def setup(tmp_path):
    (tmp_path / "chars.json").write_text(json.dumps(CHARS))
    (tmp_path / "sources.json").write_text(json.dumps(SOURCES))
    (tmp_path / "taxa.tsv").write_text("taxon\ttip_label\tsynonyms\tsources\n"
                                       "Floria alba\tFa_1\t\tpaper.pdf\nFloria rubra\t\tFloria rosea\t\n")
    raw = tmp_path / "saved" / "raw"
    raw.mkdir(parents=True)
    for name, items in (("Floria_alba", ALBA), ("Floria_rubra", RUBRA)):
        (raw / f"{name}__c0.json").write_text(json.dumps({"prompt_sha256": "x", "response": "Here:\n" + json.dumps(items)}))
    base = [str(tmp_path / f) for f in ("chars.json", "taxa.tsv", "sources.json")]
    return tmp_path, base


def extract(tmp_path, base, out="run", *extra):
    c, t, s = base
    return run("extract", "--characters", c, "--taxa", t, "--sources", s, "--out-dir", tmp_path / out,
               "--llm-backend", "none", "--replay", tmp_path / "saved", *extra)


# ── text and parsing ────────────────────────────────────────────────────────────────────────────────────────────
def test_quote_matching_and_values():
    src = pr.norm("Legs 1&ndash;19 with tibial spurs. Tergites 6 20 with complete sutures.")
    assert pr.find_quote("legs 1–19 with tibial spurs", src, 0.9)[0] == "exact"      # entity vs dash
    assert pr.find_quote("Tergites 6-20 with complete sutures", src, 0.9)[0] == "exact"   # entity dropped earlier
    assert pr.find_quote("Tergites 6-20 with complete sutrues", src, 0.9)[0] == "absent"        # one typo: 0.895
    assert pr.find_quote("Unlike F. rubra, the sepals are keeled", pr.norm("Unlike F . rubra, the sepals are keeled. Stamens 12."), 0.9)[0] == "near"
    assert pr.find_quote("Legs without any spurs at all", src, 0.9)[0] == "absent"
    assert pr.find_quote("short", src, 0.9)[0] == "too short"
    assert pr.parse_value("19-23") == (19, 23) and pr.parse_value("2,5 to 3.1") == (2.5, 3.1)
    assert pr.parse_value("21") == (21, 21) and pr.parse_value("many") is None
    assert pr.split_states("(0 1)") == ["0", "1"] and pr.split_states("0&1") == ["0", "1"]


def test_names_titles_and_mentions():
    assert pr.names_in_title("Otostigmus (Parotostigmus) salticus Schileyko", ["Otostigmus salticus"], at_start=True)
    assert not pr.names_in_title("A synthesis ... with Otostigmus salticus", ["Otostigmus salticus"], at_start=True)
    assert pr.names_in_title("Otostigmus_salticus_Zootaxa", ["Otostigmus salticus"])
    assert not pr.names_in_title("Otostigmus multisalticus", ["Otostigmus salticus"])     # whole words only
    taxa = [{"taxon": "Floria alba", "tip_label": "", "synonyms": [], "sources": []}]
    vocab = pr.name_vocabulary([dict(s, file=s.get("file", "")) for s in SOURCES], taxa)
    assert ("floria", "rubra") in vocab[0] and ("an", "unknown") not in vocab[0]
    m = pr.mentions(pr.norm("Unlike F . rubra and Floria alba"), vocab)
    assert [c if isinstance(c, str) else c[0] for _, c in m] == ["Floria rubra", "Floria alba"]


# ── init ───────────────────────────────────────────────────────────────────────────────────────────────────────
def test_init_from_morphobank_and_charstatelabels(tmp_path):
    (tmp_path / "base.nex").write_text(BASE_NEX)
    run("init", "--from-nexus", tmp_path / "base.nex", "--out", tmp_path / "c.json")
    c = json.load(open(tmp_path / "c.json"))
    assert [x["description"] for x in c] == ["Leaf blade shape", "Petal indumentum", "Sepal keel", "Petal colour"]
    assert c[1]["states"] == {"0": "glabrous", "1": "pubescent"} and [x["ordered"] for x in c] == [False, False, True, False]
    (tmp_path / "m.nex").write_text("#NEXUS\nBEGIN DATA; DIMENSIONS NTAX=1 NCHAR=2; FORMAT SYMBOLS=\"01\";\n"
                                    "CHARSTATELABELS 1 'Leaf blade shape' / lanceolate ovate, 2 petal_hairs / "
                                    "glabrous 'densely pubescent';\nMATRIX\nA 01\n;\nEND;\n")
    c2 = pr.characters_from_nexus(tmp_path / "m.nex")
    assert c2[1]["description"] == "petal_hairs" and c2[1]["states"]["1"] == "densely pubescent"
    assert run("init", "--from-nexus", tmp_path / "base.nex", "--out", tmp_path / "c.json", ok=False).returncode != 0


def test_read_matrix_keeps_spaced_polymorphisms(tmp_path):
    (tmp_path / "base.nex").write_text(BASE_NEX)
    labs, rows, n = pr.read_nexus_matrix(tmp_path / "base.nex")
    assert n == 4 and rows["Floria rubra"] == ["0", "(01)", "?", "1"] and rows["Floria viridis"] == ["1", "?", "?", "-"]


def test_molecular_labels(tmp_path):
    (tmp_path / "a.fasta").write_text(">Fa_1 voucher\nACGT\n>Floria_nigra\nACGT\n")
    (tmp_path / "a.phy").write_text("2 4\nFa_1  ACGT\nFloria_nigra ACGT\n")
    (tmp_path / "t.nwk").write_text("((Fa_1:0.1,'Floria nigra':0.2),Floria_rubra);")
    (tmp_path / "t.nex").write_text("#NEXUS\nBEGIN TREES;\nTRANSLATE 1 Fa_1, 2 Floria_rubra;\n"
                                    "TREE t1 = (1:0.1,2:0.2);\nEND;\n")
    assert pr.molecular_labels(tmp_path / "a.fasta") == ["Fa_1", "Floria_nigra"]
    assert pr.molecular_labels(tmp_path / "a.phy") == ["Fa_1", "Floria_nigra"]
    assert pr.molecular_labels(tmp_path / "t.nwk") == ["Fa_1", "Floria nigra", "Floria_rubra"]
    assert pr.molecular_labels(tmp_path / "t.nex") == ["Fa_1", "Floria_rubra"]


# ── extract (replayed replies) ──────────────────────────────────────────────────────────────────────────────────
def test_dry_run_selects_own_treatments_only(setup):
    tmp, base = setup
    out = extract(tmp, base, "dry", "--dry-run").stdout
    assert "Floria alba: 2 sources (1 own)" in out and "Floria rubra: 1 sources (1 own)" in out
    assert "    103 " not in out and "    104 " not in out                     # figure and article records
    assert not (tmp / "dry").exists()


def test_extract_checks_every_state(setup):
    tmp, base = setup
    extract(tmp, base)
    kept, rej = tsv(tmp / "run" / "review_sheet.tsv"), tsv(tmp / "run" / "rejected.tsv")
    assert len(kept) == 9 and len(rej) == 4
    reasons = sorted(r["reason"].split(" (")[0] for r in rej)
    assert reasons == ["quote absent", "state 2 not allowed", "unknown character id", "value not in the quote"]
    by = {(r["taxon"], r["char_id"], r["state"] or r["value"]): r for r in kept}
    assert by[("Floria alba", "1", "0")]["flags"] == "" and by[("Floria alba", "1", "0")]["source_id"] == "101"
    assert by[("Floria alba", "4", "25–30")]["quote_match"] == "exact"                       # &ndash; in the source
    k = by[("Floria alba", "5", "1")]["flags"]
    assert "quote names another taxon: Floria rubra" in k and "comparative statement" in k
    assert "other taxon named before" not in k                                   # synonymy list is > 400 characters up
    p = by[("Floria alba", "2", "1")]
    assert p["source_id"] == "pdf:paper.pdf" and "other taxon named before the quote: Floria rubra" in p["flags"]
    assert "source is not a treatment of this taxon" in p["flags"] and "conflicting states" in p["flags"]
    assert "conflicting states" in by[("Floria alba", "2", "0")]["flags"]
    assert "more than one state" in by[("Floria rubra", "2", "0&1")]["flags"]
    assert all(r["decision"] == "" for r in kept) and by[("Floria alba", "1", "0")]["label"] == "Fa_1"
    s = {r["taxon"]: r for r in tsv(tmp / "run" / "summary.tsv")}
    assert s["Floria alba"]["proposed"] == "10" and s["Floria alba"]["kept"] == "6" and s["Floria rubra"]["kept"] == "3"
    prov = json.load(open(tmp / "run" / "run.json"))
    assert prov["backend"] == "replay" and prov["totals"]["kept"] == 9
    assert "different prompt" in s["Floria alba"]["note"]
    # prefill clean: only unflagged, verified states are pre-accepted
    extract(tmp, base, "run2", "--prefill", "clean")
    acc = [r for r in tsv(tmp / "run2" / "review_sheet.tsv") if r["decision"] == "accept"]
    assert len(acc) == 5 and all(not r["flags"] for r in acc)


def test_extract_without_model_or_replay_stops(setup):
    tmp, (c, t, s) = setup
    r = run("extract", "--characters", c, "--taxa", t, "--sources", s, "--out-dir", tmp / "x", "--llm-backend", "none",
            ok=False)
    assert r.returncode != 0 and "needs a model" in r.stderr


# ── audit and export ────────────────────────────────────────────────────────────────────────────────────────────
def _decide(path, rule):
    rows = tsv(path)
    for r in rows:
        r["decision"] = rule(r)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def test_audit_keeps_decisions(setup):
    tmp, (c, t, s) = setup
    extract(tmp, (c, t, s))
    sheet = tmp / "run" / "review_sheet.tsv"
    _decide(sheet, lambda r: "reject" if r["source_id"] == "pdf:paper.pdf" else "accept")
    run("audit", "--states", sheet, "--characters", c, "--taxa", t, "--sources", s, "--out-dir", tmp / "aud")
    rows = tsv(tmp / "aud" / "review_sheet.tsv")
    assert len(rows) == 9 and sum(r["decision"] == "reject" for r in rows) == 1
    assert json.load(open(tmp / "aud" / "audit_report.json"))["kept"] == 9


def test_export_alone_base_and_molecular(setup):
    tmp, (c, t, s) = setup
    extract(tmp, (c, t, s))
    sheet = tmp / "run" / "review_sheet.tsv"
    r = run("export", "--review", sheet, "--characters", c, "--taxa", t, "--out-dir", tmp / "e0", ok=False)
    assert r.returncode != 0 and "no accepted states" in r.stderr
    _decide(sheet, lambda r: "reject" if r["source_id"] == "pdf:paper.pdf" else "accept")
    run("export", "--review", sheet, "--characters", c, "--taxa", t, "--out-dir", tmp / "e1")
    labs, rows, n = pr.read_nexus_matrix(tmp / "e1" / "phylorag_matrix.nex")
    assert n == 4 and rows["Fa_1"] == ["0", "0", "1", "?"] and rows["Floria_rubra"] == ["1", "(01)", "?", "1"]
    assert "ORD: 3" in (tmp / "e1" / "phylorag_matrix.nex").read_text()
    cm = {r["char_id"]: r for r in tsv(tmp / "e1" / "phylorag_matrix_counts_measurements.tsv")}
    assert (cm["3"]["min"], cm["3"]["max"], cm["4"]["min"], cm["4"]["max"], cm["4"]["unit"]) == ("12", "12", "25", "30", "cm")
    pv = tsv(tmp / "e1" / "phylorag_matrix_provenance.tsv")
    assert len(pv) == 7 and all(p["evidence"] and p["source_id"] for p in pv)
    # into an existing matrix, with molecular names
    (tmp / "base.nex").write_text(BASE_NEX)
    (tmp / "uce.fasta").write_text(">Fa_1\nACGT\n>Floria_rubra\nACGT\n>Floria_nigra\nACGT\n")
    run("export", "--review", sheet, "--characters", c, "--taxa", t, "--out-dir", tmp / "e2",
        "--base-nexus", tmp / "base.nex", "--molecular", tmp / "uce.fasta", "--add-missing-taxa")
    labs, rows, n = pr.read_nexus_matrix(tmp / "e2" / "phylorag_matrix.nex")
    assert labs == ["Floria rubra", "Floria viridis", "Fa_1", "Floria_nigra"]
    assert rows["Floria rubra"] == ["0", "(01)", "?", "1"]           # base kept: 0 not overwritten by ovate (1)
    assert rows["Floria viridis"] == ["1", "?", "?", "-"] and rows["Floria_nigra"] == ["?"] * 4
    log = {x["char_pos"]: x["action"] for x in tsv(tmp / "e2" / "phylorag_matrix_merge_log.tsv")}
    assert log == {"1": "not filled: base has 0", "2": "agrees with base", "4": "agrees with base"}
    rep = json.load(open(tmp / "e2" / "phylorag_matrix_report.json"))
    assert rep["molecular"]["in_both"] == 3 and rep["molecular"]["added_as_empty_rows"] == ["Floria_nigra"]
    assert rep["molecular"]["only_in_matrix"] == ["Floria viridis"]


def test_export_refuses_other_characters_file(setup):
    tmp, (c, t, s) = setup
    extract(tmp, (c, t, s))
    sheet = tmp / "run" / "review_sheet.tsv"
    _decide(sheet, lambda r: "accept")
    shifted = [dict(x, description=CHARS[(i + 1) % len(CHARS)]["description"]) for i, x in enumerate(CHARS)]
    (tmp / "other.json").write_text(json.dumps(shifted))
    r = run("export", "--review", sheet, "--characters", tmp / "other.json", "--taxa", t, "--out-dir", tmp / "e", ok=False)
    assert r.returncode != 0 and "different character" in r.stderr


def test_include_unreviewed_takes_clean_rows_only(setup):
    tmp, (c, t, s) = setup
    extract(tmp, (c, t, s))
    run("export", "--review", tmp / "run" / "review_sheet.tsv", "--characters", c, "--taxa", t, "--out-dir", tmp / "e",
        "--include-unreviewed")
    pv = tsv(tmp / "e" / "phylorag_matrix_provenance.tsv")
    assert len(pv) == 3 and all(p["decision"].startswith("unreviewed") for p in pv)
    assert "WARNING" in (tmp / "e" / "phylorag_matrix.nex").read_text().split("\n")[1]


def test_fetch_reads_pdfs_and_text(tmp_path):
    d = tmp_path / "txt"
    d.mkdir()
    (d / "Floria_alba_1900.txt").write_text("Floria alba. Leaf blade lanceolate. " * 5)
    (d / "short.txt").write_text("too short")
    run("fetch", "--text-dir", d, "--out", tmp_path / "src.json")
    src = json.load(open(tmp_path / "src.json"))
    assert [x["id"] for x in src] == ["txt:Floria_alba_1900.txt"] and src[0]["kind"] == "txt"
    own, extra = pr.select_sources({"taxon": "Floria alba", "synonyms": [], "sources": []}, pr.load_sources(tmp_path / "src.json"), {})
    assert len(own) == 1                                     # the file name names the taxon

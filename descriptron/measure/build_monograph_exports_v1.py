#!/usr/bin/env python3
"""
build_monograph_exports_v1.py — standards-based machine-readable monograph
==========================================================================

Exports the BioRAG v2 monograph (key + treatments + specimens + data matrix)
in the formats that biodiversity infrastructures ingest, so treatments can be
re-used, aggregated and later updated with new material or observations.

  taxpub/<code>.taxpub.xml         Plazi treatment deposit (root tp:taxon-treatment),
                                   one per species; validated against the TaxPub DTD
  taxpub/<prefix>_monograph.taxpub.xml
                                   JATS/TaxPub article: key + all treatments
  dwca_checklist.zip               Darwin Core Archive, Taxon core +
                                   GBIF Description + Distribution extensions
  dwca_occurrences.zip             Darwin Core Archive, Occurrence core (every
                                   specimen: verbatim label, sex, locality, host) +
                                   MeasurementOrFact extension (every Tier-1 value
                                   with feature ID, unit and method)
  <prefix>_treatments.sdd.xml      TDWG SDD 1.1: natural-language treatments,
                                   coded descriptions for all Tier-1 characters,
                                   identification key; validated against SDD.xsd
  <prefix>_monograph.jsonld        schema.org Dataset linking all parts
  <prefix>_monograph.md            human-readable Markdown
  export_validation_report.json    schema validation results and counts
  README_machine_readable.md       what each file is and how to update it

Plazi / TaxPub notes
  * Materials are tp:material-citation elements with Darwin Core named-content
    (dwc:catalogNumber, dwc:sex, dwc:country, dwc:eventDate, dwc:recordedBy,
    dwc:fieldNumber, dwc:verbatimLabel, dwc:verbatimCoordinates ...).
  * Undescribed species carry tp:taxon-status "undescribed (working name)";
    no nomenclatural act and no type status are asserted.
  * Every Tier-1 number cited in a treatment is wrapped as
    named-content content-type="dsc:tier1-value" with vocab-term=<feature ID>,
    linking the text to the MeasurementOrFact rows.
  * Subjective (image-read) words are listed per treatment in a "notes"
    section and flagged in the JSON-LD / TSV outputs of
    biorag_subjective_checks_v1.py.

Usage
  python build_monograph_exports_v1.py --descriptions_dir .../descriptions \
     --key_dir .../key --matrix_dir .../compiled_key_tier \
     --localities .../Diaphorina_localities_verified.tsv --plates_dir .../plates \
     --taxon-profile .../diaphorina_taxon_profile.yaml --output_dir .../machine_readable
"""

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
import biorag_feature_policy as pol  # noqa: E402

EXPORT_VERSION = "1.0"
HERE = Path(__file__).parent
TAXPUB_DTD = HERE / "biorag_prompts" / "schemas" / "taxpub" / "tax-treatment-NS0-v1_flat.dtd"
SDD_XSD = HERE / "biorag_prompts" / "schemas" / "sdd_1.1" / "SDD.xsd"
TP = "http://www.plazi.org/taxpub"
XLINK = "http://www.w3.org/1999/xlink"
DWC = "http://rs.tdwg.org/dwc/terms/"
DC = "http://purl.org/dc/terms/"
UBIF = "http://rs.tdwg.org/UBIF/2006/"
COUNTRY_CODES = {"ethiopia": "ET", "kenya": "KE", "tanzania": "TZ", "uganda": "UG", "south africa": "ZA",
                 "madagascar": "MG", "somalia": "SO", "sudan": "SD", "yemen": "YE", "comoros": "KM",
                 "australia": "AU", "united states": "US", "brazil": "BR", "germany": "DE", "puerto rico": "PR"}


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def classification(profile):
    parts = [p.strip() for p in str(profile.get("taxon", {}).get("higher_classification", "")).split(":") if p.strip()]
    out = {"kingdom": "Animalia"}
    for p in parts:
        if p.endswith("idae"):
            out["family"] = p
        elif p.endswith("oidea"):
            out["superfamily"] = p
        else:
            out.setdefault("order", p)
    if profile.get("taxon", {}).get("family"):
        out["family"] = profile["taxon"]["family"]
    return out


def name_parts(code, profile):
    """(genus, epithet_text, epithet_reg, qualifier, status_text)"""
    genus = profile.get("taxon", {}).get("genus", "")
    name = pol.species_display_name(code, profile)
    rest = name[len(genus) + 1:] if genus and name.startswith(genus + " ") else name
    status = pol.species_status(code, profile)
    qual = ""
    reg = rest
    if rest.startswith("cf. "):
        qual, reg = "cf.", rest[4:]
    if status == "undescribed" or rest.startswith("sp.") or rest.startswith("sp "):
        reg = ""
    return genus, rest, reg, qual, status


# ─────────────────────────────────────────────────────────────────────────────
# load
# ─────────────────────────────────────────────────────────────────────────────

def load_all(a):
    prof = pol.load_taxon_profile(a.taxon_profile)
    key = json.loads((Path(a.key_dir) / "key_tree.json").read_text())
    order = list(key.get("species", {}))
    treats, flags = {}, {}
    for code in order:
        d = Path(a.descriptions_dir) / code
        tj = d / f"{code}_treatment.json"
        if tj.exists():
            treats[code] = json.loads(tj.read_text())
        fj = d / f"{code}_subjective_flags.json"
        flags[code] = json.loads(fj.read_text()) if fj.exists() else []
    md = Path(a.matrix_dir)
    long = pd.read_csv(md / "specimen_matrix_long.csv")
    fdict = pd.read_csv(md / "feature_dictionary.tsv", sep="\t")
    summ = pd.read_csv(md / "species_feature_summary.csv")
    loc = pd.read_csv(a.localities, sep="\t", dtype=str).fillna("") if a.localities else pd.DataFrame()
    plates = {}
    pi = Path(a.plates_dir or "") / "plate_index.tsv"
    if pi.exists():
        for r in csv.DictReader(open(pi), delimiter="\t"):
            plates[r["species"]] = (int(r["figure_number"]), r["plate_file"])
    return prof, key, order, treats, flags, long, fdict, summ, loc, plates


def specimens(long, loc):
    """All specimens (matrix ∪ localities) with their locality record."""
    sx = long[long["sex"] != "unknown"].groupby("specimen_id")["sex"].first().to_dict()
    sp = long.groupby("specimen_id")["species"].first().to_dict()
    recs = {}
    for sid, s in sp.items():
        recs[sid] = {"specimen_id": sid, "species": s, "sex": sx.get(sid, ""), "in_matrix": True}
    for r in loc.to_dict("records") if len(loc) else []:
        rec = recs.setdefault(r["specimen_id"], {"specimen_id": r["specimen_id"], "species": r["species"],
                                                  "in_matrix": False})
        for k, v in r.items():
            if v and not rec.get(k):
                rec[k] = v
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# TaxPub
# ─────────────────────────────────────────────────────────────────────────────

def E(tag, parent=None, text=None, **attrs):
    if ":" in tag:
        pre, local = tag.split(":")
        qn = f"{{{TP if pre == 'tp' else XLINK}}}{local}"
    else:
        qn = tag
    local_ns = {"xlink": XLINK} if any(k.startswith("xlink__") for k in attrs) else None
    if parent is not None:
        el = etree.SubElement(parent, qn, nsmap=local_ns) if local_ns else etree.SubElement(parent, qn)
    else:
        el = etree.Element(qn, nsmap={"tp": TP})
    for k, v in attrs.items():
        k = k.replace("__", ":").replace("_", "-")
        if k.startswith("xlink:"):
            el.set(f"{{{XLINK}}}{k[6:]}", str(v))
        else:
            el.set(k, str(v))
    if text is not None:
        el.text = text
    return el


def taxon_name_el(parent, code, profile):
    genus, rest, reg, qual, status = name_parts(code, profile)
    tn = E("tp:taxon-name", parent)
    E("tp:taxon-name-part", tn, genus, taxon_name_part_type="genus", reg=genus)
    if reg:
        E("tp:taxon-name-part", tn, rest, taxon_name_part_type="species", reg=reg)
    else:
        E("tp:taxon-name-part", tn, rest, taxon_name_part_type="species")
    return tn


def _append(p, last, text):
    """Append text after the last child (or as p.text when there is none)."""
    if not text:
        return
    if last is None:
        p.text = (p.text or "") + text
    else:
        last.tail = (last.tail or "") + text


def p_with_values(parent, text, citations, profile, lead=None):
    """<p> with cited Tier-1 values wrapped as named-content (feature ID in vocab-term)."""
    p = E("p", parent)
    last = None
    if lead:
        last = E("italic", p, lead)
        _append(p, last, " ")
    spans = []
    for c in citations:
        v = str(c.get("value", "")).strip()
        if len(v) < 3 or not re.search(r'\d', v):
            continue
        for m in re.finditer(re.escape(v), text):
            if not any(m.start() < e and m.end() > s for s, e, _ in spans):
                spans.append((m.start(), m.end(), c.get("id", "")))
                break
    spans.sort()
    pos = 0
    for s_, e_, fid in spans:
        _append(p, last, text[pos:s_])
        last = E("named-content", p, text[s_:e_], content_type="dsc:tier1-value",
                 vocab="https://descriptron.org/ontology/", vocab_term=fid or "unknown")
        pos = e_
    _append(p, last, text[pos:])
    return p


def material_citation(parent, rec, profile):
    mc = E("tp:material-citation", parent)
    bits = []

    def nc(term, value, sep="; "):
        if not value:
            return
        el = E("named-content", mc, value, content_type=f"dwc:{term}")
        el.tail = sep
    sexsym = {"male": "♂", "female": "♀"}.get(rec.get("sex", ""), "")
    mc.text = (f"1{sexsym}" if sexsym else "1 specimen") + ": "
    if rec.get("verbatim"):
        nc("verbatimLabel", rec["verbatim"].split(" | ")[0])
    nc("country", rec.get("country"))
    nc("locality", rec.get("locality"))
    nc("verbatimCoordinates", rec.get("verbatim_coordinates"))
    if rec.get("latitude") and rec.get("longitude"):
        nc("decimalLatitude", rec["latitude"], ", ")
        nc("decimalLongitude", rec["longitude"])
    nc("eventDate", rec.get("date"))
    nc("recordedBy", rec.get("collector"))
    nc("samplingProtocol", rec.get("method"))
    if rec.get("host_plant"):
        nc("associatedTaxa", f"host plant: {rec['host_plant']}")
    nc("sex", rec.get("sex"))
    nc("fieldNumber", rec.get("field_code"))
    nc("catalogNumber", rec["specimen_id"], ".")
    return mc


def treatment_el(parent, code, t, profile, recs, flags, plates, key, standalone):
    tt = E("tp:taxon-treatment", parent, id=f"treatment-{re.sub(r'[^A-Za-z0-9_-]', '_', code)}")
    meta = E("tp:treatment-meta", tt)
    kg = E("kwd-group", meta)
    E("label", kg, "Taxon classification")
    for rank, val in classification(profile).items():
        k = E("kwd", kg)
        E("named-content", k, val, content_type=rank)
    nom = E("tp:nomenclature", tt)
    taxon_name_el(nom, code, profile)
    status = pol.species_status(code, profile)
    if status == "undescribed":
        E("tp:taxon-status", nom, "undescribed (working name; no nomenclatural act)")
    elif status == "cf":
        E("tp:taxon-status", nom, "identification uncertain (cf.)")
    cites = t.get("citations", [])
    # materials
    ms = E("tp:treatment-sec", tt, sec_type="materials examined")
    E("title", ms, "Material examined")
    mine = sorted((r for r in recs.values() if r["species"] == code), key=lambda r: natural_key(r["specimen_id"]))
    if status == "undescribed":
        tm = pol.type_material(code) if hasattr(pol, "type_material") else ""
        E("p", ms, tm if tm else
          "Proposed type material (to be confirmed by the authors; holotype not yet designated).")
    for r in mine:
        p = E("p", ms)
        material_citation(p, r, profile)
    for sec_type, title, text in (("diagnosis", "Diagnosis", t.get("diagnosis", "")),):
        s = E("tp:treatment-sec", tt, sec_type=sec_type)
        E("title", s, title)
        p_with_values(s, text, cites, profile)
    ds = E("tp:treatment-sec", tt, sec_type="description")
    E("title", ds, "Description")
    for d in t.get("description", []) or []:
        if d.get("text", "").strip():
            p_with_values(ds, d["text"].strip(), cites, profile, lead=f"{d.get('section', '')}.")
    if t.get("sexual_dimorphism", "").strip():
        s = E("tp:treatment-sec", tt, sec_type="description")
        E("title", s, "Sexual dimorphism")
        p_with_values(s, t["sexual_dimorphism"], cites, profile)
    hosts = sorted({r.get("host_plant") for r in mine if r.get("host_plant")})
    if hosts:
        s = E("tp:treatment-sec", tt, sec_type="biology_ecology")
        E("title", s, "Host plant")
        E("p", s, "; ".join(hosts) + " (slide labels).")
    countries = sorted({r.get("country") for r in mine if r.get("country")})
    s = E("tp:treatment-sec", tt, sec_type="distribution")
    E("title", s, "Distribution")
    E("p", s, (", ".join(countries) + ".") if countries else "No locality data supplied.")
    s = E("tp:treatment-sec", tt, sec_type="remarks")
    E("title", s, "Remarks")
    p_with_values(s, t.get("remarks", "") or "—", cites, profile)
    kinds = defaultdict(int)
    for f in flags:
        kinds[f"{f['kind']}: {f['status'].replace('_', ' ')}"] += 1
    s = E("tp:treatment-sec", tt, sec_type="notes")
    E("title", s, "Provenance")
    E("p", s, (f"Generated with Descriptron BioRAG v2 (evidence-tiered; {len(cites)} numbers traced to the "
               f"data matrix). Qualitative words read from images: "
               + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + "."))
    refs = [f"{r['number']}{sd.lower()}" for r in key["couplets"] for sd in ("A", "B") if r[f"{sd}_goto"] == code]
    if refs:
        E("p", s, f"Identification key: couplet {', '.join(refs)}.")
    if code in plates:
        fig = E("fig", s, id=f"F{plates[code][0]}", position="float")
        E("label", fig, f"Figure {plates[code][0]}.")
        cap = E("caption", fig)
        E("p", cap, "Specimen images with annotated structures used for measurement.")
        E("graphic", fig, xlink__href=f"plates/{plates[code][1]}", position="float")
    return tt


def validate_dtd(root_el):
    if not TAXPUB_DTD.exists():
        return None, ["TaxPub DTD not found"]
    dtd = etree.DTD(str(TAXPUB_DTD))
    ok = dtd.validate(root_el)
    return ok, [str(e) for e in dtd.error_log.filter_from_errors()][:25]


def write_xml(root, path, doctype):
    s = etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True, doctype=doctype)
    path.write_bytes(s)


def key_sec(parent, key, profile, title):
    sec = E("sec", parent, sec_type="key")
    E("title", sec, title)
    tw = E("table-wrap", sec, id="T_key", position="anchor")
    E("label", tw, "Key")
    tab = E("table", tw)
    tb = E("tbody", tab)
    for r in key["couplets"]:
        for sd in ("A", "B"):
            tr = E("tr", tb)
            E("td", tr, f"{r['number']}{sd.lower()}")
            E("td", tr, r[f"{sd}_text"])
            g = r[f"{sd}_goto"]
            E("td", tr, str(g) if isinstance(g, int) else pol.species_display_name(g, profile))
    return sec


# ─────────────────────────────────────────────────────────────────────────────
# Darwin Core Archives
# ─────────────────────────────────────────────────────────────────────────────

def meta_xml(core, core_row, core_terms, core_id, exts):
    root = etree.Element("archive", nsmap={None: "http://rs.tdwg.org/dwc/text/"},
                         metadata="eml.xml")

    def file_el(tag, row_type, location, terms, id_tag, id_idx=0):
        f = etree.SubElement(root, tag, encoding="UTF-8", fieldsTerminatedBy="\\t",
                             linesTerminatedBy="\\n", fieldsEnclosedBy="", ignoreHeaderLines="1",
                             rowType=row_type)
        loc = etree.SubElement(etree.SubElement(f, "files"), "location")
        loc.text = location
        etree.SubElement(f, id_tag, index=str(id_idx))
        for i, t in enumerate(terms):
            if i == id_idx and id_tag == "coreid":
                continue
            etree.SubElement(f, "field", index=str(i), term=t)
    file_el("core", core_row, core, core_terms, "id")
    for loc, row, terms in exts:
        file_el("extension", row, loc, terms, "coreid")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)


def eml_xml(title, authors, abstract, version):
    eml = etree.Element("{https://eml.ecoinformatics.org/eml-2.2.0}eml",
                        nsmap={"eml": "https://eml.ecoinformatics.org/eml-2.2.0"},
                        packageId=f"descriptron-biorag-{version}", system="descriptron")
    ds = etree.SubElement(eml, "dataset")
    etree.SubElement(ds, "title").text = title
    for au in [a.strip() for a in re.split(r",|&| and ", authors or "") if a.strip()]:
        c = etree.SubElement(ds, "creator")
        ind = etree.SubElement(c, "individualName")
        bits = au.split()
        if len(bits) > 1:
            etree.SubElement(ind, "givenName").text = " ".join(bits[:-1])
        etree.SubElement(ind, "surName").text = bits[-1]
    etree.SubElement(ds, "pubDate").text = date.today().isoformat()
    etree.SubElement(ds, "language").text = "eng"
    ab = etree.SubElement(ds, "abstract")
    etree.SubElement(ab, "para").text = abstract
    ir = etree.SubElement(ds, "intellectualRights")
    etree.SubElement(ir, "para").text = "Licence to be set by the authors before publication (e.g. CC BY 4.0)."
    etree.SubElement(ds, "contact").append(etree.fromstring("<positionName>Corresponding author (to be set)</positionName>"))
    return etree.tostring(eml, xml_declaration=True, encoding="UTF-8", pretty_print=True)


def tsv_bytes(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
    w.writerow(header)
    for r in rows:
        w.writerow([str(x).replace("\t", " ").replace("\n", " ") if x is not None else "" for x in r])
    return buf.getvalue().encode("utf-8")


def dwca_checklist(path, order, treats, recs, profile, title, authors, version):
    cls = classification(profile)
    tax_terms = [DWC + t for t in ("taxonID", "scientificName", "genus", "specificEpithet", "taxonRank",
                                   "family", "order", "kingdom", "taxonomicStatus", "taxonRemarks")]
    rows, desc, dist = [], [], []
    for code in order:
        genus, rest, reg, qual, status = name_parts(code, profile)
        rows.append([code, pol.species_display_name(code, profile), genus, reg, "species",
                     cls.get("family", ""), cls.get("order", ""), cls.get("kingdom", ""),
                     "accepted" if status == "described" else "doubtful" if status == "cf" else "provisional",
                     {"undescribed": "undescribed morphospecies (working name)",
                      "cf": "identification uncertain (cf.)"}.get(status, "")])
        t = treats.get(code, {})
        src = "Descriptron BioRAG v2 treatment"
        for typ, text in (("diagnosis", t.get("diagnosis", "")),
                          ("description", " ".join(f"{d.get('section')}. {d.get('text')}"
                                                   for d in t.get("description", []) or [])),
                          ("sexual dimorphism", t.get("sexual_dimorphism", "")),
                          ("remarks", t.get("remarks", ""))):
            if text:
                desc.append([code, text, typ, "en", src, authors or "", ""])
        for c in sorted({r.get("country") for r in recs.values() if r["species"] == code and r.get("country")}):
            dist.append([code, c, COUNTRY_CODES.get(c.lower(), ""), "present"])
    desc_terms = [DWC + "taxonID", DC + "description", DC + "type", DC + "language", DC + "source",
                  DC + "creator", DC + "license"]
    dist_terms = [DWC + "taxonID", DWC + "locality", DWC + "countryCode", DWC + "occurrenceStatus"]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("taxon.txt", tsv_bytes([t.split("/")[-1] for t in tax_terms], rows))
        z.writestr("description.txt", tsv_bytes(["taxonID", "description", "type", "language", "source",
                                                 "creator", "license"], desc))
        z.writestr("distribution.txt", tsv_bytes(["taxonID", "locality", "countryCode", "occurrenceStatus"], dist))
        z.writestr("meta.xml", meta_xml("taxon.txt", DWC + "Taxon", tax_terms, 0, [
            ("description.txt", "http://rs.gbif.org/terms/1.0/Description", desc_terms),
            ("distribution.txt", "http://rs.gbif.org/terms/1.0/Distribution", dist_terms)]))
        z.writestr("eml.xml", eml_xml(f"{title} — checklist and treatments", authors,
                                      "Taxa, treatments (diagnosis, description, remarks) and distribution "
                                      "generated with Descriptron BioRAG v2.", version))
    return {"taxa": len(rows), "descriptions": len(desc), "distribution": len(dist)}


def dwca_occurrences(path, recs, long, fdict, profile, title, authors, version):
    occ_terms = [DWC + t for t in ("occurrenceID", "catalogNumber", "basisOfRecord", "scientificName",
                                   "identificationQualifier", "taxonID", "sex", "country", "countryCode",
                                   "locality", "verbatimLabel", "verbatimCoordinates", "decimalLatitude",
                                   "decimalLongitude", "eventDate", "recordedBy", "fieldNumber",
                                   "samplingProtocol", "associatedTaxa", "occurrenceRemarks")]
    rows = []
    for sid in sorted(recs, key=natural_key):
        r = recs[sid]
        code = r["species"]
        genus, rest, reg, qual, status = name_parts(code, profile)
        sci = f"{genus} {reg}".strip() if reg else pol.species_display_name(code, profile)
        remarks = "; ".join(x for x in (r.get("discrepancy_notes", ""),
                                        "" if r.get("in_matrix") else "no images in the morphometric matrix")
                            if x)
        rows.append([f"descriptron:{sid}", sid, "PreservedSpecimen", sci,
                     f"{qual} {reg}".strip() if qual else "", code, r.get("sex", ""), r.get("country", ""),
                     COUNTRY_CODES.get(r.get("country", "").lower(), ""), r.get("locality", ""),
                     r.get("verbatim", "").split(" | ")[0], r.get("verbatim_coordinates", ""),
                     r.get("latitude", ""), r.get("longitude", ""), r.get("date", ""), r.get("collector", ""),
                     r.get("field_code", ""), r.get("method", ""),
                     f"host plant: {r['host_plant']}" if r.get("host_plant") else "", remarks])
    fd = fdict.set_index("feature_id")
    mof = []
    for i, x in enumerate(long.itertuples()):
        m = fd.loc[x.feature_id] if x.feature_id in fd.index else None
        unit = m["unit"] if m is not None and isinstance(m["unit"], str) else ""
        label = m["label"] if m is not None else x.feature_id
        method = (m["definition"] if m is not None else "") + \
            (f"; {m['conversion']}" if m is not None and isinstance(m["conversion"], str) and m["conversion"] else "")
        mof.append([f"descriptron:{x.specimen_id}", f"descriptron:{x.specimen_id}:{x.feature_id}", label,
                    f"{x.value:.6g}", unit, method, "Descriptron BioRAG v2 (automated image measurement)",
                    f"feature_id={x.feature_id}; tier={x.tier}; images={x.n_images}"])
    mof_terms = [DWC + t for t in ("occurrenceID", "measurementID", "measurementType", "measurementValue",
                                   "measurementUnit", "measurementMethod", "measurementDeterminedBy",
                                   "measurementRemarks")]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("occurrence.txt", tsv_bytes([t.split("/")[-1] for t in occ_terms], rows))
        z.writestr("measurementorfact.txt", tsv_bytes([t.split("/")[-1] for t in mof_terms], mof))
        z.writestr("meta.xml", meta_xml("occurrence.txt", DWC + "Occurrence", occ_terms, 0,
                                        [("measurementorfact.txt", DWC + "MeasurementOrFact", mof_terms)]))
        z.writestr("eml.xml", eml_xml(f"{title} — specimens and measurements", authors,
                                      "Specimens examined (material citations) and all hand-checkable "
                                      "(Tier-1) measurements, proportions and colour values per specimen.",
                                      version))
    return {"occurrences": len(rows), "measurements": len(mof)}


def check_dwca(path):
    problems = []
    with zipfile.ZipFile(path) as z:
        meta = etree.fromstring(z.read("meta.xml"))
        ns = {"d": "http://rs.tdwg.org/dwc/text/"}
        for part in meta.xpath("d:core|d:extension", namespaces=ns):
            loc = part.xpath("d:files/d:location", namespaces=ns)[0].text
            lines = z.read(loc).decode("utf-8").rstrip("\n").split("\n")
            ncol = len(lines[0].split("\t"))
            idx = [int(f.get("index")) for f in part.xpath("d:field|d:id|d:coreid", namespaces=ns)]
            if max(idx) >= ncol:
                problems.append(f"{loc}: meta index {max(idx)} >= {ncol} columns")
            bad = [i for i, ln in enumerate(lines[1:], 2) if len(ln.split("\t")) != ncol]
            if bad:
                problems.append(f"{loc}: {len(bad)} rows with wrong column count (e.g. line {bad[0]})")
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# SDD
# ─────────────────────────────────────────────────────────────────────────────

def sdd_xml(path, order, treats, key, summ, fdict, profile, title, version):
    U = lambda tag, parent=None, **at: (etree.SubElement(parent, f"{{{UBIF}}}{tag}", **at)  # noqa: E731
                                        if parent is not None else
                                        etree.Element(f"{{{UBIF}}}{tag}", nsmap={None: UBIF}, **at))

    def rep(parent, label, detail=None):
        r = U("Representation", parent)
        U("Label", r).text = label
        if detail:
            U("Detail", r).text = detail

    sid = lambda x: re.sub(r'[^A-Za-z0-9_.-]', '_', str(x))  # noqa: E731
    root = U("Datasets")
    tm = U("TechnicalMetadata", root, created=datetime.now().isoformat(timespec="seconds"))
    U("Generator", tm, name="Descriptron build_monograph_exports", version=EXPORT_VERSION)
    ds = U("Dataset", root)
    ds.set("{http://www.w3.org/XML/1998/namespace}lang", "en")
    rep(ds, title, f"Descriptron BioRAG v2 monograph export, version {version}. Characters are the "
                   f"hand-checkable (Tier-1) features of the data matrix.")
    tn = U("TaxonNames", ds)
    for c in order:
        rep(U("TaxonName", tn, id=f"t_{sid(c)}"), pol.species_display_name(c, profile))
    fd = fdict[fdict["tier"].isin(pol.TIER1)].sort_values(["section", "category", "feature_id"])
    chars = U("Characters", ds)
    cid = {}
    for i, r in enumerate(fd.itertuples(), 1):
        cid[r.feature_id] = f"c{i}"
        qc = U("QuantitativeCharacter", chars, id=f"c{i}")
        rep(qc, r.label, f"{r.definition} [feature_id {r.feature_id}; tier {r.tier}]")
        if isinstance(r.unit, str) and r.unit:
            mu = U("MeasurementUnit", qc)
            U("Label", mu, role="Abbrev").text = r.unit
    nls = U("NaturalLanguageDescriptions", ds)
    for c in order:
        t = treats.get(c, {})
        nl = U("NaturalLanguageDescription", nls, id=f"nl_{sid(c)}")
        rep(nl, f"Treatment of {pol.species_display_name(c, profile)}")
        sc = U("Scope", nl)
        U("TaxonName", sc, ref=f"t_{sid(c)}")
        data = U("NaturalLanguageData", nl)
        parts = [("Diagnosis", t.get("diagnosis", ""))] + \
            [(f"Description — {d.get('section')}", d.get("text", "")) for d in t.get("description", []) or []] + \
            [("Sexual dimorphism", t.get("sexual_dimorphism", "")), ("Remarks", t.get("remarks", ""))]
        for head, text in parts:
            if text:
                U("Text", data).text = f"{head}. {text} "
    cds = U("CodedDescriptions", ds)
    s = summ[summ["feature_id"].isin(cid)]
    for c in order:
        cd = U("CodedDescription", cds, id=f"d_{sid(c)}")
        rep(cd, pol.species_display_name(c, profile))
        sc = U("Scope", cd)
        U("TaxonName", sc, ref=f"t_{sid(c)}")
        sdat = U("SummaryData", cd)
        for r in s[s["species"] == c].itertuples():
            q = U("Quantitative", sdat, ref=cid[r.feature_id])
            for typ, val in (("Min", r.min), ("Max", r.max), ("Mean", r.mean), ("SD", r.sd), ("N", r.n)):
                if val == val:
                    U("Measure", q, type=typ, value=str(int(val)) if typ == "N" else repr(float(val)))
    iks = U("IdentificationKeys", ds)
    ik = U("IdentificationKey", iks, id="key1")
    rep(ik, key["meta"].get("title", "Identification key"))
    leads = U("Leads", ik)
    parent_of = {}
    for r in key["couplets"]:
        for sd in ("A", "B"):
            if isinstance(r[f"{sd}_goto"], int):
                parent_of[r[f"{sd}_goto"]] = f"L{r['number']}{sd.lower()}"
    for r in key["couplets"]:
        for sd in ("A", "B"):
            ld = U("Lead", leads, id=f"L{r['number']}{sd.lower()}")
            if r["number"] in parent_of:
                U("Parent", ld, ref=parent_of[r["number"]])
            U("Statement", ld).text = r[f"{sd}_text"]
            g = r[f"{sd}_goto"]
            if not isinstance(g, int):
                U("TaxonName", ld, ref=f"t_{sid(g)}")
    etree.ElementTree(root).write(str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True)
    if SDD_XSD.exists():
        schema = etree.XMLSchema(etree.parse(str(SDD_XSD)))
        ok = schema.validate(etree.parse(str(path)))
        return ok, [str(e) for e in schema.error_log][:20], len(cid)
    return None, ["SDD schema not found"], len(cid)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown + JSON-LD + README
# ─────────────────────────────────────────────────────────────────────────────

def markdown(path, title, authors, order, treats, key, recs, profile, plates):
    L = [f"# {title}", "", f"*{authors}*" if authors else "", "",
         f"_Draft generated {date.today().isoformat()} with Descriptron BioRAG v2._", "",
         "## Key to the species", ""]
    L += [ln for ln in Path(key["_md_path"]).read_text().splitlines()[2:]] if key.get("_md_path") else []
    for c in order:
        t = treats.get(c, {})
        L += ["", f"## {pol.species_display_name(c, profile)}", ""]
        if c in plates:
            L.append(f"Figure {plates[c][0]} — `plates/{plates[c][1]}`")
        L += ["", "**Material examined.**"]
        for r in sorted((r for r in recs.values() if r["species"] == c), key=lambda r: natural_key(r["specimen_id"])):
            sx = {"male": "♂", "female": "♀"}.get(r.get("sex", ""), "")
            bits = [f'"{r["verbatim"].split(" | ")[0]}"' if r.get("verbatim") else "", r.get("country", ""),
                    r.get("date", ""), f"leg. {r['collector']}" if r.get("collector") else ""]
            L.append(f"- 1{sx} ({r['specimen_id']}): " + "; ".join(b for b in bits if b))
        L += ["", f"**Diagnosis.** {t.get('diagnosis', '')}", "", "**Description.**", ""]
        L += [f"*{d.get('section')}.* {d.get('text')}" + "\n" for d in t.get("description", []) or []]
        if t.get("sexual_dimorphism"):
            L += [f"**Sexual dimorphism.** {t['sexual_dimorphism']}", ""]
        L += [f"**Remarks.** {t.get('remarks', '')}", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


def treatment_jsonld(code, t, profile, recs, flags, plates, key, desc_dir):
    """Complete JSON-LD treatment: the checker's JSON-LD (text, citations, subjective flags,
    validation) plus nomenclature, material examined, host, distribution, figure and key couplets."""
    base_p = Path(desc_dir) / code / f"{code}_treatment.jsonld"
    doc = json.loads(base_p.read_text()) if base_p.exists() else {
        "@context": {"@vocab": "https://schema.org/", "dsc": "https://descriptron.org/ontology/",
                     "dwc": "http://rs.tdwg.org/dwc/terms/"},
        "@type": "dsc:TaxonomicTreatment", "@id": f"#treatment-{code}",
        "dsc:diagnosis": t.get("diagnosis", ""), "dsc:remarks": {"text": t.get("remarks", "")}}
    doc["@context"]["dwc"] = "http://rs.tdwg.org/dwc/terms/"
    genus, rest, reg, qual, status = name_parts(code, profile)
    cls = classification(profile)
    doc["about"] = {"@type": "Taxon", "identifier": code, "name": pol.species_display_name(code, profile),
                    "taxonRank": "species", "dwc:genus": genus, "dwc:specificEpithet": reg or None,
                    "dwc:identificationQualifier": qual or None, "dwc:family": cls.get("family"),
                    "dwc:order": cls.get("order"), "dsc:status": status,
                    "dsc:nomenclaturalNote": {"undescribed": "undescribed (working name; no nomenclatural act)",
                                              "cf": "identification uncertain (cf.)"}.get(status, "")}
    mine = sorted((r for r in recs.values() if r["species"] == code), key=lambda r: natural_key(r["specimen_id"]))
    doc["dsc:materialsExamined"] = [{
        "@type": "dwc:Occurrence", "dwc:catalogNumber": r["specimen_id"], "dwc:basisOfRecord": "PreservedSpecimen",
        "dwc:sex": r.get("sex") or None, "dwc:verbatimLabel": (r.get("verbatim") or "").split(" | ")[0] or None,
        "dwc:country": r.get("country") or None, "dwc:locality": r.get("locality") or None,
        "dwc:verbatimCoordinates": r.get("verbatim_coordinates") or None,
        "dwc:decimalLatitude": r.get("latitude") or None, "dwc:decimalLongitude": r.get("longitude") or None,
        "dwc:eventDate": r.get("date") or None, "dwc:recordedBy": r.get("collector") or None,
        "dwc:fieldNumber": r.get("field_code") or None, "dwc:samplingProtocol": r.get("method") or None,
        "dwc:associatedTaxa": f"host plant: {r['host_plant']}" if r.get("host_plant") else None,
        "dwc:occurrenceRemarks": r.get("discrepancy_notes") or None,
        "dsc:inMorphometricMatrix": bool(r.get("in_matrix")),
        "dsc:typeStatusProposed": "proposed type material (not designated)" if status == "undescribed" else None}
        for r in mine]
    doc["dsc:hostPlant"] = sorted({r["host_plant"] for r in mine if r.get("host_plant")})
    doc["dsc:distribution"] = sorted({r["country"] for r in mine if r.get("country")})
    doc["dsc:keyCouplets"] = [f"{r['number']}{sd.lower()}" for r in key["couplets"] for sd in ("A", "B")
                              if r[f"{sd}_goto"] == code]
    if code in plates:
        doc["image"] = {"@type": "ImageObject", "name": f"Figure {plates[code][0]}",
                        "contentUrl": f"plates/{plates[code][1]}"}
    if "dsc:qualitativeObservations" not in doc:
        doc["dsc:qualitativeObservations"] = flags
    onto_p = Path(desc_dir) / code / f"{code}_ontology.jsonld"
    if onto_p.exists():                       # entity-quality annotations (biorag_ontology_annotator_v2.py)
        o = json.loads(onto_p.read_text())
        doc["@context"]["obo"] = "http://purl.obolibrary.org/obo/"
        doc["dsc:ontologies"] = o.get("dsc:ontologies")
        doc["dsc:ontologyAnnotations"] = o.get("dsc:annotations")
    audit_p = Path(desc_dir) / "confabulation_report_v2" / "per_species" / f"{code}.json"
    if audit_p.exists():
        a = json.loads(audit_p.read_text())
        errs = [r for r in a.get("records", []) if r.get("status") in ("error", "review")]
        doc["dsc:independentAudit"] = {
            "dsc:checker": "biorag_confabulation_checker_v2",
            "dsc:claimsChecked": sum(1 for r in a.get("records", [])
                                     if r.get("kind") in ("measurement", "comparison", "remarks")),
            "dsc:valuesChecked": a.get("numbers_checked"),
            "dsc:errors": [{"dsc:type": r.get("type"), "dsc:feature": r.get("feature"),
                            "dsc:note": r.get("note"), "dsc:text": r.get("context")} for r in errs]}
    return doc


README = """# Machine-readable monograph ({title})

Generated {date} by `build_monograph_exports_v1.py` (Descriptron BioRAG v2), version {version}.

| File | Standard | Content |
|---|---|---|
| `taxpub/<code>.taxpub.xml` | Plazi TaxPub (treatment deposit, root `tp:taxon-treatment`) | one treatment per species: nomenclature, material citations with Darwin Core fields, diagnosis, description, sexual dimorphism, host plant, distribution, remarks, provenance, figure |
| `taxpub/{prefix}_monograph.taxpub.xml` | JATS + TaxPub article | key (table) + all treatments |
| `dwca_checklist.zip` | Darwin Core Archive (Taxon core; GBIF Description and Distribution extensions) | taxa, treatment texts, countries |
| `dwca_occurrences.zip` | Darwin Core Archive (Occurrence core; MeasurementOrFact extension) | every specimen (verbatim label, sex, locality, host) and every hand-checkable measurement, ratio and colour value with its feature ID, unit and method |
| `{prefix}_treatments.sdd.xml` | TDWG SDD 1.1 | natural-language treatments, coded descriptions (min/max/mean/SD/n) for all Tier-1 characters, identification key |
| `key/taxonomic_key.sdd.xml` | TDWG SDD 1.1 | the key with its characters only |
| `key/taxonomic_key.jsonld`, `.md`, `.txt`, `character_matrix.tsv` | JSON-LD / text | the key (structured and readable) |
| `jsonld/<code>.jsonld` | schema.org + Darwin Core JSON-LD | one complete treatment per species: nomenclature, material examined, diagnosis, description, sexual dimorphism, remarks, host, distribution, figure, key couplets, numeric citations (feature IDs), subjective-character flags, entity-quality ontology annotations (PATO + AISM/UBERON, with the release versions), the independent audit's result, validation |
| `{prefix}_monograph.jsonld` | schema.org + Darwin Core JSON-LD | self-contained monograph: all treatments and the key embedded, plus links to the other files |
| `{prefix}_monograph.md` | Markdown | readable version |
| `export_validation_report.json` | — | TaxPub DTD and SDD XSD validation results, counts |

## Traceability
Every number in a treatment that comes from the data matrix is wrapped in the TaxPub files as
`<named-content content-type="dsc:tier1-value" vocab-term="FEATURE_ID">`. The same FEATURE_ID is
in `measurementRemarks` of `dwca_occurrences.zip` and in the SDD character details, so text,
specimens and measurements can be joined. Words read from images (colour pattern, texture,
shape, setation) are counted in each treatment's Provenance section and listed per mention in
`jsonld/<code>.jsonld` (and `descriptions/subjective_character_flags.tsv` in the run folder).

## Updating with new material or observations
1. Add the new specimens: images and COCO annotations, a group-label row, and a row in the
   localities TSV (leave unknown fields blank).
2. Re-run `run_full_pipeline_v2.py` with the same arguments. The measurements, Tier-1 matrix,
   key and treatments are rebuilt from all specimens, then these exports are regenerated.
3. Give the new export a new `--dataset_version`. Published TaxPub treatments are immutable
   records; an update is published as a new treatment that cites the earlier one (add the
   earlier Plazi treatment identifier if one exists).
4. Corrections to a single measurement go into the annotations or the exclusion list, never
   into the text. Corrections to subjective words can be made in the treatment JSON and
   recorded by changing the flag status.

## Not asserted
No nomenclatural acts and no type designations are asserted. Undescribed species carry
"undescribed (working name)". Licence and contact details must be set before deposition.
"""


def main():
    ap = argparse.ArgumentParser(description="Standards-based machine-readable monograph exports")
    ap.add_argument("--descriptions_dir", required=True)
    ap.add_argument("--key_dir", required=True)
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon-profile", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--localities", default=None)
    ap.add_argument("--plates_dir", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--authors", default="")
    ap.add_argument("--dataset_version", default=date.today().isoformat())
    ap.add_argument("--prefix", default=None)
    a = ap.parse_args()
    # type designations, if the collaborators have returned them; otherwise the placeholder stands
    n_types = pol.load_type_material(a.descriptions_dir) if hasattr(pol, "load_type_material") else 0
    if n_types:
        print(f"type material: {n_types} species have a designated type statement")

    prof, key, order, treats, flags, long, fdict, summ, loc, plates = load_all(a)
    genus = prof.get("taxon", {}).get("genus", "taxa")
    prefix = a.prefix or genus.lower() or "monograph"
    title = a.title or f"{genus} — species treatments and identification key"
    out = Path(a.output_dir)
    (out / "taxpub").mkdir(parents=True, exist_ok=True)
    recs = specimens(long, loc)
    report = {"generated": datetime.now().isoformat(), "export_version": EXPORT_VERSION,
              "dataset_version": a.dataset_version, "n_species": len(order),
              "n_treatments": len(treats), "n_specimens": len(recs)}

    doctype_tt = f'<!DOCTYPE tp:taxon-treatment SYSTEM "{TAXPUB_DTD.name}">'
    tp_ok = []
    for c in order:
        if c not in treats:
            continue
        root = treatment_el(None, c, treats[c], prof, recs, flags.get(c, []), plates, key, True)
        ok, errs = validate_dtd(root)
        tp_ok.append((c, ok, errs[:3]))
        write_xml(root, out / "taxpub" / f"{c}.taxpub.xml", doctype_tt)
    report["taxpub_treatments_valid"] = sum(1 for _, ok, _ in tp_ok if ok)
    report["taxpub_treatment_errors"] = {c: e for c, ok, e in tp_ok if not ok}

    art = etree.Element("article", nsmap={"tp": TP, "xlink": XLINK})
    art.set("article-type", "research-article")
    front = E("front", art)
    jm = E("journal-meta", front)
    E("journal-id", jm, "TO-BE-ASSIGNED", journal_id_type="publisher-id")
    E("issn", jm, "0000-0000")          # placeholder: replaced by the publisher's JATS
    am = E("article-meta", front)
    tg = E("title-group", am)
    E("article-title", tg, title)
    if a.authors:
        cg = E("contrib-group", am)
        for au in [x.strip() for x in re.split(r",|&| and ", a.authors) if x.strip()]:
            ct = E("contrib", cg, contrib_type="author")
            nm = E("name", ct)
            bits = au.split()
            E("surname", nm, bits[-1])
            if len(bits) > 1:
                E("given-names", nm, " ".join(bits[:-1]))
    pd_ = E("pub-date", am, date_type="pub", publication_format="electronic")
    E("year", pd_, str(date.today().year))
    body = E("body", art)
    key_sec(body, key, prof, key["meta"].get("title", "Key"))
    for c in order:
        if c in treats:
            treatment_el(body, c, treats[c], prof, recs, flags.get(c, []), plates, key, False)
    ok, errs = validate_dtd(art)
    report["taxpub_article_valid"] = ok
    report["taxpub_article_errors"] = errs[:10]
    write_xml(art, out / "taxpub" / f"{prefix}_monograph.taxpub.xml",
              f'<!DOCTYPE article SYSTEM "{TAXPUB_DTD.name}">')
    # copy the DTD next to the files so they validate stand-alone
    (out / "taxpub" / TAXPUB_DTD.name).write_bytes(TAXPUB_DTD.read_bytes())

    report["dwca_checklist"] = dwca_checklist(out / "dwca_checklist.zip", order, treats, recs, prof,
                                              title, a.authors, a.dataset_version)
    report["dwca_occurrences"] = dwca_occurrences(out / "dwca_occurrences.zip", recs, long, fdict, prof,
                                                  title, a.authors, a.dataset_version)
    report["dwca_structure_problems"] = check_dwca(out / "dwca_checklist.zip") + \
        check_dwca(out / "dwca_occurrences.zip")

    ok, errs, nchar = sdd_xml(out / f"{prefix}_treatments.sdd.xml", order, treats, key, summ, fdict, prof,
                              title, a.dataset_version)
    report.update({"sdd_valid": ok, "sdd_errors": errs, "sdd_characters": nchar})

    key["_md_path"] = str(Path(a.key_dir) / "taxonomic_key.md")
    markdown(out / f"{prefix}_monograph.md", title, a.authors, order, treats, key, recs, prof, plates)
    (out / "jsonld").mkdir(exist_ok=True)
    tj_docs = []
    for c in order:
        if c in treats:
            d = treatment_jsonld(c, treats[c], prof, recs, flags.get(c, []), plates, key, a.descriptions_dir)
            (out / "jsonld" / f"{c}.jsonld").write_text(json.dumps(d, indent=1, ensure_ascii=False))
            tj_docs.append(d)
    (out / "key").mkdir(exist_ok=True)
    key_jl = {}
    for fn in ("taxonomic_key.jsonld", "taxonomic_key.sdd.xml", "taxonomic_key.md", "taxonomic_key.txt",
               "character_matrix.tsv", "key_validation_report.txt"):
        src = Path(a.key_dir) / fn
        if src.exists():
            (out / "key" / fn).write_bytes(src.read_bytes())
    if (Path(a.key_dir) / "taxonomic_key.jsonld").exists():
        key_jl = json.loads((Path(a.key_dir) / "taxonomic_key.jsonld").read_text())
        key_jl.pop("@context", None)
    report["jsonld_treatments"] = len(tj_docs)
    jl = {"@context": {"@vocab": "https://schema.org/", "dsc": "https://descriptron.org/ontology/",
                       "dwc": "http://rs.tdwg.org/dwc/terms/"},
          "@type": ["Dataset", "dsc:Monograph"], "name": title, "version": a.dataset_version,
          "dateModified": date.today().isoformat(),
          "creator": [{"@type": "Person", "name": x.strip()} for x in re.split(r",|&| and ", a.authors or "") if x.strip()],
          "license": "to be set by the authors",
          "about": {"@type": "Taxon", "name": genus, "taxonRank": "genus", **{f"dwc:{k}": v for k, v in classification(prof).items()}},
          "dsc:identificationKey": key_jl,
          "hasPart": [{k: v for k, v in d.items() if k != "@context"} for d in tj_docs],
          "distribution": [
              {"@type": "DataDownload", "contentUrl": f"{prefix}_treatments.sdd.xml", "encodingFormat": "application/xml", "description": "TDWG SDD 1.1: treatments, character matrix, key"},
              {"@type": "DataDownload", "contentUrl": "key/taxonomic_key.sdd.xml", "encodingFormat": "application/xml", "description": "TDWG SDD 1.1: key and key characters"},
              {"@type": "DataDownload", "contentUrl": "key/taxonomic_key.jsonld", "encodingFormat": "application/ld+json", "description": "identification key"},
              {"@type": "DataDownload", "contentUrl": "dwca_checklist.zip", "encodingFormat": "application/zip", "description": "Darwin Core Archive: taxa, descriptions, distribution"},
              {"@type": "DataDownload", "contentUrl": "dwca_occurrences.zip", "encodingFormat": "application/zip", "description": "Darwin Core Archive: specimens + MeasurementOrFact"},
              {"@type": "DataDownload", "contentUrl": f"taxpub/{prefix}_monograph.taxpub.xml", "encodingFormat": "application/xml", "description": "Plazi TaxPub article"},
              {"@type": "DataDownload", "contentUrl": "taxpub/", "encodingFormat": "application/xml", "description": "Plazi TaxPub treatment deposits (one per species)"},
              {"@type": "DataDownload", "contentUrl": "jsonld/", "encodingFormat": "application/ld+json", "description": "JSON-LD treatments (one per species)"}],
          "dsc:validation": report}
    (out / f"{prefix}_monograph.jsonld").write_text(json.dumps(jl, indent=1, ensure_ascii=False))
    (out / "README_machine_readable.md").write_text(
        README.format(title=title, date=date.today().isoformat(), version=a.dataset_version, prefix=prefix))
    (out / "export_validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report.get(k) for k in ("n_species", "n_treatments", "n_specimens", "jsonld_treatments",
                                             "taxpub_treatments_valid", "taxpub_article_valid",
                                             "dwca_checklist", "dwca_occurrences", "dwca_structure_problems",
                                             "sdd_valid", "sdd_characters")}, indent=1))
    if report["taxpub_treatment_errors"]:
        print("TaxPub errors (first):", json.dumps(dict(list(report["taxpub_treatment_errors"].items())[:2]), indent=1))
    if not report["taxpub_article_valid"]:
        print("TaxPub article errors:", report["taxpub_article_errors"][:5])
    if not report["sdd_valid"]:
        print("SDD errors:", report["sdd_errors"][:5])


if __name__ == "__main__":
    main()

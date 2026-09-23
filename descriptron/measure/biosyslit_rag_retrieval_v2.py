#!/usr/bin/env python3
"""
biosyslit_rag_retrieval_v2.py — BioRAG v2 (evidence-tiered, backend-agnostic)
============================================================================

Thin wrapper around biosyslit_rag_retrieval.py (v1 is NOT modified), in the
same style as biosyslit_rag_retrieval_prompt_constrained.py. It changes:

  1. LLM backend: --llm-backend api | claude-code
       api          Anthropic API key (as before)
       claude-code  Claude Code subscription via `claude -p` (no API credit);
                    images are passed as base64 content blocks
  2. Prompts come from files instead of code:
       --system-prompts  universal, taxon-agnostic (default
                         biorag_prompts/biorag_system_prompts_v1.txt)
       --taxon-profile   organism-specific (names, structure terms, ratio
                         definitions, landmark sets, questions)
  3. Evidence tiers (biorag_feature_policy.py): with --matrix-dir (output of
     biorag_key_feature_filter_v2.py) every per-structure data sheet lists
     Tier-1 values (mm, ratios, CIE L*a*b*) with feature IDs, and Tier-2
     statistics in a separate block that the model may use only in REMARKS.
     The JSON schema gains "remarks" and per-trait "feature_id"; the merged
     species record and text output gain a REMARKS section.
  4. Keys are built from the data matrix, not by the LLM:
       describe-biorag --generate-key  -> biorag_key_builder_v1 (if --matrix-dir)
  5. New subcommands:
       build-key            = biorag_key_builder_v1.py
       refine-descriptions  = biorag_description_refiner_v1.py
       filter-matrix        = biorag_key_feature_filter_v2.py

Everything else (RAG index, Florence-2, caching, type-series logic, CLI of
the v1 subcommands) is inherited unchanged.

Examples
--------
  # fresh image-based descriptions on a Claude subscription
  python biosyslit_rag_retrieval_v2.py describe-biorag \
      --llm-backend claude-code \
      --taxon-profile biorag_prompts/taxon_profiles/diaphorina_taxon_profile.yaml \
      --matrix-dir  .../compiled_key_tier \
      --compiled-dir .../compiled --semilandmarks-dir .../semilandmarks \
      --image-dir .../images --coco-json .../all.json --group-labels .../group_labels.csv \
      --output-dir .../biorag_descriptions_v2 --generate-key

  # data-driven key only
  python biosyslit_rag_retrieval_v2.py build-key --matrix_dir ... --output_dir ... \
      --taxon-profile ... --llm-backend claude-code
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import biosyslit_rag_retrieval as base  # noqa: E402
from biosyslit_rag_retrieval import BioragDescriber, ClaudeDescriber  # noqa: E402
import biorag_feature_policy as pol  # noqa: E402
from biorag_llm_backend import load_prompt_library, make_llm_client  # noqa: E402

V2_VERSION = "2.0"

SUBCOMMAND_MODULES = {
    "build-key": "biorag_key_builder_v1",
    "refine-descriptions": "biorag_description_refiner_v1",
    "filter-matrix": "biorag_key_feature_filter_v2",
}

CFG = {"backend": "api", "claude_bin": None, "cc_model": None, "llm_log": None,
       "profile": pol.load_taxon_profile(None), "prompts": None, "matrix": None}


# ─────────────────────────────────────────────────────────────────────────────
# 1. backend
# ─────────────────────────────────────────────────────────────────────────────

def _describer_init(self, model=base.CLAUDE_OPUS, max_tokens=8192, max_retries=3, retry_delay=4.0):
    self.client = make_llm_client(CFG["backend"], claude_bin=CFG["claude_bin"],
                                  cc_model=CFG["cc_model"], log_path=CFG["llm_log"])
    self.model = model
    self.max_tokens = max_tokens
    self.max_retries = max_retries
    self.retry_delay = retry_delay


# ─────────────────────────────────────────────────────────────────────────────
# 2./3. prompts and tiered data sheets
# ─────────────────────────────────────────────────────────────────────────────

def _raw_section_with_context(name: str) -> str:
    """Prompt section with {evidence_policy}/{taxon_context} filled but other
    placeholders ({category}) and doubled braces kept for v1's .format()."""
    lib = CFG["prompts"]
    txt = lib.raw(name)
    esc = lambda t: t.replace("{", "{{").replace("}", "}}")  # noqa: E731
    txt = txt.replace("{evidence_policy}", esc(lib.raw("evidence_policy")))
    txt = txt.replace("{taxon_context}", esc(pol.build_taxon_context(CFG["profile"])))
    return txt


class TierMatrix:
    def __init__(self, matrix_dir):
        import pandas as pd
        md = Path(matrix_dir)
        self.summary = pd.read_csv(md / "species_feature_summary.csv")
        self.fdict = pd.read_csv(md / "feature_dictionary.tsv", sep="\t").set_index("feature_id")


def format_species_diagnosis_v2(compiled, group_label, category):
    """Tier-aware replacement for v1 format_species_diagnosis()."""
    M = CFG["matrix"]
    if M is None:
        return base._v1_format_species_diagnosis(compiled, group_label, category)
    prof = CFG["profile"]
    lines = [f"=== DATA SHEET: {pol.species_display_name(group_label, prof)} — "
             f"{pol.structure_info(category, prof)['term']} ===",
             f"(folder code '{group_label}' is internal: always write the display name above)"]
    cats = [category]
    for sex in ("male", "female"):
        cats.append(pol.sex_split_category(category, sex))
    s = M.summary[(M.summary["species"] == group_label) & (M.summary["category"].isin(cats))]
    lines.append("TIER 1 (usable in diagnosis, description and remarks):")
    for r in s.itertuples():
        if r.feature_id not in M.fdict.index:
            continue
        meta = M.fdict.loc[r.feature_id]
        if meta["tier"] not in pol.TIER1:
            continue
        unit = meta["unit"] if isinstance(meta["unit"], str) else ""
        lo, hi = pol.fmt(r.min, unit, r.median), pol.fmt(r.max, unit, r.median)
        mean = pol.fmt(r.mean, unit, r.median)
        lines.append(f"  [{r.feature_id}] {meta['label']}: {lo}–{hi} {unit} (mean {mean}; n={int(r.n)})")
    # standard proportions measured on the same specimens
    p = M.summary[(M.summary["species"] == group_label) & (M.summary["category"] == "proportions")]
    if len(p):
        lines.append("  Standard proportions of this species (whole specimen):")
        for r in p.itertuples():
            meta = M.fdict.loc[r.feature_id]
            lines.append(f"  [{r.feature_id}] {meta['label']} = {meta['definition']}: "
                         f"{pol.fmt(r.min, 'x')}–{pol.fmt(r.max, 'x')}x (mean {pol.fmt(r.mean, 'x')}; n={int(r.n)})")
    # tier 2: v1 statistics, relabelled
    old = base._v1_format_species_diagnosis(compiled, group_label, category)
    if old:
        lines.append("TIER 2 (statistical — REMARKS ONLY; raw feature names are internal and must not be printed):")
        lines.extend("  " + ln for ln in old.splitlines()[1:] if ln.strip())
    return "\n".join(lines)


def _biorag_schema_v2(self, category, group_label):
    sch = base._v1_biorag_schema(self, category, group_label)
    eo = sch["expected_output"]
    eo["taxon"] = pol.species_display_name(group_label, CFG["profile"])
    eo["diagnosis"] = ("Tier 1 only: hand-measurable or visible characters that separate this species; "
                       "no statistics, no p-values.")
    eo["description"] = ("Tier 1 only: measurements copied from the data sheet as 'range (mean; n)' "
                         "with units, proportions, colour in words, shape, sculpture, setation.")
    eo["remarks"] = "Tier 2 only: statistical results in plain language; empty if none."
    eo["traits"][0]["feature_id"] = "data-sheet ID (e.g. tibia.length_mm) or null for image observations"
    eo["traits"][0].pop("mean_sd", None)
    return sch


def _merge_species_graph_v2(self, category_results, group_label, vision_mode):
    merged = base._v1_merge_species_graph(self, category_results, group_label, vision_mode)
    rem = []
    for res in category_results:
        inner = res.get("expected_output", res)
        if inner.get("remarks"):
            rem.append(f"**{res.get('_category', '')}**: {inner['remarks']}")
    merged["full_remarks"] = "\n\n".join(rem)
    merged["display_name"] = pol.species_display_name(group_label, CFG["profile"])
    merged["method"] = f"BioRAG v{V2_VERSION} (evidence-tiered)"
    return merged


def _write_text_description_v2(self, merged, path):
    base._v1_write_text_description(self, merged, path)
    txt = Path(path).read_text()
    if merged.get("display_name"):
        txt = txt.replace(f"SPECIES DESCRIPTION: {merged.get('taxon')}",
                          f"SPECIES DESCRIPTION: {merged.get('taxon')}  ({merged['display_name']})", 1)
    if merged.get("full_remarks"):
        block = "REMARKS\n" + "-" * 40 + "\n" + merged["full_remarks"] + "\n\n"
        idx = txt.find("MATERIALS EXAMINED")
        txt = txt[:idx] + block + txt[idx:] if idx >= 0 else txt + "\n" + block
    Path(path).write_text(txt)


def _synthesize_diagnosis_v2(self, group_label, per_category_diagnoses, type_series=None):
    cached = self._get_cached_response(group_label, "_synthesis", "diagnosis_v2")
    if cached:
        return cached.get("text", "")
    lib = CFG["prompts"]
    system = lib.get("taxonomist_persona") + "\n\n" + lib.get(
        "synthesis_diagnosis", taxon_context=pol.build_taxon_context(CFG["profile"]))
    user = (f"Species: {pol.species_display_name(group_label, CFG['profile'])}\n\n"
            f"PER-STRUCTURE DIAGNOSTIC STATEMENTS:\n{per_category_diagnoses[:30000]}")
    try:
        r = self.claude.client.messages.create(model=self.claude.model, max_tokens=4000,
                                               system=system,
                                               messages=[{"role": "user", "content": user}])
        text = r.content[0].text.strip() if r.content else ""
        found = pol.find_tier2_terms(text)
        if found:
            base.logger.warning(f"  synthesis diagnosis contains Tier-2 terms {found}; sentences removed")
            text = " ".join(s for s in re.split(r'(?<=[.!?])\s+', text) if not pol.find_tier2_terms(s))
        self._save_cached_response(group_label, "_synthesis", "diagnosis_v2", {"text": text})
        return text
    except Exception as e:  # noqa: BLE001
        base.logger.error(f"  Synthesis diagnosis failed: {e}")
        return ""


def _generate_taxonomic_key_v2(self, all_results, compiled, output_dir):
    """Data-driven key (biorag_key_builder_v1) instead of an LLM-written key."""
    if not CFG.get("matrix_dir"):
        base.logger.warning("No --matrix-dir: falling back to the v1 LLM-written key "
                            "(NOT recommended; run filter-matrix first)")
        return base._v1_generate_taxonomic_key(self, all_results, compiled, output_dir)
    import subprocess
    cmd = [sys.executable, str(HERE / "biorag_key_builder_v1.py"),
           "--matrix_dir", CFG["matrix_dir"], "--output_dir", str(Path(output_dir) / "key_v2"),
           "--model", self.claude.model, "--llm-backend", CFG["backend"]]
    if CFG["profile"].get("_path"):
        cmd += ["--taxon-profile", CFG["profile"]["_path"]]
    if CFG.get("prompts_path"):
        cmd += ["--system-prompts", CFG["prompts_path"]]
    base.logger.info("Building data-driven key: " + " ".join(cmd))
    subprocess.run(cmd, check=False)
    kp = Path(output_dir) / "key_v2" / "taxonomic_key.txt"
    if kp.exists():
        (Path(output_dir) / "taxonomic_key.txt").write_text(kp.read_text())
        return kp.read_text()
    return ""


def install(args_ns):
    CFG.update({"backend": args_ns.llm_backend, "claude_bin": args_ns.claude_bin,
                "cc_model": args_ns.cc_model, "llm_log": args_ns.llm_log,
                "profile": pol.load_taxon_profile(args_ns.taxon_profile),
                "prompts": load_prompt_library(args_ns.system_prompts),
                "prompts_path": args_ns.system_prompts,
                "matrix_dir": args_ns.matrix_dir,
                "matrix": TierMatrix(args_ns.matrix_dir) if args_ns.matrix_dir else None})
    # keep originals
    base._v1_format_species_diagnosis = base.format_species_diagnosis
    base._v1_biorag_schema = BioragDescriber._biorag_schema
    base._v1_merge_species_graph = BioragDescriber._merge_species_graph
    base._v1_write_text_description = BioragDescriber._write_text_description
    base._v1_generate_taxonomic_key = BioragDescriber.generate_taxonomic_key
    # patch
    ClaudeDescriber.__init__ = _describer_init
    base.format_species_diagnosis = format_species_diagnosis_v2
    BioragDescriber.SYSTEM_PROMPT = _raw_section_with_context("describe_category")
    BioragDescriber.DIAGNOSIS_PROMPT = CFG["prompts"].raw("describe_category_user")
    BioragDescriber.DESCRIPTION_PROMPT = CFG["prompts"].raw("describe_category_user")
    BioragDescriber._biorag_schema = _biorag_schema_v2
    BioragDescriber._merge_species_graph = _merge_species_graph_v2
    BioragDescriber._write_text_description = _write_text_description_v2
    BioragDescriber._synthesize_diagnosis = _synthesize_diagnosis_v2
    BioragDescriber.generate_taxonomic_key = _generate_taxonomic_key_v2
    if CFG["profile"].get("_questions_path"):
        # use the profile's questions file when --user-prompts is not given
        if "--user-prompts" not in sys.argv:
            sys.argv += ["--user-prompts", CFG["profile"]["_questions_path"]]
    if args_ns.llm_backend != "api":
        base.load_api_key = lambda *a, **k: None
        try:
            import anthropic
            anthropic.Anthropic = lambda *a, **k: make_llm_client(  # correct-key path
                args_ns.llm_backend, claude_bin=args_ns.claude_bin, cc_model=args_ns.cc_model,
                log_path=args_ns.llm_log)
        except ImportError:
            pass
    base.save_biorag_recipe = _recipe_v2(base.save_biorag_recipe)


def _recipe_v2(orig):
    def wrapped(args, output_dir, n_user_prompts=0):
        orig(args, output_dir, n_user_prompts)
        extra = {"biorag_v2": {"version": V2_VERSION, "llm_backend": CFG["backend"],
                               "system_prompts": str(CFG["prompts"].path),
                               "taxon_profile": CFG["profile"].get("_path"),
                               "matrix_dir": CFG.get("matrix_dir"),
                               "policy_version": pol.POLICY_VERSION}}
        with open(Path(output_dir) / "biorag_recipe_v2.json", "w") as f:
            json.dump(extra, f, indent=2)
    return wrapped


def main():
    if len(sys.argv) > 1 and sys.argv[1] in SUBCOMMAND_MODULES:
        import importlib
        mod = importlib.import_module(SUBCOMMAND_MODULES[sys.argv[1]])
        sys.argv = [f"{sys.argv[0]} {sys.argv[1]}"] + sys.argv[2:]
        return mod.main()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--llm-backend", default="api", choices=["api", "claude-code"])
    pre.add_argument("--claude-bin", default=None)
    pre.add_argument("--cc-model", default=None)
    pre.add_argument("--llm-log", default=None)
    pre.add_argument("--system-prompts", default=None)
    pre.add_argument("--taxon-profile", default=None)
    pre.add_argument("--matrix-dir", default=None)
    ns, rest = pre.parse_known_args(sys.argv[1:])
    if ns.llm_log is None and "--output-dir" in rest:
        ns.llm_log = str(Path(rest[rest.index("--output-dir") + 1]) / "llm_calls.jsonl")
    sys.argv = [sys.argv[0]] + rest
    install(ns)
    if "-h" in rest or "--help" in rest:
        print(__doc__)
    base.main()


if __name__ == "__main__":
    main()

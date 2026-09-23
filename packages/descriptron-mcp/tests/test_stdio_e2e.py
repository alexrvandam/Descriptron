"""
End to end: a real MCP client launches the server over stdio, as Claude does,
and calls every kind of tool on a real monograph directory.

Needs real data, so it runs only when DESCRIPTRON_TEST_MONOGRAPH points at a
BioRAG output directory (compiled_key_tier/, descriptions/, key/), and
DESCRIPTRON_TEST_PROFILE at its taxon profile YAML. Optional:
DESCRIPTRON_TEST_SPECIES (default: the first species with a treatment).

    DESCRIPTRON_TEST_MONOGRAPH=/data/monograph DESCRIPTRON_TEST_PROFILE=/data/profile.yaml \
        pytest -s tests/test_stdio_e2e.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

MONO = os.environ.get("DESCRIPTRON_TEST_MONOGRAPH")
PROFILE = os.environ.get("DESCRIPTRON_TEST_PROFILE")
pytestmark = pytest.mark.skipif(not (MONO and PROFILE), reason="set DESCRIPTRON_TEST_MONOGRAPH and _PROFILE")


def _payload(result):
    """Tool result -> python object (structured content if present, else the text)."""
    if getattr(result, "structuredContent", None) is not None:
        sc = result.structuredContent
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
    try:
        return json.loads(texts[0]) if len(texts) == 1 else texts
    except (json.JSONDecodeError, IndexError):
        return texts


async def _session_run(fn):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=sys.executable, args=["-m", "descriptron_mcp"],
                                   env=dict(os.environ))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            return await fn(s)


def test_end_to_end():
    mono = Path(MONO)
    matrix = mono / "compiled_key_tier"
    treatments = sorted(mono.glob("descriptions/*/*_treatment.json"))
    species = os.environ.get("DESCRIPTRON_TEST_SPECIES") or treatments[0].parent.name
    treatment = json.loads((mono / "descriptions" / species / f"{species}_treatment.json").read_text())
    sheet = mono / "descriptions" / species / f"{species}_data_sheet.txt"

    async def body(s):
        call = lambda _tool, **kw: s.call_tool(_tool, kw)        # noqa: E731
        tools = {t.name for t in (await s.list_tools()).tools}
        expected = {"list_programs", "program_help", "run_program", "start_job", "job_status",
                    "job_log", "cancel_job", "list_jobs", "list_files", "read_text", "read_table",
                    "coco_summary", "view_image", "species_evidence", "audit_treatment"}
        assert expected <= tools, expected - tools

        progs = _payload(await call("list_programs"))
        names = {p["name"] for p in progs["programs"]}
        assert "biorag_confabulation_checker_v2" in names and progs["count"] > 50
        print(f"\nprograms: {progs['count']}")

        h = _payload(await call("program_help", name="biorag_key_builder"))
        assert h["help_exit_code"] == 0 and "--matrix_dir" in h["help"], h
        print("program_help ok (prefix resolved to", h["program"], ")")

        ev = _payload(await call("species_evidence", matrix_dir=str(matrix), species=species, tier="key"))
        assert ev["features"] > 0 and all(r["tier"] == "key" for r in ev["rows"])
        print(f"species_evidence: {ev['features']} key-tier features for {species}")

        t = _payload(await call("read_table", path=str(matrix / "species_feature_summary.csv"),
                                where={"species": species}, max_rows=3))
        assert t["matching_rows"] >= ev["features"]

        # the audit on the published (repaired) treatment
        a = _payload(await call("audit_treatment", species=species, treatment=treatment,
                                matrix_dir=str(matrix), taxon_profile=PROFILE,
                                data_sheet=str(sheet) if sheet.exists() else ""))
        assert a["ok"], a
        print(f"audit as published: {a['claims_checked']} claims, {a['flagged_count']} flagged")

        # plant a column confusion: a correct number, printed next to the wrong structure
        bad = json.loads(json.dumps(treatment))
        rows = [r for r in ev["rows"] if r["unit"] == "mm" and r["min"] != r["max"]]
        src, dst = rows[0], next(r for r in rows[1:] if r["feature_id"].split(".")[0] != rows[0]["feature_id"].split(".")[0])
        struct = dst["feature_id"].split(".")[0].replace("_", " ")
        bad["diagnosis"] += (f" Its {struct} length is {float(src['min']):.3f}–{float(src['max']):.3f} mm.")
        b = _payload(await call("audit_treatment", species=species, treatment=bad, matrix_dir=str(matrix),
                                taxon_profile=PROFILE, data_sheet=str(sheet) if sheet.exists() else ""))
        assert b["flagged_count"] > a["flagged_count"], b
        print(f"planted error caught: {b['flagged_count']} flagged, e.g. "
              f"{b['flagged'][0]['type'] or b['flagged'][0]['status']}: {b['flagged'][0]['context'][-120:]!r}")

        # a background job: the key builder, restricted, without the LOO test
        j = _payload(await call("start_job", name="biorag_key_builder_v1",
                                args=["--matrix_dir", str(matrix), "--output_dir",
                                      str(Path(os.environ.get("DESCRIPTRON_MCP_STATE", "/tmp")) / "e2e_key"),
                                      "--no-loo"]))
        for _ in range(120):
            st = _payload(await call("job_status", job_id=j["job_id"]))
            if st["state"] != "running":
                break
            await asyncio.sleep(2)
        assert st["state"] == "finished", st
        assert st["files_written"]["count"] > 0
        print(f"job finished in {st['elapsed_s']} s, wrote {st['files_written']['count']} files")

        img = await call("view_image", path=str(next(mono.rglob("*.png"))), max_side=512)
        assert any(getattr(c, "type", "") == "image" for c in img.content)
        print("view_image ok")

        res = await s.read_resource("descriptron://workflow")
        assert "Evidence tiers" in res.contents[0].text
        pr = await s.get_prompt("write_audited_treatment", {"species": species, "matrix_dir": str(matrix),
                                                            "taxon_profile": PROFILE})
        assert "audit_treatment" in pr.messages[0].content.text
        print("resource + prompt ok")

    asyncio.run(_session_run(body))

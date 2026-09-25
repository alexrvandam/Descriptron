#!/usr/bin/env python3
"""
zenodo_upload.py — Upload BioRAG pipeline outputs to Zenodo.

Uploads compiled CSV, JSON-LD descriptions, taxonomic key, and figure
plates as a single Zenodo deposit with structured metadata.

Usage:
    python zenodo_upload.py \
        --output_base /path/to/pipeline/output \
        --title "Diaphorina species descriptions" \
        --authors "Van Dam, A.R." \
        --token_file ~/.descriptron/zenodo_token

    python zenodo_upload.py \
        --output_base /path/to/pipeline/output \
        --title "..." --authors "..." \
        --token "YOUR_API_TOKEN" \
        --sandbox  # use sandbox.zenodo.org for testing
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

ZENODO_API = "https://zenodo.org/api"
ZENODO_SANDBOX_API = "https://sandbox.zenodo.org/api"


def load_token(token: str = None, token_file: str = None) -> str:
    if token:
        return token.strip()
    if token_file:
        p = Path(token_file).expanduser()
        if p.exists():
            return p.read_text().strip()
    env = os.environ.get("ZENODO_TOKEN")
    if env:
        return env.strip()
    raise ValueError(
        "No Zenodo API token provided. Use --token, --token_file, "
        "or set ZENODO_TOKEN environment variable."
    )


def save_token(token: str, path: str = "~/.descriptron/zenodo_token"):
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(token.strip() + "\n")
    p.chmod(0o600)
    logger.info("Token saved to %s", p)


def collect_files(output_base: str) -> list:
    base = Path(output_base)
    files = []

    compiled = base / "compiled"
    if compiled.exists():
        for f in sorted(compiled.glob("*.csv")):
            files.append(("compiled", f))
        for f in sorted(compiled.glob("*.json")):
            files.append(("compiled", f))
        for f in sorted(compiled.glob("*.jsonl")):
            files.append(("compiled", f))

    descriptions = base / "biorag_descriptions"
    if descriptions.exists():
        desc_with_figs = descriptions / "descriptions_with_figures"
        src = desc_with_figs if desc_with_figs.exists() else descriptions
        for sp_dir in sorted(src.iterdir()):
            if sp_dir.is_dir():
                for f in sorted(sp_dir.glob("*.txt")):
                    files.append(("descriptions", f))
                for f in sorted(sp_dir.glob("*.json")):
                    files.append(("descriptions", f))
                for f in sorted(sp_dir.glob("*.jsonld")):
                    files.append(("descriptions", f))

        for key_file in sorted(descriptions.glob("*key*.txt")):
            files.append(("key", key_file))

    plates = base / "biorag_descriptions" / "species_plates"
    if plates.exists():
        for f in sorted(plates.glob("*.png")):
            files.append(("figures", f))

    general_figs = base / "biorag_descriptions" / "general_figures"
    if general_figs.exists():
        for f in sorted(general_figs.glob("*.png")):
            files.append(("figures", f))

    confab = base / "biorag_descriptions" / "confabulation_report"
    if confab.exists():
        for f in sorted(confab.glob("*.json")):
            files.append(("validation", f))
        for f in sorted(confab.glob("*.csv")):
            files.append(("validation", f))

    return files


def create_deposit(api_base: str, token: str, metadata: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    r = requests.post(
        f"{api_base}/deposit/depositions",
        params={"access_token": token},
        json={"metadata": metadata},
        headers=headers,
    )
    r.raise_for_status()
    dep = r.json()
    logger.info("Created deposit %s (id=%s)", dep["links"]["html"], dep["id"])
    return dep


def upload_file(bucket_url: str, token: str, filepath: Path, remote_name: str = None):
    name = remote_name or filepath.name
    with open(filepath, "rb") as fp:
        r = requests.put(
            f"{bucket_url}/{name}",
            data=fp,
            params={"access_token": token},
        )
    r.raise_for_status()
    size_mb = filepath.stat().st_size / 1024 / 1024
    logger.info("  Uploaded %s (%.1f MB)", name, size_mb)


def publish_deposit(api_base: str, token: str, deposit_id: int) -> dict:
    r = requests.post(
        f"{api_base}/deposit/depositions/{deposit_id}/actions/publish",
        params={"access_token": token},
    )
    r.raise_for_status()
    pub = r.json()
    logger.info("Published: %s", pub["links"]["html"])
    logger.info("DOI: %s", pub.get("doi", "pending"))
    return pub


def build_metadata(title: str, authors: str, description: str = None,
                   keywords: list = None) -> dict:
    creators = []
    for author in authors.split(";"):
        author = author.strip()
        if author:
            creators.append({"name": author})

    if not description:
        description = (
            "BioRAG pipeline outputs: compiled phenomic data matrix, "
            "ontology-annotated species descriptions (JSON-LD), "
            "dichotomous taxonomic key, and annotated specimen figure plates. "
            "Generated by Descriptron v2."
        )

    if not keywords:
        keywords = [
            "taxonomy", "species descriptions", "morphometrics",
            "BioRAG", "Descriptron", "COCO JSON", "phenomics",
            "FAIR data", "ontology", "JSON-LD",
        ]

    return {
        "title": title,
        "upload_type": "dataset",
        "description": description,
        "creators": creators,
        "keywords": keywords,
        "access_right": "open",
        "license": "cc-by-4.0",
        "communities": [{"identifier": "biosyslit"}],
    }


def run_upload(output_base: str, title: str, authors: str,
               token: str, sandbox: bool = False,
               description: str = None, keywords: list = None,
               publish: bool = False, dry_run: bool = False) -> dict:
    api_base = ZENODO_SANDBOX_API if sandbox else ZENODO_API
    env_label = "SANDBOX" if sandbox else "PRODUCTION"
    logger.info("Using Zenodo %s API", env_label)

    files = collect_files(output_base)
    if not files:
        logger.error("No files found in %s", output_base)
        return {"error": "no files found"}

    logger.info("Found %d files to upload:", len(files))
    for category, f in files:
        logger.info("  [%s] %s", category, f.name)

    if dry_run:
        logger.info("DRY RUN — no deposit created")
        return {"dry_run": True, "file_count": len(files)}

    metadata = build_metadata(title, authors, description, keywords)
    deposit = create_deposit(api_base, token, metadata)
    bucket_url = deposit["links"]["bucket"]

    logger.info("Uploading %d files...", len(files))
    for category, filepath in files:
        remote_name = f"{category}/{filepath.name}"
        upload_file(bucket_url, token, filepath, remote_name)

    result = {
        "deposit_id": deposit["id"],
        "html_url": deposit["links"]["html"],
        "file_count": len(files),
    }

    if publish:
        pub = publish_deposit(api_base, token, deposit["id"])
        result["doi"] = pub.get("doi")
        result["published"] = True
    else:
        logger.info("Deposit created but NOT published. Review at: %s",
                     deposit["links"]["html"])
        result["published"] = False

    return result


def parse_args():
    p = argparse.ArgumentParser(
        description="Upload BioRAG pipeline outputs to Zenodo",
    )
    p.add_argument("--output_base", required=True,
                   help="Base output directory from run_full_pipeline.py")
    p.add_argument("--title", required=True,
                   help="Zenodo deposit title")
    p.add_argument("--authors", required=True,
                   help="Semicolon-separated author names (Last, First)")
    p.add_argument("--description", default=None,
                   help="Deposit description (auto-generated if omitted)")
    p.add_argument("--keywords", nargs="*", default=None,
                   help="Keywords for the deposit")
    p.add_argument("--token", default=None,
                   help="Zenodo API token (or use --token_file)")
    p.add_argument("--token_file", default="~/.descriptron/zenodo_token",
                   help="File containing Zenodo API token")
    p.add_argument("--save_token", action="store_true",
                   help="Save provided --token to token_file for reuse")
    p.add_argument("--sandbox", action="store_true",
                   help="Use sandbox.zenodo.org for testing")
    p.add_argument("--publish", action="store_true",
                   help="Publish immediately (default: draft only)")
    p.add_argument("--dry_run", action="store_true",
                   help="List files without uploading")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    token = load_token(args.token, args.token_file)

    if args.save_token and args.token:
        save_token(args.token, args.token_file)

    result = run_upload(
        output_base=args.output_base,
        title=args.title,
        authors=args.authors,
        token=token,
        sandbox=args.sandbox,
        description=args.description,
        keywords=args.keywords,
        publish=args.publish,
        dry_run=args.dry_run,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

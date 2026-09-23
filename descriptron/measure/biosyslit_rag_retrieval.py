#!/usr/bin/env python3
"""
biosyslit_rag_retrieval.py — BioSysLit RAG Retrieval Module for Descriptron
===========================================================================

Retrieval-Augmented Generation engine for taxonomic species descriptions.
Indexes BioSysLit treatments (from Zenodo) and user-supplied PDFs, then
retrieves the k most relevant treatments for a given specimen query.

This module supports the three description modes from the paper:
  Mode A — Whole habitus description (full specimen image)
  Mode B — Segmented region description (isolated sclerites)
  Mode C — Micro-CT / specialized imaging (3D renders, SEM, etc.)

Each mode can use the same knowledge sources (BioSysLit + user PDFs)
but retrieves with different emphasis:
  Mode A → prioritizes treatments with habitus figures from same family/genus
  Mode B → prioritizes treatments mentioning the specific body region
  Mode C → prioritizes treatments using the same imaging modality

Architecture:
  1. INGEST   — Download/parse treatments → extract text + figure metadata
  2. INDEX    — Build searchable index (taxon hierarchy, body regions, modalities)
  3. RETRIEVE — Given a query (taxon + region + modality), return k best matches
  4. FORMAT   — Package retrieved context for VLM prompt construction

Data flow:
  BioSysLit (Zenodo API) ──┐
                            ├── Ingest → Index → Retrieve → Format → VLM Prompt
  User PDFs ───────────────┘

Dependencies:
  pip install requests pdfplumber sentence-transformers numpy faiss-cpu

Usage:
  # Build index from BioSysLit search
  python biosyslit_rag_retrieval.py index --taxon "Tetramorium" --max-records 100

  # Build index from local PDFs
  python biosyslit_rag_retrieval.py index --pdf-dir ./treatments/ --output index.json

  # Retrieve context for a description query
  python biosyslit_rag_retrieval.py retrieve \\
    --index index.json \\
    --taxon "Tetramorium" --family "Formicidae" \\
    --region "head" --mode habitus \\
    --k 5

  # Full pipeline: index + retrieve + format prompt
  python biosyslit_rag_retrieval.py describe \\
    --index index.json \\
    --image specimen.jpg \\
    --coco annotations.json \\
    --taxon "Tetramorium" --family "Formicidae"

Author: Alex Van Dam (alex.vandam@mfn.berlin)
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np

# Optional imports with graceful fallback
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

import base64  # stdlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TreatmentChunk:
    """A single retrievable unit from a taxonomic treatment."""
    chunk_id: str                       # unique identifier
    text: str                           # the text content
    source_type: str                    # "biosyslit" | "user_pdf" | "user_text"
    source_id: str                      # Zenodo record ID or PDF filename
    source_doi: str = ""                # DOI if available
    source_title: str = ""              # Publication title

    # Taxonomic metadata
    taxon_name: str = ""                # Scientific name mentioned
    taxon_family: str = ""              # Family
    taxon_order: str = ""               # Order
    taxon_class: str = ""               # Class

    # Content metadata
    body_region: str = ""               # "head", "mesosoma", "leg", etc. (if region-specific)
    imaging_modality: str = "standard"  # "standard", "sem", "micro_ct", "line_drawing", "photograph"
    description_section: str = ""       # "diagnosis", "description", "etymology", "distribution", etc.
    has_figure_ref: bool = False        # whether this chunk references a figure
    figure_urls: List[str] = field(default_factory=list)  # URLs to associated figures

    # Retrieval metadata
    embedding: Optional[List[float]] = field(default=None, repr=False)  # sentence embedding


@dataclass
class RetrievalQuery:
    """Query specification for retrieval."""
    taxon_name: str = ""
    taxon_family: str = ""
    taxon_order: str = ""
    body_region: str = ""               # specific region for Mode B
    imaging_modality: str = "standard"  # for Mode C
    description_mode: str = "habitus"   # "habitus" (A), "segmented" (B), "specialized" (C)
    free_text: str = ""                 # additional search terms
    k: int = 5                          # number of results to return


@dataclass
class RetrievalResult:
    """A single retrieval result with score."""
    chunk: TreatmentChunk
    score: float                        # relevance score (higher = better)
    match_reasons: List[str] = field(default_factory=list)  # why this was retrieved


# ═══════════════════════════════════════════════════════════════════════════════
# BODY REGION VOCABULARY (maps descriptive text to canonical regions)
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical regions → list of text patterns that indicate that region
REGION_PATTERNS: Dict[str, List[str]] = {
    "head": ["head", "frons", "vertex", "clypeus", "gena", "occiput", "mandible",
             "antenna", "scape", "funiculus", "eye", "compound eye", "ocelli",
             "frontal carina", "frontal lobe", "scrobe", "labrum", "maxilla",
             "labium", "palpus", "palp"],
    "mesosoma": ["mesosoma", "alitrunk", "pronotum", "mesonotum", "scutum",
                 "scutellum", "mesopleuron", "metapleuron", "propodeum",
                 "propodeal spine", "propodeal lobe", "katepisternum",
                 "anepisternum", "mesepimeron", "metepisternum"],
    "thorax": ["thorax", "prothorax", "mesothorax", "metathorax", "notum",
               "sternum", "pleuron", "pronotum", "mesonotum", "metanotum"],
    "wing": ["wing", "forewing", "hindwing", "haltere", "wing venation",
             "stigma", "pterostigma", "crossvein", "submarginal cell",
             "marginal cell", "discal cell", "cell-", "cell_",
             "whole wing", "whole_wing", "vein"],
    "leg": ["leg", "coxa", "trochanter", "femur", "tibia", "tarsus", "pretarsus",
            "claw", "arolium", "pulvillus", "basitarsus", "tibial spur",
            "calcar", "metatarsus"],
    "metasoma": ["metasoma", "gaster", "petiole", "postpetiole", "abdominal",
                 "tergite", "sternite", "pygidium", "hypopygium"],
    "abdomen": ["abdomen", "tergum", "sternum", "elytra", "elytron", "elytral",
                "pygidium", "ventrite", "urosternite", "urotergite"],
    "genitalia": ["genitalia", "aedeagus", "paramere", "volsella", "cuspis",
                  "digitus", "penisvalve", "harpe", "valva", "ovipositor",
                  "gonostylus", "gonocoxite", "spermatheca", "bursa",
                  "proctiger", "subgenital", "circumanal", "terminalia"],
    "mouthparts": ["mouthpart", "mandible", "maxilla", "labium", "labrum",
                   "glossa", "paraglossa", "galea", "lacinia", "proboscis",
                   "haustellum", "rostrum", "labial", "lab1", "lab2"],
    # Plants
    "leaf": ["leaf", "lamina", "blade", "petiole", "stipule", "midrib",
             "venation", "margin", "apex", "base"],
    "flower": ["flower", "petal", "sepal", "stamen", "pistil", "ovary",
               "style", "stigma", "anther", "filament", "corolla", "calyx"],
    "fruit": ["fruit", "seed", "achene", "capsule", "drupe", "berry",
              "pod", "legume", "silique", "samara"],
    "stem": ["stem", "trunk", "bark", "branch", "twig", "node", "internode",
             "lenticel", "rhizome", "stolon", "tuber", "bulb"],
    "root": ["root", "rootlet", "taproot", "adventitious", "rhizoid"],
    # Fungi
    "pileus": ["pileus", "cap", "cap surface", "umbo"],
    "hymenium": ["hymenium", "gill", "lamella", "pore", "tube", "teeth"],
    "stipe": ["stipe", "stem", "annulus", "volva", "ring"],
    "spore": ["spore", "basidium", "ascus", "conidium", "basidiospore",
              "ascospore"],
}

# Imaging modality patterns
MODALITY_PATTERNS: Dict[str, List[str]] = {
    "standard": ["photograph", "habitus", "dorsal view", "lateral view",
                 "ventral view", "frontal view"],
    "sem": ["sem", "scanning electron", "electron microscop", "ultrastructure",
            "micrograph"],
    "micro_ct": ["micro-ct", "micro ct", "µct", "x-ray", "computed tomograph",
                 "3d reconstruction", "volume rendering", "virtual section"],
    "line_drawing": ["line drawing", "illustration", "fig.", "figure",
                     "drawing", "plate"],
    "stacking": ["stacking", "focus stack", "montage", "extended depth",
                 "automontage", "z-stack"],
    "confocal": ["confocal", "clsm", "laser scanning"],
    "fluorescence": ["fluorescence", "autofluorescence", "uv"],
}

# Description section patterns
SECTION_PATTERNS: Dict[str, List[str]] = {
    "diagnosis": ["diagnosis", "diagnostic", "differs from", "distinguished from",
                  "can be separated", "similar to.*but", "differs.*by"],
    "description": ["description", "worker description", "queen description",
                    "male description", "redescription", "body length"],
    "etymology": ["etymology", "named after", "named for", "from the latin",
                  "from the greek", "referring to"],
    "distribution": ["distribution", "range", "locality", "known from",
                     "recorded from", "collected in", "type locality"],
    "biology": ["biology", "ecology", "habitat", "host", "nesting",
                "foraging", "behavior", "behaviour"],
    "material_examined": ["material examined", "type material", "holotype",
                          "paratype", "additional material", "specimens examined"],
    "remarks": ["remarks", "notes", "comments", "discussion", "comparison",
                "affinities"],
    "key": ["key to species", "identification key", "couplet", "key character"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# INGEST: BioSysLit (Zenodo API)
# ═══════════════════════════════════════════════════════════════════════════════

class BioSysLitIngester:
    """Fetches and parses taxonomic treatments from BioSysLit on Zenodo."""

    ZENODO_API = "https://zenodo.org/api/records"
    COMMUNITY = "biosyslit"

    def __init__(self, cache_dir: str = "./biosyslit_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, max_records: int = 100,
               content_type: str = "all") -> List[Dict]:
        """
        Search BioSysLit for records matching query.

        Args:
            query: Search term (taxon name, keyword, etc.)
            max_records: Maximum records to fetch
            content_type: "all", "images", "pdfs"

        Returns:
            List of Zenodo record dicts
        """
        if not HAS_REQUESTS:
            raise ImportError("requests library required: pip install requests")

        records = []
        page = 1
        per_page = min(50, max_records)

        while len(records) < max_records:
            params = {
                "communities": self.COMMUNITY,
                "q": query,
                "size": per_page,
                "page": page,
                "sort": "mostrecent",
            }

            try:
                resp = requests.get(self.ZENODO_API, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"Zenodo API error: {e}")
                break

            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break

            for rec in hits:
                if content_type == "images":
                    files = [f for f in rec.get("files", [])
                             if re.search(r'\.(jpe?g|png|tiff?)$', f.get("key", ""), re.I)]
                    if not files:
                        continue
                elif content_type == "pdfs":
                    files = [f for f in rec.get("files", [])
                             if f.get("key", "").lower().endswith(".pdf")]
                    if not files:
                        continue

                records.append(rec)
                if len(records) >= max_records:
                    break

            page += 1
            total = data.get("hits", {}).get("total", 0)
            if page * per_page > total:
                break

            time.sleep(0.3)  # Rate limiting

        logger.info(f"Found {len(records)} BioSysLit records for '{query}'")
        return records

    def extract_treatment_text(self, record: Dict) -> List[TreatmentChunk]:
        """
        Extract text chunks from a BioSysLit record.
        BioSysLit records often have structured description in metadata.
        """
        chunks = []
        rec_id = str(record.get("id", ""))
        metadata = record.get("metadata", {})
        title = metadata.get("title", "")
        doi = record.get("doi", "")
        description = metadata.get("description", "")
        keywords = metadata.get("keywords", [])

        # Extract taxon from title or keywords
        taxon = self._extract_taxon(title, keywords)

        # Figure URLs
        figure_urls = []
        for f in record.get("files", []):
            if re.search(r'\.(jpe?g|png|tiff?)$', f.get("key", ""), re.I):
                url = f.get("links", {}).get("self", "")
                if url:
                    figure_urls.append(url)

        # The description field often contains the treatment text
        if description:
            # Strip HTML tags
            clean_text = re.sub(r'<[^>]+>', ' ', description)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            if len(clean_text) > 50:  # Skip very short descriptions
                # Split into sections based on common headers
                sections = self._split_into_sections(clean_text)

                for section_name, section_text in sections:
                    if len(section_text.strip()) < 20:
                        continue

                    chunk_id = f"biosyslit_{rec_id}_{section_name}_{hashlib.md5(section_text[:100].encode()).hexdigest()[:8]}"

                    chunk = TreatmentChunk(
                        chunk_id=chunk_id,
                        text=section_text.strip(),
                        source_type="biosyslit",
                        source_id=rec_id,
                        source_doi=doi,
                        source_title=title,
                        taxon_name=taxon.get("name", ""),
                        taxon_family=taxon.get("family", ""),
                        taxon_order=taxon.get("order", ""),
                        body_region=self._detect_region(section_text),
                        imaging_modality=self._detect_modality(section_text),
                        description_section=section_name,
                        has_figure_ref=bool(re.search(r'[Ff]ig\.?\s*\d|[Ff]igure\s*\d', section_text)),
                        figure_urls=figure_urls,
                    )
                    chunks.append(chunk)

        return chunks

    def _extract_taxon(self, title: str, keywords: List[str]) -> Dict[str, str]:
        """Try to extract taxon name and higher classification from metadata."""
        taxon = {"name": "", "family": "", "order": ""}

        # Try keywords first (often have structured taxonomy)
        for kw in keywords:
            # Binomial pattern
            if re.match(r'^[A-Z][a-z]+ [a-z]+', kw) and not taxon["name"]:
                taxon["name"] = kw
            # Family pattern (ends in -idae/-aceae)
            if re.search(r'(idae|aceae)$', kw, re.I) and not taxon["family"]:
                taxon["family"] = kw
            # Order patterns
            for order_suffix in ["ptera", "iformes", "ales", "odonata"]:
                if kw.lower().endswith(order_suffix) and not taxon["order"]:
                    taxon["order"] = kw

        # Fallback: try title
        if not taxon["name"]:
            m = re.search(r'([A-Z][a-z]+ [a-z]+)', title)
            if m:
                taxon["name"] = m.group(1)

        return taxon

    def _split_into_sections(self, text: str) -> List[Tuple[str, str]]:
        """Split treatment text into labeled sections."""
        # Try to find section headers
        section_pattern = re.compile(
            r'(Diagnosis|Description|Etymology|Distribution|Biology|'
            r'Material[s]? examined|Remarks|Notes|Type material|'
            r'Differential diagnosis|Key|Discussion|Comparison|'
            r'Worker|Queen|Male|Measurements)',
            re.IGNORECASE
        )

        matches = list(section_pattern.finditer(text))

        if not matches:
            # No sections found — treat as single chunk
            detected = self._detect_section_type(text)
            return [(detected, text)]

        sections = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_name = m.group(1).lower().replace(" ", "_")
            section_text = text[start:end]
            sections.append((section_name, section_text))

        # Add any text before the first section header
        if matches[0].start() > 50:
            sections.insert(0, ("preamble", text[:matches[0].start()]))

        return sections

    def _detect_region(self, text: str) -> str:
        """Detect which body region a text chunk primarily describes."""
        text_lower = text.lower()
        region_scores: Dict[str, int] = {}

        for region, patterns in REGION_PATTERNS.items():
            score = sum(1 for p in patterns if p in text_lower)
            if score > 0:
                region_scores[region] = score

        if not region_scores:
            return "whole_body"

        return max(region_scores, key=region_scores.get)

    def _detect_modality(self, text: str) -> str:
        """Detect imaging modality mentioned in text."""
        text_lower = text.lower()

        for modality, patterns in MODALITY_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                return modality

        return "standard"

    def _detect_section_type(self, text: str) -> str:
        """Detect what type of description section this text is."""
        text_lower = text.lower()

        for section, patterns in SECTION_PATTERNS.items():
            if any(re.search(p, text_lower) for p in patterns):
                return section

        return "description"  # default


# ═══════════════════════════════════════════════════════════════════════════════
# INGEST: User PDFs
# ═══════════════════════════════════════════════════════════════════════════════

class PDFIngester:
    """Extracts text chunks from user-supplied PDF files."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_from_pdf(self, pdf_path: str,
                         taxon_name: str = "",
                         taxon_family: str = "") -> List[TreatmentChunk]:
        """Extract treatment chunks from a PDF file."""
        if not HAS_PDFPLUMBER:
            raise ImportError("pdfplumber required: pip install pdfplumber")

        chunks = []
        filename = os.path.basename(pdf_path)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return chunks

        if not full_text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return chunks

        # Try section-based splitting first
        ingester = BioSysLitIngester()  # reuse section splitting logic
        sections = ingester._split_into_sections(full_text)

        for section_name, section_text in sections:
            # Further chunk long sections
            text_chunks = self._chunk_text(section_text)

            for i, chunk_text in enumerate(text_chunks):
                if len(chunk_text.strip()) < 30:
                    continue

                chunk_id = f"pdf_{hashlib.md5(filename.encode()).hexdigest()[:8]}_{section_name}_{i}"

                chunk = TreatmentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text.strip(),
                    source_type="user_pdf",
                    source_id=filename,
                    source_title=filename,
                    taxon_name=taxon_name,
                    taxon_family=taxon_family,
                    body_region=ingester._detect_region(chunk_text),
                    imaging_modality=ingester._detect_modality(chunk_text),
                    description_section=section_name,
                    has_figure_ref=bool(re.search(r'[Ff]ig\.?\s*\d|[Ff]igure\s*\d', chunk_text)),
                )
                chunks.append(chunk)

        logger.info(f"Extracted {len(chunks)} chunks from {filename}")
        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks by sentence boundaries."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sent in sentences:
            if len(current_chunk) + len(sent) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Keep overlap
                overlap_sentences = current_chunk.split('. ')
                current_chunk = '. '.join(overlap_sentences[-2:]) + ' ' if len(overlap_sentences) > 2 else ""
            current_chunk += sent + " "

        if current_chunk.strip():
            chunks.append(current_chunk)

        return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# INDEX: Build searchable treatment index
# ═══════════════════════════════════════════════════════════════════════════════

class TreatmentIndex:
    """
    Searchable index of treatment chunks.

    Supports two retrieval modes:
      1. Keyword/metadata filtering (fast, no ML dependencies)
      2. Semantic similarity with sentence embeddings (requires sentence-transformers + faiss)
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.chunks: List[TreatmentChunk] = []
        self.embedding_model_name = embedding_model
        self._encoder = None
        self._faiss_index = None
        self._embeddings_matrix = None

    def add_chunks(self, chunks: List[TreatmentChunk]):
        """Add chunks to the index."""
        self.chunks.extend(chunks)
        # Invalidate FAISS index
        self._faiss_index = None
        self._embeddings_matrix = None

    def build_embeddings(self):
        """Compute sentence embeddings for all chunks (requires sentence-transformers)."""
        if not HAS_SBERT:
            logger.warning("sentence-transformers not available — semantic search disabled. "
                           "Install: pip install sentence-transformers")
            return

        if self._encoder is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._encoder = SentenceTransformer(self.embedding_model_name)

        texts = [c.text[:512] for c in self.chunks]  # Truncate for embedding
        logger.info(f"Computing embeddings for {len(texts)} chunks...")
        embeddings = self._encoder.encode(texts, show_progress_bar=True, batch_size=32)

        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings[i].tolist()

        self._embeddings_matrix = np.array(embeddings, dtype=np.float32)

        # Build FAISS index
        if HAS_FAISS and len(self._embeddings_matrix) > 0:
            dim = self._embeddings_matrix.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
            # Normalize for cosine similarity
            faiss.normalize_L2(self._embeddings_matrix)
            self._faiss_index.add(self._embeddings_matrix)
            logger.info(f"FAISS index built: {self._faiss_index.ntotal} vectors, dim={dim}")

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve the k most relevant treatment chunks for a query.

        Uses a hybrid scoring approach:
          1. Metadata filters (taxon match, region match, modality match)
          2. Semantic similarity (if embeddings available)
          3. Combined re-ranking
        """
        candidates = []

        for chunk in self.chunks:
            score = 0.0
            reasons = []

            # === Taxonomic matching (highest weight) ===
            if query.taxon_name and chunk.taxon_name:
                # Exact species match
                if query.taxon_name.lower() == chunk.taxon_name.lower():
                    score += 10.0
                    reasons.append(f"exact_taxon:{chunk.taxon_name}")
                # Genus match
                elif query.taxon_name.split()[0].lower() == chunk.taxon_name.split()[0].lower():
                    score += 5.0
                    reasons.append(f"genus_match:{chunk.taxon_name.split()[0]}")

            if query.taxon_family and chunk.taxon_family:
                if query.taxon_family.lower() == chunk.taxon_family.lower():
                    score += 3.0
                    reasons.append(f"family_match:{chunk.taxon_family}")

            if query.taxon_order and chunk.taxon_order:
                if query.taxon_order.lower() == chunk.taxon_order.lower():
                    score += 1.0
                    reasons.append(f"order_match:{chunk.taxon_order}")

            # === Region matching (for Mode B) ===
            if query.body_region:
                if chunk.body_region == query.body_region:
                    score += 4.0
                    reasons.append(f"region_match:{chunk.body_region}")
                # Also check if the query region appears in the text
                region_patterns = REGION_PATTERNS.get(query.body_region, [])
                text_lower = chunk.text.lower()
                region_mentions = sum(1 for p in region_patterns if p in text_lower)
                if region_mentions > 0:
                    score += min(region_mentions * 0.5, 3.0)
                    reasons.append(f"region_mentions:{region_mentions}")

            # === Modality matching (for Mode C) ===
            if query.imaging_modality != "standard":
                if chunk.imaging_modality == query.imaging_modality:
                    score += 3.0
                    reasons.append(f"modality_match:{chunk.imaging_modality}")

            # === Section type preferences ===
            if query.description_mode == "habitus":
                # For whole habitus, prioritize descriptions and diagnoses
                if chunk.description_section in ("description", "diagnosis"):
                    score += 2.0
                    reasons.append(f"section_preference:{chunk.description_section}")
            elif query.description_mode == "segmented":
                # For segmented, prioritize detailed descriptions
                if chunk.description_section == "description":
                    score += 2.0
                    reasons.append("section_preference:description")
            elif query.description_mode == "specialized":
                # For specialized imaging, figure references are valuable
                if chunk.has_figure_ref:
                    score += 1.5
                    reasons.append("has_figure_ref")

            # === Text length bonus (prefer substantive chunks) ===
            if len(chunk.text) > 200:
                score += 0.5

            if score > 0:
                candidates.append(RetrievalResult(
                    chunk=chunk,
                    score=score,
                    match_reasons=reasons,
                ))

        # === Semantic re-ranking (if available) ===
        if self._faiss_index is not None and self._encoder is not None:
            query_text = self._build_query_text(query)
            query_embedding = self._encoder.encode([query_text])
            query_embedding = np.array(query_embedding, dtype=np.float32)
            faiss.normalize_L2(query_embedding)

            # Get top-k from FAISS
            k_faiss = min(query.k * 3, self._faiss_index.ntotal)
            scores, indices = self._faiss_index.search(query_embedding, k_faiss)

            # Create a map of chunk_id → semantic score
            semantic_scores = {}
            for score_val, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks):
                    semantic_scores[self.chunks[idx].chunk_id] = float(score_val)

            # Add semantic score to candidates
            for result in candidates:
                sem_score = semantic_scores.get(result.chunk.chunk_id, 0.0)
                result.score += sem_score * 5.0  # Weight semantic similarity
                if sem_score > 0.3:
                    result.match_reasons.append(f"semantic_sim:{sem_score:.2f}")

            # Also add high-semantic-score chunks that weren't caught by metadata
            for idx in indices[0]:
                if idx >= 0 and idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    if not any(r.chunk.chunk_id == chunk.chunk_id for r in candidates):
                        sem_score = semantic_scores.get(chunk.chunk_id, 0.0)
                        if sem_score > 0.4:  # Only add if strong semantic match
                            candidates.append(RetrievalResult(
                                chunk=chunk,
                                score=sem_score * 5.0,
                                match_reasons=[f"semantic_only:{sem_score:.2f}"],
                            ))

        # Sort by score, return top k
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[:query.k]

    def _build_query_text(self, query: RetrievalQuery) -> str:
        """Build a text representation of the query for embedding."""
        parts = []
        if query.taxon_name:
            parts.append(f"species description of {query.taxon_name}")
        if query.taxon_family:
            parts.append(f"family {query.taxon_family}")
        if query.body_region:
            parts.append(f"{query.body_region} morphology")
        if query.imaging_modality != "standard":
            parts.append(f"{query.imaging_modality} imaging")
        if query.free_text:
            parts.append(query.free_text)
        return " ".join(parts) if parts else "taxonomic species description morphology"

    # === Persistence ===

    def save(self, path: str):
        """Save index to JSON file, with embeddings in a paired .npy file."""
        data = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "embedding_model": self.embedding_model_name,
            "chunk_count": len(self.chunks),
            "chunks": [asdict(c) for c in self.chunks],
        }
        for chunk_data in data["chunks"]:
            chunk_data.pop("embedding", None)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        # Save embeddings matrix so load() skips re-encoding
        emb_path = path + ".embeddings.npy"
        if hasattr(self, '_embeddings_matrix') and self._embeddings_matrix is not None:
            np.save(emb_path, self._embeddings_matrix)
            logger.info(f"Saved embeddings to {emb_path}")

        logger.info(f"Saved index with {len(self.chunks)} chunks to {path}")

    def load(self, path: str):
        """Load index from JSON file. Uses paired .npy embeddings if present."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.chunks = []
        for chunk_data in data.get("chunks", []):
            chunk_data.pop("embedding", None)
            self.chunks.append(TreatmentChunk(**chunk_data))

        # Restore embeddings and FAISS index from saved .npy — no re-encoding needed
        emb_path = path + ".embeddings.npy"
        if os.path.exists(emb_path):
            self._embeddings_matrix = np.load(emb_path)
            try:
                dim = self._embeddings_matrix.shape[1]
                self._faiss_index = faiss.IndexFlatIP(dim)
                matrix_copy = np.array(self._embeddings_matrix, dtype=np.float32)
                faiss.normalize_L2(matrix_copy)
                self._faiss_index.add(matrix_copy)
                logger.info(f"Restored FAISS index from {emb_path} ({self._faiss_index.ntotal} vectors)")
            except Exception as e:
                logger.warning(f"Could not restore FAISS index from embeddings: {e}")
                self._faiss_index = None
        else:
            logger.info(f"No embeddings file found at {emb_path} — call build_semantic_index() to generate")

        logger.info(f"Loaded index with {len(self.chunks)} chunks from {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT: Build VLM prompt context from retrieved chunks
# ═══════════════════════════════════════════════════════════════════════════════

class PromptFormatter:
    """
    Formats retrieved treatment chunks into structured context for VLM prompts.

    Produces output compatible with Descriptron's JSON-LD knowledge graph schema.
    """

    # System prompt template for the VLM
    SYSTEM_PROMPT = """You are an expert taxonomist creating structured morphological descriptions.
You will be given:
1. A specimen image (whole habitus OR a segmented body region)
2. Context from related taxonomic treatments retrieved from BioSysLit
3. A COCO JSON annotation file identifying the body region shown (if segmented)

Your task is to produce a JSON-LD knowledge graph describing the visible morphological features,
following the Descriptron schema with nodes for AnatomicalRegion, MorphologicalTrait, and
SegmentationMask, linked by hasRegion, hasTrait, and hasSegmentation edges.

CRITICAL RULES:
- ONLY describe features that are VISIBLE in the provided image
- Tag each trait with evidenceSource: "visible_in_image" or "visible_in_segmented_image"
- Use the ROSETTA controlled vocabulary for attribute values (texture, sculpture, setae, color, etc.)
- Include ontology URIs (HAO, UBERON, AISM, COLAO) where available
- If you cannot determine a character state from the image, omit it — do NOT guess
- Use the retrieved treatments as reference for terminology and expected characters,
  but do NOT copy character states from them unless confirmed in the image
"""

    def format_retrieval_context(self, results: List[RetrievalResult],
                                  query: RetrievalQuery) -> str:
        """Format retrieved chunks as context for the VLM prompt."""
        if not results:
            return "No similar treatments found in the knowledge base."

        context_parts = [
            f"=== Retrieved taxonomic context ({len(results)} treatments) ===",
            f"Query: {query.taxon_name} ({query.taxon_family})",
            f"Mode: {query.description_mode}",
        ]

        if query.body_region:
            context_parts.append(f"Region: {query.body_region}")

        context_parts.append("")

        for i, result in enumerate(results):
            chunk = result.chunk
            header = f"--- Treatment {i+1} (score: {result.score:.1f}) ---"
            meta_parts = []
            if chunk.taxon_name:
                meta_parts.append(f"Taxon: {chunk.taxon_name}")
            if chunk.taxon_family:
                meta_parts.append(f"Family: {chunk.taxon_family}")
            if chunk.source_doi:
                meta_parts.append(f"DOI: {chunk.source_doi}")
            if chunk.description_section:
                meta_parts.append(f"Section: {chunk.description_section}")
            if chunk.body_region and chunk.body_region != "whole_body":
                meta_parts.append(f"Region: {chunk.body_region}")

            context_parts.append(header)
            if meta_parts:
                context_parts.append("  " + " | ".join(meta_parts))
            context_parts.append(chunk.text[:1500])  # Truncate very long chunks
            context_parts.append("")

        return "\n".join(context_parts)

    def build_description_prompt(self, results: List[RetrievalResult],
                                   query: RetrievalQuery,
                                   coco_region: Optional[Dict] = None,
                                   rosetta_vocabulary: Optional[Dict] = None) -> Dict:
        """
        Build a complete prompt dict for the VLM API call.

        Returns:
            Dict with 'system', 'context', 'user_prompt', and 'output_schema' keys
        """
        context = self.format_retrieval_context(results, query)

        # Build user prompt based on description mode
        if query.description_mode == "habitus":
            user_prompt = self._habitus_prompt(query, coco_region)
        elif query.description_mode == "segmented":
            user_prompt = self._segmented_prompt(query, coco_region)
        elif query.description_mode == "specialized":
            user_prompt = self._specialized_prompt(query, coco_region)
        else:
            user_prompt = self._habitus_prompt(query, coco_region)

        # Add vocabulary reference if available
        vocab_ref = ""
        if rosetta_vocabulary:
            # Include relevant attribute categories
            relevant_attrs = ["texture", "sculpture", "setae", "color",
                              "color_pattern", "shape", "margin", "luster",
                              "surface_covering"]
            vocab_parts = ["=== Available ROSETTA vocabulary ==="]
            for attr in relevant_attrs:
                if attr in rosetta_vocabulary:
                    values = rosetta_vocabulary[attr]
                    if isinstance(values, dict) and "values" in values:
                        values = values["values"]
                    vocab_parts.append(f"{attr}: {', '.join(values[:20])}")
            vocab_ref = "\n".join(vocab_parts)

        return {
            "system": self.SYSTEM_PROMPT,
            "context": context,
            "vocabulary": vocab_ref,
            "user_prompt": user_prompt,
            "output_schema": self._json_ld_schema(),
        }

    def _habitus_prompt(self, query: RetrievalQuery,
                         coco_region: Optional[Dict] = None) -> str:
        region_info = ""
        if coco_region:
            categories = coco_region.get("categories", [])
            region_info = f"\nThe COCO annotation identifies these regions: {', '.join(c['name'] for c in categories)}"

        return f"""Examine this habitus image of {query.taxon_name or 'the specimen'} ({query.taxon_family or 'unknown family'}).
{region_info}
Describe ALL visible morphological features, producing a JSON-LD knowledge graph.
For each visible body region, note: texture, sculpture, setae, color, color pattern, shape, and any other observable characters.
Tag every trait with evidenceSource: "visible_in_image" and a confidence level.
Use the retrieved treatments as reference for expected terminology and character states."""

    def _segmented_prompt(self, query: RetrievalQuery,
                           coco_region: Optional[Dict] = None) -> str:
        region_name = query.body_region or "body region"
        region_info = ""
        if coco_region:
            cat_name = coco_region.get("categories", [{}])[0].get("name", region_name)
            ontology = coco_region.get("categories", [{}])[0].get("ontology", "")
            region_info = f"\nCOCO annotation: category='{cat_name}', ontology='{ontology}'"

        return f"""This is a SEGMENTED image showing ONLY the {region_name} of {query.taxon_name or 'the specimen'} ({query.taxon_family or 'unknown family'}).
{region_info}
Provide a DETAILED description of this specific region only.
Because this is an isolated segment, you can observe fine details — describe:
- Surface microsculpture (texture, sculpture) at high detail
- Setae morphology (type, density, distribution, orientation)
- Color and color pattern
- Shape and margins
- Any diagnostic features visible at this magnification
Tag every trait with evidenceSource: "visible_in_segmented_image"."""

    def _specialized_prompt(self, query: RetrievalQuery,
                             coco_region: Optional[Dict] = None) -> str:
        modality = query.imaging_modality or "specialized"

        return f"""This is a {modality.upper()} image of {query.body_region or 'a body region'} from {query.taxon_name or 'the specimen'} ({query.taxon_family or 'unknown family'}).
Describe the morphological features visible in this {modality} image.
For micro-CT: describe internal structures, wall thickness, cavities, trabeculae.
For SEM: describe surface ultrastructure, microsculpture, pore patterns, setae socket morphology.
Tag every trait with evidenceSource: "visible_in_{modality}"."""

    def _json_ld_schema(self) -> Dict:
        """Return the expected JSON-LD output schema."""
        return {
            "@context": {
                "@vocab": "https://descriptron.org/ontology/",
                "dwc": "http://rs.tdwg.org/dwc/terms/",
                "uberon": "http://purl.obolibrary.org/obo/UBERON_",
                "hao": "http://purl.obolibrary.org/obo/HAO_",
                "aism": "http://purl.obolibrary.org/obo/AISM_",
            },
            "expected_node_types": [
                "Specimen", "Taxon", "Image", "AnatomicalRegion",
                "MorphologicalTrait", "SegmentationMask", "KeypointSet",
            ],
            "expected_edge_types": [
                "identifiedAs", "hasImage", "hasRegion", "hasTrait",
                "hasSegmentation", "hasKeypoints",
            ],
            "trait_properties": {
                "traitType": "ROSETTA category name (texture, sculpture, setae, etc.)",
                "value": "ROSETTA controlled vocabulary term",
                "ontology_uri": "CURIE if available (HAO:NNNNNNN, PATO:NNNNNNN, etc.)",
                "evidenceSource": "visible_in_image | visible_in_segmented_image | visible_in_sem | visible_in_micro_ct",
                "confidence": "certain | probable | possible | uncertain",
            }
        }


def load_api_key(key_file: str = None) -> None:
    """
    Load ANTHROPIC_API_KEY from a file the user specifies explicitly,
    or fall back to the environment if already set.
    """
    from dotenv import load_dotenv

    if key_file:
        path = Path(key_file).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"API key file not found: {path}")
        load_dotenv(path, override=True)
        logger.info(f"API key loaded from: {path}")
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "No API key found. Provide one with:\n"
            "  --api-key-file /path/to/your/.env\n"
            "or set ANTHROPIC_API_KEY in your environment."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-MODE DESCRIPTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class DescriptionPipeline:
    """
    Orchestrates the three-mode description pipeline:
      Mode A — Whole habitus
      Mode B — Segmented regions (one per COCO category)
      Mode C — Specialized imaging (SEM, micro-CT)

    Each mode produces a partial JSON-LD knowledge graph.
    The pipeline merges them into a single comprehensive graph.
    """

    def __init__(self, index: TreatmentIndex):
        self.index = index
        self.formatter = PromptFormatter()

    def generate_prompts(self, taxon_name: str, taxon_family: str,
                          coco_json: Optional[Dict] = None,
                          imaging_modality: str = "standard",
                          k: int = 5,
                          rosetta: Optional[Dict] = None) -> List[Dict]:
        """
        Generate all description prompts for a specimen.

        Args:
            taxon_name: Scientific name
            taxon_family: Family name
            coco_json: COCO annotation JSON (with categories and annotations)
            imaging_modality: "standard", "sem", "micro_ct"
            k: Number of retrieved treatments per query
            rosetta: ROSETTA vocabulary dict (from toSimpleATTR or toJSON)

        Returns:
            List of prompt dicts, one per description mode/region.
            Each has: mode, region, prompt_data, image_type
        """
        prompts = []

        # === Mode A: Whole habitus ===
        query_a = RetrievalQuery(
            taxon_name=taxon_name,
            taxon_family=taxon_family,
            description_mode="habitus",
            imaging_modality=imaging_modality,
            k=k,
        )
        results_a = self.index.retrieve(query_a)
        prompt_a = self.formatter.build_description_prompt(
            results_a, query_a, coco_region=coco_json, rosetta_vocabulary=rosetta
        )
        prompts.append({
            "mode": "A_habitus",
            "region": "whole_body",
            "prompt_data": prompt_a,
            "image_type": "whole_specimen",
            "retrieval_results": len(results_a),
        })

        # === Mode B: Segmented regions ===
        if coco_json and "categories" in coco_json:
            for cat in coco_json["categories"]:
                cat_name = cat.get("name", "")
                if cat_name.lower() in ("trash", "scale_bar", "label"):
                    continue

                # Map COCO category to canonical region
                region = self._map_category_to_region(cat_name)

                query_b = RetrievalQuery(
                    taxon_name=taxon_name,
                    taxon_family=taxon_family,
                    body_region=region,
                    description_mode="segmented",
                    imaging_modality=imaging_modality,
                    k=max(3, k // 2),  # Fewer per region
                )
                results_b = self.index.retrieve(query_b)
                prompt_b = self.formatter.build_description_prompt(
                    results_b, query_b,
                    coco_region={"categories": [cat]},
                    rosetta_vocabulary=rosetta,
                )
                prompts.append({
                    "mode": "B_segmented",
                    "region": region,
                    "category_name": cat_name,
                    "category_id": cat.get("id"),
                    "ontology": cat.get("ontology", ""),
                    "prompt_data": prompt_b,
                    "image_type": "segmented_region",
                    "retrieval_results": len(results_b),
                })

        # === Mode C: Specialized imaging (if applicable) ===
        if imaging_modality not in ("standard",):
            query_c = RetrievalQuery(
                taxon_name=taxon_name,
                taxon_family=taxon_family,
                imaging_modality=imaging_modality,
                description_mode="specialized",
                k=k,
            )
            results_c = self.index.retrieve(query_c)
            prompt_c = self.formatter.build_description_prompt(
                results_c, query_c, rosetta_vocabulary=rosetta
            )
            prompts.append({
                "mode": "C_specialized",
                "region": "specialized",
                "imaging_modality": imaging_modality,
                "prompt_data": prompt_c,
                "image_type": imaging_modality,
                "retrieval_results": len(results_c),
            })

        logger.info(f"Generated {len(prompts)} description prompts "
                     f"(Mode A: 1, Mode B: {sum(1 for p in prompts if p['mode']=='B_segmented')}, "
                     f"Mode C: {sum(1 for p in prompts if p['mode']=='C_specialized')})")
        return prompts

    def _map_category_to_region(self, category_name: str) -> str:
        """Map a COCO category name to a canonical body region."""
        name_lower = category_name.lower()

        for region, patterns in REGION_PATTERNS.items():
            if any(p in name_lower for p in patterns):
                return region

        # Direct mappings for common Descriptron categories
        direct_map = {
            "mandible": "head", "clypeus": "head", "frons": "head",
            "eye": "head", "antenna": "head", "scape": "head",
            "funiculus": "head", "gena": "head", "vertex": "head",
            "pronotum": "mesosoma", "mesonotum": "mesosoma",
            "propodeum": "mesosoma", "mesopleuron": "mesosoma",
            "petiole": "metasoma", "postpetiole": "metasoma",
            "gaster": "metasoma",
            "elytra": "abdomen", "elytron": "abdomen",
            "forewing": "wing", "hindwing": "wing",
            "coxa": "leg", "femur": "leg", "tibia": "leg",
        }

        for key, region in direct_map.items():
            if key in name_lower:
                return region

        return "whole_body"

    @staticmethod
    def merge_knowledge_graphs(graphs: List[Dict]) -> Dict:
        """
        Merge multiple partial JSON-LD knowledge graphs into one.

        Each graph comes from a different description mode (A, B, or C).
        Nodes are deduplicated by ID; traits from segmented descriptions
        take precedence over whole-habitus ones (higher resolution).
        """
        merged = {
            "@context": {
                "@vocab": "https://descriptron.org/ontology/",
                "dwc": "http://rs.tdwg.org/dwc/terms/",
                "uberon": "http://purl.obolibrary.org/obo/UBERON_",
                "hao": "http://purl.obolibrary.org/obo/HAO_",
                "aism": "http://purl.obolibrary.org/obo/AISM_",
            },
            "@type": "SpecimenAnnotationGraph",
            "timestamp": datetime.now().isoformat(),
            "description_modes_used": [],
            "nodes": [],
            "edges": [],
            "summary": {},
        }

        seen_node_ids = set()
        seen_edge_keys = set()

        for graph in graphs:
            mode = graph.get("description_mode", "unknown")
            if mode not in merged["description_modes_used"]:
                merged["description_modes_used"].append(mode)

            for node in graph.get("nodes", []):
                node_id = node.get("id", "")
                if node_id and node_id not in seen_node_ids:
                    # Add source mode metadata
                    node["_source_mode"] = mode
                    merged["nodes"].append(node)
                    seen_node_ids.add(node_id)
                elif node_id in seen_node_ids and node.get("type") == "MorphologicalTrait":
                    # For traits, segmented descriptions override habitus ones
                    if mode == "B_segmented":
                        # Replace the existing node
                        merged["nodes"] = [n for n in merged["nodes"] if n.get("id") != node_id]
                        node["_source_mode"] = mode
                        merged["nodes"].append(node)

            for edge in graph.get("edges", []):
                edge_key = f"{edge.get('source','')}→{edge.get('relation','')}→{edge.get('target','')}"
                if edge_key not in seen_edge_keys:
                    merged["edges"].append(edge)
                    seen_edge_keys.add(edge_key)

        # Update summary
        merged["summary"] = {
            "nodeCount": len(merged["nodes"]),
            "edgeCount": len(merged["edges"]),
            "regionCount": len([n for n in merged["nodes"] if n.get("type") == "AnatomicalRegion"]),
            "traitCount": len([n for n in merged["nodes"] if n.get("type") == "MorphologicalTrait"]),
            "modesUsed": merged["description_modes_used"],
        }

        return merged




# ═══════════════════════════════════════════════════════════════════════════════
# API KEY LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_api_key(key_file: Optional[str] = None) -> None:
    """
    Load ANTHROPIC_API_KEY from a user-specified .env file, or use whatever
    is already in the environment.

    Args:
        key_file: Path to a file containing the line:
                      ANTHROPIC_API_KEY=sk-ant-...
                  Accepts plain .env format or a file with just the key value.
                  If None, falls back to the current environment.

    Raises:
        FileNotFoundError: if key_file is given but does not exist.
        EnvironmentError:  if no key is found anywhere.
    """
    if key_file:
        path = Path(key_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"API key file not found: {path}\n"
                f"Create it with one line:  ANTHROPIC_API_KEY=sk-ant-..."
            )
        # Parse the file — accept  KEY=value  or bare  value
        text = path.read_text().strip()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                if k.strip() == "ANTHROPIC_API_KEY":
                    os.environ["ANTHROPIC_API_KEY"] = v.strip()
                    logger.info(f"API key loaded from: {path}")
                    return
            else:
                # Bare key value with no KEY= prefix
                os.environ["ANTHROPIC_API_KEY"] = line
                logger.info(f"API key loaded from: {path}")
                return
        raise EnvironmentError(
            f"ANTHROPIC_API_KEY not found inside {path}\n"
            f"File should contain:  ANTHROPIC_API_KEY=sk-ant-..."
        )

    # No file given — check environment
    if os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("API key found in environment.")
        return

    raise EnvironmentError(
        "No API key found.\n"
        "Provide one with:  --api-key-file /path/to/your.env\n"
        "File should contain one line:  ANTHROPIC_API_KEY=sk-ant-..."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLAUDE DESCRIBER
# ═══════════════════════════════════════════════════════════════════════════════

CLAUDE_SONNET = "claude-sonnet-4-6"
CLAUDE_OPUS = "claude-opus-4-6"

_MODEL_MAX_TOKENS = {
    "opus": 16384,
    "sonnet": 16384,
    "haiku": 8192,
}

def _max_output_tokens(model: str) -> int:
    """Return the maximum output tokens for a Claude model."""
    m = model.lower()
    for family, limit in _MODEL_MAX_TOKENS.items():
        if family in m:
            return limit
    return 16384

_IMAGE_MEDIA_TYPES = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".gif":  "image/gif",
}


class ClaudeDescriber:
    """
    Calls Claude Sonnet 4.6 with a specimen image plus RAG-retrieved treatment
    text to produce a JSON-LD morphological description.

    Retrieved treatment chunks are packed into the system prompt so they act as
    persistent in-context knowledge across all mode calls for one specimen.

    Usage
    -----
        index = TreatmentIndex(); index.load("my_index.json")
        pipeline  = DescriptionPipeline(index)
        describer = ClaudeDescriber()

        merged = describer.describe_specimen(
            pipeline, "Cimex lectularius", "Cimicidae",
            image_path="spec.jpg", coco_json=coco, k=5,
        )
    """

    def __init__(
        self,
        model: str = CLAUDE_OPUS,
        max_tokens: int = 8192,
        max_retries: int = 3,
        retry_delay: float = 4.0,
    ):
        if not HAS_ANTHROPIC:
            raise ImportError(
                "Run:  pip install anthropic>=0.40\n"
                "Then provide key with --api-key-file"
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set.\n"
                "Use --api-key-file /path/to/your.env"
            )
        self.client      = anthropic.Anthropic(api_key=api_key)
        self.model       = model
        self.max_tokens  = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ── Public API ─────────────────────────────────────────────────────────────

    def call(self, prompt_data: Dict, image_path: Optional[str] = None) -> Dict:
        """
        Send one prompt dict (from PromptFormatter) + optional image to Claude.
        Returns parsed JSON-LD dict.
        """
        system_text  = self._build_system(prompt_data)
        user_content = self._build_user_content(prompt_data, image_path)
        return self._call_with_retry(system_text, user_content)

    def describe_specimen(
        self,
        pipeline,
        taxon_name: str,
        taxon_family: str,
        image_path: str,
        coco_json: Optional[Dict] = None,
        imaging_modality: str = "standard",
        k: int = 5,
        rosetta: Optional[Dict] = None,
    ) -> Dict:
        """
        Full pipeline: generate prompts → call Claude per mode → merge graphs.
        """
        prompts = pipeline.generate_prompts(
            taxon_name=taxon_name,
            taxon_family=taxon_family,
            coco_json=coco_json,
            imaging_modality=imaging_modality,
            k=k,
            rosetta=rosetta,
        )

        partial_graphs = []
        for entry in prompts:
            logger.info(
                f"  [{entry['mode']}] region={entry['region']} "
                f"({entry['retrieval_results']} treatments in context)"
            )
            try:
                graph = self.call(entry["prompt_data"], image_path=image_path)
                graph["_mode"]   = entry["mode"]
                graph["_region"] = entry["region"]
                partial_graphs.append(graph)
            except Exception as e:
                logger.error(f"  Mode {entry['mode']} failed: {e}")

        if not partial_graphs:
            raise RuntimeError("All description modes failed — check logs above.")

        merged = DescriptionPipeline.merge_knowledge_graphs(partial_graphs)
        merged["_specimen_image"] = image_path
        merged["_taxon"]          = taxon_name
        return merged

    def describe_batch(
        self,
        pipeline,
        image_dir: str,
        taxon_name: str,
        taxon_family: str,
        coco_dir: Optional[str] = None,
        output_dir: str = "./descriptions/",
        k: int = 5,
    ) -> List[Dict]:
        """
        Describe every image in image_dir. Matches COCO JSONs by stem name.
        Writes one JSON-LD per specimen to output_dir.
        """
        img_dir = Path(image_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(
            f for f in img_dir.iterdir()
            if f.suffix.lower() in _IMAGE_MEDIA_TYPES
        )

        results = []
        for img_path in images:
            stem = img_path.stem
            logger.info(f"Processing {img_path.name} ...")

            coco_json = None
            if coco_dir:
                coco_path = Path(coco_dir) / f"{stem}.json"
                if coco_path.exists():
                    with open(coco_path) as f:
                        coco_json = json.load(f)

            try:
                merged   = self.describe_specimen(
                    pipeline=pipeline,
                    taxon_name=taxon_name,
                    taxon_family=taxon_family,
                    image_path=str(img_path),
                    coco_json=coco_json,
                    k=k,
                )
                out_path = out_dir / f"{stem}.jsonld"
                with open(out_path, "w") as f:
                    json.dump(merged, f, indent=2)
                logger.info(f"  ✓ → {out_path.name}")
                results.append({"image": str(img_path),
                                "output_path": str(out_path),
                                "status": "ok"})
            except Exception as e:
                logger.error(f"  ✗ {img_path.name}: {e}")
                results.append({"image": str(img_path),
                                "output_path": None,
                                "status": "failed",
                                "error": str(e)})
        return results

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_system(self, prompt_data: Dict) -> str:
        """Pack system instructions + all retrieved treatment text into system prompt."""
        parts = [prompt_data.get("system", "")]
        if prompt_data.get("context"):
            parts.append(
                "\n\n# Retrieved taxonomic treatments (reference only)\n"
                + prompt_data["context"]
            )
        if prompt_data.get("vocabulary"):
            parts.append(
                "\n\n# ROSETTA controlled vocabulary\n"
                + prompt_data["vocabulary"]
            )
        return "\n".join(parts)

    def _build_user_content(self, prompt_data: Dict,
                            image_path: Optional[str]) -> List:
        """Build user-turn content list (optional image block + text block)."""
        content = []
        if image_path:
            img_b64, media_type = self._load_image(image_path)
            content.append({
                "type": "image",
                "source": {"type": "base64",
                           "media_type": media_type,
                           "data": img_b64},
            })
        user_text = prompt_data.get("user_prompt", "Describe this specimen.")
        user_text += (
            "\n\nRespond ONLY with valid JSON-LD matching the schema below. "
            "No markdown fences, no preamble.\n\n"
            + json.dumps(prompt_data.get("output_schema", {}), indent=2)
        )
        content.append({"type": "text", "text": user_text})
        return content

    def _load_image(self, path: str):
        """Return (base64_string, media_type) for an image file."""
        p = Path(path)
        media_type = _IMAGE_MEDIA_TYPES.get(p.suffix.lower(), "image/jpeg")
        with open(p, "rb") as f:
            return base64.standard_b64encode(f.read()).decode(), media_type

    def _call_with_retry(self, system: str, user_content: List) -> Dict:
        """Call Claude with retries and JSON parsing."""
        last_err = None
        raw = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_content}],
                )
                raw = response.content[0].text.strip()
                raw = re.sub(r"^```(?:json(?:-ld)?)?\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
                return json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt}: JSON parse error — {e}")
                if attempt == self.max_retries:
                    recovered = self._recover_partial_json(raw)
                    recovered["_raw_response"] = raw
                    recovered["_parse_error"] = str(e)
                    return recovered
                last_err = e
            except Exception as e:
                logger.warning(f"Attempt {attempt}: API error — {e}")
                last_err = e
            time.sleep(self.retry_delay * attempt)
        raise RuntimeError(
            f"Claude API failed after {self.max_retries} attempts: {last_err}"
        )

    @staticmethod
    def _recover_partial_json(raw: str) -> Dict:
        """Extract diagnosis/description text from truncated JSON-LD responses."""
        result = {}
        for key in ("diagnosis", "description"):
            pattern = rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)'
            m = re.search(pattern, raw, re.DOTALL)
            if m:
                text = m.group(1)
                text = text.replace("\\n", "\n").replace('\\"', '"')
                eo = result.setdefault("expected_output", {})
                eo[key] = text
                logger.info(f"    Recovered partial '{key}' ({len(text)} chars)")
        if "traits" not in result.get("expected_output", {}):
            traits_m = re.search(r'"traits"\s*:\s*\[', raw)
            if traits_m:
                bracket_start = traits_m.end() - 1
                depth = 0
                end = len(raw)
                for i in range(bracket_start, len(raw)):
                    if raw[i] == '[': depth += 1
                    elif raw[i] == ']': depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
                try:
                    traits = json.loads(raw[bracket_start:end])
                    result.setdefault("expected_output", {})["traits"] = traits
                except json.JSONDecodeError:
                    pass
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# BioRAG — Taxonomic Observation Workflow with Literature-Enriched sYnthesis
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False


def load_compiled_data(compiled_dir: str) -> Dict:
    """Load outputs from compile_specimen_data.py."""
    if not HAS_PANDAS:
        raise ImportError("pandas required for BioRAG: pip install pandas")

    base = Path(compiled_dir)
    data = {}

    sp_path = base / "diaphorina_species_summary.csv"
    if sp_path.exists():
        data["species_summary"] = pd.read_csv(sp_path)
        logger.info(f"Species summary: {len(data['species_summary'])} rows")

    dr_path = base / "diaphorina_diagnostic_report.json"
    if dr_path.exists():
        with open(dr_path) as f:
            data["diagnostic_report"] = json.load(f)
        n_cats = len(data["diagnostic_report"].get("categories", {}))
        logger.info(f"Diagnostic report: {n_cats} categories")

    full_path = base / "diaphorina_full_features.csv"
    if full_path.exists():
        data["full_features"] = pd.read_csv(full_path, low_memory=False)
        logger.info(f"Full features: {data['full_features'].shape}")

    pw_path = base / "diaphorina_pairwise_tests.csv"
    if pw_path.exists():
        data["pairwise"] = pd.read_csv(pw_path)

    summaries_dir = base / "specimen_summaries"
    if summaries_dir.exists():
        data["specimen_summaries_dir"] = str(summaries_dir)

    return data


def generate_taxon_schema(compiled: Dict, coco_json: Dict,
                          output_path: str, family: Optional[str] = None,
                          user_prompts: Optional[Dict[str, List[str]]] = None) -> Dict:
    """Generate a universal taxon schema from compiled data + COCO JSON.

    The schema describes:
      - All annotated categories with specimen counts and available data pipelines
      - Body region groupings inferred from category names using REGION_PATTERNS
      - Species list with per-category specimen counts
      - Top diagnostic features per category (for prompt enrichment)
      - User prompt mapping per category (which taxonomist questions apply)

    This schema file makes BioRAG universal: it reads the organism's data structure
    from the schema rather than relying on hardcoded organism-specific mappings.
    """
    schema = {
        "schema_version": "1.0",
        "generated": datetime.now().isoformat(),
        "family": family or "unknown",
    }

    # --- Extract COCO category info ---
    coco_cats = {c["id"]: c["name"] for c in coco_json.get("categories", [])}
    ann_counts = {}
    for ann in coco_json.get("annotations", []):
        cid = ann.get("category_id")
        ann_counts[cid] = ann_counts.get(cid, 0) + 1

    # --- Infer body region for each category using REGION_PATTERNS ---
    def infer_region(cat_name: str) -> str:
        cat_lower = cat_name.lower()
        cat_normed = cat_lower.replace("-", " ").replace("_", " ")
        cat_tokens = set(cat_normed.split())
        best_region = "other"
        best_score = 0
        for region, patterns in REGION_PATTERNS.items():
            for pat in patterns:
                matched = False
                if pat in cat_lower or pat in cat_normed:
                    matched = True
                elif pat in cat_tokens:
                    matched = True
                if matched:
                    score = len(pat)
                    if score < 4 and pat not in cat_tokens:
                        continue
                    if score > best_score:
                        best_score = score
                        best_region = region
        return best_region

    # --- Build category entries from diagnostic report ---
    diag_report = compiled.get("diagnostic_report", {})
    categories_data = {}

    for cat, cat_info in diag_report.get("categories", {}).items():
        region = infer_region(cat)
        n_diag = cat_info.get("n_diagnostic", 0)
        n_tested = cat_info.get("n_tested", 0)
        top_features = cat_info.get("diagnostic_features", [])[:10]

        # Group top features by pipeline prefix
        pipeline_features = {}
        for feat in top_features:
            prefix = feat.split("_")[0] if "_" in feat else "other"
            pipeline_features.setdefault(prefix, []).append(feat)

        # Match user prompts if provided
        matched_prompts = []
        if user_prompts:
            matched_prompts = _match_prompts_to_category(user_prompts, cat)

        categories_data[cat] = {
            "body_region": region,
            "n_diagnostic_features": n_diag,
            "n_total_features": n_tested,
            "top_diagnostic_features": top_features,
            "pipeline_feature_groups": pipeline_features,
            "n_user_prompts_matched": len(matched_prompts),
        }

    # --- Add COCO annotation counts ---
    for cid, cname in coco_cats.items():
        norm_name = cname.lower().replace(" ", "_").replace("-", "_")
        for cat in categories_data:
            if cat.lower().replace(" ", "_").replace("-", "_") == norm_name:
                categories_data[cat]["coco_annotation_count"] = ann_counts.get(cid, 0)
                categories_data[cat]["coco_category_id"] = cid
                break

    schema["categories"] = categories_data

    # --- Group categories by body region ---
    region_groups = {}
    for cat, info in categories_data.items():
        region = info["body_region"]
        region_groups.setdefault(region, []).append(cat)
    schema["body_regions"] = {r: sorted(cats) for r, cats in sorted(region_groups.items())}

    # --- Species list with coverage ---
    full_df = compiled.get("full_features")
    species_data = {}
    if full_df is not None:
        for sp in sorted(full_df["group_label"].dropna().unique()):
            sp_rows = full_df[full_df["group_label"] == sp]
            sp_cats = sorted(sp_rows["category"].dropna().unique())
            species_data[sp] = {
                "n_specimens": int(sp_rows.drop_duplicates("image_base").shape[0]
                                   if "image_base" in sp_rows.columns else len(sp_rows)),
                "n_annotations": len(sp_rows),
                "categories": sp_cats,
            }
    schema["species"] = species_data

    # --- Pipeline summary ---
    coverage_cols = [c for c in (full_df.columns if full_df is not None else [])
                     if c.startswith("has_")]
    pipelines_available = []
    if full_df is not None:
        for col in coverage_cols:
            n_true = int((full_df[col] == True).sum())  # noqa: E712
            if n_true > 0:
                pipelines_available.append({
                    "name": col.replace("has_", ""),
                    "n_specimens_with_data": n_true,
                })
    schema["pipelines"] = pipelines_available

    # --- Write YAML (or JSON fallback) ---
    out_path = Path(output_path)
    try:
        import yaml
        with open(out_path, "w") as f:
            yaml.dump(schema, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True, width=120)
        logger.info(f"Taxon schema written: {out_path} (YAML)")
    except ImportError:
        json_path = out_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(schema, f, indent=2, default=str)
        logger.info(f"Taxon schema written: {json_path} (JSON, pyyaml not installed)")
        out_path = json_path

    # --- Summary ---
    n_cats = len(categories_data)
    n_species = len(species_data)
    n_regions = len(region_groups)
    logger.info(f"Taxon schema: {n_cats} categories in {n_regions} body regions, "
                f"{n_species} species")

    return schema


def extract_specimen_sex(image_base: str) -> str:
    """Extract sex from specimen filename convention (_m_ = male, _f_ = female)."""
    fn = image_base.lower()
    if "_m_" in fn or "_male" in fn:
        return "male"
    elif "_f_" in fn or "_female" in fn:
        return "female"
    return "unknown"


def designate_type_series(compiled: Dict, group_label: str,
                          group_labels_df=None) -> Dict:
    """Designate holotype, allotype, and paratypes for a species.

    Holotype = specimen with broadest data coverage (most non-null features).
    Allotype = paratype of opposite sex to holotype (if available).
    Returns dict with holotype, allotype, paratypes, and sex breakdown.
    """
    full = compiled.get("full_features")
    if full is None:
        return {}

    sp_rows = full[full["group_label"] == group_label]
    if sp_rows.empty:
        return {}

    specimens = sorted(sp_rows["image_base"].unique())
    if not specimens:
        return {}

    specimen_sex = {img: extract_specimen_sex(img) for img in specimens}

    # Unique specimen stems (same specimen imaged for multiple categories)
    # Group by removing the category-specific parts - use the image base directly
    # since each image = one specimen×view
    # Count how many categories each specimen base covers
    stem_coverage = {}
    for img in specimens:
        n_cats = len(sp_rows[sp_rows["image_base"] == img]["category"].unique())
        numeric_cols = sp_rows[sp_rows["image_base"] == img].select_dtypes(
            include="number"
        )
        n_features = int(numeric_cols.notna().sum().sum())
        stem_coverage[img] = (n_cats, n_features)

    # Get unique specimen identifiers (deduplicate across body regions)
    # Specimens share a common naming pattern minus the body part
    specimen_ids = {}
    for img in specimens:
        parts = img.lower().split("_")
        # Find the morph ID: everything between "morph_" and body part
        morph_idx = None
        for i, p in enumerate(parts):
            if p == "morph":
                morph_idx = i
                break
        if morph_idx is not None and morph_idx + 1 < len(parts):
            spec_id = parts[morph_idx + 1]
            # Include number suffix if present (e.g., "acok1", "sp1_1")
            if morph_idx + 2 < len(parts):
                next_part = parts[morph_idx + 2]
                if next_part.isdigit():
                    spec_id = f"{spec_id}_{next_part}"
        else:
            spec_id = img
        if spec_id not in specimen_ids:
            specimen_ids[spec_id] = []
        specimen_ids[spec_id].append(img)

    # Pick holotype: specimen ID with highest total coverage
    best_id = None
    best_score = (-1, -1)
    for spec_id, imgs in specimen_ids.items():
        total_cats = sum(stem_coverage.get(i, (0, 0))[0] for i in imgs)
        total_feats = sum(stem_coverage.get(i, (0, 0))[1] for i in imgs)
        if (total_cats, total_feats) > best_score:
            best_score = (total_cats, total_feats)
            best_id = spec_id

    holotype_imgs = specimen_ids.get(best_id, [])
    holotype_sex = "unknown"
    for img in holotype_imgs:
        s = specimen_sex.get(img, "unknown")
        if s != "unknown":
            holotype_sex = s
            break

    # Allotype: opposite sex, best coverage among paratypes
    opposite = "female" if holotype_sex == "male" else "male"
    allotype_id = None
    allotype_score = (-1, -1)
    for spec_id, imgs in specimen_ids.items():
        if spec_id == best_id:
            continue
        spec_sex = "unknown"
        for img in imgs:
            s = specimen_sex.get(img, "unknown")
            if s != "unknown":
                spec_sex = s
                break
        if spec_sex == opposite:
            total_cats = sum(stem_coverage.get(i, (0, 0))[0] for i in imgs)
            total_feats = sum(stem_coverage.get(i, (0, 0))[1] for i in imgs)
            if (total_cats, total_feats) > allotype_score:
                allotype_score = (total_cats, total_feats)
                allotype_id = spec_id

    # Build paratype list
    paratypes = []
    for spec_id, imgs in specimen_ids.items():
        if spec_id == best_id:
            continue
        spec_sex = "unknown"
        for img in imgs:
            s = specimen_sex.get(img, "unknown")
            if s != "unknown":
                spec_sex = s
                break
        entry = {
            "specimen_id": spec_id,
            "images": imgs,
            "sex": spec_sex,
            "is_allotype": spec_id == allotype_id,
        }
        paratypes.append(entry)

    males = [s for s in specimen_ids if any(
        specimen_sex.get(i) == "male" for i in specimen_ids[s])]
    females = [s for s in specimen_ids if any(
        specimen_sex.get(i) == "female" for i in specimen_ids[s])]

    return {
        "holotype": {
            "specimen_id": best_id,
            "images": holotype_imgs,
            "sex": holotype_sex,
            "n_categories": best_score[0],
            "n_features": best_score[1],
        },
        "allotype": {
            "specimen_id": allotype_id,
            "images": specimen_ids.get(allotype_id, []),
            "sex": opposite,
        } if allotype_id else None,
        "paratypes": paratypes,
        "n_specimens": len(specimen_ids),
        "n_males": len(males),
        "n_females": len(females),
        "sex_dimorphism_possible": len(males) > 0 and len(females) > 0,
    }


def format_type_material_text(type_series: Dict) -> str:
    """Format type series designation as taxonomic text for prompts and output."""
    if not type_series:
        return ""

    parts = []
    holo = type_series.get("holotype", {})
    if holo:
        sex_str = f" ({holo['sex']})" if holo.get("sex", "unknown") != "unknown" else ""
        parts.append(f"HOLOTYPE: {holo['specimen_id']}{sex_str}")

    allo = type_series.get("allotype")
    if allo and allo.get("specimen_id"):
        parts.append(f"ALLOTYPE: {allo['specimen_id']} ({allo['sex']})")

    paratypes = type_series.get("paratypes", [])
    if paratypes:
        non_allo = [p for p in paratypes if not p.get("is_allotype")]
        if non_allo:
            pt_strs = []
            for p in non_allo:
                sex = f" ({p['sex']})" if p.get("sex", "unknown") != "unknown" else ""
                pt_strs.append(f"{p['specimen_id']}{sex}")
            parts.append(f"PARATYPES ({len(non_allo)}): {', '.join(pt_strs)}")

    n_m = type_series.get("n_males", 0)
    n_f = type_series.get("n_females", 0)
    parts.append(f"Total: {type_series.get('n_specimens', 0)} specimens "
                 f"({n_m} males, {n_f} females)")

    return "\n".join(parts)


def format_species_diagnosis(compiled: Dict, group_label: str,
                             category: str) -> str:
    """Format compiled diagnostic data for one species+category into prompt text."""
    sp_summary = compiled.get("species_summary")
    if sp_summary is None:
        return ""

    sp_cat = sp_summary[
        (sp_summary["group_label"] == group_label) &
        (sp_summary["category"] == category)
    ]

    if sp_cat.empty:
        return ""

    diag = sp_cat[sp_cat["is_diagnostic"]].copy()
    if diag.empty:
        return ""

    # Prioritize interpretable features over raw grid-cell texture/homology
    diag_report = compiled.get("diagnostic_report", {})
    cat_tests = diag_report.get("categories", {}).get(category, {}).get("tests", [])
    eta_lookup = {t["feature"]: t.get("eta_squared", 0) for t in cat_tests}
    diag["eta_sq"] = diag["feature"].map(eta_lookup).fillna(0)

    # Separate interpretable from grid-cell features
    grid_pattern = re.compile(r"^(tex_r\d+c\d+|colhom_r\d+c\d+)")
    diag["is_grid"] = diag["feature"].apply(lambda f: bool(grid_pattern.match(f)))
    interpretable = diag[~diag["is_grid"]].sort_values("eta_sq", ascending=False)
    grid_feats = diag[diag["is_grid"]].sort_values("eta_sq", ascending=False)

    parts = [f"=== Quantitative measurements for {group_label} — {category} ==="]
    parts.append("(diagnostic features ranked by discriminatory power)\n")

    # Show up to 20 interpretable + 5 grid summary
    show_feats = pd.concat([interpretable.head(20), grid_feats.head(5)])

    for _, row in show_feats.iterrows():
        feat = row["feature"]
        n = int(row["n"])
        if n == 1:
            parts.append(f"  {feat}: {row['mean']:.4g} (n=1)")
        else:
            parts.append(
                f"  {feat}: {row['min']:.4g}–{row['max']:.4g} "
                f"(mean {row['mean']:.4g} ± {row['std']:.4g}, n={n}, "
                f"CV={row['cv_percent']:.1f}%)"
            )

    # Pairwise species comparisons
    diag_report = compiled.get("diagnostic_report", {})
    cat_report = diag_report.get("categories", {}).get(category, {})
    pw_sig = cat_report.get("pairwise_significant", [])

    sp_pw = [p for p in pw_sig
             if p.get("group_a") == group_label or p.get("group_b") == group_label]

    if sp_pw:
        parts.append(f"\n  Pairwise differences (Dunn's post-hoc, FDR-corrected):")
        seen = set()
        for p in sp_pw:
            other = p["group_b"] if p["group_a"] == group_label else p["group_a"]
            feat = p["feature"]
            key = (other, feat)
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > 15:
                break
            parts.append(f"    differs from {other} in {feat} (p={p['p_value_dunn']:.4f})")

    return "\n".join(parts)


def format_all_species_data(compiled: Dict, group_label: str) -> str:
    """Format ALL category data for a species into a single context block."""
    sp_summary = compiled.get("species_summary")
    if sp_summary is None:
        return ""

    sp_data = sp_summary[sp_summary["group_label"] == group_label]
    categories = sorted(sp_data["category"].unique())

    parts = []
    for cat in categories:
        cat_text = format_species_diagnosis(compiled, group_label, cat)
        if cat_text:
            parts.append(cat_text)

    return "\n\n".join(parts)


def find_foreground_mask(semilandmarks_dirs, image_base: str,
                         category: str) -> Optional[str]:
    """Find the foreground mask PNG for a specimen+category.

    semilandmarks_dirs can be a single path string or a list of paths.
    """
    if isinstance(semilandmarks_dirs, str):
        semilandmarks_dirs = [semilandmarks_dirs]

    stem = Path(image_base).stem
    stem_clean = stem.replace(" ", "_")

    for semilandmarks_dir in semilandmarks_dirs:
        base = Path(semilandmarks_dir)
        fg_dir = base / category / category
        if not fg_dir.exists():
            continue

        candidates = list(fg_dir.glob(f"*{stem}*_fg_{category}.png"))
        if candidates:
            return str(candidates[0])

        candidates = list(fg_dir.glob(f"*{stem_clean}*_fg_{category}.png"))
        if candidates:
            return str(candidates[0])
    return None


def create_contour_image(image_path: str, coco_json: Dict,
                         category: str, output_path: str) -> Optional[str]:
    """Create an image with segmentation contour overlay highlighting the target category."""
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available — contour images disabled")
        return None

    img = cv2.imread(image_path)
    if img is None:
        return None

    cat_id = None
    for cat in coco_json.get("categories", []):
        if cat["name"].replace(" ", "_") == category:
            cat_id = cat["id"]
            break
    if cat_id is None:
        return None

    drawn = False
    for ann in coco_json.get("annotations", []):
        if ann.get("category_id") != cat_id:
            continue
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                if isinstance(poly, list) and len(poly) >= 6:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                    cv2.drawContours(img, [pts], -1, (0, 255, 0), 3)
                    drawn = True

    if drawn:
        cv2.imwrite(output_path, img)
        return output_path
    return None


def pick_representative_specimens(compiled: Dict, group_label: str,
                                  category: str, n: int = 3) -> List[str]:
    """Pick n representative specimen image_bases for a species+category."""
    full = compiled.get("full_features")
    if full is None:
        return []

    sp_cat = full[
        (full["group_label"] == group_label) &
        (full["category"] == category)
    ]

    if sp_cat.empty:
        return []

    if len(sp_cat) <= n:
        return sp_cat["image_base"].tolist()

    # Pick specimens spread across size range (centroid_size or first numeric)
    size_col = None
    for c in ["shape_centroid_size_mm", "shape_centroid_size",
              "meas_length_mm", "meas_area_mm2"]:
        if c in sp_cat.columns and sp_cat[c].notna().any():
            size_col = c
            break

    if size_col:
        sorted_df = sp_cat.sort_values(size_col).dropna(subset=[size_col])
        if len(sorted_df) >= n:
            indices = np.linspace(0, len(sorted_df) - 1, n, dtype=int)
            return sorted_df.iloc[indices]["image_base"].tolist()

    return sp_cat.sample(min(n, len(sp_cat)))["image_base"].tolist()


_FLORENCE2_MODEL_BioRAG = None
_FLORENCE2_PROCESSOR_BioRAG = None


def _load_florence2_biorag():
    """Load Florence-2 for region captioning."""
    global _FLORENCE2_MODEL_BioRAG, _FLORENCE2_PROCESSOR_BioRAG
    if _FLORENCE2_MODEL_BioRAG is not None:
        return _FLORENCE2_MODEL_BioRAG, _FLORENCE2_PROCESSOR_BioRAG
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _FLORENCE2_PROCESSOR_BioRAG = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True
        )
        _FLORENCE2_MODEL_BioRAG = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True
        ).to(device)
        logger.info(f"Florence-2 loaded on {device}")
        return _FLORENCE2_MODEL_BioRAG, _FLORENCE2_PROCESSOR_BioRAG
    except Exception as e:
        logger.warning(f"Florence-2 unavailable ({e})")
        return None, None


def caption_region_florence2(image_path: str) -> str:
    """Run Florence-2 <MORE_DETAILED_CAPTION> on an image region."""
    model, processor = _load_florence2_biorag()
    if model is None:
        return ""

    try:
        import torch
        from PIL import Image
        device = next(model.parameters()).device
        img = Image.open(image_path).convert("RGB")
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=256, num_beams=3
            )
        result = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            result, task=prompt, image_size=(img.width, img.height)
        )
        return parsed.get(prompt, "").strip()
    except Exception as e:
        logger.warning(f"Florence-2 caption failed: {e}")
        return ""


def load_user_prompts(path: str) -> Dict[str, List[str]]:
    """Load taxonomist-provided prompts from .docx, .csv, .xlsx, or .txt.

    Returns dict mapping body region → list of questions.
    """
    p = Path(path)
    ext = p.suffix.lower()
    prompts: Dict[str, List[str]] = {}

    if ext == ".docx":
        from docx import Document
        doc = Document(str(p))
        current_section = "general"
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text or text == "XXX":
                continue
            is_question = ("?" in text or text.lower().startswith("what ")
                           or text.lower().startswith("is the ")
                           or text.lower().startswith("are the "))
            if not is_question and len(text.split()) <= 5:
                current_section = text.lower().replace(" ", "_")
                prompts.setdefault(current_section, [])
            else:
                prompts.setdefault(current_section, []).append(text)

    elif ext in (".csv", ".tsv"):
        import csv
        delimiter = "\t" if ext == ".tsv" else ","
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)
            region_col = 0
            question_col = 1
            if header:
                h_lower = [h.lower() for h in header]
                for i, h in enumerate(h_lower):
                    if "region" in h or "category" in h or "section" in h:
                        region_col = i
                    if "question" in h or "prompt" in h or "text" in h:
                        question_col = i
            for row in reader:
                if len(row) > max(region_col, question_col):
                    region = row[region_col].strip().lower().replace(" ", "_") or "general"
                    question = row[question_col].strip()
                    if question:
                        prompts.setdefault(region, []).append(question)

    elif ext in (".xlsx", ".xls"):
        import openpyxl
        wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return prompts
        header = [str(h or "").lower() for h in rows[0]]
        region_col = 0
        question_col = 1
        for i, h in enumerate(header):
            if "region" in h or "category" in h or "section" in h:
                region_col = i
            if "question" in h or "prompt" in h or "text" in h:
                question_col = i
        for row in rows[1:]:
            if len(row) > max(region_col, question_col):
                region = str(row[region_col] or "general").strip().lower().replace(" ", "_")
                question = str(row[question_col] or "").strip()
                if question:
                    prompts.setdefault(region, []).append(question)
        wb.close()

    elif ext == ".txt":
        current_section = "general"
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not line.endswith("?"):
                    current_section = line.lower().replace(" ", "_")
                    prompts.setdefault(current_section, [])
                else:
                    prompts.setdefault(current_section, []).append(line)
    else:
        logger.warning(f"Unsupported prompt file format: {ext}")

    total = sum(len(v) for v in prompts.values())
    logger.info(f"User prompts loaded: {total} questions across {len(prompts)} sections from {p.name}")
    return prompts


def _match_prompts_to_category(user_prompts: Dict[str, List[str]],
                                category: str) -> List[str]:
    """Find user prompts relevant to a COCO category name.

    Uses a universal matching strategy that works for any organism:
      1. Direct substring match between prompt section and category name
      2. Word-level overlap between category tokens and section/question text
      3. 'general' section always included
    No hardcoded organism-specific mappings.
    """
    cat_lower = category.lower().replace("-", "_").replace(" ", "_")
    cat_tokens = set(re.split(r"[_\-\s]+", cat_lower))
    cat_tokens.discard("")

    matched = []
    seen = set()

    for section, questions in user_prompts.items():
        sec_tokens = set(re.split(r"[_\-\s]+", section.lower()))

        if cat_lower in section or section in cat_lower:
            for q in questions:
                if q not in seen:
                    matched.append(q)
                    seen.add(q)
            continue

        if cat_tokens & sec_tokens:
            for q in questions:
                if q not in seen:
                    matched.append(q)
                    seen.add(q)
            continue

        for q in questions:
            q_lower = q.lower()
            if any(tok in q_lower for tok in cat_tokens if len(tok) > 3):
                if q not in seen:
                    matched.append(q)
                    seen.add(q)

    for q in user_prompts.get("general", []):
        if q not in seen:
            matched.append(q)
            seen.add(q)

    return matched


class BioragDescriber:
    """
    BioRAG species description pipeline.

    Combines:
      1. Compiled diagnostic data (measurements, shape, color, texture, distances)
      2. Visual evidence from foreground masks / contour-guided images
      3. Florence-2 initial captioning (optional)
      4. RAG-retrieved literature context
      5. Claude API for final structured descriptions
      6. User-provided taxonomist prompts (optional)
    """

    SYSTEM_PROMPT = """You are an expert taxonomist writing formal species descriptions and diagnoses.

You will receive:
1. QUANTITATIVE MEASUREMENT DATA for this species (ranges, means ± SD, sample sizes)
2. DIAGNOSTIC FEATURES that statistically separate this species from congeners (Kruskal-Wallis + Dunn's tests)
3. A SPECIMEN IMAGE showing a specific body region (either isolated foreground mask or contour-highlighted)
4. RETRIEVED TREATMENT TEXT from published taxonomic literature for terminology reference
5. Optionally, a FLORENCE-2 CAPTION — a machine-generated preliminary description

CRITICAL RULES:
- The quantitative data is your PRIMARY evidence. Use exact numbers, ranges, and statistics.
- Use the image to describe QUALITATIVE features not captured by measurements: color, texture, surface sculpture, setae, overall shape
- ONLY describe features VISIBLE in the image — do NOT hallucinate features from other body regions
- The image shows ONLY the {category} — do not describe other structures
- Use the retrieved treatments as TERMINOLOGY REFERENCE, not as a source of character states
- Report intraspecific variation: "length 0.42–0.51 mm (mean 0.47 ± 0.02, n=6)"
- In the diagnosis section, cite which species the current one differs from and in what character
- Use ROSETTA controlled vocabulary for qualitative terms where available
- EVERY numeric value MUST have explicit units. No exceptions. Examples:
    Length: "0.395 mm", Area: "0.0015 mm²", Perimeter: "0.917 mm"
    Dimensionless ratios: "aspect ratio 6.69 (dimensionless)"
    Procrustes distances: "lmk_dist_8_15 = 0.374 (Procrustes-normalized, dimensionless)"
    CIE LAB values: "L* = 34.5 (CIE LAB, 0–255 scale)", "b* = 151.0 (CIE LAB, 0–255 scale; >128 = yellowish)"
    HSV values: "brightness 124.4 (HSV V-channel, 0–255 scale)", "hue 42° (OpenCV HSV, 0–180°)"
    Chroma: "chroma_ab = 198.9 (CIE LAB derived, dimensionless)"
    Boundary areas: "n_boundaries_same_mm2 = 14.5 mm²" (prefer _mm2 column over raw pixel count)
    Boundary lengths: "n_boundaries_diff_mm = 0.42 mm" (prefer _mm column over raw pixel count)
    Entropy: "color_entropy = 2.31 (Shannon entropy, bits)"
    Proportions: "solidity 0.94 (dimensionless, 0–1 scale)"
    Cluster assignments: "texture cluster 2 (categorical)"
    PC scores / UMAP coordinates: "PC1 = -1.225 (dimensionless)" or "UMAP2 = 4.06 (dimensionless)"
  A referee must be able to verify every value against the raw data without guessing units.
- Output structured JSON-LD matching the schema below

# FEATURE NAME GLOSSARY — how to interpret and present measured features

When reporting quantitative color/texture features, ALWAYS provide:
  (a) the raw feature name and numeric value (for reproducibility)
  (b) a short human-readable interpretation in parentheses

## Measurement features (prefix: meas_)
  meas_length_mm          — maximum Feret length of the structure (mm)
  meas_height_mm          — maximum Feret width perpendicular to length (mm)
  meas_area_mm2           — mask area (mm²)
  meas_perimeter_mm       — mask perimeter (mm)
  meas_equivalent_diameter_mm — diameter of circle with equal area (mm)
  meas_aspect_ratio       — length/height (dimensionless)
  meas_solidity           — mask area / convex hull area (dimensionless, 0–1; 1=convex)
  meas_extent             — mask area / bounding-box area (dimensionless, 0–1)
  meas_orientation        — angle of major axis to horizontal (degrees)

## Shape semilandmark features (prefix: shape_)
  shape_centroid_size_mm  — Procrustes centroid size calibrated to mm (overall structure size)
  shape_phylo_PC1..N      — principal components of Procrustes-aligned shape
  shape_phylo_UMAP1/2     — UMAP embedding of shape
  shape_phylo_cluster     — discrete shape cluster assignment

## Color extraction features (prefix: color_)
  Two thresholding methods: "adaptive" and "median" — both measure the SAME color space (HSV).
  color_*_hue_mean        — mean hue angle (0–180 in OpenCV HSV; 0=red, 30=yellow, 60=green, 90=cyan, 120=blue, 150=magenta)
  color_*_hue_sin/cos     — sin/cos encoding of circular hue (preserves wrap-around; interpret via atan2)
  color_*_sat_mean/std    — mean/SD saturation (0=gray, 255=fully saturated)
  color_*_bri_mean/std    — mean/SD brightness (HSV Value channel; 0=black, 255=white)
  color_*_max_power       — dominant spatial frequency power (Fourier; higher=stronger periodic pattern)
  color_*_max_freq        — dominant spatial frequency (cycles/pixel; higher=finer pattern)
  color_*_sum_power       — total spectral power (overall pattern energy)
  color_*_pattern_complexity — number of significant Fourier peaks (higher=more complex patterning)
  color_*_boundary_strength — mean ΔE at color boundaries (dimensionless CIE color distance; higher=sharper transitions)
  color_*_n_boundaries_same — same-label pixel adjacency count (raw pixel pairs; higher=more uniform color)
  color_*_n_boundaries_same_mm2 — same as above calibrated to mm² (area-proportional; prefer this over raw count)
  color_*_n_boundaries_diff — cross-label pixel adjacency count (raw pixel pairs; traces internal color boundaries)
  color_*_n_boundaries_diff_mm — same as above calibrated to mm (boundary perimeter; prefer this over raw count)
  color_*_n_markings      — number of distinct color patches
  color_*_mean_aspect_ratio — mean aspect ratio of color patches
  color_*_mean_bending_energy — mean curvature energy of patch contours (higher=more irregular patches)
  color_*_mean_shape_complexity — mean Fourier complexity of patch contours
  color_*_PC1..N          — PCA of the full color feature vector
  color_*_total_marking_area_mm2 — total area of detected color markings (mm²)

## Color homology features (prefix: colhom_)
  These are measured in CIE LAB color space on a spatial grid overlaid on the structure.
  The grid divides the bounding box into rows (dorsal→ventral) × columns (proximal→distal):
    Row 0 = dorsal/anterior margin       Row 7 = ventral/posterior margin
    Col 0 = proximal/basal end           Col 9 = distal/apical end
    Mid rows (3–4) = central/median       Mid cols (4–5) = mid-length
  When reporting grid cell features, ALWAYS state the approximate anatomical position, e.g.:
    "colhom_r1c8_L_mean_abs = 195.2 (lightness in the dorso-apical region)"
    "colhom_r4c2_a_mean_abs = 142.3 (red-green axis in the ventro-basal region; >128 = reddish)"

  colhom_rNcM_L_mean_abs  — mean CIE L* lightness in grid cell (0=black, 255=white)
  colhom_rNcM_a_mean_abs  — mean CIE a* in grid cell (0–255 rescaled; 128=neutral, <128=greenish, >128=reddish)
  colhom_rNcM_b_mean_abs  — mean CIE b* in grid cell (0–255 rescaled; 128=neutral, <128=bluish, >128=yellowish)
  colhom_rNcM_L_std       — SD of lightness (higher=more L* variation within cell)
  colhom_rNcM_a_std       — SD of a* channel within cell
  colhom_rNcM_b_std       — SD of b* channel within cell
  colhom_rNcM_dL_rel      — L* difference relative to whole-structure mean (darker or lighter than average)
  colhom_rNcM_da_rel      — a* shift relative to mean (more reddish or greenish than average)
  colhom_rNcM_db_rel      — b* shift relative to mean (more yellowish or bluish than average)
  colhom_rNcM_color_entropy — Shannon entropy of color distribution in cell (higher=more diverse colors)
  colhom_rNcM_dominant_proportion — proportion of cell occupied by dominant color (lower=more mixed)
  colhom_L_mean            — whole-structure mean CIE L* (overall lightness)
  colhom_a_mean            — whole-structure mean CIE a*
  colhom_b_mean            — whole-structure mean CIE b*
  colhom_chroma_ab         — sqrt(a*² + b*²) — overall color saturation in LAB
  colhom_hue_ab_sin/cos    — circular hue in LAB a*/b* plane
  colhom_phylo_PC1..N      — PCA of the full color homology vector
  colhom_phylo_UMAP1/2     — UMAP embedding of color homology
  colhom_phylo_cluster     — discrete color cluster assignment

## Texture homology features (prefix: tex_)
  Same spatial grid as color homology. Same row/column anatomical interpretation.
  tex_rNcM_*               — texture metrics per grid cell (LBP, GLCM, Gabor, etc.)
  tex_phylo_PC1..N         — PCA of the full texture vector
  tex_phylo_UMAP1/2        — UMAP embedding of texture
  tex_phylo_cluster        — discrete texture cluster assignment

## Landmark GPA features (prefix: lmk_)
  lmk_centroid_size_mm     — Procrustes centroid size of discrete landmarks (mm)
  lmk_PC1..N               — PCs of Procrustes-aligned landmark configurations
  lmk_dist_*               — inter-landmark distances (dimensionless, Procrustes-normalized)

## Inter-mask distances (prefix: imd_)
  imd_*_mm                 — distance between centroids of two annotation categories (mm)

## Presenting color values as human-readable colors
  When CIE LAB values are available, interpret them as approximate color names:
  L* < 30 → very dark / black-brown; L* 30–80 → medium; L* > 200 → very pale / white
  a* > 140 and b* > 140 → warm brown / ferruginous / testaceous
  a* < 120 and b* > 140 → olive / greenish-yellow
  a* > 140 and b* < 120 → purplish / magenta-brown
  For HSV hue: 0–15° = red, 15–30° = orange, 30–45° = yellow-orange, 45–75° = yellow-green, 75–105° = green
  Always state both the numeric value AND the color interpretation.
"""

    DIAGNOSIS_PROMPT = """Write a TAXONOMIC DIAGNOSIS for {group_label} based on the {category} region.

A diagnosis lists ONLY the characters that distinguish this species from its congeners.
Use the pairwise comparison data to identify which species differ and in what characters.

The quantitative data below shows which features are statistically diagnostic:

{species_data}

{florence_caption}

Examine the image and combine:
1. Quantitative characters (from the data, with ranges and statistics)
2. Qualitative characters (from the image: color, texture, sculpture, shape)

Output JSON-LD with separate "diagnosis" and "description" sections.
The diagnosis should be concise and focus ONLY on distinguishing features.
The description should be comprehensive, covering all observable features.
"""

    DESCRIPTION_PROMPT = """Write a MORPHOLOGICAL DESCRIPTION of the {category} of {group_label}.

Use the quantitative data as primary evidence and supplement with visual observations.

{species_data}

{florence_caption}

Describe all observable features of this {category}:
- Dimensions (from measurements, with ranges)
- Shape and proportions
- Surface texture and sculpture
- Color and color pattern
- Setae/pubescence (if visible)
- Any distinctive features

Report all measurements with species ranges and sample sizes.
Output JSON-LD following the schema below.
"""

    def __init__(self, claude_describer: ClaudeDescriber,
                 pipeline: Optional[DescriptionPipeline] = None,
                 use_florence2: bool = True,
                 cache_dir: Optional[Path] = None,
                 user_prompts: Optional[Dict[str, List[str]]] = None,
                 taxon_schema: Optional[Dict] = None):
        self.claude = claude_describer
        self.pipeline = pipeline
        self.use_florence2 = use_florence2
        self.user_prompts = user_prompts or {}
        self.taxon_schema = taxon_schema or {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._florence2_cache = self._load_florence2_cache()

    # ── Cache helpers ─────────────────────────────────────────────────────

    def _load_florence2_cache(self) -> Dict:
        if not self.cache_dir:
            return {}
        p = self.cache_dir / "florence2_captions.json"
        if p.exists():
            with open(p) as f:
                cache = json.load(f)
            logger.info(f"Florence-2 caption cache loaded: {len(cache)} entries")
            return cache
        return {}

    def _save_florence2_cache(self):
        if not self.cache_dir or not self._florence2_cache:
            return
        p = self.cache_dir / "florence2_captions.json"
        with open(p, "w") as f:
            json.dump(self._florence2_cache, f, indent=2)

    def _get_cached_response(self, species: str, category: str,
                             vision_mode: str) -> Optional[Dict]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / species / f"{category}_{vision_mode}.json"
        if p.exists():
            with open(p) as f:
                cached = json.load(f)
            logger.info(f"    [{vision_mode}] cache hit: {category}")
            return cached
        return None

    def _save_cached_response(self, species: str, category: str,
                              vision_mode: str, result: Dict):
        if not self.cache_dir:
            return
        sp_dir = self.cache_dir / species
        sp_dir.mkdir(parents=True, exist_ok=True)
        p = sp_dir / f"{category}_{vision_mode}.json"
        with open(p, "w") as f:
            json.dump(result, f, indent=2, default=str)

    def _get_cached_florence2(self, image_path: str) -> Optional[str]:
        key = str(Path(image_path).name)
        return self._florence2_cache.get(key)

    def _save_cached_florence2(self, image_path: str, caption: str):
        key = str(Path(image_path).name)
        self._florence2_cache[key] = caption
        self._save_florence2_cache()

    def _schema_context_for_category(self, category: str, group_label: str) -> str:
        """Build schema-derived context text for a category+species prompt."""
        if not self.taxon_schema:
            return ""
        cat_info = self.taxon_schema.get("categories", {}).get(category)
        if not cat_info:
            return ""

        lines = [f"=== TAXON SCHEMA: {category} ==="]
        lines.append(f"Body region: {cat_info.get('body_region', 'unknown')}")

        region = cat_info.get("body_region", "")
        sibling_cats = self.taxon_schema.get("body_regions", {}).get(region, [])
        siblings = [c for c in sibling_cats if c != category]
        if siblings:
            lines.append(f"Related categories in {region}: {', '.join(siblings)}")

        n_diag = cat_info.get("n_diagnostic_features", 0)
        n_total = cat_info.get("n_total_features", 0)
        lines.append(f"Features: {n_diag} diagnostic / {n_total} total")

        top = cat_info.get("top_diagnostic_features", [])
        if top:
            pipe_groups = cat_info.get("pipeline_feature_groups", {})
            for pipe, feats in pipe_groups.items():
                pipe_labels = {"meas": "measurement", "shape": "shape/morphometric",
                               "color": "color", "colhom": "color homology",
                               "tex": "texture", "imd": "inter-mask distance",
                               "lmk": "landmark GPA"}
                label = pipe_labels.get(pipe, pipe)
                lines.append(f"  {label}: {', '.join(feats[:3])}")

        sp_info = self.taxon_schema.get("species", {}).get(group_label)
        if sp_info:
            lines.append(f"Species {group_label}: {sp_info.get('n_annotations', '?')} annotations "
                         f"across {len(sp_info.get('categories', []))} categories")

        n_species = len(self.taxon_schema.get("species", {}))
        lines.append(f"Total species in dataset: {n_species}")

        return "\n".join(lines)

    def describe_species(
        self,
        compiled: Dict,
        group_label: str,
        semilandmarks_dirs,
        image_dir: str,
        coco_json: Dict,
        output_dir: Path,
        vision_mode: str = "foreground",
        rosetta: Optional[Dict] = None,
        k: int = 5,
        label_dir: Optional[str] = None,
    ) -> Dict:
        """
        Generate a full species description across all diagnostic categories.

        Args:
            compiled: Output from load_compiled_data()
            group_label: Species identifier (e.g., "sp1", "cf.ender")
            semilandmarks_dirs: Path(s) to semilandmarks output (for foreground masks)
            image_dir: Path to original specimen images
            coco_json: Full COCO JSON with all annotations
            output_dir: Where to write outputs
            vision_mode: "foreground" | "contour" | "both"
            rosetta: ROSETTA vocabulary dict
            k: RAG retrieval count

        Returns:
            Merged species description JSON-LD
        """
        diag_report = compiled.get("diagnostic_report", {})
        diag_categories = set(diag_report.get("categories", {}).keys())

        # Also include categories this species has data for (even without
        # diagnostic features, e.g. head sclerites with few specimens)
        full_df = compiled.get("full_features")
        species_categories = set()
        if full_df is not None:
            sp_rows = full_df[full_df["group_label"] == group_label]
            species_categories = set(sp_rows["category"].dropna().unique())
        categories = sorted(diag_categories | species_categories)

        species_dir = output_dir / group_label
        species_dir.mkdir(parents=True, exist_ok=True)

        # Designate type series (holotype, allotype, paratypes)
        type_series = designate_type_series(compiled, group_label)
        type_material_text = format_type_material_text(type_series)
        if type_series:
            n_spec = type_series.get("n_specimens", 0)
            holo_sex = type_series.get("holotype", {}).get("sex", "?")
            has_allo = "yes" if type_series.get("allotype") else "no"
            logger.info(f"  Type series: {n_spec} specimens, "
                        f"holotype={type_series['holotype']['specimen_id']} ({holo_sex}), "
                        f"allotype={'designated' if has_allo == 'yes' else 'none'}")

        all_category_descriptions = []
        all_category_descriptions_contour = []

        for cat in categories:
            cat_features = diag_report.get("categories", {}).get(cat, {})
            n_diag = cat_features.get("n_diagnostic", 0)

            species_data = format_species_diagnosis(compiled, group_label, cat)

            # Pick representative specimens
            reps = pick_representative_specimens(compiled, group_label, cat, n=2)
            if not reps:
                logger.info(f"  [{cat}] No specimens for {group_label} — skipping")
                continue

            if not species_data and n_diag == 0:
                logger.info(f"  [{cat}] No diagnostic data — will describe from image only "
                            f"({len(reps)} specimens)")

            logger.info(f"  [{cat}] {n_diag} diagnostic features, "
                        f"{len(reps)} representative specimens")

            # Try foreground mask first
            fg_path = None
            contour_path = None
            florence_caption = ""

            for rep_img in reps:
                if vision_mode in ("foreground", "both"):
                    fg_path = find_foreground_mask(semilandmarks_dirs, rep_img, cat)
                    if fg_path:
                        break

            if fg_path and self.use_florence2:
                cached_cap = self._get_cached_florence2(fg_path)
                if cached_cap is not None:
                    florence_caption = cached_cap
                else:
                    florence_caption = caption_region_florence2(fg_path)
                    if florence_caption:
                        self._save_cached_florence2(fg_path, florence_caption)
                if florence_caption:
                    florence_caption = f"\nFlorence-2 preliminary caption: {florence_caption}"

            if vision_mode in ("contour", "both"):
                for rep_img in reps:
                    img_path = self._find_image(image_dir, rep_img)
                    if img_path:
                        contour_out = str(species_dir / f"{cat}_contour.png")
                        # Need to filter COCO to this image
                        img_coco = self._filter_coco_for_image(coco_json, rep_img)
                        contour_path = create_contour_image(
                            img_path, img_coco, cat, contour_out
                        )
                        if contour_path:
                            break

            # Build prompt
            sys_prompt = self.SYSTEM_PROMPT.format(category=cat)

            rag_context = ""
            vocab_ref = ""
            if self.pipeline:
                query = RetrievalQuery(
                    body_region=self._map_cat_to_region(cat),
                    description_mode="segmented",
                    k=k,
                )
                results = self.pipeline.index.retrieve(query)
                formatter = PromptFormatter()
                rag_context = formatter.format_retrieval_context(results, query)

            if rosetta:
                relevant = ["texture", "sculpture", "setae", "color",
                            "color_pattern", "shape", "margin", "luster"]
                vp = ["=== ROSETTA vocabulary ==="]
                for attr in relevant:
                    if attr in rosetta:
                        vals = rosetta[attr]
                        if isinstance(vals, dict) and "values" in vals:
                            vals = vals["values"]
                        if isinstance(vals, list):
                            vp.append(f"{attr}: {', '.join(vals[:20])}")
                vocab_ref = "\n".join(vp)

            schema_context = self._schema_context_for_category(cat, group_label)

            user_prompt = self.DIAGNOSIS_PROMPT.format(
                group_label=group_label,
                category=cat,
                species_data=species_data,
                florence_caption=florence_caption or "",
            )

            if schema_context:
                user_prompt += f"\n\n{schema_context}\n"

            # Inject type material and sex context
            if type_material_text:
                user_prompt += f"\n\n=== TYPE MATERIAL ===\n{type_material_text}\n"

            if type_series and type_series.get("sex_dimorphism_possible"):
                sex_note = (
                    "Both males and females are available. Where this body region "
                    "shows sexual dimorphism, describe BOTH sexes and note the "
                    "differences. State which sex the holotype represents."
                )
                # Check if this category has sex-linked structures
                cat_lower = cat.lower()
                sex_specific = any(kw in cat_lower for kw in [
                    "proctiger", "paramere", "aedeagus", "subgenital",
                    "terminalia", "circumanal", "ovipositor", "genitalia",
                ])
                if sex_specific:
                    sex_note += (
                        f"\nNote: '{cat}' is a sex-specific genitalic structure. "
                        "Indicate which sex this belongs to."
                    )
                user_prompt += f"\n\n=== SEX & DIMORPHISM ===\n{sex_note}\n"

            # Append taxonomist-provided questions for this category
            if self.user_prompts:
                cat_questions = _match_prompts_to_category(self.user_prompts, cat)
                if cat_questions:
                    user_prompt += "\n\n=== TAXONOMIST QUESTIONS (address each if possible) ===\n"
                    for qi, q in enumerate(cat_questions, 1):
                        user_prompt += f"{qi}. {q}\n"

            user_prompt += (
                "\n\nRespond ONLY with valid JSON-LD. No markdown fences.\n\n"
                + json.dumps(self._biorag_schema(cat, group_label), indent=2)
            )

            # === Foreground mask description ===
            if fg_path and vision_mode in ("foreground", "both"):
                cached = self._get_cached_response(group_label, cat, "foreground")
                if cached:
                    all_category_descriptions.append(cached)
                else:
                    prompt_data = {
                        "system": sys_prompt,
                        "context": rag_context,
                        "vocabulary": vocab_ref,
                        "user_prompt": user_prompt,
                        "output_schema": self._biorag_schema(cat, group_label),
                    }
                    try:
                        result = self.claude.call(prompt_data, image_path=fg_path)
                        result["_category"] = cat
                        result["_vision_mode"] = "foreground"
                        result["_foreground_mask"] = fg_path
                        self._save_cached_response(group_label, cat, "foreground", result)
                        all_category_descriptions.append(result)
                        logger.info(f"    [foreground] ✓ {cat}")
                    except Exception as e:
                        logger.error(f"    [foreground] ✗ {cat}: {e}")

            # === Contour-guided description ===
            if contour_path and vision_mode in ("contour", "both"):
                cached_c = self._get_cached_response(group_label, cat, "contour")
                if cached_c:
                    all_category_descriptions_contour.append(cached_c)
                else:
                    prompt_data_c = {
                        "system": sys_prompt.replace(
                            "either isolated foreground mask or contour-highlighted",
                            "full image with the target region highlighted by a GREEN contour"
                        ),
                        "context": rag_context,
                        "vocabulary": vocab_ref,
                        "user_prompt": user_prompt.replace(
                            "Examine the image",
                            "Examine the image (the target region is highlighted with a green contour)"
                        ),
                        "output_schema": self._biorag_schema(cat, group_label),
                    }
                    try:
                        result_c = self.claude.call(prompt_data_c, image_path=contour_path)
                        result_c["_category"] = cat
                        result_c["_vision_mode"] = "contour"
                        result_c["_contour_image"] = contour_path
                        self._save_cached_response(group_label, cat, "contour", result_c)
                        all_category_descriptions_contour.append(result_c)
                        logger.info(f"    [contour]    ✓ {cat}")
                    except Exception as e:
                        logger.error(f"    [contour]    ✗ {cat}: {e}")

            # === Data-only fallback (no image available) ===
            if not fg_path and not contour_path:
                cached_d = self._get_cached_response(group_label, cat, "data_only")
                if cached_d:
                    all_category_descriptions.append(cached_d)
                else:
                    prompt_data_d = {
                        "system": sys_prompt.replace(
                            "A SPECIMEN IMAGE showing a specific body region",
                            "No image available — use quantitative data only"
                        ),
                        "context": rag_context,
                        "vocabulary": vocab_ref,
                        "user_prompt": user_prompt.replace(
                            "Examine the image and combine:\n"
                            "1. Quantitative characters (from the data, with ranges and statistics)\n"
                            "2. Qualitative characters (from the image: color, texture, sculpture, shape)",
                            "Use the quantitative data to characterize this region.\n"
                            "Note: no image available — report only measurable characters."
                        ),
                        "output_schema": self._biorag_schema(cat, group_label),
                    }
                    try:
                        result_d = self.claude.call(prompt_data_d)
                        result_d["_category"] = cat
                        result_d["_vision_mode"] = "data_only"
                        self._save_cached_response(group_label, cat, "data_only", result_d)
                        all_category_descriptions.append(result_d)
                        logger.info(f"    [data-only]  ✓ {cat}")
                    except Exception as e:
                        logger.error(f"    [data-only]  ✗ {cat}: {e}")

        # Merge all category descriptions into species description
        fg_merged = self._merge_species_graph(
            all_category_descriptions, group_label, "foreground"
        )

        # Synthesis diagnosis: merge per-category diagnoses into 1-3 paragraphs
        if fg_merged.get("full_diagnosis"):
            synth = self._synthesize_diagnosis(
                group_label, fg_merged["full_diagnosis"],
                type_series=type_series)
            fg_merged["synthesis_diagnosis"] = synth

        # Add type material designation
        if type_series:
            fg_merged["type_material"] = type_series
            fg_merged["type_material_text"] = type_material_text

        # Read specimen labels
        label_text = self._read_labels(group_label, image_dir, label_dir, compiled)
        if label_text:
            fg_merged["materials_examined"] = label_text

        fg_path_out = species_dir / f"{group_label}_foreground.jsonld"
        with open(fg_path_out, "w") as f:
            json.dump(fg_merged, f, indent=2, default=str)
        logger.info(f"  Foreground description: {fg_path_out}")

        if all_category_descriptions_contour:
            ct_merged = self._merge_species_graph(
                all_category_descriptions_contour, group_label, "contour"
            )
            ct_path_out = species_dir / f"{group_label}_contour.jsonld"
            with open(ct_path_out, "w") as f:
                json.dump(ct_merged, f, indent=2, default=str)
            logger.info(f"  Contour description: {ct_path_out}")

        # Write human-readable text
        self._write_text_description(fg_merged, species_dir / f"{group_label}.txt")

        return fg_merged

    def describe_all_species(
        self,
        compiled: Dict,
        semilandmarks_dirs,
        image_dir: str,
        coco_json: Dict,
        output_dir: Path,
        species_list: Optional[List[str]] = None,
        vision_mode: str = "foreground",
        rosetta: Optional[Dict] = None,
        k: int = 5,
        label_dir: Optional[str] = None,
        generate_key: bool = False,
    ) -> Dict[str, Dict]:
        """Describe all species (or a subset) from the compiled data."""
        full = compiled.get("full_features")
        if full is None:
            raise ValueError("No full_features in compiled data")

        all_species = sorted(full["group_label"].dropna().unique())
        if species_list:
            all_species = [s for s in all_species if s in species_list]

        logger.info(f"BioRAG: describing {len(all_species)} species")

        results = {}
        for i, sp in enumerate(all_species, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Species {i}/{len(all_species)}: {sp}")
            logger.info(f"{'='*60}")

            try:
                result = self.describe_species(
                    compiled=compiled,
                    group_label=sp,
                    semilandmarks_dirs=semilandmarks_dirs,
                    image_dir=image_dir,
                    coco_json=coco_json,
                    output_dir=output_dir,
                    vision_mode=vision_mode,
                    rosetta=rosetta,
                    k=k,
                    label_dir=label_dir,
                )
                results[sp] = result
            except Exception as e:
                logger.error(f"Failed to describe {sp}: {e}")
                results[sp] = {"error": str(e)}

        # Generate taxonomic key after all species are described
        if generate_key and len([r for r in results.values() if "error" not in r]) >= 2:
            logger.info(f"\n{'='*60}")
            logger.info("Generating taxonomic key")
            logger.info(f"{'='*60}")
            self.generate_taxonomic_key(results, compiled, output_dir)

        # Write combined summary
        summary = {
            "generated": datetime.now().isoformat(),
            "method": "BioRAG (Taxonomic Observation Workflow with Literature-Enriched sYnthesis)",
            "n_species": len(results),
            "species": {sp: {"status": "error" if "error" in r else "ok",
                             "n_categories": len(r.get("category_descriptions", []))}
                        for sp, r in results.items()},
        }
        with open(output_dir / "biorag_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return results

    def _find_image(self, image_dir: str, image_base: str) -> Optional[str]:
        """Find an image file in the image directory."""
        img_dir = Path(image_dir)
        exact = img_dir / image_base
        if exact.exists():
            return str(exact)

        stem = Path(image_base).stem
        for ext in [".tif", ".tiff", ".jpg", ".jpeg", ".png"]:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                return str(candidate)

        candidates = list(img_dir.glob(f"*{stem}*"))
        if candidates:
            return str(candidates[0])
        return None

    def _filter_coco_for_image(self, coco_json: Dict, image_base: str) -> Dict:
        """Filter COCO JSON to only include annotations for a specific image."""
        img_base_norm = image_base.replace(" ", "_")
        img_id = None
        for img in coco_json.get("images", []):
            fn = Path(img["file_name"]).name.replace(" ", "_")
            if fn == img_base_norm or Path(fn).stem == Path(img_base_norm).stem:
                img_id = img["id"]
                break

        if img_id is None:
            return coco_json

        filtered = {
            "categories": coco_json.get("categories", []),
            "images": [img for img in coco_json["images"] if img["id"] == img_id],
            "annotations": [ann for ann in coco_json["annotations"]
                           if ann["image_id"] == img_id],
        }
        return filtered

    def _map_cat_to_region(self, category: str) -> str:
        """Map a category name to a canonical region for RAG retrieval.

        Uses taxon schema (data-driven) first, falls back to REGION_PATTERNS
        (vocabulary-based) only if no schema is available.
        """
        if self.taxon_schema:
            cat_info = self.taxon_schema.get("categories", {}).get(category)
            if cat_info:
                return cat_info.get("body_region", "whole_body")
        cat_lower = category.lower()
        for region, patterns in REGION_PATTERNS.items():
            if any(p in cat_lower for p in patterns):
                return region
        return "whole_body"

    def _biorag_schema(self, category: str, group_label: str) -> Dict:
        return {
            "@context": {
                "@vocab": "https://descriptron.org/ontology/",
                "dwc": "http://rs.tdwg.org/dwc/terms/",
                "uberon": "http://purl.obolibrary.org/obo/UBERON_",
                "hao": "http://purl.obolibrary.org/obo/HAO_",
                "aism": "http://purl.obolibrary.org/obo/AISM_",
            },
            "expected_output": {
                "taxon": group_label,
                "category": category,
                "diagnosis": "Characters that distinguish this species from congeners. "
                             "Cite species names and p-values from pairwise tests. "
                             "Use format: 'X differs from Y in having longer Z (0.42–0.51 vs 0.33–0.38 mm, p<0.01)'",
                "description": "Comprehensive morphological description. "
                               "Report measurements as: 'value range (mean ± SD, n=X)'. "
                               "Include qualitative observations from the image.",
                "traits": [
                    {
                        "traitType": "ROSETTA category (e.g., texture, color, shape)",
                        "value": "observed character state",
                        "measurement": "numeric value if applicable",
                        "range": "min–max across specimens",
                        "mean_sd": "mean ± SD",
                        "n": "sample size",
                        "ontology_uri": "HAO/UBERON/PATO URI if known",
                        "evidenceSource": "measurement | image_observation | both",
                        "is_diagnostic": True,
                    }
                ],
            },
        }

    # ── Synthesis, labels, and key generation ────────────────────────────

    def _synthesize_diagnosis(self, group_label: str,
                              per_category_diagnoses: str,
                              type_series: Optional[Dict] = None) -> str:
        """Call Claude to merge per-category diagnoses into 1-3 paragraph summary."""
        cached = self._get_cached_response(group_label, "_synthesis", "diagnosis")
        if cached:
            return cached.get("text", "")

        sex_context = ""
        if type_series and type_series.get("sex_dimorphism_possible"):
            n_m = type_series.get("n_males", 0)
            n_f = type_series.get("n_females", 0)
            holo = type_series.get("holotype", {})
            sex_context = (
                f"\n\nSex data: {n_m} males, {n_f} females examined. "
                f"Holotype is {holo.get('sex', 'unknown')}. "
                "If any per-category descriptions mention sexual dimorphism, "
                "include a 'Sexual dimorphism' sentence noting which characters "
                "differ between males and females."
            )

        prompt = f"""You are an expert taxonomist. Below are per-body-region diagnostic statements
for {group_label}, each identifying characters that distinguish it from congeners.

{per_category_diagnoses}{sex_context}

Write a concise 1-3 paragraph SYNTHESIS DIAGNOSIS that:
1. Opens with the most powerful distinguishing characters (largest effect sizes, most species pairs separated)
2. Groups related characters logically (size → shape → color → texture → spatial)
3. Uses formal taxonomic language
4. Cites specific congener names and p-values for the strongest contrasts
5. Omits redundant grid-cell texture/homology codes — summarize those as "wing membrane texture" etc.
6. Notes sexual dimorphism if any sex-linked structures (genitalia) are described for both sexes
7. Ends with any unique qualitative characters from image observations
8. EVERY numeric value MUST have explicit units — no exceptions:
   - Lengths/perimeters: "0.395 mm"; Areas: "0.0015 mm²"
   - Procrustes distances: "0.3747 (dimensionless, Procrustes-normalized)"
   - Dimensionless ratios: "aspect ratio 6.69 (dimensionless)"
   - CIE LAB: "L* 34.5 (CIE LAB, 0–255 scale)"
   - UMAP/PCA scores: "UMAP2 15.24 (dimensionless)"
   - Boundary areas: "n_boundaries_same_mm2 = 14.5 mm²" (prefer _mm2 over raw pixel count)
   - Boundary lengths: "n_boundaries_diff_mm = 0.42 mm" (prefer _mm over raw pixel count)
   A referee must verify every value without guessing units.

Return ONLY the diagnosis text, no JSON, no markdown headers."""

        try:
            response = self.claude.client.messages.create(
                model=self.claude.model,
                max_tokens=4000,
                system=(
                    "You are a professional taxonomist writing formal species "
                    "descriptions for peer-reviewed scientific publication. "
                    "All anatomical terms are standard taxonomic nomenclature "
                    "for organism morphology used in journals such as Systematic "
                    "Biology, Zootaxa, and the European Journal of Taxonomy."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            if not response.content:
                logger.warning(f"  Synthesis diagnosis: API returned empty content (stop_reason={response.stop_reason}), retrying with truncated input")
                truncated = per_category_diagnoses[:12000]
                response = self.claude.client.messages.create(
                    model=self.claude.model,
                    max_tokens=4000,
                    system=(
                        "You are a professional taxonomist writing formal "
                        "species descriptions for peer-reviewed scientific publication. "
                        "All anatomical terms are standard taxonomic nomenclature."
                    ),
                    messages=[{"role": "user", "content": prompt.replace(per_category_diagnoses, truncated)}],
                )
            if not response.content:
                logger.error(f"  Synthesis diagnosis: API returned empty content after retry (stop_reason={response.stop_reason})")
                return ""
            text = response.content[0].text.strip()
            self._save_cached_response(group_label, "_synthesis", "diagnosis",
                                       {"text": text})
            logger.info(f"  Synthesis diagnosis: {len(text)} chars")
            return text
        except Exception as e:
            logger.error(f"  Synthesis diagnosis failed: {e}")
            return ""

    def _read_labels(self, group_label: str, image_dir: str,
                     label_dir: Optional[str], compiled: Dict) -> str:
        """Find and OCR label images for a species using Florence-2."""
        if not label_dir:
            return ""

        label_path = Path(label_dir)
        full_features = compiled.get("full_features")
        if full_features is None:
            return ""

        sp_images = full_features[
            full_features["group_label"] == group_label
        ]["image_base"].unique() if "image_base" in full_features.columns else []

        label_imgs = []

        for img_base in sp_images:
            stem = Path(img_base).stem
            for suffix in ["_label", "_labels", "_Label"]:
                for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    candidate = label_path / f"{stem}{suffix}{ext}"
                    if candidate.exists():
                        label_imgs.append(candidate)

        sp_label_dir = label_path / group_label
        if sp_label_dir.is_dir():
            for f in sp_label_dir.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                    label_imgs.append(f)

        if not label_imgs:
            return ""

        label_texts = []
        for limg in sorted(set(label_imgs)):
            key = str(limg.name)
            cached = self._get_cached_florence2(str(limg))
            if cached is not None:
                text = cached
            else:
                text = caption_region_florence2(str(limg))
                if text:
                    self._save_cached_florence2(str(limg), text)
            if text:
                label_texts.append(f"  {limg.stem}: {text}")

        if not label_texts:
            return ""

        logger.info(f"  Labels read: {len(label_texts)} images")
        return "Materials examined:\n" + "\n".join(label_texts)

    def generate_taxonomic_key(self, all_results: Dict[str, Dict],
                               compiled: Dict, output_dir: Path) -> str:
        """Generate a dichotomous taxonomic key from all species descriptions."""
        cached = self._get_cached_response("_all", "_key", "taxonomic")
        if cached:
            key_text = cached.get("text", "")
            key_path = output_dir / "taxonomic_key.txt"
            with open(key_path, "w") as f:
                f.write(key_text)
            logger.info(f"Taxonomic key (cached): {key_path}")
            return key_text

        species_summaries = []
        for sp, result in sorted(all_results.items()):
            if "error" in result:
                continue
            synth = result.get("synthesis_diagnosis", "")
            if not synth:
                diag = result.get("full_diagnosis", "")
                if diag:
                    synth = diag[:2000]
            if synth:
                species_summaries.append(f"### {sp}\n{synth}")

        if len(species_summaries) < 2:
            logger.warning("Need at least 2 species for a key")
            return ""

        diag_report = compiled.get("diagnostic_report", {})
        top_features = []
        for cat, cat_data in diag_report.get("categories", {}).items():
            tests = cat_data.get("tests", [])
            for feat in tests[:3]:
                top_features.append(
                    f"  {cat}: {feat.get('feature', '?')} "
                    f"(H={feat.get('H_stat', 0):.1f}, "
                    f"η²={feat.get('eta_squared', 0):.3f})"
                )

        prompt = f"""You are an expert taxonomist. Below are synthesis diagnoses for {len(species_summaries)} species,
plus the top statistically powerful diagnostic features from Kruskal-Wallis tests.

=== SPECIES DIAGNOSES ===
{chr(10).join(species_summaries)}

=== TOP DIAGNOSTIC FEATURES (by effect size) ===
{chr(10).join(top_features[:60])}

All measurements are calibrated to real units (mm, mm², dimensionless ratios).
Feature names ending in _mm are millimeters, _mm2 are square millimeters.
Landmark distances (lmk_dist_*) are dimensionless Procrustes-normalized values.

Create a DICHOTOMOUS TAXONOMIC KEY that:
1. Uses the most reliable and easily observed characters first (body size, leg/wing proportions, color)
2. Falls back to quantitative measurements when qualitative characters overlap
3. For measurements, ALWAYS provide the threshold value and units (e.g., "femur length >0.38 mm" or "forewing area >1.2 mm²")
4. Each couplet contrasts two character states with clear "go to X" / "go to Y" targets
5. Every terminal leads to a single species name
6. Includes ALL {len(species_summaries)} species
7. Uses standard taxonomic key format:
   1  Character state A .......................... 2
   -  Character state B .......................... 3
   2  Character state C .......................... Species X
   -  Character state D .......................... Species Y

Prioritize characters that separate the most species pairs with the highest effect sizes.
Write the key so a human with a specimen under a microscope and ruler can follow it.

CRITICAL CONSTRAINT — HUMAN-MEASURABLE CHARACTERS ONLY:
Do NOT use any of these computational features as key characters:
- Cluster assignments (tex_phylo_cluster, colhom_phylo_cluster, color_cluster, etc.)
- UMAP coordinates (UMAP1, UMAP2, etc.)
- PCA scores (PC1, PC2, shape_PC1, etc.)
- Texture statistics (GLCM, LBP, bending_energy, shape_complexity)
- Silhouette scores, optimal_k, boundary counts (n_boundaries_same, n_boundaries_diff)
- Any feature requiring computational analysis (clustering, dimensionality reduction)

ONLY use characters a taxonomist can directly measure or observe:
- Body lengths, widths, areas, perimeters (in mm or mm²)
- Aspect ratios, solidity, equivalent diameter
- Color values (CIE L*, a*, b* on 0–255 scale; HSV hue, saturation, brightness)
- Taxonomic ratios (ratio_VL/VW, ratio_FL/FW, ratio_MF/MT, etc.)
- Landmark distances (lmk_dist_*)
- Inter-mask distances (imd_*_mm)
- Qualitative color descriptions (ferruginous, testaceous, pale, dark, etc.)
- Shape centroid size (in mm)

SPECIES CHECKLIST — every one of these {len(species_summaries)} species MUST appear as a terminal:
{chr(10).join(f'  - {sp.split(chr(10))[0].replace("### ", "")}' for sp in species_summaries)}

Verify before finishing that each species name above leads to exactly one terminal.

Return ONLY the key text. No JSON. No explanatory preamble."""

        species_names = sorted(all_results.keys())
        sys_msg = (
            "You are a professional taxonomist writing formal species "
            "descriptions for peer-reviewed scientific publication. "
            "All anatomical terms are standard taxonomic nomenclature "
            "for organism morphology used in journals such as Systematic "
            "Biology, Zootaxa, and the European Journal of Taxonomy."
        )

        try:
            response = self.claude.client.messages.create(
                model=self.claude.model,
                max_tokens=_max_output_tokens(self.claude.model),
                system=sys_msg,
                messages=[{"role": "user", "content": prompt}],
            )
            if not response.content:
                logger.error(f"Key generation: API returned empty content "
                             f"(stop_reason={response.stop_reason})")
                return ""
            key_text = response.content[0].text.strip()

            # Validate: every species must appear in the key
            missing = [sp for sp in species_names if sp not in key_text]
            if missing:
                logger.warning(f"Key missing {len(missing)} species: {missing}. "
                               f"Regenerating...")
                fix_prompt = (
                    f"The following dichotomous key is missing these species: "
                    f"{', '.join(missing)}.\n\n"
                    f"Key:\n{key_text}\n\n"
                    f"Rewrite the key to include ALL {len(species_names)} species. "
                    f"Missing species diagnoses:\n"
                    + "\n".join(sp_sum for sp_sum in species_summaries
                               if any(m in sp_sum for m in missing))
                    + "\n\nReturn ONLY the complete corrected key."
                )
                fix_resp = self.claude.client.messages.create(
                    model=self.claude.model,
                    max_tokens=_max_output_tokens(self.claude.model),
                    system=sys_msg,
                    messages=[{"role": "user", "content": fix_prompt}],
                )
                if fix_resp.content:
                    key_text = fix_resp.content[0].text.strip()
                    still_missing = [sp for sp in species_names
                                     if sp not in key_text]
                    if still_missing:
                        logger.warning(f"Key still missing after fix: "
                                       f"{still_missing}")
                    else:
                        logger.info("All species present after fix")

            self._save_cached_response("_all", "_key", "taxonomic",
                                       {"text": key_text})

            key_path = output_dir / "taxonomic_key.txt"
            with open(key_path, "w") as f:
                f.write(key_text)
            logger.info(f"Taxonomic key: {key_path} ({len(key_text)} chars)")
            return key_text
        except Exception as e:
            logger.error(f"Taxonomic key generation failed: {e}")
            return ""

    def _merge_species_graph(self, category_results: List[Dict],
                             group_label: str, vision_mode: str) -> Dict:
        """Merge per-category descriptions into a single species graph."""
        merged = {
            "@context": {
                "@vocab": "https://descriptron.org/ontology/",
                "dwc": "http://rs.tdwg.org/dwc/terms/",
            },
            "@type": "SpeciesDescription",
            "taxon": group_label,
            "generated": datetime.now().isoformat(),
            "method": "BioRAG",
            "vision_mode": vision_mode,
            "category_descriptions": [],
            "full_diagnosis": "",
            "full_description": "",
        }

        diag_parts = []
        desc_parts = []

        for result in category_results:
            cat = result.get("_category", "unknown")
            entry = {
                "category": cat,
                "vision_mode": result.get("_vision_mode", vision_mode),
            }

            inner = result.get("expected_output", result)

            if "diagnosis" in inner:
                entry["diagnosis"] = inner["diagnosis"]
                diag_parts.append(f"**{cat}**: {inner['diagnosis']}")
            if "description" in inner:
                entry["description"] = inner["description"]
                desc_parts.append(f"**{cat}**: {inner['description']}")
            if "traits" in inner:
                entry["traits"] = inner["traits"]

            entry["raw_response"] = {k: v for k, v in result.items()
                                     if not k.startswith("_")}
            merged["category_descriptions"].append(entry)

        merged["full_diagnosis"] = "\n\n".join(diag_parts)
        merged["full_description"] = "\n\n".join(desc_parts)
        merged["summary"] = {
            "n_categories": len(category_results),
            "categories_described": [r.get("_category") for r in category_results],
            "vision_modes_used": list(set(r.get("_vision_mode", "") for r in category_results)),
        }

        return merged

    def _write_text_description(self, merged: Dict, path: Path):
        """Write a human-readable text description from the merged JSON-LD."""
        lines = [
            f"SPECIES DESCRIPTION: {merged.get('taxon', 'Unknown')}",
            f"Generated: {merged.get('generated', '')}",
            f"Method: BioRAG ({merged.get('vision_mode', '')})",
            "=" * 70,
            "",
        ]

        # Type material section (before diagnosis, per taxonomic convention)
        type_text = merged.get("type_material_text", "")
        if type_text:
            lines.append("TYPE MATERIAL")
            lines.append("-" * 40)
            lines.append(type_text)
            lines.append("")

        synth = merged.get("synthesis_diagnosis", "")
        if synth:
            lines.append("SYNTHESIS DIAGNOSIS")
            lines.append("-" * 40)
            lines.append(synth)
            lines.append("")

        diag = merged.get("full_diagnosis", "")
        if diag:
            lines.append("DIAGNOSIS (per category)")
            lines.append("-" * 40)
            lines.append(diag)
            lines.append("")

        desc = merged.get("full_description", "")
        if desc:
            lines.append("DESCRIPTION")
            lines.append("-" * 40)
            lines.append(desc)
            lines.append("")

        mat = merged.get("materials_examined", "")
        if mat:
            lines.append("MATERIALS EXAMINED")
            lines.append("-" * 40)
            lines.append(mat)
            lines.append("")

        for cat_desc in merged.get("category_descriptions", []):
            cat = cat_desc.get("category", "")
            traits = cat_desc.get("traits", [])
            if traits:
                lines.append(f"\n{cat.upper()} — Trait details")
                lines.append("-" * 40)
                for t in traits:
                    line = f"  {t.get('traitType', '?')}: {t.get('value', '?')}"
                    if t.get("range"):
                        line += f" ({t['range']})"
                    if t.get("mean_sd"):
                        line += f" [mean ± SD: {t['mean_sd']}]"
                    if t.get("is_diagnostic"):
                        line += " *DIAGNOSTIC*"
                    lines.append(line)

        with open(path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"  Text description: {path}")


def save_biorag_recipe(args, output_dir: Path, n_user_prompts: int = 0):
    """Save the BioRAG run configuration as a YAML recipe file."""
    recipe = {
        "biorag_version": "1.0",
        "generated": datetime.now().isoformat(),
        "command": "describe-biorag",
        "parameters": {
            "compiled_dir": args.compiled_dir,
            "semilandmarks_dir": args.semilandmarks_dir,
            "image_dir": args.image_dir,
            "coco_json": args.coco_json,
            "output_dir": str(output_dir),
            "pdf_dir": args.pdf_dir,
            "index": args.index,
            "species": args.species,
            "vision_mode": args.vision_mode,
            "florence2": not args.no_florence2,
            "k": args.k,
            "model": args.model,
            "family": args.family,
            "label_dir": args.label_dir,
            "generate_key": args.generate_key,
            "user_prompts_file": args.user_prompts,
            "n_user_prompts": n_user_prompts,
            "cache_enabled": not args.no_cache,
        },
    }
    recipe_path = output_dir / "biorag_recipe.yaml"
    try:
        import yaml
        with open(recipe_path, "w") as f:
            yaml.dump(recipe, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        with open(recipe_path, "w") as f:
            for section, vals in recipe.items():
                if isinstance(vals, dict):
                    f.write(f"{section}:\n")
                    for k, v in vals.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{section}: {vals}\n")
    logger.info(f"Recipe saved: {recipe_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# KEY CORRECTION (standalone — usable from pipeline or CLI)
# ═══════════════════════════════════════════════════════════════════════════════

def correct_taxonomic_key(
    key_path: Path,
    check_report_path: Path,
    output_path: Path = None,
    api_key: str = None,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Correct a taxonomic key using numeric checker findings.

    Reads the existing key and the checker report, sends both to the VLM
    with explicit correction instructions, and writes the corrected key.

    Parameters
    ----------
    key_path : Path to existing taxonomic_key.txt
    check_report_path : Path to key_check_report.json from biorag_key_checker
    output_path : Where to write corrected key (default: overwrites key_path)
    api_key : Anthropic API key (reads env if None)
    model : Claude model string

    Returns
    -------
    Corrected key text, or empty string on failure.
    """
    import anthropic as _anthropic

    key_path = Path(key_path)
    check_report_path = Path(check_report_path)
    if output_path is None:
        output_path = key_path
    output_path = Path(output_path)

    if not key_path.exists():
        logger.error(f"Key file not found: {key_path}")
        return ""
    if not check_report_path.exists():
        logger.error(f"Check report not found: {check_report_path}")
        return ""

    key_text = key_path.read_text()
    with open(check_report_path) as f:
        report = json.load(f)

    issues = report.get("issue_details", [])
    if not issues:
        logger.info("No issues found in checker report — key is already correct")
        output_path.write_text(key_text)
        return key_text

    # ── Format correction table ──────────────────────────────────────────
    correction_lines = []
    for iss in issues:
        status = iss.get("status", "unknown")
        couplet = iss.get("couplet", "?")
        raw = iss.get("raw", "")
        feature = iss.get("feature_text", "")
        cat = iss.get("category", "")
        sugg = iss.get("suggestion", {})
        suggested_val = sugg.get("suggested", "?")
        accuracy = sugg.get("accuracy", "?")
        left_spp = ", ".join(iss.get("left_species", []))
        right_spp = ", ".join(iss.get("right_species", []))
        left_mean = sugg.get("left_mean", iss.get("left_mean", "?"))
        right_mean = sugg.get("right_mean", iss.get("right_mean", "?"))
        operator = iss.get("operator", "")

        if status == "direction_reversed":
            fix = (f"Couplet {couplet}: \"{raw}\" — DIRECTION REVERSED. "
                   f"{cat} {feature}: left [{left_spp}] mean={left_mean}, "
                   f"right [{right_spp}] mean={right_mean}. "
                   f"Flip the comparison direction (< → >, or > → <) "
                   f"and use threshold {suggested_val} (accuracy {accuracy}%).")
        elif status == "threshold_outside_range":
            data_range = iss.get("data_range", [])
            fix = (f"Couplet {couplet}: \"{raw}\" — THRESHOLD OUTSIDE DATA RANGE "
                   f"{data_range}. {cat} {feature}: left [{left_spp}] mean={left_mean}, "
                   f"right [{right_spp}] mean={right_mean}. "
                   f"Change threshold to {suggested_val} (accuracy {accuracy}%).")
        elif status == "threshold_too_low":
            fix = (f"Couplet {couplet}: \"{raw}\" — THRESHOLD TOO LOW. "
                   f"{cat} {feature}: left [{left_spp}] mean={left_mean}, "
                   f"right [{right_spp}] mean={right_mean}. "
                   f"Raise threshold to {suggested_val} (accuracy {accuracy}%).")
        elif status == "threshold_too_high":
            fix = (f"Couplet {couplet}: \"{raw}\" — THRESHOLD TOO HIGH. "
                   f"{cat} {feature}: left [{left_spp}] mean={left_mean}, "
                   f"right [{right_spp}] mean={right_mean}. "
                   f"Lower threshold to {suggested_val} (accuracy {accuracy}%).")
        else:
            fix = (f"Couplet {couplet}: \"{raw}\" — {status}. "
                   f"Suggested threshold: {suggested_val}.")
        correction_lines.append(fix)

    # ── Extract species list from existing key terminals ──────────────────
    import re
    terminals = set()
    for line in key_text.splitlines():
        stripped = line.strip().rstrip(".")
        parts = stripped.rsplit(" ", 1)
        if len(parts) == 2:
            candidate = parts[-1].strip()
            if (candidate and not candidate.isdigit()
                    and candidate not in ("to", "or", "and", "the", "see", "Go")):
                if re.match(r'^[A-Za-z]', candidate) and len(candidate) > 1:
                    terminals.add(candidate)
    species_from_key = sorted(terminals)

    n_issues = len(correction_lines)
    ok_count = report.get("ok", 0)
    total = report.get("total_thresholds", 0)

    prompt = f"""You are an expert taxonomist. Below is a dichotomous taxonomic key with
{n_issues} numeric threshold errors identified by an automated checker.
{ok_count} of {total} thresholds were correct.

YOUR TASK: Fix ONLY the threshold values and comparison directions listed below.
Do NOT change the key structure, couplet numbering, species terminals, or
qualitative characters. Only adjust the numeric values and operators (< / >)
as specified in the corrections.

=== ORIGINAL KEY ===
{key_text}

=== CORRECTIONS REQUIRED ({n_issues} issues) ===
{chr(10).join(correction_lines)}

=== RULES ===
1. Keep the EXACT same couplet structure and numbering
2. Keep ALL species terminals — every species in the original key must remain
3. Change ONLY the numeric threshold values and comparison directions as listed
4. Round thresholds to 2 decimal places for readability (e.g., 0.3376 → 0.34)
5. If a correction says "flip direction", swap < to > or > to <
6. If a correction says "threshold outside range", replace with the suggested value
7. Preserve all qualitative (non-numeric) characters exactly as they are
8. Return ONLY the complete corrected key — no commentary, no JSON, no preamble

SPECIES CHECKLIST — verify every one of these species appears as a terminal:
{chr(10).join(f'  - {sp}' for sp in species_from_key)}
"""

    sys_msg = (
        "You are a professional taxonomist writing formal species "
        "descriptions for peer-reviewed scientific publication. "
        "All anatomical terms are standard taxonomic nomenclature "
        "for organism morphology used in journals such as Systematic "
        "Biology, Zootaxa, and the European Journal of Taxonomy."
    )

    if api_key:
        client = _anthropic.Anthropic(api_key=api_key)
    else:
        client = _anthropic.Anthropic()

    max_tok = _max_output_tokens(model)
    logger.info(f"Correcting key: {n_issues} issues, model={model}, "
                f"max_tokens={max_tok}")

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tok,
            system=sys_msg,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            logger.error(f"Key correction: empty response "
                         f"(stop_reason={response.stop_reason})")
            return ""

        corrected = response.content[0].text.strip()
        logger.info(f"Corrected key: {len(corrected)} chars, "
                    f"stop={response.stop_reason}, "
                    f"in={response.usage.input_tokens}, "
                    f"out={response.usage.output_tokens}")

        # ── Species completeness check ───────────────────────────────────
        missing = [sp for sp in species_from_key if sp not in corrected]
        if missing:
            logger.warning(f"Corrected key missing {len(missing)} species: "
                           f"{missing}. Attempting fix...")
            fix_prompt = (
                f"The corrected key below is missing these species: "
                f"{', '.join(missing)}.\n\n"
                f"Key:\n{corrected}\n\n"
                f"Add the missing species back into the key at appropriate "
                f"positions. Return ONLY the complete key."
            )
            fix_resp = client.messages.create(
                model=model,
                max_tokens=max_tok,
                system=sys_msg,
                messages=[{"role": "user", "content": fix_prompt}],
            )
            if fix_resp.content:
                corrected = fix_resp.content[0].text.strip()
                still_missing = [sp for sp in species_from_key
                                 if sp not in corrected]
                if still_missing:
                    logger.warning(f"Still missing after fix: {still_missing}")
                else:
                    logger.info("All species present after fix")

        # ── Save ─────────────────────────────────────────────────────────
        # Back up original if overwriting
        if output_path == key_path:
            bak = Path(str(key_path) + ".bak_pre_correction")
            if not bak.exists():
                import shutil
                shutil.copy2(str(key_path), str(bak))
                logger.info(f"Backed up original key: {bak}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(corrected)
        logger.info(f"Corrected key saved: {output_path}")

        # Update cache if it exists
        cache_key = output_path.parent / "biorag_cache" / "_all" / "_key_taxonomic.json"
        if cache_key.parent.exists():
            with open(cache_key, "w") as f:
                json.dump({"text": corrected}, f, indent=2)

        return corrected

    except Exception as e:
        logger.error(f"Key correction failed: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BioSysLit RAG Retrieval for Descriptron",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index BioSysLit treatments for a taxon
  python biosyslit_rag_retrieval.py index --taxon "Tetramorium" --max-records 50 --output tetra_index.json

  # Index from local PDFs
  python biosyslit_rag_retrieval.py index --pdf-dir ./treatments/ --output local_index.json

  # Retrieve context
  python biosyslit_rag_retrieval.py retrieve --index tetra_index.json --taxon "Tetramorium caespitum" --family "Formicidae" --k 5

  # Generate description prompts
  python biosyslit_rag_retrieval.py describe --index tetra_index.json --taxon "Tetramorium caespitum" --family "Formicidae" --coco annotations.json
""")

    parser.add_argument(
        "--api-key-file", type=str, default=None,
        help="Path to a file containing ANTHROPIC_API_KEY=sk-ant-... "
             "(e.g. ~/keys/anthropic.env)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- index ---
    p_index = subparsers.add_parser("index", help="Build treatment index")
    p_index.add_argument("--taxon", type=str, help="Taxon to search in BioSysLit")
    p_index.add_argument("--pdf-dir", type=str, help="Directory of PDF treatments to index")
    p_index.add_argument("--pdf", type=str, action="append", help="Individual PDF file(s)")
    p_index.add_argument("--max-records", type=int, default=100)
    p_index.add_argument("--output", type=str, default="treatment_index.json")
    p_index.add_argument("--embeddings", action="store_true", help="Compute sentence embeddings")
    p_index.add_argument("--family", type=str, default="", help="Family for PDF imports")

    # --- retrieve ---
    p_ret = subparsers.add_parser("retrieve", help="Retrieve relevant treatments")
    p_ret.add_argument("--index", type=str, required=True)
    p_ret.add_argument("--taxon", type=str, default="")
    p_ret.add_argument("--family", type=str, default="")
    p_ret.add_argument("--region", type=str, default="")
    p_ret.add_argument("--modality", type=str, default="standard")
    p_ret.add_argument("--mode", type=str, default="habitus", choices=["habitus","segmented","specialized"])
    p_ret.add_argument("--k", type=int, default=5)

    # --- describe ---
    p_desc = subparsers.add_parser("describe", help="Generate description prompts")
    p_desc.add_argument("--index", type=str, required=True)
    p_desc.add_argument("--taxon", type=str, required=True)
    p_desc.add_argument("--family", type=str, default="")
    p_desc.add_argument("--coco", type=str, help="COCO JSON annotation file")
    p_desc.add_argument("--modality", type=str, default="standard")
    p_desc.add_argument("--k", type=int, default=5)
    p_desc.add_argument("--output", type=str, default="description_prompts.json")

    # --- describe-live ---
    p_live = subparsers.add_parser(
        "describe-live",
        help="RAG treatments into Claude Sonnet context and generate JSON-LD description",
    )
    p_live.add_argument("--index",      type=str, required=True,
                        help="Treatment index JSON from the 'index' command")
    p_live.add_argument("--image",      type=str, default=None,
                        help="Specimen image (jpg/png/webp) — single mode")
    p_live.add_argument("--coco",       type=str, default=None,
                        help="COCO JSON annotation file for the image")
    p_live.add_argument("--output",     type=str, default="description.jsonld",
                        help="Output JSON-LD file (single mode)")
    p_live.add_argument("--image-dir",  type=str, default=None,
                        help="Directory of images — batch mode")
    p_live.add_argument("--coco-dir",   type=str, default=None,
                        help="Directory of COCO JSONs matched by stem (batch mode)")
    p_live.add_argument("--output-dir", type=str, default="./descriptions/",
                        help="Output directory for batch JSON-LD files")
    p_live.add_argument("--taxon",      type=str, required=True)
    p_live.add_argument("--family",     type=str, default="")
    p_live.add_argument("--modality",   type=str, default="standard",
                        choices=["standard", "sem", "micro_ct", "stacking"])
    p_live.add_argument("--k",          type=int, default=5,
                        help="Treatment chunks to retrieve per mode")
    p_live.add_argument("--model",      type=str, default=CLAUDE_OPUS,
                        help="Claude model string")

    # --- describe-biorag ---
    p_biorag = subparsers.add_parser(
        "describe-biorag",
        help="BioRAG: data-driven species descriptions using compiled diagnostic data + VLM",
    )
    p_biorag.add_argument("--compiled-dir", type=str, required=True,
                          help="Directory from compile_specimen_data.py output")
    p_biorag.add_argument("--semilandmarks-dir", type=str, nargs="+", required=True,
                          help="Semilandmarks output dir(s) for foreground masks (can specify multiple)")
    p_biorag.add_argument("--image-dir", type=str, required=True,
                          help="Directory with original specimen images")
    p_biorag.add_argument("--coco-json", type=str, required=True,
                          help="COCO JSON with all annotations")
    p_biorag.add_argument("--output-dir", type=str, default="./biorag_descriptions/",
                          help="Output directory for species descriptions")
    p_biorag.add_argument("--index", type=str, default=None,
                          help="Treatment index JSON for RAG context (optional)")
    p_biorag.add_argument("--pdf-dir", type=str, default=None,
                          help="PDF dir to build treatment index on the fly")
    p_biorag.add_argument("--species", type=str, nargs="*", default=None,
                          help="Specific species to describe (default: all)")
    p_biorag.add_argument("--vision-mode", type=str, default="foreground",
                          choices=["foreground", "contour", "both"],
                          help="Image input mode (default: foreground)")
    p_biorag.add_argument("--no-florence2", action="store_true",
                          help="Skip Florence-2 captioning")
    p_biorag.add_argument("--k", type=int, default=5,
                          help="RAG retrieval count per category")
    p_biorag.add_argument("--model", type=str, default=CLAUDE_OPUS,
                          help="Claude model string")
    p_biorag.add_argument("--family", type=str, default="",
                          help="Taxonomic family for RAG queries")
    p_biorag.add_argument("--api-key-file", type=str, default=None,
                          help="Path to file containing ANTHROPIC_API_KEY=sk-ant-...")
    p_biorag.add_argument("--no-cache", action="store_true",
                          help="Disable response caching (re-call API for every category)")
    p_biorag.add_argument("--label-dir", type=str, default=None,
                          help="Directory with label images (*_label.jpg etc.) for Materials Examined")
    p_biorag.add_argument("--generate-key", action="store_true",
                          help="Generate a dichotomous taxonomic key after all species are described")
    p_biorag.add_argument("--user-prompts", type=str, default=None,
                          help="Taxonomist prompt file (.docx, .csv, .xlsx, .txt) with questions per body region")
    p_biorag.add_argument("--group-labels", type=str, required=True,
                          help="CSV mapping image_base → species (filename, group_label)")

    # --- correct-key ---
    p_corr = subparsers.add_parser(
        "correct-key",
        help="Correct taxonomic key thresholds using numeric checker report",
    )
    p_corr.add_argument("--key-file", type=str, required=True,
                        help="Path to taxonomic_key.txt")
    p_corr.add_argument("--check-report", type=str, required=True,
                        help="Path to key_check_report.json from biorag_key_checker")
    p_corr.add_argument("--output-file", type=str, default=None,
                        help="Output path (default: overwrites key-file)")
    p_corr.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        help="Claude model string")
    p_corr.add_argument("--api-key-file", type=str, default=None,
                        help="Path to file containing ANTHROPIC_API_KEY=sk-ant-...")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load API key for any command that needs Claude
    if args.command in ("describe-live", "describe-biorag", "correct-key"):
        load_api_key(getattr(args, "api_key_file", None))

    if args.command == "index":
        index = TreatmentIndex()

        if args.taxon:
            ingester = BioSysLitIngester()
            records = ingester.search(args.taxon, max_records=args.max_records)
            for rec in records:
                chunks = ingester.extract_treatment_text(rec)
                index.add_chunks(chunks)

        if args.pdf_dir:
            pdf_ingester = PDFIngester()
            for pdf_file in Path(args.pdf_dir).glob("*.pdf"):
                chunks = pdf_ingester.extract_from_pdf(
                    str(pdf_file), taxon_family=args.family
                )
                index.add_chunks(chunks)

        if args.pdf:
            pdf_ingester = PDFIngester()
            for pdf_file in args.pdf:
                chunks = pdf_ingester.extract_from_pdf(
                    pdf_file, taxon_family=args.family
                )
                index.add_chunks(chunks)

        if args.embeddings:
            index.build_embeddings()

        index.save(args.output)
        print(f"\nIndex saved: {args.output}")
        print(f"  Chunks: {len(index.chunks)}")

    elif args.command == "retrieve":
        index = TreatmentIndex()
        index.load(args.index)

        query = RetrievalQuery(
            taxon_name=args.taxon,
            taxon_family=args.family,
            body_region=args.region,
            imaging_modality=args.modality,
            description_mode=args.mode,
            k=args.k,
        )

        results = index.retrieve(query)

        print(f"\n=== Top {len(results)} results ===\n")
        for i, r in enumerate(results):
            print(f"{i+1}. [{r.score:.1f}] {r.chunk.taxon_name} — {r.chunk.description_section}")
            print(f"   Source: {r.chunk.source_type}:{r.chunk.source_id}")
            print(f"   Region: {r.chunk.body_region} | Modality: {r.chunk.imaging_modality}")
            print(f"   Reasons: {', '.join(r.match_reasons)}")
            print(f"   Text: {r.chunk.text[:150]}...")
            print()

    elif args.command == "describe":
        index = TreatmentIndex()
        index.load(args.index)

        coco_json = None
        if args.coco:
            with open(args.coco) as f:
                coco_json = json.load(f)

        pipeline = DescriptionPipeline(index)
        prompts = pipeline.generate_prompts(
            taxon_name=args.taxon,
            taxon_family=args.family,
            coco_json=coco_json,
            imaging_modality=args.modality,
            k=args.k,
        )

        with open(args.output, 'w') as f:
            json.dump(prompts, f, indent=2)

        print(f"\nGenerated {len(prompts)} description prompts → {args.output}")
        for p in prompts:
            print(f"  {p['mode']:20s} region={p['region']:15s} retrievals={p['retrieval_results']}")


    elif args.command == "describe-live":
        index = TreatmentIndex()
        index.load(args.index)

        pipeline  = DescriptionPipeline(index)
        describer = ClaudeDescriber(model=args.model)

        if args.image:
            # ── single specimen ──────────────────────────────────────────────
            coco_json = None
            if args.coco:
                with open(args.coco) as f:
                    coco_json = json.load(f)

            merged = describer.describe_specimen(
                pipeline=pipeline,
                taxon_name=args.taxon,
                taxon_family=args.family,
                image_path=args.image,
                coco_json=coco_json,
                imaging_modality=args.modality,
                k=args.k,
            )
            with open(args.output, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"\nDescription written → {args.output}")
            print(f"  Nodes : {merged.get('summary', {}).get('nodeCount', '?')}")
            print(f"  Traits: {merged.get('summary', {}).get('traitCount', '?')}")
            print(f"  Modes : {merged.get('description_modes_used', [])}")

        elif args.image_dir:
            # ── batch ────────────────────────────────────────────────────────
            results = describer.describe_batch(
                pipeline=pipeline,
                image_dir=args.image_dir,
                coco_dir=getattr(args, "coco_dir", None),
                taxon_name=args.taxon,
                taxon_family=args.family,
                output_dir=args.output_dir,
                k=args.k,
            )
            ok   = sum(1 for r in results if r["status"] == "ok")
            fail = len(results) - ok
            print(f"\nBatch complete: {ok} OK, {fail} failed → {args.output_dir}")
            for r in results:
                mark = "✓" if r["status"] == "ok" else "✗"
                print(f"  {mark} {Path(r['image']).name}")

        else:
            print("Provide either --image (single specimen) or --image-dir (batch).")

    elif args.command == "describe-biorag":
        # Validate group labels
        gl_path = Path(args.group_labels)
        if not gl_path.exists():
            print(f"Error: group labels file not found: {gl_path}")
            return
        import csv as csv_mod
        with open(gl_path) as f:
            reader = csv_mod.DictReader(f)
            gl_rows = list(reader)
        gl_species = sorted(set(r["group_label"] for r in gl_rows))
        logger.info(f"Group labels: {len(gl_rows)} images → {len(gl_species)} species: "
                     f"{', '.join(gl_species)}")

        # Load compiled data
        compiled = load_compiled_data(args.compiled_dir)
        if not compiled:
            print("Error: no compiled data found. Run compile_specimen_data.py first.")
            return

        # Load COCO JSON
        with open(args.coco_json) as f:
            coco_json = json.load(f)

        # Set up cache directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = None if args.no_cache else output_dir / "biorag_cache"

        # Build treatment index (optional RAG context)
        pipeline = None
        cached_index_path = str(cache_dir / "rag_index.json") if cache_dir else None

        if args.index:
            index = TreatmentIndex()
            index.load(args.index)
            pipeline = DescriptionPipeline(index)
        elif cached_index_path and Path(cached_index_path).exists() and args.pdf_dir:
            index = TreatmentIndex()
            index.load(cached_index_path)
            logger.info(f"Loaded cached RAG index: {len(index.chunks)} chunks")
            pipeline = DescriptionPipeline(index)
        elif args.pdf_dir:
            index = TreatmentIndex()
            pdf_ingester = PDFIngester()
            for pdf_file in Path(args.pdf_dir).glob("*.pdf"):
                chunks = pdf_ingester.extract_from_pdf(
                    str(pdf_file), taxon_family=args.family
                )
                index.add_chunks(chunks)
            logger.info(f"Built treatment index: {len(index.chunks)} chunks from PDFs")
            if cached_index_path:
                cache_dir.mkdir(parents=True, exist_ok=True)
                index.save(cached_index_path)
                logger.info(f"Cached RAG index: {cached_index_path}")
            pipeline = DescriptionPipeline(index)

        # Load ROSETTA vocabulary
        rosetta = None
        try:
            rosetta_path = Path(__file__).parent / "descriptron_rosetta.py"
            if rosetta_path.exists():
                sys.path.insert(0, str(rosetta_path.parent))
                from descriptron_rosetta import Rosetta
                r = Rosetta()
                rosetta = r.get_simple_attr()
                logger.info(f"ROSETTA vocabulary: {len(rosetta)} categories")
        except Exception as e:
            logger.warning(f"ROSETTA vocabulary not available: {e}")

        # Load user-provided taxonomist prompts
        user_prompts = {}
        if args.user_prompts:
            user_prompts = load_user_prompts(args.user_prompts)

        # Generate taxon schema from compiled data + COCO JSON
        schema_path = output_dir / "taxon_schema.yaml"
        taxon_schema = generate_taxon_schema(
            compiled=compiled,
            coco_json=coco_json,
            output_path=str(schema_path),
            family=args.family,
            user_prompts=user_prompts,
        )

        # Create Claude describer + BioRAG
        claude = ClaudeDescriber(model=args.model)
        biorag = BioragDescriber(
            claude_describer=claude,
            pipeline=pipeline,
            use_florence2=not args.no_florence2,
            cache_dir=cache_dir,
            user_prompts=user_prompts,
            taxon_schema=taxon_schema,
        )

        # Save run recipe
        n_user_prompts = sum(len(v) for v in user_prompts.values())
        save_biorag_recipe(args, output_dir, n_user_prompts)

        results = biorag.describe_all_species(
            compiled=compiled,
            semilandmarks_dirs=args.semilandmarks_dir,
            image_dir=args.image_dir,
            coco_json=coco_json,
            output_dir=output_dir,
            species_list=args.species,
            vision_mode=args.vision_mode,
            rosetta=rosetta,
            k=args.k,
            label_dir=args.label_dir,
            generate_key=args.generate_key,
        )

        ok = sum(1 for r in results.values() if "error" not in r)
        fail = len(results) - ok
        print(f"\nBioRAG complete: {ok} species described, {fail} failed")
        print(f"Output: {output_dir}")
        for sp, r in sorted(results.items()):
            if "error" in r:
                print(f"  ✗ {sp}: {r['error']}")
            else:
                n_cat = len(r.get("category_descriptions", []))
                print(f"  ✓ {sp}: {n_cat} categories")

    elif args.command == "correct-key":
        out = getattr(args, "output_file", None)
        corrected = correct_taxonomic_key(
            key_path=Path(args.key_file),
            check_report_path=Path(args.check_report),
            output_path=Path(out) if out else None,
            model=args.model,
        )
        if corrected:
            print(f"Key corrected: {len(corrected)} chars")
        else:
            print("Key correction failed")
            sys.exit(1)


if __name__ == "__main__":
    main()

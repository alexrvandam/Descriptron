#!/usr/bin/env python3
"""
descriptron_ontology.py — OntologyManager for Descriptron
==========================================================

Loads descriptron_ontology_templates.json and exposes the API expected by
descriptron-label-256-video-v58.py and v64.

Drop this file next to the main Descriptron script along with
descriptron_ontology_templates.json and (optionally) descriptron_rosetta.py.

File lookup order (all tried):
    1. ./descriptron_ontology_templates.json         (same dir as this script)
    2. ./measure/descriptron_ontology_templates.json
    3. ../descriptron_ontology_templates.json
    4. <path in DESCRIPTRON_ONTOLOGY_JSON env var>

This is a fresh implementation that matches the JSON schema produced by
patch_descriptron_ontology_templates.py --generate:

    {
      "version": "1.0",
      "groups": {
        "Formicidae": {
          "views": {
            "Head (dorsal)": {
              "regions": [{"id": "mandible", "label": "Mandible", "ontology": "HAO:0000506"}, ...],
              "attributes": {"texture": ["smooth", "rugose", ...], ...}
            }
          }
        }
      }
    }

Author: Alex Van Dam (alex.vandam@mfn.berlin)
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _find_templates_json() -> Optional[str]:
    """Find descriptron_ontology_templates.json in standard locations."""
    candidates = []

    # Env var override
    env_path = os.environ.get("DESCRIPTRON_ONTOLOGY_JSON")
    if env_path:
        candidates.append(env_path)

    # Relative to this module
    module_dir = Path(__file__).resolve().parent
    candidates.extend([
        str(module_dir / "descriptron_ontology_templates.json"),
        str(module_dir / "measure" / "descriptron_ontology_templates.json"),
        str(module_dir.parent / "descriptron_ontology_templates.json"),
    ])

    # Current working directory
    candidates.extend([
        "descriptron_ontology_templates.json",
        "./measure/descriptron_ontology_templates.json",
        "../descriptron_ontology_templates.json",
    ])

    for path in candidates:
        if Path(path).exists():
            return path

    return None


def _curie_to_uri(curie: str) -> str:
    """Convert a CURIE like 'HAO:0000506' to a full URI."""
    if not curie or ":" not in curie:
        return curie

    prefix, local = curie.split(":", 1)
    prefix_map = {
        "HAO":    "http://purl.obolibrary.org/obo/HAO_",
        "UBERON": "http://purl.obolibrary.org/obo/UBERON_",
        "AISM":   "http://purl.obolibrary.org/obo/AISM_",
        "COLAO":  "http://purl.obolibrary.org/obo/COLAO_",
        "PATO":   "http://purl.obolibrary.org/obo/PATO_",
        "PO":     "http://purl.obolibrary.org/obo/PO_",
        "CRU":    "http://purl.obolibrary.org/obo/CRU_",
        "LEP":    "http://purl.obolibrary.org/obo/LEP_",
        "HEM":    "http://purl.obolibrary.org/obo/HEM_",
        "AVE":    "http://purl.obolibrary.org/obo/AVE_",
        "MAM":    "http://purl.obolibrary.org/obo/MAM_",
        "NCIT":   "http://purl.obolibrary.org/obo/NCIT_",
    }
    base = prefix_map.get(prefix.upper())
    if base:
        return base + local
    return curie


# ═══════════════════════════════════════════════════════════════════════════════
# ONTOLOGY MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class OntologyManager:
    """
    Central manager for Descriptron ontology templates.

    Loads descriptron_ontology_templates.json and provides all template/
    attribute/annotation metadata operations used by the main GUI.

    Compatible with:
      - descriptron-label-256-video-v55.py
      - descriptron-label-256-video-v58.py
      - descriptron-label-256-video-v64.py
    """

    def __init__(self, templates_json_path: Optional[str] = None):
        # === Load templates ===
        if templates_json_path is None:
            templates_json_path = _find_templates_json()

        if templates_json_path is None or not Path(templates_json_path).exists():
            raise FileNotFoundError(
                "descriptron_ontology_templates.json not found. "
                "Set DESCRIPTRON_ONTOLOGY_JSON env var or place the file in the working directory."
            )

        with open(templates_json_path) as f:
            self._data = json.load(f)

        self._json_path = templates_json_path
        self.version = self._data.get("version", "1.0")
        self._groups_data = self._data.get("groups", {})

        # === Runtime state ===
        self.current_group: str = ""
        self.current_view: str = ""

        # category_name → human-readable label
        self.category_label: Dict[str, str] = {}

        # category_name → ontology CURIE/URI
        self._category_ontology: Dict[str, str] = {}

        # category_name → {attribute_name: [values]}
        self._category_attributes: Dict[str, Dict[str, List[str]]] = {}

        # annotation_id → {attribute_name: value}
        self._annotation_attributes: Dict[Any, Dict[str, str]] = {}

        # Track category id assignment
        self._next_cat_id = 1

    # ═══════════════════════════════════════════════════════════════════════
    # TEMPLATE DISCOVERY
    # ═══════════════════════════════════════════════════════════════════════

    def get_template_groups(self) -> List[str]:
        """Return list of all taxonomic group names."""
        return sorted(self._groups_data.keys())

    def get_views_for_group(self, group_name: str) -> List[str]:
        """Return list of view names for a given taxonomic group."""
        if group_name not in self._groups_data:
            return []
        views = self._groups_data[group_name].get("views", {})
        return list(views.keys())

    def get_template(self, group_name: str, view_name: str) -> Optional[Dict]:
        """
        Return the full template dict for a group+view.

        Returns: {"regions": [...], "attributes": {...}}
        """
        if group_name not in self._groups_data:
            return None
        views = self._groups_data[group_name].get("views", {})
        return views.get(view_name)

    def list_all_views(self) -> List[Tuple[str, str]]:
        """Return [(group, view), ...] across all groups."""
        result = []
        for group, gdata in self._groups_data.items():
            for view in gdata.get("views", {}):
                result.append((group, view))
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # TEMPLATE LOADING — populates runtime state
    # ═══════════════════════════════════════════════════════════════════════

    def load_template(self, group_name: str, view_name: str) -> Tuple[List[Dict], Dict]:
        """
        Load a template into the manager's active state.

        Called by the main GUI's open_template_loader() when user clicks Load.

        Returns:
            (cats, meta) where:
              cats: [{id, name, supercategory, ontology}, ...]  — COCO-ready
              meta: {category_name: {ontology, label, attributes}, ...}
        """
        tmpl = self.get_template(group_name, view_name)
        if tmpl is None:
            return [], {}

        self.current_group = group_name
        self.current_view = view_name

        # Reset per-template state
        self.category_label = {}
        self.category_ontology = {}
        self._category_attributes = {}

        # Template-level attributes (shared across all regions of this view)
        view_attributes: Dict[str, List[str]] = tmpl.get("attributes", {})

        cats: List[Dict] = []
        meta: Dict[str, Dict] = {}

        self._next_cat_id = 1
        for region in tmpl.get("regions", []):
            rid = region.get("id", "")
            label = region.get("label", rid)
            ontology = region.get("ontology", "")
            supercategory = region.get("supercategory", "sclerite")

            # Use the region label as the COCO category name (matches descriptron convention)
            # Keep it stable and readable
            cat_name = label

            cat_id = self._next_cat_id
            self._next_cat_id += 1

            cat_dict = {
                "id": cat_id,
                "name": cat_name,
                "supercategory": supercategory,
                "ontology": ontology,
            }
            cats.append(cat_dict)

            # Populate lookup dicts
            self.category_label[cat_name] = label
            self.category_ontology[cat_name] = ontology
            self._category_attributes[cat_name] = dict(view_attributes)  # shallow copy

            meta[cat_name] = {
                "ontology": ontology,
                "label": label,
                "supercategory": supercategory,
                "attributes": dict(view_attributes),
                "region_id": rid,
            }

        # Also add a "Trash" category so annotations can be marked for deletion
        trash_id = self._next_cat_id
        self._next_cat_id += 1
        cats.append({
            "id": trash_id,
            "name": "Trash",
            "supercategory": "trash",
            "ontology": "",
        })
        meta["Trash"] = {"ontology": "", "label": "Trash", "supercategory": "trash",
                         "attributes": {}, "region_id": "trash"}
        self.category_label["Trash"] = "Trash"
        self.category_ontology["Trash"] = ""
        self._category_attributes["Trash"] = {}

        return cats, meta

    # ═══════════════════════════════════════════════════════════════════════
    # ATTRIBUTE QUERIES (called by open_attribute_editor)
    # ═══════════════════════════════════════════════════════════════════════

    def get_attributes_for_category(self, category_name: str) -> List[str]:
        """Return list of attribute names available for a category (e.g. ['texture','color',...])."""
        attrs = self._category_attributes.get(category_name, {})
        return list(attrs.keys())

    def get_attribute_values(self, category_name: str, attribute_name: str) -> List[str]:
        """Return list of valid values for a category's attribute."""
        attrs = self._category_attributes.get(category_name, {})
        return attrs.get(attribute_name, [])

    # ═══════════════════════════════════════════════════════════════════════
    # PER-ANNOTATION ATTRIBUTE STORAGE
    # ═══════════════════════════════════════════════════════════════════════

    def set_annotation_attributes(self, annotation_id: Any, attrs: Dict[str, str]):
        """Store attribute values for a specific annotation."""
        # Remove keys with empty values
        clean = {k: v for k, v in attrs.items() if v and str(v).strip()}
        if clean:
            self._annotation_attributes[annotation_id] = clean
        elif annotation_id in self._annotation_attributes:
            del self._annotation_attributes[annotation_id]

    def get_annotation_attributes(self, annotation_id: Any) -> Dict[str, str]:
        """Get attribute values for a specific annotation."""
        return dict(self._annotation_attributes.get(annotation_id, {}))

    def get_annotation_notes(self, annotation_id: Any) -> str:
        """Get free-text notes for an annotation (stored under __notes__ key)."""
        return self._annotation_attributes.get(annotation_id, {}).get("__notes__", "")

    def set_annotation_notes(self, annotation_id: Any, notes: str):
        """Set free-text notes for an annotation."""
        if annotation_id not in self._annotation_attributes:
            self._annotation_attributes[annotation_id] = {}
        if notes and notes.strip():
            self._annotation_attributes[annotation_id]["__notes__"] = notes.strip()
        elif "__notes__" in self._annotation_attributes.get(annotation_id, {}):
            del self._annotation_attributes[annotation_id]["__notes__"]

    def clear_annotation_attributes(self, annotation_id: Any):
        """Remove all stored attributes for an annotation."""
        self._annotation_attributes.pop(annotation_id, None)

    def get_all_annotation_attributes(self) -> Dict[Any, Dict[str, str]]:
        """Return the full annotation_id → attributes map (read-only copy)."""
        return {aid: dict(a) for aid, a in self._annotation_attributes.items()}

    # ═══════════════════════════════════════════════════════════════════════
    # COCO IMPORT — read GBIF annotator exports
    # ═══════════════════════════════════════════════════════════════════════

    def import_gbif_coco(self, coco_data: Dict) -> Tuple[List[Dict], Dict, Dict, List]:
        """
        Import a COCO JSON from the GBIF annotator.

        Returns:
            (cats, meta, annotation_attrs, annotations)
              cats: categories list (same format as load_template)
              meta: category metadata
              annotation_attrs: {annotation_id: {attr: value}}
              annotations: list of annotation dicts
        """
        info = coco_data.get("info", {})
        group = info.get("template_group", "") or info.get("group", "")
        view = info.get("template_view", "") or info.get("view", "")

        # If a template is identified, load it to populate the vocabulary
        if group and view:
            template = self.get_template(group, view)
            if template is not None:
                self.load_template(group, view)

        # Merge in categories from the COCO file
        coco_cats = coco_data.get("categories", [])
        cats: List[Dict] = []
        meta: Dict[str, Dict] = {}

        for cat in coco_cats:
            cat_name = cat.get("name", "")
            if not cat_name:
                continue
            cat_dict = {
                "id": cat.get("id", self._next_cat_id),
                "name": cat_name,
                "supercategory": cat.get("supercategory", "sclerite"),
                "ontology": cat.get("ontology", ""),
            }
            cats.append(cat_dict)
            self.category_label[cat_name] = cat_name
            self.category_ontology[cat_name] = cat.get("ontology", "")
            meta[cat_name] = {
                "ontology": cat.get("ontology", ""),
                "label": cat_name,
                "supercategory": cat.get("supercategory", "sclerite"),
                "attributes": self._category_attributes.get(cat_name, {}),
            }

        # Extract per-annotation attributes
        annotation_attrs: Dict[Any, Dict[str, str]] = {}
        annotations = coco_data.get("annotations", [])
        for ann in annotations:
            ann_id = ann.get("id")
            attrs = ann.get("attributes", {})
            if ann_id is not None and attrs:
                # Clean string values
                clean = {k: str(v) for k, v in attrs.items() if v}
                if clean:
                    annotation_attrs[ann_id] = clean
                    self._annotation_attributes[ann_id] = clean

        return cats, meta, annotation_attrs, annotations

    # ═══════════════════════════════════════════════════════════════════════
    # COCO EXPORT — enhance with ontology metadata
    # ═══════════════════════════════════════════════════════════════════════

    def export_enhanced_coco(self, base_coco: Dict, gbif_occurrence_id: str = "") -> Dict:
        """
        Produce an enhanced COCO JSON with ontology and attributes embedded.
        """
        enhanced = {
            "info": dict(base_coco.get("info", {})),
            "images": list(base_coco.get("images", [])),
            "categories": [],
            "annotations": [],
        }

        # Add ontology info to info block
        enhanced["info"].update({
            "descriptron_version": "1.0",
            "ontology_version": self.version,
            "template_group": self.current_group,
            "template_view": self.current_view,
            "exported_at": datetime.now().isoformat(),
        })
        if gbif_occurrence_id:
            enhanced["info"]["gbif_occurrence_id"] = gbif_occurrence_id

        # Enhance categories with ontology URIs
        for cat in base_coco.get("categories", []):
            cat_enhanced = dict(cat)
            name = cat.get("name", "")
            ont = self.category_ontology.get(name, "")
            if ont and not cat_enhanced.get("ontology"):
                cat_enhanced["ontology"] = ont
                cat_enhanced["ontology_uri"] = _curie_to_uri(ont)
            enhanced["categories"].append(cat_enhanced)

        # Enhance annotations with stored attributes
        for ann in base_coco.get("annotations", []):
            ann_enhanced = dict(ann)
            ann_id = ann.get("id")
            if ann_id in self._annotation_attributes:
                ann_enhanced["attributes"] = dict(self._annotation_attributes[ann_id])
            enhanced["annotations"].append(ann_enhanced)

        return enhanced

    # ═══════════════════════════════════════════════════════════════════════
    # KNOWLEDGE GRAPH EXPORT (JSON-LD)
    # ═══════════════════════════════════════════════════════════════════════

    def export_knowledge_graph(self, enhanced_coco: Dict,
                                specimen_id: str = "unknown",
                                scientific_name: str = "") -> Dict:
        """
        Build a JSON-LD knowledge graph from enhanced COCO.

        Mirrors the knowledge graph export in descriptron-gbif-annotator.html.
        """
        spec_uri = f"urn:descriptron:specimen:{specimen_id}"
        nodes: List[Dict] = []
        edges: List[Dict] = []

        # Specimen node
        specimen_node = {
            "id": spec_uri,
            "type": "Specimen",
            "label": scientific_name or specimen_id,
            "properties": {"specimen_id": specimen_id},
        }
        nodes.append(specimen_node)

        # Taxon node
        if scientific_name:
            tax_uri = f"urn:gbif:taxon:{scientific_name.replace(' ', '_')}"
            nodes.append({
                "id": tax_uri,
                "type": "Taxon",
                "label": scientific_name,
                "properties": {"scientificName": scientific_name},
            })
            edges.append({"source": spec_uri, "target": tax_uri, "relation": "identifiedAs"})

        # Image nodes
        for img in enhanced_coco.get("images", []):
            img_id = img.get("id", "1")
            img_uri = f"{spec_uri}:image:{img_id}"
            nodes.append({
                "id": img_uri,
                "type": "Image",
                "label": img.get("file_name", ""),
                "properties": {
                    "width": img.get("width"),
                    "height": img.get("height"),
                    "file_name": img.get("file_name", ""),
                },
            })
            edges.append({"source": spec_uri, "target": img_uri, "relation": "hasImage"})

        # Build cat_id → cat lookup
        cat_map = {c["id"]: c for c in enhanced_coco.get("categories", [])}

        # Region + trait nodes from annotations
        for ann in enhanced_coco.get("annotations", []):
            cat_id = ann.get("category_id")
            cat = cat_map.get(cat_id)
            if not cat or cat.get("name", "").lower() == "trash":
                continue

            ann_id = ann.get("id", "")
            region_uri = f"{spec_uri}:region:{ann_id}"
            region_label = cat.get("name", "")
            region_ont = cat.get("ontology", "")

            nodes.append({
                "id": region_uri,
                "type": "AnatomicalRegion",
                "label": region_label,
                "properties": {
                    "ontology": region_ont,
                    "ontology_uri": _curie_to_uri(region_ont) if region_ont else "",
                    "category_id": cat_id,
                },
            })
            edges.append({"source": spec_uri, "target": region_uri, "relation": "hasRegion"})

            # Segmentation mask node
            if ann.get("segmentation") or ann.get("bbox"):
                mask_uri = f"{region_uri}:mask"
                nodes.append({
                    "id": mask_uri,
                    "type": "SegmentationMask",
                    "label": f"{region_label} mask",
                    "properties": {
                        "bbox": ann.get("bbox", []),
                        "area": ann.get("area"),
                    },
                })
                edges.append({"source": region_uri, "target": mask_uri, "relation": "hasSegmentation"})

            # Trait nodes from attributes
            attrs = ann.get("attributes", {})
            for attr_name, value in attrs.items():
                if not value or attr_name.startswith("__"):
                    continue
                trait_uri = f"{region_uri}:trait:{attr_name}"
                nodes.append({
                    "id": trait_uri,
                    "type": "MorphologicalTrait",
                    "label": f"{attr_name}: {value}",
                    "properties": {
                        "traitType": attr_name,
                        "value": value,
                        "scope": "region",
                    },
                })
                edges.append({"source": region_uri, "target": trait_uri, "relation": "hasTrait"})

        # Build JSON-LD
        kg = {
            "@context": {
                "@vocab": "https://descriptron.org/ontology/",
                "dwc":    "http://rs.tdwg.org/dwc/terms/",
                "uberon": "http://purl.obolibrary.org/obo/UBERON_",
                "hao":    "http://purl.obolibrary.org/obo/HAO_",
                "aism":   "http://purl.obolibrary.org/obo/AISM_",
                "colao":  "http://purl.obolibrary.org/obo/COLAO_",
                "pato":   "http://purl.obolibrary.org/obo/PATO_",
                "identifiedAs":      {"@type": "@id"},
                "hasImage":          {"@type": "@id"},
                "hasRegion":         {"@type": "@id"},
                "hasTrait":          {"@type": "@id"},
                "hasSegmentation":   {"@type": "@id"},
            },
            "@type": "SpecimenAnnotationGraph",
            "specimenURI": spec_uri,
            "timestamp": datetime.now().isoformat(),
            "descriptron_version": "1.0",
            "ontology_version": self.version,
            "template": f"{self.current_group}/{self.current_view}" if self.current_group else "",
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "nodeCount": len(nodes),
                "edgeCount": len(edges),
                "regionCount": sum(1 for n in nodes if n["type"] == "AnatomicalRegion"),
                "traitCount":  sum(1 for n in nodes if n["type"] == "MorphologicalTrait"),
                "maskCount":   sum(1 for n in nodes if n["type"] == "SegmentationMask"),
            },
        }

        return kg

    # ═══════════════════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════

    def summary(self) -> Dict:
        groups = self.get_template_groups()
        n_views = sum(len(self.get_views_for_group(g)) for g in groups)
        n_regions = 0
        for g in groups:
            for v in self.get_views_for_group(g):
                tmpl = self.get_template(g, v)
                if tmpl:
                    n_regions += len(tmpl.get("regions", []))
        return {
            "version": self.version,
            "json_path": self._json_path,
            "num_groups": len(groups),
            "num_views": n_views,
            "num_regions": n_regions,
            "current_group": self.current_group,
            "current_view": self.current_view,
        }


    # === CUSTOM_CATEGORY_ONTOLOGY_PATCH v1 ===
    def register_custom_category(self,
                                  category_name: str,
                                  parent_category: str = "",
                                  ontology_curie: str = "",
                                  supercategory: str = "sclerite",
                                  extra_attributes: list = None) -> dict:
        """
        Register a user-created custom category (e.g. 'left_elytron') and give
        it the full attribute schema inherited from a parent region (e.g. 'elytra').

        Called by the main GUI when user creates a new label via the Custom option
        in the Select Label dropdown.

        Args:
            category_name: New category name (e.g. 'left_elytron').
            parent_category: Existing category to inherit attributes/ontology from.
                             If empty, inherits from the current template view's
                             attribute set (same dropdowns all regions in this view get).
            ontology_curie: Optional CURIE override (e.g. 'HAO:0001234'). If empty
                             and parent_category is given, inherits parent's CURIE.
            supercategory: COCO supercategory label.
            extra_attributes: Optional list of attribute name strings to add on top
                              of inherited ones (e.g. ['evidence_source']).

        Returns:
            Dict {'category_name', 'ontology', 'supercategory',
                   'attributes', 'inherited_from'}
        """
        # Inherit attributes
        inherited_attrs = {}
        if parent_category and parent_category in self._category_attributes:
            inherited_attrs = dict(self._category_attributes[parent_category])
            # Also inherit ontology if no override given
            if not ontology_curie:
                ontology_curie = self.category_ontology.get(parent_category, "")
        elif self.current_group and self.current_view:
            # Fallback: use the template's view-level attributes
            tmpl = self.get_template(self.current_group, self.current_view)
            if tmpl:
                inherited_attrs = dict(tmpl.get("attributes", {}))

        # Add extra attributes from ROSETTA if requested
        if extra_attributes:
            try:
                from descriptron_rosetta import ROSETTA
                for attr_name in extra_attributes:
                    vals = ROSETTA.get_values(attr_name)
                    if vals:
                        inherited_attrs[attr_name] = vals
            except ImportError:
                pass  # ROSETTA not available, skip

        # Register the category in all the lookup dicts
        self.category_label[category_name] = category_name
        self.category_ontology[category_name] = ontology_curie
        self._category_attributes[category_name] = inherited_attrs

        return {
            "category_name": category_name,
            "ontology": ontology_curie,
            "supercategory": supercategory,
            "attributes": inherited_attrs,
            "inherited_from": parent_category or (f"template:{self.current_group}/{self.current_view}"
                                                    if self.current_group else ""),
        }

    def get_current_template_region_names(self) -> list:
        """Return list of region label names from the currently loaded template.
        Used by the main GUI to populate the 'parent region' dropdown when
        creating a custom label."""
        if not self.current_group or not self.current_view:
            return []
        return [name for name in self.category_label.keys() if name != "Trash"]

    def update_custom_category_ontology(self, category_name: str,
                                         ontology_curie: str) -> bool:
        """Update the ontology CURIE for an already-registered custom category.
        Returns True if updated, False if category not found."""
        if category_name not in self.category_label:
            return False
        self.category_ontology[category_name] = ontology_curie
        return True
    # === END CUSTOM_CATEGORY_ONTOLOGY_PATCH v1 ===
    def register_custom_category(self, name, parent_category=None, ontology_curie=None):
        """
        Registers a new category at runtime, inheriting attributes from a parent.
        """
        # Note the underscore in _category_attributes
        if parent_category and parent_category in self._category_attributes:
            # Inherit the attribute schema (texture, color, etc.)
            self._category_attributes[name] = self._category_attributes[parent_category]
            
            # Inherit ontology mapping if parent has one and none provided
            if not ontology_curie:
                parent_ont = self._category_ontology.get(parent_category, "")
                self._category_ontology[name] = parent_ont
        
        if ontology_curie:
            self._category_ontology[name] = ontology_curie

    def get_attributes_for_category(self, category_name):
        """Helper to get attributes, checking both templates and custom registrations"""
        return self._category_attributes.get(category_name, [])
    def __repr__(self):
        s = self.summary()
        return (f"OntologyManager(v{s['version']}, "
                f"{s['num_groups']} groups, {s['num_views']} views, "
                f"{s['num_regions']} regions)")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI diagnostic
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    try:
        mgr = OntologyManager()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Loaded: {mgr}")
    print(f"JSON:   {mgr._json_path}")
    print()

    print("Groups:")
    for g in mgr.get_template_groups():
        views = mgr.get_views_for_group(g)
        print(f"  {g:25s} ({len(views)} views)")

    if len(sys.argv) > 1 and sys.argv[1] == "--detail":
        for g in mgr.get_template_groups():
            print(f"\n=== {g} ===")
            for v in mgr.get_views_for_group(g):
                tmpl = mgr.get_template(g, v)
                regions = [r.get("label", "") for r in tmpl.get("regions", [])]
                attrs = list(tmpl.get("attributes", {}).keys())
                print(f"  {v}")
                print(f"    Regions: {', '.join(regions[:8])}{'...' if len(regions) > 8 else ''}")
                print(f"    Attrs:   {', '.join(attrs)}")

    if len(sys.argv) > 1 and sys.argv[1] == "--test-load":
        # Test loading Formicidae / Head (dorsal)
        g, v = "Formicidae", "Head (dorsal)"
        print(f"\n--- Testing load_template('{g}', '{v}') ---")
        cats, meta = mgr.load_template(g, v)
        print(f"Categories loaded: {len(cats)}")
        for c in cats[:5]:
            print(f"  id={c['id']} name={c['name']} ontology={c.get('ontology','')}")
        print(f"\nAttributes for first category ('{cats[0]['name']}'):")
        attrs = mgr.get_attributes_for_category(cats[0]['name'])
        for a in attrs:
            vals = mgr.get_attribute_values(cats[0]['name'], a)
            print(f"  {a}: {len(vals)} values — {vals[:5]}{'...' if len(vals) > 5 else ''}")

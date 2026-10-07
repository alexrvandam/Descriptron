"""v82 descriptive characters + category editing (plain-Python parts; the GUI flow is tested separately)."""
import sys
from pathlib import Path

import pytest

GUI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GUI))
import descriptron_category_edit as ce             # noqa: E402
import descriptron_descriptive_characters as dc    # noqa: E402


@pytest.fixture(scope="module")
def vocab():
    return dc.load_vocabulary()


def test_vocabulary_is_complete_and_ascii(vocab):
    assert "Arthropod cuticle" in vocab["default_sets"]
    for s, keys in vocab["sets"].items():
        for k in keys:
            assert k in vocab["characters"], (s, k)
    for k, c in vocab["characters"].items():
        assert c["type"] in ("categorical", "count")
        assert c["type"] == "count" or c["values"], k
    raw = (GUI / dc.VOCAB_FILE).read_bytes()
    assert all(b < 128 for b in raw)                # Tk labels: ASCII only


def test_annotator_characters_present(vocab):
    for k in ("texture", "sculpture", "setae", "color", "color_pattern", "luster", "surface_covering"):
        assert k in vocab["characters"]
    assert "punctate" in vocab["characters"]["texture"]["values"]


def test_clean_attributes(vocab):
    attrs, probs = dc.clean_attributes({"texture": " punctate ", "setae": "", "setae_count": "012",
                                        "puncture_count": "many"}, vocab)
    assert attrs == {"texture": "punctate", "setae_count": "12"}
    assert len(probs) == 1 and "number of punctures" in probs[0]


def test_sets_and_memory(vocab):
    keys = dc.characters_in_sets(vocab, ["Arthropod cuticle", "Counts (meristic)"])
    assert keys[0] == "texture" and "setae_count" in keys and len(keys) == len(set(keys))
    st = dc.PanelState(vocab); st.remember("elytron", {"setae": "dense"}); st.remember("elytron", {})
    assert st.last["elytron"] == {"setae": "dense"}
    assert "setae: dense" in dc.summary({"setae": "dense"}, vocab)


COCO = {"categories": [{"id": 1, "name": "mesosomu"}, {"id": 2, "name": "mesosoma"}, {"id": 3, "name": "head"}],
        "annotations": [{"id": 10, "category_id": 1}, {"id": 11, "category_id": 2}, {"id": 12, "category_id": 3}]}


def test_rename_fixes_a_misspelling():
    out = ce.rename_category({"categories": [{"id": 1, "name": "mesosomu"}], "annotations": [{"id": 1, "category_id": 1}]},
                             "mesosomu", "mesosoma")
    assert out["categories"] == [{"id": 1, "name": "mesosoma"}] and out["annotations"][0]["category_id"] == 1


def test_rename_to_existing_name_merges():
    out = ce.rename_category(COCO, "mesosomu", "mesosoma")
    assert [c["name"] for c in out["categories"]] == ["mesosoma", "head"]
    assert [a["category_id"] for a in out["annotations"]] == [2, 2, 3]
    assert COCO["categories"][0]["name"] == "mesosomu"        # input untouched


def test_delete_removes_category_and_its_annotations():
    out, n = ce.delete_category(COCO, "head")
    assert n == 1 and [c["name"] for c in out["categories"]] == ["mesosomu", "mesosoma"]
    assert [a["id"] for a in out["annotations"]] == [10, 11]
    with pytest.raises(KeyError):
        ce.delete_category(COCO, "nope")


def test_corrected_copy_never_overwrites():
    assert ce.corrected_path("/x/ann.json") == "/x/ann_categories_fixed.json"
    assert ce.corrected_path("/x/ann.rlejson") == "/x/ann_categories_fixed.rlejson"

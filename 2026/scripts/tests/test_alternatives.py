import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import stitch_text as st


def test_alt_markup_replace_delete_insert():
    assert st.alt_markup("oſa bien entreprendre trabir &", "oſa bien entreprendre trahir &") == \
        "oſa bien entreprendre trabir⟨alt:trahir⟩ &"
    assert st.alt_markup("a b c", "a c") == "a b⟨alt:⟩ c"
    assert st.alt_markup("a c", "a b c") == "a ⟨alt:+b⟩ c"


def test_mark_alternatives_rewrites_only_pointed_lines():
    page = {"blocks": [{"type": "paragraph", "lines": ["x trabir y", "unchanged"]}],
            "margin_notes": [{"key": "a", "lines": ["l. famoſi."]}],
            "uncertain": [
                {"where": "blocks[0].lines[0]", "text": "x trabir y",
                 "note": st.ALT_PREFIX + "x trabir y" + st.ALT_SEP + "x trahir y"},
                {"where": "margin_notes[0].lines[0]", "text": "l. famoſi.",
                 "note": st.ALT_PREFIX + "l. famoſi." + st.ALT_SEP + "l. famoſa."},
                {"where": "blocks[0].lines[1]", "text": "unchanged", "note": "sic"},
            ]}
    out = st.mark_alternatives(page)
    assert out["blocks"][0]["lines"] == ["x trabir⟨alt:trahir⟩ y", "unchanged"]
    assert out["margin_notes"][0]["lines"] == ["l. famoſi.⟨alt:famoſa.⟩"]
    assert page["blocks"][0]["lines"][0] == "x trabir y"      # input untouched


def test_mark_alternatives_noop_without_entries():
    page = {"blocks": [], "uncertain": [{"where": "blocks[0].lines[0]", "note": "sic"}]}
    assert st.mark_alternatives(page) is page


def test_mark_alternatives_unknown_uses_alt_query():
    page = {"blocks": [{"type": "paragraph", "lines": ["x trabir y", "a b c"]}],
            "uncertain": [
                {"where": "blocks[0].lines[0]", "text": "x trabir y", "escalate": True,
                 "note": st.ALT_PREFIX_UNKNOWN + "x trabir y" + st.ALT_SEP + "x trahir y"},
                {"where": "blocks[0].lines[1]", "text": "a b c",
                 "note": st.ALT_PREFIX + "a b c" + st.ALT_SEP + "a c"},
            ]}
    out = st.mark_alternatives(page)
    assert out["blocks"][0]["lines"] == ["x trabir⟨alt?:trahir⟩ y", "a b⟨alt:⟩ c"]
    assert st.alt_markup("a c", "a b c", "alt?") == "a ⟨alt?:+b⟩ c"

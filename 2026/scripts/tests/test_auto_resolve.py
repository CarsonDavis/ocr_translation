import sys, pathlib, copy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import auto_resolve

def page(lines, notes):
    return {"blocks": [{"type": "heading", "text": "TEXTE."}, {"type": "paragraph", "lines": lines}],
            "margin_notes": [{"key": k, "lines": v} for k, v in notes.items()], "foot_notes": []}

def test_split_form_wins_in_body_and_notes():
    a = page(["par lesloix politiques", "cẽt & onze"], {"a": ["de eo qui cog."]})
    b = page(["par les loix politiques", "cét & onze"], {"a": ["deeo qui cog."]})
    n = auto_resolve.resolve(a, b)
    assert n == 2
    assert a["blocks"][1]["lines"][0] == b["blocks"][1]["lines"][0] == "par les loix politiques"
    assert a["margin_notes"][0]["lines"][0] == b["margin_notes"][0]["lines"][0] == "de eo qui cog."
    # a genuine reading difference is left alone
    assert a["blocks"][1]["lines"][1] == "cẽt & onze" and b["blocks"][1]["lines"][1] == "cét & onze"

def test_heading_kept_in_place():
    a = page(["x"], {}); b = page(["x"], {}); b["blocks"][0]["text"] = "TEXTE ."
    auto_resolve.resolve(a, b)
    assert a["blocks"][0]["text"] == "TEXTE ." and b["blocks"][0]["text"] == "TEXTE ."

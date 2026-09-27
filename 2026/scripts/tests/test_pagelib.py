# scripts/tests/test_pagelib.py
import pathlib
import sys
import unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pagelib  # noqa: E402

align = pagelib.align_key


# --- align_key: the loose form used to align two reads --------------------

def test_long_s_folds():
    assert align("ſon fils eſt") == align("son fils est")


def test_case_folds():
    assert align("ARREST DV") == align("arrest dv")


def test_spacing_around_punctuation_is_ignored():
    assert align("bondir , & ſi") == align("bondir, & ſi")
    assert align("la  femme") == align("la femme")
    assert align("  trailing ") == align("trailing")


def test_marker_keys_survive():
    assert "{a2}" in align("à ſauteller {a2}. Et")
    assert align("bondir {u}.") != align("bondir {v}.")


def test_u_and_v_are_not_folded():
    """u/v and i/j are transcribed as printed (§2); alignment must not hide them."""
    assert align("vne") != align("une")
    assert align("iamais") != align("jamais")


def test_nfc_and_nfd_agree():
    assert align("hõnesteté") == align(unicodedata.normalize("NFD", "hõnesteté"))


def test_combining_tilde_keeps_the_word_boundary():
    """q̃ has no precomposed form, so its tilde must not eat the space after it."""
    key = align("q̃ eſt là")
    assert key == "q̃ est là"
    assert key != align("q̃eſt là")


def test_empty_line():
    assert align("") == ""


# --- points_at: does an uncertain[] `where` point into this container? -----

def test_points_at_equal():
    assert pagelib.points_at("blocks[1]", "blocks[1]")
    assert pagelib.points_at("folio", "folio")


def test_points_at_line_under_block():
    assert pagelib.points_at("blocks[1].lines[3]", "blocks[1]")
    assert pagelib.points_at("blocks[1] line 3", "blocks[1]")
    assert pagelib.points_at("margin_notes[0].lines[2]", "margin_notes[0]")


def test_points_at_is_boundary_aware():
    assert not pagelib.points_at("blocks[10].lines[0]", "blocks[1]")
    assert not pagelib.points_at("blocks[1]", "blocks[1].lines[3]")
    assert not pagelib.points_at("folio_note", "folio")


# --- block_types ----------------------------------------------------------

def test_block_types_tolerates_junk():
    page = {"blocks": [{"type": "heading"}, "ornament", {"text": "no type"}]}
    assert pagelib.block_types(page) == ["heading", None, None]


def test_long_s_marker_key_is_a_valid_marker():
    import pagelib
    page = {"blocks": [{"type": "paragraph", "lines": ["de deſpon{ſ}. impub. et {t} auſſi"]}],
            "margin_notes": [], "foot_notes": []}
    assert [k for k, _ in pagelib.markers(page)] == ["ſ", "t"]


def test_column_texts_includes_headings_in_block_order():
    import pagelib
    page = {"blocks": [{"type": "heading", "text": "TEXTE."},
                       {"type": "paragraph", "lines": ["a", "b"]},
                       {"type": "ornament", "text": "rule"},
                       {"type": "heading", "text": "ANNOTAT. I."}]}
    assert pagelib.column_texts(page) == ["TEXTE.", "a", "b", "ANNOTAT. I."]

# scripts/tests/test_stitch.py
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import stitch_text as st  # noqa: E402


# --- helpers ---------------------------------------------------------------

def para(*lines, prev=False, nxt=False):
    return {"type": "paragraph", "continues_prev": prev, "continues_next": nxt,
            "lines": list(lines)}


def head(text):
    return {"type": "heading", "text": text}


def page(page_id, *blocks, margin=(), foot=()):
    return page_id, {"id": page_id, "blocks": list(blocks),
                     "margin_notes": list(margin), "foot_notes": list(foot)}


def note(key, *lines, beside=None):
    return {"key": key, "lines": list(lines), "beside_line": beside}


def by_id(sections):
    return {s["id"]: s for s in sections}


# --- reflow: hyphens -------------------------------------------------------

def test_hyphen_join_drops_the_hyphen():
    assert st.reflow(["qui nevou-", "lut demãder"]) == "qui nevoulut demãder"


def test_plain_line_end_joins_with_one_space():
    assert st.reflow(["de la", "femme"]) == "de la femme"


def test_hyphen_keep_list_keeps_the_hyphen():
    assert st.reflow(["grand-", "mere"], keep={"grandmere"}) == "grand-mere"
    assert st.reflow(["grand-", "mere"]) == "grandmere"


def test_keep_list_skips_comments_and_folds_entries(tmp_path):
    p = tmp_path / "hyphen_keep.txt"
    p.write_text("# compound words\ngrand-mere\n\nCeſt-à-dire\n", encoding="utf-8")
    assert st.load_keep(p) == {"grandmere", "ceſtàdire"}


def test_capital_after_hyphen_keeps_the_hyphen():
    assert st.reflow(["exemple-", "Dieu ſoit"]) == "exemple-Dieu ſoit"


def test_three_lines_reflow_in_order():
    assert st.reflow(["a b", "c-", "d e"]) == "a b cd e"


# --- heading detection -----------------------------------------------------

def test_texte_headings():
    assert st.parse_heading("TEXTE.") == ("texte", None)
    assert st.parse_heading("TEXTE") == ("texte", None)


def test_annotation_headings():
    assert st.parse_heading("ANNOTATION I.") == ("annotation", 1)
    assert st.parse_heading("ANNOTAT. V.") == ("annotation", 5)
    assert st.parse_heading("ANNOT. XXIIII.") == ("annotation", 24)
    assert st.parse_heading("ANNOTAT. XLIIII.") == ("annotation", 44)
    assert st.parse_heading("ANNOTAT. XCIX.") == ("annotation", 99)
    assert st.parse_heading("ANNOT. CXI.") == ("annotation", 111)


def test_unreadable_numeral_is_an_annotation_without_a_number():
    assert st.parse_heading("ANNOTAT. [??].") == ("annotation", None)


def test_display_headings_are_not_section_headings():
    for text in ("Texte de la Toile du procés,", "& de l'arreſt.",
                 "ARGVMENT ET SOM-", "MAIRE DV FAICT.", "ARREST", "M. D. LXXII."):
        assert st.parse_heading(text) is None


# --- markers and page contributions ---------------------------------------

def test_marker_opens_each_page_contribution():
    secs = st.stitch([page("p001", head("TEXTE."), para("Au mois de")),
                      page("p002", para("Ianuier"))])
    assert secs[0]["text"] == "⟦p001⟧Au mois de\n\n⟦p002⟧Ianuier"


def test_continuation_across_a_page_break_has_no_paragraph_break():
    secs = st.stitch([page("p040", head("ANNOTAT. V."), para("qui nevou-", nxt=True)),
                      page("p041", para("lut demãder", prev=True))])
    assert secs[0]["text"] == "⟦p040⟧qui nevou⟦p041⟧lut demãder"
    assert "\n\n" not in secs[0]["text"]


def test_continuation_without_a_hyphen_joins_with_a_space():
    secs = st.stitch([page("p017", head("TEXTE."), para("de la", nxt=True)),
                      page("p018", para("femme", prev=True))])
    assert secs[0]["text"] == "⟦p017⟧de la ⟦p018⟧femme"


def test_paragraphs_that_do_not_continue_are_separated():
    secs = st.stitch([page("p017", head("TEXTE."), para("un", nxt=False)),
                      page("p018", para("deux", prev=False))])
    assert secs[0]["text"] == "⟦p017⟧un\n\n⟦p018⟧deux"


# --- section structure -----------------------------------------------------

def test_title_and_argument_pages_are_their_own_sections():
    secs = st.stitch([page("p000-title", head("ARREST"), head("MEMORABLE")),
                      page("p000-argument", head("ARGVMENT ET SOM-"),
                           head("MAIRE DV FAICT."), para("MArtin Guerre")),
                      page("p001", head("Texte de la Toile du procés,"),
                           para("AV MOIS de Ianuier"))])
    ids = [s["id"] for s in secs]
    assert ids == ["title", "argument", "texte-01"]
    assert secs[0]["kind"] == "title"
    assert secs[0]["text"] == "⟦p000-title⟧ARREST\n\nMEMORABLE"
    assert secs[1]["kind"] == "argument"
    assert secs[1]["text"].startswith("⟦p000-argument⟧ARGVMENT ET SOM-\n\nMAIRE DV FAICT.")
    # page 1's title lines and the "Texte de la Toile" heading belong to texte-01
    assert secs[2]["text"] == "⟦p001⟧Texte de la Toile du procés,\n\nAV MOIS de Ianuier"


def test_sections_are_numbered_and_carry_their_pages():
    secs = st.stitch([
        page("p001", para("un")),
        page("p002", head("ANNOTATION I."), para("annot un", nxt=True)),
        page("p004", para("suite", prev=True), head("TEXTE."), para("texte deux"),
             head("ANNOTATION II."), para("annot deux")),
    ])
    s = by_id(secs)
    assert [x["id"] for x in secs] == ["texte-01", "annot-001", "texte-02", "annot-002"]
    assert s["annot-001"]["pages"] == ["p002", "p004"]
    assert s["annot-001"]["label"] == "ANNOTATION I."
    assert s["annot-001"]["number"] == 1
    assert s["texte-02"]["pages"] == ["p004"]
    assert s["texte-02"]["number"] == 2


def test_mid_page_boundaries_are_flagged():
    secs = st.stitch([
        page("p040", head("ANNOTAT. V."), para("cinq", nxt=True)),
        page("p041", para("suite", prev=True), head("TEXTE."), para("texte")),
    ])
    s = by_id(secs)
    assert s["annot-005"]["starts_mid_page"] is False
    assert s["annot-005"]["ends_mid_page"] is True
    assert s["texte-01"]["starts_mid_page"] is True


def test_unreadable_numeral_numbers_sequentially_and_is_flagged():
    secs = st.stitch([page("p001", head("ANNOTAT. IIII."), para("a")),
                      page("p002", head("ANNOTAT. [??]."), para("b"))])
    assert [s["id"] for s in secs] == ["annot-004", "annot-005"]
    assert secs[1]["number_uncertain"] is True
    assert "number_uncertain" not in secs[0]


# --- notes -----------------------------------------------------------------

def test_notes_attach_by_marker_and_reflow():
    secs = st.stitch([page("p002", head("ANNOTATION I."), para("le mariage {a}."),
                           margin=[note("a", "Chap. der-", "nier au titre")])])
    assert secs[0]["notes"] == [{"key": "a", "page": "p002",
                                 "text": "Chap. dernier au titre"}]
    assert "missing_notes" not in secs[0]


def test_note_without_a_marker_in_the_text_is_an_orphan():
    secs = st.stitch([page("p002", head("ANNOTATION I."), para("sans marqueur"),
                           foot=[note("q", "Pline au liure xi.")])])
    assert secs[0]["notes"][0]["orphan"] is True


def test_unkeyed_note_gets_the_underscore_key():
    secs = st.stitch([page("p002", head("ANNOTATION I."), para("sans marqueur"),
                           margin=[note(None, "Gen. chap. 1.")])])
    assert secs[0]["notes"][0]["key"] == "_"


def test_marker_without_a_note_is_listed_as_missing():
    secs = st.stitch([page("p044", head("ANNOT. XXIIII."), para("ruſe {a} & {b}."),
                           margin=[note("a", "A la premiere")])])
    assert secs[0]["missing_notes"] == ["b@p044"]


def test_a_note_lands_in_the_section_that_holds_its_marker():
    secs = st.stitch([page("p044", para("ceſte de ROLS {a}"), head("TEXTE."),
                           para("Au bout {b}"),
                           margin=[note("a", "A la pre-", "miere"), note("b", "Pline")])])
    s = by_id(secs)
    assert [n["key"] for n in s["texte-01"]["notes"]] == ["a"]
    assert [n["key"] for n in s["texte-02"]["notes"]] == ["b"]


# --- prefix walk -----------------------------------------------------------

def _tree(tmp_path, ids, done, files):
    manifest = {"pages": [{"id": i, "status": {"final": "done" if i in done else "pending"}}
                          for i in ids]}
    final = tmp_path / "final"
    final.mkdir()
    for i in files:
        (final / f"{i}.json").write_text(json.dumps({"id": i, "blocks": []}), encoding="utf-8")
    return manifest, final


def test_walk_stops_at_the_first_page_that_is_not_done(tmp_path):
    ids = ["p001", "p002", "p003", "p004"]
    manifest, final = _tree(tmp_path, ids, done={"p001", "p002", "p004"}, files=set(ids))
    pages, stopped = st.walk(manifest, final)
    assert [p for p, _ in pages] == ["p001", "p002"]
    assert stopped == "p003"


def test_walk_stops_at_a_missing_final_file(tmp_path):
    ids = ["p001", "p002", "p003"]
    manifest, final = _tree(tmp_path, ids, done=set(ids), files={"p001", "p003"})
    pages, stopped = st.walk(manifest, final)
    assert [p for p, _ in pages] == ["p001"]
    assert stopped == "p002"


def test_walk_consumes_everything_when_nothing_is_pending(tmp_path):
    ids = ["p001", "p002"]
    manifest, final = _tree(tmp_path, ids, done=set(ids), files=set(ids))
    pages, stopped = st.walk(manifest, final)
    assert [p for p, _ in pages] == ids
    assert stopped is None


def test_the_open_section_at_the_stop_is_incomplete():
    secs = st.stitch([page("p001", head("TEXTE."), para("un")),
                      page("p002", head("ANNOTATION I."), para("deux", nxt=True))])
    s = by_id(secs)
    assert s["texte-01"]["complete"] is True
    assert s["annot-001"]["complete"] is False


# --- validation ------------------------------------------------------------

def test_validate_accepts_a_clean_prefix():
    pages = [page("p001", head("TEXTE."), para("un")), page("p002", para("deux"))]
    secs = st.stitch(pages)
    errors, warnings = st.validate(secs, [p for p, _ in pages])
    assert errors == []


def test_validate_reports_page_markers_out_of_order():
    secs = [{"id": "texte-01", "text": "⟦p002⟧a ⟦p001⟧b", "notes": [],
             "pages": ["p002", "p001"], "complete": True}]
    errors, _ = st.validate(secs, ["p001", "p002"])
    assert any("not in manifest order" in e for e in errors)


def test_validate_reports_a_repeated_marker_inside_one_section():
    secs = [{"id": "texte-01", "text": "⟦p001⟧a ⟦p001⟧b", "notes": [],
             "pages": ["p001"], "complete": True}]
    errors, _ = st.validate(secs, ["p001"])
    assert any("repeated page marker" in e for e in errors)


def test_validate_reports_markers_that_disagree_with_pages():
    secs = [{"id": "texte-01", "text": "⟦p001⟧a", "notes": [],
             "pages": ["p001", "p002"], "complete": True}]
    errors, _ = st.validate(secs, ["p001", "p002"])
    assert any("do not match" in e for e in errors)


def test_validate_reports_a_consumed_page_no_section_marks():
    secs = [{"id": "texte-01", "text": "⟦p001⟧a", "notes": [],
             "pages": ["p001"], "complete": True}]
    errors, _ = st.validate(secs, ["p001", "p002"])
    assert any("no section marks p002" in e for e in errors)


def test_validate_allows_one_page_to_be_marked_by_two_sections():
    pages = [page("p044", para("fin"), head("TEXTE."), para("suite"))]
    secs = st.stitch(pages)
    assert [s["text"] for s in secs] == ["⟦p044⟧fin", "⟦p044⟧suite"]
    assert st.validate(secs, ["p044"])[0] == []


def test_validate_reports_duplicate_section_ids():
    secs = [{"id": "annot-001", "text": "⟦p001⟧a", "notes": [], "pages": ["p001"],
             "complete": True},
            {"id": "annot-001", "text": "⟦p002⟧b", "notes": [], "pages": ["p002"],
             "complete": True}]
    errors, _ = st.validate(secs, ["p001", "p002"])
    assert any("duplicate section id" in e for e in errors)


def test_section_line_reports_id_pages_completeness_and_notes():
    sec = {"id": "annot-005", "pages": ["p040", "p041"], "complete": False,
           "notes": [{"key": "a"}]}
    line = st.section_line(sec)
    assert line.split() == ["annot-005", "p040,p041", "INCOMPLETE", "1", "notes"]
    sec["complete"] = True
    assert "complete" in st.section_line(sec)


def test_every_section_marks_every_page_it_draws_on():
    pages = [page("p008", para("fin du texte"), head("ANNOTAT. III."), para("annot"),
                  head("TEXTE."), para("texte", nxt=True)),
             page("p009", para("suite", prev=True))]
    secs = st.stitch(pages)
    s = by_id(secs)
    assert s["texte-01"]["text"] == "⟦p008⟧fin du texte"
    # a section wholly inside a page still opens with that page's marker
    assert s["annot-003"]["text"] == "⟦p008⟧annot"
    # and the page shared by three sections is marked in each of them
    assert s["texte-02"]["text"] == "⟦p008⟧texte ⟦p009⟧suite"
    errors, warnings = st.validate(secs, [p for p, _ in pages])
    assert errors == []
    assert not any("page marker" in w for w in warnings)


def test_each_section_marks_a_page_only_once():
    secs = st.stitch([page("p010", para("un"), head("TEXTE."), para("deux"),
                           para("trois"))])
    assert secs[1]["text"] == "⟦p010⟧deux\n\ntrois"
    assert secs[1]["pages"] == ["p010"]


def test_validate_warns_about_orphan_and_missing_notes():
    secs = st.stitch([page("p044", head("ANNOT. XXIIII."), para("ruſe {a} & {b}."),
                           margin=[note("c", "orpheline")])])
    errors, warnings = st.validate(secs, ["p044"])
    assert errors == []
    assert any("orphan" in w for w in warnings)
    assert any("b@p044" in w for w in warnings)



def test_out_of_sequence_annotation_number_is_a_misprint(tmp_path):
    """p034 prints 'ANNOTAT. XIII.' where the sequence needs XVIII: number it XVIII and
    record what was printed."""
    import stitch_text
    pages = [("p001", {"blocks": [{"type": "heading", "text": "ANNOTAT. XVII."},
                                  {"type": "paragraph", "lines": ["a"]}], "margin_notes": [], "foot_notes": []}),
             ("p002", {"blocks": [{"type": "heading", "text": "ANNOTAT. XIII."},
                                  {"type": "paragraph", "lines": ["b"]}], "margin_notes": [], "foot_notes": []})]
    secs = stitch_text.stitch(pages, set())
    ids = [s["id"] for s in secs]
    assert ids == ["annot-017", "annot-018"], ids
    assert secs[1]["number_printed"] == 13 and secs[1]["number_uncertain"] is True


# --- wrong-sort annotation headings ----------------------------------------

def test_wrong_sort_annotation_abbreviations_are_headings():
    assert st.parse_heading("ANNNT. LX.") == ("annotation", 60)      # p080
    assert st.parse_heading("ANOTAT. V.") == ("annotation", 5)
    assert st.parse_heading("ANNOTA. XII.") == ("annotation", 12)
    assert st.parse_heading("ANNOTAT, XXIIII.") == ("annotation", 24)
    assert st.parse_heading("ANNOTAT. XX,") == ("annotation", 20)


def test_fuzzy_annotation_match_leaves_display_lines_alone():
    for text in ("ARREST.", "ANNNT. de la", "ANNNT.", "MEMORABLE",
                 "TEXTE DV PROCES", "A RAISON CEDE."):
        assert st.parse_heading(text) in (None,), text


# --- wrong-sort TEXTE headings ---------------------------------------------

def test_wrong_sort_texte_headings_are_headings():
    for text in ("TFXTE.", "TBXTE.", "TEXTB.",          # p045, p058, p072
                 "TEXTF,", "TEXT.", "TEXTES.", "TEXTE ."):
        assert st.parse_heading(text) == ("texte", None), text


def test_fuzzy_texte_needs_a_stop_and_one_edit_at_most():
    for text in ("TEXTB", "TFXTE", "TBXTB.", "TEXTE DV PROCES", "Texte.", "TFXTE. de",
                 "ARREST.", "TESTER.", "EXPOSITION DES", "Guerre."):
        assert st.parse_heading(text) is None, text


def test_wrong_sort_texte_closes_the_annotation_before_it():
    secs = st.stitch([page("p071", head("ANNOTAT. L."), para("a")),
                      page("p072", head("TEXTB."), para("b"), head("ANNOT. LI."), para("c"))])
    assert [s["id"] for s in secs] == ["annot-050", "texte-01", "annot-051"]
    assert secs[0]["text"] == "⟦p071⟧a"
    assert secs[1]["label"] == "TEXTB." and secs[1]["text"] == "⟦p072⟧b"


def test_wrong_sort_heading_opens_its_annotation():
    secs = st.stitch([page("p079", head("ANNOT. LIX."), para("a")),
                      page("p080", head("TEXTE."), para("b"), head("ANNNT. LX."), para("c")),
                      page("p082", head("ANNOTAT. LXI."), para("d"))])
    assert [s["id"] for s in secs] == ["annot-059", "texte-01", "annot-060", "annot-061"]
    assert secs[2]["text"] == "⟦p080⟧c"


# --- a page that opens mid-word --------------------------------------------

def test_page_opening_mid_word_does_not_start_a_paragraph():
    """p095 ends 'meſme-', p096 opens 'ment': one word, one paragraph, even when the
    continues_* flags are false."""
    secs = st.stitch([page("p095", head("TEXTE."),
                           para("qu'il n'eſt beſoin icy d'eſcrire, meſme-", nxt=False)),
                      page("p096", para("ment que toutes ſont vaines", prev=False))])
    assert secs[0]["text"] == ("⟦p095⟧qu'il n'eſt beſoin icy d'eſcrire, "
                               "meſme⟦p096⟧ment que toutes ſont vaines")


def test_mid_word_join_also_ignores_a_missing_continues_prev():
    p96 = {"type": "paragraph", "lines": ["ment que"]}
    secs = st.stitch([page("p095", head("TEXTE."), para("meſme-")), page("p096", p96)])
    assert "\n\n" not in secs[0]["text"]


def test_hyphen_before_a_heading_does_not_join_across_it():
    secs = st.stitch([page("p095", head("TEXTE."), para("meſme-")),
                      page("p096", head("TEXTE."), para("ment"))])
    assert secs[1]["text"] == "⟦p096⟧ment"

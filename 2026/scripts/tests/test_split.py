# scripts/tests/test_split.py
import collections
import json
import pathlib
import shutil
import sys

import jsonschema
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import split_pages  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "site"
FINAL = FIX / "transcription" / "final"
IDS = ["p004", "p005", "p006", "p041"]
REAL_SECTION = ROOT / "translation" / "sections" / "annot-001.md"


def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def final(page_id):
    return read(FINAL / f"{page_id}.json")


@pytest.fixture
def built(tmp_path):
    """(counts, out_dir) for one build of the fixture book."""
    counts = split_pages.build(FIX, tmp_path)
    return counts, tmp_path


@pytest.fixture
def book(tmp_path):
    """A writable copy of the fixture book, for tests that need to break it."""
    root = tmp_path / "book"
    shutil.copytree(FIX, root)
    return root


def page(out, page_id):
    return read(out / "pages" / f"{page_id}.json")


def sections_dir(root):
    return pathlib.Path(root) / "translation" / "sections"


def write_sections(root, records, files):
    """Replace the fixture's section index and its translated files."""
    root = pathlib.Path(root)
    (root / "text").mkdir(exist_ok=True)
    (root / "text" / "sections.json").write_text(
        json.dumps({"generated": "2026-09-21T00:00:00", "sections": records}),
        encoding="utf-8")
    directory = sections_dir(root)
    shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for name, body in files.items():
        (directory / name).write_text(body, encoding="utf-8")
    return directory


def one_section(root, body, section_id="annot-001", kind="annotation", number=1,
                pages=("p004",), file_name=None):
    """Replace the fixture's sections with a single section."""
    record = {"id": section_id, "kind": kind, "number": number, "pages": list(pages)}
    return write_sections(root, [record], {(file_name or f"{section_id}.md"): body})


# --- the French layer -----------------------------------------------------

def test_french_layer_copied(built):
    counts, out = built
    rec = page(out, "p004")
    src = final("p004")
    assert rec["french"] == src["blocks"]
    assert rec["french_notes"][0] == {
        "key": "ſ",
        "kind": "margin",
        "lines": [
            "L. minorem",
            "D. de rit. nup",
            "paragr. i. de",
            "nu, aux Inſti-",
            "tutions de Iu-",
            "ſtinien c. pube",
            "res. de deſpon-",
            "ſ. impub.",
        ],
    }
    assert len(rec["french_notes"]) == len(src["margin_notes"]) + len(src["foot_notes"])
    assert rec["running_head"] == "ARREST DV"
    assert rec["folio"] == "4" and rec["side"] == "verso"
    assert rec["image"] == "p004"
    # `beside_line` and the reader's `decisions` stay out of the site data
    assert all(set(n) == {"key", "kind", "lines"} for n in rec["french_notes"])
    assert "decisions" not in rec
    # uncertain keeps only where/text/note
    assert rec["uncertain"] == [
        {k: v for k, v in u.items() if k in ("where", "text", "note")}
        for u in src["uncertain"]
    ]
    assert counts == (4, 3, 3)


def test_foot_notes_follow_margin_notes(built):
    _, out = built
    rec = page(out, "p041")
    src = final("p041")
    kinds = [n["kind"] for n in rec["french_notes"]]
    assert kinds == ["margin"] * len(src["margin_notes"]) + ["foot"] * len(src["foot_notes"])
    assert [n["key"] for n in rec["french_notes"]] == (
        [n["key"] for n in src["margin_notes"]] + [n["key"] for n in src["foot_notes"]]
    )


def test_pending_page(built):
    _, out = built
    rec = page(out, "p006")
    assert rec["french"] is None
    assert rec["french_notes"] is None
    assert rec["uncertain"] == []
    assert rec["english"] is None
    # furniture still comes from the manifest
    assert rec["folio"] == "6" and rec["side"] == "verso" and rec["running_head"] is None
    entry = {p["id"]: p for p in read(out / "index.json")["pages"]}["p006"]
    assert entry["layers"] == {"fr": False, "en": False}
    assert entry["heading"] is None


def test_missing_final_file_is_pending(book, tmp_path, capsys):
    """status.final says done but the file is not there yet: pending, not an error."""
    (book / "transcription" / "final" / "p004.json").unlink()
    out = tmp_path / "out"
    counts = split_pages.build(book, out)
    assert counts == (4, 2, 3)
    assert page(out, "p004")["french"] is None
    entry = {p["id"]: p for p in read(out / "index.json")["pages"]}["p004"]
    assert entry["layers"]["fr"] is False


def test_final_without_blocks_is_pending(book, tmp_path):
    """A done final that loads as an empty object must not count as a French layer."""
    (book / "transcription" / "final" / "p004.json").write_text(
        json.dumps({"id": "p004", "reader": "final"}), encoding="utf-8")
    out = tmp_path / "out"
    counts = split_pages.build(book, out)
    assert counts[1] == 2
    assert page(out, "p004")["french"] is None
    assert page(out, "p004")["french_notes"] is None
    entry = {p["id"]: p for p in read(out / "index.json")["pages"]}["p004"]
    assert entry["layers"]["fr"] is False


def test_prev_next_and_order(built):
    _, out = built
    index = read(out / "index.json")
    assert [p["id"] for p in index["pages"]] == IDS
    assert page(out, "p004")["prev"] is None
    assert page(out, "p004")["next"] == "p005"
    assert page(out, "p005")["prev"] == "p004"
    assert page(out, "p041")["next"] is None


def test_index_records(built):
    _, out = built
    index = read(out / "index.json")
    by_id = {p["id"]: p for p in index["pages"]}
    assert by_id["p004"] == {"id": "p004", "page": 4, "folio": "4", "heading": "TEXT",
                             "layers": {"fr": True, "en": True}}
    assert by_id["p041"]["heading"] is None  # p041 has no heading block


def test_index_heading_prefers_the_annotation():
    """A page that opens with the quoted TEXTE is listed under its annotation."""
    page_final = {"blocks": [{"type": "heading", "text": "TEXTE."},
                             {"type": "paragraph", "lines": ["…"]},
                             {"type": "heading", "text": "ANNOTAT. VI."}]}
    assert split_pages.index_heading({"id": "p012"}, page_final) == "ANNOTATION VI"
    assert split_pages.first_heading(page_final) == "TEXT"
    only_text = {"blocks": [{"type": "heading", "text": "TEXTE."}]}
    assert split_pages.index_heading({"id": "p012"}, only_text) == "TEXT"
    assert split_pages.index_heading({"id": "p012"}, {"blocks": []}) is None
    assert split_pages.index_heading({"id": "p012"}, None) is None


def test_index_heading_is_null_for_the_front_matter():
    title = {"blocks": [{"type": "heading", "text": "ARREST"}]}
    assert split_pages.index_heading({"id": "p000-title"}, title) is None
    assert split_pages.index_heading({"id": "p000-argument"}, title) is None


def test_cudl_source(built):
    _, out = built
    src = page(out, "p004")["source"]
    assert src == {"kind": "cudl", "image_no": 26,
                   "url": "https://cudl.lib.cam.ac.uk/view/PR-MONTAIGNE-00001-00007-00022/26"}


def test_gallica_source(built):
    _, out = built
    src = page(out, "p041")["source"]
    assert src["kind"] == "gallica"
    assert src["image_no"] is None
    assert src["url"] == "https://gallica.bnf.fr/ark:/12148/bpt6k52469j/f58"


def test_cudl_without_an_image_number_raises(book, tmp_path):
    manifest = read(book / "manifest.json")
    manifest["pages"][0]["image"] = None
    (book / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="p004"):
        split_pages.build(book, tmp_path / "out")


# --- the English layer ----------------------------------------------------

def test_split_sections(built):
    """The fixture's three translated sections, laid out over the pages."""
    _, out = built
    p004 = page(out, "p004")["english"]
    assert [b["type"] for b in p004] == ["heading", "paragraph", "paragraph"]
    assert p004[0] == {"type": "heading", "text": "ANNOTATION I"}
    assert p004[2]["html"].endswith("runs on to the wo")

    p005 = page(out, "p005")["english"]
    assert [b["type"] for b in p005] == ["paragraph", "heading", "paragraph"]
    assert p005[0]["html"].startswith("man of the next")

    p041 = page(out, "p041")["english"]
    assert [b["type"] for b in p041] == ["heading", "paragraph"]
    # annot-002 covers p006 but has not been translated yet
    assert page(out, "p006")["english"] is None


def test_heading_synthesized_between_two_sections_on_one_page(built):
    """p005 carries the tail of annot-001 and the head of texte-02."""
    _, out = built
    p005 = page(out, "p005")["english"]
    assert p005[1] == {"type": "heading", "text": "TEXT"}
    assert p005[2]["html"].startswith("With whom she had dwelt")


def test_an_explicit_heading_paragraph_is_not_doubled(built):
    """texte-03 writes TEXT as its own paragraph, so nothing is synthesized."""
    _, out = built
    p041 = page(out, "p041")["english"]
    assert [b for b in p041 if b["type"] == "heading"] == [{"type": "heading",
                                                            "text": "TEXT"}]


def test_section_heading_comes_from_the_section_record():
    """The printed label wins; kind and number only name a section without one."""
    heading = split_pages.section_heading
    assert heading("annotation", 1, "ANNOTATION I.") == "ANNOTATION I"
    assert heading("annotation", 3, "ANNOTAT. III.") == "ANNOTATION III"
    assert heading("texte", 2, "TEXTE.") == "TEXT"
    # the book prints IIII, so the English layer says IIII too, as the index does
    assert heading("annotation", 4, "ANNOTAT. IIII.") == "ANNOTATION IIII"
    assert heading("annotation", 4) == "ANNOTATION IV"
    assert heading("texte", 2) == "TEXT"
    assert heading("annotation", 1) == "ANNOTATION I"
    assert heading("title", None) is None
    assert heading("argument", None) is None


def test_a_labelled_section_keeps_the_printed_numeral(book, tmp_path):
    write_sections(book, [{"id": "annot-004", "kind": "annotation", "number": 4,
                           "label": "ANNOTAT. IIII.", "pages": ["p004"]}],
                   {"annot-004.md": "---\nid: annot-004\npages: [p004]\n---\n"
                                    "⟦p004⟧The fourth annotation.\n"})
    out = tmp_path / "out"
    split_pages.build(book, out)
    assert page(out, "p004")["english"][0] == {"type": "heading",
                                               "text": "ANNOTATION IIII"}


def test_roman():
    assert [split_pages.roman(n) for n in (1, 4, 5, 9, 14, 24, 40, 49, 111)] == [
        "I", "IV", "V", "IX", "XIV", "XXIV", "XL", "XLIX", "CXI"]


def test_marker_html(built):
    _, out = built
    first = page(out, "p004")["english"][1]["html"]
    assert '<sup class="mk" data-key="a">a</sup>' in first
    assert '<sup class="mk" data-key="b">b</sup>' in first
    assert "&amp; the issue of them" in first
    assert "{" not in first
    assert "<i>Parlement</i>" in page(out, "p041")["english"][1]["html"]
    note = page(out, "p004")["english"][1]["notes"][0]
    assert note["citation"] == "Digest 23.2 (<i>De ritu nuptiarum</i>), l. <i>Minorem</i>"


def test_to_html_escapes_before_marking_up():
    out = split_pages.to_html("a < b & c {a2} *is*\n  spaced")
    assert out == ('a &lt; b &amp; c <sup class="mk" data-key="a2">a2</sup> '
                   "<i>is</i> spaced")


def test_continued_detection(built):
    """The fixture breaks `woman` across p004/p005, as the print does."""
    _, out = built
    assert page(out, "p004")["english"][1]["continued"] is False
    assert page(out, "p004")["english"][2]["continued"] is False
    assert page(out, "p005")["english"][0]["continued"] is True
    # a section that starts on a page starts a fresh paragraph there
    assert page(out, "p005")["english"][2]["continued"] is False
    assert page(out, "p041")["english"][1]["continued"] is False


def test_notes_attached_in_marker_order(built):
    _, out = built
    notes = page(out, "p004")["english"][1]["notes"]
    assert [n["key"] for n in notes] == ["a", "b"]
    assert notes[0]["original"] == "L. minorem D. de rit. nup."
    assert notes[0]["gloss"] == "On the age of consent."
    assert notes[1]["gloss"] is None
    assert page(out, "p004")["english"][2]["notes"] == []
    assert [n["key"] for n in page(out, "p041")["english"][1]["notes"]] == ["a"]


def test_notes_are_scoped_by_their_page(built):
    """Key `a` is used on p004 and again on p005; they are different notes."""
    _, out = built
    p004_a = page(out, "p004")["english"][1]["notes"][0]
    p005_a = page(out, "p005")["english"][0]["notes"][0]
    assert p004_a["key"] == p005_a["key"] == "a"
    assert p004_a["citation"].startswith("Digest 23.2")
    assert p005_a["citation"].startswith("Macrobius")


def test_orphan_note_lands_on_the_first_paragraph_with_a_warning(tmp_path, capsys):
    split_pages.build(FIX, tmp_path)
    err = capsys.readouterr().err
    # the fixture's note c (p005) is never marked in the prose
    assert [n["key"] for n in page(tmp_path, "p005")["english"][0]["notes"]] == ["a", "c"]
    assert "p005" in err and "'c'" in err and "first paragraph" in err


def test_a_note_may_carry_an_aside_and_the_block_a_preamble(book, tmp_path):
    """The translator explains the print's mis-keying; the site keeps the citation."""
    one_section(book, "---\nid: annot-001\npages: [p004]\n---\n⟦p004⟧A line {a}.\n\n"
                      "## Notes\n"
                      "The print's marker alphabet is mis-set: *t* has no marker.\n"
                      "- {a} (p004) — printed with the key *b*: **Digest 24.2.6 "
                      "(*De divortiis*)** — L. vxores D. de diuort. [A gloss.]\n")
    out = tmp_path / "out"
    split_pages.build(book, out)
    note = page(out, "p004")["english"][1]["notes"][0]
    assert note["citation"] == "Digest 24.2.6 (<i>De divortiis</i>)"
    assert note["original"] == "L. vxores D. de diuort."
    assert note["gloss"] == "A gloss."


def test_unkeyed_notes_verse_and_noteless_markers(book, tmp_path, capsys):
    """`{_}` (no printed key) and `- Verse (p…)` are notes with no key, beside the
    page's first paragraph; `- {c} (p…) — aside` with no citation is a printed
    marker with no note, and no note at all."""
    one_section(book, "---\nid: annot-001\npages: [p004]\n---\n⟦p004⟧One {a}.\n\n"
                      "Two {c}.\n\n## Notes\n"
                      "- {a} (p004): **Digest 1.1** — L. i.\n"
                      "- {c} (p004) — marker with no note in the margin.\n"
                      "- {_} (p004) — unkeyed note, no printed key: **Digest 49.16.5** "
                      "— l. non omnes [A gloss.]\n"
                      "- Verse (p004): **Serenus, *Liber medicinalis*** — Interdum "
                      "exiſtit turpi verruca papilla\n")
    out = tmp_path / "out"
    split_pages.build(book, out)
    paras = [b for b in page(out, "p004")["english"] if b["type"] == "paragraph"]
    assert [(n["key"], n["citation"]) for n in paras[0]["notes"]] == [
        ("a", "Digest 1.1"), (None, "Digest 49.16.5"),
        (None, "Serenus, <i>Liber medicinalis</i>")]
    assert paras[0]["notes"][1]["gloss"] == "A gloss."
    assert paras[1]["notes"] == []
    assert "not marked in the prose" not in capsys.readouterr().err


def test_a_keyed_line_with_text_but_no_citation_still_errors(book, tmp_path):
    """Only a dash-led aside reads as a marker with no note."""
    one_section(book, "---\nid: annot-001\npages: [p004]\n---\n⟦p004⟧A line {a}.\n\n"
                      "## Notes\n- {a} (p004): Digest 1.1 — L. i.\n")
    with pytest.raises(ValueError, match="annot-001.md"):
        split_pages.build(book, tmp_path / "out")


def test_a_notes_block_may_say_there_are_none(built):
    """texte-02's `- (none: …)` line is not an entry."""
    _, out = built
    assert page(out, "p005")["english"][2]["notes"] == []


def test_a_note_line_that_does_not_parse_errors(book, tmp_path):
    one_section(book, "---\nid: annot-001\npages: [p004]\n---\n⟦p004⟧A line {a}.\n\n"
                      "## Notes\n- {a} (p004): no bold citation here\n")
    with pytest.raises(ValueError, match="annot-001.md"):
        split_pages.build(book, tmp_path / "out")


def test_marker_variants(book, tmp_path):
    assert split_pages.page_id_for_marker("p043") == "p043"
    assert split_pages.page_id_for_marker("p.43") == "p043"
    assert split_pages.page_id_for_marker("p43") == "p043"
    assert split_pages.page_id_for_marker("p000-title") == "p000-title"
    one_section(book, "---\nid: annot-001\npages: [p004, p005]\n---\n"
                      "⟦p.4⟧One. ⟦p5⟧Two.\n", pages=("p004", "p005"))
    out = tmp_path / "out"
    assert split_pages.build(book, out)[2] == 2
    assert page(out, "p004")["english"][1]["html"] == "One."
    assert page(out, "p005")["english"][0]["html"] == "Two."


def test_unknown_page_marker_errors(book, tmp_path):
    one_section(book, "---\nid: annot-009\npages: [p999]\n---\n⟦p999⟧Nowhere.\n",
                section_id="annot-009", pages=("p999",))
    with pytest.raises(ValueError, match="annot-009"):
        split_pages.build(book, tmp_path / "out")


def test_note_for_an_unknown_page_errors(book, tmp_path):
    one_section(book, "---\nid: annot-001\npages: [p004]\n---\n⟦p004⟧Here {a}.\n\n"
                      "## Notes\n- {a} (p999): **A citation** — L'original.\n")
    with pytest.raises(ValueError, match="p999"):
        split_pages.build(book, tmp_path / "out")


def test_front_matter_id_must_match_the_file_name(book, tmp_path):
    one_section(book, "---\nid: annot-002\npages: [p004]\n---\n⟦p004⟧Mismatched.\n",
                file_name="annot-001.md")
    with pytest.raises(ValueError, match="annot-001.md"):
        split_pages.build(book, tmp_path / "out")


def test_front_matter_pages_are_required(book, tmp_path):
    one_section(book, "---\nid: annot-001\n---\n⟦p004⟧No pages key.\n")
    with pytest.raises(ValueError, match="pages"):
        split_pages.build(book, tmp_path / "out")


def test_front_matter_is_required(tmp_path):
    path = tmp_path / "annot-001.md"
    path.write_text("⟦p004⟧A section with no front matter.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="front matter"):
        split_pages.parse_section(path)


def test_prose_before_the_first_marker_errors(book, tmp_path):
    one_section(book, "---\nid: annot-001\npages: [p004]\n---\nStray prose.\n\n"
                      "⟦p004⟧A line.\n")
    with pytest.raises(ValueError, match="annot-001.md"):
        split_pages.build(book, tmp_path / "out")


def test_a_page_may_not_be_covered_twice(book, tmp_path):
    write_sections(book, [
        {"id": "annot-001", "kind": "annotation", "number": 1, "pages": ["p004", "p005"]},
        {"id": "annot-002", "kind": "annotation", "number": 2, "pages": ["p004"]},
    ], {
        "annot-001.md": "---\nid: annot-001\npages: [p004, p005]\n---\n⟦p004⟧A.\n\n⟦p005⟧B.\n",
        "annot-002.md": "---\nid: annot-002\npages: [p004]\n---\n⟦p004⟧C.\n",
    })
    with pytest.raises(ValueError, match="p004"):
        split_pages.build(book, tmp_path / "out")


def test_two_sections_may_share_the_page_they_meet_on(built):
    """The tail of annot-001 and the head of texte-02 live on p005 together."""
    _, out = built
    assert len(page(out, "p005")["english"]) == 3


def test_a_section_with_no_file_is_simply_pending(book, tmp_path, capsys):
    """annot-002 is indexed but not translated; p006 stays pending, quietly."""
    out = tmp_path / "out"
    split_pages.build(book, out)
    assert page(out, "p006")["english"] is None
    assert "annot-002" not in capsys.readouterr().err


def test_index_flags_en(built):
    _, out = built
    flags = {p["id"]: p["layers"]["en"] for p in read(out / "index.json")["pages"]}
    assert flags == {"p004": True, "p005": True, "p006": False, "p041": True}


def test_no_sections_index_means_all_null(book, tmp_path):
    (book / "text" / "sections.json").unlink()
    out = tmp_path / "out"
    assert split_pages.build(book, out) == (4, 3, 0)
    assert all(page(out, page_id)["english"] is None for page_id in IDS)
    assert not any(p["layers"]["en"] for p in read(out / "index.json")["pages"])


def test_no_translated_files_means_all_null(book, tmp_path):
    for md in sections_dir(book).glob("*.md"):
        md.unlink()
    assert split_pages.build(book, tmp_path / "out") == (4, 3, 0)


@pytest.mark.skipif(not REAL_SECTION.exists(),
                    reason="the real translation is not in this checkout")
def test_a_real_section_file_parses():
    section = split_pages.parse_section(REAL_SECTION)
    assert section.id == "annot-001"
    assert section.pages == ["p002", "p003", "p004"]
    assert [p.page for p in section.pieces if p.page] == ["p002", "p003", "p004"]
    assert collections.Counter(n.page for n in section.notes) == {
        "p002": 6, "p003": 11, "p004": 8}
    assert all(n.citation and n.original for n in section.notes)
    first = section.notes[0]
    assert first.key == "a" and first.page == "p002"
    assert "<i>De frigidis" in first.citation
    assert first.original.startswith("Chap. dernier")


# --- the schema -----------------------------------------------------------

def test_schema_valid(built):
    _, out = built
    schema = read(ROOT / "scripts" / "site_schema.json")
    for path in sorted((out / "pages").glob("*.json")):
        jsonschema.validate(read(path), schema)


def test_schema_rejects_a_bad_page(built):
    _, out = built
    schema = read(ROOT / "scripts" / "site_schema.json")
    rec = page(out, "p004")
    rec["french"][0].pop("lines")          # a paragraph with neither lines nor html
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(rec, schema)
    bad = page(out, "p004")
    bad["source"]["kind"] = "flickr"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
    gone = page(out, "p004")
    del gone["french_notes"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(gone, schema)


def test_schema_checks_the_english_layer(built):
    _, out = built
    schema = read(ROOT / "scripts" / "site_schema.json")
    rec = page(out, "p004")
    jsonschema.validate(rec, schema)       # an English paragraph is valid as written
    no_html = page(out, "p004")
    del no_html["english"][1]["html"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(no_html, schema)
    no_gloss = page(out, "p004")
    del no_gloss["english"][1]["notes"][0]["gloss"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(no_gloss, schema)
    extra = page(out, "p004")
    extra["english"][1]["notes"][0]["footnote"] = "no"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(extra, schema)


def test_build_fails_loudly_on_an_invalid_page(tmp_path, monkeypatch):
    monkeypatch.setattr(split_pages, "page_record", lambda *a, **k: {"id": "p004"})
    with pytest.raises(Exception) as exc:
        split_pages.build(FIX, tmp_path)
    assert "p004" in str(exc.value)


def test_a_failed_build_leaves_the_out_dir_alone(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "pages").mkdir(parents=True)
    (out / "index.json").write_text('{"pages": ["OLD"]}\n', encoding="utf-8")
    (out / "pages" / "p999.json").write_text('{"id": "p999"}\n', encoding="utf-8")
    monkeypatch.setattr(split_pages, "page_record", lambda *a, **k: {"id": "p004"})
    with pytest.raises(Exception):
        split_pages.build(FIX, out)
    assert read(out / "index.json") == {"pages": ["OLD"]}
    assert sorted(p.name for p in out.iterdir()) == ["index.json", "pages"]
    # nothing half-written: the old pages are still exactly what was there
    assert sorted(p.name for p in (out / "pages").iterdir()) == ["p999.json"]
    assert read(out / "pages" / "p999.json") == {"id": "p999"}


def test_an_empty_manifest_raises(book, tmp_path):
    (book / "manifest.json").write_text('{"pages": []}', encoding="utf-8")
    with pytest.raises(split_pages.pagelib.PageLoadError):
        split_pages.build(book, tmp_path / "out")
    (book / "manifest.json").write_text('{"pages": "nope"}', encoding="utf-8")
    with pytest.raises(split_pages.pagelib.PageLoadError):
        split_pages.build(book, tmp_path / "out")


# --- heading normalization ------------------------------------------------

def test_normalize_heading():
    n = split_pages.normalize_heading
    assert n("ANNOTAT. V.") == "ANNOTATION V"
    assert n("ANNOTAT. XLIV.") == "ANNOTATION XLIV"
    assert n("TEXTE.") == "TEXT"
    assert n("ARGVMENT.") == "ARGUMENT"
    assert n("ANNOTATION I.") == "ANNOTATION I"
    assert n("  DV PARLEMENT DE  ") == "DV PARLEMENT DE"
    assert n("MEMORABLE") == "MEMORABLE"


# --- book.json and housekeeping -------------------------------------------

def test_book_json_written(built):
    _, out = built
    book = read(out / "book.json")
    assert book["slug"] == "martin-guerre"
    assert book["first_page"] == "p000-title"
    assert book["images"] == {"base": "img/", "ext": ".webp", "width": 2805}
    assert [layer["code"] for layer in book["layers"]] == ["en", "fr"]
    assert book["default_layer"] == "en"
    assert book["stylesheet"] is None
    assert book["description"].strip()


def test_stale_page_removed(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir(parents=True)
    stale = pages / "p999.json"
    stale.write_text("{}\n", encoding="utf-8")
    split_pages.build(FIX, tmp_path)
    assert not stale.exists()
    assert sorted(p.stem for p in pages.glob("*.json")) == sorted(IDS)


def test_files_end_with_a_newline_and_keep_unicode(built):
    _, out = built
    for rel in ("book.json", "index.json", "pages/p004.json"):
        text = (out / rel).read_text(encoding="utf-8")
        assert text.endswith("}\n")
    assert "ſ" in (out / "pages" / "p004.json").read_text(encoding="utf-8")


def test_main_prints_the_counts(tmp_path, capsys):
    rc = split_pages.main(["--root", str(FIX), "--out", str(tmp_path)])
    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines() == [
        "4 pages written (3 french, 3 english)", "3 sections written"]


# --- reader mode: text/<section>.json and text/toc.json --------------------

GOLDEN = FIX.parent / "site_golden"
PG = '<span class="pg" data-page="{}"></span>'


def section(out, section_id):
    return read(out / "text" / f"{section_id}.json")


def test_page_files_are_byte_identical_to_before_reader_mode(built):
    """site_golden/ is the fixture build from before reader mode existed."""
    _, out = built
    for golden in sorted((GOLDEN / "pages").glob("*.json")):
        assert (out / "pages" / golden.name).read_bytes() == golden.read_bytes(), golden.name
    assert (out / "index.json").read_bytes() == (GOLDEN / "index.json").read_bytes()
    assert sorted(p.name for p in (out / "pages").glob("*.json")) == sorted(
        p.name for p in (GOLDEN / "pages").glob("*.json"))


def test_toc_lists_sections_in_order_with_kinds(built):
    counts, out = built
    toc = read(out / "text" / "toc.json")["sections"]
    # annot-002 is in sections.json but untranslated: skipped, its place kept.
    assert [(s["id"], s["order"], s["kind"], s["heading"]) for s in toc] == [
        ("annot-001", 1, "annotation", "ANNOTATION I"),
        ("texte-02", 2, "text", "TEXT"),
        ("texte-03", 4, "text", "TEXT"),
    ]
    assert toc[0]["first_page"] == "p004" and toc[0]["pages"] == ["p004", "p005"]
    assert toc[2]["first_page"] == "p041"
    assert sorted(p.stem for p in (out / "text").glob("*.json")) == [
        "annot-001", "texte-02", "texte-03", "toc"]


def test_section_paragraphs_are_whole_with_the_page_turn_inside(built):
    _, out = built
    rec = section(out, "annot-001")
    assert rec["id"] == "annot-001" and rec["kind"] == "annotation"
    assert rec["pages"] == ["p004", "p005"]
    first, second = rec["blocks"]
    # The first marker of a section opens its first paragraph.
    assert first["html"].startswith(PG.format("p004") + "Marriages thus")
    # The second paragraph crosses from p004 to p005 mid-word and stays one block.
    assert "to the wo" + PG.format("p005") + "man of the next" in second["html"]
    assert second["html"].count('class="pg"') == 1
    assert "continued" not in second


def test_section_markers_and_notes_carry_their_page(built, capsys):
    _, out = built
    first, second = section(out, "annot-001")["blocks"]
    assert '<sup class="mk" data-key="a" data-page="p004">a</sup>' in first["html"]
    assert '<sup class="mk" data-key="b" data-page="p004">b</sup>' in first["html"]
    # {a} again, after the turn: the p005 note, not the p004 one.
    assert '<sup class="mk" data-key="a" data-page="p005">a</sup>' in second["html"]
    assert [(n["key"], n["page"]) for n in first["notes"]] == [("a", "p004"), ("b", "p004")]
    assert first["notes"][0]["gloss"] == "On the age of consent."
    assert first["notes"][0]["citation"].startswith("Digest 23.2 (<i>De ritu nuptiarum</i>)")
    # {c} (p005) has no marker: it sits with the first paragraph that reaches p005.
    assert [(n["key"], n["page"]) for n in second["notes"]] == [("a", "p005"), ("c", "p005")]


def test_a_heading_paragraph_is_the_section_heading_not_a_block(built):
    """texte-03 opens `⟦p041⟧TEXT`: the heading is dropped, its marker kept."""
    _, out = built
    rec = section(out, "texte-03")
    assert rec["heading"] == "TEXT"
    assert len(rec["blocks"]) == 1
    assert rec["blocks"][0]["html"].startswith(PG.format("p041") + "The said du Tilh")
    assert "<i>Parlement</i>" in rec["blocks"][0]["html"]
    assert [(n["key"], n["page"]) for n in rec["blocks"][0]["notes"]] == [("a", "p041")]


def test_section_notes_scope_by_page_within_a_paragraph(book, tmp_path):
    body = ("---\nid: annot-001\npages: [p004, p005]\n---\n"
            "⟦p004⟧One {a} runs on to ⟦p005⟧the next {a} and *ends* here.\n\n"
            "## Notes\n"
            "- {a} (p004): **First** — premier\n"
            "- {a} (p005): **Second** — second\n"
            "- {_} (p005) — unkeyed: **Loose** — libre\n")
    one_section(book, body, pages=("p004", "p005"))
    out = tmp_path / "out"
    split_pages.build(book, out)
    (block,) = section(out, "annot-001")["blocks"]
    assert [(n["key"], n["page"], n["citation"]) for n in block["notes"]] == [
        ("a", "p004", "First"), ("a", "p005", "Second"), (None, "p005", "Loose")]
    assert block["html"] == (
        PG.format("p004") + 'One <sup class="mk" data-key="a" data-page="p004">a</sup> '
        "runs on to " + PG.format("p005") + 'the next <sup class="mk" data-key="a" '
        'data-page="p005">a</sup> and <i>ends</i> here.')


def test_to_html_is_unchanged_for_the_page_files():
    assert split_pages.to_html("one {a}  *two*\n& three") == (
        'one <sup class="mk" data-key="a">a</sup> <i>two</i> &amp; three')
    slot = split_pages.PAGE_SLOT
    # An italic run may span the page turn; the span sits inside it.
    assert split_pages.to_html(f"*in {slot}two*", "p004", ["p005"]) == (
        '<i>in <span class="pg" data-page="p005"></span>two</i>')


def test_reader_kind_from_record_or_id():
    sec = split_pages.Section(id="x", pages=[], pieces=[], notes=[], path=None)
    kind = split_pages.reader_kind
    assert kind(sec._replace(kind="texte")) == "text"
    assert kind(sec._replace(kind="annotation")) == "annotation"
    assert kind(sec._replace(kind="title")) == "title"
    assert kind(sec._replace(kind="argument")) == "argument"
    assert kind(sec._replace(id="annot-007", kind=None)) == "annotation"
    assert kind(sec._replace(id="texte-07", kind=None)) == "text"


def test_book_json_names_the_reader_base(built):
    _, out = built
    assert read(out / "book.json")["reader"] == {"base": "text/"}


def test_no_translated_sections_means_no_reader(book, tmp_path):
    shutil.rmtree(sections_dir(book))
    sections_dir(book).mkdir()
    out = tmp_path / "out"
    assert split_pages.build_all(book, out) == (4, 3, 0, 0)
    assert "reader" not in read(out / "book.json")
    assert not (out / "text").exists()


def test_stale_section_file_removed(tmp_path):
    text = tmp_path / "text"
    text.mkdir(parents=True)
    stale = text / "annot-999.json"
    stale.write_text("{}\n", encoding="utf-8")
    split_pages.build(FIX, tmp_path)
    assert not stale.exists()
    assert (text / "toc.json").exists()


def test_section_files_validate_and_the_schema_rejects_a_bad_one(built):
    _, out = built
    schema = read(split_pages.SITE_SCHEMA_PATH)
    validator = split_pages._def_validator(schema, "section")
    for path in (out / "text").glob("*.json"):
        if path.name != "toc.json":
            validator.validate(read(path))
    split_pages._def_validator(schema, "toc").validate(read(out / "text" / "toc.json"))
    bad = section(out, "annot-001")
    del bad["blocks"][0]["notes"][0]["page"]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad)
    bad = section(out, "annot-001")
    bad["kind"] = "texte"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad)


def test_section_files_end_with_a_newline_and_keep_unicode(built):
    _, out = built
    text = (out / "text" / "annot-001.json").read_text(encoding="utf-8")
    assert text.endswith("}\n") and "ſ" in text and '\n "id"' in text


# --- contested readings ---------------------------------------------------

def _final(lines, decisions=(), uncertain=(), margin=None, heading=None):
    blocks = []
    if heading is not None:
        blocks.append({"type": "heading", "text": heading})
    blocks.append({"type": "paragraph", "lines": list(lines)})
    return {"blocks": blocks, "margin_notes": margin or [], "foot_notes": [],
            "decisions": list(decisions), "uncertain": list(uncertain)}


def _cut(line, spans):
    return [line[s:e] for s, e in spans]


def test_word_spans_marks_only_the_differing_words():
    line = "tu, que menaces, ou force {f}. Ioinct"
    spans = split_pages.word_spans(line, ["tu; que menaces, ou force {f}. Ioinct"])
    assert _cut(line, spans) == ["tu,"]
    # adjacent differing words share one span; a split word marks both halves
    assert _cut("me en vne de", split_pages.word_spans("me en vne de", ["meen vne de"])) == ["me en"]
    # a word only the alternative has marks the word before it
    assert _cut("a b d", split_pages.word_spans("a b d", ["a b c d"])) == ["b"]


def test_word_spans_reads_an_ellipsis_as_the_rest_of_the_line():
    line = "Ce qu'elle impetra & entre les bras de ceſte ombre rea"
    assert _cut(line, split_pages.word_spans(line, ["…ombre ren"])) == ["rea"]
    line = "que Platon, ni Ariſtote: à ſçauoir,"
    assert _cut(line, split_pages.word_spans(line, ["…, niAriſtote: à ſçauoir,"])) == ["ni Ariſtote:"]


def test_word_spans_gives_up_without_an_anchor():
    assert split_pages.word_spans("un deux trois quatre cinq", ["six sept huit neuf dix"]) is None
    assert split_pages.word_spans("", ["x"]) is None
    assert split_pages.word_spans("same words", ["same  words"]) is None


def test_word_spans_counts_utf16_like_the_viewer():
    line = "𝔄 mot"   # one astral character: two UTF-16 units
    assert split_pages.word_spans(line, ["𝔄 mots"]) == [[3, 6]]


def test_decision_becomes_a_reading_on_its_line():
    final = _final(["tu, que menaces, ou force {f}. Ioinct"], decisions=[{
        "where": "blocks[0].lines[0]", "A": "tu, que menaces, ou force {f}. Ioinct",
        "B": "tu; que menaces, ou force {f}. Ioinct", "chose": "A",
        "text": "tu, que menaces, ou force {f}. Ioinct", "reason": "comma at 4x"}])
    [r] = split_pages.contested_readings(final)
    assert r == {"where": "blocks[0].lines[0]", "target": "blocks[0].lines[0]",
                 "spans": [[0, 3]], "aligned": True,
                 "a": "tu, que menaces, ou force {f}. Ioinct",
                 "b": "tu; que menaces, ou force {f}. Ioinct",
                 "text": "tu, que menaces, ou force {f}. Ioinct", "chose": "A",
                 "by": "reconciler", "status": "decided", "reason": "comma at 4x"}


def test_by_defaults_to_reconciler_and_passes_translator_and_auto():
    line = "Loy i. ſur"
    base = {"where": "blocks[0].lines[0]", "A": "Loy 1. ſur", "B": "Loy i. ſur",
            "chose": "B", "text": line, "reason": "dot"}
    final = _final([line], decisions=[
        base, dict(base, by="translator"), dict(base, by="auto"), dict(base, by="nobody")])
    rs = split_pages.contested_readings(final)
    assert [r["by"] for r in rs] == ["reconciler", "translator", "auto", "reconciler"]
    assert [r["status"] for r in rs] == ["decided", "decided", "open", "decided"]
    # an auto-deferred reading is open: no choice, and the text shows reader A
    assert rs[2]["chose"] is None and rs[2]["text"] == "Loy 1. ſur"
    assert _cut(line, rs[0]["spans"]) == ["i."]


def test_unmarked_first_run_choices_are_the_reconcilers_and_sessions_carsons():
    """No `by`: A/B/neither came from the first run's reconciler; carson-session
    entries, and any other unmarked choice, are Carson's. An explicit `by` wins."""
    line = "Loy i. ſur"
    base = {"where": "blocks[0].lines[0]", "A": "Loy 1. ſur", "B": "Loy i. ſur",
            "text": line}
    final = _final([line], decisions=[
        dict(base, chose="A", text="Loy 1. ſur"), dict(base, chose="B"),
        dict(base, chose="neither", text="Loy j. ſur"),
        dict(base, chose="carson-session", reason="arbitration: B"),
        dict(base, chose="carson-session", reason="spot-check: B at 4x"),
        dict(base, chose="B", by="carson")])
    rs = split_pages.contested_readings(final)
    assert [r["by"] for r in rs] == ["reconciler", "reconciler", "reconciler",
                                     "carson", "carson", "carson"]
    assert all(r["status"] == "decided" for r in rs)
    schema = read(ROOT / "scripts" / "site_schema.json")
    reading = dict(schema["$defs"]["reading"], **{"$defs": schema["$defs"]})
    for r in rs:
        jsonschema.validate(r, reading)


def test_session_reason_gives_the_real_choice_and_drops_the_provenance():
    line = "aliena. P. fin."
    d = {"where": "blocks[0].lines[0]", "A": "aliena. P. fin.", "B": "alienæ. P. fin.",
         "chose": "carson-session", "text": line}
    final = _final([line], decisions=[
        dict(d, reason="arbitration: A"),
        dict(d, by="translator", reason="arbitration: A (translator: fits the sense)"),
        dict(d, by="auto", reason="arbitration: either (auto-deferred)"),
        dict(d, reason="arbitration: unknown")])
    rs = split_pages.contested_readings(final)
    assert [(r["chose"], r["status"], r["reason"]) for r in rs] == [
        ("A", "decided", None), ("A", "decided", "fits the sense"),
        (None, "open", None), (None, "open", None)]


def test_reviewer_passes_through_and_validates():
    line = "aliena. P. fin."
    d = {"where": "blocks[0].lines[0]", "A": "aliena. P. fin.", "B": "alienæ. P. fin.",
         "chose": "carson-session", "text": line}
    final = _final([line], decisions=[
        dict(d, by="reviewer", reason="arbitration: A (reviewer: same form at annot-040)"),
        dict(d, chose="A", by="reviewer", reason="plain")])
    rs = split_pages.contested_readings(final)
    assert [(r["by"], r["chose"], r["status"], r["reason"]) for r in rs] == [
        ("reviewer", "A", "decided", "same form at annot-040"),
        ("reviewer", "A", "decided", "plain")]
    schema = read(ROOT / "scripts" / "site_schema.json")
    reading = dict(schema["$defs"]["reading"], **{"$defs": schema["$defs"]})
    for r in rs:
        jsonschema.validate(r, reading)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(rs[0], by="robot"), reading)


def test_agreed_entries_are_not_contested():
    final = _final(["pour quoy les enfans"], decisions=[
        {"where": "blocks[0].lines[0]", "A": "pour quoy", "B": "pour quoy",
         "chose": "A", "text": "pour quoy les enfans", "reason": "agreed; checked"}])
    assert split_pages.contested_readings(final) == []


def test_structural_readings_have_no_target():
    final = _final(["x"], decisions=[
        {"where": "page", "A": "", "B": "", "chose": "B", "text": "", "reason": "ornaments"},
        {"where": "running_head", "A": "ARREST DV", "B": "ARREST DV.", "chose": "A",
         "text": "ARREST DV", "reason": ""},
        {"where": "margin_notes[*].beside_line", "A": None, "B": "(present)", "chose": "B"},
        {"where": "blocks[9].lines[0]", "A": "a", "B": "b", "chose": "A", "text": "a"}])
    rs = split_pages.contested_readings(final)
    assert [(r["target"], r["spans"], r["aligned"]) for r in rs] == [(None, None, False)] * 4
    assert rs[1]["reason"] is None          # an empty reason is no reason
    assert rs[2]["a"] == "" and rs[2]["text"] == "(present)"


def test_unalignable_line_falls_back_to_the_whole_line():
    line = "une ligne refaite depuis la decision"
    final = _final([line], decisions=[
        {"where": "blocks[0].lines[0]", "A": "tout autre texte ici present",
         "B": "encore autre chose ici lu", "chose": "neither", "text": "rien de cela"}])
    [r] = split_pages.contested_readings(final)
    assert r["target"] == "blocks[0].lines[0]"
    assert r["spans"] == [[0, len(line)]] and r["aligned"] is False


def test_a_quoted_excerpt_is_found_in_its_line():
    line = "blemẽt offenſez, auant que ſe doubter de luy? ou toures"
    final = _final([line], decisions=[
        {"where": "blocks[0].lines[0]", "A": "ou toutes", "B": "ou toutes",
         "chose": "neither", "text": "ou toures"}])
    # A == B but the text differs: still contested (the editor kept the print's reading)
    [r] = split_pages.contested_readings(final)
    assert _cut(line, r["spans"]) == ["toures"] and r["aligned"]
    line = "cer l'office de tabellion ou noraire, ſi toutesfois ils"
    final = _final([line], decisions=[
        {"where": "blocks[0].lines[0]", "A": "notaire", "B": "notaire",
         "chose": "neither", "text": "noraire"}])
    assert _cut(line, split_pages.contested_readings(final)[0]["spans"]) == ["noraire,"]


def test_note_lines_and_headings_are_targets():
    final = _final(["x"], heading="TEXTE.", margin=[{"key": "a", "lines": ["l j. P. vſque"]}],
                   decisions=[
                       {"where": "blocks[0].text", "A": "TEXTE.", "B": "TEXTE,", "chose": "A",
                        "text": "TEXTE."},
                       {"where": "margin_notes[0].lines[0]", "A": "l j. P. vſque",
                        "B": "l. j. P. vſque", "chose": "A", "text": "l j. P. vſque"}])
    head, note = split_pages.contested_readings(final)
    assert head["target"] == "blocks[0].text" and _cut("TEXTE.", head["spans"]) == ["TEXTE."]
    assert note["target"] == "margin_notes[0].lines[0]"
    assert _cut("l j. P. vſque", note["spans"]) == ["l"]


def test_open_arbitration_in_uncertain_becomes_a_reading_once():
    line = "Rols le receut, & careſſa commẽ mari: &"
    alt = ("arbitration: unknown; alternatives: Rols le receut, & careſſa commẽ mari: & ||| "
           "Rols le receut, & careſſa comme mari: &")
    other = {"where": "blocks[0].lines[1]", "text": "deux",
             "note": "arbitration: undecided; alternatives: deux ||| dieux", "escalate": False}
    plain = {"where": "blocks[0].lines[0]", "text": line, "note": "tilde or ink spot"}
    final = _final([line, "deux"], uncertain=[
        {"where": "blocks[0].lines[0]", "text": line, "note": alt, "escalate": True},
        other, plain], decisions=[
        {"where": "blocks[0].lines[0]", "A": line, "B": "Rols le receut, & careſſa comme mari: &",
         "chose": "carson-session", "text": line, "reason": "arbitration: unknown"}])
    rs = split_pages.contested_readings(final)
    assert [(r["where"], r["status"], r["by"]) for r in rs] == [
        ("blocks[0].lines[0]", "open", "carson"), ("blocks[0].lines[1]", "open", None)]
    assert _cut(line, rs[0]["spans"]) == ["commẽ"]
    assert rs[1]["a"] == "deux" and rs[1]["b"] == "dieux" and rs[1]["text"] == "deux"
    # the open arbitrations leave uncertain[]; the reader's own doubt stays
    assert split_pages.uncertain(final) == [plain]


def test_pages_carry_readings_and_validate(built):
    _, out = built
    schema = read(ROOT / "scripts" / "site_schema.json")
    for page_id in IDS:
        rec = page(out, page_id)
        assert isinstance(rec["readings"], list)
        jsonschema.validate(rec, schema)
    assert page(out, "p006")["readings"] == []
    assert page(out, "p004")["readings"] == split_pages.contested_readings(
        split_pages.pagelib.nfc_all(final("p004")))


def test_book_json_carries_the_about_paragraph(built):
    _, out = built
    about = read(out / "book.json")["about"]
    assert isinstance(about, list) and about
    assert "translation model" in about[0].lower()

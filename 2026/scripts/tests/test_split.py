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
    assert capsys.readouterr().out.strip() == "4 pages written (3 french, 3 english)"

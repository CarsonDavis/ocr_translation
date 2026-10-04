"""new_book.py scaffolding and bookconf root / config resolution."""
import json, pathlib, sys

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import bookconf  # noqa: E402
import new_book  # noqa: E402
import stitch_text  # noqa: E402

ARGS = ["--title", "Histoire veritable", "--author", "Iean Dupont", "--year", "1580",
        "--edition", "Lyon: Rigaud, 1580"]


@pytest.fixture
def book(tmp_path):
    root = tmp_path / "dupont"
    (root / "raw").mkdir(parents=True)
    for name in ("img010.jpg", "img002.jpg", "notes.txt", "img003.tif"):
        (root / "raw" / name).write_bytes(b"")
    assert new_book.main([str(root), *ARGS]) == 0
    return root


def test_layout_and_files(book):
    for d in new_book.DIRS:
        assert (book / d).is_dir(), d
    for f in ("book.json", "manifest.json", "docs/conventions.md", "docs/case-file.md",
              "docs/pipeline-log.md", "docs/handoff.md", "hyphen_keep.txt", ".gitignore",
              "prompts/read_single.md", "prompts/translate.md", "prompts/review.md"):
        assert (book / f).is_file(), f
    assert (book / "scripts").is_symlink()
    assert (book / "scripts").resolve() == SCRIPTS
    assert (book / "scripts" / "wave.py").is_file()


def test_manifest_lists_raw_images_in_name_order(book):
    m = json.loads((book / "manifest.json").read_text())
    assert [p["id"] for p in m["pages"]] == ["p001", "p002", "p003"]
    assert [p["raw"] for p in m["pages"]] == ["img002.jpg", "img003.tif", "img010.jpg"]
    rec = m["pages"][0]
    # render_prompt.py slims each record to these keys; they must all be present
    assert {"id", "page", "image", "side", "folio", "source"} <= set(rec)
    assert rec["source"] == "other" and set(rec["status"].values()) == {"pending"}
    assert m["edition"] == "Lyon: Rigaud, 1580"


def test_templates_are_filled_and_keep_the_numbered_sections(book):
    conv = (book / "docs/conventions.md").read_text()
    case = (book / "docs/case-file.md").read_text()
    for text in (conv, case):
        assert "{{" not in text and "Histoire veritable" in text and "Iean Dupont" in text
    assert "Coras" not in conv.split("-->", 1)[1]          # facts stripped, past the comment
    assert "Martin Guerre" not in case
    for n in range(1, 13):
        assert f"\n## {n}. " in case, n
    assert "\n## 9. Running glossary\n" in case
    assert case.rstrip().endswith("## 12. Review decision log")


def test_book_json_round_trips_through_bookconf(book):
    cfg = bookconf.load(book)
    assert cfg["title"] == "Histoire veritable" and cfg["year"] == 1580
    assert cfg["hyphen_keep"] == "hyphen_keep.txt"
    assert cfg["site"]["first_page"] == "p000-title"
    assert cfg["headings"]["texte"] == ["TEXTE"]


def test_refuses_an_existing_book(book):
    with pytest.raises(SystemExit):
        new_book.main([str(book), *ARGS])


def test_empty_raw_gives_an_empty_manifest(tmp_path):
    new_book.main([str(tmp_path / "b"), *ARGS])
    assert json.loads((tmp_path / "b/manifest.json").read_text())["pages"] == []


def test_find_root_env_then_cwd_then_toolkit(book, tmp_path):
    assert bookconf.find_root(cwd=book / "docs", env={}) == book.resolve()
    assert bookconf.find_root(cwd=tmp_path, env={"BOOK_ROOT": str(book)}) == book.resolve()
    assert bookconf.find_root(cwd=tmp_path, env={}) == SCRIPTS.parent


def test_book_prompts_override_the_toolkit(book, tmp_path):
    assert bookconf.prompt_path("translate", book) == book / "prompts/translate.md"
    assert bookconf.prompt_path("spotcheck", book) == SCRIPTS / "prompts/spotcheck.md"
    assert bookconf.prompt_path("translate", tmp_path) == SCRIPTS / "prompts/translate.md"
    tpl = bookconf.prompt_path("read_single", book).read_text()
    assert "{BOOK_ROOT}" in tpl
    assert f"under {book}/" in bookconf.fill_root(tpl, book)


def test_stitch_takes_heading_words_from_book_json(tmp_path):
    root = tmp_path / "latin"
    root.mkdir()
    (root / "book.json").write_text(json.dumps({
        "headings": {"texte": ["TEXTVS"], "annotation": ["ANNOTATIO"]},
        "front_matter": {"p000-titulus": "title"}}))
    try:
        stitch_text.configure(bookconf.load(root), root)
        assert stitch_text.parse_heading("TEXTVS.") == ("texte", None)
        assert stitch_text.parse_heading("TEXTVM.") == ("texte", None)     # one wrong sort
        assert stitch_text.parse_heading("ANNOTATIO XII.") == ("annotation", 12)
        assert stitch_text.parse_heading("TEXTE.") is None
        assert stitch_text.SPECIAL == {"p000-titulus": "title"}
        assert stitch_text.KEEP_PATH == root / "hyphen_keep.txt"
    finally:
        stitch_text.configure(bookconf.load(bookconf.ROOT))
    assert stitch_text.parse_heading("ANNOTAT. V.") == ("annotation", 5)

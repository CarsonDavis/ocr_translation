import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import render_review as rr


def sec(sid, kind, pages, text, notes=(), label=None, number=None, complete=True):
    return {"id": sid, "kind": kind, "number": number, "label": label, "pages": list(pages),
            "text": text, "notes": list(notes), "complete": complete}


def book(tmp_path, sections, translations):
    (tmp_path / "text").mkdir()
    (tmp_path / "text/sections.json").write_text(
        json.dumps({"sections": sections}, ensure_ascii=False), encoding="utf-8")
    d = tmp_path / "translation/sections"
    d.mkdir(parents=True)
    for sid, body in translations.items():
        (d / f"{sid}.md").write_text(body, encoding="utf-8")
    return tmp_path


SECTIONS = [
    sec("title", "title", ["p000-title"], "⟦p000-title⟧ARREST"),
    sec("annot-001", "annotation", ["p002", "p003"],
        "⟦p002⟧Les mariages {a} ainſi⟨alt:ainſy⟩ ⟦p003⟧contractez {a}.",
        [{"key": "a", "page": "p002", "text": "Chap. dernier."},
         {"key": "a", "page": "p003", "text": "Ariſtote."},
         {"key": None, "page": "p003", "text": "gloss"}],
        label="ANNOTATION I.", number=1),
    sec("texte-02", "texte", ["p004"], "⟦p004⟧Texte.", number=2),
    sec("texte-03", "texte", ["p005"], "⟦p005⟧Fin.", number=3, complete=False),
]
TRANS = {
    "title": "---\nid: title\npages: [p000-title]\n---\n⟦p000-title⟧DECISION\n",
    "annot-001": "---\nid: annot-001\n---\n⟦p002⟧Marriages {a} so ⟦p003⟧contracted {a}.\n\n"
                 "## Notes\n- {a} (p002): note one\n",
}


def test_render_sections_notes_and_english(tmp_path):
    md, st = rr.render(book(tmp_path, SECTIONS, TRANS))
    assert st["ids"] == ["title", "annot-001", "texte-02"] and st["skipped"] == ["texte-03"]
    assert "Skipped (complete=false, not rendered): texte-03." in md
    assert "texte-03" not in md.split("Skipped")[1].split("\n", 1)[1]
    assert "## annot-001 — annotation ANNOTATION I. — pages p002–p003" in md
    assert "## title — title — pages p000-title–p000-title" in md
    assert "## texte-02 — texte 2 — pages p004–p004" in md
    block = md.split("## annot-001")[1].split("## texte-02")[0]
    assert "### French\n\n⟦p002⟧Les mariages {a} ainſi⟨alt:ainſy⟩ ⟦p003⟧contractez {a}.\n" in block
    assert "### French notes\n\n{a} Chap. dernier.\n{a} Ariſtote.\n{_} gloss\n" in block
    assert block.rstrip().endswith("### English\n\n⟦p002⟧Marriages {a} so ⟦p003⟧contracted {a}.")
    assert "## Notes" not in md and "id: annot-001" not in md
    title = md.split("## title")[1].split("## annot-001")[0]
    assert "### French notes" not in title                     # no notes: no block
    assert md.split("## texte-02")[1].rstrip().endswith("### English\n\n(not translated)")
    assert st["missing"] == ["texte-02"]
    assert st["french_words"] == 1 + 5 + 4 + 1 and st["english_words"] == 1 + 4
    assert "3 sections, 11 French words, 5 English words." in md
    assert st["tokens"] == round(st["french_chars"] / 3.2 + st["english_chars"] / 4)
    assert st["french_chars"] + st["english_chars"] == len(md)


def test_quarters_split_contiguously_by_count():
    ids = [f"s{i}" for i in range(10)]
    assert rr.quarters(ids, 4) == [(1, "s0", "s2", 3), (2, "s3", "s5", 3),
                                   (3, "s6", "s7", 2), (4, "s8", "s9", 2)]
    assert rr.quarters(ids[:2], 4) == [(1, "s0", "s0", 1), (2, "s1", "s1", 1)]


def test_cli_writes_file_prints_table_and_tokens(tmp_path):
    root = book(tmp_path, SECTIONS, TRANS)
    out = []
    assert rr.main(["--out", str(tmp_path / "o/review.md"), "--quarters", "2"],
                   root=root, out=out.append) == 0
    md = (tmp_path / "o/review.md").read_text(encoding="utf-8")
    assert "| 1 | title | annot-001 | 2 |\n| 2 | texte-02 | texte-02 | 1 |" in md
    text = "\n".join(out)
    assert "| 1 | title | annot-001 | 2 |" in text and "tokens" in text and "texte-03" in text


def test_cli_ids_only(tmp_path):
    root = book(tmp_path, SECTIONS, TRANS)
    out = []
    assert rr.main(["--ids"], root=root, out=out.append) == 0
    assert out == ["title", "annot-001", "texte-02"]
    assert not (tmp_path / "review.md").exists()

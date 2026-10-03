# scripts/tests/test_migrate_section_ids.py
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import migrate_section_ids as mig  # noqa: E402


def sec(sid, kind, pages, text, label=None, notes=()):
    return {"id": sid, "kind": kind, "pages": list(pages), "text": text, "label": label,
            "notes": [{"key": k, "page": p, "text": "n"} for k, p in notes]}


# old cut: the misprinted "TEXTB." on p072 was a display line inside annot-050
OLD = [
    sec("texte-48", "texte", ["p071"], "⟦p071⟧t48 {m}", "TEXTE."),
    sec("annot-050", "annotation", ["p071", "p072"],
        "⟦p071⟧a50 {a} inno⟦p072⟧cent {b}\n\nTEXTB.\n\nb51 {c}", "ANNOTAT. L.",
        notes=[("a", "p071"), ("b", "p072"), ("c", "p072"), ("z", "p072")]),
    sec("annot-051", "annotation", ["p072"], "⟦p072⟧a51", "ANNOT. LI."),
    sec("texte-49", "texte", ["p073"], "⟦p073⟧t49", "TEXTE."),
]
NEW = [
    sec("texte-48", "texte", ["p071"], "⟦p071⟧t48 {m}", "TEXTE."),
    sec("annot-050", "annotation", ["p071", "p072"], "⟦p071⟧a50 {a} inno⟦p072⟧cent {b}",
        "ANNOTAT. L.", notes=[("a", "p071"), ("b", "p072")]),
    sec("texte-49", "texte", ["p072"], "⟦p072⟧b51 {c}", "TEXTB.",
        notes=[("c", "p072"), ("z", "p072")]),
    sec("annot-051", "annotation", ["p072"], "⟦p072⟧a51", "ANNOT. LI."),
    sec("texte-50", "texte", ["p073"], "⟦p073⟧t49", "TEXTE."),
]

ANNOT_050_MD = """---
id: annot-050
pages: [p071, p072]
---
⟦p071⟧As for the marriage {a}, the inno⟦p072⟧cent {b}.

TEXT

Following which opinion {c}, it seems so.

## Notes
Commentary: the TEXTB. block is translated here as TEXT.

- {a} (p071): first note
- {b} (p072): second note
- {c} (p072): third note
- {z} (p072): orphan, no marker in the body
- {q} (p072): stray entry nobody owns
"""


def md(sid, pages, body):
    return f"---\nid: {sid}\npages: [{', '.join(pages)}]\n---\n{body}\n\n## Notes\n- x\n"


def project(tmp_path):
    root = tmp_path
    s = root / "translation/sections"
    s.mkdir(parents=True)
    (s / "texte-48.md").write_text(md("texte-48", ["p071"], "⟦p071⟧t48 {m}"))
    (s / "annot-050.md").write_text(ANNOT_050_MD)
    (s / "annot-051.md").write_text(md("annot-051", ["p072"], "⟦p072⟧a51"))
    (s / "texte-49.md").write_text(md("texte-49", ["p073"], "⟦p073⟧t49"))
    r = root / "translation/reports"
    r.mkdir()
    (r / "texte-49.md").write_text("report 49")
    (r / "batch-texte-48--texte-49.md").write_text("batch report")
    a = root / "translation/alt-choices"
    a.mkdir()
    (a / "batch-texte-48--texte-49.json").write_text(
        json.dumps([{"alt_id": "p073-b0l1-1", "choice": "A", "reason": "r"}]))
    return root


def test_mapping_finds_renames_and_the_split():
    m = mig.compute_mapping(OLD, NEW)
    assert [(r["old"], r["new"]) for r in m] == [
        ("texte-48", ["texte-48"]), ("annot-050", ["annot-050", "texte-49"]),
        ("annot-051", ["annot-051"]), ("texte-49", ["texte-50"])]


def test_mapping_refuses_texts_that_do_not_align():
    bad = [dict(s) for s in NEW]
    bad[3] = dict(bad[3], text="⟦p072⟧changed")
    with pytest.raises(mig.MigrationError):
        mig.compute_mapping(OLD, bad)


def test_rename_name_maps_range_ends():
    first = {"texte-48": "texte-48", "texte-49": "texte-50", "annot-050": "annot-050"}
    last = {"texte-48": "texte-48", "texte-49": "texte-50", "annot-050": "texte-49"}
    assert mig.rename_name("batch-annot-050--texte-49.md", first, last) == \
        "batch-annot-050--texte-50.md"
    assert mig.rename_name("batch-texte-48--annot-050.json", first, last) == \
        "batch-texte-48--texte-49.json"
    assert mig.rename_name("texte-49.md", first, last) == "texte-50.md"
    assert mig.rename_name("title.md", first, last) == "title.md"


def test_dry_run_plan_touches_nothing(tmp_path):
    root = project(tmp_path)
    before = {p: p.read_text() for p in root.rglob("*") if p.is_file()}
    ops, _ = mig.build_plan(root, mig.compute_mapping(OLD, NEW), NEW)
    assert {(o["op"], o["dir"].split("/")[-1], o["src"]) for o in ops} == {
        ("split", "sections", "annot-050.md"), ("rename", "sections", "texte-49.md"),
        ("rename", "reports", "texte-49.md"),
        ("rename", "reports", "batch-texte-48--texte-49.md"),
        ("rename", "alt-choices", "batch-texte-48--texte-49.json")}
    assert before == {p: p.read_text() for p in root.rglob("*") if p.is_file()}


def test_apply_splits_renames_and_keeps_every_word(tmp_path):
    root = project(tmp_path)
    mapping = mig.compute_mapping(OLD, NEW)
    ops, _ = mig.build_plan(root, mapping, NEW)
    mig.apply_plan(root, ops)
    s = root / "translation/sections"
    assert sorted(p.name for p in s.iterdir()) == \
        ["annot-050.md", "annot-051.md", "texte-48.md", "texte-49.md", "texte-50.md"]
    # the shifted file: id rewritten, content kept
    t50 = (s / "texte-50.md").read_text()
    assert t50.startswith("---\nid: texte-50\npages: [p073]\n---\n⟦p073⟧t49")
    # the split
    a50 = (s / "annot-050.md").read_text()
    t49 = (s / "texte-49.md").read_text()
    assert "id: annot-050\npages: [p071, p072]" in a50
    assert "id: texte-49\npages: [p072]" in t49
    head1, _, notes1 = a50.partition("## Notes")
    head2, _, notes2 = t49.partition("## Notes")
    assert "As for the marriage {a}, the inno⟦p072⟧cent {b}." in head1
    assert "TEXT\n" not in head1 and "Following" not in head1
    assert "⟦p072⟧Following which opinion {c}, it seems so." in head2   # marker prepended
    assert "Commentary:" in notes1
    for k in "abq":                     # {q} cannot be attributed: stays with the first part
        assert f"- {{{k}}}" in notes1 and f"- {{{k}}}" not in notes2
    for k in "cz":                      # {c} by its marker, {z} by the French notes
        assert f"- {{{k}}}" in notes2 and f"- {{{k}}}" not in notes1
    r = root / "translation/reports"
    assert sorted(p.name for p in r.iterdir()) == ["batch-texte-48--texte-50.md", "texte-50.md"]
    assert (r / "texte-50.md").read_text() == "report 49"
    a = root / "translation/alt-choices"
    assert [p.name for p in a.iterdir()] == ["batch-texte-48--texte-50.json"]
    assert not any(p.name == ".migrate-tmp" for p in root.rglob("*"))


def test_unattributable_note_is_reported(tmp_path):
    root = project(tmp_path)
    ops, _ = mig.build_plan(root, mig.compute_mapping(OLD, NEW), NEW)
    report = next(o["report"] for o in ops if o["op"] == "split")
    assert any("{q}" in line and "kept with annot-050" in line for line in report)


def test_split_without_a_heading_line_is_refused(tmp_path):
    root = project(tmp_path)
    p = root / "translation/sections/annot-050.md"
    p.write_text(ANNOT_050_MD.replace("\nTEXT\n", "\n"))
    with pytest.raises(mig.MigrationError):
        mig.build_plan(root, mig.compute_mapping(OLD, NEW), NEW)


def test_target_owned_by_an_unmoved_file_is_refused(tmp_path):
    root = project(tmp_path)
    (root / "translation/reports/texte-50.md").write_text("someone else's")
    with pytest.raises(mig.MigrationError):
        mig.build_plan(root, mig.compute_mapping(OLD, NEW), NEW)


def test_alt_choice_section_field_is_rewritten_only_when_present(tmp_path):
    root = project(tmp_path)
    a = root / "translation/alt-choices/batch-texte-48--texte-49.json"
    assert mig.rewrite_alt_choices(a, []) is None
    a.write_text(json.dumps([{"alt_id": "p073-b0l1-1", "section": "texte-49", "choice": "A"}]))
    out = mig.rewrite_alt_choices(a, [{"alt_id": "p073-b0l1-1", "section": "texte-50"}])
    assert json.loads(out)[0]["section"] == "texte-50"


def test_main_defaults_to_dry_run(tmp_path, capsys):
    root = project(tmp_path)
    (root / "text").mkdir()
    (root / "text/sections.json").write_text(json.dumps({"sections": OLD}))
    new = tmp_path / "recut.json"
    new.write_text(json.dumps({"sections": NEW}))
    assert mig.main(["--root", str(root), "--new", str(new)]) == 0
    out = capsys.readouterr().out
    assert "SPLIT annot-050 -> annot-050 + texte-49" in out and "dry run" in out
    assert (root / "translation/sections/annot-050.md").read_text() == ANNOT_050_MD

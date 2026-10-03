# scripts/tests/test_translator_feedback.py
"""The translator feedback loop: text/alts.json from stitch_text.py, the alt table in
render_translate.py, and apply_translator_choices.py (subprocess faked, temp root)."""
import importlib.util
import json
import pathlib
import sys

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import stitch_text as st  # noqa: E402


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


render_translate = load("coras_render_translate_fb", "render_translate.py")
atc = load("coras_apply_translator_choices", "apply_translator_choices.py")

EITHER, UNKNOWN, SEP = st.ALT_PREFIX, st.ALT_PREFIX_UNKNOWN, st.ALT_SEP


def para(*lines):
    return {"type": "paragraph", "lines": list(lines)}


def unc(where, a, b, prefix=EITHER):
    return {"where": where, "text": a, "note": prefix + a + SEP + b}


def pages():
    """Two pages: a texte section across both, an annotation starting on p002."""
    p1 = {"id": "p001",
          "blocks": [{"type": "heading", "text": "TEXTE."},
                     para("x trabir y {a}", "a b c", "unchanged"),
                     para("z cõtrainte y w")],
          "margin_notes": [{"key": "a", "lines": ["l. famoſi."]}],
          "foot_notes": [],
          "uncertain": [
              unc("blocks[1].lines[0]", "x trabir y {a}", "x trahir y {a}"),
              unc("blocks[1].lines[1]", "a b c", "a c", UNKNOWN),
              unc("blocks[2].lines[0]", "z cõtrainte y w", "z cõrrainte y v"),   # two markers
              unc("margin_notes[0].lines[0]", "l. famoſi.", "l. famoſa."),
              unc("blocks[1].lines[2]", "not this text", "other"),            # unmatched
              {"where": "margin_notes[3]", "note": UNKNOWN + "[c] x" + SEP + "(no line)"},
              {"where": "blocks[1].lines[2]", "note": "sic"},
          ]}
    p2 = {"id": "p002",
          "blocks": [{"type": "heading", "text": "ANNOTAT. I."},
                     para("a b c")],                          # same marker string as p001
          "margin_notes": [], "foot_notes": [],
          "uncertain": [unc("blocks[1].lines[0]", "a b c", "a c")]}
    return [("p001", p1), ("p002", p2)]


# --- stitch: text/alts.json ----------------------------------------------------------

def test_alts_records_ids_markers_and_sections():
    alts = []
    secs = st.stitch(pages(), alts=alts)
    by = {r["alt_id"]: r for r in alts}
    assert list(by) == ["p001-b1l0-1", "p001-b1l1-1", "p001-b2l0-1", "p001-b2l0-2",
                        "p001-m0l0-1", "p002-b1l0-1"]
    assert by["p001-b1l0-1"]["marker"] == "trabir⟨alt:trahir⟩"
    assert by["p001-b1l1-1"]["marker"] == "b⟨alt?:⟩" and by["p001-b1l1-1"]["kind"] == "alt?"
    assert by["p001-b2l0-2"]["marker"] == "w⟨alt:v⟩"
    assert by["p001-m0l0-1"]["marker"] == "famoſi.⟨alt:famoſa.⟩"
    assert by["p002-b1l0-1"]["marker"] == "b⟨alt:⟩" and by["p002-b1l0-1"]["section"] == "annot-001"
    r = by["p001-b1l0-1"]
    assert (r["section"], r["page"], r["where"], r["a"], r["b"]) == \
        ("texte-01", "p001", "blocks[1].lines[0]", "x trabir y {a}", "x trahir y {a}")
    assert "x trabir⟨alt:trahir⟩ y" in r["context"]
    text = {s["id"]: s for s in secs}
    for rec in alts:                          # every marker is verbatim in its section
        sec = text[rec["section"]]
        hay = sec["text"] if rec["where"].startswith("blocks") else \
            " ".join(n["text"] for n in sec["notes"])
        assert rec["marker"] in hay


def test_alts_exclude_structural_and_unmatched_items():
    alts = []
    st.stitch(pages(), alts=alts)
    wheres = {(r["page"], r["where"]) for r in alts}
    assert ("p001", "margin_notes[3]") not in wheres
    assert ("p001", "blocks[1].lines[2]") not in wheres


def test_alt_ids_are_stable_across_runs():
    one, two = [], []
    st.stitch(pages(), alts=one)
    st.stitch(pages(), alts=two)
    assert one == two


def test_alt_markup_unchanged_by_chunking():
    assert st.alt_markup("z cõtrainte y w", "z cõrrainte y v") == "z cõtrainte⟨alt:cõrrainte⟩ y w⟨alt:v⟩"
    assert st.where_slug("foot_notes[1].lines[12]") == "f1l12"


def test_sections_unchanged_and_alts_file_written(tmp_path):
    fin = tmp_path / "transcription/final"
    fin.mkdir(parents=True)
    for pid, pg in pages():
        (fin / f"{pid}.json").write_text(json.dumps(pg, ensure_ascii=False))
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"pages": [{"id": p, "status": {"final": "done"}} for p, _ in pages()]}))
    assert st.main(["--root", str(tmp_path)]) == 0
    sec_file = json.loads((tmp_path / "text/sections.json").read_text())
    assert sec_file["sections"] == st.stitch(pages(), st.load_keep())      # same as without alts
    alts_file = json.loads((tmp_path / "text/alts.json").read_text())
    assert len(alts_file["alts"]) == 6 and alts_file["generated"] == sec_file["generated"]


# --- render_translate: the alt table -------------------------------------------------

SECS = [{"id": "s0", "complete": True, "text": "x trabir⟨alt:trahir⟩ y", "notes": []},
        {"id": "s1", "complete": True, "text": "plain", "notes": []}]
ALTS = [{"alt_id": "p001-b1l0-1", "section": "s0", "page": "p001", "where": "blocks[1].lines[0]",
         "kind": "alt", "a": "x trabir y", "b": "x trahir | y", "marker": "trabir⟨alt:trahir⟩",
         "context": "x trabir⟨alt:trahir⟩ y"}]


def test_render_embeds_alt_table_and_choices_path(tmp_path):
    tpl = (SCRIPTS / "prompts/translate.md").read_text()
    b = render_translate.render_batch(tpl, SECS, "s0", 2, root=tmp_path, alts=ALTS)
    assert "| alt_id | section | marker | a | b | context |" in b
    assert "| p001-b1l0-1 | s0 | `trabir⟨alt:trahir⟩` | `x trabir y` | `x trahir \\| y` |" in b
    assert "translation/alt-choices/batch-s0--s1.json" in b
    assert "{ALT_" not in b and "## ⟨alt⟩ choices" in b
    s = render_translate.render_single(tpl, SECS, "s1", 2, root=tmp_path, alts=ALTS)
    assert "translation/alt-choices/s1.json" in s and "**none**" in s and "`[]`" in s
    assert "p001-b1l0-1" not in s and "{ALT_" not in s


def test_render_reads_alts_json_from_root(tmp_path):
    (tmp_path / "text").mkdir()
    (tmp_path / "text/alts.json").write_text(json.dumps({"alts": ALTS}, ensure_ascii=False))
    tpl = (SCRIPTS / "prompts/translate.md").read_text()
    b = render_translate.render_batch(tpl, SECS, "s0", 1, root=tmp_path)
    assert "p001-b1l0-1" in b


# --- apply_translator_choices --------------------------------------------------------

def feedback_root(tmp_path, decisions):
    alts = [
        {"alt_id": "p001-b1l0-1", "section": "s0", "page": "p001", "where": "blocks[1].lines[0]",
         "kind": "alt", "a": "x trabir y", "b": "x trahir y", "marker": "m", "context": "c"},
        {"alt_id": "p001-b2l0-1", "section": "s0", "page": "p001", "where": "blocks[2].lines[0]",
         "kind": "alt", "a": "z cõtrainte w", "b": "z cõrrainte v", "marker": "m", "context": "c"},
        {"alt_id": "p001-b2l0-2", "section": "s0", "page": "p001", "where": "blocks[2].lines[0]",
         "kind": "alt", "a": "z cõtrainte w", "b": "z cõrrainte v", "marker": "m", "context": "c"},
        {"alt_id": "p001-m0l0-1", "section": "s0", "page": "p001", "where": "margin_notes[0].lines[0]",
         "kind": "alt", "a": "l. famoſi.", "b": "l. famoſa.", "marker": "m", "context": "c"},
        {"alt_id": "p002-b1l0-1", "section": "s1", "page": "p002", "where": "blocks[1].lines[0]",
         "kind": "alt", "a": "a b c", "b": "a c", "marker": "m", "context": "c"},
    ]
    (tmp_path / "text").mkdir(parents=True)
    (tmp_path / "text/alts.json").write_text(json.dumps({"alts": alts}, ensure_ascii=False))
    q = tmp_path / "transcription/arbitration/queue"
    q.mkdir(parents=True)
    (q / "p001.json").write_text(json.dumps({"page": "p001", "items": [
        {"id": "b-001", "kind": "body", "where_a": "blocks[1].lines[0]", "where_b": "blocks[1].lines[0]",
         "a": "x trabir y", "b": "x trahir y"},
        {"id": "b-002", "kind": "body", "where_a": "blocks[2].lines[0]", "where_b": "blocks[2].lines[0]",
         "a": "z cõtrainte w", "b": "z cõrrainte v"},
        {"id": "n-a-0", "kind": "note", "where_a": "margin_notes[0].lines[0]",
         "where_b": "margin_notes[0].lines[0]", "a": "l. famoſi.", "b": "l. famoſa."},
    ]}, ensure_ascii=False))
    (q / "p002.json").write_text(json.dumps({"page": "p002", "items": [
        {"id": "b-001", "kind": "body", "where_a": "blocks[1].lines[0]", "where_b": "blocks[1].lines[0]",
         "a": "a b c", "b": "a c"}]}, ensure_ascii=False))
    d = tmp_path / "transcription/arbitration/decisions"
    d.mkdir(parents=True)
    for pid, dec in decisions.items():
        (d / f"{pid}.json").write_text(json.dumps({"page": pid, "decisions": dec}))
    return tmp_path


AUTO = {"choice": "either", "by": "auto", "at": "t0", "reason": "auto-deferred to translator"}


def choices_file(tmp_path, entries, name="batch-s0--s1.json"):
    p = tmp_path / "translation/alt-choices" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, ensure_ascii=False))
    return p


def decisions(root, pid):
    return json.loads((root / f"transcription/arbitration/decisions/{pid}.json").read_text())["decisions"]


@pytest.fixture
def calls(monkeypatch):
    seen = []

    class R:
        returncode = 0

    monkeypatch.setattr(atc, "run", lambda args, **kw: seen.append([str(a) for a in args]) or R())
    return seen


def test_choices_map_to_queue_items_and_refinalize(tmp_path, calls):
    root = feedback_root(tmp_path, {"p001": {"b-001": AUTO, "b-002": AUTO, "n-a-0": AUTO},
                                    "p002": {"b-001": AUTO}})
    f = choices_file(tmp_path, [
        {"alt_id": "p001-b1l0-1", "choice": "B", "reason": "trahir = betray"},
        {"alt_id": "p001-b2l0-1", "choice": "A", "reason": "contrainte"},
        {"alt_id": "p001-b2l0-2", "choice": "either", "reason": "same"},
        {"alt_id": "p001-m0l0-1", "choice": "either", "reason": "same sense"},
        {"alt_id": "p002-b1l0-1", "choice": "either", "reason": "same"}])
    out = []
    assert atc.apply([f], root=root, now="t1", out=out.append) == 0
    d1 = decisions(root, "p001")
    assert d1["b-001"] == {"choice": "B", "by": "translator", "at": "t1", "reason": "trahir = betray"}
    assert d1["b-002"]["choice"] == "A" and d1["b-002"]["by"] == "translator"
    assert d1["n-a-0"] == AUTO                                 # either: auto kept
    assert decisions(root, "p002")["b-001"] == AUTO
    assert len(calls) == 1
    assert calls[0][-2:] == ["refinalize", "p001"] and calls[0][-3].endswith("wave.py")
    assert any(line.startswith("p001\t2 applied") for line in out)


def test_carson_decisions_are_never_overwritten(tmp_path, calls):
    carson = {"choice": "either", "at": "t0"}                  # no `by` = carson
    root = feedback_root(tmp_path, {"p001": {"b-001": carson,
                                             "b-002": {"choice": "A", "by": "carson", "at": "t0"},
                                             "n-a-0": {"choice": "B", "by": "translator", "at": "t0",
                                                       "reason": "old"}}})
    f = choices_file(tmp_path, [
        {"alt_id": "p001-b1l0-1", "choice": "B", "reason": "r"},
        {"alt_id": "p001-b2l0-1", "choice": "B", "reason": "r"},
        {"alt_id": "p001-m0l0-1", "choice": "A", "reason": "new"}])
    out = []
    assert atc.apply([f], root=root, now="t1", out=out.append) == 0
    d = decisions(root, "p001")
    assert d["b-001"] == carson and d["b-002"]["choice"] == "A"
    assert d["n-a-0"] == {"choice": "A", "by": "translator", "at": "t1", "reason": "new"}
    assert sum("SKIPPED (carson decision)" in line for line in out) == 2


def test_dry_run_writes_nothing(tmp_path, calls):
    root = feedback_root(tmp_path, {"p001": {"b-001": AUTO, "b-002": AUTO, "n-a-0": AUTO}})
    before = (root / "transcription/arbitration/decisions/p001.json").read_text()
    f = choices_file(tmp_path, [{"alt_id": "p001-b1l0-1", "choice": "B", "reason": "r"}])
    out = []
    assert atc.main([str(f), "--dry-run", "--root", str(root)]) == 0
    assert atc.apply([f], root=root, dry_run=True, out=out.append) == 0
    assert (root / "transcription/arbitration/decisions/p001.json").read_text() == before
    assert calls == [] and any(line.startswith("would set p001 b-001") for line in out)


def test_no_refinalize_skips_the_subprocess(tmp_path, calls):
    root = feedback_root(tmp_path, {"p001": {"b-001": AUTO}})
    f = choices_file(tmp_path, [{"alt_id": "p001-b1l0-1", "choice": "A", "reason": "r"}])
    assert atc.apply([f], root=root, refinalize=False, out=lambda s: None) == 0
    assert calls == [] and decisions(root, "p001")["b-001"]["choice"] == "A"


def test_unmatched_alt_id_fails_and_writes_nothing(tmp_path, calls):
    root = feedback_root(tmp_path, {"p001": {"b-001": AUTO}})
    before = (root / "transcription/arbitration/decisions/p001.json").read_text()
    f = choices_file(tmp_path, [{"alt_id": "p001-b1l0-1", "choice": "B", "reason": "r"},
                                {"alt_id": "p009-b0l0-1", "choice": "A", "reason": "r"}])
    out = []
    assert atc.apply([f], root=root, out=out.append) == 1
    assert (root / "transcription/arbitration/decisions/p001.json").read_text() == before
    assert any("UNMATCHED alt_id p009-b0l0-1" in line for line in out) and calls == []


def test_missing_or_ambiguous_queue_item_fails(tmp_path, calls):
    root = feedback_root(tmp_path, {"p001": {"b-001": AUTO}})
    qf = root / "transcription/arbitration/queue/p001.json"
    q = json.loads(qf.read_text())
    q["items"][0]["b"] = "changed"                                  # no match now
    qf.write_text(json.dumps(q))
    f = choices_file(tmp_path, [{"alt_id": "p001-b1l0-1", "choice": "B", "reason": "r"}])
    out = []
    assert atc.apply([f], root=root, out=out.append) == 1
    assert any("no queue item" in line for line in out)
    q["items"][0]["b"] = "x trahir y"
    q["items"].append(dict(q["items"][0], id="b-009"))              # two identical items
    qf.write_text(json.dumps(q))
    out = []
    assert atc.apply([f], root=root, out=out.append) == 1
    assert any("ambiguous" in line for line in out)


def test_conflicting_choices_on_one_line_fail(tmp_path, calls):
    root = feedback_root(tmp_path, {"p001": {"b-002": AUTO}})
    f = choices_file(tmp_path, [{"alt_id": "p001-b2l0-1", "choice": "A", "reason": "r"},
                                {"alt_id": "p001-b2l0-2", "choice": "B", "reason": "r"}])
    out = []
    assert atc.apply([f], root=root, out=out.append) == 1
    assert any("conflicting choices" in line for line in out)

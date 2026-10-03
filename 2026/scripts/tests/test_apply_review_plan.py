# scripts/tests/test_apply_review_plan.py
"""apply_review_plan.py on a temp root (check_markers subprocess faked)."""
import importlib.util
import json
import pathlib
import types

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("coras_apply_review_plan", SCRIPTS / "apply_review_plan.py")
arp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arp)

CASE = "# Case file\n\n## 9. Glossary\n\n| paillard | rogue |\n| fouasses | cakes |\n\n## 11. Sources\n\n- x\n"
SEC = ("---\nid: annot-001\npages: [p001]\n---\n"
       "⟦p001⟧The kinsmen came {a}, and the kinsmen left {b}. A rogue ⟦p002⟧spoke.\n\n"
       "## Notes\n- {a} (p001): note about kinsmen.\n- {b} (p001): rogue note.\n")


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/case-file.md").write_text(CASE, encoding="utf-8")
    (tmp_path / "translation/sections").mkdir(parents=True)
    (tmp_path / "translation/sections/annot-001.md").write_text(SEC, encoding="utf-8")
    calls = []

    def fake_run(args, **kw):
        calls.append(args)
        return types.SimpleNamespace(returncode=0, stdout="1 sections ok, 0 failed\n", stderr="")
    monkeypatch.setattr(arp, "run", fake_run)
    return types.SimpleNamespace(path=tmp_path, calls=calls)


@pytest.fixture
def root(env):
    return env.path


@pytest.fixture
def calls(env):
    return env.calls


def plan(root, name="p.json", **kw):
    d = {"pass": "pass-t", "case_file": [], "decision_log": [], "sections": []}
    d.update(kw)
    p = root / name
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return p


def go(root, p, dry=False):
    lines = []
    code = arp.apply([p], root=root, dry_run=dry, out=lines.append)
    return code, "\n".join(lines)


def sec(root, sid="annot-001"):
    return (root / f"translation/sections/{sid}.md").read_text(encoding="utf-8")


def case(root):
    return (root / "docs/case-file.md").read_text(encoding="utf-8")


def test_unique_replace_section_and_case_file(root, calls):
    p = plan(root, case_file=[{"old": "| fouasses | cakes |", "new": "| fouaces | cakes |", "reason": "r"}],
             sections=[{"id": "annot-001", "old": "A rogue", "new": "A knave", "reason": "glossary"}])
    code, out = go(root, p)
    assert code == 0
    assert "| fouaces | cakes |" in case(root)
    t = sec(root)
    assert "A knave ⟦p002⟧spoke" in t and "{b} (p001): rogue note" in t  # notes untouched
    assert t.startswith("---\nid: annot-001\npages: [p001]\n---\n")
    assert calls == [["uv", "run", "python", "scripts/check_markers.py", "annot-001"]]
    assert "touched sections (1): annot-001" in out and "check_markers: 1/1 ok" in out


def test_replacement_only_in_prose_and_in_order(root):
    # "rogue" occurs once in prose, once in notes: prose-only count makes it unique
    p = plan(root, sections=[{"id": "annot-001", "old": "rogue", "new": "knave", "reason": "a"},
                             {"id": "annot-001", "old": "A knave", "new": "A lecher", "reason": "b"}])
    code, _ = go(root, p)
    assert code == 0
    t = sec(root)
    assert "A lecher ⟦p002⟧" in t and "rogue note" in t


def test_missing_and_ambiguous(root, calls):
    p = plan(root, case_file=[{"old": "nope", "new": "zz-new", "reason": "r"}],
             sections=[{"id": "annot-001", "old": "kinsmen", "new": "kin", "reason": "amb"},
                       {"id": "annot-001", "old": "absent", "new": "y", "reason": "miss"},
                       {"id": "annot-404", "old": "a", "new": "b", "reason": "nofile"}])
    code, out = go(root, p)
    assert code == 1
    assert sec(root) == SEC and case(root) == CASE
    assert "case_file  applied 0  missing 1" in out
    assert "sections   applied 0  missing 2  already applied? 0  ambiguous 1" in out
    f = (root / "translation/review/annot-001.md").read_text(encoding="utf-8")
    assert f.startswith("# annot-001 review findings\n")
    assert "## pass-t" in f and "PROBLEM: AMBIGUOUS" in f and "PROBLEM: MISSING" in f
    assert calls == []


def test_marker_sequence_protection_reverts_whole_file(root, calls):
    p = plan(root, sections=[{"id": "annot-001", "old": "A rogue", "new": "A knave", "reason": "ok"},
                             {"id": "annot-001", "old": "left {b}", "new": "left", "reason": "drops b"}])
    code, out = go(root, p)
    assert code == 1
    assert sec(root) == SEC
    assert "MARKERS_CHANGED" in out and "markers_changed 2" in out
    assert calls == []
    f = (root / "translation/review/annot-001.md").read_text(encoding="utf-8")
    assert "PROBLEM: MARKERS_CHANGED" in f


def test_marker_reorder_detected(root):
    p = plan(root, sections=[{"id": "annot-001", "old": "A rogue ⟦p002⟧spoke",
                              "new": "⟦p002⟧A rogue spoke", "reason": "move"}])
    assert go(root, p)[0] == 0  # same sequence: fine
    p = plan(root, sections=[{"id": "annot-001", "old": "came {a}, and the kinsmen left {b}",
                              "new": "came {b}, and the kinsmen left {a}", "reason": "swap"}])
    code, out = go(root, p)
    assert code == 1 and "MARKERS_CHANGED" in out


def test_decision_log_created_and_deduped(root):
    p = plan(root, decision_log=["- pass-t: one", "- pass-t: two"])
    assert go(root, p)[0] == 0
    t = case(root)
    assert t.endswith("- x\n\n## 12. Review decision log\n\n- pass-t: one\n- pass-t: two\n")
    p2 = plan(root, "p2.json", decision_log=["- pass-t: two", "- pass-t: three"])
    code, out = go(root, p2)
    assert code == 0 and "decision_log added 1  already present 1" in out
    t = case(root)
    assert t.count("## 12. Review decision log") == 1
    assert t.endswith("## 12. Review decision log\n\n- pass-t: one\n- pass-t: two\n- pass-t: three\n")


def test_decision_log_existing_empty_heading(root):
    (root / "docs/case-file.md").write_text(CASE + "\n## 12. Review decision log\n", encoding="utf-8")
    go(root, plan(root, decision_log=["- a"]))
    assert case(root).endswith("## 12. Review decision log\n\n- a\n")


def test_dry_run_writes_nothing(root, calls):
    p = plan(root, case_file=[{"old": "| fouasses | cakes |", "new": "Z", "reason": "r"}],
             decision_log=["- pass-t: one"],
             sections=[{"id": "annot-001", "old": "A rogue", "new": "A knave", "reason": "g"}],
             uncertain=[{"id": "annot-001", "note": "hmm"}])
    code, out = go(root, p, dry=True)
    assert code == 0
    assert case(root) == CASE and sec(root) == SEC
    assert not (root / "translation/review").exists()
    assert calls == []
    assert "dry run" in out and "sections   applied 1" in out and "UNCERTAIN" in out


def test_dry_run_exit_code_on_problem(root):
    assert go(root, plan(root, case_file=[{"old": "nope", "new": "zz-new", "reason": "r"}]), dry=True)[0] == 1


def test_idempotent_second_apply_reports_already_applied(root):
    p = plan(root, sections=[{"id": "annot-001", "old": "A rogue", "new": "A knave", "reason": "g"}],
             decision_log=["- pass-t: one"])
    assert go(root, p)[0] == 0
    after = sec(root), case(root)
    code, out = go(root, p)
    assert code == 1
    assert (sec(root), case(root)) == after
    assert "already applied?" in out and "NOTE:" in out
    f = (root / "translation/review/annot-001.md").read_text(encoding="utf-8")
    assert f.count("## pass-t") == 2  # second block records the rerun's MISSING


def test_uncertain_file_and_findings_append(root):
    rv = root / "translation/review"
    rv.mkdir(parents=True)
    (rv / "annot-001.md").write_text("# Findings: annot-001 (old)\n\nprior\n", encoding="utf-8")
    p = plan(root, sections=[{"id": "annot-001", "old": "A rogue", "new": "A knave", "reason": "glossary row"}],
             uncertain=[{"id": "annot-001", "question": "rogue or knave?"}])
    code, out = go(root, p)
    assert code == 0
    f = (rv / "annot-001.md").read_text(encoding="utf-8")
    assert f.startswith("# Findings: annot-001 (old)\n\nprior\n\n## pass-t\n")
    assert "- glossary row: A rogue → A knave" in f
    u = (rv / "pass-t-uncertain.md").read_text(encoding="utf-8")
    assert "rogue or knave?" in u
    assert "uncertain (1, not applied)" in out


def test_check_markers_failure_reported(root, monkeypatch):
    monkeypatch.setattr(arp, "run", lambda a, **k: types.SimpleNamespace(
        returncode=1, stdout="PROBLEM: annot-001: page markers differ\n", stderr=""))
    p = plan(root, sections=[{"id": "annot-001", "old": "A rogue", "new": "A knave", "reason": "g"}])
    code, out = go(root, p)
    assert code == 1
    assert "check_markers: 0/1 ok" in out and "page markers differ" in out


def test_truncation():
    assert arp.trunc("x" * 300) == "x" * 200 + "…"


def test_main_defaults_to_dry_run(root):
    p = plan(root, sections=[{"id": "annot-001", "old": "A rogue", "new": "A knave", "reason": "g"}])
    assert arp.main([str(p), "--root", str(root)]) == 0
    assert sec(root) == SEC
    assert arp.main([str(p), "--root", str(root), "--apply"]) == 0
    assert "A knave" in sec(root)


def test_findings_notes_appended_not_applied(root, calls):
    p = plan(root, findings=[{"id": "annot-001", "note": "kinsmen/kin left as is"},
                             {"id": "annot-002", "note": "checked, fine"}])
    code, out = go(root, p, dry=True)
    assert code == 0 and not (root / "translation/review").exists()
    code, out = go(root, p)
    assert code == 0 and sec(root) == SEC and calls == []
    assert "findings notes 2" in out and "touched sections (0)" in out
    f = (root / "translation/review/annot-001.md").read_text(encoding="utf-8")
    assert f == "# annot-001 review findings\n\n## pass-t\n\n- kinsmen/kin left as is\n"
    assert (root / "translation/review/annot-002.md").exists()

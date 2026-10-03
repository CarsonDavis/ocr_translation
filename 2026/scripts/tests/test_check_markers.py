import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import check_markers

SEC = {"id": "annot-005", "text": "⟦p040⟧Le texte {a} continue ⟦p041⟧et finit {b}.", "notes": [{"key": "a"}, {"key": "b"}]}

def write(tmp_path, monkeypatch, body):
    monkeypatch.setattr(check_markers, "ROOT", tmp_path)
    (tmp_path / "translation/sections").mkdir(parents=True)
    (tmp_path / "translation/sections/annot-005.md").write_text(body, encoding="utf-8")

def test_good(tmp_path, monkeypatch):
    write(tmp_path, monkeypatch, "---\nid: annot-005\npages: [p040, p041]\n---\n⟦p040⟧The text {a} goes on ⟦p041⟧and ends {b}.\n\n## Notes\n- {a} (p040): **X** — a\n- {b} (p041): **Y** — b\n")
    assert check_markers.check("annot-005", {"annot-005": SEC}) == []

def test_missing_marker_and_note(tmp_path, monkeypatch):
    write(tmp_path, monkeypatch, "---\nid: annot-005\n---\n⟦p040⟧The text goes on ⟦p041⟧and ends {b}.\n\n## Notes\n- {b} (p041): **Y** — b\n")
    probs = check_markers.check("annot-005", {"annot-005": SEC})
    assert any("letter markers differ" in p for p in probs) and any("note {a}" in p for p in probs)

def test_page_marker_order(tmp_path, monkeypatch):
    write(tmp_path, monkeypatch, "---\nid: annot-005\n---\n⟦p041⟧x {a} ⟦p040⟧y {b}\n\n## Notes\n- {a}\n- {b}\n")
    assert any("page markers differ" in p for p in check_markers.check("annot-005", {"annot-005": SEC}))

def test_alternative_readings_do_not_count_as_markers(tmp_path, monkeypatch):
    sec = {"id": "annot-005",
           "text": "⟦p040⟧Le texte {a}.,⟨alt:{a},⟩ continue ⟨alt:{b}:⟩ ⟦p041⟧et finit⟨alt?:{c}⟩ {b}.",
           "notes": [{"key": "a"}, {"key": "b"}]}
    write(tmp_path, monkeypatch, "---\nid: annot-005\n---\n⟦p040⟧The text {a} goes on ⟦p041⟧and ends {b}.\n\n## Notes\n- {a}\n- {b}\n")
    assert check_markers.check("annot-005", {"annot-005": sec}) == []


def test_all_prints_one_line_per_failure_and_count(tmp_path, monkeypatch):
    write(tmp_path, monkeypatch, "---\nid: annot-005\n---\n⟦p040⟧x {a} ⟦p041⟧y {b}\n\n## Notes\n- {a}\n- {b}\n")
    (tmp_path / "translation/sections/annot-006.md").write_text(
        "---\nid: annot-006\n---\n⟦p041⟧x\n", encoding="utf-8")
    sections = {"annot-004": dict(SEC, id="annot-004"), "annot-005": SEC,
                "annot-006": dict(SEC, id="annot-006")}           # annot-004: no file, not checked
    out = []
    assert check_markers.check_all(sections, out=out.append) == 1
    assert len(out) == 2 and out[0].startswith("FAIL annot-006: ") and "page markers" in out[0]
    assert out[-1] == "1/2 ok"
    (tmp_path / "translation/sections/annot-006.md").unlink()
    out = []
    assert check_markers.check_all(sections, out=out.append) == 0 and out == ["1/1 ok"]

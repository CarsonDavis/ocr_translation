# scripts/tests/test_arbitration.py
"""Queue builder and apply script for human arbitration (arbitrate_queue.py,
apply_arbitration.py). Crops are skipped with --no-crops."""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "arbitration"
A, B = FIX / "A" / "t001.json", FIX / "B" / "t001.json"


def run(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                          capture_output=True, text=True, cwd=str(ROOT))


@pytest.fixture
def queue(tmp_path):
    r = run("arbitrate_queue.py", "t001", "--a", A, "--b", B, "--out-dir", tmp_path, "--no-crops")
    assert r.returncode == 0, r.stderr
    return tmp_path, json.loads((tmp_path / "queue" / "t001.json").read_text())


def items_by_id(q):
    return {i["id"]: i for i in q["items"]}


def test_queue_items_and_pointers(queue):
    _, q = queue
    it = items_by_id(q)
    # the punctuation-spacing difference on line 0 is normalized away, never an item
    assert set(it) == {"b-002", "u-a-004", "u-b-005", "n-a-1", "s-folio"}
    b = it["b-002"]
    assert (b["kind"], b["where_a"], b["where_b"], b["index"]) == \
        ("body", "blocks[1].lines[1]", "blocks[1].lines[1]", 2)
    assert (b["a"], b["b"]) == ("vne ligne que A lit ainſi,", "vne ligne que B lit ainſi,")
    assert b["context_before"] == "Le premier ligne {a}, & la ſuite"
    assert b["context_after"] == "ligne commune au milieu"
    ua = it["u-a-004"]
    assert (ua["kind"], ua["side"], ua["where_a"], ua["where_b"]) == \
        ("unmatched", "A", "blocks[1].lines[3]", None)
    ub = it["u-b-005"]
    assert (ub["side"], ub["where_b"], ub["a_after"]) == ("B", "blocks[1].lines[4]", "blocks[1].lines[4]")
    n = it["n-a-1"]
    assert (n["kind"], n["key"], n["line"], n["where_a"], n["a"], n["b"]) == \
        ("note", "a", 1, "margin_notes[0].lines[1]", "iure iur.", "iure iu.")
    s = it["s-folio"]
    assert (s["kind"], s["a"], s["b"]) == ("structural", "12", "21")


def test_space_only_difference_is_not_an_item(tmp_path):
    a = json.loads(A.read_text())
    b = json.loads(A.read_text())
    b["reader"] = "B"
    b["blocks"][1]["lines"][2] = "ligne communeau milieu"
    (tmp_path / "a.json").write_text(json.dumps(a, ensure_ascii=False))
    (tmp_path / "b.json").write_text(json.dumps(b, ensure_ascii=False))
    r = run("arbitrate_queue.py", "t001", "--a", tmp_path / "a.json", "--b", tmp_path / "b.json",
            "--out-dir", tmp_path, "--no-crops")
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "queue" / "t001.json").read_text())["items"] == []


def test_include_flagged(tmp_path):
    r = run("arbitrate_queue.py", "t001", "--a", A, "--b", B, "--out-dir", tmp_path,
            "--no-crops", "--include-flagged")
    assert r.returncode == 0, r.stderr
    it = items_by_id(json.loads((tmp_path / "queue" / "t001.json").read_text()))
    f = it["f-b-003"]
    assert (f["kind"], f["where_a"], f["a"]) == ("flagged", "blocks[1].lines[2]", "ligne commune au milieu")
    assert "u-a-004" in it       # a "faint ink" note is not a flag


def apply(tmp, decisions, *extra):
    (tmp / "decisions").mkdir(exist_ok=True)
    dpath = tmp / "decisions" / "t001.json"
    dpath.write_text(json.dumps({"page": "t001", "decisions": decisions}, ensure_ascii=False))
    out = tmp / "out" / "t001.json"
    r = run("apply_arbitration.py", "t001", "--a", A, "--b", B, "--queue", tmp / "queue" / "t001.json",
            "--decisions", dpath, "--out", out, *extra)
    return r, (json.loads(out.read_text()) if out.exists() else None)


def all_(choice, **over):
    ids = ["b-002", "u-a-004", "u-b-005", "n-a-1", "s-folio"]
    d = {i: {"choice": choice} for i in ids}
    d.update(over)
    return d


def body(page):
    return page["blocks"][1]["lines"]


def test_apply_all_A(queue):
    tmp, _ = queue
    r, page = apply(tmp, all_("A"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert body(page) == ["Le premier ligne {a}, & la ſuite", "vne ligne que A lit ainſi,",
                          "ligne commune au milieu", "ligne que ſeul A a vue",
                          "derniere ligne du bas."]
    assert page["folio"] == "12"
    assert page["margin_notes"][0]["lines"] == ["l. prima. D. de", "iure iur."]
    assert (page["reader"], page["model"]) == ("final", "arbitration")
    assert len(page["decisions"]) == 5
    d = {x["where"]: x for x in page["decisions"]}["blocks[1].lines[1]"]
    assert d == {"where": "blocks[1].lines[1]", "A": "vne ligne que A lit ainſi,",
                 "B": "vne ligne que B lit ainſi,", "chose": "carson-session",
                 "text": "vne ligne que A lit ainſi,", "reason": "arbitration: A"}
    notes = [u["note"] for u in page["uncertain"]]
    assert "faint ink" in notes                   # A's line is still there
    assert "last line cropped" not in notes       # B's line is not


def test_apply_all_B(queue):
    tmp, _ = queue
    r, page = apply(tmp, all_("B"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert body(page) == ["Le premier ligne {a}, & la ſuite", "vne ligne que B lit ainſi,",
                          "ligne commune au milieu", "derniere ligne du bas.",
                          "vne ligne finale de B"]
    assert page["folio"] == "21"
    assert page["margin_notes"][0]["lines"] == ["l. prima. D. de", "iure iu."]
    unc = {u["note"]: u for u in page["uncertain"]}
    assert "faint ink" not in unc                 # its line was removed
    assert unc["last line cropped"]["where"] == "blocks[1].lines[4]"


def test_apply_neither(queue):
    tmp, _ = queue
    r, page = apply(tmp, all_("A", **{"b-002": {"choice": "neither", "text": "vne ligne que C lit ainſi ,"},
                                       "u-b-005": {"choice": "neither", "text": "vne autre finale"},
                                       "u-a-004": {"choice": "neither", "text": ""}}))
    assert r.returncode == 0, r.stdout + r.stderr
    # typed text is normalized; empty typed text removes the line
    assert body(page) == ["Le premier ligne {a}, & la ſuite", "vne ligne que C lit ainſi,",
                          "ligne commune au milieu", "derniere ligne du bas.", "vne autre finale"]
    reasons = {x["reason"] for x in page["decisions"]}
    assert "arbitration: neither" in reasons


def test_apply_either(queue):
    tmp, _ = queue
    r, page = apply(tmp, all_("A", **{"b-002": {"choice": "either"}}))
    assert r.returncode == 0, r.stdout + r.stderr
    assert body(page)[1] == "vne ligne que A lit ainſi,"
    e = [u for u in page["uncertain"] if u["note"].startswith("arbitration:")]
    assert e == [{"where": "blocks[1].lines[1]", "text": "vne ligne que A lit ainſi,",
                  "note": "arbitration: undecided; alternatives: vne ligne que A lit ainſi, "
                          "||| vne ligne que B lit ainſi,", "escalate": False}]
    assert {"where": "blocks[1].lines[1]", "reason": "arbitration: either"}.items() <= \
        next(d for d in page["decisions"] if d["where"] == "blocks[1].lines[1]").items()


def test_apply_unknown_is_a_decision(queue):
    tmp, _ = queue
    r, page = apply(tmp, all_("unknown"))           # no flag needed
    assert r.returncode == 0, r.stdout + r.stderr
    # A's text and A's structure: A's extra line kept, B's extra line not inserted
    assert body(page) == ["Le premier ligne {a}, & la ſuite", "vne ligne que A lit ainſi,",
                          "ligne commune au milieu", "ligne que ſeul A a vue",
                          "derniere ligne du bas."]
    assert page["folio"] == "12"
    arb = {u["where"]: u for u in page["uncertain"] if u["note"].startswith("arbitration:")}
    assert arb["blocks[1].lines[1]"] == {
        "where": "blocks[1].lines[1]", "text": "vne ligne que A lit ainſi,",
        "note": "arbitration: unknown; alternatives: vne ligne que A lit ainſi, "
                "||| vne ligne que B lit ainſi,", "escalate": True}
    assert arb["margin_notes[0].lines[1]"]["note"] == \
        "arbitration: unknown; alternatives: iure iur. ||| iure iu."
    # the line only B has points at the nearest line of the result
    ub = [u for u in arb.values() if "vne ligne finale de B" in u["note"]][0]
    assert ub["where"] == "blocks[1].lines[4]" and ub["text"] == "derniere ligne du bas."
    assert ub["escalate"] is True
    assert all(u["escalate"] for u in arb.values())
    assert {d["reason"] for d in page["decisions"]} == {"arbitration: unknown"}
    assert len(page["decisions"]) == 5


def test_apply_undecided_fails_and_legacy_values(queue):
    tmp, _ = queue
    dec = all_("A")
    del dec["b-002"]
    r, _ = apply(tmp, dec)
    assert r.returncode == 1 and "b-002" in r.stderr
    # legacy decisions files: "skip" reads as undecided, "both" as either
    r, _ = apply(tmp, all_("A", **{"n-a-1": {"choice": "skip"}}))
    assert r.returncode == 1 and "n-a-1" in r.stderr
    r, page = apply(tmp, all_("A", **{"b-002": {"choice": "both"}}))
    assert r.returncode == 0, r.stdout + r.stderr
    notes = [u["note"] for u in page["uncertain"] if u["note"].startswith("arbitration:")]
    assert len(notes) == 1 and notes[0].startswith("arbitration: undecided; alternatives: ")
    r = run("apply_arbitration.py", "t001", "--allow-skips", "--out", tmp / "x.json")
    assert r.returncode == 2 and "unrecognized arguments" in r.stderr


def test_server_store_choices_and_legacy(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import arbitrate_server as srv
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue" / "t001.json").write_text(json.dumps(
        {"page": "t001", "items": [{"id": "x"}, {"id": "y"}, {"id": "z"}]}))
    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "t001.json").write_text(json.dumps(
        {"page": "t001", "decisions": {"x": {"choice": "both"}, "y": {"choice": "skip"}}}))
    store = srv.Store(tmp_path)
    assert store.decisions("t001") == {"x": {"choice": "either"}}
    assert store.progress("t001")["decided"] == 1
    with pytest.raises(ValueError):
        store.decide("t001", "z", "skip", None)
    prog = store.decide("t001", "z", "unknown", None)
    assert (prog["decided"], prog["done"]) == (2, False)
    raw = json.loads((tmp_path / "decisions" / "t001.json").read_text())["decisions"]
    assert raw["x"] == {"choice": "both"} and raw["y"] == {"choice": "skip"}   # not discarded
    assert store.decide("t001", "y", "either", None)["done"] is True


def test_apply_refuses_stale_queue(queue):
    tmp, q = queue
    q["items"][0]["b"] = "something else"
    (tmp / "queue" / "t001.json").write_text(json.dumps(q, ensure_ascii=False))
    r, _ = apply(tmp, all_("A"))
    assert r.returncode == 1 and "no longer match" in r.stderr


def _variant(tmp, mutate_b):
    a = json.loads(A.read_text())
    b = json.loads(A.read_text())
    b["reader"] = "B"
    mutate_b(b)
    (tmp / "a.json").write_text(json.dumps(a, ensure_ascii=False))
    (tmp / "b.json").write_text(json.dumps(b, ensure_ascii=False))
    r = run("arbitrate_queue.py", "t001", "--a", tmp / "a.json", "--b", tmp / "b.json",
            "--out-dir", tmp, "--no-crops")
    assert r.returncode == 0, r.stderr
    return json.loads((tmp / "queue" / "t001.json").read_text())


def _apply_variant(tmp, decisions):
    (tmp / "d.json").write_text(json.dumps({"decisions": decisions}))
    out = tmp / "out" / "t001.json"
    r = run("apply_arbitration.py", "t001", "--a", tmp / "a.json", "--b", tmp / "b.json",
            "--queue", tmp / "queue" / "t001.json", "--decisions", tmp / "d.json", "--out", out)
    return r, (json.loads(out.read_text()) if out.exists() else None)


def test_whole_note_only_in_B(tmp_path):
    q = _variant(tmp_path, lambda b: b["margin_notes"].append(
        {"key": None, "lines": ["Salluſte.", "in Iug."], "beside_line": "ligne commune au milieu"}))
    it = items_by_id(q)
    assert list(it) == ["un-_unkeyed_0-b"]
    u = it["un-_unkeyed_0-b"]
    assert (u["kind"], u["side"], u["where_b"], u["b"]) == \
        ("unmatched", "B", "margin_notes[1]", "Salluſte.\nin Iug.")
    r, page = _apply_variant(tmp_path, {"un-_unkeyed_0-b": {"choice": "B"}})
    assert r.returncode == 0, r.stdout + r.stderr
    assert page["margin_notes"][1] == {"key": None, "lines": ["Salluſte.", "in Iug."],
                                       "beside_line": "ligne commune au milieu"}
    r, page = _apply_variant(tmp_path, {"un-_unkeyed_0-b": {"choice": "A"}})
    assert r.returncode == 0 and len(page["margin_notes"]) == 1


def test_structure_B_takes_B_as_base(tmp_path):
    def split(b):
        lines = b["blocks"][1]["lines"]
        b["blocks"][1]["lines"] = lines[:2]
        b["blocks"].append({"type": "heading", "text": "ANNOTAT. I."})
        b["blocks"].append({"type": "paragraph", "lines": lines[2:]})
    q = _variant(tmp_path, split)
    it = items_by_id(q)
    assert set(it) == {"s-blocks", "u-b-003"}     # block count differs; B has one more column line
    r, page = _apply_variant(tmp_path, {"s-blocks": {"choice": "B"}, "u-b-003": {"choice": "B"}})
    assert r.returncode == 0, r.stdout + r.stderr
    assert [b["type"] for b in page["blocks"]] == ["heading", "paragraph", "heading", "paragraph"]
    assert page["blocks"][2]["text"] == "ANNOTAT. I."
    # the same heading refused from A's structure: dropped from B's base
    r, page = _apply_variant(tmp_path, {"s-blocks": {"choice": "B"}, "u-b-003": {"choice": "A"}})
    assert r.returncode == 0, r.stdout + r.stderr
    assert [b["type"] for b in page["blocks"]] == ["heading", "paragraph", "paragraph"]
    # A's structure with B's heading inserted
    r, page = _apply_variant(tmp_path, {"s-blocks": {"choice": "A"}, "u-b-003": {"choice": "B"}})
    assert r.returncode == 0, r.stdout + r.stderr
    assert [b["type"] for b in page["blocks"]] == ["heading", "paragraph", "heading", "paragraph"]
    assert page["blocks"][3]["lines"][0] == "ligne commune au milieu"


def test_server_next_page_in_manifest_order(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import arbitrate_server as srv
    (tmp_path / "queue").mkdir()
    for pid, ids in (("p010", ["x"]), ("p002", ["x"]), ("p030", ["x", "y"]), ("zz", ["x"])):
        (tmp_path / "queue" / f"{pid}.json").write_text(json.dumps(
            {"page": pid, "items": [{"id": i} for i in ids]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pages": [{"id": "p002"}, {"id": "p010"}, {"id": "p030"}]}))
    store = srv.Store(tmp_path, manifest)
    # manifest order, not name order; pages the manifest lacks go last
    assert [p["page"] for p in store.pages()] == ["p002", "p010", "p030", "zz"]
    # finishing p010 jumps forward to p030 ...
    prog = store.decide("p010", "x", "A", None)
    assert (prog["done"], prog["next_page"], prog["all_done"]) == (True, "p030", False)
    # ... an unfinished page does not jump
    prog = store.decide("p030", "x", "B", None)
    assert (prog["done"], prog["next_page"]) == (False, None)
    prog = store.decide("p030", "y", "either", None)
    assert prog["next_page"] == "zz"
    # finishing the last page wraps round to the first page with work left
    prog = store.decide("zz", "x", "unknown", None)
    assert prog["next_page"] == "p002"
    prog = store.decide("p002", "x", "A", None)
    assert (prog["next_page"], prog["all_done"]) == (None, True)
    # undoing across the boundary reopens the page
    prog = store.decide("p002", "x", "clear", None)
    assert (prog["done"], prog["next_page"], prog["all_done"]) == (False, None, False)
    assert store.next_page("zz") == "p002"

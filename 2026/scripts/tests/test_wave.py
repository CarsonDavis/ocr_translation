# scripts/tests/test_wave.py
"""wave.py page selection for next / queue / apply (subprocess calls faked, temp root),
render_prompt.py's context fallback to reads, and render_translate.py's batch mode."""
import argparse
import importlib.util
import json
import pathlib
import subprocess

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wave = load("coras_wave", "wave.py")
render_prompt = load("coras_render_prompt", "render_prompt.py")
render_translate = load("coras_render_translate", "render_translate.py")

IDS = ["p001", "p002", "p003", "p004", "p005", "p006"]


def put(root, rel, data):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    return p


def queue_file(root, pid, n):
    put(root, f"transcription/arbitration/queue/{pid}.json",
        {"page": pid, "items": [{"id": f"b-{i:03d}", "kind": "body"} for i in range(n)]})


def decide(root, pid, choices):
    put(root, f"transcription/arbitration/decisions/{pid}.json",
        {"page": pid, "decisions": {f"b-{i:03d}": {"choice": c} for i, c in enumerate(choices)}})


@pytest.fixture
def root(tmp_path, monkeypatch):
    pages = [{"id": pid, "status": {"readA": "pending", "readB": "pending", "final": "pending"}}
             for pid in IDS]
    put(tmp_path, "manifest.json", {"pages": pages})
    monkeypatch.setattr(wave, "ROOT", tmp_path)
    monkeypatch.setattr(wave, "SCRATCH", tmp_path / "scratch")
    return tmp_path


class FakeRun:
    """Records every command; `handlers` map a script name to a function(cmd) -> (rc, out)."""

    def __init__(self, handlers=None):
        self.calls, self.handlers = [], handlers or {}

    def __call__(self, args, **kw):
        cmd = [str(a) for a in args]
        self.calls.append(cmd)
        script = next((pathlib.Path(c).name for c in cmd if c.endswith(".py")), "")
        rc, out = self.handlers.get(script, lambda c: (0, ""))(cmd)
        return subprocess.CompletedProcess(cmd, rc, out, "")

    def scripts(self):
        return [next(pathlib.Path(c).name for c in cmd if c.endswith(".py")) for cmd in self.calls]


def set_status(root, pid, **kw):
    m = json.loads((root / "manifest.json").read_text())
    next(r for r in m["pages"] if r["id"] == pid)["status"].update(kw)
    (root / "manifest.json").write_text(json.dumps(m))


# --- next -----------------------------------------------------------------------------

def test_next_renders_only_missing_reader_and_skips_dispatched(root):
    put(root, "transcription/final/p001.json", {})
    put(root, "transcription/reads/A/p002.json", {})          # has A -> only B
    set_status(root, "p003", readA="dispatched", readB="dispatched")  # in flight
    sel = wave.select_next(wave.manifest(), 3)
    assert [(r["id"], todo) for r, todo in sel] == [("p002", ["B"]), ("p004", ["A", "B"]),
                                                   ("p005", ["A", "B"])]
    again = wave.select_next(wave.manifest(), 2, redispatch=True)
    assert [(r["id"], todo) for r, todo in again] == [("p002", ["B"]), ("p003", ["A", "B"])]


def test_next_uses_read_single_and_marks_dispatched(root, monkeypatch):
    fake = FakeRun({"render_prompt.py": lambda c: (0, "PROMPT")})
    monkeypatch.setattr(wave, "run", fake)
    put(root, "transcription/reads/A/p001.json", {})
    wave.cmd_next(argparse.Namespace(size=1, context=3, model="opus", redispatch=False))
    (cmd,) = fake.calls
    assert cmd[2:6] == ["read_single", "p001", "--reader", "B"]
    assert cmd[cmd.index("--out-dir") + 1] == "transcription/reads/{READER}"
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert (root / "scratch/read_single-p001-B.md").read_text() == "PROMPT"
    st = json.loads((root / "manifest.json").read_text())["pages"][0]["status"]
    assert st["readB"] == "dispatched" and st["readA"] == "pending"


# --- queue ----------------------------------------------------------------------------

def test_queue_selection(root):
    for pid in ("p001", "p002", "p003", "p004"):
        for rd in "AB":
            put(root, f"transcription/reads/{rd}/{pid}.json", {})
    put(root, "transcription/reads/A/p005.json", {})           # one read only
    put(root, "transcription/final/p001.json", {})             # already final
    queue_file(root, "p002", 3)                                # queue exists
    m = wave.manifest()
    assert wave.select_queue(m) == ["p003", "p004"]
    assert wave.select_queue(m, rebuild=True) == ["p002", "p003", "p004"]
    assert wave.select_queue(m, rebuild=True, only={"p002"}) == ["p002"]


def test_queue_runs_the_four_steps_and_prints_table(root, monkeypatch, capsys):
    for rd in "AB":
        put(root, f"transcription/reads/{rd}/p003.json", {})

    def fake_queue(cmd):
        queue_file(root, "p003", 4)
        return 0, "p003: 4 items"

    fake = FakeRun({"diff_reads.py": lambda c: (0, "agreement=97.5% body_diffs=2 unmatched=0 "
                                                    "structural=0 note_diffs=2"),
                    "arbitrate_queue.py": fake_queue})
    monkeypatch.setattr(wave, "run", fake)
    wave.cmd_queue(argparse.Namespace(rebuild=False, pages=[]))
    assert fake.scripts() == ["normalize_spacing.py", "auto_resolve.py", "diff_reads.py",
                              "arbitrate_queue.py"]
    assert fake.calls[0][-2:] == [str(root / "transcription/reads/A/p003.json"),
                                  str(root / "transcription/reads/B/p003.json")]
    assert "p003\t97.5%\t4" in capsys.readouterr().out
    # idempotent: second run finds nothing
    fake.calls.clear()
    wave.cmd_queue(argparse.Namespace(rebuild=False, pages=[]))
    assert fake.calls == []


def test_queue_step_failure_is_reported_and_stops_that_page(root, monkeypatch, capsys):
    for rd in "AB":
        put(root, f"transcription/reads/{rd}/p003.json", {})
    fake = FakeRun({"auto_resolve.py": lambda c: (2, "boom")})
    monkeypatch.setattr(wave, "run", fake)
    wave.cmd_queue(argparse.Namespace(rebuild=False, pages=[]))
    assert fake.scripts() == ["normalize_spacing.py", "auto_resolve.py"]
    assert "p003\tERROR\tauto_resolve exit 2" in capsys.readouterr().out


# --- apply ----------------------------------------------------------------------------

def test_apply_selection_requires_every_item_decided(root):
    queue_file(root, "p001", 2); decide(root, "p001", ["A", "B"])          # ready
    queue_file(root, "p002", 2); decide(root, "p002", ["A"])               # one undecided
    queue_file(root, "p003", 2); decide(root, "p003", ["A", "skip"])       # skip = undecided
    queue_file(root, "p004", 0)                                            # nothing to decide
    queue_file(root, "p005", 1); decide(root, "p005", ["either"])
    put(root, "transcription/final/p005.json", {})                         # already final
    # p006: no queue at all
    assert wave.progress("p002") == (1, 2)
    assert wave.progress("p006") is None
    assert wave.select_apply(wave.manifest()) == ["p001", "p004"]


def test_apply_finalizes_ready_pages_syncs_and_stitches(root, monkeypatch, capsys):
    queue_file(root, "p001", 1); decide(root, "p001", ["neither"])
    queue_file(root, "p002", 1)                                            # undecided
    queue_file(root, "p003", 1); decide(root, "p003", ["B"])               # apply fails

    def fake_apply(cmd):
        pid, out = cmd[cmd.index("--out") - 1], pathlib.Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps({"id": pid}))
        return (1, f"{pid}: INVALID") if pid == "p003" else (0, f"{pid}: 1 decisions applied")

    fake = FakeRun({"apply_arbitration.py": fake_apply,
                    "stitch_text.py": lambda c: (0, "1 pages consumed, 2 sections (1 complete), "
                                                    "stopped at p002")})
    monkeypatch.setattr(wave, "run", fake)
    wave.cmd_apply(argparse.Namespace(pages=[]))
    applied = [c[c.index("--out") - 1] for c in fake.calls if "--out" in c]
    assert applied == ["p001", "p003"]                                     # never p002
    assert (root / "transcription/final/p001.json").exists()
    assert not (root / "transcription/final/p003.json").exists()           # invalid kept out
    assert fake.scripts()[-1] == "stitch_text.py"
    m = json.loads((root / "manifest.json").read_text())
    assert [r["status"]["final"] for r in m["pages"][:3]] == ["done", "pending", "pending"]
    out = capsys.readouterr().out
    assert "finalized p001" in out and "FAILED p003" in out and "stopped at p002" in out


def test_apply_page_refuses_undecided(root):
    queue_file(root, "p001", 2); decide(root, "p001", ["A"])
    with pytest.raises(RuntimeError, match="undecided"):
        wave.apply_page("p001")


def test_status_rows(root):
    for rd in "AB":
        put(root, f"transcription/reads/{rd}/p002.json", {})
    queue_file(root, "p002", 3); decide(root, "p002", ["A", "B"])
    put(root, "transcription/reads/A/p003.json", {})
    set_status(root, "p003", readB="dispatched")
    put(root, "transcription/final/p001.json", {})
    rows = wave.status_rows(wave.manifest())
    assert rows == [("p002", "AB", "yes", "2/3", "-"), ("p003", "Ab", "-", "-", "-")]
    assert wave.status_rows(wave.manifest(), show_all=True)[0] == ("p001", "--", "-", "-", "yes")


# --- render_prompt fallback -----------------------------------------------------------

def test_render_prompt_context_falls_back_to_reads(tmp_path):
    put(tmp_path, "transcription/final/p001.json", {})
    put(tmp_path, "transcription/reads/A/p002.json", {})
    put(tmp_path, "transcription/reads/B/p002.json", {})
    put(tmp_path, "transcription/reads/B/p003.json", {})
    # p004: nothing
    got = render_prompt.context_files(["p001", "p002", "p003", "p004"], root=tmp_path)
    assert got == ["transcription/final/p001.json",
                   "transcription/reads/A/p002.json (unreconciled read)",
                   "transcription/reads/B/p003.json (unreconciled read)"]
    assert render_prompt.context_files(["p001", "p002"], reads_ok=False, root=tmp_path) == \
        ["transcription/final/p001.json"]


def test_read_single_template_explains_reads():
    tpl = (SCRIPTS / "prompts/read_single.md").read_text()
    assert "unreconciled reads, use them only for continuity" in tpl
    assert "{CONTEXT_PAGES}" in tpl


# --- render_translate batch mode ------------------------------------------------------

SECS = [{"id": f"s{i}", "complete": i < 5} for i in range(7)]


def test_translate_batch_ids_stop_at_incomplete():
    assert render_translate.batch_ids(SECS, "s1", 2) == ["s1", "s2"]
    assert render_translate.batch_ids(SECS, "s3", 12) == ["s3", "s4"]


def test_translate_batch_and_single_render(tmp_path):
    for sid in ("s0", "s1", "s2"):
        (tmp_path / "translation/sections").mkdir(parents=True, exist_ok=True)
        (tmp_path / f"translation/sections/{sid}.md").write_text("x")
    tpl = (SCRIPTS / "prompts/translate.md").read_text()
    b = render_translate.render_batch(tpl, SECS, "s3", 3, root=tmp_path)
    assert "`s3`, `s4`" in b and "batch-s3--s4.md" in b
    assert "s0 (translation/sections/s0.md)" in b and "s2 (translation" in b
    assert "<!--" not in b and "{SECTION_ID}" not in b and "text/sections.json" in b
    s = render_translate.render_single(tpl, SECS, "s3", 2, root=tmp_path)
    assert "check_markers.py s3" in s and "translation/reports/s3.md" in s
    assert "batch report" not in s and "Read before you write" not in s and "<!--" not in s and "{" + "SECTION_IDS}" not in s

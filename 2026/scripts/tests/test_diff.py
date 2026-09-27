# scripts/tests/test_diff.py
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIFF = ROOT / "scripts" / "diff_reads.py"
FIX = pathlib.Path(__file__).resolve().parent / "fixtures"
MANIFEST = FIX / "manifest.json"

STDOUT_RE = re.compile(
    r"^agreement=(\d+\.\d)% body_diffs=(\d+) unmatched=(\d+) structural=(\d+) note_diffs=(\d+)$"
)


def run(*args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, str(DIFF), *[str(a) for a in args]],
        capture_output=True, text=True, cwd=str(cwd),
    )


def parse_stdout(out):
    line = out.strip().splitlines()[-1]
    m = STDOUT_RE.match(line)
    assert m, f"bad stdout line: {line!r}"
    return float(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))


def test_ab_fixtures_differ(tmp_path):
    out = tmp_path / "p018.md"
    r = run("p018", "--a", FIX / "A" / "p018.json", "--b", FIX / "B" / "p018.json",
            "--out", out, "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout + r.stderr
    agreement, body_diffs, unmatched, structural, note_diffs = parse_stdout(r.stdout)
    assert 0 < agreement < 100
    assert body_diffs == 1
    assert unmatched == 0
    assert note_diffs >= 1
    text = out.read_text(encoding="utf-8")
    assert "agreement=" in text
    assert "d'vn fils" in text and "d'vn filz" in text
    assert "A[2]" in text and "B[2]" in text
    assert "Body line differences" in text
    assert "Note differences" in text
    assert "Summary" in text
    assert "chap. xxiii." in text


def test_identical_files_are_100(tmp_path):
    out = tmp_path / "p018.md"
    r = run("p018", "--a", FIX / "A" / "p018.json", "--b", FIX / "A" / "p018.json",
            "--out", out, "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "agreement=100.0% body_diffs=0 unmatched=0 structural=0 note_diffs=0" in r.stdout


def test_structural_differences_reported(tmp_path):
    a = json.loads((FIX / "A" / "p018.json").read_text(encoding="utf-8"))
    b = json.loads((FIX / "A" / "p018.json").read_text(encoding="utf-8"))
    b["folio"] = "81"
    b["signature"] = "C ij"
    b["blocks"].insert(0, {"type": "heading", "text": "TEXTE."})
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")
    pb.write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "p018.md"
    r = run("p018", "--a", pa, "--b", pb, "--out", out, "--manifest", MANIFEST)
    _, _, _, structural, _ = parse_stdout(r.stdout)
    assert structural >= 3
    text = out.read_text(encoding="utf-8")
    assert "Structural differences" in text
    assert "folio" in text and "signature" in text and "TEXTE." in text


def test_unmatched_lines_reported(tmp_path):
    a = json.loads((FIX / "A" / "p018.json").read_text(encoding="utf-8"))
    b = json.loads((FIX / "A" / "p018.json").read_text(encoding="utf-8"))
    del b["blocks"][0]["lines"][1]
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")
    pb.write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "p018.md"
    r = run("p018", "--a", pa, "--b", pb, "--out", out, "--manifest", MANIFEST)
    agreement, body_diffs, unmatched, _, _ = parse_stdout(r.stdout)
    assert unmatched == 1
    assert agreement == 80.0  # 4 identical aligned lines / max(5, 4)
    assert "Unmatched body lines" in out.read_text(encoding="utf-8")


def test_normalization_still_reports_exact_differences(tmp_path):
    """Alignment ignores case/long-s/spacing, but the report is character-exact."""
    a = json.loads((FIX / "A" / "p018.json").read_text(encoding="utf-8"))
    b = json.loads((FIX / "A" / "p018.json").read_text(encoding="utf-8"))
    b["blocks"][0]["lines"][0] = b["blocks"][0]["lines"][0].replace("ſ'en", "s'en")
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")
    pb.write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "p018.md"
    r = run("p018", "--a", pa, "--b", pb, "--out", out, "--manifest", MANIFEST)
    agreement, body_diffs, unmatched, _, _ = parse_stdout(r.stdout)
    assert body_diffs == 1 and unmatched == 0
    assert agreement == 80.0


def setup_workdir(tmp_path):
    for reader in ("A", "B"):
        d = tmp_path / "transcription" / "reads" / reader
        d.mkdir(parents=True)
        (d / "p018.json").write_text((FIX / reader / "p018.json").read_text(encoding="utf-8"),
                                     encoding="utf-8")
    (tmp_path / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def test_default_paths_and_manifest_update(tmp_path):
    wd = setup_workdir(tmp_path)
    r = run("p018", cwd=wd)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (wd / "transcription" / "diff" / "p018.md").exists()
    man = json.loads((wd / "manifest.json").read_text(encoding="utf-8"))
    rec = {p["id"]: p for p in man["pages"]}["p018"]
    assert rec["status"]["diffed"] == "done"
    assert isinstance(rec["agreement"], float) and 0 < rec["agreement"] < 100
    # everything else is preserved, and the file keeps indent=1
    assert man["item"] == "TEST"
    assert {p["id"]: p for p in man["pages"]}["p019"]["status"]["diffed"] == "pending"
    raw = (wd / "manifest.json").read_text(encoding="utf-8")
    assert raw.startswith('{\n "item"')


def test_no_manifest_flag(tmp_path):
    wd = setup_workdir(tmp_path)
    before = (wd / "manifest.json").read_text(encoding="utf-8")
    r = run("p018", "--no-manifest", cwd=wd)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (wd / "manifest.json").read_text(encoding="utf-8") == before


def test_spotcheck_mode_does_not_touch_manifest(tmp_path):
    wd = setup_workdir(tmp_path)
    before = (wd / "manifest.json").read_text(encoding="utf-8")
    r = run("p018", "--a", "transcription/reads/A/p018.json",
            "--out", "spot.md", cwd=wd)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (wd / "spot.md").exists()
    assert (wd / "manifest.json").read_text(encoding="utf-8") == before


def test_unkeyed_notes_are_compared(tmp_path):
    a = json.loads((FIX / "unkeyed_note" / "p018.json").read_text(encoding="utf-8"))
    b = json.loads((FIX / "unkeyed_note" / "p018.json").read_text(encoding="utf-8"))
    b["margin_notes"][1]["lines"] = ["Cicero in Verrem."]
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")
    pb.write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "p018.md"
    r = run("p018", "--a", pa, "--b", pb, "--out", out, "--manifest", MANIFEST)
    _, _, _, _, note_diffs = parse_stdout(r.stdout)
    assert note_diffs == 1
    assert "_unkeyed_0" in out.read_text(encoding="utf-8")


def test_empty_body_is_not_100_percent(tmp_path):
    out = tmp_path / "p018.md"
    r = run("p018", "--a", FIX / "no_body" / "p018.json", "--b", FIX / "no_body" / "p018.json",
            "--out", out, "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout + r.stderr
    agreement, body_diffs, unmatched, structural, _ = parse_stdout(r.stdout)
    assert agreement == 0.0
    assert structural >= 1
    text = out.read_text(encoding="utf-8")
    assert "no body lines" in text
    assert "agreement=0.0%" in text


# --- Unicode --------------------------------------------------------------

def test_nfc_and_nfd_reads_agree(tmp_path):
    out = tmp_path / "p018.md"
    r = run("p018", "--a", FIX / "A" / "p018.json", "--b", FIX / "nfd" / "p018.json",
            "--out", out, "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "agreement=100.0% body_diffs=0 unmatched=0 structural=0 note_diffs=0" in r.stdout
    text = out.read_text(encoding="utf-8")
    assert "## Body line differences\n\nnone" in text


def test_invisible_difference_is_quoted(tmp_path):
    a = json.loads((FIX / "A" / "p018.json").read_text(encoding="utf-8"))
    b = json.loads((FIX / "A" / "p018.json").read_text(encoding="utf-8"))
    b["blocks"][0]["lines"][1] = b["blocks"][0]["lines"][1].replace("iette ſur", "iette  ſur")
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")
    pb.write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "p018.md"
    r = run("p018", "--a", pa, "--b", pb, "--out", out, "--manifest", MANIFEST)
    _, body_diffs, _, _, _ = parse_stdout(r.stdout)
    assert body_diffs == 1
    text = out.read_text(encoding="utf-8")
    assert "differs only in invisible characters or spacing" in text
    shown = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("B[1]:")][0]
    body = shown[len("B[1]: "):]
    assert body[0] in "\"'", shown          # quoted, so the double space is visible
    assert "iette  ſur" in body, shown


# --- unreadable input -----------------------------------------------------

def test_missing_input_exits_2(tmp_path):
    r = run("p018", "--a", tmp_path / "nope.json", "--b", FIX / "B" / "p018.json",
            "--out", tmp_path / "o.md", "--manifest", MANIFEST)
    assert r.returncode == 2
    assert "cannot read" in r.stderr and "file not found" in r.stderr, r.stderr


def test_invalid_json_input_exits_2(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    r = run("p018", "--a", bad, "--b", FIX / "B" / "p018.json",
            "--out", tmp_path / "o.md", "--manifest", MANIFEST)
    assert r.returncode == 2
    assert "cannot read" in r.stderr and "invalid JSON" in r.stderr, r.stderr
    assert not (tmp_path / "o.md").exists()


def test_broken_manifest_exits_2_before_writing(tmp_path):
    wd = setup_workdir(tmp_path)
    (wd / "manifest.json").write_text("{oops", encoding="utf-8")
    r = run("p018", cwd=wd)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "cannot read" in r.stderr, r.stderr
    assert not (wd / "transcription" / "diff" / "p018.md").exists()


def test_id_mismatch_refuses_the_manifest_write(tmp_path):
    wd = setup_workdir(tmp_path)
    page = json.loads((wd / "transcription/reads/A/p018.json").read_text(encoding="utf-8"))
    page["id"] = "p019"
    (wd / "transcription/reads/A/p018.json").write_text(
        json.dumps(page, ensure_ascii=False), encoding="utf-8")
    before = (wd / "manifest.json").read_text(encoding="utf-8")
    r = run("p018", cwd=wd)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "has id 'p019'" in r.stderr, r.stderr
    assert (wd / "transcription" / "diff" / "p018.md").exists()
    assert (wd / "manifest.json").read_text(encoding="utf-8") == before


def test_non_dict_block_does_not_crash(tmp_path):
    a = json.loads((FIX / "A" / "p018.json").read_text(encoding="utf-8"))
    b = json.loads((FIX / "A" / "p018.json").read_text(encoding="utf-8"))
    b["blocks"].append("ornament")
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")
    pb.write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "p018.md"
    r = run("p018", "--a", pa, "--b", pb, "--out", out, "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout + r.stderr
    _, _, _, structural, _ = parse_stdout(r.stdout)
    assert structural == 1
    assert "block count: A=1 B=2" in out.read_text(encoding="utf-8")

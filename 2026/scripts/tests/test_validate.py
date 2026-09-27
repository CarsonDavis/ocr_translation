# scripts/tests/test_validate.py
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VALIDATE = ROOT / "scripts" / "validate_page.py"
FIX = pathlib.Path(__file__).resolve().parent / "fixtures"
MANIFEST = FIX / "manifest.json"


def run(*args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, str(VALIDATE), *[str(a) for a in args]],
        capture_output=True, text=True, cwd=str(cwd),
    )


def problems(out):
    return [ln for ln in out.splitlines() if "PROBLEM:" in ln]


def warnings(out):
    return [ln for ln in out.splitlines() if "WARNING:" in ln]


def test_good_page_passes_without_manifest():
    r = run(FIX / "A" / "p018.json", "--manifest", FIX / "does-not-exist.json")
    assert r.returncode == 0, r.stdout + r.stderr
    assert problems(r.stdout) == []
    assert "1 ok, 0 failed" in r.stdout


def test_good_page_passes_with_manifest():
    r = run(FIX / "A" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 ok, 0 failed" in r.stdout


def test_orphan_marker_names_the_key():
    r = run(FIX / "orphan_marker" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("marker {c} in the body has no margin_notes/foot_notes entry "
               "with key 'c'" in p for p in problems(r.stdout)), r.stdout
    assert "0 ok, 1 failed" in r.stdout


def test_note_key_without_marker_is_a_problem():
    r = run(FIX / "unmarked_note" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("note key 'c' has no {c} marker in the body" in p
               for p in problems(r.stdout)), r.stdout


def test_note_key_without_marker_excused_by_uncertain():
    r = run(FIX / "note_excused" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout


def test_duplicate_note_keys():
    r = run(FIX / "dup_keys" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("duplicate" in p.lower() for p in problems(r.stdout)), r.stdout


def test_folio_mismatch():
    r = run(FIX / "bad_folio" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("folio" in p.lower() for p in problems(r.stdout)), r.stdout


def test_folio_mismatch_excused_by_uncertain():
    r = run(FIX / "folio_excused" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout


def test_long_s_warning_does_not_fail():
    r = run(FIX / "long_s" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout
    warns = warnings(r.stdout)
    assert len(warns) == 1, r.stdout
    assert 'blocks[0].lines[4]: possible normalized long s (si, se) in ' in warns[0]
    assert 'Et si le lecteur ne se contente,"' in warns[0]


def test_extra_field_fails_schema():
    r = run(FIX / "extra_field" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("additional_field" in p for p in problems(r.stdout)), r.stdout


def test_id_must_match_filename_stem():
    r = run(FIX / "id_mismatch" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("id 'p019' does not match the filename stem 'p018'" in p
               for p in problems(r.stdout)), r.stdout


def test_whitespace_problems():
    r = run(FIX / "bad_ws" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    probs = problems(r.stdout)
    assert any("trailing" in p for p in probs), r.stdout
    assert any("double space" in p for p in probs), r.stdout


def test_spaced_caps_block_is_closed_up_and_flagged():
    # §3: spaced capitals are transcribed closed up; the flag alone records the spacing
    r = run(FIX / "spaced_caps" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout


def test_uncertain_marker_needs_an_entry():
    bad = run(FIX / "missing_uncertain" / "p018.json", "--manifest", MANIFEST)
    assert bad.returncode == 1
    assert any("[?]" in p for p in problems(bad.stdout)), bad.stdout
    good = run(FIX / "uncertain_ok" / "p018.json", "--manifest", MANIFEST)
    assert good.returncode == 0, good.stdout


def test_several_paths_and_summary():
    r = run(FIX / "A" / "p018.json", FIX / "bad_folio" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert "1 ok, 1 failed" in r.stdout
    for line in problems(r.stdout):
        assert line.split(": PROBLEM:")[0].endswith("p018.json")


def test_all_reads_and_all_final(tmp_path):
    for reader in ("A", "B"):
        d = tmp_path / "transcription" / "reads" / reader
        d.mkdir(parents=True)
        page = json.loads((FIX / reader / "p018.json").read_text(encoding="utf-8"))
        (d / "p018.json").write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
    final = tmp_path / "transcription" / "final"
    final.mkdir(parents=True)
    bad = json.loads((FIX / "bad_folio" / "p018.json").read_text(encoding="utf-8"))
    bad["reader"] = "final"
    (final / "p018.json").write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")

    r = run("--all-reads", cwd=tmp_path)
    assert r.returncode == 0, r.stdout
    assert "2 ok, 0 failed" in r.stdout

    r = run("--all-final", cwd=tmp_path)
    assert r.returncode == 1, r.stdout
    assert "0 ok, 1 failed" in r.stdout


def test_typo_inside_a_block_fails_schema():
    r = run(FIX / "bad_block_field" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("continues_previous" in p for p in problems(r.stdout)), r.stdout


def test_unkeyed_margin_note_is_fine():
    r = run(FIX / "unkeyed_note" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout


# --- whitespace beyond the body lines (conventions §1, applied to every field) ---

def test_heading_and_ornament_text_whitespace():
    r = run(FIX / "bad_heading_ws" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    probs = " | ".join(problems(r.stdout))
    assert "TEXTE." in probs and "trailing" in probs, r.stdout
    assert "newline" in probs, r.stdout
    assert "double space" in probs, r.stdout


def test_page_furniture_whitespace():
    r = run(FIX / "bad_field_ws" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    probs = problems(r.stdout)
    assert any(p.startswith(f"{FIX / 'bad_field_ws' / 'p018.json'}: PROBLEM: folio:") for p in probs), r.stdout
    assert any("signature" in p and "trailing" in p for p in probs), r.stdout
    assert any("catchword" in p and "tab" in p for p in probs), r.stdout


def test_running_head_is_closed_up_and_not_padded():
    # the good fixture's running head is "ARREST DV" (closed up, §5)
    assert run(FIX / "A" / "p018.json", "--manifest", MANIFEST).returncode == 0
    r = run(FIX / "running_head_ws" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("running_head" in p and "trailing" in p for p in problems(r.stdout)), r.stdout


# --- markers in headings, and markers where they must not be ---

def test_heading_marker_is_cross_checked():
    ok = run(FIX / "heading_marker" / "p018.json", "--manifest", MANIFEST)
    assert ok.returncode == 0, ok.stdout
    bad = run(FIX / "heading_marker_orphan" / "p018.json", "--manifest", MANIFEST)
    assert bad.returncode == 1
    assert any("{b}" in p for p in problems(bad.stdout)), bad.stdout


def test_marker_inside_a_note_line_is_a_problem():
    r = run(FIX / "marker_in_note" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("marker inside a note line" in p for p in problems(r.stdout)), r.stdout


# --- uncertain[] coverage must actually point at the line ---

def test_unrelated_uncertain_entry_does_not_excuse():
    r = run(FIX / "unrelated_uncertain" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("[?]" in p and "blocks[0]" in p for p in problems(r.stdout)), r.stdout


def test_uncertain_sign_in_a_field():
    ok = run(FIX / "folio_uncertain_ok" / "p018.json", "--manifest", MANIFEST)
    assert ok.returncode == 0, ok.stdout
    bad = run(FIX / "folio_uncertain_missing" / "p018.json", "--manifest", MANIFEST)
    assert bad.returncode == 1
    assert any("folio" in p and "[?]" in p for p in problems(bad.stdout)), bad.stdout


# --- batch flags ---

def test_all_flags_on_missing_directory(tmp_path):
    r = run("--all-final", cwd=tmp_path)
    assert r.returncode == 1, r.stdout
    assert "transcription/final: PROBLEM: directory not found" in r.stdout
    assert "0 ok, 0 failed (1 directory problem)" in r.stdout
    r = run("--all-reads", cwd=tmp_path)
    assert r.returncode == 1, r.stdout
    assert "transcription/reads: PROBLEM: directory not found" in r.stdout


def test_all_flags_on_empty_directory(tmp_path):
    (tmp_path / "transcription" / "final").mkdir(parents=True)
    r = run("--all-final", cwd=tmp_path)
    assert r.returncode == 1, r.stdout
    assert "transcription/final: PROBLEM: no page JSON files found" in r.stdout
    assert "0 ok, 0 failed (1 directory problem)" in r.stdout


# --- schema error messages ---

def test_block_schema_error_names_only_the_bogus_key():
    r = run(FIX / "bad_block_field" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    probs = problems(r.stdout)
    assert len(probs) == 1, r.stdout
    assert "continues_previous" in probs[0]
    lowered = probs[0].lower()
    assert "oneof" not in lowered and "is not valid under any" not in lowered, probs[0]
    assert "heading" not in lowered and "ornament" not in lowered, probs[0]


def test_spaced_letters_are_always_a_problem():
    # §3: letters set apart are closed up in the text; a spaced-out heading fails
    # whether or not the block declares spaced_caps
    ok = run(FIX / "spaced_caps_heading" / "p018.json", "--manifest", MANIFEST)
    assert ok.returncode == 0, ok.stdout
    bad = run(FIX / "undeclared_spaced_heading" / "p018.json", "--manifest", MANIFEST)
    assert bad.returncode == 1
    assert any("blocks[0].text" in p and ("spacing" in p or "double space" in p)
               for p in problems(bad.stdout)), bad.stdout


# --- unreadable input -----------------------------------------------------

def test_missing_file_is_a_problem(tmp_path):
    r = run(tmp_path / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("cannot read: file not found" in p for p in problems(r.stdout)), r.stdout


def test_directory_given_as_a_page(tmp_path):
    (tmp_path / "p018.json").mkdir()
    r = run(tmp_path / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("cannot read: is a directory" in p for p in problems(r.stdout)), r.stdout


def test_invalid_json(tmp_path):
    (tmp_path / "p018.json").write_text('{"id": "p018",,}', encoding="utf-8")
    r = run(tmp_path / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("cannot read: invalid JSON" in p for p in problems(r.stdout)), r.stdout


def test_not_utf8(tmp_path):
    (tmp_path / "p018.json").write_bytes(b'{"id": "p018", "folio": "\xff\xfe18"}')
    r = run(tmp_path / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("cannot read: not valid UTF-8" in p for p in problems(r.stdout)), r.stdout


def test_top_level_is_a_list(tmp_path):
    (tmp_path / "p018.json").write_text('[]', encoding="utf-8")
    r = run(tmp_path / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("cannot read: top level is list" in p for p in problems(r.stdout)), r.stdout


def test_unreadable_manifest(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text("{oops", encoding="utf-8")
    r = run(FIX / "A" / "p018.json", "--manifest", bad)
    assert r.returncode == 1
    assert "cannot read the manifest: invalid JSON" in r.stdout, r.stdout


# --- schema messages ------------------------------------------------------

def test_unknown_block_type_message():
    r = run(FIX / "bad_type" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    probs = problems(r.stdout)
    assert len(probs) == 1, r.stdout
    assert probs[0].endswith("schema: blocks/0: type must be one of "
                             "heading/paragraph/ornament/blank (got 'para')"), probs[0]


def test_missing_block_type_message():
    r = run(FIX / "no_type" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    probs = problems(r.stdout)
    assert len(probs) == 1, r.stdout
    assert probs[0].endswith("(got nothing)"), probs[0]


def test_schema_messages_never_dump_the_page():
    for name in ("bad_type", "no_type", "extra_field", "bad_block_field"):
        r = run(FIX / name / "p018.json", "--manifest", MANIFEST)
        for p in problems(r.stdout):
            detail = p.split(": PROBLEM: ", 1)[1]
            assert len(detail) <= 140, (name, detail)
            assert "ſauteller" not in detail, (name, detail)


def test_beside_line_may_be_null():
    r = run(FIX / "beside_line_null" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout


# --- NFC ------------------------------------------------------------------

def test_nfd_text_warns():
    r = run(FIX / "nfd" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout
    warns = warnings(r.stdout)
    assert any("text is not in NFC (use precomposed ã ẽ ĩ õ ũ, §2)" in w for w in warns), r.stdout
    assert any("blocks[0].lines[0]:" in w for w in warns), r.stdout


# --- uncertain[] pointers -------------------------------------------------

def test_sibling_line_entry_does_not_excuse():
    r = run(FIX / "sibling_uncertain" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("blocks[0].lines[2]: [?] with no uncertain[] entry pointing at it" in p
               for p in problems(r.stdout)), r.stdout


def test_block_level_entry_covers_its_lines():
    r = run(FIX / "block_level_uncertain" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 0, r.stdout


def test_all_signs_on_a_line_are_reported():
    r = run(FIX / "double_sign" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("blocks[0].lines[2]: [??], [...] with no uncertain[] entry at all" in p
               for p in problems(r.stdout)), r.stdout


# --- marker keys ----------------------------------------------------------

def test_capital_marker_is_reported():
    r = run(FIX / "capital_marker" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    probs = problems(r.stdout)
    assert any("marker {U} is not lowercase; marker keys run a…z, a2, a3 (§4)" in p
               for p in probs), r.stdout
    # the case problem is reported once, not doubled by the cross-check
    assert not any("marker {U} in the body has no" in p for p in probs), r.stdout


def test_mentions_key_is_word_bounded():
    r = run(FIX / "mentions_c2" / "p018.json", "--manifest", MANIFEST)
    assert r.returncode == 1
    assert any("note key 'c' has no {c} marker" in p for p in problems(r.stdout)), r.stdout


# --- CLI ------------------------------------------------------------------

def test_flags_and_paths_are_mutually_exclusive():
    r = run("--all-final", FIX / "A" / "p018.json")
    assert r.returncode == 2
    assert "take no paths" in r.stderr, r.stderr

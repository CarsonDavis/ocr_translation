#!/usr/bin/env python3
"""Validate a transcribed page against docs/conventions.md §8.

    uv run --with jsonschema python scripts/validate_page.py <path> [<path> ...]
    uv run --with jsonschema python scripts/validate_page.py --all-reads
    uv run --with jsonschema python scripts/validate_page.py --all-final

Problems are printed as `path: PROBLEM: detail` and make the exit code 1.
Warnings are printed as `path: WARNING: detail` and do not.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pagelib  # noqa: E402

# Words the print sets with a long s. If a reader silently modernized, these
# short-s spellings show up. Word-final s is legitimately short, which is why
# the list only holds words with an initial or interior s.
LONG_S_WORDS = {
    "est", "sont", "aussi", "sans", "son", "ses", "sur", "se", "si", "aussy",
    "assez", "ainsi", "chose", "choses", "cause", "sera", "seroit", "faisant",
    "disant", "iustice", "suspect", "ainsy", "plusieurs", "presente",
    "personne", "mesme", "mesmes",
}

UNCERTAIN_SIGNS = ("[??]", "[?]", "[...]", "[abbr:")
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
MAX_MESSAGE = 120


def _short(text, limit=MAX_MESSAGE):
    """Keep a schema message readable: one line, never a dump of the page."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


# --- individual checks ----------------------------------------------------

def check_schema(page, problems):
    try:
        import jsonschema  # noqa: F401
    except ImportError:  # pragma: no cover - environment problem, not data
        problems.append("jsonschema is not installed; run with "
                        "`uv run --with jsonschema python scripts/validate_page.py`")
        return False
    from jsonschema import Draft202012Validator

    schema = pagelib.load_schema()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(page), key=lambda e: list(e.absolute_path))
    if not errors:
        return True

    found = []
    for err in errors[:20]:
        path = list(err.absolute_path)
        block_errs = _block_errors(schema, page, path)
        if block_errs is not None:
            found.extend(block_errs)
            continue
        where = "/".join(str(p) for p in path) or "<root>"
        if err.validator == "oneOf":
            # the oneOf message quotes the whole instance back; say it plainly
            found.append(f"schema: {where}: does not match any allowed shape")
        else:
            found.append(f"schema: {where}: {_short(err.message)}")

    for f in found:  # the oneOf wrapper can report the same block twice
        if f not in problems:
            problems.append(f)
    return False


def _block_errors(schema, page, path):
    """Re-validate one block against the subschema for its own `type`.

    The block schema is a oneOf over the four block types, whose error message
    ("is not valid under any of the given schemas") both hides the real mistake
    and quotes the whole block back. A block that declares a known type is
    checked against that type alone, so the message names the offending key.
    """
    if len(path) < 2 or path[0] != "blocks" or not isinstance(path[1], int):
        return None
    from jsonschema import Draft202012Validator

    try:
        block = page["blocks"][path[1]]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(block, dict):
        return [f"schema: blocks/{path[1]}: block is "
                f"{type(block).__name__}, expected an object"]

    btype = block.get("type")
    if btype not in pagelib.BLOCK_TYPES:
        got = repr(btype) if btype is not None else "nothing"
        return [f"schema: blocks/{path[1]}: type must be one of "
                f"{'/'.join(pagelib.BLOCK_TYPES)} (got {got})"]

    sub = Draft202012Validator({**schema["$defs"][f"{btype}_block"],
                                "$defs": schema["$defs"]})
    out = []
    for err in sorted(sub.iter_errors(block), key=lambda e: list(e.absolute_path))[:10]:
        tail = "/".join(str(x) for x in err.absolute_path)
        where = f"blocks/{path[1]}" + (f"/{tail}" if tail else "")
        out.append(f"schema: {where}: {_short(err.message)}")
    return out or None


def _mentions_key(page, key):
    """True if some uncertain[] entry mentions this marker/note key.

    Word-boundary matched, so an entry about `c2` does not excuse `{c}`.
    """
    patterns = [re.compile(r"\{%s\}" % re.escape(key)),
                re.compile(r"\b(?:key|marker)\s+%s\b" % re.escape(key))]
    return any(p.search(blob) for blob in pagelib.uncertain_blobs(page) for p in patterns)


def check_markers(page, problems):
    for key, where in pagelib.note_markers(page):
        problems.append(f"{where}: marker inside a note line ({{{key}}}); a note's own "
                        f"key letter is the `key`, not part of its lines")

    marker_keys = set()
    for key, where in pagelib.markers(page):
        if key != key.lower():
            # reported on its own; cross-checking it too would only be noise
            problems.append(f"{where}: marker {{{key}}} is not lowercase; marker keys "
                            f"run a…z, a2, a3 (§4)")
            continue
        marker_keys.add(key)

    note_keys = [n.key for n in pagelib.notes(page) if n.key is not None]
    seen = set()
    for key in note_keys:
        if key in seen:
            problems.append(f"duplicate note key {key!r}")
        seen.add(key)

    for key in sorted(marker_keys - seen):
        if not _mentions_key(page, key):
            problems.append(
                f"marker {{{key}}} in the body has no margin_notes/foot_notes entry "
                f"with key {key!r} and no uncertain[] entry mentioning it")
    for key in sorted(seen - marker_keys):
        if not _mentions_key(page, key):
            problems.append(
                f"note key {key!r} has no {{{key}}} marker in the body and no "
                f"uncertain[] entry mentioning it")


def check_whitespace(page, problems):
    for line in pagelib.all_lines(page):
        text = line.text
        if "\n" in text:
            problems.append(f"{line.where}: contains a newline")
        if "\t" in text:
            problems.append(f"{line.where}: contains a tab")
        if text != text.strip():
            if text != text.lstrip() and text != text.rstrip():
                side = "leading and trailing"
            elif text != text.lstrip():
                side = "leading"
            else:
                side = "trailing"
            problems.append(f"{line.where}: {side} whitespace: {text!r}")
        if "  " in text:
            problems.append(f"{line.where}: double space (spaced capitals are closed up, "
                            f"§3; use single spaces): {text!r}")


def check_punct_spacing(page, problems):
    """Conventions §1: no space before , . : ; ? ! and one space after them (and around
    parentheses) when text follows. Deterministic, so `scripts/normalize_spacing.py`
    can fix it mechanically."""
    from normalize_spacing import normalize_line
    for line in pagelib.printed_lines(page):
        fixed = normalize_line(line.text)
        if fixed != line.text:
            problems.append(f"{line.where}: punctuation spacing (§1); expected {fixed!r} "
                            f"(run scripts/normalize_spacing.py on the file)")


def check_nfc(page, warnings):
    for line in pagelib.all_lines(page):
        if line.text != pagelib.nfc(line.text):
            warnings.append(f"{line.where}: text is not in NFC "
                            f"(use precomposed ã ẽ ĩ õ ũ, §2)")


def _uncertain_covers(page, line):
    """An entry covers the line only if it points at it, or quotes it.

    A block-level entry (`blocks[1]`) covers every line in that block; a
    sibling line-level entry (`blocks[1].lines[9]`) covers only its own line.
    """
    for entry in page.get("uncertain") or []:
        if not isinstance(entry, dict):
            continue
        where = str(entry.get("where", ""))
        if where == line.where or pagelib.points_at(line.where, where):
            return True
        if entry.get("text") == line.text:
            return True
    return False


def check_uncertain_signs(page, problems):
    entries = page.get("uncertain") or []
    for line in pagelib.all_lines(page):
        signs = [s for s in UNCERTAIN_SIGNS if s in line.text]
        if not signs:
            continue
        shown = ", ".join(signs)
        if not entries:
            problems.append(f"{line.where}: {shown} with no uncertain[] entry at all")
        elif not _uncertain_covers(page, line):
            problems.append(f"{line.where}: {shown} with no uncertain[] entry "
                            f"pointing at it")


def check_folio(page, manifest, problems):
    rec = pagelib.manifest_record(manifest, page.get("id"))
    if rec is None or "folio" not in rec:
        return
    expected, got = rec.get("folio"), page.get("folio")
    if expected == got:
        return
    if any("folio" in blob.lower() for blob in pagelib.uncertain_blobs(page)):
        return
    problems.append(f"folio {got!r} does not match the manifest's folio {expected!r} "
                    f"and no uncertain[] entry mentions the folio")


def check_long_s(page, warnings):
    for line in pagelib.printed_lines(page):
        tokens = [w.lower() for w in WORD_RE.findall(pagelib.nfc(line.text))]
        hits = sorted({t for t in tokens if t in LONG_S_WORDS},
                      key=lambda t: tokens.index(t))
        if hits:
            warnings.append(f'{line.where}: possible normalized long s '
                            f'({", ".join(hits)}) in "{line.text}"')


def check_id(page, path, problems):
    stem = pathlib.Path(path).stem
    if page.get("id") != stem:
        problems.append(f"id {page.get('id')!r} does not match the filename stem {stem!r}")


# --- driver ---------------------------------------------------------------

def validate_file(path, manifest):
    """Return (problems, warnings) for one file."""
    problems, warnings = [], []
    try:
        page = pagelib.load_page(path)
    except pagelib.PageLoadError as exc:
        return [f"cannot read: {exc.reason}"], []

    ok = check_schema(page, problems)
    check_id(page, path, problems)
    if ok:
        # the structural checks assume the schema held
        check_markers(page, problems)
        check_whitespace(page, problems)
        check_punct_spacing(page, problems)
        check_uncertain_signs(page, problems)
        check_folio(page, manifest, problems)
        check_nfc(page, warnings)
        check_long_s(page, warnings)
    return problems, warnings


def collect_paths(args):
    """Return (paths, directory_problems)."""
    roots = []
    if args.all_final:
        roots.append(pathlib.Path("transcription/final"))
    if args.all_reads:
        roots.append(pathlib.Path("transcription/reads"))
    if not roots:
        return [pathlib.Path(p) for p in args.paths], []

    paths, problems = [], []
    for root in roots:
        if not root.is_dir():
            problems.append((root, "directory not found"))
            continue
        found = list(pagelib.iter_pages(root))
        if not found:
            problems.append((root, "no page JSON files found"))
        paths.extend(found)
    return paths, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate transcribed page JSON.")
    ap.add_argument("paths", nargs="*", help="page JSON files")
    ap.add_argument("--all-final", action="store_true",
                    help="validate every file in transcription/final/")
    ap.add_argument("--all-reads", action="store_true",
                    help="validate every file under transcription/reads/")
    ap.add_argument("--manifest", default="manifest.json",
                    help="manifest to check folios against (default: manifest.json)")
    args = ap.parse_args(argv)

    if not args.paths and not (args.all_final or args.all_reads):
        ap.error("give at least one path, or --all-final / --all-reads")
    if args.paths and (args.all_final or args.all_reads):
        ap.error("--all-final / --all-reads take no paths; give one or the other")

    try:
        manifest = pagelib.load_manifest(args.manifest)
    except pagelib.PageLoadError as exc:
        print(f"{exc.path}: PROBLEM: cannot read the manifest: {exc.reason}")
        return 1

    paths, dir_problems = collect_paths(args)
    for root, detail in dir_problems:
        print(f"{root}: PROBLEM: {detail}")

    ok = failed = 0
    for path in paths:
        problems, warnings = validate_file(path, manifest)
        for w in warnings:
            print(f"{path}: WARNING: {w}")
        for p in problems:
            print(f"{path}: PROBLEM: {p}")
        if problems:
            failed += 1
        else:
            ok += 1

    summary = f"{ok} ok, {failed} failed"
    if dir_problems:
        n = len(dir_problems)
        summary += f" ({n} directory problem{'s' if n > 1 else ''})"
    print(summary)
    return 1 if (failed or dir_problems) else 0


if __name__ == "__main__":
    sys.exit(main())

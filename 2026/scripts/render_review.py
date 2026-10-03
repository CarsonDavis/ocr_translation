#!/usr/bin/env python3
"""Render the whole book, French and English side by side, as one markdown file for a
reviewer that reads everything in a single pass.

    uv run python scripts/render_review.py --out PATH [--quarters N]
    uv run python scripts/render_review.py --ids

Every section of text/sections.json, in book order, becomes

    ## <id> — <kind> <label> — pages pNNN–pNNN
    ### French
    <the section's text verbatim: ⟦pNNN⟧, {x} and ⟨alt…⟩ markers intact>
    ### French notes          (only when the section has notes)
    {x} <note text>           (one per note, in letter order page by page; unkeyed: {_})
    ### English
    <translation/sections/<id>.md without its front matter and its "## Notes" part,
     or "(not translated)">

A section with complete=false is left out and named in the header. The header gives the
counts (sections, French words, English words) and, with --quarters N, a table of N
contiguous runs of sections (split by section count) so a prompt can name a quarter; the
table is printed to stdout too, with a token estimate of the output (French chars / 3.2,
everything else chars / 4). --ids prints the rendered section ids, one per line, and
writes nothing.
"""
import argparse, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STRIP = re.compile(r"⟦[^⟧]*⟧|⟨alt\??:[^⟩]*⟩|\{[^}]*\}")
FRENCH_CPT, ENGLISH_CPT = 3.2, 4.0


def load_sections(root=ROOT):
    return json.loads((root / "text/sections.json").read_text(encoding="utf-8"))["sections"]


def words(text):
    return len(STRIP.sub(" ", text).split())


def english_body(root, sid):
    """The translation's body (no front matter, nothing from "## Notes" on), or None."""
    path = root / "translation/sections" / f"{sid}.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.match(r"---\n.*?\n---\n", text, re.S)
    if m:
        text = text[m.end():]
    if text.startswith("## Notes"):
        return ""
    return text.partition("\n## Notes")[0].strip()


def note_lines(sec):
    """`{x} text` per note, in the section's order (page by page, letters in order;
    letters restart on each page, so a key may repeat). Unkeyed notes are `{_}`."""
    return [f"{{{n.get('key') or '_'}}} {(n.get('text') or '').strip()}"
            for n in sec.get("notes") or []]


def heading(sec):
    label = sec.get("label") or (str(sec["number"]) if sec.get("number") is not None else "")
    pages = sec.get("pages") or []
    span = f"pages {pages[0]}–{pages[-1]}" if pages else "no pages"
    name = f"{sec['kind']} {label}".strip()
    return f"## {sec['id']} — {name} — {span}"


def quarters(ids, n):
    """N contiguous runs of ids, sizes differing by at most one (earlier runs larger)."""
    n = max(1, min(n, len(ids))) if ids else 0
    base, extra = divmod(len(ids), n) if n else (0, 0)
    out, i = [], 0
    for q in range(n):
        size = base + (1 if q < extra else 0)
        out.append((q + 1, ids[i], ids[i + size - 1], size))
        i += size
    return out


def quarter_table(rows):
    lines = ["| quarter | first | last | sections |", "|---|---|---|---|"]
    lines += [f"| {q} | {a} | {b} | {n} |" for q, a, b, n in rows]
    return "\n".join(lines)


def render(root=ROOT, n_quarters=None):
    """Returns (markdown, stats) where stats has ids, skipped, quarters, counts and
    the token estimate."""
    sections = load_sections(root)
    kept = [s for s in sections if s.get("complete", True)]
    skipped = [s["id"] for s in sections if not s.get("complete", True)]
    body, fr_chars, fr_words, en_words, missing = [], 0, 0, 0, []
    for sec in kept:
        french = sec.get("text") or ""
        notes = note_lines(sec)
        en = english_body(root, sec["id"])
        if en is None:
            missing.append(sec["id"])
        fr_words += words(french) + sum(words(n) for n in notes)
        en_words += words(en or "")
        part = [heading(sec), "### French", french]
        fr_chars += len(french)
        if notes:
            part += ["### French notes", "\n".join(notes)]
            fr_chars += sum(len(n) + 1 for n in notes)
        part += ["### English", en if en is not None else "(not translated)"]
        body.append("\n\n".join(part))

    ids = [s["id"] for s in kept]
    rows = quarters(ids, n_quarters) if n_quarters else []
    head = ["# Arrest Memorable — whole-book review text",
            "",
            f"{len(kept)} sections, {fr_words} French words, {en_words} English words."]
    if skipped:
        head.append(f"Skipped (complete=false, not rendered): {', '.join(skipped)}.")
    if missing:
        head.append(f"Not translated: {', '.join(missing)}.")
    if rows:
        head += ["", f"Quarters ({len(rows)} contiguous runs by section count):", "",
                 quarter_table(rows)]
    md = "\n".join(head) + "\n\n" + "\n\n".join(body) + "\n"
    en_chars = len(md) - fr_chars
    tokens = round(fr_chars / FRENCH_CPT + en_chars / ENGLISH_CPT)
    return md, {"ids": ids, "skipped": skipped, "missing": missing, "quarters": rows,
                "sections": len(kept), "french_words": fr_words, "english_words": en_words,
                "french_chars": fr_chars, "english_chars": en_chars, "tokens": tokens}


def main(argv=None, root=ROOT, out=print):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="markdown file to write")
    ap.add_argument("--quarters", type=int, metavar="N", help="add a table of N contiguous runs")
    ap.add_argument("--ids", action="store_true", help="print the section ids only")
    a = ap.parse_args(argv)
    if a.ids:
        for s in load_sections(root):
            if s.get("complete", True):
                out(s["id"])
        return 0
    if not a.out:
        ap.error("--out is required (or use --ids)")
    if a.quarters is not None and a.quarters < 1:
        ap.error("--quarters must be at least 1")
    md, st = render(root, a.quarters)
    path = pathlib.Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    out(f"wrote {path}: {st['sections']} sections, {st['french_words']} French words, "
        f"{st['english_words']} English words")
    if st["skipped"]:
        out(f"skipped (complete=false): {', '.join(st['skipped'])}")
    if st["missing"]:
        out(f"not translated: {', '.join(st['missing'])}")
    if st["quarters"]:
        out(quarter_table(st["quarters"]))
    out(f"~{st['tokens']} tokens ({st['french_chars']} French chars / {FRENCH_CPT}, "
        f"{st['english_chars']} other chars / {ENGLISH_CPT:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

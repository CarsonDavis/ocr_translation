#!/usr/bin/env python3
"""Stitch the finished page transcriptions into logical sections.

    uv run python scripts/stitch_text.py [--check] [--sections] [--dry-run] [--out PATH]

Reads `manifest.json` and `transcription/final/*.json` and writes
`text/sections.json`: the book as an ordered list of sections (title, argument,
`texte-NN`, `annot-NNN`) whose `text` is the diplomatic transcription reflowed
into paragraphs, with the `{x}` letter markers kept verbatim and a `⟦pNNN⟧`
marker at the start of every section's contribution on every page (so a page
shared by two sections is marked in both, and a section lying wholly inside one
page still opens with that page's marker).

The walk works on a prefix: pages are taken in manifest order and stop at the
first page whose `status.final` is not `done` or whose final file is missing. A
section that is still open at the stop is emitted with `"complete": false` so
translation can skip it.

Alongside it, `text/alts.json` (beside the `--out` file) lists one record per
inline `⟨alt:…⟩` / `⟨alt?:…⟩` marker the stitch emitted: `alt_id`
(`<page>-<where slug>-<n>`, e.g. `p071-m2l5-1` for the first marker on
`margin_notes[2].lines[5]`; `b` = blocks, `m` = margin_notes, `f` = foot_notes),
`section`, `page`, `where`, `kind` (`alt` / `alt?`), `a` and `b` (the full line
readings), `marker` (the exact string as it stands in the section text or
note) and `context` (a few words either side). The translator reports its
choice per `alt_id`; scripts/apply_translator_choices.py feeds it back.

`--check` also validates (a section's page markers are unique, in manifest order
and equal to its `pages`; every consumed page is marked by at least one section;
section ids are unique) and exits 1 on errors. Orphan and missing notes are
reported as warnings, not errors. `--sections` lists the sections one per line.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pagelib  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEEP_PATH = ROOT / "scripts/hyphen_keep.txt"

# Pages that are a section all by themselves, keyed by page id.
SPECIAL = {"p000-title": "title", "p000-argument": "argument"}

PAGE_MARKER = re.compile(r"⟦([^⟧]+)⟧")
TEXTE_RE = re.compile(r"^TEXTE\s*[.,:;]?$")
ANNOT_RE = re.compile(r"^ANNOT(?:ATIONS|ATION|AT)?(?![A-Z])\s*\.?\s*(.*)$")
# A capitalised word, then the rest: for wrong-sort abbreviations ("ANNNT. LX." on p080)
FUZZY_HEAD_RE = re.compile(r"^([A-Z]{4,12})\s*[.,]?\s*([IVXLCDM]+)\s*[.,]$")
ANNOT_FORMS = ("ANNOT", "ANNOTAT", "ANNOTATION", "ANNOTATIONS")
# A capitalised word and a stop: for wrong-sort TEXTE headings ("TFXTE." p045,
# "TBXTE." p058, "TEXTB." p072)
FUZZY_TEXTE_RE = re.compile(r"^([A-Z]{4,6})\s*[.,]$")
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


# --- reflow ---------------------------------------------------------------

def fold(word: str) -> str:
    """The lookup form of a word: lowercased, letters only."""
    return "".join(c for c in word.lower() if c.isalpha())


def load_keep(path=None) -> set[str]:
    """The hyphen keep-list: compounds whose hyphen survives the line break."""
    path = pathlib.Path(path or KEEP_PATH)
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(fold(line))
    return out


def join_pair(prev: str, nxt: str, keep=frozenset()) -> tuple[str, str]:
    """How to glue printed line `prev` to printed line `nxt`.

    Returns `(prev, separator)`: the caller writes `prev + separator + nxt`
    (a page marker, when there is one, goes between the separator and `nxt`).
    A line ending in `-` joins the next without a space and without the hyphen,
    unless the next line opens with a capital or the joined word is kept.
    """
    prev, nxt = prev.rstrip(), nxt.lstrip()
    if not prev or not nxt:
        return prev, ""
    if not prev.endswith("-"):
        return prev, " "
    stem = prev[:-1]
    tail = stem.split()[-1] if stem.split() else ""
    if nxt[:1].isupper() or fold(tail + nxt.split()[0]) in keep:
        return prev, ""          # a real hyphen: keep it
    return stem, ""              # a word broken by the compositor


def reflow(lines, keep=frozenset()) -> str:
    """One paragraph (or one note) as a single string."""
    out = ""
    for line in lines:
        if not isinstance(line, str):
            continue
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        head, sep = join_pair(out, line, keep)
        out = head + sep + line
    return out


# --- alternative readings from human arbitration ---------------------------

ALT_PREFIX = "arbitration: undecided; alternatives: "
# "unknown": the transcribers vouch for neither reading; rendered ⟨alt?:…⟩
ALT_PREFIX_UNKNOWN = "arbitration: unknown; alternatives: "
ALT_SEP = " ||| "
_POINTER = re.compile(r"^(blocks|margin_notes|foot_notes)\[(\d+)\]\.lines\[(\d+)\]$")


def alt_markup(a: str, b: str, tag: str = "alt") -> str:
    """Reading A with reading B's differing words attached as ⟨alt:…⟩ markers.

    `trabir` vs `trahir` becomes `trabir⟨alt:trahir⟩`; a word only A has gets
    `⟨alt:⟩`, a word only B has appears as `⟨alt:+word⟩`. The translator picks by
    context (scripts/prompts/translate.md). With tag="alt?" the markers read
    `⟨alt?:…⟩` (an "unknown" decision: neither reading is confirmed).
    """
    return " ".join(chunk for chunk, _ in alt_chunks(a, b, tag))


def alt_chunks(a: str, b: str, tag: str = "alt") -> list[tuple[str, bool]]:
    """The pieces of `alt_markup`'s output as `(text, is_marker)`, joined by spaces."""
    import difflib
    mark = tag
    ta, tb = a.split(), b.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if tag == "equal":
            out.extend((t, False) for t in ta[i1:i2])
        elif tag == "replace":
            out.append((" ".join(ta[i1:i2]) + f"⟨{mark}:" + " ".join(tb[j1:j2]) + "⟩", True))
        elif tag == "delete":
            out.append((" ".join(ta[i1:i2]) + f"⟨{mark}:⟩", True))
        else:  # insert
            out.append((f"⟨{mark}:+" + " ".join(tb[j1:j2]) + "⟩", True))
    return out


def where_slug(where: str) -> str:
    """`blocks[2].lines[5]` -> `b2l5`; `margin_notes[3].lines[0]` -> `m3l0`."""
    m = _POINTER.match(where)
    return f"{m.group(1)[0]}{m.group(2)}l{m.group(3)}" if m else re.sub(r"\W+", "", where)


def mark_alternatives(page: dict, record=None) -> dict:
    """A copy of the page whose undecided lines carry ⟨alt:…⟩ markers.

    Lines are found through the uncertain[] entries that apply_arbitration.py writes
    for an "either" decision (note = ALT_PREFIX + A + ALT_SEP + B, rendered ⟨alt:…⟩)
    and for an "unknown" decision (ALT_PREFIX_UNKNOWN, rendered ⟨alt?:…⟩).
    With a list as `record`, appends one dict per rewritten line that carries at
    least one marker: field, block/note index, line index, where, kind, a, b and
    `markers` (the marker strings in line order).
    """
    import copy
    todo = {}
    for u in page.get("uncertain") or []:
        note = u.get("note") or ""
        m = _POINTER.match(u.get("where") or "")
        prefix = next((p for p in (ALT_PREFIX, ALT_PREFIX_UNKNOWN) if note.startswith(p)), None)
        if prefix is None or not m or ALT_SEP not in note:
            continue
        a, b = note[len(prefix):].split(ALT_SEP, 1)
        tag = "alt?" if prefix == ALT_PREFIX_UNKNOWN else "alt"
        todo[(m.group(1), int(m.group(2)), int(m.group(3)))] = (a, b, tag)
    if not todo:
        return page
    page = copy.deepcopy(page)
    for (field, i, j), (a, b, tag) in todo.items():
        try:
            lines = page[field][i]["lines"]
            if lines[j].strip() == a.strip():
                chunks = alt_chunks(a, b, tag)
                lines[j] = " ".join(c for c, _ in chunks)
                markers = [c for c, is_marker in chunks if is_marker]
                if record is not None and markers:
                    record.append({"field": field, "index": i, "line": j,
                                   "where": f"{field}[{i}].lines[{j}]", "kind": tag,
                                   "a": a, "b": b, "markers": markers})
        except (KeyError, IndexError, TypeError):
            continue
    return page


# --- headings -------------------------------------------------------------

def roman_value(text: str):
    """The value of a Roman numeral, or None if it is not one.

    Additive forms the book uses (IIII, XLIIII, XCIX, CXI) all work.
    """
    text = text.strip().upper()
    if not text or any(c not in ROMAN for c in text):
        return None
    total, values = 0, [ROMAN[c] for c in text]
    for i, v in enumerate(values):
        total += -v if i + 1 < len(values) and v < values[i + 1] else v
    return total


def within_one_edit(a: str, b: str) -> bool:
    """True if `a` becomes `b` by at most one substitution, insertion or deletion."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    short, long_ = sorted((a, b), key=len)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def parse_heading(text: str):
    """`("texte", None)`, `("annotation", n_or_None)`, or None for a display line."""
    text = (text or "").strip()
    if TEXTE_RE.match(text):
        return "texte", None
    # A wrong-sort TEXTE (one letter wrong, missing or extra) counts only with a stop.
    f = FUZZY_TEXTE_RE.match(text)
    if f and within_one_edit(f.group(1), "TEXTE"):
        return "texte", None
    m = ANNOT_RE.match(text)
    if not m:
        # A wrong-sort abbreviation (one letter wrong, missing or extra: "ANNNT. LX.",
        # "ANOTAT. V.") counts only when a Roman numeral and a stop follow it.
        f = FUZZY_HEAD_RE.match(text)
        if f and any(within_one_edit(f.group(1), form) for form in ANNOT_FORMS):
            return "annotation", roman_value(f.group(2))
        return None
    tail = m.group(1).strip()
    if any(c.islower() for c in tail):
        return None              # "ANNOTATIONS de..." is prose, not a heading
    return "annotation", roman_value(tail.strip(" .,")) if tail else None


# --- the walk -------------------------------------------------------------

def walk(manifest, final_dir):
    """Pages in manifest order up to the first one that is not finished.

    Returns `(pages, stopped_at)` where `pages` is a list of `(id, page)` and
    `stopped_at` is the id of the page that stopped the walk (None if none did).
    """
    final_dir = pathlib.Path(final_dir)
    pages = []
    for rec in (manifest or {}).get("pages") or []:
        page_id = rec.get("id")
        path = final_dir / f"{page_id}.json"
        if (rec.get("status") or {}).get("final") != "done" or not path.exists():
            return pages, page_id
        pages.append((page_id, pagelib.load_page(path)))
    return pages, None


# --- stitching ------------------------------------------------------------

def stitch(pages, keep=frozenset(), alts=None):
    """Build the section list from `(page_id, page)` pairs in book order.

    With a list as `alts`, also fills it with the alt-marker records (see
    `locate_alts`); the sections are the same either way."""
    sections, cur = [], None
    alt_lines = []                  # (page_id, line record) from mark_alternatives
    block_home = {}                 # (page_id, block index) -> working section
    texte_n = annot_n = 0
    page_notes = []
    pending_cont = False            # previous page ended on an open paragraph
    pending_hyphen = False          # ...or on a line ending in a break hyphen

    def close(mid):
        nonlocal cur
        if cur is not None:
            cur["ends_mid_page"] = mid
            cur["complete"] = True
            cur = None

    def start(sid, kind, number, label, mid, uncertain=False):
        nonlocal cur
        cur = {"id": sid, "kind": kind, "number": number, "label": label,
               "pages": [], "paras": [], "notes": [], "markers": [], "lines": {},
               "starts_mid_page": mid, "ends_mid_page": False, "complete": False}
        if uncertain:
            cur["number_uncertain"] = True
        sections.append(cur)

    def add(page_id, body, raw_lines, join=False):
        # every section marks every page it draws on, at its first contribution
        mark = ""
        if page_id not in cur["pages"]:
            cur["pages"].append(page_id)
            mark = f"⟦{page_id}⟧"
        cur["lines"].setdefault(page_id, []).extend(raw_lines)
        if join and cur["paras"]:
            head, sep = join_pair(cur["paras"][-1], body, keep)
            cur["paras"][-1] = head + sep + mark + body
        else:
            cur["paras"].append(mark + body)
        for key in pagelib.MARKER_RE.findall(body):
            cur["markers"].append((key, page_id))

    for page_id, page in pages:
        found = []
        page = mark_alternatives(page, found)
        alt_lines.extend((page_id, rec) for rec in found)
        used = False                # has this page contributed text yet?
        armed = pending_cont        # may the first paragraph join the last one?
        hyph = pending_hyphen       # does the last page end mid-word?
        blocks = [(bi, b) for bi, b in enumerate(page.get("blocks") or [])
                  if isinstance(b, dict) and b.get("type") in ("heading", "paragraph")]
        if page_id in SPECIAL:
            close(used)
            start(SPECIAL[page_id], SPECIAL[page_id], None, None, used)
            armed = hyph = False
        open_para = ends_hyphen = False
        for bi, block in blocks:
            if block.get("type") == "heading":
                text = (block.get("text") or "").strip()
                kind = parse_heading(text)
                if kind is None:                      # a display line: keep it
                    if cur is None:
                        texte_n += 1
                        start(f"texte-{texte_n:02d}", "texte", texte_n, None, used)
                    add(page_id, reflow([text], keep), [text])
                    used = True
                elif kind[0] == "texte":
                    close(used)
                    texte_n += 1
                    start(f"texte-{texte_n:02d}", "texte", texte_n, text, used)
                else:
                    close(used)
                    number, uncertain = kind[1], kind[1] is None
                    printed = number
                    # The print misnumbers some annotations (p034 "XIII." for XVIII,
                    # p045 repeats "XXIIII."). The sequence is authoritative: a heading
                    # whose number is not the next expected one is a misprint.
                    if number is None or (annot_n > 0 and number != annot_n + 1):
                        number, uncertain = annot_n + 1, True
                    annot_n = number
                    start(f"annot-{annot_n:03d}", "annotation", annot_n, text,
                          used, uncertain)
                    if printed is not None and printed != annot_n:
                        cur["number_printed"] = printed
                armed, open_para, ends_hyphen, hyph = False, False, False, False
            else:
                lines = [ln for ln in (block.get("lines") or []) if isinstance(ln, str)]
                if cur is None:
                    texte_n += 1
                    start(f"texte-{texte_n:02d}", "texte", texte_n, None, used)
                body = reflow(lines, keep)
                # A page that opens mid-word ("meſme-" | "ment") continues the
                # paragraph whatever the continues_* flags say.
                mid_word = hyph and body[:1].islower()
                join = (mid_word or (armed and bool(block.get("continues_prev")))) \
                    and bool(cur["paras"])
                add(page_id, body, lines, join)
                block_home[(page_id, bi)] = cur
                used, armed, hyph = True, False, False
                open_para = bool(block.get("continues_next"))
                last = next((ln.strip() for ln in reversed(lines) if ln.strip()), "")
                ends_hyphen = last.endswith("-")
        if page_id in SPECIAL:       # a one-page section: it ends with its page
            close(False)
            open_para = ends_hyphen = False
        pending_cont = open_para
        pending_hyphen = ends_hyphen
        page_notes.append((page_id, pagelib.notes(page)))

    note_home = {}
    attach_notes(sections, page_notes, keep, note_home)
    out = [finish(s) for s in sections]
    if alts is not None:
        public = {id(s): rec for s, rec in zip(sections, out)}
        alts.extend(locate_alts(alt_lines, block_home, note_home, public))
    return out


def locate_alts(alt_lines, block_home, note_home, public, words=6):
    """One record per emitted alt marker, with the section it landed in.

    A block line's markers are looked up in its section's text, a note line's in
    that note's text; repeated identical marker strings are taken in book order.
    A marker that cannot be found is left out."""
    out, cursor = [], {}         # cursor: (container key, marker) -> search start
    for page_id, rec in alt_lines:
        if rec["field"] == "blocks":
            sec = block_home.get((page_id, rec["index"]))
            if sec is None:
                continue
            pub = public[id(sec)]
            text, key = pub["text"], ("text", id(sec))
        else:
            hit = note_home.get((page_id, rec["field"], rec["index"]))
            if hit is None:
                continue
            sec, note = hit
            pub = public[id(sec)]
            text, key = note["text"], ("note", id(note))
        slug = f"{page_id}-{where_slug(rec['where'])}"
        for n, marker in enumerate(rec["markers"], 1):
            start = cursor.get((key, marker), 0)
            pos = text.find(marker, start)
            if pos < 0:
                continue
            cursor[(key, marker)] = pos + len(marker)
            before = text[:pos].split()[-words:]
            after = text[pos + len(marker):].split()[:words]
            out.append({"alt_id": f"{slug}-{n}", "section": pub["id"], "page": page_id,
                        "where": rec["where"], "kind": rec["kind"],
                        "a": rec["a"], "b": rec["b"], "marker": marker,
                        "context": " ".join(before + [marker] + after)})
    return out


def attach_notes(sections, page_notes, keep, home=None):
    """Hang each margin/foot note on the section that carries its marker.

    With a dict as `home`, maps (page, note kind, note index) to (section, record)."""
    by_page = {}
    for sec in sections:
        for page_id in sec["pages"]:
            by_page.setdefault(page_id, []).append(sec)
    for page_id, notes in page_notes:
        on_page = by_page.get(page_id, [])
        for note in notes:
            key = note.key if note.key else "_"
            target = None
            if note.key:
                target = next((s for s in on_page
                               if (key, page_id) in s["markers"]), None)
            orphan = target is None
            if target is None and note.beside_line:
                target = next((s for s in on_page
                               if note.beside_line in s["lines"].get(page_id, [])), None)
            if target is None:
                target = on_page[0] if on_page else None
            if target is None:
                continue            # a page that contributed no text at all
            record = {"key": key, "page": page_id, "text": reflow(note.lines, keep)}
            if orphan:
                record["orphan"] = True
            target["notes"].append(record)
            if home is not None:
                home[(page_id, note.kind, note.index)] = (target, record)


def finish(sec) -> dict:
    """The public section record, in field order, with the working keys dropped."""
    have = {(n["key"], n["page"]) for n in sec["notes"]}
    missing, seen = [], set()
    for pair in sec["markers"]:
        if pair not in have and pair not in seen:
            seen.add(pair)
            missing.append(f"{pair[0]}@{pair[1]}")
    out = {"id": sec["id"], "kind": sec["kind"], "number": sec["number"],
           "label": sec["label"], "pages": sec["pages"],
           "text": "\n\n".join(sec["paras"]), "notes": sec["notes"],
           "starts_mid_page": sec["starts_mid_page"],
           "ends_mid_page": sec["ends_mid_page"], "complete": sec["complete"]}
    if sec.get("number_uncertain"):
        out["number_uncertain"] = True
    if sec.get("number_printed") is not None:
        out["number_printed"] = sec["number_printed"]
    if missing:
        out["missing_notes"] = missing
    return out


# --- validation -----------------------------------------------------------

def validate(sections, page_ids):
    """`(errors, warnings)`. Errors are fatal; note problems are not.

    Inside a section the page markers must be unique, in manifest order and
    equal to its `pages`; across all sections every consumed page must be
    marked at least once (a page shared by two sections is marked in both).
    """
    errors, warnings = [], []
    order = {page_id: i for i, page_id in enumerate(page_ids)}
    seen, marked = set(), set()
    for sec in sections:
        if sec["id"] in seen:
            errors.append(f"duplicate section id {sec['id']}")
        seen.add(sec["id"])
        if not sec["text"].strip():
            warnings.append(f"{sec['id']}: empty section")
        found = PAGE_MARKER.findall(sec["text"])
        marked.update(found)
        unknown = [p for p in found if p not in order]
        if unknown:
            errors.append(f"{sec['id']}: marker for unconsumed page "
                          f"{', '.join(sorted(set(unknown)))}")
        elif len(set(found)) != len(found):
            errors.append(f"{sec['id']}: repeated page marker "
                          f"{', '.join(sorted({p for p in found if found.count(p) > 1}))}")
        elif [order[p] for p in found] != sorted(order[p] for p in found):
            errors.append(f"{sec['id']}: page markers are not in manifest order "
                          f"({', '.join(found)})")
        elif found != list(sec["pages"]):
            errors.append(f"{sec['id']}: page markers {found} do not match "
                          f"pages {sec['pages']}")
        for note in sec["notes"]:
            if note.get("orphan"):
                warnings.append(
                    f"{sec['id']}: orphan note {{{note['key']}}} on {note['page']}")
        for item in sec.get("missing_notes", []):
            warnings.append(f"{sec['id']}: marker {{{item.split('@')[0]}}} "
                            f"has no note ({item})")
    unmarked = [p for p in page_ids if p not in marked]
    if unmarked:
        errors.append(f"no section marks {', '.join(unmarked)}")
    return errors, warnings


def section_line(sec) -> str:
    """One line per section for `--sections`."""
    return (f"{sec['id']:<10} {','.join(sec['pages']):<34} "
            f"{'complete' if sec['complete'] else 'INCOMPLETE':<10} "
            f"{len(sec['notes'])} notes")


# --- cli ------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="validate and exit 1 on errors")
    ap.add_argument("--sections", action="store_true",
                    help="print one line per section: id, pages, complete, notes")
    ap.add_argument("--dry-run", action="store_true", help="do not write the file")
    ap.add_argument("--root", default=ROOT, help="project root")
    ap.add_argument("--out", default=None, help="output path")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.root)
    out = pathlib.Path(args.out) if args.out else root / "text/sections.json"
    manifest = pagelib.load_manifest(root / "manifest.json") or {"pages": []}
    pages, stopped = walk(manifest, root / "transcription/final")
    alts = []
    sections = stitch(pages, load_keep(), alts)

    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        generated = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        payload = {"generated": generated, "sections": sections}
        out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        alts_payload = {"generated": generated, "sections_file": out.name, "alts": alts}
        (out.parent / "alts.json").write_text(
            json.dumps(alts_payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.sections:
        for sec in sections:
            print(section_line(sec))
    complete = sum(1 for s in sections if s["complete"])
    print(f"{len(pages)} pages consumed, {len(sections)} sections "
          f"({complete} complete), stopped at {stopped or 'end of manifest'}; "
          f"{len(alts)} alt markers")
    if not args.check:
        return 0
    errors, warnings = validate(sections, [p for p, _ in pages])
    for warning in warnings:
        print("WARNING:", warning)
    for error in errors:
        print("PROBLEM:", error)
    print(f"{len(errors)} problems, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

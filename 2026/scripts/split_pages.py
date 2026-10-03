#!/usr/bin/env python3
"""Turn the manifest, the finished transcriptions and the translation into the viewer's data.

    uv run --with jsonschema python scripts/split_pages.py [--root DIR] [--out DIR]

Reads `manifest.json`, `transcription/final/<id>.json`, `text/sections.json` and
`translation/sections/<id>.md` (never writes any of them) and writes, under `--out`
(default `<root>/site/data`):

    book.json          the book-level constants the viewer and the landing card need
    index.json         one short record per page, in reading order
    pages/<id>.json    the layers of one page, per scripts/site_schema.json

A page whose final is missing or not yet marked done gets `french: null`; a page no
translated section covers gets `english: null`; `index.json` records which layers a
page has. Every page record is validated against scripts/site_schema.json before
anything is written, so a bad run leaves the previous data in place.

The section files are written by the translator to the format in
scripts/prompts/translate.md, which is the authority on it; this script follows it.
docs/site-data-contract.md describes both ends.
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
import sys
from typing import NamedTuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pagelib  # noqa: E402

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
SITE_SCHEMA_PATH = SCRIPTS_DIR / "site_schema.json"
DEFAULT_ROOT = SCRIPTS_DIR.parent

CUDL_ITEM = "https://cudl.lib.cam.ac.uk/view/PR-MONTAIGNE-00001-00007-00022"
# p041 is missing from the Cambridge scan; its image comes from the Gallica copy.
GALLICA_PAGE = "https://gallica.bnf.fr/ark:/12148/bpt6k52469j/f58"

BOOK_DESCRIPTION = (
    "Jean de Coras's 1572 account of the Martin Guerre case: the arrest of the "
    "Parlement de Tholose on the man who came back to Artigat as another man's "
    "husband, printed with Coras's hundred and eleven annotations. A diplomatic, "
    "line-by-line transcription of the French and an English translation, both "
    "still in progress, from the Cambridge University Library scans."
)

# The viewer's About box, after the description: one plain paragraph per string.
BOOK_ABOUT = [
    "Each page of the French was transcribed in two independent passes. A dotted "
    "underline in the French marks a reading where the two passes disagreed; "
    "hover over it, or tap it, to see both readings and how the difference was "
    "resolved. Each was decided by the editor; by the reconciliation model, the "
    "language model that settled the two passes' differences in the first run "
    "over the early pages; by the translation model, the language model that "
    "writes the English, choosing from context the reading that fits the "
    "sentence; or is not yet decided, and the text shows the first pass's "
    "version, reader A. Every "
    "contested reading on a page, including those about layout that have no "
    "single word to underline, is listed under Readings below the page's French "
    "text; the contested readings button in the header turns both off.",
]

# `ANNOTAT. V.` / `ANNOT. XXIIII.` / `ANNOTATION I.` are all the same heading in
# this print; the site spells it out. The numeral is roman, upper or lower case.
ANNOTATION_RE = re.compile(r"^ANNOT(?:AT)?(?:ION)?\.?\s*([IVXLC]+)\.?$", re.IGNORECASE)
HEADING_WORDS = {"TEXTE": "TEXT", "ARGVMENT": "ARGUMENT"}
# Most annotations open with the quoted TEXTE they annotate, so the jump menu wants
# the annotation's own heading, not the word TEXT.
GENERIC_HEADING = "TEXT"
# The front matter has no annotation number; its headings are the title-page lines.
FRONT_PAGES = ("p000-title", "p000-argument")

UNCERTAIN_KEYS = ("where", "text", "note")

# --- translation sections (scripts/prompts/translate.md) ------------------

# ⟦p043⟧ / ⟦p000-title⟧, as scripts/check_markers.py matches them.
PAGE_MARKER_RE = re.compile(r"⟦([^⟧]*)⟧")
PAGE_TOKEN_RE = re.compile(r"^p\.?\s*(\d+)$", re.IGNORECASE)
FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
NOTES_FENCE = "\n## Notes"
# - {a} (p002): **Seneca, *On Benefits* 4.2** — Seneque au liu. des benefices. [gloss]
# An aside may sit between the page and the citation, as in
# `- {t} (p007) — orphan note, no marker in the body: **Digest 34.5.9** — L. qui duos`.
# `{_}` is a margin note the print sets with no key (the final's `key: null`), and a
# capitalised label in place of the key — `- Verse (p062): **…** — …` — is a long
# quotation the translator moved into the notes; both are notes with no key.
# A keyed line with only an aside and no citation — `- {c} (p044) — marker with no
# note in the margin.` — records a printed marker that has no note, and is no note.
NOTE_LINE_RE = re.compile(
    r"^-\s*(?:\{([A-Za-zſ]+\d*|_)\}|([A-Z][a-z]+))\s*\(\s*([^)]+?)\s*\)\s*(.*)$")
UNKEYED = "_"
NOTE_NONE_RE = re.compile(r"^-\s*\(\s*none\b.*\)\s*$", re.IGNORECASE)
CITATION_RE = re.compile(r"^\*\*(.+)\*\*\s*(?:[—–-]\s*(.*))?$")
GLOSS_RE = re.compile(r"\s*\[(.+)\]\s*$")
ITALIC_RE = re.compile(r"\*([^*\n]+)\*")
# A paragraph the translator set as a heading in its own right.
HEADING_PARAGRAPH_RE = re.compile(
    r"^(TEXT|ANNOTATION [IVXLC]+|ARGUMENT(?: AND SUMMARY OF THE FACTS)?\.?)$")
ROMAN_STEPS = ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"),
               (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
TEXTE_KINDS = ("texte", "text")
ANNOTATION_KINDS = ("annotation", "annot")


def _warn(message: str) -> None:
    print(f"split_pages: WARNING: {message}", file=sys.stderr)


class SectionNote(NamedTuple):
    """One sidenote of the English layer, from a `## Notes` line."""

    key: str | None
    page: str
    citation: str
    original: str
    gloss: str | None


class Piece(NamedTuple):
    """One run of English prose that belongs to a single page.

    `page` is the page a `⟦…⟧` marker switched to, or None for the start of a
    paragraph that stays on the page the previous paragraph ended on. `continued`
    is True when the marker sat mid-paragraph — mid-word, even — so the prose runs
    on from the page before.
    """

    page: str | None
    text: str
    continued: bool


class Section(NamedTuple):
    """One file of `translation/sections/`, with its `text/sections.json` record."""

    id: str
    pages: list[str]
    pieces: list[Piece]
    notes: list[SectionNote]
    path: pathlib.Path
    kind: str | None = None
    number: int | None = None
    heading: str | None = None

    @property
    def first_page(self) -> str | None:
        return next((p.page for p in self.pieces if p.page is not None), None)


# --- headings -------------------------------------------------------------

def normalize_heading(text: str) -> str:
    """The printed French heading as the site says it.

    `ANNOTAT. V.` -> `ANNOTATION V`, `TEXTE.` -> `TEXT`, `ARGVMENT.` -> `ARGUMENT`;
    anything else only loses its trailing period(s) and surrounding whitespace, so
    a heading this function has never seen still reads as printed.
    """
    stripped = pagelib.nfc(text).strip()
    m = ANNOTATION_RE.match(stripped)
    if m:
        return f"ANNOTATION {m.group(1).upper()}"
    bare = stripped.rstrip(".").strip()
    return HEADING_WORDS.get(bare.upper(), bare)


def roman(number: int) -> str:
    """4 -> IV, 9 -> IX, 24 -> XXIV. The annotations run to 111."""
    if number < 1:
        raise ValueError(f"no roman numeral for {number}")
    out = []
    left = number
    for value, numeral in ROMAN_STEPS:
        while left >= value:
            out.append(numeral)
            left -= value
    return "".join(out)


def section_heading(kind, number, label=None) -> str | None:
    """The heading the English layer prints at the head of a section.

    The stitch step strips the printed heading blocks out of the French section
    text, so neither the French `text` nor the translation carries them; the
    section record is what is left to say which heading this is. Its `label` is the
    heading as the book prints it, so it wins — `ANNOTAT. IIII.` reaches the reader
    as `ANNOTATION IIII`, the same as in the jump menu — and only a section without
    one is named from its kind and number. The title page and the argument print
    their own display lines, so they get none.
    """
    if label:
        return normalize_heading(label)
    if kind in TEXTE_KINDS:
        return GENERIC_HEADING
    if kind in ANNOTATION_KINDS:
        return f"ANNOTATION {roman(number)}" if number else "ANNOTATION"
    return None


def page_headings(final: dict | None) -> list[str]:
    """Every heading printed on the page, normalized, in order."""
    if final is None:
        return []
    return [normalize_heading(t) for t in pagelib.headings(final)
            if isinstance(t, str) and t.strip()]


def first_heading(final: dict | None) -> str | None:
    """The first heading printed on the page, normalized; None if there is none."""
    return next(iter(page_headings(final)), None)


def index_heading(manifest_rec: dict, final: dict | None) -> str | None:
    """What the jump menu shows for this page.

    A page that opens with the quoted `TEXTE.` and then carries `ANNOTAT. VI.` is
    listed under the annotation, since that is what a reader is looking for; the
    front matter is listed by name, not by its title-page lines.
    """
    if manifest_rec.get("id") in FRONT_PAGES:
        return None
    headings = page_headings(final)
    if not headings:
        return None
    for heading in headings:
        if heading != GENERIC_HEADING:
            return heading
    return headings[0]


# --- the English layer ----------------------------------------------------

def page_id_for_marker(token: str) -> str:
    """`p043` is already an id; `p.43` and `p43` are accepted and normalized."""
    stripped = token.strip()
    m = PAGE_TOKEN_RE.match(stripped)
    if m:
        return f"p{int(m.group(1)):03d}"
    return stripped


def to_html(text: str) -> str:
    """One paragraph of English prose as the viewer's HTML.

    Whitespace collapsed, HTML escaped, `*x*` italicised, and `{a}` turned into the
    marker the viewer ties to its sidenote.
    """
    collapsed = " ".join(text.split())
    escaped = html.escape(collapsed, quote=False)
    italicised = ITALIC_RE.sub(r"<i>\1</i>", escaped)
    return pagelib.MARKER_RE.sub(
        lambda m: f'<sup class="mk" data-key="{m.group(1)}">{m.group(1)}</sup>',
        italicised)


def _rich(text: str) -> str:
    """Escaped, with `*x*` italicised: for a citation or a gloss."""
    return ITALIC_RE.sub(r"<i>\1</i>", html.escape(text.strip(), quote=False))


def marker_keys(text: str) -> list[str]:
    """The `{a}` keys of one paragraph, in printed order, without repeats."""
    out = []
    for m in pagelib.MARKER_RE.finditer(text):
        if m.group(1) not in out:
            out.append(m.group(1))
    return out


def _front_matter(label: str, text: str) -> tuple[dict, str]:
    """The `---` block check_markers.py reads, and the body after it."""
    m = FRONT_MATTER_RE.match(text)
    if not m:
        raise ValueError(f"{label}: no `---` front matter")
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"{label}: front matter line is not `key: value`: "
                             f"{line.strip()!r}")
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, text[m.end():]


def _note_line(label: str, line: str) -> SectionNote | None:
    """One `- {a} (p002): **citation** — original [gloss]` line; None for `(none…)`."""
    if NOTE_NONE_RE.match(line.strip()):
        return None
    head = NOTE_LINE_RE.match(line.strip())
    if not head:
        raise ValueError(f"{label}: note line is not "
                         f"`- {{key}} (page): **citation** — original [gloss]`: "
                         f"{line.strip()[:80]!r}")
    key = head.group(1) if head.group(1) not in (None, UNKEYED) else None
    page, rest = page_id_for_marker(head.group(3)), head.group(4)
    # Anything between the page and the bold citation is an aside to the reviewer
    # ("orphan note, no marker in the body"); the site has nowhere to put it.
    start = rest.find("**")
    if start < 0 and head.group(1) not in (None, UNKEYED) and re.match(r"[—–-]", rest):
        # A printed marker the margin has no note for: nothing to show.
        return None
    body = CITATION_RE.match(rest[start:].strip()) if start >= 0 else None
    if not body:
        raise ValueError(f"{label}: note {{{key}}} ({page}) has no **citation**: "
                         f"{rest.strip()[:80]!r}")
    tail = (body.group(2) or "").strip()
    gloss = None
    found = GLOSS_RE.search(tail)
    if found:
        gloss = found.group(1)
        tail = tail[:found.start()]
    return SectionNote(key=key, page=page, citation=_rich(body.group(1)),
                       original=html.escape(tail.strip(), quote=False),
                       gloss=_rich(gloss) if gloss else None)


def _notes(label: str, text: str) -> list[SectionNote]:
    """The `## Notes` block.

    A line that is not a new `- {key} (page)` entry continues the one above it; the
    block may open with a paragraph addressed to the reviewer (what the print
    mis-keys, say), which the site does not show.
    """
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        if raw.lstrip().startswith("-"):
            lines.append(raw.strip())
        elif lines:
            lines[-1] = f"{lines[-1]} {raw.strip()}"
    return [note for note in (_note_line(label, line) for line in lines)
            if note is not None]


def _pieces(label: str, body: str) -> list[Piece]:
    """The prose, cut into the runs that belong to each page."""
    pieces: list[Piece] = []
    seen_marker = False
    for paragraph in re.split(r"\n\s*\n", body):
        if not paragraph.strip():
            continue
        parts = PAGE_MARKER_RE.split(paragraph)
        head = " ".join(parts[0].split())
        seen_text = False
        if head:
            if not seen_marker:
                raise ValueError(f"{label}: prose before the first ⟦page⟧ marker: "
                                 f"{head[:40]!r}")
            pieces.append(Piece(None, head, False))
            seen_text = True
        for i in range(1, len(parts), 2):
            seen_marker = True
            run = " ".join(parts[i + 1].split())
            pieces.append(Piece(page_id_for_marker(parts[i]), run,
                                seen_text and bool(run)))
            if run:
                seen_text = True
    if not seen_marker:
        raise ValueError(f"{label}: no ⟦page⟧ marker anywhere in the file")
    return pieces


def parse_section(path) -> Section:
    """One `translation/sections/<id>.md` -> its prose pieces and its notes.

    `kind`, `number` and the synthesized `heading` come from `text/sections.json`
    and are filled in by `load_sections`.
    """
    path = pathlib.Path(path)
    text = pagelib.nfc(path.read_text(encoding="utf-8"))
    meta, body = _front_matter(path.name, text)
    section_id = meta.get("id")
    if not section_id:
        raise ValueError(f"{path.name}: front matter has no `id`")
    if section_id != path.stem:
        raise ValueError(f"{path.name}: front matter id {section_id!r} does not match "
                         f"the file name")
    if "pages" not in meta:
        raise ValueError(f"{path.name}: front matter has no `pages`")
    pages = [p.strip() for p in meta["pages"].strip().strip("[]").split(",") if p.strip()]
    if not pages:
        raise ValueError(f"{path.name}: front matter `pages` is empty")
    prose, _, notes_text = body.partition(NOTES_FENCE)
    return Section(id=section_id, pages=pages,
                   pieces=_pieces(path.name, prose),
                   notes=_notes(path.name, notes_text),
                   path=path)


def load_sections(root) -> list[Section]:
    """Every translated section, in the order `text/sections.json` lists them.

    A section with no file under `translation/sections/` has simply not been
    translated yet; its pages stay pending.
    """
    root = pathlib.Path(root)
    index_path = root / "text" / "sections.json"
    if not index_path.is_file():
        return []
    records = json.loads(index_path.read_text(encoding="utf-8")).get("sections") or []
    sections = []
    for record in records:
        if not isinstance(record, dict) or not record.get("id"):
            continue
        path = root / "translation" / "sections" / f"{record['id']}.md"
        if not path.is_file():
            continue
        section = parse_section(path)
        sections.append(section._replace(
            kind=record.get("kind"), number=record.get("number"),
            heading=section_heading(record.get("kind"), record.get("number"),
                                    record.get("label"))))
    return sections


def _attach_notes(page_id: str, texts: list[str],
                  notes: list[SectionNote]) -> list[list[SectionNote]]:
    """The notes of one page, spread over its paragraphs by where their keys print."""
    by_key: dict[str, SectionNote] = {}
    keyless = [note for note in notes if note.key is None]
    for note in notes:
        if note.key is None:
            continue
        if note.key in by_key:
            _warn(f"{page_id}: two notes keyed {note.key!r}; keeping the first")
            continue
        by_key[note.key] = note
    attached: list[list[SectionNote]] = [[] for _ in texts]
    taken: set[str] = set()
    for i, text in enumerate(texts):
        for key in marker_keys(text):
            if key in by_key and key not in taken:
                attached[i].append(by_key[key])
                taken.add(key)
    for key, note in by_key.items():
        if key in taken:
            continue
        if not attached:
            _warn(f"{page_id}: note {key!r} has no English paragraph on this page")
            continue
        _warn(f"{page_id}: note {key!r} is not marked in the prose; "
              f"attached to the first paragraph")
        attached[0].append(note)
    # A note with no key (unkeyed in the print, or a quotation moved to the notes)
    # has no marker to follow; it sits with the page's first paragraph.
    for note in keyless:
        if attached:
            attached[0].append(note)
        else:
            _warn(f"{page_id}: a note with no key has no English paragraph on this page")
    return attached


def split_english(sections: list[Section], page_ids) -> dict[str, list[dict] | None]:
    """The English blocks of every page, sections in order. -> {page id: blocks|None}."""
    known = list(page_ids)
    known_set = set(known)

    # What lands on which page, in reading order, before notes are attached.
    texts: dict[str, list[str]] = {}
    continued: dict[str, list[bool]] = {}
    headings: dict[str, list[tuple[int, str]]] = {}
    current: str | None = None
    visited: dict[str, str] = {}

    def visit(page: str, section: Section) -> None:
        nonlocal current
        if page == current:
            return
        if page not in known_set:
            raise ValueError(f"section {section.id}: marker ⟦{page}⟧ is not a page "
                             f"of this book")
        if page in visited:
            raise ValueError(f"section {section.id}: page {page} is marked again "
                             f"after section {visited[page]} moved past it")
        visited[page] = section.id
        current = page

    for section in sections:
        if section.first_page is None:
            raise ValueError(f"section {section.id}: no ⟦page⟧ marker anywhere")
        for note in section.notes:
            if note.page not in known_set:
                raise ValueError(f"section {section.id}: note {{{note.key}}} is for "
                                 f"⟦{note.page}⟧, which is not a page of this book")
        first_text = next((p.text for p in section.pieces if p.text), None)
        # The translator may have written the heading out as its own paragraph; if
        # so it becomes a heading block below and must not be printed twice.
        if section.heading and first_text != section.heading:
            visit(section.first_page, section)
            headings.setdefault(section.first_page, []).append(
                (len(texts.get(section.first_page, [])), section.heading))
        for piece in section.pieces:
            if piece.page is not None:
                visit(piece.page, section)
            if not piece.text:
                continue
            texts.setdefault(current, []).append(piece.text)
            continued.setdefault(current, []).append(piece.continued)

    page_notes: dict[str, list[SectionNote]] = {}
    for section in sections:
        for note in section.notes:
            page_notes.setdefault(note.page, []).append(note)

    out: dict[str, list[dict] | None] = {page_id: None for page_id in known}
    for page_id in known:
        if page_id not in texts and page_id not in headings:
            continue
        paragraphs = texts.get(page_id, [])
        notes = _attach_notes(page_id, paragraphs, page_notes.get(page_id, []))
        blocks: list[dict] = []
        pending: dict[int, list[str]] = {}
        for at, heading in headings.get(page_id, []):
            pending.setdefault(at, []).append(heading)
        for i, text in enumerate(paragraphs):
            for heading in pending.pop(i, []):
                blocks.append({"type": "heading", "text": heading})
            if HEADING_PARAGRAPH_RE.match(text):
                blocks.append({"type": "heading", "text": text})
                continue
            blocks.append({
                "type": "paragraph",
                "html": to_html(text),
                "continued": continued[page_id][i],
                "notes": [{"key": n.key, "citation": n.citation,
                           "original": n.original, "gloss": n.gloss} for n in notes[i]],
            })
        for i in sorted(pending):
            for heading in pending[i]:
                blocks.append({"type": "heading", "text": heading})
        out[page_id] = blocks
    for page_id, notes in page_notes.items():
        if out.get(page_id) is None and notes:
            _warn(f"{page_id}: notes but no English prose; the page stays pending")
    return out


# --- one page -------------------------------------------------------------

def _source(rec: dict) -> dict:
    """Where this page's scan came from, and where a reader can see it."""
    kind = rec.get("source")
    if kind == "gallica":
        return {"kind": kind, "image_no": None, "url": GALLICA_PAGE}
    image_no = rec.get("image")
    if image_no is None:
        raise ValueError(f"{rec.get('id')}: source {kind!r} with no image number")
    return {"kind": kind, "image_no": image_no, "url": f"{CUDL_ITEM}/{image_no}"}


def french_notes(final: dict) -> list[dict]:
    """margin notes then foot notes, keyed by their printed letter.

    `beside_line` is a proof-reading aid for the transcription and is dropped: the
    viewer places a note by its marker in the text.
    """
    out = []
    for note in pagelib.notes(final):
        out.append({
            "key": note.key,
            "kind": "margin" if note.kind == "margin_notes" else "foot",
            "lines": list(note.lines),
        })
    return out


def uncertain(final: dict | None) -> list[dict]:
    """The reader's doubts about particular lines, without the `escalate` flag.

    Open arbitrations ("arbitration: undecided|unknown; alternatives: …") are left
    out: `contested_readings` carries them, with both readings.
    """
    if final is None:
        return []
    out = []
    for entry in final.get("uncertain") or []:
        # An open arbitration between the two readings is a contested reading,
        # carried in `readings` instead.
        if isinstance(entry, dict) and _open_alternatives(entry) is None:
            out.append({k: entry[k] for k in UNCERTAIN_KEYS if k in entry})
    return out


# --- contested readings ---------------------------------------------------
#
# Where the two transcription passes disagreed, the final carries one reading in
# its text and the pair in decisions[] (A, B, the text kept, who chose). A reading
# left open is an uncertain[] entry "arbitration: undecided|unknown; alternatives:
# A ||| B" whose line shows A. The site gets one `readings` record per contested
# spot, with the character spans of the words that differ, so the viewer can
# underline just those words and list the rest (page layout, the running head,
# note structure) in the page's apparatus.

ALT_PREFIXES = ("arbitration: undecided; alternatives: ",
                "arbitration: unknown; alternatives: ")
ALT_SEP = " ||| "
DECIDERS = ("carson", "translator", "reviewer", "auto")
# A decision with no `by` whose choice is one of these was the first run's
# reconciliation model's; the site names it `reconciler`.
RECONCILER_CHOICES = ("A", "B", "neither")
# An arbitration-session reason: "arbitration: A", "arbitration: B (translator: why)",
# "arbitration: either (auto-deferred)", "arbitration: neither (carson: why)".
SESSION_REASON_RE = re.compile(
    r"^arbitration:\s*(A|B|neither|either|unknown|undecided)\b"
    r"(?:\s*\((?:carson|translator|reviewer|auto-deferred)(?::\s*(.*))?\))?\s*$", re.S)
OPEN_CHOICES = ("either", "unknown", "undecided")
LINE_TARGET_RE = re.compile(r"^(blocks|margin_notes|foot_notes)\[(\d+)\]\.lines\[(\d+)\]$")
HEADING_TARGET_RE = re.compile(r"^blocks\[(\d+)\]\.text$")
# A line this short needs no shared word to anchor the alignment.
SHORT_LINE = 3


def _reading_str(value) -> str:
    """A reading as text: None is empty, a structure is its compact JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return pagelib.nfc(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _utf16(text: str, index: int) -> int:
    """A code-point offset as the UTF-16 offset a JavaScript string uses."""
    return len(text[:index].encode("utf-16-le")) // 2


def target_text(final: dict, where: str):
    """(target, text) for a pointer at one printed line or heading; (None, None) otherwise."""
    m = LINE_TARGET_RE.match(where or "")
    try:
        if m:
            field, i, j = m.group(1), int(m.group(2)), int(m.group(3))
            line = final[field][i]["lines"][j]
            if isinstance(line, str):
                return where, pagelib.nfc(line)
        m = HEADING_TARGET_RE.match(where or "")
        if m:
            block = final["blocks"][int(m.group(1))]
            if block.get("type") == "heading" and isinstance(block.get("text"), str):
                return where, pagelib.nfc(block["text"])
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    return None, None


def _is_ellipsis(word: str) -> bool:
    """A reading quoted in part elides the rest of the line with `…`."""
    return word.startswith("…") or word.strip(".") == ""


def _split_ellipses(words: list[str]) -> list[str]:
    """`…ombre` -> `…`, `ombre`: the elision and the word it runs into."""
    out = []
    for word in words:
        core = word.strip("…")
        if core != word and not any(ch.isalnum() for ch in core):
            out.append("…")       # `…,`: the elision and a stray point
        elif core and core != word:
            if word.startswith("…"):
                out.append("…")
            out.append(core)
            if word.endswith("…"):
                out.append("…")
        else:
            out.append(word)
    return out


def _differing(words: list[str], alternatives) -> set[int] | None:
    """Indices of `words` that differ from any alternative; None if unalignable."""
    import difflib
    if not words:
        return None
    hit: set[int] = set()
    for alt in alternatives:
        raw = _split_ellipses((alt or "").split())
        # `…` is a gap: the words of the line it stands for are not in dispute.
        other: list[str] = []
        gaps: set[int] = set()
        for word in raw:
            if _is_ellipsis(word):
                gaps.add(len(other))
            else:
                other.append(word)
        if other == words and not gaps:
            continue
        matcher = difflib.SequenceMatcher(None, words, other, autojunk=False)
        opcodes = matcher.get_opcodes()
        if len(words) > SHORT_LINE and not any(tag == "equal" for tag, *_ in opcodes):
            return None
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                continue
            if tag == "delete" and (j1 in gaps):
                continue
            if tag == "replace" and (j1 in gaps or j2 in gaps):
                # The gap swallows some of these words; the alternative's own words
                # answer for as many of the line's as it takes to spell them.
                hit.update(_beside_gap(words, i1, i2, other[j1:j2], at_start=j1 in gaps))
                continue
            if i2 > i1:
                hit.update(range(i1, i2))
            else:
                # words only the alternative has: mark the word before them
                hit.add(max(0, min(i1 - 1, len(words) - 1)))
    return hit or None


def _beside_gap(words, i1, i2, theirs, at_start):
    """The line's words, out of words[i1:i2], that `theirs` stands against.

    Taken from the end of the run when the gap comes first (`…, niAriſtote:`
    against `que Platon, ni Ariſtote:` -> `ni Ariſtote:`), from the start when it
    comes last: as many as it takes to reach the alternative's length, spaces
    aside.
    """
    want = len("".join(theirs))
    order = range(i2 - 1, i1 - 1, -1) if at_start else range(i1, i2)
    out, have = [], 0
    for i in order:
        out.append(i)
        have += len(words[i])
        if have >= want:
            break
    return out


def _spans(line: str, tokens: list[tuple[int, int]], hit: set[int]) -> list[list[int]]:
    """Token indices -> merged [start, end) UTF-16 spans of `line`."""
    spans: list[list[int]] = []
    last = None
    for i in sorted(hit):
        if last is not None and i == last + 1:
            spans[-1][1] = tokens[i][1]
        else:
            spans.append([tokens[i][0], tokens[i][1]])
        last = i
    return [[_utf16(line, s), _utf16(line, e)] for s, e in spans]


def _tokens(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def word_spans(line: str, alternatives) -> list[list[int]] | None:
    """The [start, end) spans of `line` whose words differ from any alternative.

    Words are whitespace-separated, compared as stitch_text.alt_markup compares
    them. A word only an alternative has marks its neighbour in `line`; a `…` in
    an alternative quoted in part stands for whatever the line has there.
    Adjacent differing words share one span; offsets are UTF-16, as the viewer's
    JavaScript counts them. None when the alignment cannot say which words
    differ: no words, nothing differs word for word, or no word in common to
    anchor on (in a line of more than SHORT_LINE words).
    """
    tokens = _tokens(line)
    hit = _differing([line[s:e] for s, e in tokens], alternatives)
    return None if hit is None else _spans(line, tokens, hit)


def _same(a: str, b: str) -> bool:
    return a.split() == b.split()


def _find_words(line: str, excerpt: str) -> int | None:
    """Index of the first of `line`'s words where the words of `excerpt` run; or None.

    The last quoted word may stop short of the line's word (`noraire` in
    `noraire,`).
    """
    want = excerpt.split()
    words = [line[s:e] for s, e in _tokens(line)]
    if not want:
        return None
    last = len(want) - 1

    def fits(word, k):
        return word == want[k] or (k == last and word.startswith(want[k]))

    for i in range(len(words) - len(want) + 1):
        if all(fits(words[i + k], k) for k in range(len(want))):
            return i
    return None


def _line_spans(line: str, a: str, b: str, text: str) -> list[list[int]] | None:
    """The spans of `line` to underline for readings a / b, or None if unalignable.

    The readings may be the whole line, the line quoted in part with `…`, or just
    the words in dispute; in that last case the kept reading is found in the line
    and the readings are compared there.
    """
    readings = [x for x in (text, a, b) if x]
    if any(_same(line, x) for x in readings) or any("…" in x for x in readings):
        return word_spans(line, [x for x in (a, b) if not _same(line, x)])
    for kept in readings:
        first = _find_words(line, kept)
        if first is None:
            continue
        # Compare the quoted words, then mark the printed words they stand for.
        hit = _differing(kept.split(), [x for x in (a, b) if not _same(kept, x)])
        if hit is None:
            return None
        tokens = _tokens(line)
        return _spans(line, tokens, {first + i for i in hit})
    return None


def reading_record(final: dict, where: str, a: str, b: str, text: str, chose,
                   by, status: str, reason) -> dict:
    """One contested reading, placed on its line when it has one.

    `spans` are UTF-16 offsets into the line as the final prints it (markers and
    all); `aligned` is False when the whole line is marked because the words
    could not be aligned. A reading with no single line to mark has
    `target: null` and `spans: null`, and so has `spans: null` one whose readings
    are word-for-word the same (the passes differed in spacing or layout only);
    both appear only in the page's list.
    """
    target, line = target_text(final, where)
    spans = None
    aligned = False
    if target is not None and not (_same(a, b) and _same(a, text)):
        spans = _line_spans(line, a, b, text)
        aligned = spans is not None
        if spans is None:
            spans = [[0, _utf16(line, len(line))]]
    return {"where": where, "target": target, "spans": spans, "aligned": aligned,
            "a": a, "b": b, "text": text, "chose": chose, "by": by,
            "status": status, "reason": reason or None}


def _open_alternatives(entry: dict):
    """(A, B) of an open arbitration entry in uncertain[]; None for any other entry."""
    note = entry.get("note") if isinstance(entry, dict) else None
    if not isinstance(note, str):
        return None
    for prefix in ALT_PREFIXES:
        if note.startswith(prefix):
            a, sep, b = note[len(prefix):].partition(ALT_SEP)
            return pagelib.nfc(a), pagelib.nfc(b) if sep else ""
    return None


def _default_by(chose) -> str:
    """Who decided an entry that does not say.

    The first pipeline run left `by` off: its reconciliation model chose A, B or
    neither, and Carson's own calls in that run say `chose: "carson-session"`.
    Anything else unmarked is taken as Carson's.
    """
    return "reconciler" if chose in RECONCILER_CHOICES else "carson"


def contested_readings(final: dict | None) -> list[dict]:
    """Every reading the two passes disagreed on, decided or open, in file order.

    From a decisions[] entry: `by` is carson | translator | reviewer | auto; absent or
    unknown, it is reconciler for a first-run choice of A, B or neither, else
    carson (see `_default_by`); an arbitration-session entry ("chose": "carson-session")
    takes its real choice from the reason ("arbitration: B"), whose provenance
    tail is dropped and whose own words, if any, become `reason`. A decision that
    chose "either"/"unknown", or that `by: auto` deferred, is `status: "open"` and
    shows reader A. An open uncertain[] entry with no decision at the same place
    adds a record of its own with `by: null`.
    """
    if final is None:
        return []
    out: list[dict] = []
    for entry in final.get("decisions") or []:
        if not isinstance(entry, dict):
            continue
        where = str(entry.get("where") or "").strip()
        if not where:
            continue
        a, b = _reading_str(entry.get("A")), _reading_str(entry.get("B"))
        text_given = _reading_str(entry.get("text"))
        if a and a == b and (not text_given or _same(text_given, a)
                             or _find_words(text_given, a) is not None):
            # Both passes read it the same and the text kept it: the entry records
            # a check, not a dispute.
            continue
        chose = entry.get("chose")
        by = entry.get("by") if entry.get("by") in DECIDERS else _default_by(chose)
        reason = entry.get("reason") if isinstance(entry.get("reason"), str) else None
        session = SESSION_REASON_RE.match(reason or "")
        if session:
            chose = session.group(1)
            reason = (session.group(2) or "").strip() or None
        elif chose == "carson-session":
            chose = "neither"
        if chose not in ("A", "B", "neither") + OPEN_CHOICES:
            chose = "neither"
        status = "open" if by == "auto" or chose in OPEN_CHOICES else "decided"
        if "text" in entry and entry["text"] is not None:
            text = _reading_str(entry["text"])
        else:
            text = {"A": a, "B": b}.get(chose, a)
        if status == "open":
            chose, text = None, a
        out.append(reading_record(final, where, a, b, text, chose, by, status, reason))
    placed = {r["where"] for r in out}
    for entry in final.get("uncertain") or []:
        pair = _open_alternatives(entry)
        if pair is None:
            continue
        where = str(entry.get("where") or "").strip()
        if not where or where in placed:
            continue
        placed.add(where)
        out.append(reading_record(final, where, pair[0], pair[1], pair[0], None, None,
                                  "open", None))
    return out


def page_record(manifest, page_id: str, final: dict | None,
                prev_id: str | None, next_id: str | None,
                english: list[dict] | None = None) -> dict:
    """The site record for one page. `final` / `english` are None while pending."""
    rec = pagelib.manifest_record(manifest, page_id)
    if rec is None:
        raise KeyError(f"{page_id}: not in the manifest")
    # The final is the authority on what the page itself prints; the manifest
    # holds what only it knows (the side, the scan number).
    folio = final.get("folio") if final is not None else None
    if folio is None:
        folio = rec.get("folio")
    return {
        "id": page_id,
        "page": rec.get("page"),
        "folio": folio,
        "side": rec.get("side"),
        "image": page_id,
        "source": _source(rec),
        "running_head": final.get("running_head") if final is not None else None,
        "prev": prev_id,
        "next": next_id,
        "english": english,
        "french": pagelib.blocks(final) if final is not None else None,
        "french_notes": french_notes(final) if final is not None else None,
        "uncertain": uncertain(final),
        "readings": contested_readings(final),
    }


def index_record(manifest_rec: dict, final: dict | None,
                 english: list[dict] | None = None) -> dict:
    """The short form the viewer's jump menu and the landing card's progress use."""
    folio = final.get("folio") if final is not None else None
    if folio is None:
        folio = manifest_rec.get("folio")
    return {
        "id": manifest_rec.get("id"),
        "page": manifest_rec.get("page"),
        "folio": folio,
        "heading": index_heading(manifest_rec, final),
        "layers": {"fr": final is not None, "en": english is not None},
    }


def book_record() -> dict:
    return {
        "slug": "martin-guerre",
        "title": "Arrest memorable du Parlement de Tholose",
        "short_title": "Martin Guerre",
        "author": "Jean de Coras",
        "year": 1572,
        "description": BOOK_DESCRIPTION,
        "about": list(BOOK_ABOUT),
        "source": {
            "name": "Cambridge University Library",
            "item_url": f"{CUDL_ITEM}/1",
            "license": "CC BY-NC 4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        },
        "images": {"base": "img/", "ext": ".webp", "width": 2805},
        "layers": [{"code": "en", "label": "English"},
                   {"code": "fr", "label": "French"}],
        "default_layer": "en",
        "stylesheet": None,
        "first_page": "p000-title",
    }


# --- the run --------------------------------------------------------------

def write_json(path: pathlib.Path, data) -> None:
    """One file, UTF-8, keys in the order they were built."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


def load_final(root: pathlib.Path, rec: dict) -> dict | None:
    """The finished transcription of this page, or None if there is not one yet.

    A final that exists but is not marked done in the manifest is still being
    worked on, so the site does not show it; neither does a "done" final with no
    blocks, which cannot be a transcription of anything.
    """
    page_id = rec.get("id")
    if (rec.get("status") or {}).get("final") != "done":
        return None
    path = pathlib.Path(root) / "transcription" / "final" / f"{page_id}.json"
    if not path.exists():
        return None
    final = pagelib.load_page(path)
    if not isinstance(final.get("blocks"), list) or not final["blocks"]:
        _warn(f"{page_id}: final has no blocks; the French layer stays pending")
        return None
    return final


def build(root: pathlib.Path, out: pathlib.Path) -> tuple[int, int, int]:
    """Write book.json, index.json and pages/<id>.json. -> (pages, french, english).

    Everything is built and validated before anything is written, so a page that
    does not fit the schema leaves the previous `site/data` untouched.
    """
    import jsonschema

    root = pathlib.Path(root)
    out = pathlib.Path(out)
    manifest = pagelib.load_manifest(root / "manifest.json")
    if not manifest:
        raise pagelib.PageLoadError(root / "manifest.json", "file not found")
    if not isinstance(manifest.get("pages"), list):
        raise pagelib.PageLoadError(root / "manifest.json", "`pages` is not a list")
    records = [r for r in manifest["pages"] if isinstance(r, dict)]
    if not records:
        raise pagelib.PageLoadError(root / "manifest.json", "no page records")
    ids = [r.get("id") for r in records]

    english = split_english(load_sections(root), ids)

    schema = json.loads(SITE_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    pages: list[tuple[str, dict]] = []
    index: list[dict] = []
    n_french = n_english = 0
    for i, rec in enumerate(records):
        page_id = rec.get("id")
        final = load_final(root, rec)
        page = pagelib.nfc_all(page_record(
            manifest, page_id, final,
            ids[i - 1] if i else None,
            ids[i + 1] if i + 1 < len(records) else None,
            english.get(page_id)))
        error = jsonschema.exceptions.best_match(validator.iter_errors(page))
        if error is not None:
            where = "/".join(str(p) for p in error.absolute_path)
            raise jsonschema.ValidationError(
                f"{page_id}: invalid site page record"
                f"{' at ' + where if where else ''}: {error.message}")
        pages.append((page_id, page))
        index.append(pagelib.nfc_all(index_record(rec, final, english.get(page_id))))
        n_french += page["french"] is not None
        n_english += page["english"] is not None

    pages_dir = out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for page_id, page in pages:
        write_json(pages_dir / f"{page_id}.json", page)
    keep = {f"{page_id}.json" for page_id, _ in pages}
    for stale in sorted(pages_dir.glob("*.json")):
        if stale.name not in keep:
            stale.unlink()
    write_json(out / "book.json", book_record())
    write_json(out / "index.json", {"pages": index})
    return len(records), n_french, n_english


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT,
                    help="the 2026/ directory holding manifest.json, transcription/, "
                         "text/ and translation/")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="where to write the data files (default <root>/site/data)")
    args = ap.parse_args(argv)
    out = args.out if args.out is not None else args.root / "site" / "data"
    n, fr, en = build(args.root, out)
    print(f"{n} pages written ({fr} french, {en} english)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

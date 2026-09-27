"""Shared helpers for the page tools (validate_page.py, diff_reads.py).

A "page" is the JSON object described by scripts/page_schema.json and
docs/conventions.md: the diplomatic, line-by-line transcription of one printed
page.
"""
from __future__ import annotations

import json
import pathlib
import re
import unicodedata
from typing import Iterator, NamedTuple

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
SCHEMA_PATH = SCRIPTS_DIR / "page_schema.json"

# Marker keys run a…z, restarting as a2, a3 (§4). Capitals are matched so the
# validator can complain about them instead of silently ignoring them.
MARKER_RE = re.compile(r"\{([A-Za-zſ]+\d*)\}")

# fields of the page that hold transcribed text but are not "lines"
FURNITURE = ("running_head", "folio", "signature", "catchword")
BLOCK_TYPES = ("heading", "paragraph", "ornament", "blank")
TEXT_BLOCKS = ("heading", "ornament", "blank")


class PageLoadError(Exception):
    """A page or manifest file could not be read as a JSON object."""

    def __init__(self, path, reason: str):
        self.path = pathlib.Path(path)
        self.reason = reason
        super().__init__(f"{self.path}: {reason}")


class Line(NamedTuple):
    """One transcribed string, with enough context to point at it."""

    where: str          # "blocks[1].lines[3]" / "margin_notes[0].lines[2]" / "folio"
    container: str      # "blocks[1]" / "margin_notes[0]" / "folio"
    kind: str           # "body" | "margin_note" | "foot_note" | "heading_text" | …
    index: int          # index of the line inside its container
    text: str
    double_space_ok: bool   # spaced capitals (§3) and the running head


class Note(NamedTuple):
    kind: str           # "margin_notes" | "foot_notes"
    index: int
    key: str | None
    lines: list[str]
    beside_line: str | None

    @property
    def where(self) -> str:
        return f"{self.kind}[{self.index}]"


# --- loading --------------------------------------------------------------

def _load_json_object(path) -> dict:
    p = pathlib.Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except IsADirectoryError:
        raise PageLoadError(p, "is a directory, not a file") from None
    except FileNotFoundError:
        raise PageLoadError(p, "file not found") from None
    except UnicodeDecodeError as exc:
        raise PageLoadError(p, f"not valid UTF-8 ({exc.reason})") from None
    except OSError as exc:
        raise PageLoadError(p, f"cannot open ({exc.strerror or exc})") from None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PageLoadError(p, f"invalid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise PageLoadError(p, f"top level is {type(data).__name__}, expected a JSON object")
    return data


def load_page(path) -> dict:
    """Load one page file. Raises PageLoadError with a usable reason."""
    return _load_json_object(path)


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_manifest(path):
    """Load the manifest, or None if there is no file at `path`."""
    p = pathlib.Path(path)
    if not p.exists():
        return None
    return _load_json_object(p)


def manifest_record(manifest, page_id: str):
    if not manifest:
        return None
    for rec in manifest.get("pages") or []:
        if isinstance(rec, dict) and rec.get("id") == page_id:
            return rec
    return None


def write_manifest(path, manifest: dict) -> None:
    """Atomic: serialize fully first (so a non-serializable value raises before anything
    touches disk), write to a sibling temp file, then rename over the target."""
    path = pathlib.Path(path)
    text = json.dumps(manifest, indent=1, ensure_ascii=False, default=_json_default) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _json_default(o):
    """Accept numpy scalars and other int/float look-alikes."""
    for attr in ("item", "__int__", "__float__"):
        if hasattr(o, attr):
            v = o.item() if attr == "item" else (int(o) if attr == "__int__" else float(o))
            return v
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# --- Unicode --------------------------------------------------------------

def nfc(text: str) -> str:
    """Precomposed form. The conventions ask for ã ẽ ĩ õ ũ, not a + U+0303 (§2)."""
    return unicodedata.normalize("NFC", text)


def nfc_all(obj):
    """Recursively NFC-normalize every string in a loaded page."""
    if isinstance(obj, str):
        return nfc(obj)
    if isinstance(obj, list):
        return [nfc_all(v) for v in obj]
    if isinstance(obj, dict):
        return {k: nfc_all(v) for k, v in obj.items()}
    return obj


# --- structure ------------------------------------------------------------

def blocks(page: dict) -> list[dict]:
    """The block list, with anything that is not an object dropped."""
    return [b for b in (page.get("blocks") or []) if isinstance(b, dict)]


def block_types(page: dict) -> list:
    """The `type` of each block, in order (None for a block without one)."""
    return [b.get("type") if isinstance(b, dict) else None
            for b in (page.get("blocks") or [])]


def body_lines(page: dict) -> list[Line]:
    """Every paragraph line of the body column, in reading order."""
    out: list[Line] = []
    for bi, block in enumerate(page.get("blocks") or []):
        if not isinstance(block, dict) or block.get("type") != "paragraph":
            continue
        spaced = bool(block.get("spaced_caps", False))
        for li, text in enumerate(block.get("lines") or []):
            if isinstance(text, str):
                out.append(Line(f"blocks[{bi}].lines[{li}]", f"blocks[{bi}]",
                                "body", li, text, spaced))
    return out


def body_texts(page: dict) -> list[str]:
    return [ln.text for ln in body_lines(page)]


def column_texts(page: dict) -> list[str]:
    """Every printed line of the body column in block order: heading text and
    paragraph lines alike. Used for read-vs-read alignment, so that a line one
    reader put in a heading and the other in a paragraph is still compared."""
    out: list[str] = []
    for block in page.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "heading" and isinstance(block.get("text"), str):
            out.append(block["text"])
        elif block.get("type") == "paragraph":
            out.extend(t for t in (block.get("lines") or []) if isinstance(t, str))
    return out


def notes(page: dict) -> list[Note]:
    """margin_notes then foot_notes, in file order."""
    out: list[Note] = []
    for kind in ("margin_notes", "foot_notes"):
        for i, note in enumerate(page.get(kind) or []):
            if not isinstance(note, dict):
                continue
            lines = [t for t in (note.get("lines") or []) if isinstance(t, str)]
            out.append(Note(kind, i, note.get("key"), lines, note.get("beside_line")))
    return out


def note_lines(page: dict) -> list[Line]:
    out: list[Line] = []
    for note in notes(page):
        kind = "margin_note" if note.kind == "margin_notes" else "foot_note"
        for li, text in enumerate(note.lines):
            out.append(Line(f"{note.where}.lines[{li}]", note.where, kind, li, text, False))
    return out


def block_text_lines(page: dict) -> list[Line]:
    """The `text` of heading / ornament / blank blocks.

    Only a heading can be set in spaced capitals, so only a heading may keep
    its double spaces (§3).
    """
    out: list[Line] = []
    for bi, block in enumerate(page.get("blocks") or []):
        if not isinstance(block, dict) or block.get("type") not in TEXT_BLOCKS:
            continue
        text = block.get("text")
        if not isinstance(text, str):
            continue
        spaced = block["type"] == "heading" and bool(block.get("spaced_caps", False))
        out.append(Line(f"blocks[{bi}].text", f"blocks[{bi}]",
                        f"{block['type']}_text", 0, text, spaced))
    return out


def furniture(page: dict) -> list[Line]:
    """running_head, folio, signature, catchword (the ones actually present).

    The running head is set in spaced capitals (§3), so it alone keeps its
    double spaces.
    """
    out: list[Line] = []
    for field in FURNITURE:
        text = page.get(field)
        if isinstance(text, str):
            out.append(Line(field, field, "field", 0, text, field == "running_head"))
    return out


def all_lines(page: dict) -> list[Line]:
    """Every transcribed string on the page, whatever holds it."""
    return body_lines(page) + note_lines(page) + block_text_lines(page) + furniture(page)


def printed_lines(page: dict) -> list[Line]:
    """Lines that reproduce the printed text of the page itself.

    Ornament/blank `text` is an editorial description and page furniture is
    mostly capitals, so neither is worth scanning for normalized long s.
    """
    headings = [ln for ln in block_text_lines(page) if ln.kind == "heading_text"]
    return body_lines(page) + note_lines(page) + headings


def note_map(page: dict) -> dict[str, list[str]]:
    """Note lines keyed by note key; unkeyed notes get a synthetic key."""
    out: dict[str, list[str]] = {}
    unkeyed = 0
    for note in notes(page):
        key = note.key
        if key is None:
            key = f"_unkeyed_{unkeyed}"
            unkeyed += 1
        elif key in out:
            # duplicate keys are a validator problem; keep both visible here
            suffix = 2
            while f"{key}#{suffix}" in out:
                suffix += 1
            key = f"{key}#{suffix}"
        out[key] = list(note.lines)
    return out


def markers(page: dict) -> list[tuple[str, str]]:
    """(key, where) for every {x} marker in the body, in order.

    Headings carry markers too, so they count as body text here.
    """
    found = []
    sources = body_lines(page) + [ln for ln in block_text_lines(page)
                                  if ln.kind == "heading_text"]
    for line in sources:
        for m in MARKER_RE.finditer(line.text):
            found.append((m.group(1), line.where))
    return found


def note_markers(page: dict) -> list[tuple[str, str]]:
    """(key, where) for every {x} found inside a note's own lines.

    Notes never carry markers: the key letter printed at the head of a note is
    the `key`, not part of `lines` (§9).
    """
    found = []
    for line in note_lines(page):
        for m in MARKER_RE.finditer(line.text):
            found.append((m.group(1), line.where))
    return found


def headings(page: dict) -> list[str]:
    return [b.get("text", "") for b in blocks(page) if b.get("type") == "heading"]


def uncertain_blobs(page: dict) -> list[str]:
    """One "<where> <note>" string per uncertain[] entry, as written.

    Callers lowercase it themselves when they want a case-insensitive match.
    """
    out = []
    for entry in page.get("uncertain") or []:
        if isinstance(entry, dict):
            out.append(f"{entry.get('where', '')} {entry.get('note', '')}")
    return out


# `where` may be written "blocks[1].lines[3]", "blocks[1] line 3",
# "blocks[1]: line 3" or "blocks[1], line 3"; anything else is a different
# container, so "blocks[10]" never points into "blocks[1]".
POINTER_BOUNDARY = (".", ",", ":", " ")


def points_at(where: str, container: str) -> bool:
    """True if an uncertain[]/decisions[] `where` points into `container`."""
    if where == container:
        return True
    return (where.startswith(container)
            and where[len(container):len(container) + 1] in POINTER_BOUNDARY)


# --- normalization used for diff alignment --------------------------------

_SPACE_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s*([^\w\s])\s*", re.UNICODE)


def _tighten_punctuation(match: re.Match) -> str:
    """Drop the spaces around a punctuation mark, but not around a combining one.

    A combining mark (Mn) such as the tilde of `q̃` is not a word character to
    `re`, but it is part of its word: eating the space around it would glue
    `q̃ eſt` into one token and knock the alignment off.
    """
    char = match.group(1)
    if unicodedata.category(char) == "Mn":
        return match.group(0)
    return char


def align_key(text: str) -> str:
    """Loose form used only to *align* two reads.

    NFC first, then lowercase, map long s to s, collapse whitespace and strip
    spaces around punctuation, so that exactly the differences we want to
    report (long s, u/v, spacing, case) do not knock the alignment off.
    """
    t = nfc(text).replace("ſ", "s").replace("ẛ", "s").lower()
    t = _SPACE_RE.sub(" ", t)
    t = _PUNCT_SPACE_RE.sub(_tighten_punctuation, t)
    return t.strip()


def iter_pages(root: pathlib.Path) -> Iterator[pathlib.Path]:
    if not root.exists():
        return
    yield from sorted(root.rglob("*.json"))

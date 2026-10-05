"""Fetch and build the canon-law source corpora for the site's source pane.

Four corpora, written per docs/sources-contract.md to site/data/sources/<corpus>/:

  decretum     Gratian's Decretum, Friedberg (Leipzig 1879), from the MGH/BSB full text
               at geschichte.digitale-sammlungen.de/decretum-gratiani (one web page per
               canon, dictum, question, distinctio or causa; footnotes not included).
               Units: D.1..D.101, C.1..C.36, De pen. D.1..D.7 (= C.33 q.3), De cons. D.1..D.5.
  decretals    Liber Extra (X), Friedberg (Leipzig 1881), from the Bibliotheca Augustana
               transcription (Angus Graham), one page per title. Units: book.title.
  sext         Liber Sextus, Friedberg vol. 2, from the archive.org OCR (_djvu.txt) of the
  clementines  1955 Graz reprint (BD1141952). OCR quality; footnote blocks are dropped by
               heuristics. Units: book.title.

Every remote file is fetched once into the cache (sequential, with a pause), and the
build reads only the cache, so it can be re-run offline.

    python scripts/fetch_canon_law.py fetch [--only decretum|decretals|ocr] [--cache DIR]
    python scripts/fetch_canon_law.py build [--cache DIR] [--out site/data/sources]
"""
from __future__ import annotations

import argparse
import html
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "site" / "data" / "sources"
DEFAULT_CACHE = pathlib.Path(os.environ.get(
    "CANON_CACHE",
    "/private/tmp/claude-502/-Users-cdavis-github-translator/"
    "d3de2614-6075-4c4c-8061-e08283898989/scratchpad/sources-cache/canon"))

UA = "Mozilla/5.0 (compatible; coras-translation-sources/1.0; scholarly use)"
PAUSE = 0.35

MDZ_BASE = "https://geschichte.digitale-sammlungen.de/decretum-gratiani"
MDZ_LAST = 4176
AUG_BASE = "https://www.hs-augsburg.de/~harsch/Chronologia/Lspost13/GregoriusIX"
OCR_VOL2 = "https://archive.org/download/BD1141952/BD1141952_djvu.txt"
SCAN_VOL1 = "https://archive.org/details/corpus-iuris-canonici-t.-i-decretum-gratiani"
SCAN_VOL2 = "https://archive.org/details/BD1141952"


# --- roman numerals ---------------------------------------------------------

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman(s: str) -> int | None:
    """Roman numeral to int; None if `s` is not one."""
    s = s.strip().upper().rstrip(".")
    if not s or any(ch not in _ROMAN for ch in s):
        return None
    total = 0
    for i, ch in enumerate(s):
        v = _ROMAN[ch]
        if i + 1 < len(s) and _ROMAN[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


# OCR confusions in numerals: n->II, m->III, H->II, U->II, l/1/i/r->I, y->V, ffl->III.
_OCR_CHARS = [("ffl", "III"), ("fl", "II"), ("n", "II"), ("m", "III"), ("U", "II"),
              ("u", "II"), ("H", "II"), ("E", "II"), ("l", "I"), ("1", "I"), ("|", "I"),
              ("i", "I"), ("r", "I"), ("T", "I"), ("y", "V"), ("Y", "V"), ("v", "V"),
              ("x", "X"), ("c", "C")]


def ocr_roman(s: str) -> tuple[int | None, bool]:
    """Roman numeral as OCR'd in Friedberg headings ('XVHL', 'm', 'xxrv', 'XXX VH').
    Returns (value, clean): `clean` when the token was already a well-formed numeral."""
    s = re.sub(r"[^A-Za-z0-9|]", "", s)
    if not s:
        return None, False
    if s.isdigit() and len(s) <= 2 and s not in ("1", "11", "111"):
        return int(s), False
    r = roman(s) if s.isupper() else None
    if r is not None and not (len(s) > 1 and s.endswith("L")):
        return r, True
    t = s
    if len(t) > 1 and t.endswith("L"):          # 'XIIL' = 'XIII' / 'XII.'
        t = t[:-1] + "I"
    if t == "L":
        t = "I"
    for x, y in _OCR_CHARS:
        t = t.replace(x, y)
    return roman(t), False


def plausible(raw: str, prev: int | None) -> int:
    """Number for a title/chapter heading given the previous one in sequence.
    A clean numeral is trusted; an OCR-repaired one must follow `prev` closely."""
    v, clean = ocr_roman(raw)
    base = prev or 0
    if v is not None and (clean or base < v <= base + 4):
        return v
    return base + 1


# --- fetching ---------------------------------------------------------------

def _get(url: str, dest: pathlib.Path, binary_ok: bool = True) -> bool:
    """Fetch `url` to `dest` once. Returns True if fetched now, False if cached/failed."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            time.sleep(PAUSE)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                dest.with_suffix(".404").write_text(url)
                return False
            time.sleep(5 * (attempt + 1))
        except Exception:  # network hiccup
            time.sleep(5 * (attempt + 1))
    print("failed:", url, file=sys.stderr)
    return False


def mdz_chapter_id(n: int) -> str:
    return f"dc_chapter_{n // 1000}_{n:04d}"


def mdz_order() -> list[int]:
    """Fetch order: C.27-C.36 first (most cited), then the distinctiones, then the rest."""
    first = list(range(3067, 3767))
    second = list(range(1, 1077))
    rest = [n for n in range(1, MDZ_LAST + 1) if n not in set(first) | set(second)]
    return first + second + rest


def fetch_decretum(cache: pathlib.Path) -> None:
    d = cache / "mdz_decretum"
    got = 0
    for i, n in enumerate(mdz_order()):
        if _get(f"{MDZ_BASE}/kapitel/{mdz_chapter_id(n)}", d / f"{n:04d}.html"):
            got += 1
        if i % 200 == 0:
            print(f"decretum {i}/{MDZ_LAST} (fetched {got})", flush=True)


def fetch_decretals(cache: pathlib.Path) -> None:
    d = cache / "augustana_x"
    _get(f"{AUG_BASE}/gre_0000.html", cache / "aug_gre_0000.html")
    idx = (cache / "aug_gre_0000.html").read_text(encoding="latin-1")
    pages = sorted(set(re.findall(r'(?i)href="(gre_\dt\d\d\.html)"', idx)))
    for p in pages:
        _get(f"{AUG_BASE}/{p}", d / p)
    print(f"decretals: {len(pages)} title pages", flush=True)


def fetch_ocr(cache: pathlib.Path) -> None:
    _get(OCR_VOL2, cache / "BD1141952_djvu.txt")


# --- Decretum (MDZ) ---------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")


def mdz_page(raw: str) -> tuple[str, str]:
    """(heading, text) of one MDZ chapter page."""
    m = re.search(r'<h2 class="content">(.*?)</h2>(.*?)<form', raw, re.S)
    if not m:
        return "", ""
    head = " ".join(html.unescape(_TAG.sub("", m.group(1))).split())
    body = m.group(2).replace("\r", "").replace("\n", " ")   # source newlines are layout
    body = re.sub(r"<br[^>]*>", "\n", body)
    body = html.unescape(_TAG.sub("", body))
    return head, join_lines(body.split("\n"))


def join_lines(lines: list[str]) -> str:
    """Join hard-wrapped lines into paragraphs; blank lines separate paragraphs.
    Line-end hyphens ('duo-' + 'bus') are joined."""
    paras, cur = [], []
    for ln in lines:
        ln = " ".join(ln.split())
        if not ln:
            if cur:
                paras.append(cur)
                cur = []
            continue
        cur.append(ln)
    if cur:
        paras.append(cur)
    out = []
    for p in paras:
        s = ""
        for ln in p:
            if s.endswith("-") and ln[:1].islower():
                s = s[:-1] + ln
            else:
                s = (s + " " + ln) if s else ln
        out.append(s.strip())
    return "\n\n".join(x for x in out if x)


_CANON_HEAD = re.compile(r"^\[?(PALEA\.\s*)?(?:C|CAP)\.\s*([IVXLC]+|UN)\b\.?\]?")


def decretum_units(pages: list[tuple[int, str, str]]) -> dict[str, dict]:
    """Group MDZ pages (n, heading, text) in reading order into contract units.

    Headings drive a small state machine: DISTINCTIO (part I, De pen., De cons.),
    CAUSA, QUESTIO, and canons ('C. IV. rubric', '[C. II.] inscription')."""
    units: dict[str, dict] = {}
    part = 1            # 1 = distinctiones, 2 = causae, 3 = de consecratione
    causa = q = None
    depen = False
    unit_id = None
    prefix = ""        # passage id prefix

    def start(uid: str, title: str) -> None:
        nonlocal unit_id
        unit_id = uid
        units.setdefault(uid, {"corpus": "decretum", "unit": uid, "title": title,
                               "url": f"{MDZ_BASE}/kapitel/{mdz_chapter_id(cur_n)}",
                               "passages": []})

    def add(pid: str, label: str, text: str) -> None:
        if unit_id is None or not text:
            return
        ps = units[unit_id]["passages"]
        if any(p["id"] == pid for p in ps):          # duplicate numbering: keep both
            k = 2
            while any(p["id"] == f"{pid}#{k}" for p in ps):
                k += 1
            pid = f"{pid}#{k}"
        ps.append({"id": pid, "label": label, "text": text,
                   "url": f"{MDZ_BASE}/kapitel/{mdz_chapter_id(cur_n)}"})

    cur_n = 0
    for n, head, text in pages:
        cur_n = n
        H = head.upper()
        if n >= 3767 and part == 2:      # MDZ numbering: De consecratione starts at 3767
            part, depen = 3, False
        if H.startswith("DECRETI PARS SECUNDA"):
            part, depen = 2, False
            continue
        if H.startswith("DECRETI PARS TERTIA"):
            part, depen = 3, False
            continue
        if H.startswith(("TITELBLATT", "PROLEGOMENA", "ARBOR", "CONSANGUINITAS",
                         "DECLARATIO", "INDICES", "CANONUM DECRETI", "INDEX",
                         "ADDENDA")) or n >= 4169:
            unit_id = None
            continue
        if H.startswith("CONCORDIA DISCORDANTIUM"):
            unit_id = None
            continue
        m = re.match(r"CAUSA\s+([IVXL]+)", H)
        if m and part in (1, 2):
            part = 2
            causa, q, depen = roman(m.group(1)), None, False
            start(f"C.{causa}", f"Decretum, C. {causa}")
            prefix = f"C.{causa}"
            add(f"C.{causa} pr", f"C. {causa} (casus)", strip_head(text, head))
            continue
        m = re.match(r"QU?A?ESTIO\s+([IVXL]+)", H)
        if m and part == 2 and causa:
            q = roman(m.group(1))
            if causa == 33 and q == 3:
                depen = True          # De penitentia: distinctiones follow
                prefix = "De pen."
                start("de-pen-D.1", "Decretum, De penitentia D. 1")
                add("De pen. pr", "C. 33 q. 3 (De penitentia)", strip_head(text, head))
                continue
            if depen:                 # C.33 q.4, q.5 after De penitentia
                depen = False
                start(f"C.{causa}", f"Decretum, C. {causa}")
            prefix = f"C.{causa} q.{q}"
            add(f"{prefix} pr", f"C. {causa} q. {q} d. a. c. 1", strip_head(text, head))
            continue
        m = re.match(r"DISTINCTIO\s+([IVXLC]+|PRIMA)", H)
        if m:
            d = 1 if m.group(1) == "PRIMA" else roman(m.group(1))
            if part == 1:
                start(f"D.{d}", f"Decretum, D. {d}")
                prefix = f"D.{d}"
            elif part == 2 and depen:
                start(f"de-pen-D.{d}", f"Decretum, De penitentia D. {d}")
                prefix = f"De pen. D.{d}"
            elif part == 3:
                start(f"de-cons-D.{d}", f"Decretum, De consecratione D. {d}")
                prefix = f"De cons. D.{d}"
            else:
                continue
            lab = prefix.replace("D.", "D. ")
            add(f"{prefix} pr", f"{lab} d. a. c. 1", strip_head(text, head))
            continue
        m = _CANON_HEAD.match(head)
        if m and unit_id:
            c = 1 if m.group(2) == "UN" else roman(m.group(2))
            lab = prefix.replace("D.", "D. ").replace("q.", "q. ").replace("C.", "C. ", 1) \
                if prefix.startswith("C.") else prefix.replace("D.", "D. ")
            body = text
            # the page text repeats the heading as its first line
            body = strip_head(body, head)
            rubric = head[m.end():].strip(" ]")
            label = f"{lab} c. {c}" + (" (Palea)" if m.group(1) else "") \
                + (f" ({rubric})" if rubric else "")
            add(f"{prefix} c.{c}", label, body if not rubric else
                (rubric + "\n\n" + body if body else rubric))
            continue
        # anything else inside a unit (e.g. 'GRATIANUS.' dicta, Palea headings)
        if unit_id:
            add(f"{prefix} {n}", head or prefix, strip_head(text, head))
    for u in units.values():
        u["passages"] = [p for p in u["passages"] if p["text"].strip()]
    return units


def strip_head(text: str, head: str) -> str:
    """Drop the heading the MDZ page repeats as its first paragraph."""
    if not head:
        return text.strip()
    norm = lambda s: re.sub(r"\W+", "", s).lower()  # noqa: E731
    paras = text.split("\n\n")
    if paras and norm(paras[0]).startswith(norm(head)[:40]):
        first = paras[0]
        # cut the heading off the front of the first paragraph, keep the rest
        hn = norm(head)
        acc, i = "", 0
        while i < len(first) and len(norm(acc)) < len(hn):
            acc += first[i]
            i += 1
        rest = first[i:].strip(" .")
        paras = ([rest] if rest else []) + paras[1:]
    return "\n\n".join(paras).strip()


def build_decretum(cache: pathlib.Path) -> tuple[dict, dict]:
    d = cache / "mdz_decretum"
    pages = []
    missing = []
    for n in range(1, MDZ_LAST + 1):
        f = d / f"{n:04d}.html"
        if not f.exists():
            missing.append(n)
            continue
        head, text = mdz_page(f.read_text(encoding="latin-1"))
        pages.append((n, head, text))
    units = decretum_units(pages)
    meta = {
        "id": "decretum",
        "title": "Decretum Gratiani",
        "language": "la",
        "edition": "Friedberg, Corpus Iuris Canonici I (Leipzig 1879), text as digitised "
                   "by the MGH and the Bayerische Staatsbibliothek (footnotes omitted)",
        "license": "public domain (Friedberg 1879 edition text); the site states no "
                   "licence for its digital text, used with attribution to MGH / "
                   "Bayerische Staatsbibliothek (Münchener DigitalisierungsZentrum)",
        "attribution": "Text: Decretum Gratiani, geschichte.digitale-sammlungen.de "
                       "(MGH / BSB)",
        "url": f"{MDZ_BASE}/online/angebot",
        "unit_scheme": "D.n | C.n | de-pen-D.n | de-cons-D.n (file-safe slugs for "
                       "De penitentia = C.33 q.3, and De consecratione)",
        "external_url_note": "each unit and passage carries `url`, its MDZ page",
        "passage_scheme": "D.n c.n | C.n q.n c.n | De pen. D.n c.n | De cons. D.n c.n "
                          "(pr = dictum ante / casus)",
        "quality": "clean",
        "scan_url_template": SCAN_VOL1,
        "missing_pages": len(missing),
    }
    return meta, units


# --- Liber Extra (Augustana) -----------------------------------------------

def augustana_title(raw: str) -> tuple[str, list[tuple[int, str]]]:
    """(title heading, [(chapter number, text)]) of one Augustana title page."""
    raw = raw.replace("\r", "")
    # title heading: 'Titulus XV' + rubric, in the first f_ruber/f_canus spans
    body = raw
    chunks = re.split(r'<SPAN CLASS="f_ruber">\s*Capitulum\s*([IVXLC]+|unicum|unic\.?)'
                      r'\.?\s*</SPAN>', body, flags=re.I)
    head_html = chunks[0]
    title = ""
    m = re.search(r'(?is)<SPAN CLASS="f_viridis">(.*?)</SPAN>', head_html)
    if m:
        parts = re.split(r"(?i)<BR[^>]*>", m.group(1), maxsplit=1)
        rubric = " ".join(html.unescape(_TAG.sub(" ", parts[1] if len(parts) > 1 else ""))
                          .split())
        num = html.unescape(_TAG.sub("", parts[0])).replace("\xa0", " ")
        num = re.sub(r"\s+", "", num).replace("Titulus", "").strip(".")
        title = f"Titulus {num}. {rubric}".strip()
    out = []
    for i in range(1, len(chunks), 2):
        num = chunks[i]
        c = 1 if num.lower().startswith("unic") else roman(num)
        txt = chunks[i + 1]
        txt = re.split(r'(?i)<SPAN CLASS="f_roseus">|</DL>|</BODY>', txt)[0]
        txt = re.sub(r"(?i)<BR[^>]*>", "\n", txt)
        txt = re.sub(r"(?i)<D[DT][^>]*>", "\n\n", txt)
        txt = html.unescape(_TAG.sub("", txt))
        out.append((c, join_lines(txt.split("\n"))))
    return title, out


def build_decretals(cache: pathlib.Path) -> tuple[dict, dict]:
    units = {}
    for f in sorted((cache / "augustana_x").glob("gre_*t*.html")):
        m = re.match(r"gre_(\d)t(\d\d)\.html", f.name)
        b, t = int(m.group(1)), int(m.group(2))
        title, chaps = augustana_title(f.read_text(encoding="latin-1"))
        rubric = re.sub(r"^Titulus\s+[IVXLC]+\.?\s*", "", title)
        uid = f"{b}.{t}"
        units[uid] = {
            "corpus": "decretals", "unit": uid,
            "title": f"X {uid} {rubric}".strip(),
            "passages": [{"id": f"{uid}.{c}", "label": f"X {uid}.{c}", "text": txt}
                         for c, txt in chaps if txt],
        }
    meta = {
        "id": "decretals",
        "title": "Decretals of Gregory IX (Liber Extra)",
        "language": "la",
        "edition": "Friedberg, Corpus Iuris Canonici II (Leipzig 1881), as transcribed "
                   "for the Bibliotheca Augustana by Angus Graham",
        "license": "public domain (Friedberg 1881 edition text); the Bibliotheca Augustana "
                   "states no licence for its transcriptions, used with attribution "
                   "(U. Harsch, Hochschule Augsburg; digital version Angus Graham)",
        "attribution": "Text: Bibliotheca Augustana, hs-augsburg.de/~harsch "
                       "(digital version: Angus Graham)",
        "url": f"{AUG_BASE}/gre_0000.html",
        "external_url_template": AUG_BASE + "/gre_{book}t{title:02d}.html",
        "unit_scheme": "book.title",
        "passage_scheme": "book.title.chapter",
        "quality": "clean",
        "scan_url_template": SCAN_VOL2,
    }
    return meta, units


# --- Sext and Clementines (archive.org OCR) --------------------------------

# running heads: 'SEXTI DECRETAL. LIB. I. TIT. III. DE RESCRIPTIS, c. 4' (OCR-mangled)
_RUNHEAD = re.compile(r"^.{0,6}?\b(SEX\S{1,3}|CLEMENTIN\S*|CLEM\.)\s+(DE|L\S{2,3}\.)",
                      re.I)
_TITLE = re.compile(r"^\s*T\s?I\s?T\s?U\s?L\s?U\s?S\s+(PRIMUS|UNICUS|[A-Za-z1|]{1,6}"
                    r"(?: [A-Za-z]{1,3})?)[.,]?\s*\*?\s*$", re.I)
_CAP = re.compile(r"^\s*(?:[Cc]\s?[Aa]\s?[PpIil1JVr]\S{0,2}|C\s?ap)[.,]?\s+"
                  r"(un|UN|[A-Za-z1|]{1,6})[.,»]?\**\s*$")
_LIBER = re.compile(r"^\s*LIBER\s+(PRIMUS|SECUNDUS|TERTIUS|QUARTUS|QUINTUS)\.?\s*$")
_FOOT = re.compile(r"(deest|^\s*(Tit\.|T\s?i\s?t\.|Cap\.?\s*[IVXLun]+\.?\s+[a1-9]\))"
                   r"|^\s*C\s?ap[.,]\s+[IVXLCHnmU]+\.\s+\S+\)|^\s*[a-z1-9]\)\s|Codd\.|"
                   r"\b[A-Z]{3,}[a-z]{0,6}\s*$|\b(Comp\.|Cone\.|Conc\.)\s)")
_BOOKS = {"PRIMUS": 1, "SECUNDUS": 2, "TERTIUS": 3, "QUARTUS": 4, "QUINTUS": 5}


def is_footnote_block(block: list[str]) -> bool:
    """Heuristic: an apparatus block (critical notes) rather than edition text."""
    joined = " ".join(block)
    if not joined.strip():
        return False
    hits = sum(1 for ln in block if _FOOT.search(ln))
    sigla = len(re.findall(r"\b[A-Z]{2,}[a-z]*\b\s*[;.:]?", joined))
    markers = len(re.findall(r"(?:^|\s)(?:[a-z]|\d{1,3})\)\s", joined))
    short_note = len(block) <= 3 and markers >= 1 and ":" in joined
    return (short_note or hits >= max(1, len(block) // 2) or markers >= 2 or
            re.search(r"\bde[ec]st\b|Codd\.", joined) is not None or (sigla >= 3 and markers >= 1))


def segment_ocr(lines: list[str], corpus: str) -> dict[str, dict]:
    """Segment Friedberg OCR text of one collection (already sliced) into
    {book.title: unit}, each with chapters. Running heads and apparatus blocks dropped."""
    blocks, cur = [], []
    for ln in lines:
        if not ln.strip():
            if cur:
                blocks.append(cur)
                cur = []
        elif _TITLE.match(ln) or _CAP.match(ln) or _LIBER.match(ln):
            if cur:                       # a heading always starts its own block
                blocks.append(cur)
            cur = [ln.rstrip()]
        else:
            cur.append(ln.rstrip())
    if cur:
        blocks.append(cur)

    units: dict[str, dict] = {}
    book, tit, cap = 1, None, None
    just_booked = True
    expect_rubric = False
    want_title_rubric = False
    buf: list[str] = []
    label = {"sext": "VI", "clementines": "Clem."}[corpus]

    def flush():
        nonlocal buf
        if tit is not None and cap is not None and buf:
            uid = f"{book}.{tit}"
            u = units[uid]
            text = clean_ocr(join_lines(buf))
            pid = f"{uid}.{cap}"
            for p in u["passages"]:
                if p["id"] == pid:
                    p["text"] += "\n\n" + text
                    break
            else:
                u["passages"].append({"id": pid, "label": f"{label} {pid}", "text": text})
        buf = []

    for blk in blocks:
        first = blk[0]
        if _RUNHEAD.search(first) and len(blk) <= 3:
            continue
        if re.fullmatch(r"\s*\d{1,4}\s*", first) and len(blk) == 1:
            continue                                      # bare column number
        m = _LIBER.match(first)
        if m:
            flush()
            book, tit, cap, just_booked = _BOOKS[m.group(1)], None, None, True
            continue
        m = _TITLE.match(first)
        if m:
            flush()
            v = m.group(1).strip()
            new = 1 if v.upper() in ("PRIMUS", "UNICUS") else plausible(v, tit)
            if new == 1 and tit is not None and not just_booked:
                book += 1                 # title numbering restarts: next book
            tit, cap, just_booked = new, None, False
            uid = f"{book}.{tit}"
            units.setdefault(uid, {"corpus": corpus, "unit": uid,
                                   "title": f"{label} {uid}", "passages": []})
            want_title_rubric = True
            continue
        m = _CAP.match(first)
        if m and tit is not None:
            flush()
            v = m.group(1).strip()
            cap = 1 if v.lower() == "un" else plausible(v, cap)
            expect_rubric = True
            want_title_rubric = False
            if len(blk) > 1:
                buf.extend(blk[1:])
            continue
        if want_title_rubric and tit is not None and first.isupper():
            units[f"{book}.{tit}"]["title"] += " " + " ".join(
                " ".join(blk).split()).title().replace(" Et ", " et ")
            want_title_rubric = False
            continue
        if is_footnote_block(blk):
            continue
        if cap is None:
            if tit is None or want_title_rubric:
                continue
            cap = 1                       # chapter heading lost in OCR
        if buf:
            buf.append("")
        # inline running head at the top of a block ('SEXTI DECRETAL. LIB. ...')
        if _RUNHEAD.search(first):
            blk = blk[1:]
        buf.extend(blk)
        expect_rubric = False
    flush()
    return units


def clean_ocr(text: str) -> str:
    """Drop footnote call marks the OCR glued to words ('obtinebas1*', 'VIII.*')."""
    text = re.sub(r"(?<=[A-Za-z.,;:])\d{0,2}\*+", "", text)
    return re.sub(r"[ \t]+([,;:.])", r"\1", text).replace("  ", " ")


def slice_vol2(text: str) -> dict[str, list[str]]:
    """Cut the vol. 2 OCR into the Sext and the Clementines."""
    lines = text.split("\n")
    def find(pat, start=0):
        rx = re.compile(pat)
        for i in range(start, len(lines)):
            if rx.search(lines[i]):
                return i
        return None
    s0 = find(r"^SEXTI\s+DECRETALIUM\s*$")
    c0 = find(r"^\s*CLEMENTIS\s+PAPAE\s+V\.?\s*$", s0 or 0)
    e0 = find(r"^\s*EXTRA\s?VAGANTES\s*$", c0 or s0 or 0)
    out = {}
    if s0 is not None and c0 is not None:
        out["sext"] = lines[s0:c0]
    if c0 is not None:
        out["clementines"] = lines[c0:e0]
    return out


def build_ocr(cache: pathlib.Path) -> dict[str, tuple[dict, dict]]:
    f = cache / "BD1141952_djvu.txt"
    parts = slice_vol2(f.read_text(encoding="utf-8", errors="replace"))
    res = {}
    for corpus, lines in parts.items():
        units = segment_ocr(lines, corpus)
        title = {"sext": "Liber Sextus of Boniface VIII",
                 "clementines": "Clementines (Constitutions of Clement V)"}[corpus]
        res[corpus] = ({
            "id": corpus, "title": title, "language": "la",
            "edition": "Friedberg, Corpus Iuris Canonici II (Leipzig 1881; repr. Graz "
                       "1955), machine OCR from archive.org (BD1141952)",
            "license": "public domain (edition text); OCR from the Internet Archive",
            "attribution": "Text: Internet Archive OCR of Friedberg vol. 2 (BD1141952), "
                           "uncorrected",
            "url": SCAN_VOL2,
            "unit_scheme": "book.title",
            "passage_scheme": "book.title.chapter",
            "quality": "ocr",
            "scan_url_template": SCAN_VOL2,
        }, units)
    return res


# --- writing ----------------------------------------------------------------

def unit_sort_key(u: str):
    m = re.match(r"(de-pen-|de-cons-)?([DC])\.(\d+)(?:-([a-z]))?$", u)
    if m:
        part = {"": 0, "de-pen-": 2, "de-cons-": 3}[m.group(1) or ""]
        if m.group(2) == "C" and not m.group(1):
            part = 1
        return (part, int(m.group(3)), m.group(4) or "")
    m = re.fullmatch(r"([\d.]+)(?:-([a-z]))?", u)
    if m:
        return tuple(int(x) for x in m.group(1).split(".")) + (m.group(2) or "",)
    return (99, u)


MAX_UNIT_BYTES = 300_000


def split_unit(u: dict, limit: int = MAX_UNIT_BYTES) -> list[dict]:
    """Split a unit whose JSON exceeds `limit` into <unit>-a, <unit>-b, ... at passage
    boundaries (the Vulgate corpus does the same; see its split_units)."""
    size = len(json.dumps(u, ensure_ascii=False).encode())
    if size <= limit:
        return [u]
    n = -(-size // limit)
    target = size / n
    parts, cur, acc = [], [], 0
    for p in u["passages"]:
        b = len(json.dumps(p, ensure_ascii=False).encode())
        if cur and acc + b > target and len(parts) < n - 1:
            parts.append(cur)
            cur, acc = [], 0
        cur.append(p)
        acc += b
    parts.append(cur)
    out = []
    for i, ps in enumerate(parts):
        suf = "abcdefghij"[i]
        out.append(dict(u, unit=f"{u['unit']}-{suf}", passages=ps,
                        title=f"{u['title']} ({suf})"))
    return out


def write_corpus(out: pathlib.Path, meta: dict, units: dict) -> dict:
    d = out / meta["id"]
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.json"):
        old.unlink()
    order, split, total = [], {}, 0
    for uid in sorted((u for u in units if units[u]["passages"]), key=unit_sort_key):
        parts = split_unit(units[uid])
        if len(parts) > 1:
            split[uid] = [{"unit": p["unit"],
                           "passage_range": [p["passages"][0]["id"], p["passages"][-1]["id"]]}
                          for p in parts]
        for u in parts:
            order.append(u["unit"])
            total += len(u["passages"])
            (d / f"{u['unit']}.json").write_text(
                json.dumps(u, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    meta = dict(meta, units=order, passages=total,
                status="text" if order else "scan-only")
    if split:
        meta["split_units"] = split
        meta["split_note"] = ("Units over ~300 KB are split at passage boundaries into "
                              "<unit>-a, <unit>-b; split_units gives each part's first "
                              "and last passage (passage_range).")
    (d / "corpus.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n",
                                   encoding="utf-8")
    size = sum(f.stat().st_size for f in d.glob("*.json"))
    print(f"{meta['id']}: {len(order)} units, {total} passages, {size / 1e6:.1f} MB")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["fetch", "build"])
    ap.add_argument("--only", choices=["decretum", "decretals", "ocr"])
    ap.add_argument("--cache", type=pathlib.Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    a.cache.mkdir(parents=True, exist_ok=True)
    if a.cmd == "fetch":
        if a.only in (None, "ocr"):
            fetch_ocr(a.cache)
        if a.only in (None, "decretals"):
            fetch_decretals(a.cache)
        if a.only in (None, "decretum"):
            fetch_decretum(a.cache)
        return 0
    if a.only in (None, "decretum"):
        write_corpus(a.out, *build_decretum(a.cache))
    if a.only in (None, "decretals"):
        write_corpus(a.out, *build_decretals(a.cache))
    if a.only in (None, "ocr"):
        for meta, units in build_ocr(a.cache).values():
            write_corpus(a.out, meta, units)
    return 0


if __name__ == "__main__":
    sys.exit(main())

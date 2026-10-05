#!/usr/bin/env python3
"""Build the Roman-law source corpora (digest, code, institutes, novels) for the site.

Source: The Roman Law Library, droitromain.univ-grenoble-alpes.fr (Y. Lassard, A. Koptev):
Mommsen-Krueger Digest, Krueger Code, Krueger Institutes, Schoell-Kroll Novels (Latin).
Each page is fetched once into a cache directory (sequential, polite delay); every later run
can rebuild from the cache with --offline. Output follows docs/sources-contract.md:

    site/data/sources/<corpus>/<unit>.json   unit files (< ~300 KB; oversize titles are split)
    site/data/sources/<corpus>/corpus.json   the corpus's index.json entry (merged elsewhere)
    site/data/sources/code/concordance.json  medieval-vulgate -> Krueger Code numbering

Usage:
    python3 scripts/fetch_roman_law.py [--offline] [--cache DIR] [--only digest,code]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site/data/sources"
DEFAULT_CACHE = Path("/private/tmp/claude-502/-Users-cdavis-github-translator/"
                     "d3de2614-6075-4c4c-8061-e08283898989/scratchpad/sources-cache/roman")
SITE = "https://droitromain.univ-grenoble-alpes.fr/"
BASE = SITE + "Corpus/"
UA = "Coras-translation-corpus-builder/1.0 (scholarly citation viewer; sequential, cached fetch)"
DELAY = 1.5
MAX_UNIT_BYTES = 300_000

LICENSE = "public domain (edition text); site courtesy of Université Grenoble Alpes"
ATTRIB = "Text: droitromain.univ-grenoble-alpes.fr (Y. Lassard, A. Koptev)"

CORPORA = {
    "digest": dict(title="Digest of Justinian", abbr="D.",
                   edition="Mommsen–Krüger text as published at droitromain.univ-grenoble-alpes.fr",
                   unit_scheme="book.title", passage_scheme="book.title.fragment[.paragraph]"),
    "code": dict(title="Code of Justinian", abbr="C.",
                 edition="Krüger text as published at droitromain.univ-grenoble-alpes.fr",
                 unit_scheme="book.title", passage_scheme="book.title.constitution[.paragraph]"),
    "institutes": dict(title="Institutes of Justinian", abbr="Inst.",
                       edition="Krüger text as published at droitromain.univ-grenoble-alpes.fr",
                       unit_scheme="book", passage_scheme="book.title[.paragraph]"),
    "novels": dict(title="Novels of Justinian", abbr="Nov.",
                   edition="Schöll–Kroll Latin text as published at droitromain.univ-grenoble-alpes.fr",
                   unit_scheme="novel", passage_scheme="novel.chapter"),
}


# ---------------------------------------------------------------- fetching

class Fetcher:
    def __init__(self, cache: Path, offline: bool):
        self.cache, self.offline = cache, offline
        self.cache.mkdir(parents=True, exist_ok=True)
        self._last = 0.0
        self.missing: list[str] = []

    def get(self, page: str, base: str | None = None) -> str | None:
        path = self.cache / page
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if self.offline:
                self.missing.append(page)
                return None
            wait = DELAY - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            req = urllib.request.Request((base or BASE) + page, headers={"User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
            except Exception as e:  # noqa: BLE001 - report, never guess text
                print(f"  fetch failed {page}: {e}", file=sys.stderr)
                self.missing.append(page)
                return None
            finally:
                self._last = time.time()
            path.write_bytes(data)
            print(f"  fetched {page} ({len(data)} bytes)", file=sys.stderr)
        return decode_page(path.read_bytes())


def decode_page(data: bytes) -> str:
    """Most pages are Latin-1/cp1252; a few Institutes pages are UTF-8 despite their meta."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def page_list(corpus: str, fetcher: Fetcher) -> list[str]:
    if corpus == "digest":
        return [f"d-{n:02d}.htm" for n in range(1, 51)]
    if corpus == "code":
        return [f"CJ{n}.htm" for n in range(1, 13)]
    if corpus == "institutes":
        return ["just_proem.htm"] + [f"just{n}.gr.htm" for n in range(1, 5)]
    if corpus == "novels":
        idx = fetcher.get("Novellae.htm") or ""
        nums = sorted({int(n) for n in re.findall(r'href="Nov(\d+)\.htm"', idx)})
        return [f"Nov{n:02d}.htm" if n < 10 else f"Nov{n}.htm" for n in nums]
    raise ValueError(corpus)


# ---------------------------------------------------------------- tokenising

TOKEN = re.compile(r"<!--.*?-->|<script\b.*?</script>|<style\b.*?</style>|<[^>]*>|[^<]+",
                   re.S | re.I)
BLOCK = {"p", "br", "div", "td", "tr", "li", "table", "hr", "ul", "h1", "h2", "h3", "h4"}
ITAL = {"i", "em"}
ANCHOR_NAME = re.compile(r"""\bname\s*=\s*["']?([^"'\s>]+)""", re.I)


def tokens(src: str):
    """Yield ('anchor', name) / ('end_anchor', None) / ('break', None) / ('ital', +1|-1) /
    ('text', str)."""
    for m in TOKEN.finditer(src):
        t = m.group(0)
        if not t.startswith("<"):
            yield ("text", html.unescape(t))
            continue
        if t.startswith("<!--") or t[:7].lower() in ("<script", "<style"):
            continue
        tm = re.match(r"<\s*(/?)\s*([a-zA-Z0-9]+)", t)
        if not tm:
            continue
        close, tag = tm.group(1) == "/", tm.group(2).lower()
        if tag == "a":
            if close:
                yield ("end_anchor", None)
            else:
                nm = ANCHOR_NAME.search(t)
                yield ("anchor", nm.group(1) if nm else None)
        elif tag in BLOCK:
            yield ("break", None)
        elif tag in ITAL:
            yield ("ital", -1 if close else 1)


def clean(s: str) -> str:
    s = s.replace("\xa0", " ").replace("�", "")
    s = re.sub(r"[ \t\r\n]+", " ", s)
    return s.strip()


def join_blocks(blocks: list[str]) -> str:
    parts = [clean(b) for b in blocks]
    parts = [p for p in parts if p and not re.fullmatch(r"[\s.,;:]*", p)]
    text = "\n\n".join(parts)
    text = re.sub(r"^[\s.]+", "", text)            # stray "." left after a number anchor
    return text.strip()


# ---------------------------------------------------------------- generic anchor parser

class Section:
    """A run of text that belongs to one anchor id."""

    def __init__(self, sid: str, kind: str):
        self.id, self.kind = sid, kind
        self.blocks: list[str] = [""]
        self.inscription: list[str] = []
        self.anchor_text = ""
        self.insc_closed = False

    def add(self, txt: str):
        self.blocks[-1] += txt

    def brk(self):
        if self.blocks[-1].strip():
            self.blocks.append("")

    @property
    def text(self) -> str:
        return join_blocks(self.blocks)


def scan(src: str, classify, stop_markers=()) -> list[Section]:
    """Walk the page, starting a new Section at every anchor that `classify(name)` maps to a
    kind. Anchor inner text (numbers like '48.5.39 (38)', 'pr.', 'CAPUT V.') is kept apart
    in `anchor_text`. Italic text that opens a fragment/constitution, before any roman text,
    is collected as its inscription."""
    for mk in stop_markers:
        i = src.find(mk)
        if i >= 0:
            src = src[:i]
    secs: list[Section] = []
    cur: Section | None = None
    in_anchor = False
    ital = 0
    for kind, val in tokens(src):
        if kind == "anchor":
            k = classify(val) if val else None
            if k:
                cur = Section(val, k)
                secs.append(cur)
                in_anchor = True
            continue
        if kind == "end_anchor":
            in_anchor = False
            continue
        if kind == "ital":
            ital = max(0, ital + val)
            continue
        if cur is None:
            continue
        if kind == "break":
            in_anchor = False
            if clean("".join(cur.inscription)):
                cur.insc_closed = True
            cur.brk()
            continue
        if in_anchor:
            cur.anchor_text += val
            continue
        if (ital and (cur.kind == "frag" or cur.id.endswith(".pr.")) and not cur.insc_closed
                and not re.sub(r"[\W_]+", "", "".join(cur.blocks))):
            cur.inscription.append(val)
            continue
        cur.add(val)
    return secs


# ---------------------------------------------------------------- Digest and Code

def _num(x: str) -> tuple:
    return tuple(int(p) if p.isdigit() else -1 for p in re.split(r"[.]", x))


SINGLE_TITLE_BOOKS = {30, 31, 32}
ROMAN_INSC = re.compile(r"^(?:[A-Z][\w]+\s+){1,4}(?:libro|libris|lib\.|notat|notis|ex|ad|de|in|"
                        r"singulari|responsorum|epistularum|quaestionum|digestorum)\b")


def parse_book_title_page(src: str, corpus: str, book: int) -> list[dict]:
    """Digest d-NN.htm / Code CJN.htm -> list of unit dicts (unsplit)."""
    b = str(book)
    single = corpus == "digest" and book in SINGLE_TITLE_BOOKS
    # single-title books (D. 30-32) number fragments book.fragment[.par] with no title level
    re_title = re.compile(r"^$" if single else rf"^{b}\.(\d+)$")
    re_frag = re.compile(rf"^{b}\.(\d+)$" if single else rf"^{b}\.(\d+)\.(\d+)$")
    re_par = re.compile(rf"^{b}\.(\d+)\.(pr\.?|\d+)$" if single
                        else rf"^{b}\.(\d+)\.(\d+)\.(pr\.?|\d+)$")
    depth = 2 if single else 3            # components in a fragment id
    toc = {f"{b}.{t}": clean(n).rstrip(" .") for t, n in
           re.findall(rf"""<a\s+href=["']?#\d+["']?[^>]*>\s*(?:<[^>]+>\s*)*{b}\.(\d+)\.0\.\s*([^<]+)""",
                      src)}

    def classify(name):
        if re_title.match(name):
            return "title"
        if re_frag.match(name):
            return "frag"
        if re_par.match(name):
            return "par"
        return None

    abbr = CORPORA[corpus]["abbr"]
    units: dict[str, dict] = {}
    order: list[str] = []
    frag_meta: dict[str, dict] = {}
    for s in scan(src, classify):
        if s.kind == "title":
            uid = s.id
            name = clean(" ".join(s.blocks)).rstrip(" .") or toc.get(uid, "")
            if uid not in units:
                units[uid] = {"corpus": corpus, "unit": uid,
                              "title": f"{abbr} {uid} {name}".strip(), "passages": []}
                order.append(uid)
            continue
        parts = s.id.split(".")
        uid = b if single else ".".join(parts[:2])
        if uid not in units:  # no title heading anchor on the page: name from the contents list
            if single:
                m0 = re.search(rf"{b}\.0\.\s*([^<]+)", src)
                name = clean(m0.group(1)).rstrip(" .") if m0 else ""
            else:
                name = toc.get(uid, "")
            units[uid] = {"corpus": corpus, "unit": uid,
                          "title": f"{abbr} {uid} {name}".strip(), "passages": []}
            order.append(uid)
        u = units[uid]
        if s.kind == "frag" and corpus == "digest" and not clean(" ".join(s.inscription)):
            # some pages set the inscription in roman type: take a short leading block that
            # reads like one ("Gaius libro 3 ad edictum provinc.")
            blocks = [x for x in s.blocks if clean(x) and re.sub(r"[\W_]+", "", x)]
            while blocks and re.fullmatch(r"\(\s*[\d,\s]+[a-z]?\s*\)", clean(blocks[0])):
                s.anchor_text += " " + clean(blocks[0])       # second numbering, e.g. (52,30)
                s.blocks = s.blocks[s.blocks.index(blocks[0]) + 1:] or [""]
                blocks = blocks[1:]
            if blocks and len(clean(blocks[0])) < 160 and ROMAN_INSC.match(clean(blocks[0])):
                s.inscription = [blocks[0]]
                s.blocks = s.blocks[s.blocks.index(blocks[0]) + 1:] or [""]
        if s.kind == "par" and s.id.endswith(".pr.") and clean(" ".join(s.inscription)):
            fid0 = ".".join(parts[:depth])
            if not frag_meta.get(fid0, {}).get("inscription"):
                frag_meta.setdefault(fid0, {"alt": None})["inscription"] = \
                    clean(" ".join(s.inscription)).rstrip(" .").strip()
        if s.kind == "frag":
            insc = clean(" ".join(s.inscription)).rstrip(" .").strip()
            alt = re.search(r"\((\d+[a-z]?(?:,\s*\d+)?)\)", clean(s.anchor_text))
            frag_meta[s.id] = {"inscription": insc, "alt": alt.group(1) if alt else None}
            txt = s.text
            u["passages"].append({"id": s.id, "_frag": s.id, "text": txt})
        else:
            par = parts[depth].rstrip(".")
            fid = ".".join(parts[:depth])
            pid = f"{fid}.{par}"
            # text sitting directly under the fragment header becomes its principium
            prev = u["passages"][-1] if u["passages"] else None
            if prev is not None and prev["id"] == fid:
                if prev["text"] and par != "pr":
                    prev["id"] = f"{fid}.pr"
                elif not prev["text"]:
                    u["passages"].pop()
                elif par == "pr":  # text before an explicit pr anchor: fold into pr
                    u["passages"].pop()
                    s.blocks.insert(0, prev["text"])
            u["passages"].append({"id": pid, "_frag": fid, "text": s.text})
    out = []
    for uid in order:
        u = units[uid]
        seen = set()
        passages = []
        for p in u["passages"]:
            fid, ref = p["_frag"], p["id"]
            meta = frag_meta.get(fid, {})
            label = f"{abbr} {ref[:-3]} pr." if ref.endswith(".pr") else f"{abbr} {ref}"
            text = p["text"]
            greek = bool(GREEK_GAP.search(text))
            if greek:
                text = GREEK_GAP.sub("", text).strip()
            if not text or re.fullmatch(r"[.\s]*", text):
                u.setdefault("_empty", []).append(ref)  # no text at the source: not stored
                continue
            q = {"id": ref, "label": label, "text": text}
            if greek:
                q["greek_not_online"] = True
            if fid not in seen:
                seen.add(fid)
                who = author_of(meta.get("inscription", ""), corpus)
                if who:
                    q["label"] += f" ({who})"
                if meta.get("inscription"):
                    q["inscription"] = meta["inscription"]
                if meta.get("alt"):
                    q["alt_number"] = meta["alt"]
            passages.append(q)
        u["passages"] = passages
        out.append(u)
    return out


GREEK_GAP = re.compile(r"\[\s*Here there is a Greek text\.?[^\]]*\]\s*", re.I)


def author_of(insc: str, corpus: str) -> str:
    insc = clean(insc).strip(" .")
    if not insc:
        return ""
    if corpus == "digest":
        # "Papinianus libro 36 quaestionum" -> "Papinianus"; "Idem libro..." kept as Idem
        m = re.search(r"\b([A-Z]\w+)\s+notat\b", insc)     # "... Marcellus notat"
        if m:
            return m.group(1)
        # the jurist's name is the run of capitalised words before "libro", "notat", ...
        m = re.match(r"((?:[A-Z][\w]*\.?\s*)+?)(?=\s+[a-z0-9]|$)", insc)
        return (m.group(1) if m else insc.split()[0]).strip(" ,.")
    # Code: "Imperatores Severus, Antoninus" -> "Severus, Antoninus"
    who = re.sub(r"^(Imperator(?:es)?|Impp?\.|Augusti?)\s+", "", insc, flags=re.I)
    return clean(who).strip(" ,.") or insc


# ---------------------------------------------------------------- Institutes

def parse_institutes_page(src: str, book: str) -> dict:
    """just{N}.gr.htm -> one unit for book N; just_proem.htm -> unit 'proem'."""
    if book == "proem":
        def classify(name):
            if re.fullmatch(r"(?:proem|prooem)[\w.]*", name, re.I):
                return "par"
            return None
        secs = scan(src, classify)
        passages = []
        for s in secs:
            m = re.search(r"\.(pr|\d+)\.?$", s.id)
            if not m:
                continue
            pid = f"proem.{m.group(1)}"
            passages.append({"id": pid, "label": f"Inst. proem. {m.group(1)}", "text": s.text})
        return {"corpus": "institutes", "unit": "proem",
                "title": "Inst. proem. Imperatoriam maiestatem", "passages": passages}
    b = book
    re_title = re.compile(rf"^{b}\.(\d+)$")
    re_par = re.compile(rf"^{b}\.(\d+)\.(pr\.?|\d+)$")

    def classify(name):
        if re_title.match(name):
            return "title"
        if re_par.match(name):
            return "par"
        return None

    passages, titles = [], {}
    for s in scan(src, classify):
        if s.kind == "title":
            nm = clean(" ".join(s.blocks)).rstrip(" .")
            nm = re.sub(r"^(?:TIT\.?\s*[\dIVXLC]+\.?\s*)", "", nm, flags=re.I)
            titles[s.id] = nm[:1] + nm[1:].lower() if nm.isupper() else nm
            continue
        bt, par = s.id.rsplit(".", 1) if not s.id.endswith(".") else s.id[:-1].rsplit(".", 1)
        par = par.rstrip(".")
        pid = f"{bt}.{par}"
        label = f"Inst. {bt} pr." if par == "pr" else f"Inst. {pid}"
        passages.append({"id": pid, "label": label, "text": s.text})
    # a title heading without its own anchor ("TIT. 1 ... DE IUSTITIA ET IURE.")
    plain = clean(" ".join(v for k, v in tokens(src) if k == "text"))
    for t, nm in re.findall(r"TIT\.\s*(\d+)\s+(?!TIT\b)([A-Z][A-Z ,]+?)\.", plain):
        titles.setdefault(f"{b}.{t}", nm[:1] + nm[1:].lower())
    titles = dict(sorted(titles.items(), key=lambda kv: int(kv[0].split(".")[1])))
    return {"corpus": "institutes", "unit": b, "title": f"Inst. {b}", "passages": passages,
            "titles": titles}


# ---------------------------------------------------------------- Novels

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def parse_novel_page(src: str, nov: int) -> dict:
    n = str(nov)
    re_any = re.compile(rf"^{n}\.(praefatio|epilogus|\d+)(?:\.(pr\.?|\d+))?$")

    def classify(name):
        return "par" if re_any.match(name) else None

    # header: "~ NOV. XC ~  DE TESTIBUS.  ( AD 539 )" then inscription, before first anchor
    head_src = src
    first = re.search(rf"""name=["']?{n}\.""", src)
    if first:
        head_src = src[:first.start()]
    head = clean(" ".join(v for k, v in tokens(head_src) if k == "text"))
    m = re.search(r"NOV\.\s*[CLXVI]+\s*~?\s*(.*?)\s*\(\s*AD\s*(\d*)\s*\??\s*\)", head)
    rubric = clean(m.group(1)).strip(" ~") if m else ""
    year = m.group(2) if m and m.group(2).isdigit() else None
    if m and "?" in m.group(0):
        year = None  # date marked uncertain at the source
    # the body starts after the "( Based upon the Latin text ... )" credit box, closed by <hr>
    cb = src.find("Based upon")
    body_start = src.find("<hr", cb) if cb >= 0 else -1
    stop = min([i for i in (src.find("&#9658;"), src.find("\u25ba")) if i >= 0] or [len(src)])
    lead_end = src.rfind("<", 0, first.start()) if first else stop
    lead_blocks = []
    if body_start >= 0:
        cur = ""
        for k, v in tokens(src[body_start:max(body_start, lead_end)]):
            if k == "text":
                cur += v
            elif k == "break" and clean(cur):
                lead_blocks.append(clean(cur))
                cur = ""
        if clean(cur):
            lead_blocks.append(clean(cur))
    lead_blocks = [x for x in lead_blocks
                   if not re.fullmatch(r"[~\s]*|Translated from the Greek|.*Ingo\s+Maier.*", x)]
    secs = scan(src, classify, stop_markers=("&#9658;", "\u25ba"))
    chapters: dict[str, dict] = {}
    order: list[str] = []
    for s in secs:
        mm = re_any.match(s.id)
        ch, par = mm.group(1), (mm.group(2) or "").rstrip(".")
        cid = {"praefatio": "pr", "epilogus": "epilogus"}.get(ch, ch)
        pid = f"{n}.{cid}"
        if pid not in chapters:
            lab = {"pr": f"Nov. {n} praef.", "epilogus": f"Nov. {n} epil."}.get(cid, f"Nov. {n}.{cid}")
            chapters[pid] = {"id": pid, "label": lab, "_parts": []}
            order.append(pid)
        txt = s.text
        txt = re.sub(r"^<?\s*(Praefatio|Epilogus)\.?\s*>\s*", "", txt, flags=re.I)
        txt = re.sub(r"^>\s*", "", txt)
        if par and par != "pr" and txt:
            txt = f"{par}. {txt}"
        if txt:
            chapters[pid]["_parts"].append(txt)
    inscription = ""
    if (lead_blocks and 3 < len(lead_blocks[0]) < 200 and ". . ." not in lead_blocks[0]
            and (order or len(lead_blocks) > 1)):
        inscription = lead_blocks.pop(0)
    if order and lead_blocks and f"{n}.pr" not in chapters:
        # an unanchored praefatio between the inscription and chapter 1
        parts = [re.sub(r"^<?\s*Praefatio\s*>?\.?\s*", "", x, flags=re.I) for x in lead_blocks]
        chapters[f"{n}.pr"] = {"id": f"{n}.pr", "label": f"Nov. {n} praef.", "_parts": parts}
        order.insert(0, f"{n}.pr")
    elif not order and lead_blocks:
        # novel without chapter anchors at the source: the whole body is one passage
        chapters[f"{n}.pr"] = {"id": f"{n}.pr", "label": f"Nov. {n}", "_parts": lead_blocks}
        order.append(f"{n}.pr")
    passages = []
    for pid in order:
        c = chapters[pid]
        passages.append({"id": c["id"], "label": c["label"], "text": "\n\n".join(c.pop("_parts"))})
    rubric = rubric.rstrip(" .")
    if rubric.isupper():
        rubric = rubric[:1] + rubric[1:].lower()
    unit = {"corpus": "novels", "unit": n, "title": f"Nov. {n} {rubric}".strip(),
            "passages": passages}
    if inscription:
        unit["inscription"] = inscription
    if not first:
        unit["no_chapters"] = True
    if year:
        unit["year"] = int(year)
    return unit


# ---------------------------------------------------------------- splitting, validating

def unit_bytes(u: dict) -> int:
    return len(json.dumps(u, ensure_ascii=False).encode("utf-8"))


def split_unit(u: dict, limit: int = MAX_UNIT_BYTES) -> list[dict]:
    """Split an oversize unit at fragment boundaries into u.a, u.b, ... each under limit."""
    if unit_bytes(u) <= limit:
        return [u]
    groups: list[list[dict]] = []
    cur: list[dict] = []
    cur_frag = None
    frag_block: list[dict] = []
    blocks: list[list[dict]] = []
    for p in u["passages"]:
        f = ".".join(p["id"].split(".")[:len(u["unit"].split(".")) + 1])
        if f != cur_frag and frag_block:
            blocks.append(frag_block)
            frag_block = []
        cur_frag = f
        frag_block.append(p)
    if frag_block:
        blocks.append(frag_block)
    shell = unit_bytes({**u, "passages": []}) + 200
    size = shell
    for blk in blocks:
        bsz = sum(len(json.dumps(p, ensure_ascii=False).encode()) + 2 for p in blk)
        if cur and size + bsz > limit:
            groups.append(cur)
            cur, size = [], shell
        cur.extend(blk)
        size += bsz
    if cur:
        groups.append(cur)
    out = []
    for i, g in enumerate(groups):
        suffix = "abcdefghijklmnopqrstuvwxyz"[i]
        out.append({**u, "unit": f"{u['unit']}{suffix}", "parent_unit": u["unit"],
                    "title": u["title"] + f" (part {i + 1} of {len(groups)})",
                    "passage_range": [g[0]["id"], g[-1]["id"]], "passages": g})
    return out


def validate_unit(u: dict, corpus: str) -> list[str]:
    errs = []
    for k in ("corpus", "unit", "title", "passages"):
        if k not in u:
            errs.append(f"{u.get('unit')}: missing {k}")
    if u.get("corpus") != corpus:
        errs.append(f"{u.get('unit')}: corpus mismatch")
    ids = [p.get("id") for p in u.get("passages", [])]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        errs.append(f"{u.get('unit')}: duplicate passage ids {dup[:5]}")
    if not ids:
        errs.append(f"{u.get('unit')}: no passages")
    for p in u.get("passages", []):
        if not all(isinstance(p.get(k), str) for k in ("id", "label", "text")):
            errs.append(f"{u.get('unit')}: passage {p.get('id')} lacks id/label/text")
        elif re.search(r"</(?:p|a|font|span|strong|b|i|em|div|td)>|<a\s+(?:href|name)\s*="
                       r"|<(?:p|br|font|span|strong|div|td|tr|table|hr|li|ul|em)\b[^>]*>",
                       p["text"], re.I):
            errs.append(f"{u.get('unit')}: markup in {p['id']}")
    if unit_bytes(u) > MAX_UNIT_BYTES:
        errs.append(f"{u.get('unit')}: {unit_bytes(u)} bytes > {MAX_UNIT_BYTES}")
    return errs


def unit_sort_key(uid: str):
    return tuple((int(x) if x.isdigit() else 10 ** 6, re.sub(r"\d", "", x))
                 for x in re.findall(r"\d+|[a-z]+", uid))


def write_corpus(corpus: str, units: list[dict], notes: dict) -> dict:
    d = OUT / corpus
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.json"):
        if old.name not in ("concordance.json",):
            old.unlink()
    final, splits, errs = [], {}, []
    for u in units:
        parts = split_unit(u) if corpus in ("digest", "code") else [u]
        if len(parts) > 1:
            splits[u["unit"]] = [{"unit": p["unit"], "passage_range": p["passage_range"]}
                                 for p in parts]
        final.extend(parts)
    for u in final:
        errs += validate_unit(u, corpus)
        (d / f"{u['unit']}.json").write_text(json.dumps(u, ensure_ascii=False, indent=0) + "\n",
                                            encoding="utf-8")
    meta = CORPORA[corpus]
    entry = {"id": corpus, "title": meta["title"], "language": "la", "edition": meta["edition"],
             "license": LICENSE, "attribution": ATTRIB,
             "url": "https://droitromain.univ-grenoble-alpes.fr/",
             "unit_scheme": meta["unit_scheme"], "passage_scheme": meta["passage_scheme"],
             "units": [u["unit"] for u in sorted(final, key=lambda x: unit_sort_key(x["unit"]))]}
    if splits:
        entry["split_units"] = splits
        entry["split_note"] = ("Titles larger than ~300 KB are split at fragment boundaries into "
                               "<unit>a, <unit>b, ...; split_units maps the book.title to its "
                               "parts with each part's first and last passage id (passage_range).")
    entry.update(notes)
    (d / "corpus.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1) + "\n",
                                   encoding="utf-8")
    entry["_stats"] = {"units": len(final), "passages": sum(len(u["passages"]) for u in final),
                       "bytes": sum(f.stat().st_size for f in d.glob("*.json")), "errors": errs}
    return entry


# ---------------------------------------------------------------- main

def build(corpus: str, fetcher: Fetcher) -> dict:
    units, failures, empty_at_source, no_text_units = [], [], [], []
    for page in page_list(corpus, fetcher):
        src = fetcher.get(page)
        if src is None:
            failures.append(f"{page}: not fetched")
            continue
        try:
            if corpus in ("digest", "code"):
                book = int(re.search(r"(\d+)", page).group(1))
                got = parse_book_title_page(src, corpus, book)
            elif corpus == "institutes":
                book = "proem" if "proem" in page else re.search(r"just(\d)", page).group(1)
                got = [parse_institutes_page(src, book)]
            else:
                got = [parse_novel_page(src, int(re.search(r"(\d+)", page).group(1)))]
        except Exception as e:  # noqa: BLE001
            failures.append(f"{page}: parse error {e!r}")
            continue
        for u in got:
            if not u["passages"]:
                no_text_units.append(u["unit"])
                empty_at_source.extend(u.pop("_empty", []))
        had_any = bool(got)
        got = [u for u in got if u["passages"]]
        if not got and not had_any:
            failures.append(f"{page}: no passages parsed")
        for u in got:
            empty_at_source.extend(u.pop("_empty", []))
            empty = [p["id"] for p in u["passages"] if not p["text"]]
            if empty:
                failures.append(f"{page}: {u['unit']} empty passages {empty[:8]}"
                                + (" ..." if len(empty) > 8 else ""))
        units.extend(got)
    notes = {}
    if no_text_units:
        notes["units_without_text"] = no_text_units
        notes["units_without_text_note"] = ("Titles listed at the source with no Latin text "
                                            "(Greek constitutions not online, or a lacuna); "
                                            "no unit file is written for them.")
    if empty_at_source:
        notes["empty_at_source"] = empty_at_source
        notes["empty_note"] = ("Numbers present at the source page with no text (lost, Greek, or "
                               "dotted lacuna); no passage is stored for them.")
    if corpus == "code":
        notes["greek_note"] = ("Passages flagged greek_not_online: the source page has a "
                               "placeholder for a Greek constitution; only the subscription, if "
                               "any, is stored.")
    if corpus == "novels":
        notes["passage_note"] = ("One passage per chapter: <novel>.pr is the praefatio, "
                                 "<novel>.epilogus the epilogue; chapter paragraphs are kept "
                                 "inside the chapter text, numbered '1. ', '2. ' and separated "
                                 "by blank lines.")
        idx = fetcher.get("Novellae.htm") or ""
        have = {int(x) for x in re.findall(r'href="Nov(\d+)\.htm"', idx)}
        notes["absent"] = [n for n in range(1, 169) if n not in have]
        notes["absent_note"] = "Novels with no Latin text at the source site."
        notes["no_chapters_note"] = ("Units flagged no_chapters have no chapter anchors at the "
                                     "source; their whole body is the single passage <novel>.pr.")
    if corpus in ("digest", "code"):
        if corpus == "digest":
            notes["single_title_books"] = sorted(SINGLE_TITLE_BOOKS)
            notes["single_title_note"] = ("Books 30-32 (De legatis et fideicommissis I-III) have "
                                          "one title each; their unit is the book ('30') and "
                                          "passage ids are book.fragment[.par] ('30.39.pr'), as "
                                          "in Mommsen's numbering.")
        notes["passage_note"] = ("A fragment/constitution without numbered paragraphs is one "
                                 "passage with the bare id (48.1.1); otherwise 'pr' is the "
                                 "principium. The first passage of each fragment carries "
                                 "'inscription' (and 'alt_number' where the page gives a "
                                 "second number in parentheses).")
    if corpus == "institutes":
        notes["passage_note"] = ("Unit 'proem' holds the constitutio Imperatoriam (proem.pr, "
                                 "proem.1...). Each book unit carries 'titles' (book.title -> "
                                 "rubric).")
    entry = write_corpus(corpus, units, notes)
    entry["_stats"]["failures"] = failures
    entry["_units"] = units
    return entry


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="use the cache only, no network")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--only", default="digest,code,institutes,novels")
    ap.add_argument("--fetch-only", action="store_true", help="fill the cache, do not parse")
    a = ap.parse_args(argv)
    fetcher = Fetcher(a.cache, a.offline)
    corpora = [c for c in a.only.split(",") if c]
    if a.fetch_only:
        for c in corpora:
            for p in page_list(c, fetcher):
                fetcher.get(p)
        print("missing:", fetcher.missing)
        return 0
    bad = 0
    for c in corpora:
        e = build(c, fetcher)
        st = e["_stats"]
        print(f"{c}: {st['units']} units, {st['passages']} passages, "
              f"{st['bytes'] / 1e6:.2f} MB")
        for f in st["failures"]:
            print(f"  FAIL {f}")
        for er in st["errors"]:
            print(f"  INVALID {er}")
        bad += len(st["errors"])
        if c == "code":
            conc, cfail = build_concordance(e["_units"], fetcher)
            write_concordance(conc)
            print(f"code concordance: {len(conc) - 1} vulgate numbers differ from Krueger")
            for f in cfail:
                print(f"  FAIL {f}")
    return 1 if bad else 0


# ---------------------------------------------------------------- Code concordance
#
# Coras cites the Code in the medieval vulgate numbering (the glossed Codex). The S. P. Scott
# English translation (1932), also on droitromain (Anglica/CJ{n}_Scott*.htm), follows the
# Gothofredus / vulgate numbering. Each constitution in both names its addressee; aligning the
# per-title sequence of addressees (Krueger subscription "* SEV. ET ANT. AA. CASSIAE. *"
# against Scott's heading "... to Cassia.") with a longest-common-subsequence match, with a
# loose date check, gives the vulgate number -> Krueger number pairs. Only pairs that differ
# are stored.

SCOTT_PAGES = ["CJ1_Scott.htm", "CJ2_Scott.htm", "CJ3_Scott.htm", "CJ4_Scott.htm",
               "CJ5_Scott.htm", "CJ6_Scott.htm", "CJ7_Scott.htm", "CJ8_Scott.gr.htm",
               "CJ9_Scott.gr.htm", "CJ10_Scott.gr.html", "CJ11_Scott.gr.html",
               "CJ12_Scott.gr.html"]


def _norm_name(x: str) -> str:
    x = x.lower()
    if x.startswith(("pop", "peop")):
        return "pop"
    x = x.replace("j", "i").replace("v", "u").replace("ph", "f").replace("y", "i")
    x = x.replace("ch", "c").replace("th", "t")
    return re.sub(r"[^a-z]", "", x)[:3]


SCOTT_SKIP = {"the", "soldier", "soldiers", "veteran", "praetorian", "prefect", "decurion",
              "most", "illustrious", "his", "her", "a", "an", "all", "count", "master"}


def parse_scott_book(src: str) -> dict[int, list[dict]]:
    """Scott page -> {title: [{num, year, to}]} (to = first letters of the addressee)."""
    out: dict[int, list[dict]] = {}
    pieces = re.split(r"""<a\s+name=["']?(\d+)["']?[^>]*>""", src)
    for i in range(1, len(pieces) - 1, 2):
        t = int(pieces[i])
        text = clean(html.unescape(re.sub(r"<[^>]+>", " ", pieces[i + 1])))
        cons = []
        heads = list(re.finditer(r"(?:^|\s)(\d{1,3})\.\s+(?=The (?:Emperors?|Same|Emperor)\b|"
                                 r"This [Ll]aw|The Divine|Emperor\b)", text))
        for j, m in enumerate(heads):
            body = text[m.end():heads[j + 1].start() if j + 1 < len(heads) else len(text)]
            yr = re.findall(r"\b([1-5]\d\d)\s*\.?\s*$", body.strip()[-40:])
            to = None
            hm = re.match(r"[^.]*?\bto\s+([^.]{0,80})", body[:220])
            if hm:
                for w in re.findall(r"[A-Za-z]+", hm.group(1)):
                    if w.lower() not in SCOTT_SKIP:
                        to = _norm_name(w)
                        break
            cons.append({"num": int(m.group(1)), "year": yr[-1] if yr else None, "to": to})
        if cons and t not in out:
            out[t] = cons
    return out


def kruger_constitutions(units: list[dict]) -> dict[tuple, list[dict]]:
    """Krueger Code units -> {(book, title): [{num, year, to}]} from the subscriptions."""
    out: dict[tuple, list[dict]] = {}
    for u in units:
        b, t = (int(x) for x in u["unit"].split(".")[:2])
        by: dict[int, list[str]] = {}
        for p in u["passages"]:
            by.setdefault(int(p["id"].split(".")[2]), []).append(p["text"])
        cons = []
        for num in sorted(by):
            txt = " ".join(by[num])
            yr = re.findall(r"<\s*a\.?\s*([1-5]\d\d)\b", txt, re.I)
            sub = re.findall(r"\*\s*([^*<]{3,160}?)\s*\*\s*<", txt)
            to = None
            if sub:
                tail = re.split(r"\b(?:AAA|AA|A|CCC|CC|C|DD)\.\s*", sub[-1])
                if len(tail) > 1:
                    w = [x for x in re.findall(r"[A-Za-z]+", tail[-1]) if x.upper() not in ("ET", "AD")]
                    to = _norm_name(w[0]) if w else None
            cons.append({"num": num, "year": yr[-1] if yr else None, "to": to})
        out[(b, t)] = cons
    return out


def build_concordance(units: list[dict], fetcher: "Fetcher") -> tuple[dict, list[str]]:
    import difflib
    kr = kruger_constitutions(units)
    mapping: dict[str, str] = {}
    fails: list[str] = []
    checked = 0
    for page in SCOTT_PAGES:
        b = int(re.search(r"CJ(\d+)", page).group(1))
        src = fetcher.get("Anglica/" + page, base=SITE)
        if src is None:
            fails.append(f"Anglica/{page}: not fetched")
            continue
        sc_book = parse_scott_book(src)
        for t, sc in sc_book.items():
            k = kr.get((b, t))
            if not k:
                continue
            checked += 1
            # align on the addressee; a constitution with none never matches
            ky = [c["to"] or f"k{c['num']}" for c in k]
            sy = [c["to"] or f"s{c['num']}" for c in sc]
            sm = difflib.SequenceMatcher(None, ky, sy, autojunk=False)
            for blk in sm.get_matching_blocks():
                for d in range(blk.size):
                    kc, scn = k[blk.a + d], sc[blk.b + d]
                    if kc["year"] and scn["year"] and abs(int(kc["year"]) - int(scn["year"])) > 3:
                        continue  # same addressee, dates far apart: not a safe pair
                    if kc["num"] != scn["num"]:
                        mapping[f"{b}.{t}.{scn['num']}"] = f"{b}.{t}.{kc['num']}"
    conc = {
        "_note": ("Vulgate (glossed Codex / Gothofredus) constitution number -> Krueger number, "
                  "only where they differ. Method: per title, the sequence of constitution "
                  "addressees in Krueger (subscriptions on droitromain) is aligned with the "
                  "sequence in S. P. Scott's translation (droitromain Anglica/, vulgate "
                  "numbering) by longest common subsequence; matched pairs are kept unless both "
                  "give a year and the years differ by more than 3 (Scott's dates often differ "
                  "from Krueger's by a year). Paragraph numbers carry over unchanged: apply the "
                  "map to book.title.constitution and keep the rest of the id. Pairs rest on "
                  "addressee agreement and should be checked against the text when a citation "
                  "matters. C. 9.9.30 (Quamvis adulterii) -> Krueger 9.9.29 is the reference "
                  "case."),
    }
    conc["_note"] += f" Titles compared: {checked}."
    conc.update(sorted(mapping.items(), key=lambda kv: [int(x) for x in kv[0].split(".")]))
    return conc, fails


def write_concordance(conc: dict):
    d = OUT / "code"
    d.mkdir(parents=True, exist_ok=True)
    (d / "concordance.json").write_text(json.dumps(conc, ensure_ascii=False, indent=1) + "\n",
                                        encoding="utf-8")

if __name__ == "__main__":
    sys.exit(main())

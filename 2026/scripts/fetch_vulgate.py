#!/usr/bin/env python3
"""Build the Clementine Vulgate corpus for the site (docs/sources-contract.md).

Source: the Clementine Vulgate Project text (M. Tweedale et al., VulSearch 4.x plain-text
files, public domain), mirrored with the project's own README in
github.com/jrichter/ClementineVulgateConverter (latin/*.lat), pinned to one commit.
One file per book, one verse per line: "chapter:verse text", cp1252, with markup
'\\' paragraph, '[' ']' verse-setting, '/' line break, '<Name>' speaker / Hebrew letter.

Output: site/data/sources/vulgate/<unit>.json, one unit per book (unit id is a slug such as
'1-kings'; passage ids are 'Book chapter:verse', e.g. 'Genesis 17:5', '1 Kings 3:5',
'Psalms 9:21' in Vulgate numbering) and vulgate/corpus.json (index entry).
A book over ~300 KB is split by chapter range into <unit>-a, <unit>-b (recorded in corpus.json).

Usage: python3 scripts/fetch_vulgate.py [--offline] [--cache DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site/data/sources/vulgate"
DEFAULT_CACHE = Path("/private/tmp/claude-502/-Users-cdavis-github-translator/"
                     "d3de2614-6075-4c4c-8061-e08283898989/scratchpad/sources-cache/vulgate")
COMMIT = "38292f3f91e874d3db9e7e9bf7abd23da3217054"
RAW = f"https://raw.githubusercontent.com/jrichter/ClementineVulgateConverter/{COMMIT}/latin/"
UA = "Coras-translation-corpus-builder/1.0 (scholarly citation viewer; sequential, cached fetch)"
MAX_UNIT_BYTES = 300_000

# (source file, passage-id book name, Latin Vulgate name, common citation abbreviations)
BOOKS = [
    ("Gn", "Genesis", "Genesis", ["Gen.", "Gn."]),
    ("Ex", "Exodus", "Exodus", ["Exod.", "Ex."]),
    ("Lv", "Leviticus", "Leviticus", ["Lev.", "Lv."]),
    ("Nm", "Numbers", "Numeri", ["Num.", "Nm."]),
    ("Dt", "Deuteronomy", "Deuteronomium", ["Deut.", "Dt."]),
    ("Jos", "Joshua", "Josue", ["Jos.", "Iosue"]),
    ("Jdc", "Judges", "Judicum", ["Jud.", "Judic.", "Jdc."]),
    ("Rt", "Ruth", "Ruth", ["Ruth"]),
    ("1Rg", "1 Kings", "Regum I", ["1 Reg.", "I Reg.", "1 Sam."]),
    ("2Rg", "2 Kings", "Regum II", ["2 Reg.", "II Reg.", "2 Sam."]),
    ("3Rg", "3 Kings", "Regum III", ["3 Reg.", "III Reg."]),
    ("4Rg", "4 Kings", "Regum IV", ["4 Reg.", "IV Reg."]),
    ("1Par", "1 Chronicles", "Paralipomenon I", ["1 Par.", "I Paralip."]),
    ("2Par", "2 Chronicles", "Paralipomenon II", ["2 Par.", "II Paralip."]),
    ("Esr", "Ezra", "Esdrae I", ["1 Esdr.", "Esdr."]),
    ("Neh", "Nehemiah", "Nehemiae (Esdrae II)", ["Neh.", "2 Esdr."]),
    ("Tob", "Tobit", "Tobiae", ["Tob."]),
    ("Jdt", "Judith", "Judith", ["Judith", "Jdt."]),
    ("Est", "Esther", "Esther", ["Esth."]),
    ("Job", "Job", "Job", ["Job"]),
    ("Ps", "Psalms", "Psalmi", ["Ps.", "Psal."]),
    ("Pr", "Proverbs", "Proverbia", ["Prov.", "Pr."]),
    ("Ecl", "Ecclesiastes", "Ecclesiastes", ["Eccl.", "Eccle."]),
    ("Ct", "Song of Songs", "Canticum Canticorum", ["Cant.", "Ct."]),
    ("Sap", "Wisdom", "Sapientia", ["Sap.", "Wis."]),
    ("Sir", "Ecclesiasticus", "Ecclesiasticus", ["Eccli.", "Sir."]),
    ("Is", "Isaiah", "Isaias", ["Is.", "Isai."]),
    ("Jr", "Jeremiah", "Jeremias", ["Jer.", "Hier."]),
    ("Lam", "Lamentations", "Lamentationes", ["Lam.", "Thren."]),
    ("Bar", "Baruch", "Baruch", ["Bar."]),
    ("Ez", "Ezekiel", "Ezechiel", ["Ezech.", "Ez."]),
    ("Dn", "Daniel", "Daniel", ["Dan.", "Dn."]),
    ("Os", "Hosea", "Osee", ["Os.", "Osee"]),
    ("Joel", "Joel", "Joel", ["Joel"]),
    ("Am", "Amos", "Amos", ["Am.", "Amos"]),
    ("Abd", "Obadiah", "Abdias", ["Abd."]),
    ("Jon", "Jonah", "Jonas", ["Jon."]),
    ("Mch", "Micah", "Michaeas", ["Mich."]),
    ("Nah", "Nahum", "Nahum", ["Nah."]),
    ("Hab", "Habakkuk", "Habacuc", ["Hab."]),
    ("Soph", "Zephaniah", "Sophonias", ["Soph."]),
    ("Agg", "Haggai", "Aggaeus", ["Agg."]),
    ("Zach", "Zechariah", "Zacharias", ["Zach."]),
    ("Mal", "Malachi", "Malachias", ["Mal."]),
    ("1Mcc", "1 Maccabees", "Machabaeorum I", ["1 Mach.", "I Mach."]),
    ("2Mcc", "2 Maccabees", "Machabaeorum II", ["2 Mach.", "II Mach."]),
    ("Mt", "Matthew", "Matthaeus", ["Matth.", "Mt."]),
    ("Mc", "Mark", "Marcus", ["Marc.", "Mc."]),
    ("Lc", "Luke", "Lucas", ["Luc.", "Lc."]),
    ("Jo", "John", "Joannes", ["Joan.", "Io."]),
    ("Act", "Acts", "Actus Apostolorum", ["Act."]),
    ("Rom", "Romans", "ad Romanos", ["Rom."]),
    ("1Cor", "1 Corinthians", "ad Corinthios I", ["1 Cor.", "I Cor."]),
    ("2Cor", "2 Corinthians", "ad Corinthios II", ["2 Cor.", "II Cor."]),
    ("Gal", "Galatians", "ad Galatas", ["Gal."]),
    ("Eph", "Ephesians", "ad Ephesios", ["Eph."]),
    ("Phlp", "Philippians", "ad Philippenses", ["Phil."]),
    ("Col", "Colossians", "ad Colossenses", ["Col."]),
    ("1Thes", "1 Thessalonians", "ad Thessalonicenses I", ["1 Thess."]),
    ("2Thes", "2 Thessalonians", "ad Thessalonicenses II", ["2 Thess."]),
    ("1Tim", "1 Timothy", "ad Timotheum I", ["1 Tim."]),
    ("2Tim", "2 Timothy", "ad Timotheum II", ["2 Tim."]),
    ("Tit", "Titus", "ad Titum", ["Tit."]),
    ("Phlm", "Philemon", "ad Philemonem", ["Philem."]),
    ("Hbr", "Hebrews", "ad Hebraeos", ["Hebr."]),
    ("Jac", "James", "Jacobi", ["Jac.", "Iac."]),
    ("1Ptr", "1 Peter", "Petri I", ["1 Pet.", "I Petr."]),
    ("2Ptr", "2 Peter", "Petri II", ["2 Pet.", "II Petr."]),
    ("1Jo", "1 John", "Joannis I", ["1 Joan.", "I Io."]),
    ("2Jo", "2 John", "Joannis II", ["2 Joan."]),
    ("3Jo", "3 John", "Joannis III", ["3 Joan."]),
    ("Jud", "Jude", "Judae", ["Jud.", "Iudae"]),
    ("Apc", "Apocalypse", "Apocalypsis", ["Apoc.", "Rev."]),
]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def fetch(src: str, cache: Path, offline: bool) -> str | None:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"{src}.lat"
    if not p.exists():
        if offline:
            return None
        req = urllib.request.Request(RAW + f"{src}.lat", headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                p.write_bytes(r.read())
        except Exception as e:  # noqa: BLE001
            print(f"  fetch failed {src}: {e}", file=sys.stderr)
            return None
        time.sleep(0.5)
    return p.read_bytes().decode("cp1252")


def clean_verse(t: str) -> str:
    """Strip the Clementine markup to plain text. Verse lines ('/') become newlines,
    paragraph marks and verse brackets are dropped, '<Aleph>' becomes 'Aleph. '."""
    t = t.replace("\\", " ")
    t = re.sub(r"\s*[\[\]]\s*", " ", t)
    t = re.sub(r"<([^>]+)>\s*", r"\1. ", t)
    t = re.sub(r"\s*/\s*", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" ([:;?!])", r"\1", t)      # French spacing in the source -> plain
    t = "\n".join(line.strip() for line in t.split("\n"))
    return t.strip()


def parse_book(raw: str, name: str) -> list[dict]:
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+):(\d+)\s+(.*)$", line)
        if not m:
            raise ValueError(f"{name}: unparsed line {line[:60]!r}")
        ch, vs, text = m.groups()
        ref = f"{name} {int(ch)}:{int(vs)}"
        out.append({"id": ref, "label": ref, "text": clean_verse(text)})
    return out


def ubytes(u: dict) -> int:
    return len(json.dumps(u, ensure_ascii=False, indent=0).encode("utf-8"))


def split_by_chapter(u: dict, limit: int = MAX_UNIT_BYTES) -> list[dict]:
    if ubytes(u) <= limit:
        return [u]
    chapters: list[list[dict]] = []
    for p in u["passages"]:
        ch = p["id"].rsplit(" ", 1)[1].split(":")[0]
        if not chapters or chapters[-1][0]["id"].rsplit(" ", 1)[1].split(":")[0] != ch:
            chapters.append([])
        chapters[-1].append(p)
    total = ubytes(u)
    nparts = -(-total // int(limit * 0.9))
    target = total / nparts
    groups, cur, size = [], [], 0
    for c in chapters:
        csz = sum(len(json.dumps(p, ensure_ascii=False).encode()) + 2 for p in c)
        if cur and size + csz > target and len(groups) < nparts - 1:
            groups.append(cur)
            cur, size = [], 0
        cur.extend(c)
        size += csz
    groups.append(cur)
    parts = []
    for i, g in enumerate(groups):
        parts.append({**u, "unit": f"{u['unit']}-{'abcdefgh'[i]}", "parent_unit": u["unit"],
                      "title": f"{u['title']} (part {i + 1} of {len(groups)})",
                      "passage_range": [g[0]["id"], g[-1]["id"]], "passages": g})
    return parts


def validate(u: dict) -> list[str]:
    errs = []
    ids = [p["id"] for p in u["passages"]]
    if len(ids) != len(set(ids)):
        errs.append(f"{u['unit']}: duplicate verse ids")
    if not ids:
        errs.append(f"{u['unit']}: empty")
    for p in u["passages"]:
        if not p["text"]:
            errs.append(f"{u['unit']}: empty verse {p['id']}")
        if re.search(r"[<>\[\]\\/]", p["text"]):
            errs.append(f"{u['unit']}: markup left in {p['id']}")
    if ubytes(u) > MAX_UNIT_BYTES:
        errs.append(f"{u['unit']}: {ubytes(u)} bytes")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    units, splits, aliases, errs, failures = [], {}, {}, [], []
    for src, name, latin, abbrs in BOOKS:
        raw = fetch(src, a.cache, a.offline)
        if raw is None:
            failures.append(f"{src}.lat not available")
            continue
        try:
            passages = parse_book(raw, name)
        except ValueError as e:
            failures.append(str(e))
            continue
        u = {"corpus": "vulgate", "unit": slug(name), "title": f"{name} ({latin})",
             "passages": passages}
        aliases[name] = {"unit": slug(name), "latin": latin, "abbreviations": abbrs}
        parts = split_by_chapter(u)
        if len(parts) > 1:
            splits[u["unit"]] = [{"unit": p["unit"], "passage_range": p["passage_range"]}
                                 for p in parts]
            aliases[name]["parts"] = [p["unit"] for p in parts]
        for p in parts:
            errs += validate(p)
            (OUT / f"{p['unit']}.json").write_text(json.dumps(p, ensure_ascii=False, indent=0) + "\n",
                                                   encoding="utf-8")
            units.append(p)
    entry = {
        "id": "vulgate", "title": "Vulgate (Clementine)", "language": "la",
        "edition": "Biblia Sacra Vulgatae Editionis (Clementine, 1592), text of the Clementine "
                   "Vulgate Project (after Colunga–Turrado 1946 and Vercellone 1861)",
        "license": "public domain (Clementine Vulgate Project text)",
        "attribution": "Text: The Clementine Vulgate Project (M. Tweedale et al.), via VulSearch; "
                       f"mirror github.com/jrichter/ClementineVulgateConverter @ {COMMIT[:12]}",
        "url": "https://vulsearch.sourceforge.net/",
        "unit_scheme": "book (unit id = slug of the book name, e.g. '1-kings')",
        "passage_scheme": "Book chapter:verse (Vulgate names and numbering: 1 Kings = 1 Samuel, "
                          "3 Kings = 1 Kings; Psalms in Vulgate/LXX numbering)",
        "units": [u["unit"] for u in units],
        "books": aliases,
        "text_note": "Markup removed: paragraph marks and verse brackets dropped, poetic line "
                     "breaks kept as newlines, speaker/letter tags rendered 'Aleph. '. The "
                     "prologues of Lamentations and Ecclesiasticus sit at the start of 1:1 as in "
                     "the source. Appendix books (3-4 Esdras, Prayer of Manasses) are not included.",
    }
    if splits:
        entry["split_units"] = splits
        entry["split_note"] = ("Books over ~300 KB are split at chapter boundaries into "
                               "<unit>-a, <unit>-b; split_units gives each part's first and last "
                               "verse (passage_range).")
    (OUT / "corpus.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1) + "\n",
                                     encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.glob("*.json"))
    print(f"vulgate: {len(units)} units, {sum(len(u['passages']) for u in units)} passages, "
          f"{size / 1e6:.2f} MB")
    for f in failures:
        print("  FAIL", f)
    for e in errs:
        print("  INVALID", e)
    return 1 if errs or failures else 0


if __name__ == "__main__":
    sys.exit(main())

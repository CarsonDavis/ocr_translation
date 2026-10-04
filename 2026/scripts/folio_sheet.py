# scripts/folio_sheet.py
"""Build contact sheets of page-header bands so the image->page mapping can be checked.

For every manifest page that has a raw image we crop the band from 7% to 13% of the
image height (full width) -- the running head, which carries the printed folio
number -- resize it to 1400px wide, and stack 20 of them per sheet. Each strip is
labelled at the left, in red, with "<id> / <expected folio>". The label lives in a
gutter to the left of the strip so it can never cover the printed number (which on
verso pages sits at the outer, i.e. left, edge).
"""
import argparse
import json
import pathlib

from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bookconf import ROOT  # noqa: E402  (the book root: see bookconf.py)
RAW = ROOT / "raw"
OUT = ROOT / "docs" / "checks"
# The spec called for 3%-9% of image height, but on these scans that band lands on the
# black scanner backdrop / blank top margin: the running head (which carries the printed
# folio number) starts between ~7.7% and ~10.6% depending on how the leaf sits. 7%-13%
# is the same 6%-tall band, shifted down to cover the head on every page. Override with
# --top/--bot.
BAND_TOP, BAND_BOT = 0.07, 0.13
STRIP_W = 1400
GUTTER = 210
PER_SHEET = 20
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_SIZE = 28


def load_font():
    try:
        return ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except OSError:
        return ImageFont.load_default()


def strips(records, top=BAND_TOP, bot=BAND_BOT):
    out = []
    for rec in records:
        if rec["image"] is None:
            print(f"skip {rec['id']}: no image in manifest")
            continue
        p = RAW / f"img{rec['image']:03d}.jpg"
        if not p.exists():
            print(f"skip {rec['id']}: {p.name} not on disk")
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size
        band = im.crop((0, int(h * top), w, int(h * bot)))
        sh = max(1, round(band.height * STRIP_W / band.width))
        out.append((rec, band.resize((STRIP_W, sh), Image.LANCZOS)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-ids", help="comma-separated manifest ids to include")
    ap.add_argument("--top", type=float, default=BAND_TOP)
    ap.add_argument("--bot", type=float, default=BAND_BOT)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    font = load_font()
    m = json.loads((ROOT / "manifest.json").read_text())
    recs = m["pages"]
    if a.only_ids:
        want = [s.strip() for s in a.only_ids.split(",") if s.strip()]
        recs = [r for r in recs if r["id"] in want]
    items = strips(recs, a.top, a.bot)
    if not items:
        print("nothing to do")
        return
    for k in range(0, len(items), PER_SHEET):
        chunk = items[k:k + PER_SHEET]
        sheet_h = sum(s.height for _, s in chunk)
        sheet = Image.new("RGB", (GUTTER + STRIP_W, sheet_h), "white")
        d = ImageDraw.Draw(sheet)
        y = 0
        for rec, s in chunk:
            sheet.paste(s, (GUTTER, y))
            folio = rec["folio"] if rec["folio"] is not None else "-"
            d.text((8, y + max(2, s.height // 2 - FONT_SIZE)), f"{rec['id']}", font=font, fill=(220, 0, 0))
            d.text((8, y + max(2, s.height // 2 - FONT_SIZE) + FONT_SIZE + 2), f"/ {folio}", font=font, fill=(220, 0, 0))
            d.line([(0, y), (GUTTER + STRIP_W, y)], fill=(180, 180, 180), width=1)
            y += s.height
        n = k // PER_SHEET + 1
        path = OUT / f"folio-sheet-{n:02d}.jpg"
        sheet.save(path, "JPEG", quality=88)
        print(f"wrote {path.relative_to(ROOT)} ({len(chunk)} strips, {sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()

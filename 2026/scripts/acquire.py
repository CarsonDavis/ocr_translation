# scripts/acquire.py
"""Download native-resolution page images from Cambridge IIIF as region tiles and stitch them.

The IIIF server caps whole-image requests at 2000px on the long edge, but serves
arbitrary regions at native resolution as long as the output is <= 2000x2000.
So we request the page as a 2x3 grid of non-overlapping native-resolution tiles
and paste each at its true offset: no scaling happens anywhere, so tile seams
are pixel-exact.
"""
import argparse
import io
import json
import os
import pathlib
import sys
import time

import requests
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
RAW.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) coras-transcription/1.0"}
W, H = 2941, 4711
# non-overlapping tiles: x in {0, 2000} (widths 2000, 941), y in {0, 2000, 4000} (heights 2000, 2000, 711)
TILES = [(x, y, min(2000, W - x), min(2000, H - y)) for y in (0, 2000, 4000) for x in (0, 2000)]
SLEEP = 0.2


def fetch(url, tries=4):
    r = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.content
        except requests.RequestException as e:
            print("  retry", e, flush=True)
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url} -> {r.status_code if r is not None else 'no response'}")


def stitch(iiif_base, out):
    canvas = Image.new("RGB", (W, H))
    for x, y, w, h in TILES:
        tile = Image.open(io.BytesIO(fetch(f"{iiif_base}/{x},{y},{w},{h}/full/0/default.jpg")))
        assert tile.size == (w, h), (tile.size, (w, h))
        canvas.paste(tile, (x, y))
        time.sleep(SLEEP)
    tmp = out.with_suffix(f".part{os.getpid()}")  # unique: two workers may meet on the same page
    canvas.save(tmp, "JPEG", quality=92)
    tmp.rename(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--reverse", action="store_true", help="walk the manifest backwards (lets a second worker meet the first in the middle)")
    a = ap.parse_args()
    m = json.loads((ROOT / "manifest.json").read_text())
    missing = wrong = ok = 0
    order = list(reversed(m["pages"])) if a.reverse else m["pages"]
    for rec in order:
        if rec["image"] is None:
            continue
        if a.only and rec["id"] != a.only:
            continue
        out = RAW / f"img{rec['image']:03d}.jpg"
        if not out.exists() and not a.check:
            print("fetch", rec["id"], rec["image"], flush=True)
            stitch(rec["iiif"], out)
        if not out.exists():
            missing += 1
            continue
        size = Image.open(out).size
        if abs(size[0] - W) > 2 or abs(size[1] - H) > 2:
            wrong += 1
            continue
        ok += 1
        rec["status"]["acquired"] = "done"
    if not a.check:
        __import__("pagelib").write_manifest(ROOT / "manifest.json", m)
    print(f"{ok} raw images ok, {missing} missing, {wrong} wrong size")
    sys.exit(1 if missing or wrong else 0)


if __name__ == "__main__":
    main()

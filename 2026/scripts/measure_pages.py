#!/usr/bin/env python3
"""Measure the paper bounding box of every raw CUDL scan.

The 161 CUDL scans are all 2941x4711 and were photographed on a black backdrop,
with the leaf roughly centred, a sliver of the facing page sometimes showing at
one edge, and a burned-in copyright band in the black area below the leaf.

This script reports, per image, the detected paper box, and then the min /
median / max of each edge across all images.  Task 3's decision rule reads that
spread: if max-min of *every* edge is < 60px a single fixed box (the median of
each edge) is good enough for all CUDL pages; otherwise crop.py detects per
page.

Detection (see crop.detect_paper_box) works at 1/4 scale on grayscale:
paper = luminance > 60, rows that are > 30% paper, then columns that are > 30%
paper within those rows, largest contiguous runs wider than 40% / taller than
50% of the frame, scaled back x4.

Usage:
    uv run --with pillow,numpy python scripts/measure_pages.py [--log] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from crop import EDGE_NAMES, FIXED_BOX_SPREAD_LIMIT, detect_paper_box  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
LOG = ROOT / "docs" / "pipeline-log.md"
LOG_HEADING = "## 2026-09-21 Task 3: crop"


def cudl_records(manifest: dict) -> list[dict]:
    return [r for r in manifest["pages"] if r.get("image") is not None]


def measure(records, quiet=False):
    """Return [(id, image_no, (x0, y0, x1, y1)), ...] for every scan on disk."""
    rows = []
    for rec in records:
        path = RAW / f"img{rec['image']:03d}.jpg"
        if not path.exists():
            if not quiet:
                print(f"skip {rec['id']}: {path.name} not on disk")
            continue
        with Image.open(path) as im:
            box = detect_paper_box(im)
        rows.append((rec["id"], rec["image"], box))
    return rows


def spread(rows):
    """min / median / max per edge, plus max-min, over the measured boxes."""
    out = {}
    for i, name in enumerate(EDGE_NAMES):
        vals = sorted(box[i] for _, _, box in rows)
        lo, hi = vals[0], vals[-1]
        out[name] = {
            "min": lo,
            "median": int(round(statistics.median(vals))),
            "max": hi,
            "range": hi - lo,
        }
    return out


def decision(sp):
    worst = max(sp[name]["range"] for name in EDGE_NAMES)
    mode = "fixed" if worst < FIXED_BOX_SPREAD_LIMIT else "detected"
    return mode, worst


def format_report(rows, sp, mode, worst):
    lines = []
    lines.append(f"{len(rows)} CUDL images measured")
    lines.append("")
    lines.append(f"{'id':<14}{'img':>5}{'x0':>7}{'y0':>7}{'x1':>7}{'y1':>7}{'w':>7}{'h':>7}")
    for pid, img, (x0, y0, x1, y1) in rows:
        lines.append(f"{pid:<14}{img:>5}{x0:>7}{y0:>7}{x1:>7}{y1:>7}"
                     f"{x1 - x0:>7}{y1 - y0:>7}")
    lines.append("")
    lines.append(f"{'edge':<6}{'min':>8}{'median':>8}{'max':>8}{'range':>8}")
    for name in EDGE_NAMES:
        s = sp[name]
        lines.append(f"{name:<6}{s['min']:>8}{s['median']:>8}{s['max']:>8}{s['range']:>8}")
    lines.append("")
    med = tuple(sp[n]["median"] for n in EDGE_NAMES)
    lines.append(f"worst edge range = {worst}px "
                 f"({'<' if worst < FIXED_BOX_SPREAD_LIMIT else '>='} "
                 f"{FIXED_BOX_SPREAD_LIMIT}px) -> mode = {mode}")
    lines.append(f"median box = {list(med)}  "
                 f"({med[2] - med[0]} x {med[3] - med[1]})")
    return "\n".join(lines)


def summary_block(rows, sp, mode, worst):
    """The short form that goes into docs/pipeline-log.md."""
    med = [sp[n]["median"] for n in EDGE_NAMES]
    out = [f"- `measure_pages.py` over {len(rows)} CUDL scans (all 2941x4711), "
           "paper box detected at 1/4 scale (luminance > 60, >30% rows/columns, "
           "largest run >40% wide / >50% tall):", "",
           "  | edge | min | median | max | range |",
           "  |---|---|---|---|---|"]
    for name in EDGE_NAMES:
        s = sp[name]
        out.append(f"  | {name} | {s['min']} | {s['median']} | {s['max']} | {s['range']} |")
    out.append("")
    out.append(f"- Worst edge range {worst}px, threshold {FIXED_BOX_SPREAD_LIMIT}px "
               f"-> **crop mode = `{mode}`**"
               + (f" (one fixed box {med}, {med[2] - med[0]}x{med[3] - med[1]}, "
                  "for every CUDL page)" if mode == "fixed"
                  else " (per-page detected boxes)")
               + ", plus a 1% outward margin clamped to the frame.")
    return "\n".join(out)


def write_log(text):
    body = LOG.read_text(encoding="utf-8")
    if LOG_HEADING in body:
        head, _, rest = body.partition(LOG_HEADING)
        # drop the old section (up to the next "## " heading, if any)
        tail = ""
        nxt = rest.find("\n## ")
        if nxt != -1:
            tail = rest[nxt:]
        body = head.rstrip("\n") + "\n\n" + LOG_HEADING + "\n" + text.rstrip("\n") + "\n" + tail
    else:
        body = body.rstrip("\n") + "\n\n" + LOG_HEADING + "\n" + text.rstrip("\n") + "\n"
    LOG.write_text(body, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true",
                    help="write the summary into docs/pipeline-log.md")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images")
    ap.add_argument("--json", help="also dump the per-image boxes as JSON here")
    a = ap.parse_args(argv)

    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    recs = cudl_records(manifest)
    if a.limit:
        recs = recs[:a.limit]
    t0 = time.time()
    rows = measure(recs)
    if not rows:
        print("no images measured")
        return 1
    sp = spread(rows)
    mode, worst = decision(sp)
    print(format_report(rows, sp, mode, worst))
    print(f"\nmeasured in {time.time() - t0:.1f}s")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"boxes": {pid: box for pid, _, box in rows},
             "spread": sp, "mode": mode, "worst": worst}, indent=1), encoding="utf-8")
    if a.log:
        write_log(summary_block(rows, sp, mode, worst))
        print(f"wrote summary to {LOG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

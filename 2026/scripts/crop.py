#!/usr/bin/env python3
"""Crop the raw scans to the leaf, and cut each page into reading strips.

Why strips exist
----------------
The reading model sees at most ~1.15 megapixels per image (~1568px on the long
edge).  A whole 2941x4711 page handed over at once is downscaled to about
979x1568, at which size the small italic marginal notes are illegible.  So each
page is also cut into pieces small enough that nothing is downscaled: body
strips and margin strips, every one of them <= 1.1 MP at native resolution.

Outputs
-------
    pages/full/<id>.jpg              the (deskewed) paper box, native res, q92
    pages/read/<id>.jpg              the same crop at 1600px wide, q85
    pages/strips/<id>/body-K.jpg     body column, cut into <=1.1MP strips
    pages/strips/<id>/margin-K.jpg   margin column (+60px inward), same
    pages/strips/<id>/foot.jpg       bottom 18% of the crop, full width
    docs/checks/crop-sheet-NN.jpg    contact sheets with the bands drawn on
    manifest.json                    per page: crop.{box,body,margin,tilt,mode}

Usage
-----
    uv run --with pillow,numpy python scripts/crop.py [--only p001,p004]
    uv run --with pillow,numpy python scripts/crop.py --sheets
    uv run --with pillow,numpy python scripts/crop.py --check
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
PAGES = ROOT / "pages"
FULL_DIR = PAGES / "full"
READ_DIR = PAGES / "read"
STRIP_DIR = PAGES / "strips"
CHECKS = ROOT / "docs" / "checks"
MANIFEST = ROOT / "manifest.json"
BOX_CACHE = CHECKS / "crop-boxes.json"

EDGE_NAMES = ("x0", "y0", "x1", "y1")

# --- paper detection ------------------------------------------------------
DETECT_SCALE = 4          # work at 1/4 scale
PAPER_LUM = 60            # backdrop is near black; paper is well above this
ROW_PAPER_FRAC = 0.30
COL_PAPER_FRAC = 0.30
MIN_COL_RUN = 0.40        # of frame width -- excludes the facing-page sliver
MIN_ROW_RUN = 0.50        # of frame height
FIXED_BOX_SPREAD_LIMIT = 60   # px; below this one fixed box serves every page
OUTLIER_LIMIT = 150       # px from the median before a page is worth a warning
MARGIN_FRAC = 0.01        # 1% of page width/height added outward

# --- tilt -----------------------------------------------------------------
TILT_LO, TILT_HI, TILT_STEP = -1.5, 1.5, 0.1
TILT_REGION = 0.60        # horizontal middle of the box stands in for the body
DESKEW_MIN = 0.5          # only deskew when |tilt| exceeds this

# --- columns --------------------------------------------------------------
INK_LUM = 128
SMOOTH = 25               # moving average, in native px
SMOOTH_FINE = 9           # ... and a sharper one, used only to split a weld
DENSE_FRAC = 0.15         # of the smoothed projection's max
MIN_MARGIN_FRAC = 0.06    # a margin band must be >= 6% of the page width
BACKDROP_FRAC = 0.35      # a row/column this dark this often is backdrop
# The gutter shadow on the inner edge is not black enough to be backdrop but is
# far too dark to be type: a column of this book's type is 9-20% ink, and 35%
# even on the title page, while a shadow or a paper edge runs 60-100%.
SHADOW_COL_FRAC = 0.45
SHADOW_ROW_FRAC = 0.70    # rows: a line of type can be much denser than that
WELD_FRAC = 0.66          # a run this much of the leaf is body+margin welded
WELD_TROUGH = 0.60        # ... and splits only at a real trough in it
EDGE_GUARD = 120          # px of leaf edge ignored when hunting a lone note
MERGE_GAP = 8             # dense runs closer than this are one band
BAND_PAD = 30             # px, native scale, outward on each side
MARGIN_INWARD = 220       # px of body edge included in the margin strips (keys sit left of the note text; 60 clipped them on p037)

# --- strips ---------------------------------------------------------------
MAX_PIXELS = 1_100_000
MAX_STRIP_H = 1400
OVERLAP = 80
CUT_SEARCH = 60           # +/- rows searched for a gap between text lines
FOOT_FRAC = 0.18
READ_KB_MIN, READ_KB_MAX = 200, 1200   # sanity band for pages/read

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


# =========================================================================
# small helpers
# =========================================================================

def runs_of(mask) -> list[tuple[int, int]]:
    """[(start, stop_exclusive), ...] for every contiguous True run."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def widest(runs):
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def close_gaps(runs, gap):
    """Join runs separated by less than `gap`.

    A column of type is not perfectly dense: a line of white space between two
    paragraphs, or a short line, can dip the projection under the threshold for
    a few pixels and split one text block into three, after which "the widest
    run" picks a third of the page.  The gutter between the body and the margin
    column is never this narrow (25px on these scans), so closing pinholes of a
    few pixels cannot weld the two columns together.
    """
    if not runs:
        return []
    out = [list(runs[0])]
    for lo, hi in runs[1:]:
        if lo - out[-1][1] < gap:
            out[-1][1] = hi
        else:
            out.append([lo, hi])
    return [tuple(r) for r in out]


def moving_average(a, n):
    """Box filter of width n, edge-padded so the ends are not attenuated."""
    a = np.asarray(a, dtype=np.float64)
    if n <= 1 or a.size == 0:
        return a
    pad = n // 2
    padded = np.pad(a, (pad, n - 1 - pad), mode="edge")
    kernel = np.ones(n) / n
    return np.convolve(padded, kernel, mode="valid")


def load_font(size=34):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


# =========================================================================
# paper box detection
# =========================================================================

def detect_paper_box(im, scale=DETECT_SCALE, lum=PAPER_LUM):
    """Bounding box of the leaf in `im`, in full-scale pixels.

    Works at 1/4 scale on grayscale.  Paper is luminance > `lum` (the backdrop
    is near black).  Rows that are more than 30% paper make the row mask; the
    largest run of those taller than 50% of the frame gives the vertical
    extent.  Columns that are more than 30% paper *within those rows* make the
    column mask; the largest run wider than 40% of the frame gives the
    horizontal extent -- which is what keeps the narrow sliver of the facing
    page out of the box.
    """
    g = im.convert("L")
    sw, sh = max(1, g.width // scale), max(1, g.height // scale)
    a = np.asarray(g.resize((sw, sh), Image.BILINEAR))
    paper = a > lum
    h, w = paper.shape

    row_mask = paper.mean(axis=1) > ROW_PAPER_FRAC
    row_runs = runs_of(row_mask)
    tall = [r for r in row_runs if (r[1] - r[0]) > MIN_ROW_RUN * h]
    y0, y1 = widest(tall) or widest(row_runs) or (0, h)

    col_mask = paper[y0:y1].mean(axis=0) > COL_PAPER_FRAC
    col_runs = runs_of(col_mask)
    wide = [r for r in col_runs if (r[1] - r[0]) > MIN_COL_RUN * w]
    x0, x1 = widest(wide) or widest(col_runs) or (0, w)

    return (x0 * scale, y0 * scale, x1 * scale, y1 * scale)


def expand_box(box, size, frac=MARGIN_FRAC):
    """Grow `box` outward by `frac` of its own width/height, clamped to `size`."""
    x0, y0, x1, y1 = box
    W, H = size
    dx = int(round((x1 - x0) * frac))
    dy = int(round((y1 - y0) * frac))
    return (max(0, x0 - dx), max(0, y0 - dy), min(W, x1 + dx), min(H, y1 + dy))


# =========================================================================
# tilt
# =========================================================================

def estimate_tilt(im, box, scale=DETECT_SCALE):
    """Skew of the text lines, in degrees, positive = counter-clockwise.

    The angle that maximizes the variance of the horizontal ink projection is
    the one at which the printed lines line up with image rows.  Searched from
    -1.5 to +1.5 degrees in 0.1 steps on a 1/4-scale image, over the middle
    60% of the paper box (which is body text on rectos and versos alike).
    Deskewing is `rotate(-tilt)`.
    """
    x0, y0, x1, y1 = box
    g = im.convert("L").crop((x0, y0, x1, y1))
    sw, sh = max(8, g.width // scale), max(8, g.height // scale)
    a = np.asarray(g.resize((sw, sh), Image.BILINEAR))
    # middle TILT_REGION horizontally, and drop the top/bottom 5% (head, foot)
    mx = int(sw * (1 - TILT_REGION) / 2)
    my = max(1, int(sh * 0.05))
    a = a[my:sh - my, mx:sw - mx]
    if a.size == 0:
        return 0.0
    ink = Image.fromarray(((a < INK_LUM) * 255).astype(np.uint8), "L")

    best_angle, best_var = 0.0, -1.0
    steps = int(round((TILT_HI - TILT_LO) / TILT_STEP)) + 1
    for i in range(steps):
        t = round(TILT_LO + i * TILT_STEP, 3)
        rot = ink if t == 0.0 else ink.rotate(-t, resample=Image.BILINEAR,
                                              fillcolor=0)
        proj = np.asarray(rot, dtype=np.float32).sum(axis=1)
        v = float(proj.var())
        if v > best_var:
            best_var, best_angle = v, t
    return round(best_angle, 2)


def deskew(im, box, tilt):
    """Rotate `im` by -tilt about the centre of `box`, white fill."""
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return im.rotate(-tilt, resample=Image.BICUBIC, center=(cx, cy),
                     fillcolor=(255, 255, 255))


# =========================================================================
# column detection
# =========================================================================

def ink_threshold(a):
    """Luminance below which a pixel counts as ink, scaled to the page's paper.

    The CUDL scans photograph cream paper at about 205, and 62% of that is 127
    -- the 128 the spec calls for.  The Gallica microfilm of p.41 is a much
    darker, near-bilevel image whose paper sits around 130, and at a fixed 128
    more than a third of the leaf reads as ink.  Taking the threshold from the
    page's own paper level (the 75th percentile, which is paper on every page
    here) keeps the CUDL pages at the spec's value and makes the microfilm
    behave.
    """
    paper = float(np.percentile(a, 75))
    return float(np.clip(round(0.62 * paper), 60, 150))


def page_ink(full_im):
    """(eroded ink mask, leaf row mask, leaf column mask) at native scale.

    Two things have to be taken out before a "luminance < 128" projection says
    anything about columns of type:

    * **Backdrop.**  The 1% outward margin leaves a thin band of black backdrop
      along the outer edges, and backdrop is ink by the luminance test.  A
      backdrop column is dark top to bottom -- ten times any column of type --
      so it alone would set the projection's max and drag the 15% threshold
      above the printed text.  Rows and columns that are darker than PAPER_LUM
      over more than BACKDROP_FRAC of their length are dropped.
    * **Paper.**  This is 450-year-old laid paper with foxing; at a luminance
      < 128 threshold every blank column still reads about 3% "ink", speckle a
      pixel or two across.  The gutter between the body and the margin column
      in this book is only about 25px wide, so that speckle floor is enough to
      keep the gutter above 15% of the max and weld the two columns into one.
      A 3x3 cross erosion at *native* scale deletes the speckle and leaves the
      type (strokes are 4-8px), after which a blank column really is blank.
      The erosion is why this runs at native scale rather than 1/2: at 1/2
      scale the same erosion would eat the type as well.
    """
    a = np.asarray(full_im.convert("L"))
    ink = a < ink_threshold(a)
    dark = a <= PAPER_LUM
    keep_row = ((dark.mean(axis=1) <= BACKDROP_FRAC)
                & (ink.mean(axis=1) <= SHADOW_ROW_FRAC))
    keep_col = ((dark.mean(axis=0) <= BACKDROP_FRAC)
                & (ink.mean(axis=0) <= SHADOW_COL_FRAC))
    m = ink & keep_row[:, None] & keep_col[None, :]
    e = np.zeros_like(m)
    e[1:-1, 1:-1] = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1]
                     & m[1:-1, :-2] & m[1:-1, 2:])
    return e, keep_row, keep_col


def column_projection(full_im):
    """Smoothed vertical ink projection of the full crop, native scale."""
    e, _, _ = page_ink(full_im)
    return moving_average(e.sum(axis=0).astype(np.float64), SMOOTH)


def detect_columns(full_im, side, whole_ink=False):
    """(body, margin) as [x0, x1] pairs in the full crop's native pixels.

    body   = the widest run of dense columns (projection > 15% of its max).
    margin = the widest remaining dense run on the *outer* side (right on a
             recto, left on a verso) at least 6% of the page wide.  Pages that
             carry only one or two notes have no *dense* outer run -- three
             lines of italic in a 4000px column are nowhere near 15% of the
             body's density -- so when the dense test finds nothing the outer
             leaf area outside the body is taken as the band instead (see
             _sparse_margin), so that a page carrying only one or two notes
             still gets margin strips.  None when there is no outer band.
    Both bands are padded 30px outward.
    """
    W = full_im.width
    e, keep_row, keep_col = page_ink(full_im)
    colsum = e.sum(axis=0).astype(np.float64)
    proj = moving_average(colsum, SMOOTH)
    if proj.max() <= 0:
        return [0, W], None

    if whole_ink:
        lit = np.flatnonzero(proj > 0.05 * proj.max())
        return _pad([int(lit[0]), int(lit[-1] + 1)], W), None

    rs = close_gaps(runs_of(proj > DENSE_FRAC * proj.max()), MERGE_GAP)
    if not rs:
        return [0, W], None
    body_run = widest(rs)

    min_w = MIN_MARGIN_FRAC * W
    # A run that spans most of the leaf is body and margin welded together, and
    # splitting it has to come first: the separate run further out on such a
    # page is the paper's edge, not the margin.
    split = _split_welded(moving_average(colsum, SMOOTH_FINE), body_run,
                          side, int(keep_col.sum()), min_w)
    if split:
        body_run, margin_run = split
    else:
        if side == "verso":
            cands = [r for r in rs if r[1] <= body_run[0] and (r[1] - r[0]) >= min_w]
        else:
            cands = [r for r in rs if r[0] >= body_run[1] and (r[1] - r[0]) >= min_w]
        outer = (0, body_run[0]) if side == "verso" else (body_run[1], W)
        margin_run = widest(cands) or _sparse_margin(
            e, keep_row, keep_col, outer, min_w, side)

    body = _pad([body_run[0], body_run[1]], W)
    margin = _pad([margin_run[0], margin_run[1]], W) if margin_run else None
    return body, margin


def _split_welded(proj, run, side, leaf_w, min_w):
    """Split a run that is body and margin welded together, or None.

    On the CUDL scans the gutter between the text block and the margin column
    is 25px and the two come out as separate runs.  On the Gallica microfilm of
    p.41 it is about 15px and the grain of the film bridges it, so the body
    run swallows the margin and spans nearly the whole leaf.  A run wider than
    WELD_FRAC of the leaf is taken to be welded, and is cut at the quietest
    column in its outer third -- but only if that column really is a trough
    (below WELD_TROUGH of the run's own median), and only if both halves are
    still wide enough to be bands.
    """
    lo, hi = run
    if (hi - lo) <= WELD_FRAC * max(1, leaf_w):
        return None
    span = hi - lo
    # the cut has to leave a band of at least min_w on each side, and the run's
    # own ends are always its lowest points, so keep well clear of them
    pad = int(min_w)
    if side == "verso":
        search = (lo + pad, lo + max(pad + 1, int(span * 0.35)))
    else:
        search = (hi - max(pad + 1, int(span * 0.35)), hi - pad)
    if search[1] - search[0] < 3:
        return None
    seg = proj[search[0]:search[1]]
    cut = int(np.argmin(seg)) + search[0]
    if seg.min() > WELD_TROUGH * float(np.median(proj[lo:hi])):
        return None
    body, margin = ((cut, hi), (lo, cut)) if side == "verso" else ((lo, cut), (cut, hi))
    if (body[1] - body[0]) < min_w or (margin[1] - margin[0]) < min_w:
        return None
    return body, margin


def _sparse_margin(e, keep_row, keep_col, outer, min_w, side):
    """The outer margin of the leaf, for a page with no *dense* margin column.

    Pages that carry one or two notes instead of a full column of them have no
    dense outer run: three lines of italic in a 4000px column are nowhere near
    15% of the body's density.  Every statistic tried for "is there type out
    here" -- mean eroded ink, the densest 120px cell -- overlaps between a page
    with a two-line note (p029) and a page whose outer margin holds only a
    stain or some show-through (p090, p145), so there is no honest test.

    The band is therefore taken geometrically: the whole outer margin, from
    EDGE_GUARD inside the paper's edge to the body.  A note is then never lost,
    at the cost of three largely blank strips on the dozen or so pages that
    really have nothing out there.  A missing note is a hole in the
    transcription; a blank strip is one cheap call.  `None` is still returned
    when the body reaches the paper's edge and there is no margin to speak of.
    """
    cols = np.flatnonzero(keep_col[outer[0]:outer[1]]) + outer[0]
    rows = np.flatnonzero(keep_row)
    if len(cols) < 2 or len(rows) < 2:
        return None
    band = cols[EDGE_GUARD:] if side == "verso" else cols[:-EDGE_GUARD]
    if len(band) < min_w:
        return None
    return int(band[0]), int(band[-1]) + 1


def _pad(band, W, pad=BAND_PAD):
    return [max(0, band[0] - pad), min(W, band[1] + pad)]


# =========================================================================
# strip cutting
# =========================================================================

def strip_limit(band_w, max_px=MAX_PIXELS, max_h=MAX_STRIP_H):
    """Tallest strip of this width that still fits under the pixel budget."""
    return max(1, min(max_h, int(max_px // max(1, band_w))))


def strip_rows(band_h, band_w, proj=None, overlap=OVERLAP, search=CUT_SEARCH,
               max_px=MAX_PIXELS, max_h=MAX_STRIP_H):
    """[(y0, y1), ...] covering 0..band_h with `overlap` rows shared.

    Every strip is at most `strip_limit(band_w)` tall, so band_w * height stays
    under the pixel budget.  Each cut is nudged to the quietest row (the
    smallest horizontal ink projection, i.e. the gap between two printed lines)
    within +/-`search` of its nominal position, so no line is sliced through.
    """
    limit = strip_limit(band_w, max_px, max_h)
    if band_h <= limit:
        return [(0, band_h)], limit
    if limit <= overlap + 2:                      # pathological: no room to step
        return [(0, band_h)], limit

    proj = None if proj is None else np.asarray(proj, dtype=np.float64)
    rows, y = [], 0
    while True:
        if band_h - y <= limit:
            rows.append((y, band_h))
            break
        nominal = y + limit - search
        lo = max(y + overlap + 1, nominal - search)
        hi = min(band_h - 1, y + limit)
        end = min(max(nominal, lo), hi)
        if proj is not None and hi > lo:
            end = lo + int(np.argmin(proj[lo:hi + 1]))
        end = int(min(max(end, lo), hi))
        rows.append((y, end))
        y = end - overlap

    # a sliver at the foot reads badly; fold it into the strip above if it fits
    if len(rows) >= 2:
        a, b = rows[-2], rows[-1]
        if (b[1] - b[0]) < 250 and (b[1] - a[0]) <= limit:
            rows[-2:] = [(a[0], b[1])]
    return rows, limit


def row_projection(im, box):
    """Horizontal ink projection (one value per row) of `im` inside `box`."""
    a = np.asarray(im.convert("L").crop(box), dtype=np.uint8)
    return (a < INK_LUM).sum(axis=1).astype(np.float64)


def cut_band(full_im, x0, x1, out_dir, prefix, quality=90):
    """Write <prefix>-K.jpg strips for the column band [x0, x1). Returns paths."""
    x0, x1 = int(max(0, x0)), int(min(full_im.width, x1))
    band_w = x1 - x0
    band_h = full_im.height
    if band_w <= 0:
        return []
    proj = row_projection(full_im, (x0, 0, x1, band_h))
    rows, _ = strip_rows(band_h, band_w, proj)
    out = []
    for k, (y0, y1) in enumerate(rows, 1):
        path = out_dir / f"{prefix}-{k}.jpg"
        full_im.crop((x0, y0, x1, y1)).save(path, "JPEG", quality=quality)
        out.append(path)
    return out


def cut_foot(full_im, out_dir, quality=90):
    """Bottom 18% of the crop, full width; downscaled only if over budget."""
    h = full_im.height
    top = int(h * (1 - FOOT_FRAC))
    foot = full_im.crop((0, top, full_im.width, h))
    scaled = False
    if foot.width * foot.height > MAX_PIXELS:
        f = math.sqrt(MAX_PIXELS / (foot.width * foot.height))
        foot = foot.resize((max(1, int(foot.width * f)), max(1, int(foot.height * f))),
                           Image.LANCZOS)
        scaled = True
    path = out_dir / "foot.jpg"
    foot.save(path, "JPEG", quality=quality)
    return path, scaled


# =========================================================================
# manifest plumbing
# =========================================================================

def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def save_manifest(m):
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import pagelib
    pagelib.write_manifest(MANIFEST, m)   # atomic, numpy-safe


def raw_path(rec):
    if rec.get("image") is not None:
        return RAW / f"img{rec['image']:03d}.jpg"
    return RAW / f"gallica-{rec['id']}.jpg"


# =========================================================================
# measuring pass (shared with measure_pages.py)
# =========================================================================

def measure_all(records, cache=True, remeasure=False):
    """{id: box} for every CUDL record with a scan on disk."""
    if cache and not remeasure and BOX_CACHE.exists():
        data = json.loads(BOX_CACHE.read_text(encoding="utf-8"))
        boxes = {k: tuple(v) for k, v in data["boxes"].items()}
        if all(r["id"] in boxes for r in records if raw_path(r).exists()):
            return boxes
    boxes = {}
    for rec in records:
        p = raw_path(rec)
        if not p.exists():
            continue
        with Image.open(p) as im:
            boxes[rec["id"]] = detect_paper_box(im)
    if cache:
        CHECKS.mkdir(parents=True, exist_ok=True)
        BOX_CACHE.write_text(json.dumps(
            {"boxes": {k: list(v) for k, v in boxes.items()}}, indent=1),
            encoding="utf-8")
    return boxes


def median_box(boxes):
    med = []
    for i in range(4):
        vals = sorted(b[i] for b in boxes.values())
        n = len(vals)
        med.append(int(round((vals[n // 2] if n % 2 else
                              (vals[n // 2 - 1] + vals[n // 2]) / 2))))
    return tuple(med)


def decide_mode(boxes):
    """('fixed'|'detected', worst edge range, median box)."""
    worst = 0
    for i in range(4):
        vals = [b[i] for b in boxes.values()]
        worst = max(worst, max(vals) - min(vals))
    return ("fixed" if worst < FIXED_BOX_SPREAD_LIMIT else "detected",
            worst, median_box(boxes))


# =========================================================================
# per-page work
# =========================================================================

def process_page(rec, box, mode, verbose=True):
    """Write full/read/strips for one page.

    Returns (crop_record, was_deskewed, foot_was_downscaled).
    """
    pid = rec["id"]
    src = raw_path(rec)
    im = Image.open(src).convert("RGB")

    tilt = estimate_tilt(im, box)
    if abs(tilt) > DESKEW_MIN:
        work = deskew(im, box, tilt)
        deskewed = True
    else:
        work = im
        deskewed = False

    FULL_DIR.mkdir(parents=True, exist_ok=True)
    READ_DIR.mkdir(parents=True, exist_ok=True)
    full = work.crop(box)
    full_path = FULL_DIR / f"{pid}.jpg"
    full.save(full_path, "JPEG", quality=92)

    read = Image.open(full_path).convert("RGB")
    rw = 1600
    rh = max(1, round(read.height * rw / read.width))
    read.resize((rw, rh), Image.LANCZOS).save(READ_DIR / f"{pid}.jpg",
                                              "JPEG", quality=85)

    saved = Image.open(full_path).convert("RGB")
    body, margin = detect_columns(saved, rec.get("side", "recto"),
                                  whole_ink=(pid == "p000-title"))

    out_dir = STRIP_DIR / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    nb = len(cut_band(saved, body[0], body[1], out_dir, "body"))
    nm = 0
    if margin:
        # The margin strip always runs to the paper's outer edge: the detected band can
        # stop short of long citation lines and clip their last word (found on p009).
        page_w = saved.width
        if rec.get("side") == "verso":
            mx0, mx1 = 0, margin[1] + MARGIN_INWARD
        else:
            mx0, mx1 = margin[0] - MARGIN_INWARD, page_w
        nm = len(cut_band(saved, mx0, mx1, out_dir, "margin"))
    _, foot_scaled = cut_foot(saved, out_dir)

    im.close()
    if verbose:
        print(f"{pid:<14} box={list(box)} tilt={tilt:+.1f}"
              f"{' deskewed' if deskewed else '':<9}"
              f" body={body} margin={margin} strips={nb}b/{nm}m"
              f"{' foot-scaled' if foot_scaled else ''}")
    return ({"box": list(box), "body": body, "margin": margin,
             "tilt": tilt, "mode": mode}, deskewed, foot_scaled)


# =========================================================================
# commands
# =========================================================================

def cmd_run(args):
    m = load_manifest()
    recs = m["pages"]
    if args.only:
        want = [s.strip() for s in args.only.replace(",", " ").split() if s.strip()]
        recs = [r for r in recs if r["id"] in want]
        missing = [w for w in want if w not in {r["id"] for r in recs}]
        if missing:
            print(f"unknown ids: {missing}")

    cudl = [r for r in m["pages"] if r.get("image") is not None]
    boxes = measure_all(cudl, remeasure=args.remeasure)
    mode, worst, med = decide_mode(boxes)
    print(f"measured {len(boxes)} CUDL scans; worst edge range {worst}px "
          f"-> mode={mode}; median box {list(med)}")

    if mode == "detected":
        warned = []
        for pid, b in sorted(boxes.items()):
            off = max(abs(b[i] - med[i]) for i in range(4))
            if off > OUTLIER_LIMIT:
                warned.append(pid)
                print(f"  WARNING {pid}: detected box {list(b)} is {off}px "
                      f"from the median on one edge -- review")
        if warned:
            print(f"  ({len(warned)} warnings. Rectos and versos sit at "
                  "opposite ends of the frame -- the sliver of the facing "
                  "page falls on the left of a recto and the right of a "
                  "verso -- so one median cannot suit both and about half "
                  "the book is 'far from the median' by construction.)")

    t0 = time.time()
    deskewed, footscaled = [], []
    by_id = {r["id"]: r for r in m["pages"]}
    for rec in recs:
        pid = rec["id"]
        src = raw_path(rec)
        if not src.exists():
            print(f"skip {pid}: {src.name} not on disk")
            continue
        override = rec.get("crop_override")
        if override:
            box = tuple(override)
            pmode = "override"
        else:
            with Image.open(src) as probe:
                size = probe.size
            raw_box = boxes.get(pid)
            if raw_box is None:
                with Image.open(src) as probe2:
                    raw_box = detect_paper_box(probe2)
            box = expand_box(med if mode == "fixed" else raw_box, size)
            pmode = mode
        crop, dsk, fsc = process_page(rec, box, pmode)
        by_id[pid]["crop"] = crop
        by_id[pid].setdefault("status", {})["cropped"] = "done"
        if dsk:
            deskewed.append((pid, crop["tilt"]))
        if fsc:
            footscaled.append(pid)
    save_manifest(m)
    print(f"\n{len(recs)} pages in {time.time() - t0:.1f}s")
    if deskewed:
        print("deskewed (|tilt| > 0.5): " +
              ", ".join(f"{p} {t:+.1f}" for p, t in deskewed))
    else:
        print("deskewed: none")
    print(f"foot strips downscaled to fit 1.1MP: {len(footscaled)}")
    return 0


def cmd_check(args):
    m = load_manifest()
    recs = m["pages"]
    issues = []
    read_kb = []
    ok_full = ok_read = ok_strips = True
    for rec in recs:
        pid = rec["id"]
        f = FULL_DIR / f"{pid}.jpg"
        r = READ_DIR / f"{pid}.jpg"
        d = STRIP_DIR / pid
        if not f.exists():
            issues.append(f"{pid}: missing full")
            ok_full = False
        if not r.exists():
            issues.append(f"{pid}: missing read")
            ok_read = False
        else:
            kb = r.stat().st_size / 1024
            read_kb.append(kb)
            # The task sheet expected 250-600KB at quality 85; the grain of
            # these scans actually puts every page between 730 and 1010KB, so
            # the band here is the measured one -- wide enough to still catch a
            # truncated or blank page, which is what this check is for.
            if not (READ_KB_MIN <= kb <= READ_KB_MAX):
                issues.append(f"{pid}: read is {kb:.0f}KB "
                              f"(expected {READ_KB_MIN}-{READ_KB_MAX}KB)")
        crop = rec.get("crop")
        if not crop:
            issues.append(f"{pid}: no crop record in manifest")
            continue
        bodies = sorted(d.glob("body-*.jpg"))
        margins = sorted(d.glob("margin-*.jpg"))
        foot = d / "foot.jpg"
        if not bodies:
            issues.append(f"{pid}: no body strips")
            ok_strips = False
        if crop.get("margin") and not margins:
            issues.append(f"{pid}: margin band recorded but no margin strips")
            ok_strips = False
        if not crop.get("margin") and margins:
            issues.append(f"{pid}: margin strips but no margin band")
            ok_strips = False
        if not foot.exists():
            issues.append(f"{pid}: missing foot.jpg")
            ok_strips = False
        for p in bodies + margins + ([foot] if foot.exists() else []):
            with Image.open(p) as s:
                px = s.width * s.height
            if px > MAX_PIXELS:
                issues.append(f"{pid}/{p.name}: {px} px > {MAX_PIXELS}")
                ok_strips = False
    if read_kb:
        read_kb.sort()
        print(f"pages/read: {len(read_kb)} files, "
              f"{read_kb[0]:.0f}-{read_kb[-1]:.0f}KB "
              f"(median {read_kb[len(read_kb) // 2]:.0f}KB)")
    print(f"{len(recs)} pages: full {'ok' if ok_full else 'FAIL'}, "
          f"read {'ok' if ok_read else 'FAIL'}, "
          f"strips {'ok' if ok_strips else 'FAIL'}; {len(issues)} issues")
    for i in issues:
        print(f"  {i}")
    return 0


SHEET_COLS, SHEET_ROWS, THUMB_W = 3, 4, 400
PER_SHEET = SHEET_COLS * SHEET_ROWS


def cmd_sheets(args):
    m = load_manifest()
    recs = [r for r in m["pages"] if (READ_DIR / f"{r['id']}.jpg").exists()]
    if not recs:
        print("no pages/read images; run crop.py first")
        return 1
    CHECKS.mkdir(parents=True, exist_ok=True)
    font = load_font(26)
    label_h, gap = 34, 12
    made = 0
    for k in range(0, len(recs), PER_SHEET):
        chunk = recs[k:k + PER_SHEET]
        thumbs = []
        for rec in chunk:
            im = Image.open(READ_DIR / f"{rec['id']}.jpg").convert("RGB")
            scale = THUMB_W / im.width
            th = max(1, round(im.height * scale))
            t = im.resize((THUMB_W, th), Image.LANCZOS)
            crop = rec.get("crop") or {}
            full_w = (crop.get("box") or [0, 0, im.width, 0])
            fw = max(1, full_w[2] - full_w[0])
            s = THUMB_W / fw
            d = ImageDraw.Draw(t)
            body = crop.get("body")
            if body:
                d.rectangle([body[0] * s, 1, body[1] * s - 1, th - 2],
                            outline=(0, 200, 0), width=3)
            margin = crop.get("margin")
            if margin:
                d.rectangle([margin[0] * s, 1, margin[1] * s - 1, th - 2],
                            outline=(230, 0, 0), width=3)
            thumbs.append((rec["id"], t))
        cell_h = max(t.height for _, t in thumbs) + label_h
        sheet_w = SHEET_COLS * (THUMB_W + gap) + gap
        sheet_h = SHEET_ROWS * (cell_h + gap) + gap
        sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
        d = ImageDraw.Draw(sheet)
        for i, (pid, t) in enumerate(thumbs):
            cx = gap + (i % SHEET_COLS) * (THUMB_W + gap)
            cy = gap + (i // SHEET_COLS) * (cell_h + gap)
            d.text((cx + 2, cy + 4), pid, font=font, fill=(0, 0, 0))
            sheet.paste(t, (cx, cy + label_h))
        n = k // PER_SHEET + 1
        path = CHECKS / f"crop-sheet-{n:02d}.jpg"
        sheet.save(path, "JPEG", quality=85)
        made += 1
        print(f"wrote {path.relative_to(ROOT)} ({len(chunk)} pages)")
    print(f"{made} sheets")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", help="comma/space separated manifest ids")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--remeasure", action="store_true",
                    help="ignore the cached paper boxes")
    a = ap.parse_args(argv)
    if a.check:
        return cmd_check(a)
    if a.sheets:
        return cmd_sheets(a)
    return cmd_run(a)


if __name__ == "__main__":
    raise SystemExit(main())

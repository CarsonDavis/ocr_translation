# scripts/tests/test_crop.py
"""Synthetic-image tests for scripts/crop.py.

Everything here is generated in memory: a white rectangle on a black backdrop
for the paper detector, two ink bands for the column detector, a ruled page for
the tilt estimator.  No real scan is touched, so the suite runs anywhere.

    uv run --with pillow,numpy,pytest python -m pytest scripts/tests/test_crop.py
"""
import pathlib
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import crop  # noqa: E402


# --- paper detection ------------------------------------------------------

def paper_on_black(size, box, lum=230):
    im = Image.new("RGB", size, (8, 8, 10))
    ImageDraw.Draw(im).rectangle(box, fill=(lum, lum - 4, lum - 12))
    return im


@pytest.mark.parametrize("box", [
    (200, 300, 1000, 1700),      # aligned to the 1/4-scale grid
    (201, 303, 1001, 1699),      # not aligned
    (140, 260, 1060, 1820),
])
def test_detect_paper_box_within_4px(box):
    im = paper_on_black((1200, 2000), (box[0], box[1], box[2] - 1, box[3] - 1))
    got = crop.detect_paper_box(im)
    for name, g, w in zip(crop.EDGE_NAMES, got, box):
        assert abs(g - w) <= 4, f"{name}: got {g}, want {w} ({got} vs {box})"


def test_detect_ignores_facing_page_sliver():
    """A narrow bright strip at one edge must not widen the box."""
    im = paper_on_black((1200, 2000), (300, 300, 1099, 1699))
    ImageDraw.Draw(im).rectangle((0, 250, 90, 1750), fill=(210, 206, 198))
    x0, y0, x1, y1 = crop.detect_paper_box(im)
    assert abs(x0 - 300) <= 4 and abs(x1 - 1100) <= 4


def test_expand_box_adds_one_percent_and_clamps():
    box = (100, 200, 1100, 2200)          # 1000 x 2000
    assert crop.expand_box(box, (1200, 3000)) == (90, 180, 1110, 2220)
    assert crop.expand_box((0, 0, 1000, 2000), (1000, 2000)) == (0, 0, 1000, 2000)


# --- column detection -----------------------------------------------------

# Ink is drawn at luminance ~70: dark enough to be ink (< 128) but not as dark
# as the photographic backdrop (<= 60), which is how real ink on this paper
# behaves -- only 0.09% of body pixels on a real scan fall below 60.
INK = (72, 70, 68)
INK2 = (84, 82, 80)


def two_band_page(w=2400, h=3600, wide=(200, 1800), narrow=(2000, 2300),
                  line_h=18, gap=24, note_rows=None):
    """White page with a wide ink band and a narrow one, ruled like text.

    `note_rows` limits the narrow band to that many ruled lines, which is what
    a page carrying a single marginal note looks like.
    """
    im = Image.new("RGB", (w, h), (246, 242, 232))
    d = ImageDraw.Draw(im)

    def line(x0, x1, y, fill, n):
        """Word-shaped blocks, so a row is about 45% ink as real type is.

        The words are shifted from line to line: in real type they fall in
        different places on every line, so a column through the text block is
        evenly inked, which is what the column projection relies on.
        """
        x = x0 - (n * 37) % 100
        while x < x1:
            lo, hi = max(x0, x), min(x1, x + 44)
            if hi > lo:
                d.rectangle((lo, y, hi - 1, y + line_h), fill=fill)
            x += 100
        d.rectangle((max(x0, x1 - 44), y, x1 - 1, y + line_h), fill=fill)

    y, n = 200, 0
    while y < h - 200:
        line(wide[0], wide[1], y, INK, n)
        if note_rows is None or n < note_rows:
            line(narrow[0], narrow[1], y, INK2, n)
        y += line_h + gap
        n += 1
    return im


def blank_outer_page(w=2400, h=3600):
    return two_band_page(w, h, note_rows=0)


# The 25px moving average turns a sharp band edge into a ramp, so the 15%
# threshold is crossed about (0.5 - 0.15) * 25 = 9 half-scale px early: bands
# come out ~18 native px wider than the ink on each side, before the 30px pad.
# That is the spec'd filter doing its job, so the tolerance allows for it.
SMEAR = 22


def test_columns_recto_body_wide_margin_narrow():
    im = two_band_page()
    body, margin = crop.detect_columns(im, "recto")
    assert -SMEAR <= (200 - crop.BAND_PAD) - body[0] <= SMEAR
    assert -SMEAR <= body[1] - (1800 + crop.BAND_PAD) <= SMEAR
    assert margin is not None
    assert -SMEAR <= (2000 - crop.BAND_PAD) - margin[0] <= SMEAR
    assert -SMEAR <= margin[1] - (2300 + crop.BAND_PAD) <= SMEAR


def test_columns_margin_null_when_band_is_on_the_wrong_side():
    """Same page read as a verso: the narrow band is inboard, so no margin."""
    im = two_band_page()
    body, margin = crop.detect_columns(im, "verso")
    assert -SMEAR <= body[1] - (1800 + crop.BAND_PAD) <= SMEAR
    assert margin is None


def test_columns_verso_finds_a_left_hand_margin():
    im = two_band_page(wide=(600, 2200), narrow=(100, 400))
    body, margin = crop.detect_columns(im, "verso")
    assert -SMEAR <= (600 - crop.BAND_PAD) - body[0] <= SMEAR
    assert margin is not None and margin[1] <= body[0] + crop.BAND_PAD * 2


def test_columns_narrow_band_below_six_percent_is_not_its_own_margin():
    """A 90px band (3.75% of the page) is too narrow to be the margin column.

    The outer margin is still handed over as a band -- see _sparse_margin, a
    note is never dropped -- but it is the geometric margin, not that run.
    """
    im = two_band_page(wide=(200, 1800), narrow=(2200, 2290))
    body, margin = crop.detect_columns(im, "recto")
    assert body[1] < 2000
    assert margin is not None and margin[0] < 2200 and margin[1] > 2290


def test_sparse_margin_finds_a_two_line_note():
    """Three lines of italic in a 3600px column are nowhere near 15% dense."""
    im = two_band_page(note_rows=3)
    body, margin = crop.detect_columns(im, "recto")
    assert -SMEAR <= body[1] - (1800 + crop.BAND_PAD) <= SMEAR
    assert margin is not None, "a marginal note must not be dropped"
    assert margin[0] < 2000 + crop.BAND_PAD and margin[1] > 2300 - crop.BAND_PAD


def test_empty_outer_margin_still_gets_a_band():
    """No honest test separates a two-line note from a stain, so the band is
    geometric and the blank margin is read too."""
    body, margin = crop.detect_columns(blank_outer_page(), "recto")
    assert margin is not None and margin[0] >= body[1] - 2 * crop.BAND_PAD


def test_no_margin_when_the_body_reaches_the_paper_edge():
    im = two_band_page(w=2400, wide=(100, 2330), note_rows=0)
    body, margin = crop.detect_columns(im, "recto")
    assert margin is None


def test_whole_ink_mode_spans_everything():
    im = two_band_page()
    body, margin = crop.detect_columns(im, "recto", whole_ink=True)
    assert margin is None
    assert body[0] <= 200 and body[1] >= 2300


# --- strip cutting --------------------------------------------------------

def test_strip_limit_keeps_pixels_under_budget():
    for w in (400, 785, 1000, 1700, 1900, 2600):
        h = crop.strip_limit(w)
        assert w * h <= crop.MAX_PIXELS
        assert h <= crop.MAX_STRIP_H


@pytest.mark.parametrize("band_h,band_w", [
    (4300, 1700), (4300, 460), (3600, 1900), (2350, 1100), (900, 1700),
    (600, 300), (5000, 2600),
])
def test_strip_rows_overlap_and_budget(band_h, band_w):
    rows, limit = crop.strip_rows(band_h, band_w)
    assert rows[0][0] == 0 and rows[-1][1] == band_h
    for y0, y1 in rows:
        assert 0 < y1 - y0 <= limit
        assert band_w * (y1 - y0) <= crop.MAX_PIXELS
    for (a0, a1), (b0, b1) in zip(rows, rows[1:]):
        assert a1 - b0 == crop.OVERLAP, f"overlap {a1 - b0} != {crop.OVERLAP}"
    covered = np.zeros(band_h, dtype=bool)
    for y0, y1 in rows:
        covered[y0:y1] = True
    assert covered.all()


def test_strip_rows_cut_between_lines():
    """The cut must land in a gap, not through a ruled line."""
    band_h, band_w = 3000, 1700
    proj = np.zeros(band_h)
    line_h, gap = 30, 20
    y = 0
    while y < band_h:
        proj[y:y + line_h] = 900.0
        y += line_h + gap
    rows, limit = crop.strip_rows(band_h, band_w, proj)
    assert len(rows) > 1
    for _, y1 in rows[:-1]:
        assert proj[y1] == 0.0, f"cut at row {y1} slices a line"


def test_strip_rows_single_strip_when_short():
    rows, limit = crop.strip_rows(300, 1700)
    assert rows == [(0, 300)]


# --- tilt -----------------------------------------------------------------

def ruled_page(w=1800, h=2600, line_h=6, pitch=46, inset=150):
    im = Image.new("RGB", (w, h), (250, 247, 238))
    d = ImageDraw.Draw(im)
    for y in range(inset, h - inset, pitch):
        d.rectangle((inset, y, w - inset - 1, y + line_h), fill=(15, 15, 15))
    return im


def framed(page, pad=120):
    """The page on a black backdrop, so the box is a real sub-rectangle."""
    im = Image.new("RGB", (page.width + 2 * pad, page.height + 2 * pad), (9, 9, 11))
    im.paste(page, (pad, pad))
    return im


@pytest.mark.parametrize("want", [1.0, -1.0, 0.6, 0.0])
def test_estimate_tilt(want):
    page = ruled_page()
    if want:
        page = page.rotate(want, resample=Image.BICUBIC, fillcolor=(250, 247, 238))
    im = framed(page)
    box = (120, 120, 120 + page.width, 120 + page.height)
    got = crop.estimate_tilt(im, box)
    assert abs(got - want) <= 0.2, f"tilt {got}, want {want}"


def test_deskew_reduces_tilt_to_zero():
    page = ruled_page().rotate(1.0, resample=Image.BICUBIC,
                               fillcolor=(250, 247, 238))
    im = framed(page)
    box = (120, 120, 120 + page.width, 120 + page.height)
    tilt = crop.estimate_tilt(im, box)
    fixed = crop.deskew(im, box, tilt)
    assert abs(crop.estimate_tilt(fixed, box)) <= 0.2


# --- misc -----------------------------------------------------------------

def test_runs_of():
    assert crop.runs_of([0, 1, 1, 0, 0, 1]) == [(1, 3), (5, 6)]
    assert crop.runs_of([]) == []
    assert crop.runs_of([1, 1, 1]) == [(0, 3)]


def test_decide_mode():
    tight = {"a": (10, 20, 30, 40), "b": (15, 25, 35, 45)}
    assert crop.decide_mode(tight)[0] == "fixed"
    loose = {"a": (10, 20, 30, 40), "b": (200, 25, 35, 45)}
    assert crop.decide_mode(loose)[0] == "detected"


def test_median_box():
    boxes = {"a": (10, 0, 0, 0), "b": (20, 0, 0, 0), "c": (60, 0, 0, 0)}
    assert crop.median_box(boxes)[0] == 20


# --- welded body + margin -------------------------------------------------

def welded_projection(w=1450, body=(110, 1050), margin=(1065, 1255), floor=30.0):
    """One dense run covering body and margin, with a shallow dip between."""
    p = np.full(w, 2.0)
    p[body[0]:body[1]] = 110.0
    p[margin[0]:margin[1]] = 95.0
    p[body[1]:margin[0]] = floor          # the gutter, bridged by film grain
    return p


def test_split_welded_cuts_at_the_gutter():
    p = welded_projection()
    got = crop._split_welded(p, (110, 1255), "recto", 1450, 0.06 * 1450)
    assert got is not None
    body, margin = got
    assert abs(body[1] - 1050) <= 20 and abs(margin[0] - 1050) <= 20
    assert body[0] == 110 and margin[1] == 1255


def test_split_welded_verso_cuts_on_the_left():
    p = welded_projection()
    p[110:300], p[315:1255] = 95.0, 110.0
    p[300:315] = 30.0
    got = crop._split_welded(p, (110, 1255), "verso", 1450, 0.06 * 1450)
    assert got is not None
    body, margin = got
    assert abs(margin[1] - 307) <= 20 and body[1] == 1255


def test_split_welded_leaves_a_normal_run_alone():
    """A body run that is only 60% of the leaf is not a weld."""
    p = np.full(2800, 2.0)
    p[560:2240] = 110.0
    assert crop._split_welded(p, (560, 2240), "recto", 2800, 0.06 * 2800) is None


def test_split_welded_needs_a_real_trough():
    p = welded_projection(floor=90.0)      # barely a dip: not a gutter
    assert crop._split_welded(p, (110, 1255), "recto", 1450, 0.06 * 1450) is None

#!/usr/bin/env python3
"""Build a human-arbitration queue for one page: every place where reads A and B still
disagree after the mechanical fixes, with an image crop of the disputed line.

    uv run --with pillow python scripts/arbitrate_queue.py p066
    uv run --with pillow python scripts/arbitrate_queue.py p057 \
        --a transcription/pilot/opus/A/p057.json --b transcription/pilot/opus/B/p057.json

Writes <out-dir>/queue/<id>.json and <out-dir>/crops/<id>/<item>.jpg. The arbitration
page (scripts/arbitrate_server.py) shows the queue; scripts/apply_arbitration.py turns the
decisions into a final page.

Before diffing, both reads are normalized in memory (normalize_spacing.normalize_page) and
space-only differences are levelled (auto_resolve.resolve); neither read file is touched.

Item kinds
    body        an aligned pair of body-column lines (heading text included) that differ
    note        an aligned pair of note lines (same key) that differ
    unmatched   a body line, note line or whole note that only one reader has
    structural  a furniture field (running_head/folio/signature/catchword) or the block
                structure differs
    note-structure  the same note text keyed or split differently (A: one 7-line note; B:
                a 3-line note + a 4-line unkeyed note): one decision for the whole region,
                instead of one unmatched item per line
    flagged     (off by default; --include-flagged turns it on) a line both reads agree on that either reader's
                uncertain[] flags as sic / wrong sort / could be / may be

Crop geometry (no line segmentation; proportional placement):
    body   The body column's text runs from the first ink run below the running head to
           the last ink row above the bottom edge. Each column line gets a height share
           proportional to 1/(characters per line of its block), so large-type TEXTE
           lines get more room than small-type annotation lines; a heading gets twice the
           share of the page's smallest line (it carries white space above and below).
           The window is 3.5 line heights centred on the line.
    margin The concatenated margin-note lines (file order) are laid onto the ink runs of the
           margin column (see margin_positions); window of 5 margin-line pitches, with
           260px of the body beside it so the marker line is in view.
    foot   Proportional over the ink of the bottom 18% of the page (pages/strips/<id>/foot.jpg).
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import auto_resolve  # noqa: E402
import diff_reads  # noqa: E402
import normalize_spacing  # noqa: E402
import pagelib  # noqa: E402

FLAG_RE = re.compile(r"\bsic\b|wrong sort|could be|may be", re.IGNORECASE)
FURNITURE = pagelib.FURNITURE

# --- crop geometry constants (pages/full pixels) ---------------------------
INK_LUM = 128
INK_ROW_FRAC = 0.05       # a row is text if >= 5% of the column width is ink
EDGE_IGNORE = 130         # rows this close to the top/bottom are paper edge / backdrop
WINDOW_LINES = 3.5
SIDE_PAD = 40
UPSCALE = 2.5
QUALITY = 85
HEADING_WEIGHT = 2.0      # x the smallest line share
MARGIN_WINDOW_LINES = 5.0
MARGIN_INTO_BODY = 260    # px of body column shown beside a margin note (the marker line)
FOOT_FRAC = 0.18
STRIP_OVERLAP = 80        # scripts/crop.py OVERLAP


# =========================================================================
# reads
# =========================================================================

def load_pair(a_path, b_path):
    """Both reads, NFC, normalized, with space-only differences levelled."""
    a = pagelib.nfc_all(pagelib.load_page(a_path))
    b = pagelib.nfc_all(pagelib.load_page(b_path))
    normalize_spacing.normalize_page(a)
    normalize_spacing.normalize_page(b)
    auto_resolve.resolve(a, b)
    return a, b


def column_pointers(page):
    """`where` for each entry of pagelib.column_texts(page), in the same order."""
    out = []
    for bi, block in enumerate(page.get("blocks") or []):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "heading" and isinstance(block.get("text"), str):
            out.append(f"blocks[{bi}].text")
        elif block.get("type") == "paragraph":
            for li, t in enumerate(block.get("lines") or []):
                if isinstance(t, str):
                    out.append(f"blocks[{bi}].lines[{li}]")
    return out


def note_table(page):
    """{note_map key: pagelib.Note}, with exactly pagelib.note_map's key naming."""
    out, unkeyed = {}, 0
    for note in pagelib.notes(page):
        key = note.key
        if key is None:
            key = f"_unkeyed_{unkeyed}"
            unkeyed += 1
        elif key in out:
            suffix = 2
            while f"{key}#{suffix}" in out:
                suffix += 1
            key = f"{key}#{suffix}"
        out[key] = note
    return out


def _safe(s):
    """ASCII-only id fragment (keys may be `ſ`, `_unkeyed_0`, `a#2`)."""
    return "".join(c if c.isascii() and (c.isalnum() or c == "_") else f"x{ord(c):x}"
                   for c in str(s))


# =========================================================================
# items
# =========================================================================

def _anchor(pairs, side_idx, other_idx, j, pointers):
    """For a line at index j on one side with no partner, return (after, before) pointers
    on the other side: the partner of the nearest aligned line before / after it."""
    after = before = None
    for p in pairs:
        if p[side_idx] < j:
            after = pointers[p[other_idx]]
        elif p[side_idx] > j and before is None:
            before = pointers[p[other_idx]]
    return after, before


def build_items(a, b, include_flagged=False):
    """The arbitration items for two prepared reads (no crop fields)."""
    items = []
    ca, cb = pagelib.column_texts(a), pagelib.column_texts(b)
    pa, pb = column_pointers(a), column_pointers(b)
    pairs, unmatched = diff_reads.align_body(ca, cb)

    def ctx(i, lines):
        return (lines[i - 1] if 0 <= i - 1 < len(lines) else None,
                lines[i + 1] if i + 1 < len(lines) else None)

    body = []
    for ai, bi, ta, tb in pairs:
        if ta == tb:
            continue
        before, after = ctx(ai, ca)
        body.append((ai, {"id": f"b-{ai:03d}", "kind": "body", "where_a": pa[ai],
                          "where_b": pb[bi], "index": ai, "index_b": bi, "a": ta, "b": tb,
                          "context_before": before, "context_after": after}))
    for side, idx, text in unmatched:
        if side == "A":
            before, after = ctx(idx, ca)
            ins_after, ins_before = _anchor(pairs, 0, 1, idx, pb)
            item = {"id": f"u-a-{idx:03d}", "kind": "unmatched", "side": "A",
                    "where_a": pa[idx], "where_b": None, "index": idx,
                    "a": text, "b": None, "b_after": ins_after, "b_before": ins_before,
                    "context_before": before, "context_after": after}
            order = idx
        else:
            before, after = ctx(idx, cb)
            ins_after, ins_before = _anchor(pairs, 1, 0, idx, pa)
            # place it in A's order just after its anchor, for a sensible queue order
            order = next((p[0] for p in pairs if p[1] > idx), len(ca)) - 0.5
            item = {"id": f"u-b-{idx:03d}", "kind": "unmatched", "side": "B",
                    "where_a": None, "where_b": pb[idx], "index_b": idx,
                    "a": None, "b": text, "a_after": ins_after, "a_before": ins_before,
                    "context_before": before, "context_after": after,
                    "b_is_heading": pb[idx].endswith(".text")}
        body.append((order, item))
    body.sort(key=lambda t: t[0])
    items.extend(i for _, i in body)

    # --- notes -------------------------------------------------------------
    na, nb = note_table(a), note_table(b)
    order_b = list(nb)
    aligned = {k: diff_reads.align_body(na[k].lines, nb[k].lines) for k in na if k in nb}
    groups, grouped = note_groups(na, nb, aligned)
    for key in list(na) + [k for k in nb if k not in na]:
        in_a, in_b = key in na, key in nb
        if in_a and in_b:
            A, B = na[key], nb[key]
            npairs, nun = aligned[key]
            wa = [f"{A.where}.lines[{i}]" for i in range(len(A.lines))]
            wb = [f"{B.where}.lines[{i}]" for i in range(len(B.lines))]
            for ai, bi, ta, tb in npairs:
                if ta != tb:
                    items.append({"id": f"n-{_safe(key)}-{ai}", "kind": "note", "key": key,
                                  "line": ai, "note_kind": A.kind, "where_a": wa[ai],
                                  "where_b": wb[bi], "a": ta, "b": tb,
                                  "context_before": A.lines[ai - 1] if ai else None,
                                  "context_after": A.lines[ai + 1] if ai + 1 < len(A.lines) else None})
            for side, idx, text in nun:
                if (side, key, idx) in grouped:
                    continue            # part of a note-structure item
                if side == "A":
                    aft, bef = _anchor(npairs, 0, 1, idx, wb)
                    items.append({"id": f"un-{_safe(key)}-a{idx}", "kind": "unmatched",
                                  "side": "A", "key": key, "line": idx, "note_kind": A.kind,
                                  "where_a": wa[idx], "where_b": None, "a": text, "b": None,
                                  "b_after": aft, "b_before": bef, "b_note": B.where})
                else:
                    aft, bef = _anchor(npairs, 1, 0, idx, wa)
                    items.append({"id": f"un-{_safe(key)}-b{idx}", "kind": "unmatched",
                                  "side": "B", "key": key, "line": idx, "note_kind": B.kind,
                                  "where_a": None, "where_b": wb[idx], "a": None, "b": text,
                                  "a_after": aft, "a_before": bef, "a_note": A.where})
        else:
            N = na[key] if in_a else nb[key]
            side = "A" if in_a else "B"
            if (side, key, None) in grouped:
                continue                # part of a note-structure item
            item = {"id": f"un-{_safe(key)}-{side.lower()}", "kind": "unmatched", "side": side,
                    "key": key, "line": None, "note_kind": N.kind, "whole_note": True,
                    "where_a": N.where if in_a else None, "where_b": None if in_a else N.where,
                    "a": "\n".join(N.lines) if in_a else None,
                    "b": None if in_a else "\n".join(N.lines),
                    "note_key": N.key, "beside_line": N.beside_line}
            if not in_a:
                # insert after the A note holding the key that precedes it in B's order
                prev = [k for k in order_b[:order_b.index(key)] if k in na]
                item["a_after"] = na[prev[-1]].where if prev else None
            items.append(item)
    # after the line items: apply rewrites line texts first, then the note layout
    items.extend(note_structure_item(g, na, nb, aligned) for g in groups)

    # --- structure -----------------------------------------------------------
    for field in FURNITURE:
        if a.get(field) != b.get(field):
            items.append({"id": f"s-{field}", "kind": "structural", "field": field,
                          "where_a": field, "where_b": field,
                          "a": a.get(field), "b": b.get(field),
                          "text": f"{field}: A={a.get(field)!r} B={b.get(field)!r}"})
    block_diffs = [s for s in diff_reads.structural_diffs(a, b)
                   if not s.split(":")[0] in FURNITURE]
    if pagelib.block_types(a) == pagelib.block_types(b):
        # same blocks, only a heading's text differs: the heading is a column line, so its
        # body item already decides it; a structural item would ask the same thing twice
        block_diffs = [s for s in block_diffs if not s.startswith("headings:")]
    if block_diffs:
        # block count / types / headings are one decision: whose block structure is the base
        sa, sb = _block_summary(a), _block_summary(b)
        items.append({"id": "s-blocks", "kind": "structural", "field": "blocks",
                      "where_a": "blocks", "where_b": "blocks", "a": None, "b": None,
                      "text": "; ".join(block_diffs),
                      "a_blocks": sa, "b_blocks": sb, "block_rows": block_rows(sa, sb)})

    if include_flagged:
        items.extend(_flagged(a, b, ca, pa, pairs, na, nb))
    return items


# --- note-structure groups ------------------------------------------------------

GROUP_RATIO = 0.85      # joined texts this similar are the same material, split differently
GROUP_MAX_RUN = 3       # at most this many consecutive fragments on the other side


def _fragments(side, mine, other, aligned):
    """Material of one read's notes that has no partner line in the other read: a whole
    note whose key the other read lacks, or a run of consecutive unmatched lines inside a
    note both reads have. [(side, key, line indices or None, texts)] in file order."""
    out = []
    for key, note in mine.items():
        if key not in other:
            out.append((side, key, None, list(note.lines)))
            continue
        nun = aligned[key][1]
        idx = sorted(i for s_, i, _ in nun if s_ == side)
        run = []
        for i in idx + [None]:
            if run and (i is None or i != run[-1] + 1):
                out.append((side, key, run, [note.lines[j] for j in run]))
                run = []
            if i is not None:
                run.append(i)
    return out


def _similar(t1, t2):
    import difflib
    k1, k2 = pagelib.align_key(" ".join(t1)), pagelib.align_key(" ".join(t2))
    if not k1 or not k2:
        return 0.0
    return difflib.SequenceMatcher(None, k1, k2, autojunk=False).ratio()


def note_groups(na, nb, aligned):
    """Unmatched note material that is the same text on both sides under a different key or
    split (p057: A's note a lines 3-6 = B's separate unkeyed note). Each group is one
    decision. Returns ([(a_fragments, b_fragments)], {(side, key, line or None)})."""
    fa = _fragments("A", na, nb, aligned)
    fb = _fragments("B", nb, na, aligned)
    used_a, used_b, groups = set(), set(), []

    def match(one, many, used_one, used_many):
        for i, f in enumerate(one):
            if i in used_one:
                continue
            best = None
            for s0 in range(len(many)):
                for n in range(1, GROUP_MAX_RUN + 1):
                    run = list(range(s0, s0 + n))
                    if run[-1] >= len(many) or any(j in used_many for j in run):
                        break
                    frags = [many[j] for j in run]
                    if all(g[1] == f[1] for g in frags):
                        continue        # same note on both sides: a line difference, not a split
                    r = _similar(f[3], [t for g in frags for t in g[3]])
                    if r >= GROUP_RATIO and (best is None or r > best[0]):
                        best = (r, run)
            if best:
                used_one.add(i)
                used_many.update(best[1])
                yield f, [many[j] for j in best[1]]

    for f, frags in list(match(fa, fb, used_a, used_b)):
        groups.append(([f], frags))
    for f, frags in list(match(fb, fa, used_b, used_a)):
        groups.append((frags, [f]))
    grouped = set()
    for fas, fbs in groups:
        for side, key, idx, _ in fas + fbs:
            for i in (idx or [None]):
                grouped.add((side, key, i))
    return groups, grouped


def region_text(notes):
    """Whole notes as one editable string: `[key] first line` then the other lines; an
    unkeyed note is `[–]`. apply_arbitration.parse_region reads it back."""
    out = []
    for n in notes:
        for j, t in enumerate(n.lines):
            out.append(f"[{'–' if n.key is None else n.key}] {t}" if j == 0 else t)
    return "\n".join(out)


def note_structure_item(group, na, nb, aligned):
    """One item for a group: the region is every note holding grouped material, closed over
    real keys (a region note's key-mate on the other side belongs to the region too)."""
    fas, fbs = group
    keys_a = {f[1] for f in fas}
    keys_b = {f[1] for f in fbs}
    while True:
        real = {k for k in keys_a | keys_b if not k.startswith("_unkeyed_")}
        new_a = keys_a | {k for k in real if k in na}
        new_b = keys_b | {k for k in real if k in nb}
        if (new_a, new_b) == (keys_a, keys_b):
            break
        keys_a, keys_b = new_a, new_b
    notes_a = [na[k] for k in na if k in keys_a]      # file order
    notes_b = [nb[k] for k in nb if k in keys_b]

    def layout(target, target_tbl_side, src_tbl, frag_pairs):
        """Target read's region notes, each line saying which source-read line it reuses."""
        out = []
        for key, n in target:
            lines = []
            for j, t in enumerate(n.lines):
                src, keep = None, False
                if key in src_tbl and key in aligned:
                    for ai, bi, _, _ in aligned[key][0]:
                        tj, sj = (bi, ai) if target_tbl_side == "B" else (ai, bi)
                        if tj == j:
                            src, keep = f"{src_tbl[key].where}.lines[{sj}]", True
                if src is None and (key, j) in frag_pairs:
                    src = frag_pairs[(key, j)]
                lines.append({"text": t, "from": src, "keep_text": keep})
            out.append({"where": n.where, "kind": n.kind, "key": n.key,
                        "beside_line": n.beside_line, "lines": lines})
        return out

    def frag_map(tgt_frags, src_frags, src_tbl):
        """Positional line pairs between matched fragments of equal length."""
        tl = [(f[1], i) for f in tgt_frags for i in (f[2] or range(len(f[3])))]
        sl = [(f[1], i) for f in src_frags for i in (f[2] or range(len(f[3])))]
        if len(tl) != len(sl):
            return {}
        return {t: f"{src_tbl[s[0]].where}.lines[{s[1]}]" for t, s in zip(tl, sl)}

    lay_b = layout([(k, nb[k]) for k in nb if k in keys_b], "B", na, frag_map(fbs, fas, na))
    lay_a = layout([(k, na[k]) for k in na if k in keys_a], "A", nb, frag_map(fas, fbs, nb))
    first = notes_a[0] if notes_a else notes_b[0]
    return {"id": f"ns-{_safe(first.key if first.key is not None else 'unkeyed')}",
            "kind": "note-structure",
            "where_a": notes_a[0].where if notes_a else None,
            "where_b": notes_b[0].where if notes_b else None,
            "notes_a": [n.where for n in notes_a], "notes_b": [n.where for n in notes_b],
            "a": region_text(notes_a), "b": region_text(notes_b),
            "layout_a": lay_a, "layout_b": lay_b,
            "hint": "the same note text is split or keyed differently; 1/2 takes that read's "
                    "note layout (line differences inside are decided by their own items)"}


def block_rows(sa, sb, context=1):
    """The two block lists aligned, reduced to the blocks that differ plus `context`
    unchanged blocks above and below each difference; the rest collapses to gap rows.
    [{"a": text or None, "b": text or None, "status": "same" | "diff" | "gap"}]."""
    import difflib
    rows = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sa, sb, autojunk=False).get_opcodes():
        if tag == "equal":
            rows += [{"a": sa[i], "b": sb[j1 + k], "status": "same"} for k, i in enumerate(range(i1, i2))]
        else:
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                rows.append({"a": sa[i1 + k] if i1 + k < i2 else None,
                             "b": sb[j1 + k] if j1 + k < j2 else None, "status": "diff"})
    diff_at = [n for n, r in enumerate(rows) if r["status"] == "diff"]
    keep = {m for n in diff_at for m in range(n - context, n + context + 1) if 0 <= m < len(rows)}
    out = []
    for n, r in enumerate(rows):
        if n in keep:
            out.append(r)
        elif not out or out[-1]["status"] != "gap":
            out.append({"a": None, "b": None, "status": "gap"})
    return out


def _block_summary(page):
    out = []
    for blk in pagelib.blocks(page):
        if blk.get("type") == "paragraph":
            lines = blk.get("lines") or []
            n = len(lines)
            first = lines[0] if lines and isinstance(lines[0], str) else ""
            out.append(f"paragraph ({n} line{'s' if n != 1 else ''}): {first}")
        else:
            out.append(f"{blk.get('type')}: {blk.get('text', '')}")
    return out


def _flags(page):
    """[(where, text, note)] for uncertain[] entries whose note flags a doubtful reading."""
    out = []
    for e in page.get("uncertain") or []:
        if isinstance(e, dict) and FLAG_RE.search(str(e.get("note", ""))):
            out.append((str(e.get("where", "")), e.get("text"), str(e.get("note", ""))))
    return out


def _flagged(a, b, ca, pa, pairs, na, nb):
    fa, fb = _flags(a), _flags(b)
    items = []

    def notes_for(where_a, where_b, text):
        found = []
        for side, flags, w in (("A", fa, where_a), ("B", fb, where_b)):
            for fw, ft, note in flags:
                if fw == w or (ft is not None and ft == text):
                    found.append(f"{side}: {note}")
        return found

    def hint(found):
        by = sorted({f[0] for f in found})
        return ("Both readers agree on this line; flagged by " + " and ".join(by) + ".\n"
                + "\n".join(f"{f[0]}: {f[3:]}" for f in found))

    for ai, bi, ta, tb in pairs:
        if ta != tb:
            continue
        pbw = column_pointers(b)[bi]
        found = notes_for(pa[ai], pbw, ta)
        if found:
            items.append({"id": f"f-b-{ai:03d}", "kind": "flagged", "where_a": pa[ai],
                          "where_b": pbw, "index": ai, "a": ta, "b": tb, "flags": found, "hint": hint(found),
                          "context_before": ca[ai - 1] if ai else None,
                          "context_after": ca[ai + 1] if ai + 1 < len(ca) else None})
    for key in na:
        if key not in nb:
            continue
        A, B = na[key], nb[key]
        npairs, _ = diff_reads.align_body(A.lines, B.lines)
        for ai, bi, ta, tb in npairs:
            if ta != tb:
                continue
            wa, wb = f"{A.where}.lines[{ai}]", f"{B.where}.lines[{bi}]"
            found = notes_for(wa, wb, ta)
            if found:
                items.append({"id": f"f-n-{_safe(key)}-{ai}", "kind": "flagged", "key": key,
                              "line": ai, "note_kind": A.kind, "where_a": wa, "where_b": wb,
                              "a": ta, "b": tb, "flags": found, "hint": hint(found),
                              "context_before": A.lines[ai - 1] if ai else None,
                              "context_after": A.lines[ai + 1] if ai + 1 < len(A.lines) else None})
    return items


# =========================================================================
# crop geometry
# =========================================================================

def _ink_rows(im_l, x0, x1, y0, y1):
    """Per-row ink fraction of the box (PIL only: threshold, then box-average to 1px)."""
    from PIL import Image
    box = im_l.crop((x0, y0, x1, y1)).point(lambda v: 255 if v < INK_LUM else 0)
    col = box.resize((1, y1 - y0), Image.BOX)
    return [v / 255.0 for v in col.getdata()]


def _runs(flags):
    runs, s = [], None
    for y, v in enumerate(list(flags) + [False]):
        if v and s is None:
            s = y
        elif not v and s is not None:
            runs.append((s, y))
            s = None
    return runs


def text_extent(im_l, x0, x1, skip_head=True, y_lo=None, y_hi=None):
    """(top, bottom) of the text in the column band, in image rows.

    Rows within EDGE_IGNORE of the image's top/bottom are paper edge or backdrop. With
    skip_head, the first ink run (the running head with the folio) is dropped: the body
    text starts at the next run. (The head sits only ~20px above the first body line,
    so a "first gap > 60px" rule would stop at the paper-edge noise instead.)"""
    lo = EDGE_IGNORE if y_lo is None else y_lo
    hi = im_l.height - EDGE_IGNORE if y_hi is None else y_hi
    prof = _ink_rows(im_l, x0, x1, lo, hi)
    runs = [r for r in _runs(v >= INK_ROW_FRAC for v in prof) if r[1] - r[0] >= 8]
    if not runs:
        return lo, hi
    if skip_head and len(runs) > 1:
        runs = runs[1:]
    return lo + runs[0][0], lo + runs[-1][1]


def line_weights(page):
    """Relative height share of each column line (see module docstring)."""
    weights = []
    for block in page.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "heading" and isinstance(block.get("text"), str):
            weights.append(None)
        elif block.get("type") == "paragraph":
            lines = [t for t in block.get("lines") or [] if isinstance(t, str)]
            if not lines:
                continue
            longest = max(len(t) for t in lines)
            full = [len(t) for t in lines if len(t) >= 0.75 * longest] or [longest]
            w = 1.0 / max(10, statistics.median(full))
            weights.extend([w] * len(lines))
    real = [w for w in weights if w is not None]
    small = min(real) if real else 1.0
    return [HEADING_WEIGHT * small if w is None else w for w in weights], small


def body_positions(page, top, bottom):
    """[(y_centre, line_height)] per column line, and the small-type line height."""
    weights, small = line_weights(page)
    total = sum(weights) or 1.0
    scale = (bottom - top) / total
    out, y = [], float(top)
    for w in weights:
        h = w * scale
        out.append((y + h / 2, h))
        y += h
    return out, small * scale


def margin_band(side, body, margin):
    """The margin column without the body's edge: the manifest bands overlap by up to
    ~100px, and the gutter is too ragged to find, so stay 40px clear of the padded body."""
    if side == "verso":
        return margin[0], min(margin[1], body[0] + 30) - 40
    return max(margin[0], body[1] - 30) + 40, margin[1]


def margin_positions(page, gray, band, top, bottom):
    """{(note kind, note index, line index): y_centre} for margin notes, and a line pitch.

    Notes are not reliably printed beside their markers (a crowded margin starts a note
    well above or below it), so the concatenated margin-note lines (file order) are laid
    onto the ink runs of the margin column itself: fragments (accents, stray dots) are
    dropped smallest-first until there are as many runs as lines; if there are fewer runs
    than lines, lines are spread proportionally over the runs."""
    lines = [(n.kind, n.index, j) for n in pagelib.notes(page) if n.kind == "margin_notes"
             for j in range(len(n.lines))]
    if not lines:
        return {}, 60.0
    lo, hi = max(0, top - 60), min(gray.height, bottom + 60)
    prof = _ink_rows(gray, band[0], band[1], lo, hi)
    runs = [r for r in _runs(v >= 0.03 for v in prof) if r[1] - r[0] >= 8]
    while len(runs) > len(lines):
        runs.remove(min(runs, key=lambda r: r[1] - r[0]))
    if not runs:
        pitch = (bottom - top) / len(lines)
        return {ln: top + (k + 0.5) * pitch for k, ln in enumerate(lines)}, pitch
    centres = [lo + (r[0] + r[1]) / 2 for r in runs]
    gaps = [b - a for a, b in zip(centres, centres[1:])]
    pitch = statistics.median(gaps) if gaps else 60.0
    L, R = len(lines), len(runs)
    out = {}
    for k, ln in enumerate(lines):
        idx = k if L == R else round(k * (R - 1) / max(1, L - 1))
        out[ln] = centres[idx]
    return out, pitch


def strip_ranges(strip_dir, prefix):
    """[(name, y0, y1)] for pages/strips/<id>/<prefix>-K.jpg in full-page rows.

    crop.py cuts strips top to bottom with an 80px overlap, so each strip starts 80 rows
    above the end of the previous one; the heights give the exact ranges."""
    from PIL import Image
    out, y = [], 0
    k = 1
    while (strip_dir / f"{prefix}-{k}.jpg").exists():
        with Image.open(strip_dir / f"{prefix}-{k}.jpg") as im:
            h = im.height
        out.append((f"{prefix}-{k}.jpg", y, y + h))
        y = y + h - STRIP_OVERLAP
        k += 1
    return out


def best_strip(ranges, y):
    """The strip whose range holds y furthest from its edges."""
    best, score = None, None
    for name, y0, y1 in ranges:
        if y0 <= y < y1:
            s = min(y - y0, y1 - y)
            if score is None or s > score:
                best, score = name, s
    if best is None and ranges:
        best = ranges[-1][0] if y >= ranges[-1][2] else ranges[0][0]
    return best


def _save_window(full, x0, x1, yc, h, path):
    from PIL import Image
    y0 = int(max(0, yc - h / 2))
    y1 = int(min(full.height, yc + h / 2))
    x0, x1 = int(max(0, x0)), int(min(full.width, x1))
    win = full.crop((x0, y0, x1, max(y1, y0 + 1)))
    win = win.resize((int(win.width * UPSCALE), int(win.height * UPSCALE)), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    win.save(path, "JPEG", quality=QUALITY)


class Geometry:
    """Crop placement for one page, built on the read the item points into."""

    def __init__(self, page_id, rec, pages_dir):
        from PIL import Image
        self.pid = page_id
        self.full = Image.open(pages_dir / "full" / f"{page_id}.jpg").convert("RGB")
        self.gray = self.full.convert("L")
        self.body = rec["crop"]["body"]
        self.margin = rec["crop"].get("margin")
        self.side = rec.get("side", "recto")
        self.top, self.bottom = text_extent(self.gray, *self.body)
        sdir = pages_dir / "strips" / page_id
        self.body_strips = strip_ranges(sdir, "body")
        self.margin_strips = strip_ranges(sdir, "margin")
        self.pages_dir = pages_dir
        self._cache = {}

    def for_page(self, page):
        key = id(page)
        if key not in self._cache:
            pos, small_h = body_positions(page, self.top, self.bottom)
            mpos, mh = {}, small_h
            if self.margin:
                band = margin_band(self.side, self.body, self.margin)
                mpos, mh = margin_positions(page, self.gray, band, self.top, self.bottom)
            self._cache[key] = (pos, small_h, mpos, mh)
        return self._cache[key]

    def body_window(self, page, i, path):
        pos, small_h, _, _ = self.for_page(page)
        yc, h = pos[min(i, len(pos) - 1)]
        # the window is 3.5 lines of this line's type, but never less than 3.5 small lines
        hh = WINDOW_LINES * max(h, small_h)
        _save_window(self.full, self.body[0] - SIDE_PAD, self.body[1] + SIDE_PAD, yc, hh, path)
        return "pages/strips/%s/%s" % (self.pid, best_strip(self.body_strips, yc))

    def note_window(self, page, where, path):
        m = re.match(r"(margin_notes|foot_notes)\[(\d+)\](?:\.lines\[(\d+)\])?", where)
        kind, ni, li = m.group(1), int(m.group(2)), int(m.group(3) or 0)
        if kind == "foot_notes":
            return self.foot_window(page, ni, li, path)
        _, _, mpos, mh = self.for_page(page)
        yc = mpos.get((kind, ni, li))
        if yc is None or not self.margin:
            return self.page_image(path)
        mx0, mx1 = self.margin
        if self.side == "verso":      # margin on the left, body to its right
            x0, x1 = mx0 - SIDE_PAD, mx1 + MARGIN_INTO_BODY
        else:
            x0, x1 = mx0 - MARGIN_INTO_BODY, mx1 + SIDE_PAD
        _save_window(self.full, x0, x1, yc, MARGIN_WINDOW_LINES * mh, path)
        return "pages/strips/%s/%s" % (self.pid, best_strip(self.margin_strips, yc))

    def region_window(self, page, wheres, path):
        """A window over whole notes (a note-structure item): first to last line."""
        _, _, mpos, mh = self.for_page(page)
        ys = []
        for w in wheres:
            m = re.match(r"(margin_notes|foot_notes)\[(\d+)\]", w)
            kind, ni = m.group(1), int(m.group(2))
            ys += [y for (k, i, _), y in mpos.items() if k == kind and i == ni]
        if not ys or not self.margin:
            return self.page_image(path)
        mx0, mx1 = self.margin
        if self.side == "verso":
            x0, x1 = mx0 - SIDE_PAD, mx1 + MARGIN_INTO_BODY
        else:
            x0, x1 = mx0 - MARGIN_INTO_BODY, mx1 + SIDE_PAD
        top, bottom = min(ys), max(ys)
        _save_window(self.full, x0, x1, (top + bottom) / 2, bottom - top + 3 * mh, path)
        return "pages/strips/%s/%s" % (self.pid, best_strip(self.margin_strips, (top + bottom) / 2))

    def foot_window(self, page, ni, li, path):
        h = self.full.height
        y_lo = int(h * (1 - FOOT_FRAC))
        top, bottom = text_extent(self.gray, 0, self.full.width, skip_head=False,
                                  y_lo=max(y_lo, self.bottom - 10), y_hi=h - EDGE_IGNORE)
        flat = [(n.index, j) for n in pagelib.notes(page) if n.kind == "foot_notes"
                for j in range(len(n.lines))]
        k = flat.index((ni, li)) if (ni, li) in flat else 0
        lh = (bottom - top) / max(1, len(flat))
        _save_window(self.full, 0, self.full.width, top + (k + 0.5) * lh,
                     max(WINDOW_LINES * lh, 200), path)
        return f"pages/strips/{self.pid}/foot.jpg"

    def page_image(self, path):
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(self.pages_dir / "read" / f"{self.pid}.jpg") as im:
            im.convert("RGB").save(path, "JPEG", quality=QUALITY)
        return f"pages/read/{self.pid}.jpg"


def add_crops(items, a, b, page_id, out_dir, manifest_path, pages_dir):
    rec = pagelib.manifest_record(pagelib.load_manifest(manifest_path), page_id)
    if rec is None or "crop" not in rec:
        raise SystemExit(f"{page_id}: no crop record in {manifest_path}; use --no-crops")
    geo = Geometry(page_id, rec, pages_dir)
    cdir = out_dir / "crops" / page_id
    for it in items:
        rel = f"crops/{page_id}/{it['id']}.jpg"
        path = out_dir / rel
        # crop from the read that has the line (A unless it is B-only)
        page, where = (a, it.get("where_a")) if it.get("where_a") else (b, it.get("where_b"))
        if it["kind"] == "structural":
            if not (cdir / "page.jpg").exists():
                geo.page_image(cdir / "page.jpg")
            it["crop"] = f"crops/{page_id}/page.jpg"
            it["strip"] = f"pages/read/{page_id}.jpg"
            continue
        if it["kind"] == "note-structure":
            wheres = it["notes_a"] if it.get("notes_a") else it["notes_b"]
            it["strip"] = geo.region_window(a if it.get("notes_a") else b, wheres, path)
        elif where and where.startswith("blocks"):
            idx = column_pointers(page).index(where)
            it["strip"] = geo.body_window(page, idx, path)
        elif where and where.startswith(("margin_notes", "foot_notes")):
            it["strip"] = geo.note_window(page, where, path)
        else:
            it["strip"] = geo.page_image(path)
        it["crop"] = rel


# =========================================================================
# driver
# =========================================================================

def build_queue(page_id, a_path, b_path, include_flagged=False):
    a, b = load_pair(a_path, b_path)
    items = build_items(a, b, include_flagged)
    return {"page": page_id, "a": str(a_path), "b": str(b_path),
            "include_flagged": include_flagged, "items": items}, a, b


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the arbitration queue for one page.")
    ap.add_argument("page_id")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--out-dir", default="transcription/arbitration")
    ap.add_argument("--no-flagged", dest="include_flagged", action="store_false",
                    help="(the default) leave out flagged-but-agreed lines")
    ap.add_argument("--include-flagged", dest="include_flagged", action="store_true",
                    help="add lines both reads agree on but a reader flagged (off by default)")
    ap.set_defaults(include_flagged=False)
    ap.add_argument("--no-crops", action="store_true")
    ap.add_argument("--manifest", default=str(ROOT / "manifest.json"))
    ap.add_argument("--pages", default=str(ROOT / "pages"))
    x = ap.parse_args(argv)
    pid = x.page_id
    a_path = pathlib.Path(x.a or f"transcription/reads/A/{pid}.json")
    b_path = pathlib.Path(x.b or f"transcription/reads/B/{pid}.json")
    try:
        queue, a, b = build_queue(pid, a_path, b_path, x.include_flagged)
    except pagelib.PageLoadError as exc:
        print(f"cannot read {exc.path}: {exc.reason}", file=sys.stderr)
        return 2
    out_dir = pathlib.Path(x.out_dir)
    if not x.no_crops:
        add_crops(queue["items"], a, b, pid, out_dir, x.manifest, pathlib.Path(x.pages))
    qpath = out_dir / "queue" / f"{pid}.json"
    qpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = qpath.with_name(qpath.name + ".tmp")
    tmp.write_text(json.dumps(queue, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(qpath)
    counts = {}
    for it in queue["items"]:
        counts[it["kind"]] = counts.get(it["kind"], 0) + 1
    print(f"{pid}: {len(queue['items'])} items "
          + " ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f" -> {qpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

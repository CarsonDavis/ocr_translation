#!/usr/bin/env python3
"""Apply Carson's arbitration decisions to two reads and write the final page.

    uv run --with jsonschema python scripts/apply_arbitration.py p066 --out /tmp/p066.json
    uv run --with jsonschema python scripts/apply_arbitration.py p057 \
        --a transcription/pilot/opus/A/p057.json --b transcription/pilot/opus/B/p057.json \
        --out transcription/final/p057.json

Starts from read A (normalized, with auto_resolve's space-only fixes applied, exactly as
arbitrate_queue.py prepared it), or from read B if a block-structure item was decided B,
and applies every decision at its `where` pointer:

    A / B     take that read's text (for a line only one read has: keep it / drop it, or
              insert it from the other read)
    neither   the typed text (empty text removes the line; for a note-structure item, the
              whole region typed as `[key] first line` / following lines, `[–]` = unkeyed)
    either    "it is one of these two": keep the base text (A's) and add an uncertain[]
              entry "arbitration: undecided; alternatives: <A> ||| <B>", escalate false;
              the translator chooses from context (stitch_text renders ⟨alt:…⟩)
    unknown   "neither reading is confirmed": keep the base text (A's; for a line only one
              read has and for structure, A's structure) and add an uncertain[] entry
              "arbitration: unknown; alternatives: <A> ||| <B>", escalate true
              (stitch_text renders ⟨alt?:…⟩). It is a decision: the page can be finalized.

A note-structure item (the same note text keyed or split differently) is applied after the
line items, in one step: A/B take that read's note layout for the whole region, reusing the
base read's lines rather than copying them, so no line can appear twice.

A decisions entry may carry `by` (carson | auto | translator; absent = carson) and a
`reason`; both pass into the final's decisions[] entry: `by` as is, the reason folded
into `reason`, e.g. "arbitration: either (auto-deferred)" or
"arbitration: B (translator: <reason>)".

An item with no decision fails the run with a list of the undecided items. A legacy
decisions file is read leniently: "both" means either, "skip" means undecided.

The queue file is used as a staleness check: the items are rebuilt from the reads and must
match the queue Carson decided on. Output gets reader "final", model "arbitration", and one
decisions[] entry per decided item with chose "carson-session". The output is normalized
(normalize_spacing) and validated (validate_page); exit 1 if validation fails.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import arbitrate_queue as aq  # noqa: E402
import normalize_spacing  # noqa: E402
import pagelib  # noqa: E402

CHOICES = ("A", "B", "neither", "either", "unknown")
BY = ("carson", "auto", "translator")           # who decided; absent = carson
LEGACY = {"both": "either", "skip": None}     # older decisions files
SIGNS = ("[?]", "[??]", "[...]", "[abbr:")
_ids = itertools.count()


class ArbitrationError(Exception):
    pass


# =========================================================================
# an editable view of a page: a flat list of entries per column / note
# =========================================================================

class Entry:
    """One printed line (or a whole non-text block) with a stable identity."""

    def __init__(self, text, block, kind, where=None):
        self.id = next(_ids)
        self.text = text          # None for ornament/blank blocks
        self.block = block        # the source block index, or a fresh ("new", n) tag
        self.kind = kind          # "heading" | "paragraph" | "other"
        self.where = where        # the pointer in the base read, None if inserted
        self.deleted = False


class Editable:
    def __init__(self, page):
        self.page = page
        self.col = []
        self.by_where = {}
        for bi, block in enumerate(page.get("blocks") or []):
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "heading" and isinstance(block.get("text"), str):
                self._add(self.col, Entry(block["text"], bi, "heading", f"blocks[{bi}].text"))
            elif t == "paragraph":
                for li, text in enumerate(block.get("lines") or []):
                    self._add(self.col, Entry(text, bi, "paragraph", f"blocks[{bi}].lines[{li}]"))
            else:
                self.col.append(Entry(None, bi, "other"))
        # notes: [(kind, note dict, [entries], deleted flag holder)]
        self.notes = []
        for kind in ("margin_notes", "foot_notes"):
            for ni, note in enumerate(page.get(kind) or []):
                if not isinstance(note, dict):
                    continue
                entries = []
                for li, text in enumerate(note.get("lines") or []):
                    self._add(entries, Entry(text, None, "note", f"{kind}[{ni}].lines[{li}]"))
                rec = {"kind": kind, "note": note, "lines": entries, "deleted": False,
                       "where": f"{kind}[{ni}]", "id": next(_ids)}
                self.notes.append(rec)
                self.by_where[rec["where"]] = rec
        self._new = itertools.count()

    def _add(self, lst, e):
        lst.append(e)
        self.by_where[e.where] = e

    def entry(self, where):
        e = self.by_where.get(where)
        if e is None:
            raise ArbitrationError(f"pointer {where!r} not found in the base read")
        return e

    def _list_of(self, e):
        if e in self.col:
            return self.col
        for rec in self.notes:
            if e in rec["lines"]:
                return rec["lines"]
        raise ArbitrationError("internal: entry not found")

    # --- operations ------------------------------------------------------
    def insert_line(self, text, after=None, before=None, heading=False, note=None):
        """Insert a line next to an anchor pointer of the base read.

        A body line joins the paragraph of its anchor; when the anchor is a heading (or
        there is none) it joins the paragraph on the far side of it, else it starts a new
        paragraph block. A heading always becomes its own block."""
        if note is not None:
            rec = self.by_where.get(note)
            if not isinstance(rec, dict):
                raise ArbitrationError(f"note {note!r} not found in the base read")
            lst = rec["lines"]
            e = Entry(text, None, "note")
            if after:
                lst.insert(lst.index(self.entry(after)) + 1, e)
            elif before:
                lst.insert(lst.index(self.entry(before)), e)
            else:
                lst.append(e)
            return e
        lst = self.col
        if after:
            anchor = self.entry(after)
            pos = lst.index(anchor) + 1
            neighbour = lst[pos] if pos < len(lst) else None
        elif before:
            anchor = self.entry(before)
            pos = lst.index(anchor)
            neighbour = lst[pos - 1] if pos > 0 else None
        else:
            anchor, pos = None, 0
            neighbour = lst[0] if lst else None
        if heading:
            e = Entry(text, ("new", next(self._new)), "heading")
        else:
            if anchor is not None and anchor.kind == "paragraph":
                block = anchor.block
            elif neighbour is not None and neighbour.kind == "paragraph":
                block = neighbour.block
            else:
                block = ("new", next(self._new))
            e = Entry(text, block, "paragraph")
        lst.insert(pos, e)
        return e

    def insert_note(self, kind, note, after=None):
        rec = {"kind": kind, "note": note, "id": next(_ids), "deleted": False, "where": None,
               "lines": [Entry(t, None, "note") for t in note["lines"]]}
        if after and after in self.by_where:
            idx = self.notes.index(self.by_where[after]) + 1
        else:
            idx = next((i for i, r in enumerate(self.notes) if r["kind"] == kind), len(self.notes))
        self.notes.insert(idx, rec)
        return rec

    # --- rebuild ------------------------------------------------------------
    def rebuild(self):
        """Write the entries back into the page; return {entry id: final pointer}."""
        page, src = self.page, self.page.get("blocks") or []
        blocks, where = [], {}
        runs = []   # consecutive entries of the same block
        for e in self.col:
            if e.deleted:
                continue
            if runs and runs[-1][0] == e.block and e.kind == runs[-1][1] and e.kind == "paragraph":
                runs[-1][2].append(e)
            else:
                runs.append((e.block, e.kind, [e]))
        for block_id, kind, entries in runs:
            bi = len(blocks)
            orig = src[block_id] if isinstance(block_id, int) else {}
            if kind == "other":
                blocks.append(copy.deepcopy(orig))
            elif kind == "heading":
                b = {"type": "heading", "text": entries[0].text}
                if orig.get("type") == "heading" and "spaced_caps" in orig:
                    b["spaced_caps"] = orig["spaced_caps"]
                blocks.append(b)
                where[entries[0].id] = f"blocks[{bi}].text"
            else:
                b = {k: v for k, v in orig.items() if k != "lines"} if orig.get("type") == "paragraph" \
                    else {"type": "paragraph", "continues_prev": False, "continues_next": False}
                b["type"] = "paragraph"
                b["lines"] = [x.text for x in entries]
                blocks.append(b)
                for li, x in enumerate(entries):
                    where[x.id] = f"blocks[{bi}].lines[{li}]"
        # a paragraph split by an inserted heading: only the first part keeps continues_next
        # of the original and only the last keeps continues_prev (cosmetic, both schema-valid)
        page["blocks"] = blocks
        out = {"margin_notes": [], "foot_notes": []}
        for rec in self.notes:
            lines = [x for x in rec["lines"] if not x.deleted]
            if rec["deleted"] or not lines:
                continue
            ni = len(out[rec["kind"]])
            note = dict(rec["note"])
            note["lines"] = [x.text for x in lines]
            out[rec["kind"]].append(note)
            where[rec["id"]] = f"{rec['kind']}[{ni}]"
            for li, x in enumerate(lines):
                where[x.id] = f"{rec['kind']}[{ni}].lines[{li}]"
        page["margin_notes"], page["foot_notes"] = out["margin_notes"], out["foot_notes"]
        return where


# =========================================================================
# decisions
# =========================================================================

def load_decisions(path):
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("decisions", data) if isinstance(data, dict) else {}


def _norm(text):
    return normalize_spacing.normalize_line(pagelib.nfc(text.strip())) if text else ""


def _alt(t):
    return "(no line)" if t is None else t.replace("\n", " / ")


def apply(page_id, a, b, queue, decisions):
    """Return the final page dict. `a`, `b` are prepared as by arbitrate_queue.load_pair."""
    items = aq.build_items(a, b, bool(queue.get("include_flagged")))
    # staleness: the queue Carson decided on must be what the reads give now
    q_items = {i["id"]: (i.get("a"), i.get("b")) for i in queue.get("items") or []}
    now = {i["id"]: (i.get("a"), i.get("b")) for i in items}
    if q_items != now:
        changed = sorted(set(q_items.items()) ^ set(now.items()))
        raise ArbitrationError("the reads no longer match the queue (rebuild it): "
                               + ", ".join(sorted({c[0] for c in changed}))[:400])

    def choice_of(it):
        d = decisions.get(it["id"]) or {}
        c = d.get("choice")
        c = LEGACY.get(c, c)
        if c not in CHOICES:
            c = None
        return c, d.get("text")

    def provenance(it):
        d = decisions.get(it["id"]) or {}
        by = d.get("by") if d.get("by") in BY else "carson"
        reason = d.get("reason") if isinstance(d.get("reason"), str) else ""
        return by, reason.strip()

    undecided = [it["id"] for it in items if choice_of(it)[0] is None]
    if undecided:
        raise ArbitrationError("undecided items: " + ", ".join(undecided))

    # --- the base read ---------------------------------------------------------
    struct = [it for it in items if it["kind"] == "structural" and it["field"] == "blocks"]
    picks = {choice_of(it)[0] for it in struct} & {"A", "B"}
    if picks == {"A", "B"}:
        raise ArbitrationError("block-structure items were decided both A and B: "
                               + ", ".join(it["id"] for it in struct))
    base_side = "B" if picks == {"B"} else "A"
    other_side = "A" if base_side == "B" else "B"
    base = copy.deepcopy(b if base_side == "B" else a)
    ed = Editable(base)
    bk, ok = base_side.lower(), other_side.lower()

    records = []         # (item, choice, final text, entry-or-note or None, where fallback)

    for it in items:
        choice, typed = choice_of(it)
        # either / unknown both keep the base read as it is
        eff = "keep" if choice in ("either", "unknown") else choice
        text_a, text_b = it.get("a"), it.get("b")
        typed_raw = typed if eff == "neither" else None
        typed = _norm(typed) if eff == "neither" and it["kind"] != "note-structure" else None
        kind = it["kind"]
        target, final = None, None

        if kind == "structural":
            if it["field"] == "blocks":
                final = f"base read {base_side}"
            else:
                field = it["field"]
                val = {"A": text_a, "B": text_b, "keep": base.get(field),
                       "neither": typed or None}[eff]
                base[field] = val
                final = val
            where = it["where_a"]
        elif kind in ("body", "note", "flagged"):
            target = ed.entry(it["where_" + bk])
            final = {"A": text_a, "B": text_b, "keep": target.text, "neither": typed}[eff]
            if eff == "neither" and not typed:
                target.deleted = True
            else:
                target.text = final
            where = it["where_" + bk]
        elif kind == "unmatched":
            has = it["side"]                           # the read that has it
            present_in_base = has == base_side
            want_present = {"A": has == "A", "B": has == "B", "keep": present_in_base,
                            "neither": bool(typed)}[eff]
            new_text = typed if eff == "neither" else (text_a if has == "A" else text_b)
            if it.get("whole_note"):
                if present_in_base:
                    target = ed.by_where[it["where_" + bk]]
                    if not want_present:
                        target["deleted"] = True
                    elif eff == "neither":
                        target["lines"] = [Entry(t, None, "note") for t in new_text.split("\n") if t.strip()]
                elif want_present:
                    note = {"key": it.get("note_key"), "lines": [t for t in new_text.split("\n") if t.strip()]}
                    if it.get("beside_line") is not None:
                        note["beside_line"] = it["beside_line"]
                    target = ed.insert_note(it["note_kind"], note, after=it.get(f"{bk}_after"))
                final = new_text if want_present else ""
                where = it.get("where_" + bk) or it.get("where_" + ok)
            else:
                if present_in_base:
                    target = ed.entry(it["where_" + bk])
                    if not want_present:
                        target.deleted = True
                    elif eff == "neither":
                        target.text = new_text
                elif want_present:
                    if it.get("key") is not None:          # a note line
                        target = ed.insert_line(new_text, after=it.get(f"{bk}_after"),
                                                before=it.get(f"{bk}_before"),
                                                note=it.get(f"{bk}_note"))
                    else:
                        target = ed.insert_line(new_text, after=it.get(f"{bk}_after"),
                                                before=it.get(f"{bk}_before"),
                                                heading=bool(it.get("b_is_heading")) and has == "B")
                final = new_text if want_present else ""
                where = it.get("where_" + bk) or it.get("where_" + ok)
        elif kind == "note-structure":
            final, target = apply_note_structure(ed, it, eff, typed_raw, base_side, bk, ok)
            where = it.get("where_" + bk) or it.get("where_" + ok)
        else:
            raise ArbitrationError(f"unknown item kind {kind!r}")
        if choice in ("either", "unknown") and target is None and not present_in_base_of(it, base_side):
            target = _nearest(ed, it, bk)
        records.append((it, choice, final, target, where))

    wmap = ed.rebuild()

    def final_where(target, fallback):
        if isinstance(target, Entry):
            return wmap.get(target.id, fallback)
        if isinstance(target, dict):
            return wmap.get(target["id"], fallback)
        return fallback

    decisions_out, arb_uncertain = [], []
    for it, choice, final, target, where in records:
        by, why = provenance(it)
        w = final_where(target, where)
        alt = f"alternatives: {_alt(it.get('a'))} ||| {_alt(it.get('b'))}"
        if it["kind"] == "structural" and it["field"] == "blocks":
            alt = f"alternatives: {it['text']}"
        kept = final if isinstance(final, str) else None
        if choice in ("either", "unknown"):
            label = "undecided" if choice == "either" else "unknown"
            e = {"where": w, "note": f"arbitration: {label}; {alt}",
                 "escalate": choice == "unknown"}
            if isinstance(target, Entry) and not target.deleted:
                kept = target.text           # an absent line points at its nearest line
            elif isinstance(target, dict) and target.get("lines"):
                kept = target["lines"][0].text
            if kept:
                e["text"] = kept.split("\n")[0] if "\n" in kept else kept
            arb_uncertain.append(e)
        if choice == "neither" and final and any(s in final for s in SIGNS):
            arb_uncertain.append({"where": w, "text": final,
                                  "note": "arbitration: the typed reading marks an unreadable span",
                                  "escalate": False})
        decisions_out.append({"where": w, "A": it.get("a"), "B": it.get("b"),
                              "chose": "carson-session", "by": by,
                              "text": final if isinstance(final, str) else "",
                              "reason": f"arbitration: {choice}" + reason_suffix(by, why)})

    base["uncertain"] = merge_uncertain(base, a, b, base_side) + arb_uncertain
    base["reader"] = "final"
    base["model"] = "arbitration"
    base["id"] = page_id
    base["decisions"] = decisions_out
    normalize_spacing.normalize_page(base)
    return base


def reason_suffix(by, why):
    """The provenance tail of a decisions[] reason: (auto-deferred), (translator: why),
    (carson: why), or nothing for Carson without a reason."""
    if by == "auto":
        return " (auto-deferred)" if not why or why.startswith("auto-deferred") \
            else f" (auto-deferred: {why})"
    if by == "translator":
        return f" (translator: {why})" if why else " (translator)"
    return f" (carson: {why})" if why else ""


def parse_region(text):
    """`[key] line` starts a note (`[–]` = unkeyed), other lines continue it; the inverse of
    arbitrate_queue.region_text. Returns [(key, [lines])]."""
    notes = []
    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^\[([^\]]*)\]\s*(.*)$", line)
        if m:
            key = m.group(1).strip()
            notes.append((None if key in ("", "–", "-") else key, []))
            line = m.group(2).strip()
            if not line:
                continue
        elif not notes:
            raise ArbitrationError("typed note text must start with [key] (e.g. `[a] l. prima.`)")
        notes[-1][1].append(_norm(line))
    return [(k, ls) for k, ls in notes if ls]


def apply_note_structure(ed, it, eff, typed, base_side, bk, ok):
    """Rewrite a note region in one step: the base read's region notes are replaced by the
    other read's layout (reusing, never copying, the base lines it maps to) or by the typed
    notes. Returns (final text, first record of the region)."""
    region = [ed.by_where[w] for w in it.get("notes_" + bk) or [] if w in ed.by_where]
    if eff in ("keep", base_side) or (eff not in ("neither",) and not it.get("layout_" + ok)):
        return it.get(bk) or "", (region[0] if region else None)
    kind0 = region[0]["kind"] if region else "margin_notes"
    new = []
    if eff == "neither":
        old = {r["note"].get("key"): r["note"] for r in region}
        parsed = parse_region(typed)
        if not parsed:
            raise ArbitrationError(f"{it['id']}: the typed notes are empty")
        for key, lines in parsed:
            note = {"key": key, "lines": lines}
            if key in old and "beside_line" in old[key]:
                note["beside_line"] = old[key]["beside_line"]
            new.append({"kind": kind0, "note": note, "id": next(_ids), "deleted": False,
                        "where": None, "lines": [Entry(t, None, "note") for t in lines]})
        final = typed
    else:
        used = set()
        for spec in it["layout_" + ok]:
            entries = []
            for ln in spec["lines"]:
                src = ed.by_where.get(ln["from"]) if ln.get("from") else None
                if isinstance(src, Entry) and src.id not in used:
                    used.add(src.id)
                    if not ln.get("keep_text"):
                        src.text = _norm(ln["text"])
                    entries.append(src)
                else:
                    entries.append(Entry(_norm(ln["text"]), None, "note"))
            note = {"key": spec["key"], "lines": [e.text for e in entries]}
            if "beside_line" in spec:
                note["beside_line"] = spec["beside_line"]
            new.append({"kind": spec["kind"], "note": note, "id": next(_ids), "deleted": False,
                        "where": None, "lines": entries})
        final = it.get(ok) or ""
    pos = ed.notes.index(region[0]) if region else len(ed.notes)
    for r in region:
        r["deleted"] = True
    ed.notes[pos:pos] = new
    return final, new[0]


def present_in_base_of(it, base_side):
    """True unless the item is a line or note that only the other read has."""
    return it["kind"] != "unmatched" or it["side"] == base_side


def _nearest(ed, it, bk):
    """For a line / note absent from the result: the base-read line or note it would sit
    next to, so an uncertain[] entry can point at the nearest line."""
    for key in (f"{bk}_after", f"{bk}_before", f"{bk}_note"):
        w = it.get(key)
        if w and w in ed.by_where:
            return ed.by_where[w]
    return None


def merge_uncertain(result, a, b, base_side):
    """Both reads' uncertain[] entries that still apply to the result.

    An entry quoting a line's text is kept only if a line with that text is in the
    result, and is re-pointed at it; an entry without text, or whose text quotes no line
    of either read (a marker `{c}`, a bare note key, a whole note joined with " / "), is
    kept from the base read, or from the other read when it points at page furniture
    rather than a line (a word quote pointing at a line is kept only while that line
    still holds it). So a reader's "note c has no marker" entry survives arbitration."""
    lines = {}
    for ln in pagelib.all_lines(result):
        lines.setdefault(ln.text, []).append(ln.where)
    read_lines = {ln.text for page in (a, b) for ln in pagelib.all_lines(page)}
    where_text = {ln.where: ln.text for ln in pagelib.all_lines(result)}
    out, seen = [], set()
    for side, page in (("A", a), ("B", b)):
        for e in page.get("uncertain") or []:
            if not isinstance(e, dict):
                continue
            e = dict(e)
            text = e.get("text")
            if isinstance(text, str) and text and (text in lines or text in read_lines):
                if text not in lines:
                    continue
                if e.get("where") not in lines[text] and re.match(r"(blocks|margin_notes|foot_notes)\[\d+\]\.", str(e.get("where", ""))):
                    e["where"] = lines[text][0]
            elif side != base_side and re.match(r"(blocks|margin_notes|foot_notes)\[", str(e.get("where", ""))):
                continue
            elif isinstance(text, str) and text and e.get("where") in where_text \
                    and text not in where_text[e["where"]]:
                continue                     # a word quote whose line no longer holds it
            key = (e.get("where"), e.get("text"), e.get("note"))
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
    return out


# =========================================================================
# driver
# =========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description="Apply arbitration decisions to one page.")
    ap.add_argument("page_id")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--decisions")
    ap.add_argument("--queue")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "manifest.json"))
    x = ap.parse_args(argv)
    pid = x.page_id
    queue_path = pathlib.Path(x.queue or f"transcription/arbitration/queue/{pid}.json")
    dec_path = pathlib.Path(x.decisions or f"transcription/arbitration/decisions/{pid}.json")
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read the queue {queue_path}: {exc}", file=sys.stderr)
        return 2
    a_path = pathlib.Path(x.a or queue.get("a") or f"transcription/reads/A/{pid}.json")
    b_path = pathlib.Path(x.b or queue.get("b") or f"transcription/reads/B/{pid}.json")
    try:
        a, b = aq.load_pair(a_path, b_path)
    except pagelib.PageLoadError as exc:
        print(f"cannot read {exc.path}: {exc.reason}", file=sys.stderr)
        return 2
    try:
        page = apply(pid, a, b, queue, load_decisions(dec_path))
    except ArbitrationError as exc:
        print(f"{pid}: {exc}", file=sys.stderr)
        return 1

    out = pathlib.Path(x.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(page, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out)

    import validate_page
    problems, warnings = validate_page.validate_file(out, pagelib.load_manifest(x.manifest))
    for w in warnings:
        print(f"{out}: WARNING: {w}")
    for p in problems:
        print(f"{out}: PROBLEM: {p}")
    n = len(page.get("decisions") or [])
    print(f"{pid}: {n} decisions applied -> {out}" + (" (INVALID)" if problems else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

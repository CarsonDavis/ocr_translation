#!/usr/bin/env python3
"""Diff two independent reads of the same page, line by line.

    uv run python scripts/diff_reads.py p018
    uv run python scripts/diff_reads.py p018 --a transcription/final/p018.json \
        --b transcription/spotcheck/p018.json --out transcription/spotcheck/p018.diff.md

Writes a markdown report for the reconciler and prints a one-line summary.
Stdlib only.
"""
from __future__ import annotations

import argparse
import difflib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pagelib  # noqa: E402

INVISIBLE = "differs only in invisible characters or spacing"


def align_body(a_lines, b_lines):
    """Align two line lists on their loose forms.

    Returns (pairs, unmatched) where pairs are (ai, bi, a_text, b_text) and
    unmatched are ("A"|"B", index, text).
    """
    sm = difflib.SequenceMatcher(None, [pagelib.align_key(t) for t in a_lines],
                                 [pagelib.align_key(t) for t in b_lines], autojunk=False)
    pairs, unmatched = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                pairs.append((i1 + k, j1 + k, a_lines[i1 + k], b_lines[j1 + k]))
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                pairs.append((i1 + k, j1 + k, a_lines[i1 + k], b_lines[j1 + k]))
            for i in range(i1 + n, i2):
                unmatched.append(("A", i, a_lines[i]))
            for j in range(j1 + n, j2):
                unmatched.append(("B", j, b_lines[j]))
        elif tag == "delete":
            for i in range(i1, i2):
                unmatched.append(("A", i, a_lines[i]))
        elif tag == "insert":
            for j in range(j1, j2):
                unmatched.append(("B", j, b_lines[j]))
    return pairs, unmatched


def structural_diffs(a, b):
    out = []
    for field in pagelib.FURNITURE:
        if a.get(field) != b.get(field):
            out.append(f"{field}: A={a.get(field)!r} B={b.get(field)!r}")
    types_a, types_b = pagelib.block_types(a), pagelib.block_types(b)
    if len(types_a) != len(types_b):
        out.append(f"block count: A={len(types_a)} B={len(types_b)}")
    elif types_a != types_b:
        out.append(f"block types: A={types_a} B={types_b}")
    ha, hb = pagelib.headings(a), pagelib.headings(b)
    if ha != hb:
        out.append(f"headings: A={ha} B={hb}")
    return out


def note_diffs(a, b):
    """Differences between the two note tables, plus note-line agreement."""
    ma, mb = pagelib.note_map(a), pagelib.note_map(b)
    out = []
    for k in ma:
        if k not in mb:
            out.append(f'note key "{k}" only in A ({len(ma[k])} lines)')
    for k in mb:
        if k not in ma:
            out.append(f'note key "{k}" only in B ({len(mb[k])} lines)')
    same = 0
    total_a = sum(len(v) for v in ma.values())
    total_b = sum(len(v) for v in mb.values())
    for k, la in ma.items():
        if k not in mb:
            continue
        lb = mb[k]
        for i in range(max(len(la), len(lb))):
            ta = la[i] if i < len(la) else None
            tb = lb[i] if i < len(lb) else None
            if ta == tb:
                same += 1
            else:
                sa = "(no line)" if ta is None else repr(ta)
                sb = "(no line)" if tb is None else repr(tb)
                note = f"  [{INVISIBLE}]" if ta is not None and tb is not None and \
                    pagelib.align_key(ta) == pagelib.align_key(tb) else ""
                out.append(f'note "{k}" line {i}: A={sa} B={sb}{note}')
    return out, same, total_a, total_b


def pct(num, den, empty=100.0):
    """Percentage, with an explicit value for the empty case."""
    return round(100.0 * num / den, 1) if den else empty


def build_report(page_id, a_path, b_path, a, b):
    a_lines, b_lines = pagelib.column_texts(a), pagelib.column_texts(b)
    pairs, unmatched = align_body(a_lines, b_lines)
    body_diffs = [(ai, bi, ta, tb) for ai, bi, ta, tb in pairs if ta != tb]
    identical = len(pairs) - len(body_diffs)
    # nothing to agree about is not agreement: a page with no body lines in
    # either read is a structural problem, not a perfect match.
    agreement = pct(identical, max(len(a_lines), len(b_lines)), empty=0.0)

    struct = structural_diffs(a, b)
    if not a_lines and not b_lines:
        struct.insert(0, "no body lines: A=0 B=0")
    ndiffs, note_same, note_total_a, note_total_b = note_diffs(a, b)
    note_agreement = pct(note_same, max(note_total_a, note_total_b))

    report = []
    report.append(f"# Diff {page_id}")
    report.append("")
    report.append(f"- A: `{a_path}`")
    report.append(f"- B: `{b_path}`")
    report.append("")
    report.append(f"agreement={agreement:.1f}%")
    report.append("")

    report.append("## Body line differences")
    report.append("")
    if body_diffs:
        for n, (ai, bi, ta, tb) in enumerate(body_diffs, 1):
            invisible = pagelib.align_key(ta) == pagelib.align_key(tb)
            head = f"{n}. A[{ai}] / B[{bi}]"
            report.append(head + (f" ({INVISIBLE})" if invisible else ""))
            report.append("")
            # a difference you cannot see on the page is shown quoted
            report.append(f"   A[{ai}]: {ta!r}" if invisible else f"   A[{ai}]: {ta}")
            report.append(f"   B[{bi}]: {tb!r}" if invisible else f"   B[{bi}]: {tb}")
            report.append("")
    else:
        report.append("none")
        report.append("")

    report.append("## Unmatched body lines")
    report.append("")
    if unmatched:
        for side, idx, text in unmatched:
            report.append(f"- {side}[{idx}] only in {side}: {text}")
    else:
        report.append("none")
    report.append("")

    report.append("## Structural differences")
    report.append("")
    if struct:
        for s in struct:
            report.append(f"- {s}")
    else:
        report.append("none")
    report.append("")

    report.append("## Note differences")
    report.append("")
    if ndiffs:
        for s in ndiffs:
            report.append(f"- {s}")
    else:
        report.append("none")
    report.append("")

    report.append("## Summary")
    report.append("")
    report.append(f"- body lines: A={len(a_lines)} B={len(b_lines)}, "
                  f"aligned={len(pairs)}, identical={identical}")
    report.append(f"- body agreement: {agreement:.1f}%")
    report.append(f"- body line differences: {len(body_diffs)}")
    report.append(f"- unmatched body lines: {len(unmatched)}")
    report.append(f"- structural differences: {len(struct)}")
    report.append(f"- note lines: A={note_total_a} B={note_total_b}, "
                  f"identical={note_same}, agreement {note_agreement:.1f}%")
    report.append(f"- note differences: {len(ndiffs)}")
    report.append("")

    stats = dict(agreement=agreement, body_diffs=len(body_diffs),
                 unmatched=len(unmatched), structural=len(struct),
                 note_diffs=len(ndiffs), note_agreement=note_agreement)
    return "\n".join(report), stats


def record_in_manifest(manifest, manifest_path, page_id, agreement):
    """Set status.diffed and the agreement on the page's manifest record."""
    rec = pagelib.manifest_record(manifest, page_id)
    if rec is None:
        print(f"warning: {page_id} is not in {manifest_path}; not recording the diff",
              file=sys.stderr)
        return
    status = rec.get("status")
    if not isinstance(status, dict):
        status = {}
        rec["status"] = status
    status["diffed"] = "done"
    rec["agreement"] = agreement
    pagelib.write_manifest(manifest_path, manifest)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Diff two reads of one page.")
    ap.add_argument("page_id")
    ap.add_argument("--a", help="default: transcription/reads/A/<id>.json")
    ap.add_argument("--b", help="default: transcription/reads/B/<id>.json")
    ap.add_argument("--out", help="default: transcription/diff/<id>.md")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--no-manifest", action="store_true",
                    help="do not record the result in the manifest")
    args = ap.parse_args(argv)

    pid = args.page_id
    # a spot-check compares files of its own choosing; only a plain A-vs-B run
    # of the pipeline is allowed to write the manifest back
    custom = bool(args.a or args.b)
    a_path = pathlib.Path(args.a or f"transcription/reads/A/{pid}.json")
    b_path = pathlib.Path(args.b or f"transcription/reads/B/{pid}.json")
    out_path = pathlib.Path(args.out or f"transcription/diff/{pid}.md")

    try:
        a = pagelib.nfc_all(pagelib.load_page(a_path))
        b = pagelib.nfc_all(pagelib.load_page(b_path))
    except pagelib.PageLoadError as exc:
        print(f"cannot read {exc.path}: {exc.reason}", file=sys.stderr)
        return 2

    record = not args.no_manifest and not custom
    for side, page, path in (("A", a, a_path), ("B", b, b_path)):
        if page.get("id") != pid:
            print(f"warning: {path} has id {page.get('id')!r}, not {pid!r}; "
                  f"not recording the diff in the manifest", file=sys.stderr)
            record = False

    # load the manifest before writing anything, so a broken manifest does not
    # leave a report behind with no record of it
    manifest = None
    if record:
        try:
            manifest = pagelib.load_manifest(args.manifest)
        except pagelib.PageLoadError as exc:
            print(f"cannot read {exc.path}: {exc.reason}", file=sys.stderr)
            return 2
        if manifest is None:
            print(f"warning: no manifest at {args.manifest}; not recording the diff",
                  file=sys.stderr)
            record = False

    report, stats = build_report(pid, a_path, b_path, a, b)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")

    if record:
        record_in_manifest(manifest, args.manifest, pid, stats["agreement"])

    print("agreement={agreement:.1f}% body_diffs={body_diffs} unmatched={unmatched} "
          "structural={structural} note_diffs={note_diffs}".format(**stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())

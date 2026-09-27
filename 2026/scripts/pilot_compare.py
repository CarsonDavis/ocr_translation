#!/usr/bin/env python3
"""Score pilot reads against the verified finals.

    uv run python scripts/pilot_compare.py [--root transcription/pilot] [--models opus sonnet] [--pages p010 ...]

For every model and page with reads A and B under <root>/<model>/{A,B}/<page>.json:
  * A–B agreement after normalize_spacing + auto_resolve (on temp copies), as the live
    pipeline would see it (agreement %, body diffs, unmatched, structural, note diffs);
  * each read against transcription/final/<page>.json: body line diffs + unmatched,
    note diffs, structural diffs (errors a reconciler would have to catch);
  * shared errors: final lines where A and B agree with each other but not with the
    final (what a reconciler cannot catch).
Writes docs/checks/pilot-single-pass.md and prints the table.
"""
from __future__ import annotations

import argparse, json, pathlib, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pagelib, diff_reads  # noqa: E402
from normalize_spacing import normalize_page  # noqa: E402
from auto_resolve import resolve  # noqa: E402


def load(p):
    page = json.loads(p.read_text()); normalize_page(page); return page


def vs_final(read: dict, final: dict):
    _, stats = diff_reads.build_report("x", "a", "b", read, final)
    return stats


def shared_errors(a: dict, b: dict, final: dict):
    fa = dict((fi, ta) for ai, fi, ta, tf in diff_reads.align_body(pagelib.column_texts(a), pagelib.column_texts(final))[0])
    fb = dict((fi, tb) for bi, fi, tb, tf in diff_reads.align_body(pagelib.column_texts(b), pagelib.column_texts(final))[0])
    fl = pagelib.column_texts(final)
    body = [(i, fl[i], fa[i]) for i in range(len(fl)) if i in fa and i in fb and fa[i] == fb[i] and fa[i] != fl[i]]
    na, nb, nf = pagelib.note_map(a), pagelib.note_map(b), pagelib.note_map(final)
    notes = []
    for k, lines in nf.items():
        if k in na and k in nb and na[k] == nb[k] and na[k] != lines:
            notes.append((k, lines, na[k]))
    return body, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="transcription/pilot")
    ap.add_argument("--models", nargs="+", default=["opus", "sonnet"])
    ap.add_argument("--pages", nargs="*")
    ap.add_argument("--out", default="docs/checks/pilot-single-pass.md")
    args = ap.parse_args()
    root = ROOT / args.root
    rows, details = [], []
    for model in args.models:
        pages = args.pages or sorted(p.stem for p in (root / model / "A").glob("p*.json"))
        for pid in pages:
            pa, pb = root / model / "A" / f"{pid}.json", root / model / "B" / f"{pid}.json"
            pf = ROOT / "transcription/final" / f"{pid}.json"
            if not (pa.exists() and pb.exists() and pf.exists()):
                rows.append((model, pid, "missing read or final", *[""] * 9)); continue
            a, b, final = load(pa), load(pb), load(pf)
            a2, b2 = json.loads(json.dumps(a)), json.loads(json.dumps(b))
            resolve(a2, b2)
            _, ab = diff_reads.build_report(pid, "A", "B", a2, b2)
            sa, sb = vs_final(a, final), vs_final(b, final)
            sh_body, sh_notes = shared_errors(a, b, final)
            rows.append((model, pid, f"{ab['agreement']:.1f}%", ab["body_diffs"] + ab["unmatched"], ab["note_diffs"], ab["structural"],
                         sa["body_diffs"] + sa["unmatched"], sa["note_diffs"], sb["body_diffs"] + sb["unmatched"], sb["note_diffs"],
                         len(sh_body) + len(sh_notes)))
            for i, tf, ta in sh_body:
                details.append(f"- {model} {pid} body[{i}] shared: final `{tf}` / both reads `{ta}`")
            for k, lf, la in sh_notes:
                details.append(f"- {model} {pid} note {k} shared: final `{' | '.join(lf)}` / both reads `{' | '.join(la)}`")
            for tag, s, read in (("A", sa, a), ("B", sb, b)):
                rep, _ = diff_reads.build_report(pid, tag, "final", read, final)
                (ROOT / "docs/checks/pilot-diffs").mkdir(parents=True, exist_ok=True)
                (ROOT / "docs/checks/pilot-diffs" / f"{model}-{pid}-{tag}-vs-final.md").write_text(rep)
    hdr = "| model | page | A–B agree | A–B body diffs | A–B note diffs | A–B struct | A vs final body | A notes | B vs final body | B notes | shared errors |"
    sep = "|" + "---|" * 11
    lines = [hdr, sep] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    out = "# Single-pass pilot: reads vs verified finals\n\nBody columns count differing + unmatched body lines against the final; note columns count note-line differences; shared errors are lines where A and B agree with each other but not the final (uncatchable by a reconciler).\n\n" + "\n".join(lines) + "\n\n## Shared errors\n\n" + ("\n".join(details) if details else "none") + "\n"
    (ROOT / args.out).write_text(out)
    print("\n".join(lines)); print(); print("\n".join(details) if details else "shared errors: none")


if __name__ == "__main__":
    main()

"""Resolve read-vs-read differences that a convention settles mechanically, before a
reconciler sees the page.

Rule applied (conventions §1, word division): when two aligned lines are identical once
all spaces are removed, the reading with MORE spaces wins (the compositor's tight setting
is normalized to one space per word gap). Both read files are rewritten with the winning
text so the diff afterwards shows only genuine reading disagreements.

usage: auto_resolve.py PAGE_ID [--a PATH] [--b PATH]     prints the number of lines resolved
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bookconf import ROOT  # noqa: E402  (the book root: see bookconf.py)
sys.path.insert(0, str(ROOT / "scripts"))
import pagelib, diff_reads  # noqa: E402


def _nospace(t): return t.replace(" ", "")


def _resolve_lists(a_lines, b_lines):
    """Return (new_a, new_b, count) for two lists of strings."""
    pairs, _ = diff_reads.align_body(a_lines, b_lines)
    new_a, new_b, n = list(a_lines), list(b_lines), 0
    for ai, bi, ta, tb in pairs:
        if ta != tb and _nospace(ta) == _nospace(tb):
            win = ta if ta.count(" ") >= tb.count(" ") else tb
            new_a[ai] = win; new_b[bi] = win; n += 1
    return new_a, new_b, n


def _set_column(page, texts):
    """Write a flat list of column texts back into heading/paragraph blocks in order."""
    it = iter(texts)
    for block in page.get("blocks") or []:
        if not isinstance(block, dict): continue
        if block.get("type") == "heading" and isinstance(block.get("text"), str):
            block["text"] = next(it)
        elif block.get("type") == "paragraph":
            block["lines"] = [next(it) for _ in block.get("lines") or []]


def resolve(a, b):
    total = 0
    ca, cb, n = _resolve_lists(pagelib.column_texts(a), pagelib.column_texts(b)); total += n
    _set_column(a, ca); _set_column(b, cb)
    for key in ("margin_notes", "foot_notes"):
        na = {n_.get("key"): n_ for n_ in a.get(key) or [] if isinstance(n_, dict)}
        nb = {n_.get("key"): n_ for n_ in b.get(key) or [] if isinstance(n_, dict)}
        for k in na.keys() & nb.keys():
            if k is None: continue
            la, lb, n = _resolve_lists(na[k].get("lines") or [], nb[k].get("lines") or []); total += n
            na[k]["lines"], nb[k]["lines"] = la, lb
    return total


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("page_id"); ap.add_argument("--a"); ap.add_argument("--b")
    x = ap.parse_args()
    pa = pathlib.Path(x.a or ROOT / "transcription/reads/A" / f"{x.page_id}.json")
    pb = pathlib.Path(x.b or ROOT / "transcription/reads/B" / f"{x.page_id}.json")
    a, b = pagelib.load_page(pa), pagelib.load_page(pb)
    n = resolve(a, b)
    for p, d in ((pa, a), (pb, b)):
        p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    print(f"{x.page_id}: {n} space-only differences resolved")


if __name__ == "__main__":
    main()

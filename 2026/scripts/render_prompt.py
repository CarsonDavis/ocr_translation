"""Fill an agent prompt template for one page.

usage: render_prompt.py read|read_single|reconcile|spotcheck PAGE_ID [--reader A|B] [--model NAME] [--context N] [--out-dir DIR]
Prints the rendered prompt to stdout. Context = the N preceding pages (manifest order),
each as transcription/final/<id>.json when it exists. For the reader prompts (read,
read_single) a preceding page with no final yet falls back to transcription/reads/A/<id>.json,
then reads/B, so readers beyond the finals frontier keep continuity; such files are labelled
"(unreconciled read)" in the list. Pages with neither are left out; "none" if nothing is left.
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bookconf import ROOT  # noqa: E402  (the book root: see bookconf.py)

READER_KINDS = ("read", "read_single")


def context_files(prev_ids, reads_ok=True, root=None):
    """The context list for the pages `prev_ids` (manifest order): the final if present,
    else (reads_ok) read A, else read B, labelled as an unreconciled read."""
    root = pathlib.Path(root or ROOT)
    out = []
    for pid in prev_ids:
        if (root / "transcription/final" / f"{pid}.json").exists():
            out.append(f"transcription/final/{pid}.json")
            continue
        if not reads_ok:
            continue
        for rd in ("A", "B"):
            if (root / "transcription/reads" / rd / f"{pid}.json").exists():
                out.append(f"transcription/reads/{rd}/{pid}.json (unreconciled read)")
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["read", "read_single", "reconcile", "spotcheck"])
    ap.add_argument("page_id")
    ap.add_argument("--reader", default="A")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--context", type=int, default=3)
    ap.add_argument("--out-dir", default="transcription/reads/{READER}")
    a = ap.parse_args()
    m = json.loads((ROOT / "manifest.json").read_text())
    ids = [r["id"] for r in m["pages"]]
    rec = next(r for r in m["pages"] if r["id"] == a.page_id)
    i = ids.index(a.page_id)
    ctx = context_files(ids[max(0, i - a.context):i], reads_ok=a.kind in READER_KINDS)
    slim = {k: rec[k] for k in ("id", "page", "image", "side", "folio", "source")}
    import bookconf
    tpl = bookconf.fill_root(bookconf.prompt_path(a.kind, ROOT).read_text(), ROOT)
    out = (tpl.replace("{PAGE_ID}", a.page_id).replace("{READER}", a.reader)
              .replace("{MODEL}", a.model)
              .replace("{OUT_DIR}", a.out_dir.replace("{READER}", a.reader))
              .replace("{CONTEXT_PAGES}", ", ".join(ctx) if ctx else "none (no earlier pages finished yet)")
              .replace("{MANIFEST_RECORD}", json.dumps(slim, ensure_ascii=False)))
    sys.stdout.write(out)

if __name__ == "__main__":
    main()

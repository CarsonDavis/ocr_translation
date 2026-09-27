"""Fill an agent prompt template for one page.

usage: render_prompt.py read|read_single|reconcile|spotcheck PAGE_ID [--reader A|B] [--model NAME] [--context N] [--out-dir DIR]
Prints the rendered prompt to stdout. Context = the N preceding pages (manifest order) that
already have transcription/final/<id>.json; listed as paths, or "none" for the first pages.
"""
import argparse, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]

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
    ctx = [f"transcription/final/{pid}.json" for pid in ids[max(0, i - a.context):i]
           if (ROOT / "transcription/final" / f"{pid}.json").exists()]
    slim = {k: rec[k] for k in ("id", "page", "image", "side", "folio", "source")}
    tpl = (ROOT / "scripts/prompts" / f"{a.kind}.md").read_text()
    out = (tpl.replace("{PAGE_ID}", a.page_id).replace("{READER}", a.reader)
              .replace("{MODEL}", a.model)
              .replace("{OUT_DIR}", a.out_dir.replace("{READER}", a.reader))
              .replace("{CONTEXT_PAGES}", ", ".join(ctx) if ctx else "none (no earlier pages finished yet)")
              .replace("{MANIFEST_RECORD}", json.dumps(slim, ensure_ascii=False)))
    sys.stdout.write(out)

if __name__ == "__main__":
    main()

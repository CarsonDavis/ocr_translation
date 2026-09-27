"""Render the translator prompt (scripts/prompts/translate.md) to stdout.

  render_translate.py SECTION_ID [--context N]
      One section; context = the N preceding sections that already have an English file.
  render_translate.py --batch START_ID [--size 12]
      One prompt covering START_ID and the following sections, SIZE in all (fewer if the
      stitched text ends or reaches an incomplete section first). Context = the English
      files of the SIZE sections before START_ID (the previous batch). Report path:
      translation/reports/batch-<first>--<last>.md.

Redirect stdout into the scratchpad (the prompt is not written anywhere by this script).
"""
import argparse, json, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCK = r"<!-- {tag} -->(.*?)<!-- /{tag} -->"


def select_mode(tpl, mode):
    """Keep the `mode` blocks' contents, drop the other mode's blocks and the comment."""
    other = "batch" if mode == "single" else "single"
    tpl = re.sub(r"<!-- render note:.*?-->\n", "", tpl, flags=re.S)
    tpl = re.sub(BLOCK.format(tag=other), "", tpl, flags=re.S)
    tpl = re.sub(BLOCK.format(tag=mode), lambda m: m.group(1), tpl, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", tpl)


def context(ids, start, n, root=ROOT):
    return [f"{sid} (translation/sections/{sid}.md)" for sid in ids[max(0, start - n):start]
            if (root / "translation/sections" / f"{sid}.md").exists()]


def batch_ids(secs, start_id, size):
    ids = [s["id"] for s in secs]
    i = ids.index(start_id)
    out = []
    for s in secs[i:i + size]:
        if not s.get("complete", True):
            break
        out.append(s["id"])
    return out


def render_single(tpl, secs, section_id, n, root=ROOT):
    ids = [s["id"] for s in secs]
    ctx = context(ids, ids.index(section_id), n, root)
    return (select_mode(tpl, "single").replace("{SECTION_ID}", section_id)
            .replace("{CONTEXT_SECTIONS}", ", ".join(ctx) if ctx else "none (this is the first section)"))


def render_batch(tpl, secs, start_id, size, root=ROOT):
    ids = [s["id"] for s in secs]
    batch = batch_ids(secs, start_id, size)
    if not batch:
        raise SystemExit(f"{start_id} is not complete in text/sections.json; nothing to translate")
    ctx = context(ids, ids.index(start_id), size, root)
    report = f"translation/reports/batch-{batch[0]}--{batch[-1]}.md"
    return (select_mode(tpl, "batch")
            .replace("{SECTION_IDS}", ", ".join(f"`{b}`" for b in batch))
            .replace("{BATCH_SIZE}", f"{len(batch)} sections")
            .replace("{REPORT_PATH}", report)
            .replace("{CONTEXT_SECTIONS}", ", ".join(ctx) if ctx else "none (this is the first batch)"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("section_id", nargs="?")
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--batch", metavar="START_ID")
    ap.add_argument("--size", type=int, default=12)
    a = ap.parse_args(argv)
    if bool(a.section_id) == bool(a.batch):
        ap.error("give either SECTION_ID or --batch START_ID")
    secs = json.loads((ROOT / "text/sections.json").read_text())["sections"]
    tpl = (ROOT / "scripts/prompts/translate.md").read_text()
    if a.batch:
        sys.stdout.write(render_batch(tpl, secs, a.batch, a.size))
    else:
        sys.stdout.write(render_single(tpl, secs, a.section_id, a.context))


if __name__ == "__main__":
    main()

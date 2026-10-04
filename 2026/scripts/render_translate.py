"""Render the translator prompt (scripts/prompts/translate.md) to stdout.

  render_translate.py SECTION_ID [--context N]
      One section; context = the N preceding sections that already have an English file.
  render_translate.py --batch START_ID [--size 12]
      One prompt covering START_ID and the following sections, SIZE in all (fewer if the
      stitched text ends or reaches an incomplete section first). Context = the English
      files of the SIZE sections before START_ID (the previous batch). Report path:
      translation/reports/batch-<first>--<last>.md.

Both forms embed the alt-marker table for the sections in the prompt (from text/alts.json,
written by stitch_text.py) and name the alt-choices file the translator must write:
translation/alt-choices/<SECTION_ID>.json or translation/alt-choices/batch-<first>--<last>.json
(fed back by scripts/apply_translator_choices.py).

Redirect stdout into the scratchpad (the prompt is not written anywhere by this script).
"""
import argparse, json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bookconf import ROOT  # noqa: E402  (the book root: see bookconf.py)
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


def load_alts(root=ROOT):
    """The records of text/alts.json, or [] (with a warning) when it is missing."""
    f = root / "text/alts.json"
    if not f.exists():
        print(f"warning: {f} not found (re-run stitch_text.py); no alt table", file=sys.stderr)
        return []
    return json.loads(f.read_text(encoding="utf-8")).get("alts") or []


def _cell(t):
    return "`" + str(t).replace("|", "\\|").replace("`", "'") + "`"


def alt_table(alts, section_ids, secs=None):
    """The prompt's alt-marker table for `section_ids` (a sentence when there are none)."""
    rows = [r for r in alts if r.get("section") in set(section_ids)]
    if not rows:
        return ("Alt markers in these sections: **none**. Still write the alt-choices file, "
                "containing an empty list `[]`.")
    if secs is not None:                 # alts.json older than sections.json?
        text = {s["id"]: s.get("text", "") + "".join(n.get("text", "") for n in s.get("notes") or [])
                for s in secs}
        stale = [r["alt_id"] for r in rows if r["marker"] not in text.get(r["section"], "")]
        if stale:
            print("warning: alt markers not found in text/sections.json (stale alts.json?): "
                  + ", ".join(stale), file=sys.stderr)
    out = [f"Alt markers in these sections ({len(rows)}); `a` / `b` are the whole printed line "
           "as reading A / B:", "",
           "| alt_id | section | marker | a | b | context |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| " + " | ".join([r["alt_id"], r["section"], _cell(r["marker"]), _cell(r["a"]),
                                       _cell(r["b"]), _cell(r["context"])]) + " |")
    return "\n".join(out)


def fill_alts(text, alts, section_ids, choices_path, secs=None):
    return (text.replace("{ALT_TABLE}", alt_table(alts, section_ids, secs))
            .replace("{ALT_CHOICES_PATH}", choices_path))


def render_single(tpl, secs, section_id, n, root=ROOT, alts=None):
    ids = [s["id"] for s in secs]
    ctx = context(ids, ids.index(section_id), n, root)
    alts = load_alts(root) if alts is None else alts
    tpl = fill_alts(select_mode(tpl, "single"), alts, [section_id],
                    f"translation/alt-choices/{section_id}.json", secs)
    return (tpl.replace("{SECTION_ID}", section_id).replace("{BOOK_ROOT}", str(root))
            .replace("{CONTEXT_SECTIONS}", ", ".join(ctx) if ctx else "none (this is the first section)"))


def render_batch(tpl, secs, start_id, size, root=ROOT, alts=None):
    ids = [s["id"] for s in secs]
    batch = batch_ids(secs, start_id, size)
    if not batch:
        raise SystemExit(f"{start_id} is not complete in text/sections.json; nothing to translate")
    ctx = context(ids, ids.index(start_id), size, root)
    report = f"translation/reports/batch-{batch[0]}--{batch[-1]}.md"
    alts = load_alts(root) if alts is None else alts
    tpl = fill_alts(select_mode(tpl, "batch"), alts, batch,
                    f"translation/alt-choices/batch-{batch[0]}--{batch[-1]}.json", secs)
    return (tpl
            .replace("{BOOK_ROOT}", str(root))
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
    import bookconf
    tpl = bookconf.prompt_path("translate", ROOT).read_text()
    if a.batch:
        sys.stdout.write(render_batch(tpl, secs, a.batch, a.size))
    else:
        sys.stdout.write(render_single(tpl, secs, a.section_id, a.context))


if __name__ == "__main__":
    main()

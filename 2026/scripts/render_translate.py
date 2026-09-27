"""Render the translator prompt for one section: render_translate.py SECTION_ID [--context N]"""
import argparse, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("section_id"); ap.add_argument("--context", type=int, default=2)
a = ap.parse_args()
secs = json.loads((ROOT / "text/sections.json").read_text())["sections"]
ids = [s["id"] for s in secs]; i = ids.index(a.section_id)
ctx = [f"{sid} (translation/sections/{sid}.md)" for sid in ids[max(0, i - a.context):i]
       if (ROOT / "translation/sections" / f"{sid}.md").exists()]
tpl = (ROOT / "scripts/prompts/translate.md").read_text()
sys.stdout.write(tpl.replace("{SECTION_ID}", a.section_id)
                 .replace("{CONTEXT_SECTIONS}", ", ".join(ctx) if ctx else "none (this is the first section)"))

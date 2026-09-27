"""Verify that a section translation keeps its page and letter markers.

usage: check_markers.py SECTION_ID | --all
Rules: the sequence of ⟦pNNN⟧ page markers in the English body must equal the French
section's; the sequence of {x} letter markers must be equal too; every note key in the
French section has a Notes entry `- {x}`; the front matter's id matches.
"""
import json, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = re.compile(r"⟦(p[0-9]{3}(?:-[a-z]+)?)⟧")
MARK = re.compile(r"\{([a-zſ]+\d*)\}")


def check(section_id, sections):
    sec = sections.get(section_id)
    path = ROOT / "translation/sections" / f"{section_id}.md"
    problems = []
    if sec is None:
        return [f"{section_id}: no such section in text/sections.json"]
    if not path.exists():
        return [f"{section_id}: {path} missing"]
    text = path.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        problems.append("no front matter")
        body = text
    else:
        fm, body = m.group(1), text[m.end():]
        if not re.search(rf"^id:\s*{re.escape(section_id)}\s*$", fm, re.M):
            problems.append("front matter id does not match")
    head, _, notes = body.partition("\n## Notes")
    fp, ep = PAGE.findall(sec["text"]), PAGE.findall(head)
    if fp != ep:
        problems.append(f"page markers differ: French {fp} vs English {ep}")
    fm_, em_ = MARK.findall(sec["text"]), MARK.findall(head)
    if fm_ != em_:
        problems.append(f"letter markers differ: French {fm_} vs English {em_}")
    keys = [n["key"] for n in sec.get("notes", []) if n.get("key") not in (None, "_")]
    have = set(MARK.findall(notes))
    for k in keys:
        if k not in have:
            problems.append(f"note {{{k}}} has no Notes entry")
    if not head.strip():
        problems.append("empty English body")
    return [f"{section_id}: {p}" for p in problems]


def main():
    sections = {s["id"]: s for s in json.loads((ROOT / "text/sections.json").read_text())["sections"]}
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    ids = [s for s in sections if (ROOT / "translation/sections" / f"{s}.md").exists()] if sys.argv[1] == "--all" else sys.argv[1:]
    problems = [p for sid in ids for p in check(sid, sections)]
    for p in problems: print("PROBLEM:", p)
    ok = len(ids) - len({p.split(":")[0] for p in problems})
    print(f"{ok} sections ok, {len(ids) - ok} failed")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

"""Verify that a section translation keeps its page and letter markers.

usage: check_markers.py SECTION_ID... | --all
--all checks every section with a translation file: one FAIL line per failing section,
then `N/M ok`; exit 1 on any failure.
Rules: the sequence of ⟦pNNN⟧ page markers in the English body must equal the French
section's; the sequence of {x} letter markers must be equal too; every note key in the
French section has a Notes entry `- {x}`; the front matter's id matches.
"""
import json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bookconf import ROOT  # noqa: E402  (the book root: see bookconf.py)
PAGE = re.compile(r"⟦(p[0-9]{3}(?:-[a-z]+)?)⟧")
MARK = re.compile(r"\{([a-zſ]+\d*)\}")
# Inline alternative readings (⟨alt:…⟩, ⟨alt?:…⟩) may quote a marker; reading A is
# the text before the span, so the span itself is dropped before counting markers.
ALT = re.compile(r"⟨alt\??:[^⟩]*⟩")


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
    french = ALT.sub("", sec["text"])
    fp, ep = PAGE.findall(french), PAGE.findall(head)
    if fp != ep:
        problems.append(f"page markers differ: French {fp} vs English {ep}")
    fm_, em_ = MARK.findall(french), MARK.findall(head)
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


def check_all(sections, out=print):
    """Check every section that has a translation file, in book order: one line per failing
    section, then `N/M ok`. Returns the exit code (1 on any failure)."""
    ids = [s for s in sections if (ROOT / "translation/sections" / f"{s}.md").exists()]
    failed = 0
    for sid in ids:
        problems = check(sid, sections)
        if problems:
            failed += 1
            prefix = f"{sid}: "
            out(f"FAIL {sid}: " + "; ".join(p[len(prefix):] if p.startswith(prefix) else p
                                          for p in problems))
    out(f"{len(ids) - failed}/{len(ids)} ok")
    return 1 if failed else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    sections = {s["id"]: s for s in json.loads((ROOT / "text/sections.json").read_text())["sections"]}
    if not argv:
        print(__doc__); return 2
    if argv[0] == "--all":
        return check_all(sections)
    problems = [p for sid in argv for p in check(sid, sections)]
    for p in problems: print("PROBLEM:", p)
    ok = len(argv) - len({p.split(":")[0] for p in problems})
    print(f"{ok} sections ok, {len(argv) - ok} failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Carry translation files across a re-cut of text/sections.json.

    uv run python scripts/migrate_section_ids.py --new NEW_SECTIONS.json [--old PATH] [--apply]

When stitch_text.py starts recognising a heading it used to miss (the misprinted
TEXTE headings "TFXTE." p045, "TBXTE." p058, "TEXTB." p072 and "ANNNT. LX." p080),
the section that swallowed it splits in two and every later id of that kind shifts.
This script aligns the old section list with the new one and moves the translation
files to match:

* the alignment walks both lists in order: an old section whose text (page markers
  and spacing ignored) equals the next new one maps to it; one whose text is the next
  new section's text followed by `<label> <text>` of the one after (and so on) is a
  split. Anything else stops the script: re-cut first, then migrate.
* `translation/sections/<old>.md` is renamed to `<new>.md` with its front-matter `id`
  (and `pages`, if they changed) rewritten. A split file is cut at the translated
  heading line ("TEXT", "ANNOTATION LX", ...), which is dropped since the heading is
  the section's label; each part gets its own front matter, the second part opens with
  its first page marker if the English lacks it, and the `## Notes` entries go to the
  part whose prose carries their `{x}` marker (else the part whose French notes have
  that key on that page; else the first part, reported). The translator's commentary
  paragraph stays with the first part.
* files in `translation/reports/` and `translation/alt-choices/` whose names embed
  old ids are renamed (in a range name `batch-A--B`, A becomes the first new id of A
  and B the last new id of B). Alt-choice entries are keyed by page-based `alt_id`;
  a `section` field, if one is ever present, is rewritten from the new alts.json.

Renames go through a temporary directory so a shift by one never overwrites a file.
The default is a dry run that prints the full plan; `--apply` carries it out.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE_MARK = re.compile(r"⟦(p[0-9]{3}(?:-[a-z]+)?)⟧")
LETTER = re.compile(r"\{([a-zſ]+\d*)\}")
ALT = re.compile(r"⟨alt\??:[^⟩]*⟩")
ID_RE = re.compile(r"(texte-\d+|annot-\d{3}|title|argument)")
NAME_ID_RE = re.compile(r"texte-\d+|annot-\d{3}")
# A translated heading line: TEXT / TEXTE / TEXT. / ANNOTATION LX / ANNOT. LX., maybe bold
HEADING_LINE = re.compile(r"^(?:#+\s*)?(?:\*\*)?\s*(TEXTE?|TEXT[A-Z]|ANNOT[A-Z]*)\b"
                          r"[^a-z]{0,30}?(?:\*\*)?\s*$")
ENTRY_RE = re.compile(r"^- \{([^}]+)\}(?:\s*\((p[0-9]{3}(?:-[a-z]+)?)[^)]*\))?")


class MigrationError(Exception):
    pass


def norm(text: str) -> str:
    return " ".join(PAGE_MARK.sub("", text or "").split())


def load_sections(path) -> list[dict]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))["sections"]


# --- alignment --------------------------------------------------------------

def compute_mapping(old: list[dict], new: list[dict]) -> list[dict]:
    """One record per old section: {"old", "new": [ids], "old_pages", "new_pages": [...]}."""
    out, i = [], 0
    for o in old:
        if i >= len(new):
            raise MigrationError(f"{o['id']}: no new section left to align with")
        rest = norm(o["text"])
        first = norm(new[i]["text"])
        if not rest.startswith(first):
            raise MigrationError(f"{o['id']} ({o['pages']}) does not align with new "
                                 f"{new[i]['id']} ({new[i]['pages']}): texts differ")
        parts, rest = [new[i]], rest[len(first):].strip()
        i += 1
        while rest:
            if i >= len(new):
                raise MigrationError(f"{o['id']}: text left over after the last new section")
            nxt = new[i]
            expect = " ".join(filter(None, [norm(nxt.get("label") or ""), norm(nxt["text"])]))
            if not rest.startswith(expect):
                raise MigrationError(f"{o['id']}: leftover text does not begin new "
                                     f"{nxt['id']} ({nxt.get('label')!r})")
            parts.append(nxt)
            rest = rest[len(expect):].strip()
            i += 1
        out.append({"old": o["id"], "new": [p["id"] for p in parts],
                    "old_pages": o["pages"], "new_pages": [p["pages"] for p in parts],
                    "labels": [p.get("label") for p in parts]})
    if i != len(new):
        raise MigrationError(f"new sections left unaligned: {[s['id'] for s in new[i:]]}")
    return out


# --- markdown files -----------------------------------------------------------

def split_front_matter(text: str):
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise MigrationError("no front matter")
    return m.group(1), text[m.end():]


def render_front_matter(fm: str, sid: str, pages: list[str]) -> str:
    lines, seen_id, seen_pages = [], False, False
    for line in fm.split("\n"):
        if re.match(r"^id:", line):
            line, seen_id = f"id: {sid}", True
        elif re.match(r"^pages:", line):
            line, seen_pages = f"pages: [{', '.join(pages)}]", True
        lines.append(line)
    if not seen_id:
        lines.insert(0, f"id: {sid}")
    if not seen_pages:
        lines.append(f"pages: [{', '.join(pages)}]")
    return "---\n" + "\n".join(lines) + "\n---\n"


def retarget(text: str, sid: str, pages: list[str]) -> str:
    fm, body = split_front_matter(text)
    return render_front_matter(fm, sid, pages) + body


def parse_notes(notes: str):
    """(preamble lines, [entry text]) from the part after '## Notes'."""
    pre, entries = [], []
    for line in notes.split("\n"):
        if line.startswith("- "):
            entries.append([line])
        elif entries:
            entries[-1].append(line)
        else:
            pre.append(line)
    return pre, ["\n".join(e).rstrip("\n") for e in entries]


def heading_kind(line: str):
    m = HEADING_LINE.match(line.strip())
    if not m:
        return None
    return "annotation" if m.group(1).startswith("ANNOT") else "texte"


def split_file(text: str, rec: dict, new_by_id: dict) -> tuple[list[tuple[str, str]], list[str]]:
    """Cut one translation into the parts of a split section.

    Returns ([(new_id, file text)], report lines)."""
    fm, body = split_front_matter(text)
    head, sep, notes = body.partition("\n## Notes")
    parts = [new_by_id[sid] for sid in rec["new"]]
    paras = re.split(r"(\n\s*\n)", head)          # keep the separators
    cuts = []                                       # indices into paras
    want = iter(parts[1:])
    target = next(want)
    for idx, chunk in enumerate(paras):
        if idx % 2 or not chunk.strip() or "\n" in chunk.strip():
            continue
        if heading_kind(chunk) == target["kind"]:
            cuts.append(idx)
            target = next(want, None)
            if target is None:
                break
    if len(cuts) != len(parts) - 1:
        raise MigrationError(f"{rec['old']}: found {len(cuts)} translated heading line(s), "
                             f"need {len(parts) - 1}")
    report = [f"cut at heading line {paras[c].strip()!r} (dropped; it is "
              f"{parts[k + 1]['id']}'s label {parts[k + 1].get('label')!r})"
              for k, c in enumerate(cuts)]
    bounds = [0] + cuts + [len(paras)]
    bodies = []
    for k in range(len(parts)):
        lo = bounds[k] + (1 if k else 0)            # skip the heading itself
        hi = bounds[k + 1]
        bodies.append("".join(paras[lo:hi]).strip("\n"))
    # page markers: each part opens with its first page marker, as the French does
    for k, part in enumerate(parts):
        m = PAGE_MARK.match(part["text"])
        if k and m and not bodies[k].startswith(m.group(0)):
            bodies[k] = m.group(0) + bodies[k]
            report.append(f"{part['id']}: prepended page marker {m.group(0)}")
    # notes
    pre, entries = parse_notes(notes) if sep else ([], [])
    marks = [set(LETTER.findall(ALT.sub("", b))) for b in bodies]
    french = [{(n.get("key"), n.get("page")) for n in p.get("notes") or []} for p in parts]
    assigned = [[] for _ in parts]
    for entry in entries:
        m = ENTRY_RE.match(entry)
        key, page = (m.group(1), m.group(2)) if m else (None, None)
        by_mark = [k for k, s in enumerate(marks) if key in s]
        by_note = [k for k, s in enumerate(french) if (key, page) in s]
        if len(by_mark) == 1:
            k = by_mark[0]
        elif len(by_note) == 1:
            k = by_note[0]
        else:
            k = 0
            report.append(f"note {entry[:40]!r}... not attributable; kept with {parts[0]['id']}")
        assigned[k].append(entry)
    outs = []
    for k, part in enumerate(parts):
        text_k = render_front_matter(fm, part["id"], part["pages"]) + bodies[k] + "\n"
        if sep:
            pre_k = "\n".join(pre).strip("\n") if k == 0 else (
                f"Split from {rec['old']} when the re-cut recognised the heading "
                f"{part.get('label')!r}; the translator's commentary is in {parts[0]['id']}.")
            block = "\n\n".join(filter(None, [pre_k, "\n".join(assigned[k])]))
            text_k += "\n## Notes\n" + block + "\n"
        outs.append((part["id"], text_k))
        report.append(f"{part['id']}: {len(assigned[k])} note entries, pages {part['pages']}")
    check_words(head, cuts, paras, [b for b in bodies], rec)
    return outs, report


def words(text: str) -> list[str]:
    return PAGE_MARK.sub(" ", text).split()


def check_words(head, cuts, paras, bodies, rec):
    """Every word of the English body survives the cut, bar the dropped heading lines."""
    dropped = set(cuts)
    expect = words("".join(c for i, c in enumerate(paras) if i not in dropped))
    if words(" ".join(bodies)) != expect:
        raise MigrationError(f"{rec['old']}: the cut would lose or reorder words")


# --- the plan -----------------------------------------------------------------

def id_maps(mapping):
    first = {r["old"]: r["new"][0] for r in mapping}
    last = {r["old"]: r["new"][-1] for r in mapping}
    return first, last


def rename_name(name: str, first: dict, last: dict) -> str:
    hits = list(NAME_ID_RE.finditer(name))
    if not hits:
        return name
    out, pos = [], 0
    for n, m in enumerate(hits):
        table = last if (n == len(hits) - 1 and len(hits) > 1) else first
        out.append(name[pos:m.start()] + table.get(m.group(0), m.group(0)))
        pos = m.end()
    return "".join(out) + name[pos:]


def build_plan(root, mapping, new_sections, new_alts=None):
    """A list of operations; nothing is touched."""
    root = pathlib.Path(root)
    new_by_id = {s["id"]: s for s in new_sections}
    by_old = {r["old"]: r for r in mapping}
    first, last = id_maps(mapping)
    ops, notes = [], []
    sec_dir = root / "translation/sections"
    for path in sorted(sec_dir.glob("*.md")) if sec_dir.exists() else []:
        rec = by_old.get(path.stem)
        if rec is None:
            notes.append(f"sections/{path.name}: id not in the old sections.json; left alone")
            continue
        if len(rec["new"]) > 1:
            parts, report = split_file(path.read_text(encoding="utf-8"), rec, new_by_id)
            ops.append({"op": "split", "dir": "translation/sections", "src": path.name,
                        "parts": [(f"{sid}.md", txt) for sid, txt in parts],
                        "report": report})
            continue
        sid, pages = rec["new"][0], rec["new_pages"][0]
        if sid == rec["old"] and pages == rec["old_pages"]:
            continue
        text = retarget(path.read_text(encoding="utf-8"), sid, pages)
        ops.append({"op": "rename", "dir": "translation/sections", "src": path.name,
                    "dst": f"{sid}.md", "text": text,
                    "pages": None if pages == rec["old_pages"] else (rec["old_pages"], pages)})
    for sub in ("translation/reports", "translation/alt-choices"):
        d = root / sub
        for path in sorted(d.iterdir()) if d.exists() else []:
            if not path.is_file():
                continue
            text = None
            if path.suffix == ".json":
                text = rewrite_alt_choices(path, new_alts)
            dst = rename_name(path.name, first, last)
            single = [m.group(0) for m in NAME_ID_RE.finditer(path.name)]
            if len(single) == 1 and len(by_old.get(single[0], {}).get("new", [])) > 1:
                notes.append(f"{sub}/{path.name}: covers split section {single[0]} "
                             f"(now {', '.join(by_old[single[0]]['new'])}); named for the first")
            if dst != path.name or text is not None:
                ops.append({"op": "rename", "dir": sub, "src": path.name, "dst": dst,
                            "text": text, "pages": None})
    check_collisions(root, ops)
    return ops, notes


def rewrite_alt_choices(path, new_alts):
    """New file text if any entry carries a `section` field that needs rewriting, else None."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("choices", [])
    if not any(isinstance(e, dict) and "section" in e for e in entries):
        return None
    if new_alts is None:
        raise MigrationError(f"{path.name}: entries carry `section`; need the new alts.json")
    home = {a["alt_id"]: a["section"] for a in new_alts}
    changed = False
    for e in entries:
        if isinstance(e, dict) and "section" in e and e.get("alt_id") in home \
                and e["section"] != home[e["alt_id"]]:
            e["section"], changed = home[e["alt_id"]], True
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n" if changed else None


def check_collisions(root, ops):
    for d in {op["dir"] for op in ops}:
        moving = {op["src"] for op in ops if op["dir"] == d}
        targets = []
        for op in ops:
            if op["dir"] != d:
                continue
            targets += [n for n, _ in op["parts"]] if op["op"] == "split" else [op["dst"]]
        dupes = {t for t in targets if targets.count(t) > 1}
        if dupes:
            raise MigrationError(f"{d}: two files would become {sorted(dupes)}")
        for t in targets:
            if (root / d / t).exists() and t not in moving:
                raise MigrationError(f"{d}/{t} exists and is not part of the migration")


def apply_plan(root, ops):
    root = pathlib.Path(root)
    for d in sorted({op["dir"] for op in ops}):
        tmp = root / d / ".migrate-tmp"
        if tmp.exists():
            raise MigrationError(f"{tmp} exists; a previous run did not finish")
        tmp.mkdir()
        mine = [op for op in ops if op["dir"] == d]
        for op in mine:                                  # 1. everything out of the way
            shutil.move(str(root / d / op["src"]), str(tmp / op["src"]))
        for op in mine:                                  # 2. into place
            src = tmp / op["src"]
            if op["op"] == "split":
                for name, text in op["parts"]:
                    (root / d / name).write_text(text, encoding="utf-8")
                src.unlink()
            elif op["text"] is not None:
                (root / d / op["dst"]).write_text(op["text"], encoding="utf-8")
                src.unlink()
            else:
                shutil.move(str(src), str(root / d / op["dst"]))
        tmp.rmdir()


def print_plan(mapping, ops, notes, out=print):
    changed = [r for r in mapping if r["new"] != [r["old"]]]
    splits = [r for r in mapping if len(r["new"]) > 1]
    out(f"alignment: {len(mapping)} old sections -> {sum(len(r['new']) for r in mapping)} new; "
        f"{len(changed)} change id or split, {len(splits)} split")
    for r in changed:
        tag = "SPLIT " if len(r["new"]) > 1 else ""
        out(f"  {tag}{r['old']} -> {' + '.join(r['new'])}"
            + (f"  (at {r['labels'][1]!r})" if len(r["new"]) > 1 else ""))
    out("")
    for op in ops:
        if op["op"] == "split":
            out(f"SPLIT  {op['dir']}/{op['src']} -> {', '.join(n for n, _ in op['parts'])}")
            for line in op["report"]:
                out(f"         {line}")
        else:
            extra = ""
            if op["pages"]:
                extra = f"  pages {op['pages'][0]} -> {op['pages'][1]}"
            if op["dst"] == op["src"]:
                out(f"EDIT   {op['dir']}/{op['src']}{extra}")
            else:
                out(f"RENAME {op['dir']}/{op['src']} -> {op['dst']}{extra}")
    for n in notes:
        out(f"NOTE   {n}")
    counts = {}
    for op in ops:
        key = (op["dir"], op["op"])
        counts[key] = counts.get(key, 0) + 1
    out("")
    out("summary: " + "; ".join(f"{d.split('/')[-1]} {o} {n}"
                               for (d, o), n in sorted(counts.items())) if counts
        else "summary: nothing to do")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default=None, help="old sections.json (default text/sections.json)")
    ap.add_argument("--new", required=True, help="re-cut sections.json (alts.json beside it)")
    ap.add_argument("--root", default=ROOT, help="project root")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True, help="print the plan (default)")
    g.add_argument("--apply", action="store_true", help="carry the plan out")
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root)
    old = load_sections(args.old or root / "text/sections.json")
    new_path = pathlib.Path(args.new)
    new = load_sections(new_path)
    alts_path = new_path.parent / "alts.json"
    new_alts = json.loads(alts_path.read_text(encoding="utf-8"))["alts"] \
        if alts_path.exists() else None
    try:
        mapping = compute_mapping(old, new)
        ops, notes = build_plan(root, mapping, new, new_alts)
    except MigrationError as e:
        print("PROBLEM:", e)
        return 1
    print_plan(mapping, ops, notes)
    if args.apply:
        apply_plan(root, ops)
        print(f"applied {len(ops)} operations")
    else:
        print("dry run: nothing written (use --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

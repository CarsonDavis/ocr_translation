"""Apply a review pass's plan JSON to the case file and the section translations.

usage: apply_review_plan.py PLAN.json... [--dry-run | --apply]

--dry-run (the default) prints what would change and every problem, writing nothing;
--apply writes. A plan has the shape
  {"pass": "pass-0-glossary",
   "case_file":    [{"old", "new", "reason"}],   exact, unique replacement in docs/case-file.md
   "decision_log": ["- pass-0-glossary: ..."],   appended under "## 12. Review decision log"
   "sections":     [{"id", "old", "new", "reason"}],  exact, unique replacement in the prose
                                                 of translation/sections/<id>.md (before ## Notes)
   "findings":     [{"id", "note"}],             not edits: each note becomes a bullet in
                                                 translation/review/<id>.md
   "uncertain":    [{...}]}                      never applied; listed only
Each entry's "old" must occur exactly once in the text it targets (0 = MISSING, >1 =
AMBIGUOUS); such entries are skipped. Entries apply in order on the evolving text. If a
section's ⟦pNNN⟧/{x} marker sequence changes, all of that file's edits are reverted
(MARKERS_CHANGED). After --apply: check_markers.py runs on each touched section, findings
go to translation/review/<id>.md and uncertain entries to translation/review/<pass>-uncertain.md.
A MISSING entry whose "new" is present is flagged "already applied?" (a rerun of the same
plan reports these, harmlessly). Exit 1 on any problem, 0 otherwise.
"""
import argparse, json, pathlib, re, subprocess, sys

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from bookconf import ROOT  # noqa: E402  (the book root: see bookconf.py)
CASE_FILE = "docs/case-file.md"
LOG_HEADING = "## 12. Review decision log"
MARKERS = re.compile(r"⟦p[0-9]{3}(?:-[a-z]+)?⟧|\{[a-zſ]+\d*\}")
KEYS = ("case_file", "sections")
TRUNC = 200


def run(args, **kw):
    """subprocess.run with cwd=ROOT (tests replace this)."""
    kw.setdefault("cwd", ROOT)
    return subprocess.run([str(x) for x in args], text=True, capture_output=True, **kw)


def trunc(s, n=TRUNC):
    s = s.replace("\n", "⏎")
    return s if len(s) <= n else s[:n] + "…"


def replace_unique(text, old, new):
    """(new_text, status): status is APPLIED, MISSING, MISSING_DONE (new present) or AMBIGUOUS."""
    if not old:
        return text, "MISSING"
    n = text.count(old)
    if n == 1:
        return text.replace(old, new, 1), "APPLIED"
    if n == 0:
        return text, "MISSING_DONE" if new and new in text else "MISSING"
    return text, "AMBIGUOUS"


def split_section(text):
    """(front matter incl. closing ---, prose, notes tail starting at '\\n## Notes' or '')."""
    m = re.match(r"---\n.*?\n---\n", text, re.S)
    fm, body = (m.group(0), text[m.end():]) if m else ("", text)
    i = body.find("\n## Notes")
    return (fm, body, "") if i < 0 else (fm, body[:i], body[i:])


def markers(prose):
    return MARKERS.findall(prose)


def label(status):
    return {"MISSING_DONE": "MISSING (already applied?)"}.get(status, status)


def is_problem(status):
    return status != "APPLIED"


class Result:
    def __init__(self):
        self.counts = {k: {"applied": 0, "missing": 0, "already_applied": 0, "ambiguous": 0,
                           "markers_changed": 0} for k in KEYS}
        self.log_added = 0
        self.log_skipped = 0
        self.notes = 0
        self.touched = []            # section ids with >=1 applied edit kept
        self.findings = {}           # id -> [(pass, entry, status)]
        self.uncertain = []          # (pass, entry)
        self.check = {}              # id -> (ok, output)
        self.problems = 0

    def count(self, key, status):
        col = {"APPLIED": "applied", "MISSING": "missing", "MISSING_DONE": "already_applied",
               "AMBIGUOUS": "ambiguous", "MARKERS_CHANGED": "markers_changed"}[status]
        self.counts[key][col] += 1
        if is_problem(status):
            self.problems += 1


def apply_case_file(plan, root, res, out):
    path = root / CASE_FILE
    if not plan.get("case_file") and not plan.get("decision_log"):
        return None
    text = path.read_text(encoding="utf-8")
    for e in plan.get("case_file", []):
        text, st = replace_unique(text, e.get("old", ""), e.get("new", ""))
        res.count("case_file", st)
        out(f"  case_file {label(st)}: {trunc(e.get('old', ''), 80)!r} -> "
            f"{trunc(e.get('new', ''), 80)!r}")
    text = add_log(text, plan.get("decision_log", []), res, out)
    return text


def add_log(text, lines, res, out):
    """Append lines under LOG_HEADING (created at the end if absent); skip verbatim repeats."""
    if not lines:
        return text
    present = set(text.splitlines())
    new = []
    for ln in lines:
        ln = ln.rstrip("\n")
        if ln in present or ln in new:
            res.log_skipped += 1
            out(f"  decision_log already present: {trunc(ln, 100)}")
        else:
            new.append(ln)
            res.log_added += 1
            out(f"  decision_log add: {trunc(ln, 100)}")
    if not new:
        return text
    lines_ = text.split("\n")
    idx = next((i for i, l in enumerate(lines_) if l.strip() == LOG_HEADING), None)
    if idx is None:
        text = text.rstrip("\n") + "\n\n" + LOG_HEADING + "\n\n" + "\n".join(new) + "\n"
        return text
    # end of the log section: the next level-2 heading (or EOF)
    end = next((j for j in range(idx + 1, len(lines_)) if lines_[j].startswith("## ")),
               len(lines_))
    sec = "\n".join(lines_[idx:end]).rstrip("\n")
    sep = "\n\n" if sec.strip() == LOG_HEADING else "\n"
    sec = sec + sep + "\n".join(new) + "\n"
    rest = "\n".join(lines_[end:])
    if rest:
        sec += "\n" + rest
    head = "\n".join(lines_[:idx])
    return (head + "\n" if head else "") + sec


def apply_sections(plan, root, res, out):
    """Returns {id: new_text} for files whose kept edits change them."""
    groups = {}
    for e in plan.get("sections", []):
        groups.setdefault(e.get("id", ""), []).append(e)
    writes = {}
    for sid, entries in groups.items():
        path = root / "translation/sections" / f"{sid}.md"
        if not sid or not path.exists():
            for e in entries:
                res.count("sections", "MISSING")
                res.findings.setdefault(sid, []).append((plan["pass"], e, "MISSING", "no such file"))
                out(f"  {sid}: MISSING (no file {path.name})")
            continue
        text = path.read_text(encoding="utf-8")
        fm, prose, notes = split_section(text)
        before = markers(prose)
        statuses = []
        for e in entries:
            prose, st = replace_unique(prose, e.get("old", ""), e.get("new", ""))
            statuses.append(st)
        if markers(prose) != before:
            statuses = ["MARKERS_CHANGED" if s == "APPLIED" else s for s in statuses]
            out(f"  {sid}: MARKERS_CHANGED, all edits to this file reverted")
            prose = None
        for e, st in zip(entries, statuses):
            res.count("sections", st)
            res.findings.setdefault(sid, []).append((plan["pass"], e, st, ""))
            out(f"  {sid} {label(st)}: {trunc(e.get('old', ''), 80)!r} -> "
                f"{trunc(e.get('new', ''), 80)!r}")
        if prose is not None and "APPLIED" in statuses:
            writes[sid] = fm + prose + notes
            if sid not in res.touched:
                res.touched.append(sid)
    return writes


def findings_block(pass_, items):
    lines = [f"## {pass_}", ""]
    for _, e, st, why in items:
        if st == "NOTE":
            lines.append(f"- {e.get('note', '').strip()}")
            continue
        reason = e.get("reason", "").strip() or "(no reason given)"
        lines.append(f"- {reason}: {trunc(e.get('old', ''))} → {trunc(e.get('new', ''))}")
        if is_problem(st):
            lines.append(f"  - PROBLEM: {label(st)}" + (f" ({why})" if why else "")
                         + (" — skipped" if st != "MARKERS_CHANGED" else " — reverted"))
    return "\n".join(lines) + "\n"


def write_findings(root, res, out):
    for sid, items in res.findings.items():
        if not sid:
            continue
        by_pass = {}
        for it in items:
            by_pass.setdefault(it[0], []).append(it)
        path = root / "translation/review" / f"{sid}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8") if path.exists() else f"# {sid} review findings\n"
        changed = False
        for p, its in by_pass.items():
            block = findings_block(p, its)
            if block in text:
                continue
            text = text.rstrip("\n") + "\n\n" + block
            changed = True
        if changed:
            path.write_text(text, encoding="utf-8")
            out(f"  wrote {path.relative_to(root)}")


def write_uncertain(root, pass_, entries, out):
    path = root / "translation/review" / f"{pass_}-uncertain.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {pass_}: uncertain items (not applied)", ""]
    body += [f"- {json.dumps(e, ensure_ascii=False)}" for e in entries]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    out(f"  wrote {path.relative_to(root)}")


def apply(plan_paths, root=ROOT, dry_run=True, out=print):
    root = pathlib.Path(root)
    res = Result()
    for pp in plan_paths:
        plan = json.loads(pathlib.Path(pp).read_text(encoding="utf-8"))
        plan.setdefault("pass", pathlib.Path(pp).stem)
        out(f"== {plan['pass']} ({pp}) ==")
        cf = apply_case_file(plan, root, res, out)
        writes = apply_sections(plan, root, res, out)
        for e in plan.get("findings") or []:
            res.notes += 1
            res.findings.setdefault(e.get("id", ""), []).append((plan["pass"], e, "NOTE", ""))
            out(f"  {e.get('id', '')} FINDING (note): {trunc(e.get('note', ''), 100)}")
        unc = plan.get("uncertain") or []
        for e in unc:
            res.uncertain.append((plan["pass"], e))
            out(f"  UNCERTAIN (not applied): {json.dumps(e, ensure_ascii=False)}")
        if not dry_run:
            if cf is not None:
                (root / CASE_FILE).write_text(cf, encoding="utf-8")
            for sid, t in writes.items():
                (root / "translation/sections" / f"{sid}.md").write_text(t, encoding="utf-8")
            if unc:
                write_uncertain(root, plan["pass"], unc, out)
    if not dry_run:
        write_findings(root, res, out)
        for sid in res.touched:
            r = run(["uv", "run", "python", "scripts/check_markers.py", sid], cwd=root)
            res.check[sid] = (r.returncode == 0, (r.stdout or "") + (r.stderr or ""))
            if r.returncode != 0:
                res.problems += 1
    summary(res, dry_run, out)
    return 1 if res.problems else 0


def summary(res, dry_run, out):
    out("")
    out(f"== SUMMARY ({'dry run, nothing written' if dry_run else 'applied'}) ==")
    if any(c["already_applied"] for c in res.counts.values()):
        out("NOTE: 'already applied?' = old text absent but new text present "
            "(expected on a rerun; harmless)")
    for k in KEYS:
        c = res.counts[k]
        out(f"{k:<10} applied {c['applied']}  missing {c['missing']}  "
            f"already applied? {c['already_applied']}  ambiguous {c['ambiguous']}"
            + (f"  markers_changed {c['markers_changed']}" if k == "sections" else ""))
    out(f"decision_log added {res.log_added}  already present {res.log_skipped}")
    out(f"findings notes {res.notes}")
    out(f"touched sections ({len(res.touched)}): {', '.join(res.touched) or '-'}")
    if res.uncertain:
        out(f"uncertain ({len(res.uncertain)}, not applied):")
        for p, e in res.uncertain:
            out(f"  {p}: {trunc(json.dumps(e, ensure_ascii=False), 160)}")
    if res.check:
        bad = [s for s, (ok, _) in res.check.items() if not ok]
        out(f"check_markers: {len(res.check) - len(bad)}/{len(res.check)} ok")
        for s in bad:
            for ln in res.check[s][1].strip().splitlines():
                out(f"  {s}: {ln}")
    elif not dry_run:
        out("check_markers: no sections touched")
    out(f"problems: {res.problems}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plans", nargs="+")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True)
    g.add_argument("--apply", action="store_true")
    ap.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    return apply(a.plans, root=a.root, dry_run=not a.apply)


if __name__ == "__main__":
    sys.exit(main())

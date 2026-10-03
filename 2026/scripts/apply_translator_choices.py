#!/usr/bin/env python3
"""Feed the translator's ⟨alt⟩ choices back into the arbitration decisions.

    uv run python scripts/apply_translator_choices.py [--dry-run] [--no-refinalize] CHOICES.json...

Each CHOICES file (translation/alt-choices/<id>.json, written by the translator, see
scripts/prompts/translate.md) is a JSON list of {"alt_id", "choice": "A"|"B"|"either",
"reason"}. Every alt_id is looked up in text/alts.json (written by stitch_text.py) to get
its page, `where` and the two line readings, and matched to the one item of
transcription/arbitration/queue/<page>.json whose readings are the same (narrowed by
`where` when several are). For choice A or B the decision in
transcription/arbitration/decisions/<page>.json becomes
{"choice", "by": "translator", "at", "reason"}:

  - a decision by Carson (`"by": "carson"`, or no `by` at all) is never overwritten; it is
    reported as skipped;
  - an `auto` (or earlier `translator`) decision is replaced;
  - "either" changes nothing: the auto decision stays and the French keeps reading A.

Nothing is written when any alt_id is unknown, any queue match is missing or ambiguous, or
two markers of one line were given different choices (exit 1). Afterwards
`wave.py refinalize` runs for the pages that changed, unless --no-refinalize. --dry-run
prints what would change and writes nothing.
"""
import argparse, datetime, json, os, pathlib, subprocess, sys, tempfile

SCRIPTS = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
CHOICES = ("A", "B", "either")
PROTECTED = "carson"


def run(args, **kw):
    """subprocess.run with cwd=ROOT (tests replace this)."""
    kw.setdefault("cwd", ROOT)
    return subprocess.run([str(x) for x in args], text=True, **kw)


def _alt(t):
    """The form apply_arbitration writes a reading in (mirrors apply_arbitration._alt)."""
    return "(no line)" if t is None else t.replace("\n", " / ")


class FeedbackError(Exception):
    pass


def load_choices(paths):
    """[(file, entry)] from the alt-choices files; malformed entries raise."""
    out = []
    for p in paths:
        data = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise FeedbackError(f"{p}: expected a JSON list")
        for e in data:
            if not isinstance(e, dict) or not e.get("alt_id") or e.get("choice") not in CHOICES:
                raise FeedbackError(f"{p}: bad entry {e!r} (need alt_id and choice A|B|either)")
            out.append((str(p), e))
    return out


def match_item(items, rec):
    """The one queue item for an alt record (by readings, then by where)."""
    hits = [it for it in items if _alt(it.get("a")) == rec["a"] and _alt(it.get("b")) == rec["b"]]
    if len(hits) > 1:
        hits = [it for it in hits if rec["where"] in (it.get("where_a"), it.get("where_b"))]
    if len(hits) != 1:
        what = "no" if not hits else f"{len(hits)} (ambiguous)"
        raise FeedbackError(f"{rec['alt_id']}: {what} queue item in {rec['page']} with "
                            f"a={rec['a']!r} b={rec['b']!r} where={rec['where']}")
    return hits[0]


def read_decisions(path, pid):
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if "decisions" not in data and data and all(isinstance(v, dict) for v in data.values()):
        data = {"page": pid, "decisions": data}          # bare {item: decision} file
    data.setdefault("page", pid)
    data.setdefault("decisions", {})
    return data


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def plan(choices, alts, root=ROOT):
    """Resolve every choice to (page, item id). Returns
    ({page: {item_id: (choice, reason, [alt_ids])}}, unmatched alt_ids, errors)."""
    by_id = {r["alt_id"]: r for r in alts}
    queues, out, unmatched, errors = {}, {}, [], []
    for src, e in choices:
        rec = by_id.get(e["alt_id"])
        if rec is None:
            unmatched.append(e["alt_id"])
            continue
        pid = rec["page"]
        if pid not in queues:
            q = root / "transcription/arbitration/queue" / f"{pid}.json"
            if not q.exists():
                errors.append(f"{e['alt_id']}: no queue file {q.relative_to(root)}")
                continue
            queues[pid] = json.loads(q.read_text(encoding="utf-8")).get("items") or []
        try:
            item = match_item(queues[pid], rec)
        except FeedbackError as exc:
            errors.append(str(exc))
            continue
        page = out.setdefault(pid, {})
        reason = str(e.get("reason") or "").strip()
        prev = page.get(item["id"])
        if prev is None:
            page[item["id"]] = (e["choice"], reason, [e["alt_id"]])
            continue
        # several markers on one line: the line has one choice ("either" yields)
        choice = prev[0]
        if e["choice"] != "either":
            if choice not in ("either", e["choice"]):
                errors.append(f"{item['id']} on {pid}: conflicting choices {choice} "
                              f"({', '.join(prev[2])}) and {e['choice']} ({e['alt_id']})")
                continue
            choice = e["choice"]
        reasons = prev[1] if reason in prev[1] else "; ".join(x for x in (prev[1], reason) if x)
        page[item["id"]] = (choice, reasons, prev[2] + [e["alt_id"]])
    return out, unmatched, errors


def apply(choice_paths, root=ROOT, dry_run=False, refinalize=True, now=None, out=print):
    """Returns the process exit code."""
    root = pathlib.Path(root)
    alts_file = root / "text/alts.json"
    if not alts_file.exists():
        out(f"ERROR: {alts_file} not found (run stitch_text.py)")
        return 1
    alts = json.loads(alts_file.read_text(encoding="utf-8")).get("alts") or []
    try:
        choices = load_choices(choice_paths)
    except (FeedbackError, json.JSONDecodeError, OSError) as exc:
        out(f"ERROR: {exc}")
        return 1
    resolved, unmatched, errors = plan(choices, alts, root)
    if unmatched or errors:
        for a in unmatched:
            out(f"UNMATCHED alt_id {a} (not in text/alts.json)")
        for e in errors:
            out(f"ERROR: {e}")
        out(f"nothing written: {len(unmatched)} unmatched alt_ids, {len(errors)} errors")
        return 1

    at = now or datetime.datetime.now().isoformat(timespec="seconds")
    changed, skipped, summary = [], [], []
    for pid in sorted(resolved):
        dpath = root / "transcription/arbitration/decisions" / f"{pid}.json"
        data = read_decisions(dpath, pid)
        dec = data["decisions"]
        n_set = n_either = n_same = 0
        for item_id, (choice, reason, alt_ids) in sorted(resolved[pid].items()):
            cur = dec.get(item_id) if isinstance(dec.get(item_id), dict) else None
            if choice == "either":
                n_either += 1
                continue
            if cur and cur.get("choice") and cur.get("by", PROTECTED) == PROTECTED:
                skipped.append(f"{pid} {item_id} ({', '.join(alt_ids)}): carson chose "
                               f"{cur.get('choice')}, translator {choice}")
                continue
            if cur and cur.get("by") == "translator" and cur.get("choice") == choice \
                    and cur.get("reason") == reason:
                n_same += 1
                continue
            new = {"choice": choice, "by": "translator", "at": at, "reason": reason}
            out(f"{'would set' if dry_run else 'set'} {pid} {item_id}: "
                f"{(cur or {}).get('choice', '-')} ({(cur or {}).get('by', 'none')}) -> "
                f"{choice} (translator)  [{', '.join(alt_ids)}]")
            dec[item_id] = new
            n_set += 1
        summary.append(f"{pid}\t{n_set} applied\t{n_either} either (kept)\t{n_same} unchanged")
        if n_set:
            changed.append(pid)
            if not dry_run:
                write_json(dpath, data)

    out("page\tapplied\teither\tunchanged")
    for line in summary:
        out(line)
    for s in skipped:
        out(f"SKIPPED (carson decision): {s}")
    out(f"{sum(len(v) for v in resolved.values())} lines, {len(changed)} pages changed, "
        f"{len(skipped)} carson decisions skipped, 0 unmatched alt_ids"
        + ("   (dry run: nothing written)" if dry_run else ""))
    if dry_run or not changed:
        return 0
    if not refinalize:
        out("not refinalized (--no-refinalize); run: uv run python scripts/wave.py refinalize "
            + " ".join(changed))
        return 0
    r = run([sys.executable, SCRIPTS / "wave.py", "refinalize", *changed], cwd=root)
    return 1 if r.returncode else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("choices", nargs="+", help="translation/alt-choices/*.json files")
    ap.add_argument("--dry-run", action="store_true", help="print what would change")
    ap.add_argument("--no-refinalize", action="store_true",
                    help="write decisions but do not run wave.py refinalize")
    ap.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    return apply(a.choices, root=a.root, dry_run=a.dry_run, refinalize=not a.no_refinalize)


if __name__ == "__main__":
    sys.exit(main())

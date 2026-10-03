"""Wave helpers for the full transcription run (single-pass reads + human arbitration).

  wave.py next [--size 8] [--context 3] [--model opus] [--redispatch]
      Pick the next pages in manifest order that lack a final and are missing read A
      and/or read B, render a `read_single` prompt for each missing reader into the
      scratchpad ($CORAS_SCRATCH), mark status.read<X> = dispatched, and print one line per
      prompt: page, reader, prompt path. A reader already marked `dispatched` is skipped
      (it is in flight) unless --redispatch is given (for stale marks of dead agents).
  wave.py queue [--rebuild] [PAGE ...]
      For every page with both reads, no final and no arbitration queue yet: normalize
      spacing on both reads, auto_resolve, diff_reads (transcription/diff/<id>.md), then
      arbitrate_queue.py <id>. Prints page, agreement, items. Pages whose queue exists are
      skipped unless --rebuild.
  wave.py apply [PAGE ...]
      For every page without a final whose queue has every item decided in
      transcription/arbitration/decisions/<id>.json (a queue of 0 items needs no decisions
      file), run apply_arbitration.py; the final is moved into transcription/final/ only
      if apply exits 0 (valid). Then sync the manifest and re-stitch text/sections.json.
      A page with an undecided item is never applied.
  wave.py refinalize PAGE [PAGE ...]
      Re-run apply_arbitration for pages that already have a final (after a translator's
      or Carson's re-decision), replacing the final atomically only if apply exits 0;
      then sync the manifest and re-stitch as apply does. The page must still be fully
      decided, and apply's staleness check refuses a queue that no longer matches the
      reads (the old final is then left as it was).
  wave.py defer [--dry-run] [PAGE ...]          (also: scripts/auto_defer.py ...)
      For every queue page (or only PAGE ...), write a decision for every item that has
      none: "either" for line items (body / note / unmatched / flagged), "unknown" for
      structure (structural, note-structure), each with "by": "auto", "at" and
      "reason": "auto-deferred to translator". An existing decision is never
      overwritten (a legacy "skip" counts as none). --dry-run writes nothing. Prints
      page, deferred either, deferred unknown, already decided.
  wave.py status [--all]
      Sync manifest flags with the files on disk, then list per page: reads present,
      queue present, decided/total, final present (pages with any activity and no final;
      --all lists final pages too).

Run under `uv run --with pillow --with jsonschema python scripts/wave.py ...`; when the
current interpreter lacks pillow/jsonschema the helper scripts are started via `uv run`.
"""
import argparse, datetime, importlib.util, json, os, pathlib, subprocess, sys, tempfile

SCRIPTS = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
SCRATCH = pathlib.Path(os.environ.get("CORAS_SCRATCH",
    "/private/tmp/claude-502/-Users-cdavis-github-translator/f7bb6907-97b3-42f5-958f-8758af1eb5f7/scratchpad/waves"))
PY = sys.executable
READERS = ("A", "B")
# mirrors apply_arbitration.CHOICES / LEGACY ("skip" = undecided)
CHOICES = {"A", "B", "neither", "either", "unknown", "both"}
STRUCTURE_KINDS = {"structural", "note-structure"}     # deferred as "unknown"
DEFER_REASON = "auto-deferred to translator"


def run(args, **kw):
    """subprocess.run with text output, cwd=ROOT (tests replace this)."""
    kw.setdefault("cwd", ROOT)
    return subprocess.run([str(x) for x in args], capture_output=True, text=True, **kw)


def pycmd(script, needs=()):
    """Command prefix for a helper script, adding uv --with for missing modules."""
    missing = [m for m in needs if importlib.util.find_spec(m) is None]
    if not missing:
        return [PY, SCRIPTS / script]
    uv = ["uv", "run"]
    for m in missing:
        uv += ["--with", m]
    return uv + ["python", SCRIPTS / script]


def manifest():
    return json.loads((ROOT / "manifest.json").read_text())


def write_manifest(m):
    sys.path.insert(0, str(SCRIPTS)); import pagelib
    pagelib.write_manifest(ROOT / "manifest.json", m)


def path(kind, pid):
    return ROOT / "transcription" / kind / f"{pid}.json"


def has(kind, pid):
    return path(kind, pid).exists()


def both_reads(pid):
    return all(has(f"reads/{rd}", pid) for rd in READERS)


def progress(pid):
    """(decided, total) for the page's arbitration queue, or None if there is no queue."""
    q = path("arbitration/queue", pid)
    if not q.exists():
        return None
    items = json.loads(q.read_text()).get("items") or []
    d = path("arbitration/decisions", pid)
    dec = {}
    if d.exists():
        data = json.loads(d.read_text())
        dec = data.get("decisions", data) if isinstance(data, dict) else {}
    ok = sum(1 for it in items if isinstance(dec.get(it["id"]), dict)
             and dec[it["id"]].get("choice") in CHOICES)
    return ok, len(items)


# --- next -------------------------------------------------------------------

def render(pid, reader, context=3, model="opus"):
    out = SCRATCH / f"read_single-{pid}-{reader}.md"
    r = run([PY, SCRIPTS / "render_prompt.py", "read_single", pid, "--reader", reader,
             "--context", context, "--model", model, "--out-dir", "transcription/reads/{READER}"])
    if r.returncode:
        raise SystemExit(f"render_prompt failed for {pid} {reader}: {r.stderr}")
    out.write_text(r.stdout)
    return out


def select_next(m, size, redispatch=False):
    """[(record, [missing readers])] for the next `size` pages needing reads."""
    picked = []
    for r in m["pages"]:
        pid = r["id"]
        if has("final", pid):
            continue
        todo = [rd for rd in READERS if not has(f"reads/{rd}", pid)
                and (redispatch or r["status"].get(f"read{rd}") != "dispatched")]
        if todo:
            picked.append((r, todo))
            if len(picked) == size:
                break
    return picked


def cmd_next(a):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    m = manifest()
    picked = select_next(m, a.size, a.redispatch)
    for r, todo in picked:
        for rd in todo:
            print(r["id"], rd, render(r["id"], rd, a.context, a.model))
            r["status"][f"read{rd}"] = "dispatched"
    if picked:
        write_manifest(m)
    else:
        print("nothing left to read")


# --- queue ------------------------------------------------------------------

def select_queue(m, rebuild=False, only=None):
    out = []
    for r in m["pages"]:
        pid = r["id"]
        if only and pid not in only:
            continue
        if has("final", pid) or not both_reads(pid):
            continue
        if has("arbitration/queue", pid) and not rebuild:
            continue
        out.append(pid)
    return out


def build_queue(pid):
    """Run the four steps for one page; return (agreement, items) or raise RuntimeError."""
    reads = [path(f"reads/{rd}", pid) for rd in READERS]
    steps = [
        ("normalize_spacing", pycmd("normalize_spacing.py") + reads),
        ("auto_resolve", pycmd("auto_resolve.py") + [pid]),
        ("diff_reads", pycmd("diff_reads.py") + [pid]),
        ("arbitrate_queue", pycmd("arbitrate_queue.py", needs=("PIL",)) + [pid]),
    ]
    agreement = "?"
    for name, cmd in steps:
        r = run(cmd)
        if r.returncode:
            raise RuntimeError(f"{name} exit {r.returncode}: {(r.stderr or r.stdout).strip()[:300]}")
        if name == "diff_reads":
            for kv in r.stdout.split():
                if kv.startswith("agreement="):
                    agreement = kv.split("=", 1)[1]
    prog = progress(pid)
    return agreement, (prog[1] if prog else "?")


def cmd_queue(a):
    todo = select_queue(manifest(), a.rebuild, set(a.pages) or None)
    if not todo:
        print("no pages to queue")
        return
    print("page\tagreement\titems")
    for pid in todo:
        if has("arbitration/decisions", pid):
            print(f"{pid}\tWARNING\trebuilding over an existing decisions file; check it still matches")
        try:
            agreement, items = build_queue(pid)
            print(f"{pid}\t{agreement}\t{items}")
        except RuntimeError as exc:
            print(f"{pid}\tERROR\t{exc}")


# --- apply ------------------------------------------------------------------

def select_apply(m, only=None):
    """Pages with a queue, every item decided, and no final."""
    out = []
    for r in m["pages"]:
        pid = r["id"]
        if only and pid not in only:
            continue
        if has("final", pid):
            continue
        prog = progress(pid)
        if prog is not None and prog[0] == prog[1]:
            out.append(pid)
    return out


def apply_page(pid):
    """Run apply_arbitration into a temp file; move it to final/ only on exit 0.

    The temp file sits beside final/ (same filesystem), so the move is an atomic replace:
    an existing final (refinalize) is either kept whole or replaced whole."""
    prog = progress(pid)
    if prog is None or prog[0] != prog[1]:           # belt and braces
        raise RuntimeError(f"undecided items ({prog})")
    dest = path("final", pid)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent.parent, prefix=".apply-") as td:
        tmp = pathlib.Path(td) / f"{pid}.json"
        r = run(pycmd("apply_arbitration.py", needs=("jsonschema",)) + [pid, "--out", tmp])
        if r.returncode or not tmp.exists():
            raise RuntimeError(f"apply exit {r.returncode}: {(r.stderr + r.stdout).strip()[-400:]}")
        os.replace(tmp, dest)
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


def cmd_apply(a):
    todo = select_apply(manifest(), set(a.pages) or None)
    if not todo:
        print("no fully decided pages without a final")
    finalize(todo, "finalized")


def cmd_refinalize(a):
    todo = []
    for pid in a.pages:
        if not has("final", pid):
            print(f"SKIPPED {pid}: no final yet (use apply)")
        elif not has("arbitration/queue", pid):
            print(f"SKIPPED {pid}: no arbitration queue (final not made by apply_arbitration)")
        else:
            todo.append(pid)
    finalize(todo, "refinalized")


def finalize(todo, verb):
    """apply_page each page, then sync the manifest and re-stitch if anything changed."""
    done = []
    for pid in todo:
        try:
            msg = apply_page(pid)
            done.append(pid)
            print(f"{verb} {pid} -> transcription/final/{pid}.json  {msg}")
        except RuntimeError as exc:
            print(f"FAILED {pid}: {exc}")
    n, total = sync_finals()
    print(f"finals: {total} ({n} newly marked)")
    if done:
        r = run(pycmd("stitch_text.py"))
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        print(("stitch: " if not r.returncode else "stitch FAILED: ") + last)
        if r.returncode:
            print(r.stderr.strip())


# --- defer ------------------------------------------------------------------

def defer_choice(kind):
    return "unknown" if kind in STRUCTURE_KINDS or str(kind).startswith("structural") else "either"


def defer_page(pid, dry_run=False, now=None):
    """Auto-defer every undecided item of one queue page.

    Returns {"either": n, "unknown": n, "decided": n}; writes the decisions file
    (atomically) unless dry_run or nothing to add. Existing decisions are kept as is."""
    items = json.loads(path("arbitration/queue", pid).read_text()).get("items") or []
    dpath = path("arbitration/decisions", pid)
    data = json.loads(dpath.read_text()) if dpath.exists() else {}
    if not isinstance(data, dict):
        raise RuntimeError(f"{dpath} is not a JSON object")
    if "decisions" not in data and data and all(isinstance(v, dict) for v in data.values()):
        data = {"page": pid, "decisions": data}          # bare {item: decision} file
    data.setdefault("page", pid)
    dec = data.setdefault("decisions", {})
    at = now or datetime.datetime.now().isoformat(timespec="seconds")
    counts = {"either": 0, "unknown": 0, "decided": 0}
    for it in items:
        cur = dec.get(it["id"])
        if isinstance(cur, dict) and cur.get("choice") in CHOICES:
            counts["decided"] += 1
            continue
        choice = defer_choice(it.get("kind"))
        counts[choice] += 1
        dec[it["id"]] = {"choice": choice, "by": "auto", "at": at, "reason": DEFER_REASON}
    if not dry_run and (counts["either"] or counts["unknown"]):
        dpath.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dpath.parent, prefix=dpath.name + ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        os.replace(tmp, dpath)
    return counts


def defer_kinds(pid):
    """{kind: n} of the undecided items of a page (for --dry-run)."""
    items = json.loads(path("arbitration/queue", pid).read_text()).get("items") or []
    dpath = path("arbitration/decisions", pid)
    data = json.loads(dpath.read_text()) if dpath.exists() else {}
    dec = data.get("decisions", data) if isinstance(data, dict) else {}
    out = {}
    for it in items:
        cur = dec.get(it["id"])
        if not (isinstance(cur, dict) and cur.get("choice") in CHOICES):
            out[it.get("kind")] = out.get(it.get("kind"), 0) + 1
    return out


def queue_pages(only=None):
    """Page ids with a queue file, in manifest order (ids the manifest lacks last)."""
    qdir = ROOT / "transcription" / "arbitration" / "queue"
    pids = {p.stem for p in qdir.glob("*.json")} if qdir.is_dir() else set()
    if only:
        missing = sorted(set(only) - pids)
        for pid in missing:
            print(f"SKIPPED {pid}: no arbitration queue")
        pids &= set(only)
    rank = {r["id"]: n for n, r in enumerate(manifest()["pages"])}
    return sorted(pids, key=lambda p: (rank.get(p, len(rank)), p))


def cmd_defer(a):
    pids = queue_pages(set(a.pages) or None)
    if not pids:
        print("no queue pages"); return
    print("page\tdeferred either\tdeferred unknown\talready decided"
          + ("\tby kind (dry run)" if a.dry_run else ""))
    tot = {"either": 0, "unknown": 0, "decided": 0}
    for pid in pids:
        c = defer_page(pid, dry_run=a.dry_run)
        for k in tot:
            tot[k] += c[k]
        row = f"{pid}\t{c['either']}\t{c['unknown']}\t{c['decided']}"
        if a.dry_run:
            row += "\t" + (", ".join(f"{k}={n}" for k, n in sorted(defer_kinds(pid).items())) or "-")
        print(row)
    print(f"total\t{tot['either']}\t{tot['unknown']}\t{tot['decided']}"
          + ("   (dry run: nothing written)" if a.dry_run else ""))


# --- status -----------------------------------------------------------------

def sync_finals():
    """Mark status.final = done for every page whose final file exists, and
    status.read<X> = done for every read on disk."""
    m = manifest(); n = 0
    for r in m["pages"]:
        if has("final", r["id"]) and r["status"].get("final") != "done":
            r["status"]["final"] = "done"; n += 1
        for rd in READERS:
            if has(f"reads/{rd}", r["id"]) and r["status"].get(f"read{rd}") != "done":
                r["status"][f"read{rd}"] = "done"
    write_manifest(m)
    return n, sum(r["status"].get("final") == "done" for r in m["pages"])


def status_rows(m, show_all=False):
    rows = []
    for r in m["pages"]:
        pid, st = r["id"], r["status"]
        final = has("final", pid)
        reads = "".join(rd if has(f"reads/{rd}", pid) else
                        (rd.lower() if st.get(f"read{rd}") == "dispatched" else "-")
                        for rd in READERS)
        prog = progress(pid)
        if final and not show_all:
            continue
        if not final and reads == "--" and prog is None:
            continue
        rows.append((pid, reads, "yes" if prog else "-",
                     f"{prog[0]}/{prog[1]}" if prog else "-", "yes" if final else "-"))
    return rows


def cmd_status(a):
    n, total = sync_finals()
    print(f"finals: {total} ({n} newly marked)")
    rows = status_rows(manifest(), a.all)
    if not rows:
        print("no pages pending"); return
    print("page\treads\tqueue\tdecided\tfinal   (reads: A/B on disk, a/b dispatched, - none)")
    for row in rows:
        print("\t".join(row))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("next"); n.add_argument("--size", type=int, default=8)
    n.add_argument("--context", type=int, default=3); n.add_argument("--model", default="opus")
    n.add_argument("--redispatch", action="store_true", help="ignore stale `dispatched` marks")
    q = sub.add_parser("queue"); q.add_argument("--rebuild", action="store_true"); q.add_argument("pages", nargs="*")
    p = sub.add_parser("apply"); p.add_argument("pages", nargs="*")
    rf = sub.add_parser("refinalize"); rf.add_argument("pages", nargs="+")
    d = sub.add_parser("defer"); d.add_argument("--dry-run", action="store_true")
    d.add_argument("pages", nargs="*")
    s = sub.add_parser("status"); s.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    {"next": cmd_next, "queue": cmd_queue, "apply": cmd_apply, "status": cmd_status,
     "refinalize": cmd_refinalize, "defer": cmd_defer}[a.cmd](a)


if __name__ == "__main__":
    main()

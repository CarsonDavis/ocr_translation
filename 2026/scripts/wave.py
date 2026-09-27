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
  wave.py status [--all]
      Sync manifest flags with the files on disk, then list per page: reads present,
      queue present, decided/total, final present (pages with any activity and no final;
      --all lists final pages too).

Run under `uv run --with pillow --with jsonschema python scripts/wave.py ...`; when the
current interpreter lacks pillow/jsonschema the helper scripts are started via `uv run`.
"""
import argparse, importlib.util, json, os, pathlib, shutil, subprocess, sys, tempfile

SCRIPTS = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
SCRATCH = pathlib.Path(os.environ.get("CORAS_SCRATCH",
    "/private/tmp/claude-502/-Users-cdavis-github-translator/f7bb6907-97b3-42f5-958f-8758af1eb5f7/scratchpad/waves"))
PY = sys.executable
READERS = ("A", "B")
# mirrors apply_arbitration.CHOICES / LEGACY ("skip" = undecided)
CHOICES = {"A", "B", "neither", "either", "unknown", "both"}


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
    """Run apply_arbitration into a temp file; move it to final/ only on exit 0."""
    prog = progress(pid)
    if prog is None or prog[0] != prog[1]:           # belt and braces
        raise RuntimeError(f"undecided items ({prog})")
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / f"{pid}.json"
        r = run(pycmd("apply_arbitration.py", needs=("jsonschema",)) + [pid, "--out", tmp])
        if r.returncode or not tmp.exists():
            raise RuntimeError(f"apply exit {r.returncode}: {(r.stderr + r.stdout).strip()[-400:]}")
        dest = path("final", pid)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), str(dest))
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


def cmd_apply(a):
    todo = select_apply(manifest(), set(a.pages) or None)
    done = []
    for pid in todo:
        try:
            msg = apply_page(pid)
            done.append(pid)
            print(f"finalized {pid} -> transcription/final/{pid}.json  {msg}")
        except RuntimeError as exc:
            print(f"FAILED {pid}: {exc}")
    if not todo:
        print("no fully decided pages without a final")
    n, total = sync_finals()
    print(f"finals: {total} ({n} newly marked)")
    if done:
        r = run(pycmd("stitch_text.py"))
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        print(("stitch: " if not r.returncode else "stitch FAILED: ") + last)
        if r.returncode:
            print(r.stderr.strip())


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
    s = sub.add_parser("status"); s.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    {"next": cmd_next, "queue": cmd_queue, "apply": cmd_apply, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Wave helpers for the full transcription run.

  wave.py next [--size 8] [--context 3] [--model opus]
      Pick the next pages in manifest order that lack a final and lack reads, render
      read prompts A and B for each into the scratchpad, and print the page ids and
      prompt paths (one line per prompt) for dispatch.
  wave.py status
      For every page with both reads: normalize spacing, run the diff, and classify as
      AUTO (100% agreement, no structural or note differences → make_final from A) or
      RECONCILE (render the reconcile prompt). Print a table. Pages with a final are
      skipped. Pages with only one read are listed as WAITING.
"""
import argparse, json, os, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRATCH = pathlib.Path(os.environ.get("CORAS_SCRATCH",
    "/private/tmp/claude-502/-Users-cdavis-github-translator/b3292afe-bea5-456a-a8de-96de57a2f4ef/scratchpad/waves"))
PY = sys.executable

def manifest():
    return json.loads((ROOT / "manifest.json").read_text())

def has(kind, pid):
    return (ROOT / "transcription" / kind / f"{pid}.json").exists()

def render(kind, pid, **kw):
    out = SCRATCH / (f"{kind}-{pid}" + (f"-{kw['reader']}" if kw.get("reader") else "") + ".md")
    args = [PY, ROOT / "scripts/render_prompt.py", kind, pid, "--context", str(kw.get("context", 3)),
            "--model", kw.get("model", "opus")]
    if kw.get("reader"): args += ["--reader", kw["reader"]]
    out.write_text(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    return out

def cmd_next(a):
    """Pages are skipped once dispatched (manifest status.readA == "dispatched"), so
    readers still in flight are not picked twice."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    m = manifest(); picked = []
    for r in m["pages"]:
        pid = r["id"]
        if has("final", pid) or has("reads/A", pid) or has("reads/B", pid):
            continue
        if r["status"].get("readA") in ("dispatched", "done"):
            continue
        picked.append(r)
        if len(picked) == a.size: break
    for r in picked:
        for rd in ("A", "B"):
            print(r["id"], rd, render("read", r["id"], reader=rd, context=a.context, model=a.model))
            r["status"][f"read{rd}"] = "dispatched"
    if picked:
        sys.path.insert(0, str(ROOT / "scripts")); import pagelib
        pagelib.write_manifest(ROOT / "manifest.json", m)
    else:
        print("nothing left to read")

def sync_finals():
    """Mark status.final = done for every page whose final file exists (reconcilers and
    make_final write the file; the manifest flag follows)."""
    sys.path.insert(0, str(ROOT / "scripts")); import pagelib
    m = manifest(); n = 0
    for r in m["pages"]:
        if has("final", r["id"]) and r["status"].get("final") != "done":
            r["status"]["final"] = "done"; n += 1
        for rd in ("A", "B"):
            if has(f"reads/{rd}", r["id"]) and r["status"].get(f"read{rd}") != "done":
                r["status"][f"read{rd}"] = "done"
    pagelib.write_manifest(ROOT / "manifest.json", m)
    return n, sum(r["status"]["final"] == "done" for r in m["pages"])


def cmd_status(a):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    n, total = sync_finals()
    print(f"finals: {total} ({n} newly marked)")
    rows = []
    for r in manifest()["pages"]:
        pid = r["id"]
        if has("final", pid): continue
        ra, rb = has("reads/A", pid), has("reads/B", pid)
        if not (ra and rb):
            if ra or rb: rows.append((pid, "WAITING", "A" if ra else "B", ""))
            continue
        for rd in ("A", "B"):
            subprocess.run([PY, ROOT / "scripts/normalize_spacing.py", ROOT / "transcription/reads" / rd / f"{pid}.json"],
                           capture_output=True, check=True)
        subprocess.run([PY, ROOT / "scripts/auto_resolve.py", pid], capture_output=True, check=True, cwd=ROOT)
        out = subprocess.run([PY, ROOT / "scripts/diff_reads.py", pid], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        stats = dict(kv.split("=") for kv in out.split())
        clean = stats.get("agreement") == "100.0%" and all(stats.get(k) == "0" for k in ("unmatched", "structural", "note_diffs"))
        if clean:
            rows.append((pid, "AUTO", out, ""))
        else:
            rows.append((pid, "RECONCILE", out, str(render("reconcile", pid, context=a.context, model="fable"))))
    for row in rows: print("\t".join(row))
    if not rows: print("no pages pending")

ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
n = sub.add_parser("next"); n.add_argument("--size", type=int, default=8); n.add_argument("--context", type=int, default=3); n.add_argument("--model", default="opus")
s = sub.add_parser("status"); s.add_argument("--context", type=int, default=3)
a = ap.parse_args(); {"next": cmd_next, "status": cmd_status}[a.cmd](a)

#!/usr/bin/env python3
"""Local server for the arbitration page (tools/arbitrate/). Stdlib only.

    uv run python scripts/arbitrate_server.py [--port 8765] [--root transcription/arbitration]

then open http://127.0.0.1:8765/ . Binds to 127.0.0.1 only (use the 127.0.0.1 address, not
"localhost", which may resolve to ::1 and reach a different server on the same port).

Keys on the page
    1  A                 the line is as reading A has it
    2  B                 the line is as reading B has it
    e  edit              type the line (saved as "neither"; empty text removes the line)
    3  either            "it is one of these two": the translator chooses from context
    4  unknown           "neither reading is confirmed": kept as A, escalated
    u  undo              re-open the last decided item and clear its decision
    ←  →                 previous / next item

Routes
    /                    tools/arbitrate/index.html (and its other static files)
    /crops/...           <root>/crops/...
    /strips/...          pages/strips/...
    /read/...            pages/read/...
    GET  /api/pages      [{page, items, decided, done}] for every <root>/queue/*.json, in
                         manifest order
    GET  /api/queue/<id> the queue with each item's saved decision merged in as `decision`
                         (with `by`: carson | auto | translator; absent in the file = carson)
    POST /api/decide     {"page", "item", "choice": A|B|neither|either|unknown|clear, "text"}
                         upserts <root>/decisions/<id>.json (atomic) and returns progress;
                         "clear" removes the item's decision (used by undo). A decision
                         made here is stored with "by": "carson", replacing an "auto" or
                         "translator" one. When the
                         decision finishes the page, `next_page` names the next page (manifest
                         order, wrapping) that still has undecided items; `all_done` is true
                         when there is none. The page moves there by itself.

An older decisions file may hold "both" (read as either) or "skip" (read as undecided);
such entries are left in the file as written until the item is decided again.
"""
from __future__ import annotations

import argparse
import datetime
import json
import mimetypes
import os
import pathlib
import re
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote, urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATIC = ROOT / "tools" / "arbitrate"
PAGES = ROOT / "pages"
CHOICES = {"A", "B", "neither", "either", "unknown"}
LEGACY = {"both": "either", "skip": None}      # older decisions files
PAGE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


class Store:
    def __init__(self, root, manifest=None):
        self.root = root
        self.manifest = pathlib.Path(manifest) if manifest else ROOT / "manifest.json"

    def order(self, pids):
        """Page ids in manifest order; ids the manifest lacks go last, by name."""
        m = read_json(self.manifest, {})
        rank = {r.get("id"): n for n, r in enumerate(m.get("pages") or []) if isinstance(r, dict)} \
            if isinstance(m, dict) else {}
        return sorted(pids, key=lambda p: (rank.get(p, len(rank)), p))

    def queue(self, pid):
        return read_json(self.root / "queue" / f"{pid}.json", None)

    def decisions(self, pid):
        d = read_json(self.root / "decisions" / f"{pid}.json", {})
        raw = d.get("decisions", {}) if isinstance(d, dict) else {}
        out = {}
        for item, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            choice = LEGACY.get(entry.get("choice"), entry.get("choice"))
            if choice in CHOICES:
                out[item] = dict(entry, choice=choice, by=entry.get("by") or "carson")
        return out

    def progress(self, pid, queue=None, decisions=None):
        queue = queue or self.queue(pid) or {"items": []}
        decisions = self.decisions(pid) if decisions is None else decisions
        ids = [i["id"] for i in queue.get("items", [])]
        decided = sum(1 for i in ids if (decisions.get(i) or {}).get("choice") in CHOICES)
        return {"page": pid, "items": len(ids), "decided": decided, "done": decided == len(ids)}

    def pages(self):
        qdir = self.root / "queue"
        pids = [p.stem for p in qdir.glob("*.json")] if qdir.is_dir() else []
        return [self.progress(p) for p in self.order(pids)]

    def next_page(self, after):
        """The next page (manifest order, wrapping round) with undecided items, or None."""
        pages = self.pages()
        ids = [p["page"] for p in pages]
        start = ids.index(after) + 1 if after in ids else 0
        for p in pages[start:] + pages[:start]:
            if not p["done"] and p["page"] != after:
                return p["page"]
        return None

    def decide(self, pid, item, choice, text):
        queue = self.queue(pid)
        if queue is None:
            raise ValueError(f"no queue for {pid}")
        if item not in {i["id"] for i in queue.get("items", [])}:
            raise ValueError(f"{item} is not an item of {pid}")
        path = self.root / "decisions" / f"{pid}.json"
        data = read_json(path, {})
        if not isinstance(data, dict):
            data = {}
        data.setdefault("page", pid)
        dec = data.setdefault("decisions", {})
        if choice == "clear":
            dec.pop(item, None)
        else:
            if choice not in CHOICES:
                raise ValueError(f"bad choice {choice!r}")
            entry = {"choice": choice, "by": "carson",
                     "at": datetime.datetime.now().isoformat(timespec="seconds")}
            if choice == "neither":
                entry["text"] = text if isinstance(text, str) else ""
            dec[item] = entry
        write_atomic(path, data)
        prog = self.progress(pid, queue)
        # when this decision finished the page, say where to go next
        prog["next_page"] = self.next_page(pid) if prog["done"] else None
        prog["all_done"] = prog["done"] and prog["next_page"] is None
        return prog


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):          # quiet: only errors
            if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
                super().log_message(fmt, *args)

        def send_json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, base, rel):
            base = base.resolve()
            path = (base / rel).resolve()
            if base not in path.parents or not path.is_file():
                return self.send_json({"error": "not found"}, 404)
            data = path.read_bytes()
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                return self.send_file(STATIC, "index.html")
            if path == "/api/pages":
                return self.send_json(store.pages())
            m = re.match(r"^/api/queue/([^/]+)$", path)
            if m:
                pid = m.group(1)
                queue = store.queue(pid) if PAGE_RE.match(pid) else None
                if queue is None:
                    return self.send_json({"error": f"no queue for {pid}"}, 404)
                dec = store.decisions(pid)
                for it in queue.get("items", []):
                    if it["id"] in dec:
                        it["decision"] = dec[it["id"]]
                queue["progress"] = store.progress(pid, queue, dec)
                return self.send_json(queue)
            for prefix, base in (("/crops/", store.root / "crops"), ("/strips/", PAGES / "strips"),
                                 ("/read/", PAGES / "read")):
                if path.startswith(prefix):
                    return self.send_file(base, path[len(prefix):])
            return self.send_file(STATIC, path.lstrip("/"))

        def do_POST(self):
            if urlparse(self.path).path != "/api/decide":
                return self.send_json({"error": "not found"}, 404)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                pid = str(body.get("page", ""))
                if not PAGE_RE.match(pid):
                    raise ValueError("bad page id")
                prog = store.decide(pid, str(body.get("item", "")), body.get("choice"),
                                    body.get("text"))
            except (ValueError, json.JSONDecodeError) as exc:
                return self.send_json({"error": str(exc)}, 400)
            return self.send_json(prog)

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve the arbitration page on 127.0.0.1.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default="transcription/arbitration")
    x = ap.parse_args(argv)
    root = pathlib.Path(x.root)
    if not root.is_absolute():
        root = (pathlib.Path.cwd() / root) if (pathlib.Path.cwd() / root).exists() else ROOT / root
    server = HTTPServer(("127.0.0.1", x.port), make_handler(Store(root.resolve())))
    print(f"arbitration: http://127.0.0.1:{x.port}/  (root {root})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

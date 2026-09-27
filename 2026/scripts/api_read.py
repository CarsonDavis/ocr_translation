#!/usr/bin/env python3
"""Single-call page read through the Claude API (no agent loop).

    uv run --python cpython@3.13 --with anthropic --with jsonschema python scripts/api_read.py p064 \
        --reader A --model opus [--out-dir transcription/reads/A] [--context 2] [--effort high]
    ... --dry-run          # build the request, print its shape and a token estimate, send nothing

One request per read: system = reader instructions + conventions + schema (cached prefix,
1h TTL, shared by every read of every page); user = manifest record, the preceding finals,
the whole-page image, every strip. The reply is the page JSON. If the validator rejects it,
one repair turn sends the validator output back (at most --repairs times, default 2).
Usage and cost are appended to <out-dir>/../usage.jsonl.

Requires ANTHROPIC_API_KEY (falls back to the `export KEY=...` lines of ../.env).
"""
from __future__ import annotations

import argparse, base64, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

MODELS = {"opus": "claude-opus-5", "sonnet": "claude-sonnet-5", "fable": "claude-fable-5-1",
          "haiku": "claude-haiku-4-5"}
# $/MTok: input, cache write (5m), cache read, output — list prices, for the running estimate only
PRICES = {"claude-opus-5": (5, 6.25, 0.5, 25), "claude-sonnet-5": (2, 2.5, 0.2, 10),
          "claude-fable-5-1": (10, 12.5, 0.25, 50), "claude-haiku-4-5": (1, 1.25, 0.1, 5)}

INSTRUCTIONS = """You are transcribing ONE page of a 1572 French printed book for a diplomatic edition:
Jean de Coras, *Arrest memorable du Parlement de Tholose* (Paris, Galliot du Pré, 1572), the
account of the Martin Guerre case with Coras's numbered annotations. Accuracy matters far more
than speed. You are one of two independent readers; your output is diffed against the other
reader's line by line and every disagreement is settled by a separate reconciler, so do not
skip anything and do not tidy anything.

The conventions document below defines every rule of the output; the JSON schema after it
defines the exact shape. Read both before the images.

How to work:
1. Use the whole-page image only for the layout: how many paragraphs, headings, markers,
   margin notes, whether there is a foot block, a signature, a catchword.
2. Transcribe from the strips, which carry the scan's native resolution. Body strips overlap
   by a few lines: do not transcribe an overlapped line twice. Margin strips hold the outer
   margin column; foot.jpg the bottom of the page.
3. Every printed line becomes one string, in order, per the conventions (long s as ſ, u/v and
   i/j as printed, tildes kept, markers as {x}). Attend to the three error classes careful
   readers get wrong on this print: (a) ſſ vs ſs — the print often sets long s + round s
   inside a word (profeſsion, auſsi); (b) sentence punctuation — a comma has a tail below the
   baseline, a period is a round dot on the baseline, a colon has two dots; decide from the
   shape, never from the sense; (c) wrong sorts — this print has a wrong letter roughly once
   a page (raporrera, cſgalle, qni, viute, Cuerre); your eye reads the expected word, the
   print did not print it; transcribe the misprint as printed with an uncertain[] note "sic".
4. Cross-check: every {x} in the body has a note with key x and every note has a marker; the
   line count of each paragraph matches the whole-page image.
5. Fill uncertain[] for every doubtful reading, every [?], every missing marker or note. Where
   a glyph is genuinely ambiguous at this resolution, give your best reading and record it;
   never guess silently, never modernize, never expand abbreviations, never fix misprints,
   never merge or split printed lines.
6. The crops keep a narrow strip of the facing page along the gutter edge (inner edge: right
   on versos, left on rectos). Ignore it completely.
7. Use the preceding pages' finals only for continuity (continues_prev, a word broken across
   the page boundary, the running marker alphabet). Never copy from them.

Reply with the page JSON object only: no prose, no code fence. Set "id", "reader" and
"model" to the values given in the request."""


def load_env_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = ROOT.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")


def image_block(path: pathlib.Path):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(path.read_bytes()).decode()}}


def image_tokens(path: pathlib.Path) -> int:
    """Rough token estimate: (w*h)/750 after the API's downscale to ~1.15MP."""
    try:
        from PIL import Image
        w, h = Image.open(path).size
    except Exception:
        return 1500
    px = w * h
    if px > 1_150_000:
        px = 1_150_000
    return px // 750


def context_pages(page_id: str, n: int, manifest: dict) -> list[tuple[str, dict]]:
    ids = [r["id"] for r in manifest["pages"]]
    i = ids.index(page_id)
    out = []
    for pid in ids[max(0, i - n):i]:
        p = ROOT / "transcription/final" / f"{pid}.json"
        if p.exists():
            page = json.loads(p.read_text())
            slim = {k: page.get(k) for k in ("id", "running_head", "folio", "blocks", "margin_notes", "foot_notes")}
            out.append((pid, slim))
    return out


def strip_paths(page_id: str) -> list[pathlib.Path]:
    d = ROOT / "pages/strips" / page_id
    bodies = sorted(d.glob("body-*.jpg"), key=lambda p: int(p.stem.split("-")[1]))
    margins = sorted(d.glob("margin-*.jpg"), key=lambda p: int(p.stem.split("-")[1]))
    foot = [d / "foot.jpg"] if (d / "foot.jpg").exists() else []
    return bodies + margins + foot


def build_request(page_id: str, reader: str, model_key: str, n_context: int):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    rec = next(r for r in manifest["pages"] if r["id"] == page_id)
    slim = {k: rec[k] for k in ("id", "page", "image", "side", "folio", "source")}
    system = [
        {"type": "text", "text": INSTRUCTIONS},
        {"type": "text", "text": "# docs/conventions.md\n\n" + (ROOT / "docs/conventions.md").read_text()},
        {"type": "text", "text": "# scripts/page_schema.json\n\n" + (ROOT / "scripts/page_schema.json").read_text(),
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]
    content = [{"type": "text", "text": f"Page id: {page_id}. Reader: {reader}. Model: {model_key}.\n"
                                        f"Manifest record: {json.dumps(slim, ensure_ascii=False)}"}]
    ctx = context_pages(page_id, n_context, manifest)
    if ctx:
        content.append({"type": "text", "text": "Preceding pages (final transcriptions, for continuity only):\n" +
                        "\n".join(f"## {pid}\n{json.dumps(page, ensure_ascii=False)}" for pid, page in ctx)})
    else:
        content.append({"type": "text", "text": "Preceding pages: none (no earlier page is finished yet)."})
    images = []
    read_img = ROOT / "pages/read" / f"{page_id}.jpg"
    content.append({"type": "text", "text": "Whole page, reading size (layout only):"})
    content.append(image_block(read_img)); images.append(read_img)
    for p in strip_paths(page_id):
        content.append({"type": "text", "text": f"Strip {p.name} (native resolution):"})
        content.append(image_block(p)); images.append(p)
    content.append({"type": "text", "text": f"Transcribe page {page_id} now. Reply with the JSON object only."})
    return system, [{"role": "user", "content": content}], images


def extract_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        t = t.rsplit("```", 1)[0]
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in reply")
    return json.loads(t[start:end + 1])


def validate(path: pathlib.Path) -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(ROOT / "scripts/validate_page.py"), str(path)],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def cost_of(model: str, u) -> float:
    i, cw, cr, o = PRICES.get(model, (0, 0, 0, 0))
    return (u.input_tokens * i + (u.cache_creation_input_tokens or 0) * cw +
            (u.cache_read_input_tokens or 0) * cr + u.output_tokens * o) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page_id")
    ap.add_argument("--reader", default="A", choices=["A", "B", "spotcheck"])
    ap.add_argument("--model", default="opus", choices=sorted(MODELS))
    ap.add_argument("--out-dir", default=None, help="default transcription/reads/<reader>")
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--repairs", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    model = MODELS[a.model]
    out_dir = ROOT / (a.out_dir or f"transcription/reads/{a.reader}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{a.page_id}.json"
    system, messages, images = build_request(a.page_id, a.reader, a.model, a.context)

    if a.dry_run:
        text_chars = sum(len(b["text"]) for b in system) + sum(len(c["text"]) for c in messages[0]["content"] if c["type"] == "text")
        img_tok = sum(image_tokens(p) for p in images)
        print(f"{a.page_id} {a.reader} {model}: {len(images)} images (~{img_tok} tok), "
              f"~{text_chars // 4} text tok (of which ~{sum(len(b['text']) for b in system) // 4} cached system prefix)")
        return

    load_env_key()
    import anthropic
    client = anthropic.Anthropic(max_retries=3, timeout=900)
    kwargs = dict(model=model, max_tokens=32000, system=system, output_config={"effort": a.effort})
    t0 = time.time(); usages = []; repairs = 0; page = None; status = "failed"; problems = ""
    while True:
        with client.messages.stream(messages=messages, **kwargs) as stream:
            resp = stream.get_final_message()
        usages.append(resp.usage)
        if resp.stop_reason == "refusal":
            problems = f"refusal: {resp.stop_details}"; break
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            page = extract_json(text)
            page.update(id=a.page_id, reader=a.reader, model=a.model)
            out_path.write_text(json.dumps(page, indent=1, ensure_ascii=False) + "\n")
            subprocess.run([sys.executable, str(ROOT / "scripts/normalize_spacing.py"), str(out_path)],
                           capture_output=True, text=True)
            rc, problems = validate(out_path)
        except (ValueError, json.JSONDecodeError) as e:
            rc, problems = 1, f"reply was not valid JSON: {e}"
        if rc == 0:
            status = "ok"; break
        if repairs >= a.repairs:
            status = "invalid"; break
        repairs += 1
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"The validator rejected that transcription:\n\n{problems}\n\n"
                                        "Fix every reported problem by looking at the images again and reply with "
                                        "the complete corrected JSON object only."}]
    secs = round(time.time() - t0, 1)
    tot = {k: sum(getattr(u, k) or 0 for u in usages) for k in
           ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    cost = sum(cost_of(model, u) for u in usages)
    rec = dict(page=a.page_id, reader=a.reader, model=model, effort=a.effort, status=status, repairs=repairs,
               seconds=secs, cost_usd=round(cost, 4), **tot, images=len(images))
    with (out_dir.parent / "usage.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    if status != "ok":
        print(problems, file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reconcile two reads through one Claude API call, sending only the disputed material.

    uv run --python cpython@3.13 --with anthropic --with jsonschema python scripts/api_reconcile.py p064 \
        [--a transcription/reads/A/p064.json] [--b transcription/reads/B/p064.json] \
        [--diff transcription/diff/p064.md] [--out transcription/final/p064.json] [--model fable] [--dry-run]

The request carries: the reconciler instructions + conventions (cached prefix), the diff
report, both reads, and only the strips that hold disputed lines (body strips chosen by
line position with one neighbour each side; all margin strips if any note differs; foot.jpg
if any foot note differs). The reply is the final page JSON with a "decisions" array.
Usage and cost go to transcription/usage.jsonl. Run scripts/diff_reads.py first.
"""
from __future__ import annotations

import argparse, json, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pagelib  # noqa: E402
from api_read import MODELS, PRICES, cost_of, image_block, image_tokens, load_env_key, extract_json, validate  # noqa: E402

INSTRUCTIONS = """Two independent readers transcribed one page of Coras, *Arrest memorable* (1572). Their
outputs differ in places. Decide every difference by looking at the page images, and write the
final master transcription. Do not defer to either reader by default. Accuracy over speed, but
spend your attention only on the disputed lines, the structural differences, and the lines
named in either reader's uncertain[]: lines the readers agree on are not re-read here (a
separate spot-check samples for shared mistakes).

The conventions document below is the rule book. The diff report says exactly where the reads
differ and how the body lines are numbered (column order: headings and paragraph lines in block
order). Only the strips that hold disputed material are attached; body strips overlap by a few
lines, so a line may appear in two.

Procedure:
1. For every differing line pair, find the line in the attached strips and read it yourself.
   Decide A, B, or neither (write the correct text).
2. For structural differences (a missing line, a different paragraph split, a missing note,
   different keys) decide the same way, from the whole-page image if attached.
3. If a reading is genuinely undecidable from the images, keep the best reading (or [?]) and
   add an uncertain[] entry with "escalate": true and a precise pointer (strip file, line), so
   a human can look.
4. Output the complete final page JSON: same schema, "reader": "final", "model" as given, every
   agreed line copied verbatim from the reads, plus a "decisions" array with one entry per
   difference: {"where": "blocks[1].lines[3]", "A": "…", "B": "…", "chose": "A"|"B"|"neither",
   "text": "…", "reason": "…"}. Keep both readers' uncertain[] entries that still apply.

Never modernize, expand, or correct the print. Reply with the JSON object only: no prose, no
code fence."""


def disputed_positions(diff_text: str, n_lines: int):
    """Body line indexes (A side) named in the diff, plus whether notes / foot notes differ."""
    body = set()
    for m in re.finditer(r"A\[(\d+)\]", diff_text):
        body.add(int(m.group(1)))
    for m in re.finditer(r"B\[(\d+)\]", diff_text):
        body.add(int(m.group(1)))
    notes = "## Note differences\n\nnone" not in diff_text
    struct = "## Structural differences\n\nnone" not in diff_text
    return sorted(i for i in body if 0 <= i < max(n_lines, 1)), notes, struct


def choose_strips(page_id: str, a: dict, diff_text: str):
    d = ROOT / "pages/strips" / page_id
    bodies = sorted(d.glob("body-*.jpg"), key=lambda p: int(p.stem.split("-")[1]))
    margins = sorted(d.glob("margin-*.jpg"), key=lambda p: int(p.stem.split("-")[1]))
    n_lines = len(pagelib.column_texts(a))
    idxs, notes_differ, struct = disputed_positions(diff_text, n_lines)
    chosen = []
    if struct:
        chosen.append(ROOT / "pages/read" / f"{page_id}.jpg")
    if bodies and (idxs or struct):
        per = max(n_lines, 1) / len(bodies)
        want = set()
        for i in idxs:
            s = int(i / per)
            want.update({s - 1, s, s + 1})
        if struct and not idxs:
            want.update(range(len(bodies)))
        chosen += [bodies[s] for s in sorted(want) if 0 <= s < len(bodies)]
    if notes_differ or struct:
        chosen += margins
        foot = d / "foot.jpg"
        if foot.exists() and ("foot" in diff_text.lower() or struct):
            chosen.append(foot)
    return chosen


def build_request(page_id: str, a_path: pathlib.Path, b_path: pathlib.Path, diff_path: pathlib.Path, model_key: str):
    a = json.loads(a_path.read_text()); b = json.loads(b_path.read_text())
    diff_text = diff_path.read_text()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    rec = next(r for r in manifest["pages"] if r["id"] == page_id)
    slim = {k: rec[k] for k in ("id", "page", "image", "side", "folio", "source")}
    system = [
        {"type": "text", "text": INSTRUCTIONS},
        {"type": "text", "text": "# docs/conventions.md\n\n" + (ROOT / "docs/conventions.md").read_text(),
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]
    content = [
        {"type": "text", "text": f"Page id: {page_id}. Model: {model_key}. Manifest record: {json.dumps(slim, ensure_ascii=False)}"},
        {"type": "text", "text": "# Diff report\n\n" + diff_text},
        {"type": "text", "text": "# Read A\n\n" + json.dumps(a, ensure_ascii=False)},
        {"type": "text", "text": "# Read B\n\n" + json.dumps(b, ensure_ascii=False)},
    ]
    images = choose_strips(page_id, a, diff_text)
    for p in images:
        label = "Whole page, reading size" if p.parent.name == "read" else f"Strip {p.name} (native resolution)"
        content.append({"type": "text", "text": f"{label}:"})
        content.append(image_block(p))
    content.append({"type": "text", "text": f"Write the final transcription of {page_id} now. Reply with the JSON object only."})
    return system, [{"role": "user", "content": content}], images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page_id")
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--diff"); ap.add_argument("--out")
    ap.add_argument("--model", default="fable", choices=sorted(MODELS))
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--repairs", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pid = a.page_id
    a_path = ROOT / (a.a or f"transcription/reads/A/{pid}.json")
    b_path = ROOT / (a.b or f"transcription/reads/B/{pid}.json")
    diff_path = ROOT / (a.diff or f"transcription/diff/{pid}.md")
    out_path = ROOT / (a.out or f"transcription/final/{pid}.json")
    model = MODELS[a.model]
    system, messages, images = build_request(pid, a_path, b_path, diff_path, a.model)
    if a.dry_run:
        text_tok = (sum(len(b["text"]) for b in system) + sum(len(c["text"]) for c in messages[0]["content"] if c["type"] == "text")) // 4
        print(f"{pid} {model}: {len(images)} images ({', '.join(p.name for p in images)}; ~{sum(image_tokens(p) for p in images)} tok), ~{text_tok} text tok")
        return
    load_env_key()
    import anthropic
    client = anthropic.Anthropic(max_retries=3, timeout=900)
    kwargs = dict(model=model, max_tokens=32000, system=system, output_config={"effort": a.effort})
    t0 = time.time(); usages = []; repairs = 0; status = "failed"; problems = ""
    while True:
        with client.messages.stream(messages=messages, **kwargs) as stream:
            resp = stream.get_final_message()
        usages.append(resp.usage)
        if resp.stop_reason == "refusal":
            problems = f"refusal: {resp.stop_details}"; break
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            page = extract_json(text)
            page.update(id=pid, reader="final", model=a.model)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(page, indent=1, ensure_ascii=False) + "\n")
            subprocess.run([sys.executable, str(ROOT / "scripts/normalize_spacing.py"), str(out_path)], capture_output=True, text=True)
            rc, problems = validate(out_path)
        except (ValueError, json.JSONDecodeError) as e:
            rc, problems = 1, f"reply was not valid JSON: {e}"
        if rc == 0:
            status = "ok"; break
        if repairs >= a.repairs:
            status = "invalid"; break
        repairs += 1
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": f"The validator rejected that file:\n\n{problems}\n\nFix every reported problem and reply with the complete corrected JSON object only."}]
    tot = {k: sum(getattr(u, k) or 0 for u in usages) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    rec = dict(page=pid, reader="reconcile", model=model, effort=a.effort, status=status, repairs=repairs,
               seconds=round(time.time() - t0, 1), cost_usd=round(sum(cost_of(model, u) for u in usages), 4), **tot, images=len(images))
    with (ROOT / "transcription/usage.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    if status != "ok":
        print(problems, file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()

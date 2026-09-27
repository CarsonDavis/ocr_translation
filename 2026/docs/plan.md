# Coras 1572 Retranslation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A verified diplomatic transcription and a reviewed English translation of Coras's *Arrest memorable* (1572), served as a static side-by-side site, built under `2026/`.

**Architecture:** Small stand-alone Python scripts (run with `uv run --with <deps>`) do every deterministic step (manifest, download, crop, diff, stitch, split, site build). Vision-capable agents do every reading, translating and reviewing step, always writing to their own output file, never to a master. `manifest.json` is the single source of truth for page identity and stage status. Each stage has a validation script or agent check that must pass before the next stage starts.

**Tech Stack:** Python 3.12 via `uv run --with pillow,numpy,requests,jsonschema`; IIIF Image API 2.0 at Cambridge; Opus/Fable subagents through the Agent tool; plain HTML/CSS/JS for the site (no build step).

**Conventions used in this plan:**
- All paths are relative to `/Users/cdavis/github/translator/2026/` unless they start with `/`.
- `pNNN` is the zero-padded **true page number** (`p001`…`p160`); the title page is `p000-title`, the Argument is `p000-argument`. Sort order is title, argument, 1…160.
- Run every script from `/Users/cdavis/github/translator/2026/`.
- Nothing is committed without Carson's explicit go-ahead. Tasks below say "ready to commit"; they do not commit.

---

## File structure

| Path | Responsibility |
|---|---|
| `manifest.json` | One record per page: identity, image source, status per stage. Written by `scripts/build_manifest.py`, status fields updated by later scripts. |
| `scripts/build_manifest.py` | Encodes the image→page mapping rules and writes the manifest. |
| `scripts/acquire.py` | Downloads IIIF tiles, stitches to `raw/`. Fetches page 41 from Gallica. |
| `scripts/folio_sheet.py` | Contact sheet of header strips for agent verification of folio numbers. |
| `scripts/measure_pages.py` | Measures paper rectangle on every raw scan; prints spread. Decides fixed trim vs per-page. |
| `scripts/crop.py` | Writes `pages/full/`, `pages/read/`, and reading strips `pages/strips/pNNN/`. |
| `scripts/page_schema.json` | JSON schema for a transcribed page. |
| `scripts/validate_page.py` | Validates one or all read/final JSON files against the schema and conventions. |
| `scripts/diff_reads.py` | Line-level diff of read A vs read B, writes `transcription/diff/pNNN.md`, updates manifest. |
| `scripts/stitch_text.py` | Final pages → `text/sections.json` (ordered logical sections, reflowed, page markers). |
| `scripts/check_markers.py` | Verifies a translation keeps every page marker and letter marker of its source section. |
| `scripts/split_pages.py` | Section translations → `translation/pages/pNNN.json` and `site/data/pages/pNNN.json`. |
| `scripts/prompts/*.md` | Every agent prompt, versioned. |
| `docs/conventions.md` | The transcription rules the readers follow. |
| `docs/case-file.md` | Narrative and factual reference for translators and reviewers. |
| `docs/pipeline-log.md` | Running log: what ran, agreement rates, escalations, decisions. |
| `site/index.html`, `site/app.js`, `site/style.css` | The viewer. |

---

### Task 1: Manifest

**Goal:** `manifest.json` with 162 correct page records.

**Files:**
- Create: `scripts/build_manifest.py`
- Create: `manifest.json`
- Create: `scripts/tests/test_manifest.py`

**Acceptance Criteria:**
- [ ] 162 records: `p000-title` (image 7), `p000-argument` (image 22), `p001`…`p160`.
- [ ] Mapping: pages 1–40 ← images 23–62; page 41 ← `gallica`; pages 42–159 ← images 63–180 with odd image *i* → page *i*−20 and even *i* → page *i*−22; page 160 ← image 182. Image 181 is unused.
- [ ] Printed folio recorded: page 44 → `"24"`, 45 → `"44"`, 48 → `"58"`, otherwise the page number as a string; title and argument have `null`.
- [ ] `side` is `"recto"` for odd pages, `"verso"` for even; title `"recto"`, argument `"verso"` (verify on image 22 during Task 2 and fix if wrong).
- [ ] No image number appears twice.

**Verify:** `uv run --with pytest pytest scripts/tests/test_manifest.py -q` → `5 passed`

**Steps:**

- [ ] **Step 1: Write the test**

```python
# scripts/tests/test_manifest.py
import json, subprocess, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]

def load():
    subprocess.run([sys.executable, ROOT/"scripts/build_manifest.py"], check=True, cwd=ROOT)
    return json.loads((ROOT/"manifest.json").read_text())

def by_id(m): return {r["id"]: r for r in m["pages"]}

def test_count():
    assert len(load()["pages"]) == 162

def test_mapping_samples():
    p = by_id(load())
    assert p["p000-title"]["image"] == 7
    assert p["p000-argument"]["image"] == 22
    assert p["p001"]["image"] == 23 and p["p040"]["image"] == 62
    assert p["p041"]["image"] is None and p["p041"]["source"] == "gallica"
    assert p["p042"]["image"] == 64 and p["p043"]["image"] == 63
    assert p["p158"]["image"] == 180 and p["p159"]["image"] == 179
    assert p["p160"]["image"] == 182

def test_folios():
    p = by_id(load())
    assert p["p044"]["folio"] == "24" and p["p045"]["folio"] == "44" and p["p048"]["folio"] == "58"
    assert p["p046"]["folio"] == "46" and p["p000-title"]["folio"] is None

def test_sides():
    p = by_id(load())
    assert p["p001"]["side"] == "recto" and p["p002"]["side"] == "verso"

def test_no_duplicate_images():
    imgs = [r["image"] for r in load()["pages"] if r["image"] is not None]
    assert len(imgs) == len(set(imgs))
```

- [ ] **Step 2: Run it, expect failure** (`build_manifest.py` missing).

- [ ] **Step 3: Implement**

```python
# scripts/build_manifest.py
"""Write manifest.json: the single source of truth for page identity and stage status."""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
CUDL_ITEM = "PR-MONTAIGNE-00001-00007-00022"
IIIF = f"https://images.lib.cam.ac.uk/iiif/{CUDL_ITEM}-000-{{n:05d}}.jp2"
MISPRINTED_FOLIOS = {44: "24", 45: "44", 48: "58"}
STAGES = ["acquired", "cropped", "readA", "readB", "diffed", "final", "spotchecked", "translated", "reviewed"]

def image_for_page(page: int):
    if 1 <= page <= 40: return page + 22
    if page == 41: return None
    if 42 <= page <= 159:
        # openings shot recto-first: odd image i -> page i-20, even image i -> page i-22
        return page + 20 if (page + 20) % 2 == 1 else page + 22
    if page == 160: return 182
    raise ValueError(page)

def record(pid, page, image, side, folio, source="cudl"):
    return {"id": pid, "page": page, "image": image, "side": side, "folio": folio,
            "source": source,
            "iiif": IIIF.format(n=image) if image else None,
            "status": {s: "pending" for s in STAGES}}

pages = [record("p000-title", None, 7, "recto", None),
         record("p000-argument", None, 22, "verso", None)]
for pg in range(1, 161):
    img = image_for_page(pg)
    pages.append(record(f"p{pg:03d}", pg, img, "recto" if pg % 2 else "verso",
                        MISPRINTED_FOLIOS.get(pg, str(pg)),
                        source="cudl" if img else "gallica"))
assert all(image_for_page(p) is None or (image_for_page(p) % 2 == 1) == (p % 2 == 1)
           for p in range(42, 160)), "odd images must be rectos"
manifest = {"item": CUDL_ITEM, "edition": "Paris: Galliot du Pré, 1572",
            "native_size": [2941, 4711], "pages": pages}
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
print(f"wrote {len(pages)} pages")
```

Note the recto check: image 63 is page 43 (odd, recto). Odd images are rectos throughout 63–180, so for pages 42–159 the formula reduces to: page odd → image = page+20; page even → image = page+22.

- [ ] **Step 4: Run the tests, expect `5 passed`.**
- [ ] **Step 5: Ready to commit** (`scripts/build_manifest.py`, `scripts/tests/test_manifest.py`, `manifest.json`).

---

### Task 2: Acquire full-resolution scans

**Goal:** `raw/imgNNN.jpg` at 2941×4711 for every CUDL image in the manifest, plus `raw/gallica-p041.jpg`, and an agent-verified folio check.

**Files:**
- Create: `scripts/acquire.py`, `scripts/folio_sheet.py`
- Create: `raw/` (gitignored)
- Modify: `manifest.json` (status.acquired)
- Append: `docs/pipeline-log.md`

**Acceptance Criteria:**
- [ ] 161 CUDL images stitched; each 2941×4711 (allow ±2px), JPEG quality 92.
- [ ] Tile seams are pixel-exact: the script requests tiles `x∈{0,1941}` (w=2000, last w=1000) × `y∈{0,2000,4000}` (h=2000, last h=711) and pastes at the same offsets, so no resampling occurs.
- [ ] Page 41 fetched from Gallica (`https://gallica.bnf.fr/ark:/12148/bpt6k52469j`); the right view is found by an agent reading the header strips of candidate views around view 60 (Gallica's view index does not equal page number). Saved as `raw/gallica-p041.jpg` at the largest size Gallica serves (`/f{view}.highres`).
- [ ] `folio_sheet.py` writes `docs/checks/folio-sheet-{k}.jpg` (20 header strips per sheet, each labelled with the manifest id). A Fable agent reads every sheet and confirms every folio matches `manifest.folio`; mismatches are listed in `docs/pipeline-log.md`. Zero mismatches required.
- [ ] `status.acquired = "done"` for every page.

**Verify:** `uv run --with pillow python scripts/acquire.py --check` → `161 raw images ok, 0 missing, 0 wrong size`

**Steps:**

- [ ] **Step 1: Write `scripts/acquire.py`**

```python
# scripts/acquire.py
"""Download native-resolution page images from Cambridge IIIF as region tiles and stitch them."""
import argparse, io, json, pathlib, sys, time, requests
from PIL import Image
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"; RAW.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) coras-transcription/1.0"}
W, H = 2941, 4711
TILES = [(x, y, min(2000, W - x), min(2000, H - y)) for y in (0, 2000, 4000) for x in (0, 1941)]

def fetch(url, tries=4):
    for i in range(tries):
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code == 200: return r.content
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url} -> {r.status_code}")

def stitch(iiif_base, out):
    canvas = Image.new("RGB", (W, H))
    for x, y, w, h in TILES:
        tile = Image.open(io.BytesIO(fetch(f"{iiif_base}/{x},{y},{w},{h}/full/0/default.jpg")))
        assert tile.size == (w, h), (tile.size, (w, h))
        canvas.paste(tile, (x, y))
    canvas.save(out, "JPEG", quality=92)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--only")
    a = ap.parse_args()
    m = json.loads((ROOT / "manifest.json").read_text())
    missing = wrong = ok = 0
    for rec in m["pages"]:
        if rec["image"] is None: continue
        if a.only and rec["id"] != a.only: continue
        out = RAW / f"img{rec['image']:03d}.jpg"
        if not out.exists() and not a.check:
            print("fetch", rec["id"], rec["image"], flush=True); stitch(rec["iiif"], out)
        if not out.exists(): missing += 1; continue
        if abs(Image.open(out).size[0] - W) > 2 or abs(Image.open(out).size[1] - H) > 2: wrong += 1; continue
        ok += 1; rec["status"]["acquired"] = "done"
    if not a.check: (ROOT / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
    print(f"{ok} raw images ok, {missing} missing, {wrong} wrong size")
    sys.exit(1 if missing or wrong else 0)

if __name__ == "__main__": main()
```

- [ ] **Step 2: Run on one page first**: `uv run --with pillow,requests python scripts/acquire.py --only p043`, open `raw/img063.jpg`, confirm 2941×4711 and no visible seams at y=2000/4000 or x=1941 (crop a 400×400 window around (1941,2000) and view it).
- [ ] **Step 3: Run the full download** in the background (`run_in_background`, ~161 × 6 requests; expect 10–20 minutes). Then `--check`.
- [ ] **Step 4: Gallica page 41.** Dispatch an agent: fetch `https://gallica.bnf.fr/ark:/12148/bpt6k52469j/f{v}.highres` for v in 58..66 into the scratchpad, crop the top 8% of each, view them, identify the view whose running head reads `PARLEMENT DE THOLOSE. 41` and whose first line continues p.40's last line (`… de ladite de Rols, qui nevou-`). Save that view as `raw/gallica-p041.jpg`, record its view number and pixel size in `docs/pipeline-log.md`, set `status.acquired = "done"` for `p041`.
- [ ] **Step 5: Write `scripts/folio_sheet.py`**: for every page in manifest order, load the raw image (or the Gallica file), crop the band from 3% to 9% of height across the full width, resize to 1400px wide, stack 20 per sheet with a red label `id / expected folio`, write `docs/checks/folio-sheet-{k:02d}.jpg`. Dispatch a Fable agent to read all 9 sheets and report `id: seen folio` for every page and a list of mismatches. Any mismatch is investigated and the manifest mapping fixed before continuing.
- [ ] **Step 6: Log** counts, timings, the Gallica view number, and the folio-check result in `docs/pipeline-log.md`. Ready to commit scripts and log (not `raw/`).

---

### Task 3: Crop decision, crops, and reading strips

**Goal:** Cropped page images for the site and near-native-resolution strips for the readers.

**Why strips:** the model sees at most about 1.15 megapixels per image (roughly 1568px on the long edge). A whole 2941×4711 page is downscaled to about 979×1568, worse than the old scans. To read at native resolution the page is cut into strips small enough not to be downscaled.

**Files:**
- Create: `scripts/measure_pages.py`, `scripts/crop.py`
- Create: `pages/full/`, `pages/strips/` (gitignored), `pages/read/` (committed)
- Create: `docs/checks/crop-sheet-*.jpg`
- Modify: `manifest.json` (status.cropped, crop box per page)

**Acceptance Criteria:**
- [ ] `measure_pages.py` prints, for all raw images, the detected paper bounding box (threshold on luminance > 60 against the black backdrop, largest connected region, ignoring the facing-page sliver on the left by requiring width > 40% of image) and the min/median/max of each edge. The decision (fixed trim or per-page) is written to `docs/pipeline-log.md` with those numbers.
- [ ] `pages/full/pNNN.jpg`: the paper region plus a 1% margin, copyright band removed (it lies below the paper; verify it is outside the detected box), no deskew unless measured tilt > 0.5° (log the tilt; deskew is via `Image.rotate(angle, resample=BICUBIC, expand=False)`).
- [ ] `pages/read/pNNN.jpg`: same crop at 1600px wide, quality 85, expected 250–500KB.
- [ ] `pages/strips/pNNN/body-K.jpg`: the body text column cut into horizontal strips of at most 1400px height at native scale, 80px overlap between consecutive strips, each ≤ 1.1 megapixels. Body column = the widest dense band of the vertical ink projection (column sums of `pixel < 128`, smoothed with a 25px window, threshold at 15% of max). Margin column = the remaining dense band on the outer side (right on rectos, left on versos).
- [ ] `pages/strips/pNNN/margin-K.jpg`: the margin column (plus 60px inward so it captures the body edge) at native scale, ≤ 1.1 megapixels per strip. Pages with no detected margin band get no margin strips, and the manifest records `"margin": false`.
- [ ] `pages/strips/pNNN/foot.jpg`: the bottom 18% of the page at native scale (catches foot-of-page citation blocks, signatures, catchwords).
- [ ] Contact sheets `docs/checks/crop-sheet-{k}.jpg` (12 read-size pages each, with the detected body/margin bands drawn as coloured rectangles) reviewed by a Fable agent, which lists any page with clipped text, leftover backdrop, or a wrong column split. Every listed page is fixed (manual crop box in `manifest.json` under `"crop_override"`) and re-run. Zero open issues required.
- [ ] Argument page and title page get the same treatment (title page may have no columns; a single body band covering the whole ink area is acceptable).

**Verify:** `uv run --with pillow,numpy python scripts/crop.py --check` → `162 pages: full ok, read ok, strips ok; 0 issues`

**Steps:**

- [ ] **Step 1: `measure_pages.py`.** Load each raw image at 1/4 scale (grayscale), threshold, find the bounding box of the largest bright region wider than 40% of the frame (use `numpy` row/column projections: rows with > 30% bright pixels, columns with > 30% bright pixels within those rows). Print per-image box and the aggregate spread. Run it on all 161 and paste the summary into `docs/pipeline-log.md`.
- [ ] **Step 2: Decide.** If max−min of every edge < 60px at full scale, use one fixed box (median of each edge) for all pages. Otherwise use per-page boxes. Record the decision.
- [ ] **Step 3: `crop.py`.** Implements the crop, the read-size derivative, the projection-based column detection, the strips, the foot crop, and `--check`. Stores the crop box and column bands in `manifest.json` per page (`"crop": {"box": [...], "body": [x0, x1], "margin": [x0, x1] | null}`). `--sheets` writes the contact sheets.
- [ ] **Step 4: Pilot on p001, p004, p008, p043, p159, p000-title.** View the strips for p008 yourself; confirm the margin strips are legible at 1:1 and the body strips are not downscaled (check pixel sizes ≤ 1568 on the long edge).
- [ ] **Step 5: Run all, generate sheets, dispatch the Fable review agent, fix issues, re-run `--check`.**
- [ ] **Step 6: Log and ready to commit** (`scripts/`, `pages/read/`, `docs/checks/`, manifest).

---

### Task 4: Conventions, schema, validator, diff tool, and reader prompts

**Goal:** Everything an agent needs to read a page consistently, and the tools to check and compare its output.

**Files:**
- Create: `docs/conventions.md`
- Create: `scripts/page_schema.json`, `scripts/validate_page.py`, `scripts/diff_reads.py`
- Create: `scripts/prompts/read.md`, `scripts/prompts/reconcile.md`, `scripts/prompts/spotcheck.md`
- Create: `scripts/tests/test_validate.py`, `scripts/tests/test_diff.py`

**Page JSON contract** (this is the master format; everything downstream depends on it):

```json
{
  "id": "p043",
  "reader": "A",
  "model": "opus",
  "running_head": "PARLEMENT DE THOLOSE.",
  "folio": "43",
  "blocks": [
    {"type": "heading", "text": "ANNOTAT. V."},
    {"type": "paragraph", "continues_prev": true, "continues_next": false,
     "lines": ["ſont dignes de louange grande. Mais reprenans nos", "…"]},
    {"type": "paragraph", "spaced_caps": false, "lines": ["…{a}…"]}
  ],
  "margin_notes": [
    {"key": "a", "lines": ["Seneque au", "liu. des be-", "nefices."], "beside_line": "…the body line text it sits next to…"}
  ],
  "foot_notes": [
    {"key": "d", "lines": ["…"]}
  ],
  "signature": "F iij",
  "catchword": null,
  "ornaments": ["decorated initial A", "headpiece"],
  "uncertain": [
    {"where": "blocks[1].lines[3]", "note": "second word could be 'vne' or 'une'; ink faint"}
  ]
}
```

Rules encoded in `docs/conventions.md` (write the full document; the essentials):
- One string per printed line, in order, hyphens as printed. Never join or wrap lines.
- Long s → `ſ`. `u/v`, `i/j` as printed. `æ`, `œ` kept. Tilde vowels `ã ẽ ĩ õ ũ`, tilde consonants with combining U+0303 (`q̃`, `m̃`, `n ̃`). `&` kept. The `ꝑ` (per/par), `ꝓ` (pro), `ꝰ` (-us) signs kept when they appear; if a sign cannot be typed, `[abbr: description]`.
- Letter markers in the body: `{a}` at the exact spot. Marker letters run a–z then start over on the page; `u` and `v`, `i` and `j` are distinct only if the print uses both.
- Spaced capitals kept with spaces; block gets `"spaced_caps": true`.
- Capitalization and punctuation exactly as printed, including the colon-spacing quirks.
- Unreadable character → `[?]`; damaged/lost span → `[...]`; every use goes in `uncertain[]` with a note.
- Page number, running head, signature, catchword transcribed, not normalized.
- The "fancy t" the old notes puzzled over is the Fraktur-like `t` in the italic margin font; transcribe as `t`.

**Acceptance Criteria:**
- [ ] `validate_page.py pNNN.json` checks schema, that every `{x}` marker in the body has a note with key `x` in `margin_notes` or `foot_notes` on that page or an entry in `uncertain[]` explaining its absence, and that every note key appears as a marker (same rule). Exit 1 with a readable list of problems.
- [ ] `diff_reads.py pNNN` aligns read A and read B **line by line** (sequence-align the flattened list of body lines with `difflib.SequenceMatcher` on normalized text, then compare aligned pairs character-exact), does the same for margin/foot notes by key, and writes `transcription/diff/pNNN.md` with: identical-line count, differing pairs shown as `A: …` / `B: …`, unmatched lines, structural differences (block count, headings, keys). It prints `agreement=NN.N%` (identical lines / max lines) and writes it to the manifest under `status.diffed` as `"done"` plus `"agreement": 97.4`.
- [ ] Prompts are complete, self-contained instructions (a fresh agent with no other context can follow them) and reference the exact input paths and the exact output path.
- [ ] Tests: a hand-written valid page passes; a page with an orphan `{c}` fails with a message naming `c`; two reads differing in one character produce agreement `< 100` and a diff file containing both variants.

**Verify:** `uv run --with pytest,jsonschema pytest scripts/tests/test_validate.py scripts/tests/test_diff.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write `docs/conventions.md`** in full (readable by a human; the reader prompt links to it and also inlines it).
- [ ] **Step 2: Write `scripts/page_schema.json`** matching the contract above (`required`: id, reader, running_head, folio, blocks, margin_notes, foot_notes, uncertain; block `type` enum: heading, paragraph, ornament, blank).
- [ ] **Step 3: Tests for validator and diff, then implement both.**
- [ ] **Step 4: `scripts/prompts/read.md`.** Content, in this order: role ("you are transcribing one page of a 1572 French book for a diplomatic edition; accuracy over speed"); the conventions inlined; the input list (`pages/read/pNNN.jpg` for layout, `pages/strips/pNNN/body-*.jpg` in order, `margin-*.jpg`, `foot.jpg`, and `transcription/final/` files for the previous 2–3 pages for context); the procedure (read the whole page image first to count paragraphs, headings, markers, and margin notes; then read each body strip and transcribe line by line; then each margin strip; then the foot; then cross-check that every marker has a note; then fill `uncertain`); the output (write exactly one file, `transcription/reads/{READER}/pNNN.json`, then run `uv run --with jsonschema python scripts/validate_page.py transcription/reads/{READER}/pNNN.json` and fix until it passes); and a final rule: never guess silently, use `[?]` and `uncertain[]`.
- [ ] **Step 5: `scripts/prompts/reconcile.md`.** Inputs: reads A and B, `transcription/diff/pNNN.md`, all strips, previous final pages. Procedure: for every differing pair, look at the strip that contains the line and decide; write `transcription/final/pNNN.json` with `"reader": "final"` and a `"decisions": [{"where", "chose": "A"|"B"|"neither", "text", "reason"}]` array; anything undecidable stays `[?]` and is listed in `uncertain` with `"escalate": true`. Validate before finishing.
- [ ] **Step 6: `scripts/prompts/spotcheck.md`.** Inputs: the final page and the strips only (not the reads). Procedure: transcribe the page fresh into `transcription/spotcheck/pNNN.json`, then run `diff_reads.py --a transcription/final/pNNN.json --b transcription/spotcheck/pNNN.json` and report every difference with a verdict (final correct / spotcheck correct / undecidable).
- [ ] **Step 7: Ready to commit.**

---

### Task 5: Transcription pilot and reader-model decision

**Goal:** A tuned reading process proven on five graded pages, with a documented choice of reader model.

**Files:**
- Create: `transcription/reads/A/`, `transcription/reads/B/`, `transcription/diff/`, `transcription/final/` for pilot pages
- Modify: `scripts/prompts/read.md`, `docs/conventions.md` (tuning), `docs/pipeline-log.md`

**Pilot pages:** `p001` (TEXTE opening with ornament and initial), `p004` (dense annotation with many margin notes), `p159` (foot-of-page citation block), `p044` (misprinted folio "24"), `p000-title`.

**Acceptance Criteria:**
- [ ] Two Opus reads per pilot page (Agent tool, `model: "opus"`, prompt = `read.md` with `READER=A|B`), both validated, diffed.
- [ ] Reconciliation by a Fable agent for every page with agreement < 100%.
- [ ] **My own read** (main session) of every pilot page: I view the strips and compare against `final/` line by line, recording every error in `docs/pipeline-log.md` as `page / line / final text / correct text / who was right (A, B, neither)`.
- [ ] Metrics logged: A–B agreement per page, errors in final per page, error types (long-s, u/v, marker placement, margin note key, punctuation, missed line).
- [ ] Decision written in the log: **Opus stays as reader** only if the reconciled final has zero substantive errors on all five pages (substantive = anything except a spacing quirk) after at most two rounds of prompt tuning. Otherwise **Fable reads** from here on (both A and B, or A by Fable and B by Opus with Fable reconciling; choose and record).
- [ ] The pilot pages' `final/` files are corrected to match my read and marked `status.final = "done"`, `status.spotchecked = "done"`.

**Verify:** `docs/pipeline-log.md` contains a section `## Pilot results` with the metrics table and a line starting `Decision:`.

**Steps:**
- [ ] Step 1: Dispatch 10 read agents (5 pages × A/B) in parallel. Each must finish with a passing validator.
- [ ] Step 2: Run `diff_reads.py` on all five; read the diffs (they are short).
- [ ] Step 3: Dispatch Fable reconcilers for pages with differences.
- [ ] Step 4: Personal read. I open each strip with the Read tool and compare. Fix `final/`.
- [ ] Step 5: Tune prompts/conventions for the error types seen; if tuning was needed, re-run A/B on the worst page to confirm the fix.
- [ ] Step 6: Log results and the decision. Ready to commit.

---

### Task 6: Full transcription in waves with spot checks

**Goal:** `transcription/final/` complete for all 162 pages, every page validated, spot-checked at the agreed rate, no open escalations.

**Files:**
- Create: `transcription/reads/{A,B}/*.json`, `transcription/diff/*.md`, `transcription/final/*.json`, `transcription/spotcheck/*.json`
- Modify: `manifest.json`, `docs/pipeline-log.md`

**Wave protocol:** pages are processed in manifest order in waves of 8. A wave's readers receive the `final/` files of the 3 pages preceding the wave (context). Within a wave the 16 reads run in parallel. After the wave: diff all, reconcile those < 100%, validate, escalate `uncertain[].escalate` items to me, then mark `status.final = "done"` and start the next wave. About 20 waves.

**Acceptance Criteria:**
- [ ] Every page has `status.final = "done"` and passes `validate_page.py`.
- [ ] Every escalated item was resolved by me (main session) and the resolution recorded in the page's `decisions` with `"chose": "carson-session"`; count logged.
- [ ] Spot checks: every 10th page in manifest order plus every page whose reconciler made more than 5 decisions gets `spotcheck.md` run by a Fable agent. Every spot-check difference is adjudicated and logged. If any wave's spot checks find a substantive error in `final/`, the two neighbouring pages are also spot-checked, and if the wave's substantive-error rate exceeds 1 per 10 pages, the reader model switches to Fable for all remaining waves (logged as a decision).
- [ ] `docs/pipeline-log.md` has a per-wave table: pages, mean agreement, reconciled count, escalations, spot-check results.
- [ ] Manifest counts: `python -c "…"` reports 162 final, 162 validated, N spot-checked with N ≥ 17.

**Verify:** `uv run --with jsonschema python scripts/validate_page.py --all-final` → `162 ok, 0 failed`

**Steps:** run the wave protocol above; log after each wave; ready to commit after every 5 waves.

---

### Task 7: Stitch into logical sections

**Goal:** `text/sections.json`: the whole book as ordered sections with reflowed text and page markers, computed from `final/` by script.

**Files:**
- Create: `scripts/stitch_text.py`, `scripts/tests/test_stitch.py`
- Create: `text/sections.json`

**Section record:**

```json
{"id": "annot-005", "kind": "annotation", "number": 5, "label": "ANNOTAT. V.",
 "pages": ["p040", "p041", "p042", "p043"],
 "text": "⟦p040⟧… reflowed diplomatic text with {a} markers …⟦p041⟧…",
 "notes": [{"key": "a", "page": "p040", "text": "Seneque au liu. des benefices."}],
 "starts_mid_page": true, "ends_mid_page": false}
```

Reflow rules (deterministic): join lines of a paragraph with a single space; a line ending in `-` joins the next without space and without the hyphen **unless** the next line starts with a capital or the joined word is in `scripts/hyphen_keep.txt` (compound words to preserve; starts empty and grows during translation review); a `⟦pNNN⟧` marker is inserted at the start of each page's text; paragraphs separated by `\n\n`; headings become section boundaries (`TEXTE.` → `texte-NN` numbered in order; `ANNOTAT. N.` → `annot-NNN`); title and argument are their own sections (`title`, `argument`).

**Acceptance Criteria:**
- [ ] Sections come out in book order; `annot-001`…`annot-111` all present exactly once; `texte-*` sections numbered consecutively.
- [ ] Every page id appears exactly once as a marker across all sections, in manifest order.
- [ ] Every note key is unique within its page, and every `{x}` in a section's text has a note on the same page.
- [ ] Tests cover: hyphen join, hyphen keep-list, marker insertion at page start, a paragraph that continues across a page break (no `\n\n` at the boundary when `continues_next` is true), heading detection for `ANNOTAT. XLIIII.`-style Roman numerals including forms like `IIII`, `XCIX`, `CXI`.

**Verify:** `uv run --with pytest pytest scripts/tests/test_stitch.py -q` and `uv run python scripts/stitch_text.py --check` → `162 pages, 111 annotations, NN texte sections, 0 problems`

---

### Task 8: Case file

**Goal:** `docs/case-file.md`, the factual backbone every translator and reviewer receives.

**Files:**
- Create: `docs/case-file.md`, `scripts/prompts/casefile.md`

**Acceptance Criteria (sections the file must contain, each with sources cited):**
- [ ] Timeline of the affair with dates: the 1538 marriage in Artigat, Martin's departure c. 1548, Arnaud du Tilh's arrival 1556, the Rieux trial 1559–60, the Toulouse appeal, the return of the real Martin Guerre, the sentence and execution 16 September 1560.
- [ ] People: Martin Guerre, Bertrande de Rols, Arnaud du Tilh ("Pansette"), Pierre Guerre, Sanxi Guerre, the Rieux judge, Coras and the other Toulouse judges (Mansencal, etc.), with the spellings Coras uses.
- [ ] Places: Artigat, Hendaye, Sajas, Rieux, Toulouse, Le Pin, and the diocese/seneschalsy structure.
- [ ] Procedure: what a *parlement* was, *rapporteur*, *enquête*, *confrontation*, *récolement*, *arrêt*, *appel*, the sentence forms; how to render each in English consistently (a glossary table with the chosen English term).
- [ ] Legal citation conventions Coras uses: `l.`/`ff.`/`D.` (Digest), `C.` (Code), `Inst.` (Institutes), `Nov.`/`Auth.` (Novels), `c.` (Decretals/canon), glossators (Bartolus, Baldus, Accursius), and how each is expanded in English; a table of the most frequent abbreviations seen in the margin notes of the pilot pages.
- [ ] Classical and biblical sources Coras cites (Plautus's *Amphitruo*, Pliny, Aristotle, Macrobius, Seneca, Münster, Vives) with standard English titles.
- [ ] Money, measures, calendar, forms of address.
- [ ] A running **glossary** table (French term → chosen English rendering) that translation review appends to.
- [ ] Sources: Davis, *The Return of Martin Guerre* (1983); Ringold & Lewis (1982); the Argument page's own summary; standard references for Roman law citation.

**Verify:** a Fable agent reads the file against the transcribed `texte-*` sections and reports no contradictions between the case file and what Coras's own text says (the text wins; the case file is corrected).

---

### Task 9: Translation

**Goal:** `translation/sections/*.md` for every section, in order, with markers intact.

**Files:**
- Create: `scripts/prompts/translate.md`, `scripts/check_markers.py`, `scripts/tests/test_check_markers.py`
- Create: `translation/sections/{title,argument,texte-01,annot-001,…}.md`
- Modify: `docs/case-file.md` (glossary appends), `manifest.json`

**Output format per section file:**

```markdown
---
id: annot-005
pages: [p040, p041, p042, p043]
---
⟦p040⟧ …English prose with {a} markers exactly where the French has them… ⟦p041⟧ …

## Notes
- {a} (p040): **Seneca, *On Benefits*** — Seneque au liu. des benefices. [one-line gloss of the point cited, if recoverable]
```

**Translation conventions** (in the prompt): modern, clear, readable English that keeps Coras's sentence rhythm where it does not hurt clarity; legal terms per the case-file glossary; names per the case file; keep the tone of the court record formal and the annotations discursive; never omit a clause; keep every `⟦pNNN⟧` and `{x}` marker at the corresponding place; translate every margin note and expand its citation (Digest/Code/Institutes book.title.law where identifiable; classical works by standard English title); when a passage is genuinely obscure, translate it literally and add `[unclear: …]`; do not add commentary beyond the notes.

**Agent inputs:** the section from `text/sections.json`; `docs/case-file.md`; `docs/conventions.md` (so it can read the diplomatic French); the preceding 2 sections' French and English (`text/sections.json` entries and `translation/sections/` files); for `texte-*` sections, the Ringold & Lewis PDF text is **not** provided (it is used only in review).

**Acceptance Criteria:**
- [ ] `check_markers.py annot-005` confirms the page markers and letter markers in the English body equal those in the French text in count and order, and that every note key has a Notes entry. Exit 1 otherwise.
- [ ] Every section translated by a Fable agent, in book order, each passing `check_markers.py` before the next section starts (batches of 4 consecutive sections may run in parallel when they share the same preceding context; log which).
- [ ] New glossary terms appended to `docs/case-file.md` by the translator when it makes a rendering choice for a recurring term.
- [ ] `status.translated = "done"` for every page.

**Verify:** `uv run python scripts/check_markers.py --all` → `NNN sections ok, 0 failed`

---

### Task 10: Review passes and fixes

**Goal:** Every section reviewed for fidelity, consistency, and (for TEXTE) divergence from Ringold & Lewis, with findings resolved.

**Files:**
- Create: `scripts/prompts/review-fidelity.md`, `review-consistency.md`, `review-ringold.md`, `fix.md`
- Create: `translation/review/<section>.md`
- Create: `docs/reference/ringold-lewis-1982.txt` (text of the PDF, for review use only)
- Modify: `translation/sections/*.md`, `manifest.json`

**Acceptance Criteria:**
- [ ] Fidelity review per section by a fresh Fable agent: French and English side by side, sentence by sentence; every omission, addition, or mistranslation listed with the French, the current English, and a proposed English. Severity `high` (meaning changed) / `low` (style).
- [ ] Consistency review per section: names, dates, places, legal terms and glossary renderings checked against `docs/case-file.md`; contradictions listed.
- [ ] Ringold review for each `texte-*` section: passages where our meaning diverges from theirs listed with both versions and the French; the reviewer states which is right or that it is undecidable. Our text is not changed to match theirs unless the French supports it.
- [ ] Fixer agent applies accepted findings, re-runs `check_markers.py`, and writes a resolution line per finding in the review file (`applied` / `rejected: reason`).
- [ ] Every `high` finding is resolved; every `undecidable` Ringold divergence is escalated to me and my decision recorded.
- [ ] `status.reviewed = "done"` for every page; summary counts in `docs/pipeline-log.md`.

**Verify:** `grep -L "^resolution:" translation/review/*.md` → empty; `check_markers.py --all` → 0 failed.

---

### Task 11: Split translation per page and emit site data

**Goal:** `translation/pages/pNNN.json` and `site/data/pages/pNNN.json` for every page.

**Files:**
- Create: `scripts/split_pages.py`, `scripts/tests/test_split.py`
- Create: `translation/pages/*.json`, `site/data/pages/*.json`, `site/data/index.json`

**Page site record:**

```json
{"id": "p043", "page": 43, "folio": "43", "side": "recto", "image": "img/p043.jpg",
 "source": {"kind": "cudl", "image_no": 63, "url": "https://cudl.lib.cam.ac.uk/view/PR-MONTAIGNE-00001-00007-00022/63"},
 "running_head": "PARLEMENT DE THOLOSE.",
 "prev": "p042", "next": "p044",
 "english": [
   {"type": "heading", "text": "ANNOTATION V"},
   {"type": "paragraph", "continued": true, "html": "…text with <sup class=\"mk\" data-key=\"a\">a</sup>…",
    "notes": [{"key": "a", "citation": "Seneca, *On Benefits*", "original": "Seneque au liu. des benefices.", "gloss": "…"}]}
 ],
 "french": [
   {"type": "heading", "text": "ANNOTAT. V."},
   {"type": "paragraph", "lines": ["…", "…"]}
 ],
 "french_notes": [{"key": "a", "lines": ["…"]}],
 "uncertain": [ … ]}
```

**Acceptance Criteria:**
- [ ] The English for a page is exactly the text between its `⟦pNNN⟧` marker and the next marker, across whichever section(s) it falls in; a page containing the end of one section and the start of the next gets both, with the heading between.
- [ ] Headings normalized for display: `ANNOTAT. V.` → `ANNOTATION V`; `TEXTE.` → `TEXT`.
- [ ] `continued: true` on a first paragraph that starts mid-sentence (page marker not preceded by `\n\n` or a heading).
- [ ] Notes attached to the paragraph containing their marker; markers rendered as `<sup>` with `data-key`.
- [ ] `site/data/index.json`: ordered list of page ids with page numbers and the first heading on each page (for the jump menu).
- [ ] `site/img/` is a symlink or copy of `pages/read/` (choose copy; keep the site self-contained).
- [ ] Tests: split of a two-section synthetic input; heading normalization; `continued` detection; note attachment.

**Verify:** `uv run --with pytest pytest scripts/tests/test_split.py -q` and `uv run python scripts/split_pages.py` → `162 pages written`

---

### Task 12: The site

**Goal:** A static viewer that opens instantly, shows page image left and translation right with sidenotes in place, has a French toggle, and works on a phone.

**Files:**
- Create: `site/index.html`, `site/app.js`, `site/style.css`
- Use: `site/data/`, `site/img/`

**Acceptance Criteria:**
- [ ] Hash routing: `#p043` loads that page; no hash → title page. Prev/next buttons, ← → keys, a jump box accepting a page number or `title`/`argument`.
- [ ] Left: page image, `object-fit: contain` to the panel height; click toggles a zoomed view that can be panned (CSS transform, mouse drag and touch).
- [ ] Right: heading(s) and paragraphs; a paragraph with notes is a two-column grid on screens ≥ 1100px (text ~62ch, notes column ~18rem) with each sidenote vertically aligned to the top of its paragraph, stacked in marker order; on narrower screens markers are tappable and the note expands inline below the paragraph.
- [ ] Sidenote shows the expanded citation in English, the original French in a smaller italic line, and the gloss if present. Hovering a marker highlights its sidenote and vice versa.
- [ ] A `continued` paragraph shows a faint "⋯ continued" marker before its first word.
- [ ] Toggle button `English | French`; French mode renders the diplomatic lines one per line in a serif font with the margin notes in the sidenote column keyed by letter, so it can be read against the image.
- [ ] Header shows `p. 43` and, when the folio differs, `printed as 24`, plus `image 63 · Cambridge University Library` linking to the CUDL view; the Gallica page says so. Footer carries the CC BY-NC 4.0 credit and a link to the item.
- [ ] Uncertain readings are shown in French mode with a dotted underline and the note on hover.
- [ ] No framework, no build, no external scripts. Fonts: system serif stack. Light and dark via `prefers-color-scheme`.
- [ ] Works when opened over `python -m http.server` (fetch of JSON needs http, not `file://`; the README says so).
- [ ] Checked in a real browser (page-capture skill or Playwright) on the five pilot pages at 1440px and 390px widths; screenshots saved to `docs/checks/site-*.png` and reviewed by me.

**Verify:** `cd site && python3 -m http.server 8765` then open `http://localhost:8765/#p043`; screenshots exist and show the notes aligned beside their paragraph.

---

### Task 13: README and log finalization

**Goal:** `2026/README.md` explaining what this is, how to regenerate every stage, the conventions in brief, the source and its quirks, the status, and credits. The root `README.md` gets a two-line pointer to `2026/` and a correction of the edition (1572 Paris, not 1561 Lyon).

**Acceptance Criteria:**
- [ ] README lists each script with its one-line purpose and command.
- [ ] README states the image→page mapping rule and the missing page 41, with the Gallica view number.
- [ ] README states the verification results (agreement, spot-check counts, escalations) with the date.
- [ ] No AI attribution anywhere.

**Verify:** read-through by a fresh agent following the README to run `--check` on every stage succeeds.

---

## Self-review

- Spec coverage: design sections 1–9 map to Tasks 1–13 (acquire→2, crop→3, transcribe→4/5/6, stitch→7, case file→8, translate→9, review→10, re-split/site→11/12, housekeeping done before planning). The semi-diplomatic layer is conditional in the design and is deliberately not a task; if the pilot or review demands it, it becomes a derived view inside Task 7 and this plan is amended.
- Type consistency: page ids `pNNN`/`p000-title`/`p000-argument` everywhere; section ids `texte-NN`/`annot-NNN`/`title`/`argument`; markers `⟦pNNN⟧` and `{a}`; manifest status keys as defined in Task 1's `STAGES`.
- Ordering dependency: Task 8 (case file) only needs `texte-*` transcriptions for its verification step, so it can start after Task 5 and finish after Task 7.

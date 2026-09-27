# Site handoff: building the side-by-side viewer

For a fresh session whose job is the **static site** (design §6, plan Tasks 11–12). The
transcription pipeline is running in a separate session at the same time; this document
says what exists, what the data looks like, what the site must do, and which files each
session owns so the two do not collide.

## 1. What this project is, in one paragraph

We are producing a verified diplomatic transcription and an English translation of Jean de
Coras, *Arrest memorable du Parlement de Tholose* (Paris, 1572), the judge's own account of
the Martin Guerre case with his 111 learned annotations, from Cambridge University Library
scans (one page from Gallica). The site shows each page image on the left and, on the
right, the English for that page with the marginal citations rendered as sidenotes at the
letter where they belong, plus a toggle to show the diplomatic French instead. Read
`docs/design.md` first (10 minutes), then plan Tasks 11 and 12 in `docs/plan.md`.

## 2. State of the pipeline (2026-09-21, evening)

| Stage | State |
|---|---|
| Manifest (`manifest.json`, 162 pages) | done; image→page mapping verified on every page |
| Full-resolution scans (`raw/`, gitignored) | done, 2941×4711 |
| Crops (`pages/full/` gitignored, `pages/read/` 1600px wide, gitignored) | done for all 162 |
| Transcription (`transcription/final/<id>.json`) | **in progress**, 15 pages final, waves of 5 pages running; expect the full book over the next day or two |
| Stitch into sections (`text/sections.json`) | not started (plan Task 7) |
| Case file (`docs/case-file.md`) | draft done |
| Translation (`translation/sections/*.md`) | not started; begins after transcription |
| Re-split per page (`translation/pages/`, `site/data/`) | not started (plan Task 11) |
| Site (`site/`) | not started (plan Task 12) — **this session's job** |

Finalized pages you can use as real sample data now: `p000-title`, `p000-argument`, `p001`
through `p006`, `p008`, `p009`, `p012`, `p013`, `p041`, `p044`, `p159`. More arrive
continuously; `manifest.json` → `pages[].status.final == "done"` is the source of truth.

## 3. Page identity

- `id` is `pNNN` by **true page number** (`p001`…`p160`), plus `p000-title` and
  `p000-argument`. Manifest order is the reading order.
- `folio` is the **printed** page number, which is wrong on four pages (44→"24", 45→"44",
  48→"58", 77→"78"). Show both when they differ.
- `image` is the Cambridge image number; the CUDL viewer URL for a page is
  `https://cudl.lib.cam.ac.uk/view/PR-MONTAIGNE-00001-00007-00022/<image>`. `p041` has
  `image: null` and `source: "gallica"` (`https://gallica.bnf.fr/ark:/12148/bpt6k52469j/f58`).
- `side`: recto/verso (margin notes sit on the outer side: right on rectos, left on versos).

## 4. The transcription JSON (the French layer)

`transcription/final/<id>.json`, schema in `scripts/page_schema.json`, rules in
`docs/conventions.md`. Essentials for rendering:

```json
{
  "id": "p004", "running_head": "ARREST DV", "folio": "4",
  "blocks": [
    {"type": "heading", "text": "TEXTE.", "spaced_caps": true},
    {"type": "paragraph", "continues_prev": true, "continues_next": false,
     "spaced_caps": false,
     "lines": ["me {ſ}. meſme qu'en ceſt aage, on void quelquesfois adue-", "…"]},
    {"type": "ornament", "text": "woodcut headpiece"}
  ],
  "margin_notes": [{"key": "ſ", "lines": ["L. minorem", "D. de rit. nup", "…"], "beside_line": "…"}],
  "foot_notes":   [{"key": "q", "lines": ["…"]}],
  "signature": "A iij", "catchword": null, "ornaments": ["decorated initial A, 5 lines"],
  "uncertain": [{"where": "blocks[1].lines[3]", "text": "…", "note": "…", "escalate": false}],
  "decisions": [ … reconciliation record, not for display … ]
}
```

- **Lines are printed lines**; render the French view one line per printed line, in a serif
  face, so it can be read against the image. Line-end hyphens are as printed.
- `{x}` inside a line is a **marker**: the small letter keying a marginal citation. Render
  as a superscript `x` linked to the note with the same `key` (`margin_notes` first, then
  `foot_notes`). Keys follow the printer's alphabet `a b c d e f g h i k l m n o p q r ſ t u x y z`;
  a restarted alphabet on the same page uses `a2`, `a3`. A marker may have **no note on its
  page** (the citation ran on to the next page) and a note may have no marker (its marker
  was on the previous page); both are recorded in `uncertain[]`. Render what exists.
- `ſ` is long s (U+017F); `u/v`, `i/j` are as printed; tildes (`ẽ õ ã`) are abbreviation
  marks. Do not normalize. The Unicode is NFC.
- `spaced_caps: true` means the text was letterspaced in the print; render with CSS
  `letter-spacing`, the text itself is closed up (`ARREST DV`).
- `continues_prev` / `continues_next`: the paragraph runs across the page boundary.
- `uncertain[]` entries point at a line (`where`) and explain a doubtful reading; the
  French view should show a dotted underline on such lines with the note on hover. `[?]`,
  `[??]`, `[...]` may appear in text for unreadable characters.
- The page image for `<id>` is `pages/read/<id>.jpg` (1600px wide). It is **not in git**:
  for development, run the site over `python -m http.server` from `2026/` and reference
  `../pages/read/<id>.jpg`, or copy a few into `site/img/`. In production the images will
  be served from **S3 behind CloudFront**; make the image base URL a single constant.

## 5. The translation JSON (the English layer) — contract, not yet produced

The translation pipeline will emit `site/data/pages/<id>.json` with this shape (plan
Task 11); build the site against it and generate fixtures for the sample pages by hand or
with placeholder English until real data exists:

```json
{"id": "p043", "page": 43, "folio": "43", "side": "recto", "image": "img/p043.jpg",
 "source": {"kind": "cudl", "image_no": 63, "url": "https://cudl.lib.cam.ac.uk/view/PR-MONTAIGNE-00001-00007-00022/63"},
 "running_head": "PARLEMENT DE THOLOSE.", "prev": "p042", "next": "p044",
 "english": [
   {"type": "heading", "text": "ANNOTATION V"},
   {"type": "paragraph", "continued": true,
    "html": "…text with <sup class=\"mk\" data-key=\"a\">a</sup>…",
    "notes": [{"key": "a", "citation": "Seneca, <i>On Benefits</i>", "original": "Seneque au liu. des benefices.", "gloss": "…"}]}
 ],
 "french": [ {"type": "heading", "text": "ANNOTAT. V."}, {"type": "paragraph", "lines": ["…"]} ],
 "french_notes": [{"key": "a", "lines": ["…"]}],
 "uncertain": [ … ]}
```

The English is split per page from section-level translations at `⟦pNNN⟧` markers, so a
page's English is exactly the text printed on that page, with `continued: true` when it
starts mid-sentence. Headings are normalized for display (`ANNOTAT. V.` → `ANNOTATION V`,
`TEXTE.` → `TEXT`). Each note carries the expanded citation, the original French, and an
optional one-line gloss. `site/data/index.json` will list page ids in order with the first
heading on each page, for the jump menu.

## 6. What the site must do (design §6, plan Task 12)

- One `index.html`, plain HTML/CSS/JS, **no framework, no build step, no external scripts**.
  Hash routing `#p043`; no hash → title page. Prev/next, ← → keys, jump box (page number,
  `title`, `argument`).
- Left: the page image, fit to panel height; click to zoom, drag/touch to pan.
- Right: headings and paragraphs; each paragraph with notes is a two-column grid at
  ≥1100px (text ~62ch, sidenote column ~18rem), sidenotes aligned to the top of their
  paragraph and stacked in marker order; below 1100px, markers are tappable and the note
  expands inline. Hovering a marker highlights its sidenote and vice versa.
- Toggle `English | French`. French mode = diplomatic lines one per printed line, serif,
  notes in the sidenote column keyed by letter, uncertain lines dotted-underlined.
- Header: `p. 43`, `printed as 24` when the folio differs, `image 63 · Cambridge University
  Library` linking to CUDL (or the Gallica line for p041). Footer: CC BY-NC 4.0 credit and
  the item link. Light/dark via `prefers-color-scheme`. Works on a phone (390px).
- Data is loaded per page on demand (one small JSON each).
- It goes live on codebycarson only after the translation is finished; until then it is a
  local dev site.

## 7. Ownership: who touches what

**Site session (you) owns:** `site/**`, `scripts/split_pages.py` and its tests,
`scripts/tests/test_split.py`, `docs/site-*.md`. You may also draft `scripts/stitch_text.py`
(plan Task 7) if you want to feed the site from real sections, but say so in
`docs/pipeline-log.md` first so the pipeline session does not write it too.

**Pipeline session owns:** `manifest.json`, `transcription/**`, `pages/**`, `raw/**`,
`scripts/{acquire,crop,measure_pages,folio_sheet,build_manifest,validate_page,diff_reads,
auto_resolve,normalize_spacing,make_final,render_prompt,wave}.py`, `scripts/prompts/**`,
`docs/conventions.md`, `docs/case-file.md`, `docs/pipeline-log.md` (append-only for you).

Rules for both:
- Never edit `manifest.json` by hand; if a script must write it, use
  `pagelib.write_manifest` (atomic). A truncated manifest cost an hour once already.
- Read finals, never write them.
- **No git commits, no `git add`,** unless Carson says so explicitly. No AI attribution
  anywhere, ever (no "Generated with", no Co-Authored-By).
- Images are never committed (`pages/read/` is gitignored).
- Run Python with `uv run --with <deps> python …`; system Python has no packages.
- Tests: `uv run --with pytest,jsonschema pytest scripts/tests -q` (128 passing now).

## 8. Where to look

- `docs/design.md` — the approved design (§6 is the site).
- `docs/plan.md` — Tasks 11 (split + site data) and 12 (site) with acceptance criteria.
- `docs/conventions.md` — what every character in the French means.
- `transcription/final/p004.json` — a dense annotation page with 8 markers and notes.
- `transcription/final/p159.json` — a page with margin notes overflowing into a foot block.
- `transcription/final/p000-title.json` — display lines only, ornaments described.
- `docs/case-file.md` — the story, the people, the legal citation conventions (useful for
  writing good placeholder English and for the notes' expanded citations).
- `docs/pipeline-log.md` — running record; append a dated section for site work.

## Status 2026-09-22: site built, awaiting go-live

The site session finished plan Tasks 11–12 in a revised form. Read these instead of §6 above
where they differ:

- `docs/site-design.md` — approved design. Hosting is `translations.codebycarson.com`
  (multi-book root; this book at `/martin-guerre/`), served from a new bucket +
  CloudFront distribution in the `code-by-carson` CDK stack. Dark by default with a light
  toggle. Images are the full-resolution crops as WebP q70.
- `docs/site-plan.md` (+ `.tasks.json`) — the tasks and their state.
- `docs/site-data-contract.md` — what the split reads (`text/sections.json`,
  `translation/sections/*.md` in the translator-prompt format) and what it writes
  (`site/data/`). **The pipeline changes nothing**: the split follows the format in
  `scripts/prompts/translate.md`.
- `scripts/split_pages.py` → `site/data/` (committed; `.gitignore` now un-ignores it).
  `scripts/site_images.py` → `site/img/` (gitignored). Tests in `scripts/tests/test_split.py`,
  `test_site_images.py`.
- The viewer and landing page live in `~/github/code-by-carson/translations/`; run locally per
  `site/README.md`. Browser checks and screenshots: `docs/checks/site-*.png`,
  `docs/checks/site-checklist.md`.
- After new pages are transcribed or translated: rerun `split_pages.py`, commit `site/data/`,
  push, then `gh workflow run deploy-translations.yml -R CarsonDavis/code-by-carson`.

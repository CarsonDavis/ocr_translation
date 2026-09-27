# translations.codebycarson.com Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Spec: `docs/site-design.md`. Context: `docs/site-handoff.md`.

**Goal:** A live static site at `translations.codebycarson.com` whose `/martin-guerre/` viewer shows each scan beside its English translation or diplomatic French, renders now with partial data, and redeploys in one click as data arrives.

**Architecture:** Two repos. `ocr_translation/2026` generates per-book data (`site/data/`) and WebP images (`site/img/`, gitignored, uploaded once to S3). `code-by-carson` holds the generic viewer, the landing page, the CDK resources (one bucket, one CloudFront distribution), and a workflow that checks out the book repo and assembles the site tree.

**Tech Stack:** Python 3.12 via `uv run`, pytest, jsonschema, cwebp; plain HTML/CSS/JS (no framework, no build, no external requests); AWS CDK (Python) + GitHub Actions OIDC; Playwright for screenshots.

**Rules that override everything below:** no `git add`/commit/push unless Carson says so; no AI attribution anywhere; never write `manifest.json`, `transcription/**`, `pages/**`; Python via `uv run --with <deps> python …`; append (never rewrite) `docs/pipeline-log.md`.

---

## File structure

**ocr_translation/2026** (this repo; cwd for Tasks 1–3, 8, 11)
```
scripts/site_schema.json          JSON schema for site/data/pages/<id>.json
scripts/split_pages.py            manifest + finals (+ translation sections) -> site/data/
scripts/site_images.py            pages/full/*.jpg -> site/img/*.webp; prints or runs the S3 upload
scripts/tests/test_split.py       unit tests (synthetic fixtures under scripts/tests/fixtures/site/)
scripts/tests/test_site_images.py
site/data/book.json               generated, committed
site/data/index.json              generated, committed
site/data/pages/<id>.json         generated, committed (162 files)
site/img/<id>.webp                generated, gitignored
site/README.md                    what is generated, how, and how to run the viewer against it
docs/site-data-contract.md        the translation-section format the pipeline must emit
docs/checks/site-*.png            browser screenshots
```

**code-by-carson** (cwd for Tasks 4–7, 9, 10, 11)
```
translations/landing/index.html   root page
translations/landing/style.css
translations/viewer/index.html    generic viewer shell
translations/viewer/style.css     tokens (dark default, light), layout, sidenotes, image panel
translations/viewer/app.js        routing, loading, rendering, zoom/pan, toggles
translations/viewer/data          -> symlink to ../../../translator/2026/site/data (gitignored)
translations/viewer/img           -> symlink to ../../../translator/2026/site/img  (gitignored)
translations/README.md            local dev + how to add a book
.github/workflows/deploy-translations.yml
cdk/stacks/portfolio_stack.py     + Translations bucket/distribution/alias/grants/outputs
cdk/cdk.json                      + "attach_translations_domain": "true"
INDEX.md, docs/IMPLEMENTATION_LOG.md, docs/REQUIREMENTS.md   updated
```

---

## Phase 1 — Book data (ocr_translation/2026)

### Task 1: Site schema and split_pages.py (French layer)

**Goal:** `uv run python scripts/split_pages.py` writes `site/data/book.json`, `index.json`, and 162 `pages/<id>.json` from `manifest.json` and whatever finals exist; every page validates against `scripts/site_schema.json`.

**Files:**
- Create: `scripts/site_schema.json`, `scripts/split_pages.py`, `scripts/tests/test_split.py`, `scripts/tests/fixtures/site/` (a mini manifest with 3 pages and 2 finals copied from `transcription/final/p004.json` and `p005.json`, ids kept)
- Read only: `manifest.json`, `transcription/final/*.json`, `scripts/pagelib.py` (`load_manifest`, `load_page`, `blocks`, `note_map`, `MARKER_RE`, `nfc_all`)

**Output shapes (exact):**

`book.json` — the literal from `docs/site-design.md` §4, with `first_page: "p000-title"`, `images: {"base": "img/", "ext": ".webp", "width": 2805}`, `layers: [{"code":"en","label":"English"},{"code":"fr","label":"French"}]`, `default_layer: "en"`, `stylesheet: null`, `description`: one paragraph (Coras, 1572, Martin Guerre, source, in progress).

`index.json`
```json
{"pages": [{"id": "p000-title", "page": 0, "folio": null, "heading": null, "layers": {"fr": true, "en": false}}, …]}
```
`page` is the manifest `page` (0 for the two front pages); `heading` = first `heading` block's text, display-normalized (Task 2's `normalize_heading`, stub it here as identity if Task 2 is not yet done — no: implement `normalize_heading` in Task 1 since both need it).

`pages/<id>.json`
```json
{"id": "p004", "page": 4, "folio": "4", "side": "verso",
 "image": "p004",
 "source": {"kind": "cudl", "image_no": 26, "url": "https://cudl.lib.cam.ac.uk/view/PR-MONTAIGNE-00001-00007-00022/26"},
 "running_head": "ARREST DV", "prev": "p003", "next": "p005",
 "english": null,
 "french": [
   {"type": "paragraph", "lines": ["…"], "continues_prev": true, "continues_next": false, "spaced_caps": true},
   {"type": "heading", "text": "TEXTE.", "spaced_caps": true},
   {"type": "ornament", "text": "woodcut headpiece"}
 ],
 "french_notes": [{"key": "ſ", "kind": "margin", "lines": ["L. minorem", "…"]}, {"key": "q", "kind": "foot", "lines": ["…"]}],
 "uncertain": [{"where": "blocks[1].lines[3]", "text": "…", "note": "…"}]}
```
- `image` is the page id; the viewer builds `book.images.base + id + book.images.ext`.
- `source.kind` = manifest `source`; for `gallica`: `image_no: null`, `url: "https://gallica.bnf.fr/ark:/12148/bpt6k52469j/f58"`.
- `french` = the final's `blocks` verbatim (all keys kept: `type`, `text`, `lines`, `continues_prev`, `continues_next`, `spaced_caps`), or `null` when `manifest.pages[].status.final != "done"` or the file is missing. `french_notes` = `margin_notes` then `foot_notes`, each `{key, kind, lines}` (`kind` = `"margin"|"foot"`; `key` may be null); `[]` when none, `null` when the layer is null. `uncertain` = the final's entries minus `escalate`, `[]` when none.
- `prev`/`next` are neighbours in manifest order, `null` at the ends.
- Everything NFC (`pagelib.nfc_all`). Output JSON `ensure_ascii=False`, `indent=1`, trailing newline, sorted keys off (keep the order above).

**CLI:** `split_pages.py [--root DIR] [--out DIR]` defaults `root=2026/`, `out=root/site/data`. Prints `162 pages written (15 french, 0 english)`.

**Schema (`scripts/site_schema.json`):** JSON Schema draft 2020-12 for the page record above; `english`/`french` are `["array","null"]`; block items require `type`; `paragraph` requires `lines` (French) or `html` (English, Task 2); `additionalProperties: true` on blocks so a book can extend.

**Acceptance Criteria:**
- [ ] `test_french_layer_copied`: for fixture p004, `french` equals the final's `blocks`, `french_notes[0] == {"key":"ſ","kind":"margin","lines":[…8 lines…]}`, foot notes follow margin notes.
- [ ] `test_pending_page`: a manifest page with no final gets `french: null`, `french_notes: null`, `uncertain: []`, and `index.json` `layers.fr == false`.
- [ ] `test_prev_next_and_order`: first page `prev` null, last `next` null, `index.json` order == manifest order.
- [ ] `test_gallica_source`: p041 record has `source.kind == "gallica"`, `image_no is None`, the Gallica URL.
- [ ] `test_schema_valid`: every written page validates with `jsonschema.validate`.
- [ ] `test_normalize_heading`: `ANNOTAT. V.` → `ANNOTATION V`, `TEXTE.` → `TEXT`, `ANNOTAT. XLIV.` → `ANNOTATION XLIV`, `ARGVMENT.` → `ARGUMENT`, unknown → stripped of trailing period only.
- [ ] Running against the real repo: `162 pages written (N french, 0 english)` with N == count of `final == "done"`; `site/data/index.json` has 162 entries.

**Verify:** `uv run --with pytest,jsonschema pytest scripts/tests/test_split.py -q` → all pass; `uv run --with jsonschema python scripts/split_pages.py` → `162 pages written (15 french, 0 english)` (15 may be higher; the pipeline is running).

**Steps:**
- [ ] Write `scripts/tests/fixtures/site/manifest.json` (3 pages: p004 final done, p005 final done, p006 final pending; copy the real records, keep `status`) and copy `transcription/final/p004.json`, `p005.json` into `scripts/tests/fixtures/site/final/`.
- [ ] Write the failing tests above (use `subprocess` like `test_manifest.py`, or import `split_pages` and call `build(root, out)`; prefer import).
- [ ] Implement `normalize_heading`, `page_record`, `index_record`, `book_record`, `build(root, out) -> (n_pages, n_fr, n_en)`, `main()`.
- [ ] Write `scripts/site_schema.json`; validate in `build` (fail loudly on the first invalid page).
- [ ] Run tests, run the real build, eyeball `site/data/pages/p004.json` against the final.

---

### Task 2: English layer from translation sections

**Goal:** `split_pages.py` also reads `translation/sections/*.md` when present and fills `english` per page; the section format is documented in `docs/site-data-contract.md` for the pipeline session.

**Files:**
- Create: `docs/site-data-contract.md`, `scripts/tests/fixtures/site/sections/` (two synthetic sections spanning three pages)
- Modify: `scripts/split_pages.py`, `scripts/tests/test_split.py`, `scripts/site_schema.json`

**Section file format (write this into `docs/site-data-contract.md` verbatim, then implement it):**
```
---
section: annot-005
order: 12
heading: ANNOTAT. V.
---
⟦p043⟧ English prose of the section begins here, with letter markers {a} kept in place
exactly where the French has them {b}. Paragraph breaks are blank lines.

Second paragraph continues ⟦p044⟧ across the page boundary mid-sentence {c}.

## notes ⟦p043⟧
a | Seneca, *On Benefits* 4.2 | Seneque au liu. des benefices. | Seneca on gratitude between unequal parties.
b | Digest 23.2, *De ritu nuptiarum*, lex *Minorem* | L. minorem D. de rit. nup. |

## notes ⟦p044⟧
c | Aristotle, *History of Animals* 10.5 | Ariſtote au v. de la nature des animaux x. c. v. | 
```
Rules: front matter is required (`section`, `order` integer, `heading` as printed in the French, or `heading: TEXTE.` for text sections). Page markers are `⟦pNNN⟧` using the page **id** (`⟦p000-title⟧`, `⟦p043⟧`); the script also accepts `⟦p.43⟧` and `⟦p43⟧` and normalizes. The first marker must precede all prose. A marker inside a paragraph (not at a paragraph start) means the paragraph continues across pages. Notes: one `## notes ⟦pNNN⟧` block per page that has notes; each line `key | citation | original | gloss` (gloss may be empty; `*x*` in citation → `<i>x</i>`). Sections are concatenated in `order`; a page can therefore contain the tail of one section and the head of the next, with the next section's heading between.

**Split algorithm:** concatenate sections in `order` into a stream of events: `heading(text)` at each section start, then prose split at markers. For each page id in manifest order, `english` = `[]` of blocks in stream order between its marker and the next: a `heading` block (`normalize_heading`) when the section starts on that page; `paragraph` blocks `{type, html, continued, notes}` where `html` = paragraph text with `{a}` → `<sup class="mk" data-key="a">a</sup>` and `*x*` → `<i>x</i>`, HTML-escaped otherwise; `continued: true` on the first paragraph when the page marker sat mid-paragraph; `notes` = the page's note entries whose key appears in that paragraph, in marker order, each `{key, citation, original, gloss}` (`gloss` null when empty). A page with no marker anywhere → `english: null`. A marker for an unknown page id → error naming the section and marker.

**Acceptance Criteria:**
- [ ] `test_split_two_sections`: fixture sections yield p043 = `[heading ANNOTATION V, paragraph, paragraph(continued? no)]`, p044 = `[paragraph continued=true …]`; the section-2 heading lands on the page where its first marker is.
- [ ] `test_marker_html`: `{a}` becomes `<sup class="mk" data-key="a">a</sup>`; `&` becomes `&amp;`; `*On Benefits*` becomes `<i>On Benefits</i>`.
- [ ] `test_notes_attached_to_paragraph_in_marker_order`: note `b` attaches to the paragraph containing `{b}`; a note with no marker on the page is kept on the page's first paragraph and reported on stderr.
- [ ] `test_marker_variants`: `⟦p.43⟧`, `⟦p43⟧`, `⟦p043⟧` all map to `p043`.
- [ ] `test_unknown_page_marker_errors`: `⟦p999⟧` raises `ValueError` mentioning the section name.
- [ ] `test_index_flags_en`: `index.json` `layers.en` true only for pages with English.
- [ ] Schema accepts English paragraph blocks (`html` string, `continued` bool, `notes` array).

**Verify:** `uv run --with pytest,jsonschema pytest scripts/tests/test_split.py -q` → pass; real build still prints `162 pages written (N french, 0 english)` (no sections exist yet).

**Steps:**
- [ ] Write `docs/site-data-contract.md` (format above + the page JSON shape from Task 1 + `index.json` + `book.json`).
- [ ] Write fixture sections and the failing tests.
- [ ] Implement `parse_section(path) -> Section`, `split_english(sections, page_ids) -> dict[id, list[block] | None]`, wire into `build`.
- [ ] Run tests and the real build.

---

### Task 3: site_images.py and repo housekeeping

**Goal:** `site/img/<id>.webp` for all 162 pages at full resolution, quality 70; an upload command ready for Carson; gitignore and log updated.

**Files:**
- Create: `scripts/site_images.py`, `scripts/tests/test_site_images.py`, `site/README.md`
- Modify: `/Users/cdavis/github/translator/.gitignore` (add `2026/site/img/`), `docs/pipeline-log.md` (append a dated "Site" section: what was added, where, how to regenerate)

**Behaviour:** `site_images.py [--src pages/full] [--out site/img] [--quality 70] [--only p004,p005] [--upload --bucket NAME --profile PROFILE]`. Uses `cwebp -q 70 -m 6 -quiet in -o out`; skips outputs newer than their source; prints `162 images, 152.3 MB`. Without `--upload` it prints the exact upload command:
```
aws s3 sync site/img/ s3://<bucket>/martin-guerre/img/ --profile <profile> --cache-control "public, max-age=31536000, immutable" --content-type image/webp --size-only
```
With `--upload` it runs it via `subprocess` and refuses if `--bucket` or `--profile` is missing. Never touches `pages/**`.

**Acceptance Criteria:**
- [ ] `test_converts_and_skips_fresh`: on a tmp dir with one small JPEG, produces a `.webp`, second run reports `0 converted`.
- [ ] `test_upload_command_text`: `upload_command("b", "prof")` returns the string above.
- [ ] `test_upload_requires_args`: `--upload` without bucket/profile exits 2.
- [ ] Real run produces 162 files in `site/img/`; `site/img/p004.webp` is 2805×3962 (`sips -g pixelWidth`).

**Verify:** `uv run --with pytest pytest scripts/tests/test_site_images.py -q`; `uv run python scripts/site_images.py` → `162 images, ~150 MB`.

---

## Phase 2 — Viewer and landing (code-by-carson)

### Task 4: Viewer shell — routing, data loading, header, footer, toggles, theme

**Goal:** `translations/viewer/` opens over `python3 -m http.server`, loads `book.json` + `index.json`, routes by hash, shows the header/footer, switches layer and theme. Text and image panels are placeholders until Tasks 5–6.

**Files:**
- Create: `translations/viewer/index.html`, `translations/viewer/style.css`, `translations/viewer/app.js`, `translations/README.md`
- Modify: `.gitignore` (add `translations/viewer/data`, `translations/viewer/img`)
- Create symlinks (not committed): `translations/viewer/data -> ../../../translator/2026/site/data`, `translations/viewer/img -> ../../../translator/2026/site/img`

**index.html skeleton:**
```html
<!doctype html><html lang="en" data-theme="dark"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Martin Guerre — translations.codebycarson.com</title>
<link rel="stylesheet" href="style.css"><meta name="darkreader-lock">
</head><body>
<header class="bar">
  <a class="home" href="../">translations</a><span class="sep">/</span><span class="book-title"></span>
  <nav class="pager"><button class="prev" aria-label="Previous page">←</button>
    <form class="jump"><input name="p" inputmode="numeric" placeholder="p." aria-label="Go to page"></form>
    <button class="next" aria-label="Next page">→</button></nav>
  <div class="meta"><span class="pageno"></span><span class="folio"></span><a class="src" target="_blank" rel="noopener"></a></div>
  <div class="controls"><div class="layers" role="tablist"></div><button class="theme" aria-label="Toggle theme"></button></div>
</header>
<main class="stage"><section class="image-panel"></section><section class="text-panel"></section></main>
<footer class="foot"></footer>
<script src="app.js"></script></body></html>
```
The `<title>` is set from `book.json` at runtime; the static one is a fallback.

**app.js structure (one file, ~400 lines by the end of Task 6; keep these module-level functions):**
`loadBook()`, `loadIndex()`, `loadPage(id)` (fetch `data/pages/${id}.json`, cache in a Map), `route()` (hash → id; invalid → `book.first_page`; unknown id → text panel "No such page"), `render(page)`, `renderHeader(page)`, `renderFooter(book)`, `setLayer(code)` (state in `localStorage['tc.layer']`, falls back to first available layer for the page), `setTheme(t)` (`document.documentElement.dataset.theme`, `localStorage['tc.theme']`, default dark), `bindKeys()` (← → prev/next unless focus is in the jump input; `Escape` un-zooms), `jump(value)` (number → `p${zero-padded 3}`, `title`/`argument` → `p000-title`/`p000-argument`), `prefetch(id)` (`new Image().src`).
Header: `p. 43` (front pages: `Title page` / `Argument`); `printed as 24` when `folio` differs from `String(page)`; `image 63 · Cambridge University Library` linking to `source.url`, or `Gallica (BnF)` for `kind == "gallica"`. Footer: `Scans © Cambridge University Library, CC BY-NC 4.0 · Transcription and translation in progress` with links from `book.source`.
Layer tabs: one button per `book.layers`; disabled with `title="translation pending"` / `"transcription pending"` when the page's layer is null.

**style.css tokens:**
```css
:root, :root[data-theme="dark"] { --bg:#181a1b; --surface:#1e2021; --border:#3c4143; --text:#e8e6e3; --muted:#a7a096; --accent:#8eb4f1; --accent-hover:#b1cbf5; --note:#cfc9bf; --paper:#111; }
:root[data-theme="light"] { --bg:#f6f4ef; --surface:#fffdf8; --border:#d9d4c8; --text:#1f1f1f; --muted:#6b665e; --accent:#2f5fb3; --accent-hover:#1e4a94; --note:#4a4640; --paper:#e9e5dc; }
--serif: "Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif; --sans: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
```
Layout: `.bar` sticky top (`top: env(safe-area-inset-top,0px)`), `.stage` = CSS grid `grid-template-columns: minmax(0,1fr) minmax(0,1fr)` at ≥900px, single column (image above text) below; `.stage` height `calc(100dvh - var(--bar-h))` on wide screens with each panel `overflow:auto`. No `*{margin:0}` reset that would fight the text; scope resets to the chrome.

**Acceptance Criteria:**
- [ ] `http://localhost:8765/#p004` shows header `p. 4`, `image 26 · Cambridge University Library` (href to CUDL 26), French tab enabled, English tab disabled with tooltip, footer credit; `#p044` shows `printed as 24`; `#p041` shows the Gallica line; `#p000-title` shows `Title page`; no hash → `#p000-title`.
- [ ] ← → change the hash; typing `12` + Enter in the jump box goes to `#p012`; `title` goes to the title page.
- [ ] Theme toggle flips `data-theme`, persists across reload; first visit is dark.
- [ ] No console errors; no network requests except `data/*` and `img/*`.
- [ ] `translations/README.md` explains the symlinks, the server command, and the data contract link.

**Verify:** `cd translations/viewer && python3 -m http.server 8765`, then Playwright (page-capture skill) at 1440×900 loading the five URLs above; console clean.

---

### Task 5: Text panel — English blocks with sidenotes, French diplomatic lines, pending states

**Goal:** The right panel renders both layers per the spec with sidenotes aligned to their paragraphs at ≥1100px and inline expansion below.

**Files:**
- Modify: `translations/viewer/app.js` (add `renderText(page, layer)`, `renderEnglishBlock`, `renderFrenchBlock`, `renderNotes`, `bindNoteHover`), `translations/viewer/style.css`

**Rendering rules:**
- English `heading` → `<h2 class="blk-heading">`; `paragraph` → `<div class="para has-notes?"><p class="body">[continued mark]html</p><aside class="notes">…</aside></div>`; `continued` → `<span class="cont" aria-label="continued from previous page">⋯</span>` before the text. Each note: `<div class="note" data-key="a"><sup class="nk">a</sup><span class="cite">citation</span><span class="orig">original</span><span class="gloss">gloss</span></div>`.
- French `heading` → `<h2 class="blk-heading fr" data-spaced>`; `paragraph` → `<div class="para fr"><p class="lines">` one `<span class="ln" data-where="blocks[i].lines[j]">` per line, `{a}` → `<sup class="mk" data-key="a">a</sup>`; `spaced_caps` → class `spaced` (`letter-spacing:.12em`); `ornament` → `<div class="blk-ornament">〔woodcut headpiece〕</div>`; `blank` → `<div class="blk-blank"></div>`; any other type → `<div class="blk blk-{type}">text or lines</div>`. French notes: rendered as a single `<aside class="notes fr">` beside the **first** paragraph containing each key's marker (fall back to the first paragraph), each `<div class="note fr" data-key>` with `<sup class="nk">` and the note lines joined by `<br>`; `kind == "foot"` notes get class `foot`.
- `uncertain[]`: for each entry, the `.ln[data-where]` matching `where` gets class `uncertain` and `title=note` (match on the `blocks[i].lines[j]` form; also accept `blocks[i] line j`).
- Pending: layer null → text panel shows `<p class="pending">Translation pending</p>` or `Transcription in progress`; if both null → `Transcription in progress` and the image alone.
- Layout: `.para.has-notes { display:grid; grid-template-columns: minmax(0,62ch) 18rem; gap:1.5rem; align-items:start }` at ≥1100px; below, `.notes` hidden until a marker is tapped (`.para.open .notes {display:block}` after the paragraph). Hover/focus on `.mk[data-key]` adds `.hl` to the matching `.note[data-key]` in the same `.para` and vice versa.
- Body text: serif, 1.05rem, line-height 1.55, `max-width:62ch`; French lines `white-space:pre-wrap` off, one line per `.ln` via `display:block`, 0.98rem, `hyphens:none`.

**Acceptance Criteria:**
- [ ] `#p004` French: 3 blocks in order, 36 `.ln` in the first paragraph, `<sup class="mk">ſ</sup>` in line 1, notes aside with keys `ſ t u x y z a b` (+ foot notes), spaced-caps heading `TEXTE.`; hovering `ſ` highlights its note.
- [ ] A hand-written English fixture (write `translations/viewer/dev/p004-en.json` — a copy of p004 with a plausible English `english` array containing two paragraphs and three notes — and a `?fixture=` query param that loads it instead of `data/`) renders headings, a `continued` mark, sidenotes aligned to their paragraph top at 1440px, and inline expansion on tap at 390px.
- [ ] `#p007` (pending) shows the image and `Transcription in progress`.
- [ ] `uncertain` lines dotted-underlined with the note as tooltip (p004 has entries; verify one).
- [ ] No horizontal scrolling at 390px.

**Verify:** Playwright screenshots at 1440×900 and 390×844 of `#p004` (fr), `#p004?fixture=1` (en), `#p007`; saved to `/Users/cdavis/github/translator/2026/docs/checks/site-task5-*.png`.

---

### Task 6: Image panel — fit, zoom, pan, prefetch, jump menu

**Goal:** Left panel shows the page image fit to the panel, click toggles 1:1 zoom with mouse/touch pan; next image is prefetched; the jump box gets a datalist of pages with done markers.

**Files:**
- Modify: `translations/viewer/app.js` (`renderImage(page)`, `initZoom(panel)`, `buildJumpList(index)`), `translations/viewer/style.css`

**Behaviour:**
- `<img class="scan" src="img/p004.webp" alt="Page 4 scan" decoding="async">` inside `.image-panel .frame`; `object-fit:contain; width:100%; height:100%` at rest; `.image-panel.zoomed .scan { width:auto; height:auto; max-width:none; transform:translate(x,y) scale(1) }` with the panel `overflow:hidden` and `cursor:grab`. Click (not drag) toggles `.zoomed` centred on the click point; pointer events (`pointerdown/move/up`, `setPointerCapture`) pan; wheel with ctrl/pinch not required. `Escape` un-zooms. Zoom resets on page change.
- While loading, panel background `var(--paper)` and a 0.3s fade-in on `load`. On error, show `Image unavailable` and the CUDL link.
- `prefetch(next)` after the current image loads.
- Jump: `<datalist id="pages">` with `<option value="43">p. 43 · ANNOTATION V ✓</option>` (✓ when the current layer is done); front pages as `title` / `argument`.

**Acceptance Criteria:**
- [ ] `#p004` at 1440: image fills the left panel height without cropping; click zooms to native pixels; drag pans; Escape resets; navigating resets.
- [ ] At 390: image above text, full width, zoom/pan works with touch (Playwright `tap` + touch move).
- [ ] Network shows `img/p005.webp` requested after `p004` loads.
- [ ] Missing image (`#p999` style test: temporarily point to a nonexistent id via the fixture) shows the fallback.

**Verify:** Playwright screenshots `site-task6-*.png` (rest + zoomed, 1440 and 390) into the same `docs/checks/`.

---

### Task 7: Landing page

**Goal:** `translations/landing/index.html` lists the books with progress computed from each book's `index.json`.

**Files:**
- Create: `translations/landing/index.html`, `translations/landing/style.css`

**Content:** `<h1>Translations</h1>`, one paragraph ("Early printed books, transcribed and translated, shown beside the original pages."), a card grid; the Martin Guerre card: thumbnail `martin-guerre/img/p000-title.webp` (shown at 200px wide), title + author/year from `martin-guerre/data/book.json`, one-line description, progress `N of 162 pages transcribed · M translated` computed by fetching `martin-guerre/data/index.json` (fallback text "in progress" if fetch fails), link to `martin-guerre/`. Cards are declared in a small `BOOKS = [{slug:"martin-guerre"}]` array in an inline script so adding a book is one entry. Same tokens as the viewer (copy the `:root` block; a shared file is not worth a third path). Footer link back to `codebycarson.com`. Dark only here is acceptable (matches portfolio) but must set an explicit body background.

**Acceptance Criteria:**
- [ ] Served from an assembled tree (Task 10's assembly step run locally into `/tmp/out`), `http://localhost:8766/` shows the card with the real progress numbers and thumbnail; the card links to `/martin-guerre/`.
- [ ] 390px: single column, no horizontal scroll.

**Verify:** screenshot `site-task7-landing.png`.

---

### Task 8: Browser verification pass and spec checklist

**Goal:** Every acceptance criterion in `docs/site-design.md` §5 is checked in a real browser and evidenced by screenshots; findings fixed.

**Files:**
- Create: `docs/checks/site-final-*.png` (ocr_translation/2026), `docs/checks/site-checklist.md` (one line per spec bullet: pass/fail + screenshot name)
- Modify: viewer files as needed for fixes

**Verify:** `docs/checks/site-checklist.md` has no `fail`; Carson reviews the screenshots before Phase 3 goes to CI.

---

## Phase 3 — Infrastructure and deploy (code-by-carson)

### Task 9: CDK — Translations bucket, distribution, domain, grants, outputs

**Goal:** `npx cdk synth` produces the new resources following the llms block exactly.

**Files:**
- Modify: `cdk/stacks/portfolio_stack.py` (insert after the LLMs block, before `# ── Outputs`), `cdk/cdk.json` (`"attach_translations_domain": "true"`), `INDEX.md` (stack line)

**Code (adapt names; keep the structure of the llms block, lines ~477–577):**
```python
        # ── Translations site (translations.codebycarson.com) ───────
        # Landing page + one viewer per book under /<slug>/; book data is
        # checked out from its own repo by deploy-translations.yml. Page
        # images live in this bucket under /<slug>/img/ and are uploaded
        # once from a workstation (scripts/site_images.py in the book repo);
        # the workflow's sync excludes that prefix.
        attach_translations_domain = self.node.try_get_context("attach_translations_domain") == "true"
        # (add f"translations.{DOMAIN}" to cert_sans under this flag, next to the llms line)
        translations_bucket = s3.Bucket(self, "TranslationsBucket", …same options as LlmsBucket…)
        translations_rewrite_function = cloudfront.Function(self, "TranslationsIndexRewrite", …same code…, comment="Resolve static subpath index files for translations site")
        translations_distribution = cloudfront.Distribution(self, "TranslationsDistribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(translations_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                compress=True,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
                response_headers_policy=cloudfront.ResponseHeadersPolicy.SECURITY_HEADERS,
                function_associations=[…rewrite…]),
            domain_names=[f"translations.{DOMAIN}"] if attach_translations_domain else None,
            certificate=certificate if attach_translations_domain else None,
            default_root_object="index.html",
            minimum_protocol_version=cloudfront.SecurityPolicyProtocol.TLS_V1_2_2021,
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,
            error_responses=[…404/403 → /404.html as llms…])
        # ARecord "TranslationsAlias" under the flag; grant_read_write; CreateInvalidation on this distribution only
        CfnOutput(self, "TranslationsBucketName", value=translations_bucket.bucket_name)
        CfnOutput(self, "TranslationsDistributionId", value=translations_distribution.distribution_id)
```
Note the cert SAN list is built before the certificate; move the `attach_translations_domain` flag read up next to the other two flags. Add a `translations/landing/404.html` (tiny, same style) so the error responses resolve.

**Acceptance Criteria:**
- [ ] `cd cdk && uv venv && uv pip install -r requirements.txt && npx cdk synth` succeeds; the template contains `TranslationsBucket`, `TranslationsDistribution`, `TranslationsAlias`, both outputs, and the cert SAN list includes `translations.codebycarson.com`.
- [ ] `npx cdk diff` (needs AWS creds; Carson runs it) shows only additions plus the certificate replacement.
- [ ] Deploy role policy additions are scoped to the new bucket and the new distribution ARN only.

**Verify:** `npx cdk synth > /dev/null && grep -c Translations cdk.out/CodeByCarsonStack.template.json` > 0.

---

### Task 10: deploy-translations.yml

**Goal:** Push to `master` touching `translations/**` (or manual dispatch) assembles and deploys the site; images untouched.

**Files:**
- Create: `.github/workflows/deploy-translations.yml`, `translations/scripts/assemble.sh`

**assemble.sh** (used by CI and by Task 7 locally): `assemble.sh <book-src-dir> <out>`: copies `translations/landing/*` → `out/`, `translations/viewer/{index.html,app.js,style.css}` → `out/martin-guerre/`, `<book-src>/2026/site/data` → `out/martin-guerre/data`; refuses if `data/index.json` is missing; `find out -type f | sort`.

**Workflow:** copy `deploy-llms.yml` structure: triggers (`paths: ["translations/**", ".github/workflows/deploy-translations.yml"]`, `workflow_dispatch`), `concurrency: translations`, checkout self + `CarsonDavis/ocr_translation@main` into `book-src`, run `translations/scripts/assemble.sh book-src out`, OIDC creds, resolve `TranslationsBucketName`/`TranslationsDistributionId` with the same retry loop, then:
```
aws s3 sync out/ "s3://$BUCKET/" --delete --exclude "*/img/*" --exclude "*.html" --exclude "*.json" --cache-control "public, max-age=3600"
aws s3 sync out/ "s3://$BUCKET/" --delete --exclude "*/img/*" --exclude "*" --include "*.html" --include "*.json" --cache-control "public, max-age=300, must-revalidate"
aws cloudfront create-invalidation --distribution-id "$DIST" --paths "/*"
```
(`--delete` with `--exclude "*/img/*"` leaves the image prefix alone; verify this with `--dryrun` in the acceptance step.)

**Acceptance Criteria:**
- [ ] `bash translations/scripts/assemble.sh ../translator out-test` locally produces `out-test/index.html`, `out-test/martin-guerre/index.html`, `out-test/martin-guerre/data/index.json`, and the listing.
- [ ] `actionlint` (or `gh workflow view` after push) reports no errors; the YAML has no `Co-Authored-By` or AI text.
- [ ] Dry run documented in the workflow comment: `aws s3 sync … --dryrun` shows no `delete:` lines under `martin-guerre/img/` (Carson runs once after the image upload).

---

### Task 11: Documentation and logs

**Goal:** Both repos' docs reflect the work.

**Files:**
- code-by-carson: `INDEX.md` (translations/ tree, workflow, stack line), `docs/IMPLEMENTATION_LOG.md` (dated entry: what, verification, not done), `docs/REQUIREMENTS.md` (new section `R? – Translations site` with checked boxes), `translations/README.md` (from Task 4, extend with "adding a book" = data contract link + assemble line + landing entry + upload command).
- ocr_translation/2026: `docs/pipeline-log.md` (append), `site/README.md` (from Task 3), `docs/site-handoff.md` (append a "Status 2026-09-2x" paragraph pointing at design/plan/contract).

**Acceptance Criteria:**
- [ ] Every new file appears in `INDEX.md`; no AI attribution anywhere (`grep -ri "claude\|generated with\|co-authored" translations .github/workflows/deploy-translations.yml` → nothing).

---

### Task 12: Go-live (Carson, with the orchestrator's checklist)

Not automated. The orchestrator hands Carson this list:
1. Review `git status` in both repos; commit ocr_translation (`2026/site/data`, scripts, docs, `.gitignore`) and code-by-carson (`translations/`, workflow, cdk, docs) — Carson decides messages; no attribution.
2. Push code-by-carson → `deploy.yml` runs `cdk deploy` (cert replacement + new resources) and `deploy-translations.yml` deploys the tree. Watch both in `gh run list`.
3. `uv run python scripts/site_images.py --upload --bucket <TranslationsBucketName> --profile <profile>` from `2026/`.
4. Verify: `curl -I https://translations.codebycarson.com/`, `/martin-guerre/`, `/martin-guerre/data/pages/p004.json`, `/martin-guerre/img/p004.webp` (200, `image/webp`, `immutable`). Open the site on a phone.
5. Blog: in `CarsonDavis.github.io/_posts/2025-03-09-troublesome_translations.md` replace `you can find it here.` with `you can find it [here](https://translations.codebycarson.com/martin-guerre/).` and mention it is a work in progress; commit in that repo.
6. Re-deploy after each data push: `gh workflow run deploy-translations.yml -R CarsonDavis/code-by-carson`.

---

## Self-review

- Spec §2 hosting → Task 9; §3.1 layout/workflow → Tasks 4, 7, 10; §3.2 → Tasks 1–3; §3.3 images → Task 3 + Task 12; §4 contract → Tasks 1–2 (+ `site-data-contract.md`); §5 viewer → Tasks 4–6; landing → Task 7; §6 verification → Tasks 1–3 tests, 8 screenshots, 9 synth, 12 curl; §7 order → phases; blog link → Task 12.
- Names used consistently: `normalize_heading`, `build`, `split_english`, `parse_section`, `renderText`, `renderImage`, `setLayer`, `setTheme`, `TranslationsBucketName`, `TranslationsDistributionId`, `assemble.sh`.
- Open item deliberately left to the pipeline session: producing `translation/sections/*.md` in the Task 2 format. Recorded in `docs/site-data-contract.md` and `docs/pipeline-log.md`.

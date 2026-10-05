# Reader mode: continuous English text

Decided 2026-10-04 after the phone audit (`docs/checks/site-phone-audit.md`). On a phone
the scan is a reference, not the thing being read. Reader mode shows the English as one
scrolling document in print order; the scan is one tap away via page markers and is never
forced on the reader. English only for now; the diplomatic French stays page-bound.

Approach chosen: the pipeline emits per-section JSON (one source of truth, the section is
the natural unit), and the viewer gains a second mode that loads sections lazily.

## 1. Data (ocr_translation/2026, `scripts/split_pages.py`)

`split_pages.py` already parses `translation/sections/*.md` (front matter `section`,
`order`, `heading`; page markers `⟦pNNN⟧`; `## notes ⟦pNNN⟧` blocks). Today it splits the
prose at page markers into `site/data/pages/<id>.json`. It must additionally write, without
changing any existing output:

`site/data/text/toc.json`
```json
{"sections": [
  {"id": "annot-005", "order": 12, "heading": "ANNOTATION V", "kind": "annotation",
   "first_page": "p043", "pages": ["p043", "p044"]}, …]}
```
`kind` is one of `title`, `argument`, `text`, `annotation` (derive from the section id /
heading; `TEXTE.` sections are `text`). Order is `order`.

`site/data/text/<section-id>.json`
```json
{"id": "annot-005", "order": 12, "heading": "ANNOTATION V", "kind": "annotation",
 "pages": ["p043", "p044"],
 "blocks": [
   {"type": "paragraph",
    "html": "…<span class=\"pg\" data-page=\"p043\"></span>English prose with <sup class=\"mk\" data-key=\"a\" data-page=\"p043\">a</sup> …",
    "notes": [{"key": "a", "page": "p043", "citation": "…", "original": "…", "gloss": null}]}
 ]}
```
Rules:
- Paragraphs are whole; they are not split at page breaks. A page marker becomes an empty
  `<span class="pg" data-page="pNNN"></span>` at the exact character position where the
  printed page turns, including mid-sentence. The first marker of a section sits at the
  start of its first paragraph.
- Marker letters restart on every printed page, so `<sup class="mk">` and each note carry
  `data-page` / `page`; the viewer pairs marker and note on (page, key). The `html`
  conversion (`{a}` → sup, `*x*` → `<i>`, escaping) is the same code path as the page
  files; refactor so both outputs share it rather than duplicating.
- Notes attach to the paragraph that contains their marker, in marker order, as now. A
  citation-less "marker with no note" line produces no note, as the contract already says.
- `book.json` gains `"reader": {"base": "text/"}` so the viewer can find it; absent key
  means no reader mode.
- Everything NFC, `ensure_ascii=False`, `indent=1`. Validate every section file against a
  schema (`scripts/site_schema.json` gains a `section` definition or a sibling
  `site_section_schema.json`). Document the format in `docs/site-data-contract.md`.
- Tests in `scripts/tests/test_split.py` with the existing synthetic fixtures: section
  emission, page-marker position inside a paragraph that crosses a page, note `page` keys,
  toc order and kinds, and that the per-page output is byte-identical to before.
- CLI prints an extra line: `N sections written`.

## 2. Viewer (code-by-carson, `translations/viewer/`)

New files `reader.js` and `reader.css` (no build step, no framework, no external requests),
loaded from `index.html` after `app.js` / `style.css`. Keep edits to `app.js` to the hooks
below so the uncommitted cited-sources work in that file is not disturbed. **Add the new
files to `translations/scripts/assemble.sh`**, which copies viewer files explicitly.

Routing (hash):
- `#p043` and friends: page mode, unchanged.
- `#read` → reader at the top; `#read/annot-005` → reader scrolled to that section;
  `#read/p043` → reader scrolled to that page marker.
- No hash: viewports narrower than 900px open reader mode; wider open page mode. The last
  mode chosen is remembered in `localStorage['tc.mode']` and wins over the width rule.
  A deep link to a page id always opens page mode.

Header in reader mode: book title (home link as now), a `Pages | Text` mode toggle (also
shown in page mode), a `Contents` button, the theme toggle. No pager, jump box, image link
or layer tabs. Keep it to one row on a 390px phone.

Contents: a panel (dialog or slide-down) listing the toc: Title, Argument, then Texte and
Annotations in print order, each a link to `#read/<id>`. Annotation entries show the
roman numeral heading; `text` entries show "Text" with the first page number.

Rendering:
- Sections render in toc order into one scrolling column. Load lazily: render the first
  three, then use an IntersectionObserver sentinel to fetch and append the next section
  when the reader is within two screens of the end. Jumping to a section via the toc or a
  `#read/<id>` hash renders everything up to and including that section before scrolling.
  Rendered sections stay in the DOM (the whole book's English is small).
- Each section: `<section class="rd-section" id="sec-annot-005">` with `<h2>` from
  `heading` and the same paragraph/notes markup as page mode so `style.css` applies
  (sidenotes beside paragraphs at ≥1100px, tap-to-expand inline below; marker/note
  highlight pairs on `data-page` + `data-key`). Reuse the existing note rendering
  functions from `app.js` by calling them; do not copy them.
- Page markers: each `.pg` span gets a small margin label `p. 43` (absolute, in the left
  gutter on wide screens; a right-aligned superscript-style label on phones) that is a link
  to `#p043`, title "Show the printed page". Front pages label `Title` / `Argument`.
- As the reader scrolls, update the hash to `#read/<current section>` with
  `history.replaceState` so a reload or a shared link returns to the same place.
- Typography: body 1.05rem / 1.55, max-width 62ch, centred, 16px side gutters on phones;
  no horizontal scroll at 390px. Dark default and light theme both complete.
- `?embed=1` continues to work in both modes; embed mode hides the home link and footer as
  now and keeps the mode toggle. Do not change the blog post.

Page mode, one audit fix only: in the stacked layout (below 900px) cap the scan frame at
`clamp(240px, 52dvh, 560px)` so the text panel starts on the first screen. No other
page-mode changes.

## 3. Verification

- `uv run --with pytest --with jsonschema --with pillow python -m pytest scripts/tests -q`
  passes; real build prints `162 pages written (162 french, 162 english)` and the sections
  line; `git diff --stat site/data/pages` is empty after the rebuild.
- `bash translations/scripts/assemble.sh ../translator <out>` lists `reader.js`,
  `reader.css`, `data/text/toc.json` and section files.
- Playwright (Chrome channel) against the assembled tree served on a 879x port, with
  `img` symlinked to the book repo's `site/img`:
  - iPhone 13 portrait, no hash → reader mode; first screen shows the title and prose;
    `document.documentElement.scrollWidth == clientWidth`; Contents opens and a jump to
    ANNOTATION V lands on its heading; a page-marker tap opens `#p043` in page mode; the
    `Pages | Text` toggle returns to `#read/annot-005`; a marker tap expands its note
    inline and it is visible without scrolling.
  - 1440×900, no hash → page mode; toggle to Text shows sidenotes beside paragraphs; page
    labels in the gutter; scrolling updates the hash.
  - `#read/p100` lands on page 100's marker; `#p100` still opens page mode.
  - No console errors in any run. Screenshots to
    `ocr_translation/2026/docs/checks/site-reader-*.png`.
- Record the work in code-by-carson's `INDEX.md`, `docs/IMPLEMENTATION_LOG.md`,
  `docs/REQUIREMENTS.md`, `translations/README.md`; append a dated section to
  `ocr_translation/2026/docs/pipeline-log.md`.

## 4. Rules

No `git add`/commit/push in any repo. No AI attribution anywhere. Never write
`manifest.json`, `transcription/**`, `translation/**`, `text/**`, `pages/**`. Python via
`uv run --with <deps> python …`. Do not revert or reformat the uncommitted cited-sources
edits in `translations/viewer/app.js`, `style.css`, `dev/README.md`; build on top of them.
Ports 8765 and 8766 are taken by other projects.

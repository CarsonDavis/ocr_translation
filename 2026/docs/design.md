# Coras, *Arrest memorable* (1572): transcription, translation, and side-by-side site

Design document. Written 2026-09-21. Everything for this effort lives under `2026/`; the
legacy pipeline in the repo root is left untouched and is not reused.

## 1. Goal

Produce, for the 1572 Paris edition of Jean de Coras's *Arrest memorable du Parlement de
Tholose* (Cambridge University Library, Montaigne.1.7.22):

1. A **fully diplomatic French transcription** of the title page, the *Argument et
   sommaire du faict* (the one-page summary of the case printed before page 1), and
   printed pages 1–160,
   faithful to the print line by line, verified by independent reads.
2. A **modern English translation** of the whole text, including all 111 annotations and
   every lettered marginal citation, checked for consistency with the known facts of the
   case and the period.
3. A **lightweight static HTML site** showing each page image on the left and its
   translation on the right, with the marginal citations placed as sidenotes at their
   letter markers, and a toggle to show the French instead.

Out of scope for now (may be added later): the royal privilege (image 8), the printer's
notice (9–10), and the alphabetical table (11–21).

## 2. Facts about the source that shape the design

- **CUDL item** `PR-MONTAIGNE-00001-00007-00022`, 185 images. IIIF Image API at
  `https://images.lib.cam.ac.uk/iiif/PR-MONTAIGNE-00001-00007-00022-000-{NNNNN}.jp2/`.
  Native size 2941×4711. Whole-image requests are capped at 2000px, but **regions are
  served at native resolution** if the output is ≤2000×2000, so a full-resolution page is
  six stitched tiles.
- **Image-to-page mapping is not linear.** Images 23–62 are pages 1–40. **Printed page 41
  was never photographed.** From image 63 to 180 each opening was shot recto first:
  odd image *i* is page *i*−20, even image *i* is page *i*−22. Image 182 is page 160.
  Image 7 is the title page. Page 41 comes from Gallica's copy of the same edition
  (`ark:/12148/bpt6k52469j`, a microfilm reproduction) and is flagged as such.
- **Pagination errors in the print** (catalogue note, confirmed): pages 44, 45, 48 are
  printed as 24, 44, 58. The manifest records both the printed folio string and the true
  page number.
- **Three textual layers** on a page: `TEXTE.` blocks (the court record), `ANNOTAT. I`
  through `ANNOTAT. CXI` blocks (Coras's commentary) set in the same column, and
  **lettered marginal citations** (a, b, c … repeating) in the outer margin keyed to
  superscript letters in the body. When the margin overflows, citations continue as a
  small-type block at the foot of the page. Running heads: verso `N ARREST DV`, recto
  `PARLEMENT DE THOLOSE. N`. Signature marks at the foot of some rectos.
- **Prior English translation**: Ringold and Lewis, *TriQuarterly* 55 (1982), about ten
  pages of English covering the `TEXTE` blocks only, from this edition. It is **partial**:
  it does not cover the annotations and may abridge or skip parts of the `TEXTE` itself, so
  its absence of a passage means nothing. Used as a cross-check for meaning where it
  exists, never as a source to copy from. The annotations have no published English
  translation.

## 3. Folder layout

```
2026/
  README.md                     status, how to run each stage, conventions summary
  docs/
    design.md                   this file
    conventions.md              transcription character/markup conventions (section 5)
    case-file.md                narrative, people, places, dates, legal terms, sources
    pipeline-log.md             what ran when, agreement rates, escalations
  manifest.json                 one entry per page: image no., true page, printed folio,
                                side, source (cudl|gallica), status per stage
  scripts/                      small Python scripts (uv run --with ...), one per stage
  raw/                          full-res stitched originals, gitignored
  pages/
    full/p041.jpg               cropped, deskewed, full resolution, gitignored
    read/p041.jpg               cropped, ~1600px wide, committed, used by the site
    margin/p041.jpg             optional: margin-column crop for hard pages
  transcription/
    reads/A/p041.json           independent read A (raw agent output)
    reads/B/p041.json           independent read B
    diff/p041.md                line-level disagreement report
    final/p041.json             reconciled diplomatic master
  text/
    sections.json               the book as an ordered list of logical sections, each
                                with reflowed text and page-break markers
  translation/
    sections/annot-005.md       one file per logical section, English with markers
    review/annot-005.md         reviewer findings and resolution
    pages/p041.json             re-split per page for the site
  site/
    index.html, app.js, style.css, data/pages/p041.json, img -> ../pages/read
```

## 4. Pipeline stages

Each stage has a script or an agent prompt, an input, an output, and a validation step
that must pass before the next stage runs. Status per page is tracked in `manifest.json`.

### Stage 0: Manifest
Build `manifest.json` from the mapping rules in section 2. Validation: 162 entries
(title page, Argument, pages 1–160), page 41 marked `source: gallica`, no duplicate
images.

### Stage 1: Acquire
For each entry, download six IIIF region tiles at native resolution and stitch them into
`raw/imgNNN.jpg`. Fetch page 41 from Gallica at the best size available. Validation:
every raw image exists and is 2941×4711 (Gallica excepted); a contact sheet of the top
strips of all pages is generated and an agent reads every folio number and compares it to
the manifest. Any mismatch stops the pipeline.

### Stage 2: Crop (decided on the pilot, may be reduced to a fixed trim)
The scans come from one capture rig with a black backdrop, so the page position should be
fairly stable, but this is checked rather than assumed. On the pilot pages, measure the
detected paper rectangle across all 161 raw images and look at the spread. If it is tight,
use a single fixed trim (safe and predictable). If it varies, use per-page detection with
a generous margin so nothing can be clipped. Either way the copyright band is removed and
`pages/full/` and `pages/read/` are written. Validation: an agent reviews a contact sheet
of all cropped pages for clipped text or leftover backdrop and lists pages to redo by hand.
If cropping proves unreliable, the fallback is to serve the raw images uncropped; nothing
downstream depends on the crop.

### Stage 3: Transcribe (the core)
Two **independent** reads of each page by Opus agents given the full-resolution page
image, the conventions document, the manifest entry, and **the final transcriptions of
the previous two or three pages** so the reader knows the sentence and section it is
continuing. Both reads get the same context, so their independence is preserved. Pages
are therefore processed in order, in waves, each wave receiving the finalized output of
the one before. Output is JSON (section 5).
Then a **line-level diff**. Pages with zero disagreements are accepted. Pages with
disagreements go to a Fable reconciliation agent that sees both reads and the image and
must decide every disputed line, citing what it sees. Lines it cannot decide are marked
`uncertain` and escalated to me; I read those crops personally in the main session.

Hard pages get a **margin-column crop** as a second image so citations are read at
the largest possible scale.

**Pilot first, with a model decision at the end.** Before scaling, run the full stage
on five pages of graded difficulty (a `TEXTE` page, a dense annotation page, a page with
a foot-of-page citation block, a page with a misprinted folio, and the title page).
Measure the agreement rate between reads A and B, and compare the reconciled result
line by line against **my own reading of every pilot page** (Fable, main session).
Adjust the conventions and prompts until the pilot is clean. Then decide the reader
model: if the Opus reads are not accurate enough after tuning, **Fable agents do the
reads instead** and Opus is dropped from this stage. Accuracy is the priority over cost.

**Ongoing spot checks.** After the pilot, a Fable agent independently re-reads a random
sample of at least one page in ten of the accepted pages, plus every page the reconciler
touched heavily, and compares against the final. Any error found triggers a re-read of
the neighbouring pages and is logged in `pipeline-log.md`. If the sample error rate is
not near zero, the reader model is switched to Fable for the remaining pages.

### Stage 4: Stitch
Derive `text/sections.json` from the final transcriptions: split the book into ordered
logical sections (`TEXTE` blocks and numbered annotations), reflow the diplomatic lines
into paragraphs, resolve end-of-line hyphenation, and insert `⟦p.43⟧` page-break markers.
This derivation is a **script**, not an agent, so it is repeatable and testable. The
fully diplomatic master is never altered; the reflowed form is a computed view.
Validation: every page appears exactly once, in order; every annotation number I–CXI
appears exactly once; every letter marker in the body has a matching marginal note on
the same page (or the note is explicitly recorded as absent).

If testing shows a **semi-diplomatic layer** (long-s resolved, abbreviations expanded)
measurably improves translation quality, it is added here as a second computed view,
with the expansion rules written down and spot-checked. It is not built by default.

### Stage 5: Case file
Before any translation, a research agent writes `docs/case-file.md`: the sequence of
events of the Martin Guerre affair with dates, the people and their relationships, the
places, the court procedure of a sixteenth-century parlement, the legal sources Coras
cites (Digest, Code, Institutes, glossators, canon law) and how citations are abbreviated,
and the period's units, money, and calendar. Sources: Natalie Zemon Davis, Ringold and
Lewis, the Argument page, and standard references. Every translation and review agent
receives this file. It is updated as the translation surfaces new facts.

### Stage 6: Translate
One Fable agent per logical section, given: the section's reflowed French with page
markers, its marginal citations, the case file, the conventions for English output, and
**the last few pages of finished French and English** (the preceding sections, both
languages) so terminology, names, and tone stay continuous. Sections are therefore
translated in order. Output: English prose that keeps every
`⟦p.N⟧` marker and every letter marker in place, plus a translated and **expanded**
version of each marginal citation (for example, `l. minorem D. de ritu nupt.` becomes
"Digest 23.2, *De ritu nuptiarum*, lex *Minorem*", with a one-line gloss of what the
law says when that is recoverable). Validation: markers in the output match the input
exactly in count and order.

### Stage 7: Review
Three reviewer passes per section, each by a fresh agent with the French and English side
by side: (a) fidelity, which hunts omissions, additions, and mistranslations sentence by
sentence; (b) consistency against the case file (names, dates, places, titles, legal
terms, and the running glossary); (c) for `TEXTE` sections only, a meaning comparison
against Ringold and Lewis that flags divergences for a human decision. Findings are
written to `translation/review/`, applied by a fixer agent, and re-checked.

### Stage 8: Re-split and build site
A script splits each section's English at its page markers into `translation/pages/`,
attaches the sidenotes to their page, and emits `site/data/pages/pNNN.json`. The site is
plain HTML, CSS, and JavaScript with no build step and no framework.

## 5. Transcription conventions (summary; full text in `conventions.md`)

The master is **fully diplomatic and line-faithful**:

- One JSON string per printed line, in print order, inside each block. Line-end hyphens
  kept as printed.
- Long s is `ſ`. `u`/`v` and `i`/`j` exactly as printed. Ligatures `æ`, `œ` kept.
  Tilde abbreviations kept with the tilde (`ẽ`, `õ`, `ã`, `q̃`). `&` kept. Other
  abbreviation signs recorded with an agreed Unicode character (list in `conventions.md`).
- Spaced capitals (`P R AE`) transcribed with the spaces, marked `"spaced": true`.
- Inline letter markers written as `{a}` at the exact position in the line.
- Characters the reader cannot resolve are `[?]` with a note; damaged text is `[...]`.
- Blocks: `running_head`, `folio` (as printed), `heading` (`TEXTE.`, `ANNOTAT. V.`),
  `paragraph` (with `continues_prev`/`continues_next` flags), `margin_note` (key, lines,
  and the body line it sits beside), `foot_note`, `signature`, `catchword`, `ornament`.

## 6. Site design

- One `index.html`. Hash routing `#p43`. Prev/next, jump-to-page, keyboard arrows.
- Left panel: the reading-size page image with click-to-zoom and drag to pan.
- Right panel: the page's English, with headings (`TEXT`, `ANNOTATION V`) and paragraphs.
  Letter markers are superscripts; each has its sidenote rendered in a narrow column to
  the right of the paragraph on wide screens and as a tap-to-expand note on phones.
  A page that begins mid-sentence shows a faint "continued" mark.
- Toggle: English / diplomatic French (line-faithful, so it can be read against the image).
- Header shows true page number, printed folio if different, image number, and source
  (CUDL or Gallica), with a link to the CUDL viewer. Footer carries the CC BY-NC credit.
- Data is one small JSON per page, loaded on demand, so the site opens instantly.
- Page images are **not** committed to git. They will be uploaded to S3 and served through
  CloudFront (cheap, cached); the site references them by URL, with a local `img/` fallback
  for development. Going live on codebycarson happens only after the translation is done.

## 7. Agents and models

- Page reads: Opus, two per page, run in parallel batches of about eight pages.
- Reconciliation: Fable, but only on pages with disagreements, and only on the disputed and
  uncertain lines (agreed lines are not re-read). Pages with full agreement are finalized
  by script. Fable token use is kept to the minimum that accuracy requires.
- Translation, review: Fable agents, one section each.
- Pilot verification and escalations: the main session (me), reading page crops directly.
- Spot checks of accepted pages: Fable agents, independent of the original readers.
- All agent prompts are saved under `2026/scripts/prompts/` so runs are reproducible.
- Every agent writes only to its own output path and never edits the master files.

## 8. Verification and honesty rules

- No stage is marked done in `manifest.json` until its validation has run and passed.
- Agreement rates, escalation counts, and pages redone are logged in `pipeline-log.md`.
- The site is checked in a real browser on at least the pilot pages before any claim that
  the alignment works.
- Where the text is uncertain, the uncertainty is shown, not hidden: `[?]` in French,
  and a bracketed note in English.

## 9. Housekeeping outside `2026/` (proposed, needs your go-ahead)

- Local `main` is three commits behind `origin/main` (a contributor's typo PR). Fast-forward it.
- `.env` in the repo root holds live API keys. Rotate them if this repo is ever shared.
- Two `example_data/2_ocr/*.md` files have uncommitted edits; leave them as they are.
- Add `2026/raw/` and `2026/pages/full/` to `.gitignore`.

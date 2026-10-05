# Playbook: transcribing and translating an early printed book with model agents

Written 2026-10-03 from the Coras *Arrest memorable* (1572) project, after the method had
settled. This is the repeatable version: what to do, in what order, with which model, and what
we learned not to do. Project-specific numbers are given as calibration, not as targets.

The pipeline turns page images of one printed book into (1) a diplomatic transcription in
JSON, one file per page, (2) a stitched, sectioned source text, (3) a complete translation with
every marginal citation identified, and (4) a static side-by-side site. A human sets the rules,
reads the reports, and commits. Human time spent on individual readings is the thing this
method minimises.

## Running this on a new book

`PIPELINE.md` at the repo root is the entry point for a coordinator session: what to ask
the human for, `scripts/new_book.py` to scaffold a book root (`book.json`, manifest,
templated `docs/conventions.md` and `docs/case-file.md`, a `scripts` symlink to this
toolkit), and every stage with its exact agent dispatch text, caps and checks. The
scripts find the book from `$BOOK_ROOT` or the nearest `book.json` (`2026/book.json` is
the Coras one).

## 0. Roles and cost rules

- **Coordinator** (the session you talk to): plans, dispatches agents, reads their three-line
  summaries, runs the scripts, commits. It never reads page images or long files itself and
  never writes code beyond one-liners. Its context is the scarcest resource; the first project
  burned a coordinator to 937k tokens before this rule existed.
- **Readers** (Opus): one page, one pass, no cropping or zooming. Cheap and good enough when
  two of them read independently.
- **Translator and reviewer** (the strongest model available; Fable here): the hard reading
  comprehension lives here. Translation batches of about 12 sections; review as whole-book
  passes.
- **Mechanical helpers** (Opus): code, scripts, data migrations, mechanical sweeps, image
  checks of specific words. Every Opus prompt says "do not spawn sub-agents": left to itself
  one sweep fanned out to six.
- **Cost model that held**: an agent's cost is dominated by what it must read in, not by how
  many edits it makes. A whole-book reviewer cost ~520k tokens whether it changed 2 sections
  or 26. Design agents around read-in: read each input once, in the largest chunks the tool
  allows, never re-read, write outputs in a few large writes.
- Watch running agents without reading their transcripts: count tool calls and the last
  `cache_read_input_tokens` in the transcript file with `grep -c` / `grep -o`. A reader at 20
  calls, a reviewer at 30, is normal. A reader at 70 is cropping.

## 1. Set up the book

1. Page images into `pages/`, with a `manifest.json` listing page ids in reading order and
   their stage flags. Expect the scan order to be wrong somewhere; fix the manifest, not the
   files.
2. Cut each page into native **strips** (`pages/strips/<page>/body-N.jpg`, margin strips,
   foot strips) at the scan's own resolution. Readers read strips, never whole pages and
   never crops they make themselves.
3. Write the three documents the agents depend on, before any model touches text:
   - `docs/conventions.md`: how the diplomatic transcription records the print (long s,
     u/v, i/j, tildes, ligatures, `{x}` for a citation marker, how wrong sorts are kept as
     printed and noted as *sic*).
   - `docs/case-file.md`: the story, people with the author's spellings, places, dates, the
     procedure glossary, citation conventions (how `l. minorem D. de ritu nup.` expands),
     classical sources, a **running glossary table** (French → English, section id), and a
     **review decision log** section at the end. This is the shared memory of every agent.
   - `scripts/page_schema.json`: the per-page JSON schema. Every reading, decision and
     uncertainty gets a `where` pointer and a provenance field `by`.
4. Tests from day one (`scripts/tests/`, run with `uv run --with pytest … -m pytest`). The
   data migrations later in the project were only safe because they were tested on fixtures.

## 2. Transcribe: two single-pass reads per page

```
uv run python scripts/wave.py status                 # per-page reads / queue / decided / final
uv run python scripts/wave.py next --size 12         # render read_single prompts for the next pages
#   one Opus agent per prompt, 12 concurrent, backfill as they return:
#   "Read the file <prompt path> and carry out the task it describes exactly, including where to
#    save your output and the hard rules on tools (no image cropping, zooming or processing of any
#    kind; read each image once). Work from <repo>/2026. Do not run git commands. Return only the
#    three-line summary the prompt asks for."
uv run python scripts/wave.py queue                  # normalize spacing, auto-resolve, diff A vs B, build items
```

- Prompt: `scripts/prompts/read_single.md`. Two readers (A, B) per page, same prompt,
  independent. Each writes `transcription/reads/A|B/<page>.json` and validates it.
- Calibration: ~20 tool calls, 70–80k tokens, 1–3 minutes, ≈$0.66 per read. Agreement
  between reads 94–100%. The first project's zoom-everything readers cost ten times that for
  the same accuracy with a different error profile.
- `wave.py queue` writes `transcription/arbitration/queue/<page>.json`: one item per
  disagreement, by kind (body word, note word, note-structure, blocks, running head), with
  crops for a human tool. Items where both readers agree but flagged doubt are hidden by
  default.

## 3. Resolve disagreements: defer to the translator, keep provenance

Human arbitration of fine-glyph calls (period/comma, h/b, i/l) was the project's bottleneck
and added little over a translator choosing from context. Replace it:

```
uv run python scripts/wave.py defer          # every undecided item -> either (word items) / unknown (structural), by:auto
uv run python scripts/wave.py apply          # finals for fully decided pages, manifest sync, stitch
```

- An `either` decision keeps reading A in the final and records B; the stitch turns it into an
  inline marker `wordA⟨alt:wordB⟩` that the translator sees. `unknown` does the same with
  `⟨alt?:…⟩` and an escalation flag.
- Every decision carries `by`: `auto`, `translator`, `reviewer`, or `carson` (the human).
  The arbitration tool (`scripts/arbitrate_server.py --port 8766`, page at
  `http://127.0.0.1:8766/`) stays usable to override any of them later; it labels who decided.
- Expect validation failures the first time on pages where a margin note has no marker: the
  reader-level uncertainty note must survive into the final (fixed in `merge_uncertain`).
- Known errors in already-finalised pages (wrong sorts the earlier pipeline normalised) are
  fixed in the final with a `decisions[]` + `uncertain[]` sic entry, never silently.

## 4. Stitch into sections

```
uv run python scripts/stitch_text.py         # text/sections.json + text/alts.json
```

- Sections are cut at headings (`TEXTE.` / `ANNOTAT. N.`) and numbered by running count, not
  by the printed numeral, because printed numerals are misprinted. **Heading detection must
  tolerate one-letter misprints** (`TFXTE.`, `ANNNT. LX.`); the first cut missed four and every
  id after p045 was off by one to three. Scan all heading blocks for near-misses before
  translating anything.
- The last section must close at end of manifest, not at the next heading (there is none).
- `text/alts.json` lists every inline alt marker with a page-based `alt_id`, so choices made
  against one cut survive a re-cut.
- If a re-cut renumbers sections after translation has started,
  `scripts/migrate_section_ids.py --old <saved sections.json> --new text/sections.json --apply`
  renames and splits the translation files. Save the old `sections.json` first.

## 5. Translate in batches, feed choices back

```
uv run python scripts/render_translate.py --batch <first-id> --size 12 > <scratch>/translate-<id>.md
#   one Fable agent per batch: "Read the file <prompt path> and carry out the task it describes exactly …
#   write ONLY your sections, your batch report and your alt-choices file; do NOT edit docs/case-file.md:
#   put glossary additions in the report."
uv run python scripts/apply_translator_choices.py --dry-run translation/alt-choices/*.json
uv run python scripts/apply_translator_choices.py translation/alt-choices/*.json   # by:translator, refinalize, restitch
```

- Prompt: `scripts/prompts/translate.md`. Output per section: front matter, English prose
  with every `⟦pNNN⟧` page marker and `{x}` citation marker in place, then `## Notes` with
  every citation expanded and identified. `scripts/check_markers.py <id>` enforces the
  markers; the translator runs it per section.
- The prompt embeds the batch's alt table; the translator writes
  `translation/alt-choices/<batch>.json` (`alt_id`, `choice` A/B/either, reason). Apply all
  batches' choices in one pass after translation, not between parallel batches: refinalising
  shifts line indices.
- Run batches in parallel (6 at a time worked); they write only their own files. Glossary
  rows go into the batch reports and are merged into the case file once, by one agent,
  because parallel appends clobber each other.
- Calibration: 12 sections ≈ 180–240k tokens, 15–35 minutes, ≈$0.37 per section. The whole
  book (225 sections) took 11 batches.

## 6. Review: whole-book passes, in sequence, emitting plans

```
uv run python scripts/render_review.py --out <scratch>/book-render.md --quarters 4   # whole book, no Notes, ~184k tokens
#   fill scripts/prompts/review.md placeholders (PASS_NAME, PASS_SCOPE, RENDER_PATH, PLAN_PATH,
#   ALT_CHOICES_PATH, MISREADINGS_PATH, REPORT_PATH); one Fable agent per pass, one pass at a time
uv run python scripts/apply_review_plan.py translation/review/plan-<pass>.json           # dry-run
uv run python scripts/apply_review_plan.py --apply translation/review/plan-<pass>.json   # edits, findings, decision log, check_markers
#   re-render, next pass
uv run python scripts/apply_translator_choices.py --by reviewer translation/alt-choices/review-*.json
```

- Each reviewer reads the **whole book** (French, French notes, English prose) plus the case
  file, then works one scope. Whole-book knowledge is what catches inconsistency; a reviewer
  with a quarter of the book can only check against the glossary.
- Passes run **sequentially** so each inherits the previous pass's glossary decisions from
  the case file's decision log. Order that worked: pass 0 consistency and glossary over the
  whole book, then four quarter passes doing fidelity + consistency + comparison with the
  prior published translation (meaning only, never wording; it is copyrighted).
- Reviewers **do not edit**. They emit a plan: exact-substring `old → new` replacements per
  section and per case-file entry, decision-log lines, findings, uncertain items. The applier
  requires each `old` to occur exactly once, protects the marker sequence, writes per-section
  findings files, and runs `check_markers`. Zero misses across 50 entries in this project.
- Transcription doubts from reviewers go to two files: alt flips (`alt-choices`, applied with
  `--by reviewer`) and suspected misreadings. The misreadings go to an Opus **page-image
  pass** that reads the native strips once and decides. In this project the print agreed with
  the transcription in all 46 cases: reviewers see sense, readers see glyphs, and the print has
  wrong sorts. Record those as *sic* notes; render the intended sense in English.
- Finish with an Opus **mechanical sweep** (sentence alignment, numbers, names, Latin
  quotations, negations, paragraph counts, and a diff review of what the review passes
  changed). It caught two negations a review pass had reversed. Its confident fixes go through
  the same plan applier; its judgment items go to a narrow fixer with only those sections.
- Calibration: each whole-book Fable pass ≈ 520–530k tokens, ~30 tool calls, 15–35 minutes.
  One pass stalled during read-in and resumed cleanly from its transcript when messaged.

## 7. Build the site

```
uv run python scripts/split_pages.py         # site/data/pages/pNNN.json from finals + translations
```

- The French pane shows contested readings inline (dotted underline, popover with both
  readings and who decided), a per-page Readings list, and a header toggle. Provenance labels
  are honest: editor, reconciliation model, translation model, review model, undecided.
- The viewer is a static page with no build step. Keep its data contract in
  `docs/site-data-contract.md` and change the two together.

## 8. Commit and record

- Commit at every clean checkpoint (checks pass, tests pass): after each wave, each apply,
  each review pass. Author is the human; no model attribution anywhere.
- `git diff --cached --name-only` before every commit; image files never enter git.
- Keep `docs/pipeline-log.md` (dated sections, what changed and why) and `docs/handoff.md`
  (state table, runbook, open items) current. A fresh session reads the handoff first.
- Before pushing to a public remote, check what reference material is in the tree.

## 9. What we would not do again

- Zoomed, per-word readers and a model reconciler. Two cheap single-pass reads plus
  deferral to the translator gave the same accuracy at a tenth of the cost.
- Human arbitration as a gate on the pipeline. Keep the tool for overrides.
- Sizing review agents by the translation batch size. Review input is small and output is
  small; whole-book context is what matters.
- Letting review agents edit directly when they hold a huge context. Plan emission costs the
  same and gives a dry-run, a diff, and a record.
- Trusting heading detection on a diplomatic transcription without scanning for misprints.
- Parallel agents appending to the same file.
- Opus agents without "do not spawn sub-agents".

## 10. Order of operations, compressed

1. Images → strips → manifest → conventions, case file, schema, tests.
2. Reads A and B for every page (Opus, single pass, 12 concurrent). Queue.
3. Defer all disagreements; apply; validate; fix any final that will not validate.
4. Scan headings for misprints; stitch; confirm section count matches the book.
5. Translate in 12-section batches (Fable, parallel). Merge glossary rows once. Apply all alt
   choices in one pass.
6. Review: pass 0 glossary, then quarters, sequential, plan mode. Apply reviewer alt flips.
   Page-image pass on misreadings. Mechanical sweep. Narrow fixer.
7. Build site. Update log and handoff. Commit. Decide on push and go-live.

## 11. Link citations to their sources

Optional, after review: make each note citation open the cited passage beside the page.

- **Verify the identifications first.** The translators' note citations are partly guessed.
  Web agents check the uncertain ones against the source texts (a quarter each), then one
  residue pass resolves or marks the rest explicitly unidentified.
- **Contract first** (`docs/sources-contract.md`): a corpus index with edition, licence and
  attribution; one small JSON file per unit; a passage scheme per corpus; a citations file
  keyed by page and marker with a status (passage, unit, work, scan, none).
- **One fetcher per corpus, in parallel**, each caching raw downloads so it rebuilds offline.
- **Locator**: parse the translators' citation forms, adapt each corpus's numbering scheme,
  add concordances where editions number differently, write a coverage report of why refs
  stop short of a passage. A corpus still being fetched can be treated as absent.
- **Incipit pass**: laws cited by opening words need a separate lookup against the corpus.
- **Viewer pane**: affordance only on cited sidenotes; third column on wide screens, bottom
  sheet on phones; highlight the passage range; credit every corpus.
- **Open text exists** for Roman law (Grenoble droitromain site), the Vulgate, most classical
  authors (Perseus, The Latin Library), the Decretals (Bibliotheca Augustana) and the Decretum
  (MGH/BSB Friedberg); Sext and Clementines only as archive.org OCR. Medieval and early modern
  **commentaries and humanist works have no usable open text**: scan links or nothing.

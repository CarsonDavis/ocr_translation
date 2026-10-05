# PIPELINE: running the method on a new book

For a Claude Code **coordinator** session. A human (Carson) hands you page scans of an
early printed book that has the Coras shape: a main text cut into sections by a printed
heading, numbered annotations with their own headings, and lettered marginal citations
(mostly Latin). You run the pipeline end to end by dispatching agents and running
scripts. The method, its reasons and its calibration numbers are in
`2026/docs/playbook.md`; read it once, fully, before you start. `2026/docs/handoff.md`
and `2026/docs/pipeline-log.md` are the Coras run's runbook and log, for reference only.
Ask the human only for the book-specific facts in §1; everything else is decided below.

The toolkit is `2026/scripts/` (the Coras book's own directory doubles as the toolkit
home). Every script works on the **book root**: `$BOOK_ROOT` if set, else the current
directory or nearest parent holding a `book.json`, else `2026/`. A new book gets a
`scripts` symlink to the toolkit, so you always `cd` to the book root and run
`uv run python scripts/<name>.py …` exactly as the Coras runbook does.

## 0. Rules for you, the coordinator (not negotiable)

- **You never read page images, strips, crops or contact sheets**, and you never read a
  long file (a final, `text/sections.json`, a render, a section translation, a report
  longer than a screen). Agents read; you read their three-line summaries and the last
  lines of script output (`| tail -5`). The first project's coordinator reached 937k
  tokens before this rule existed.
- **You never write code** beyond one-liners. Script changes go to an Opus agent with a
  precise spec and "add a test".
- **Every Opus prompt says "Do not spawn sub-agents."** (Left to itself one sweep fanned
  out to six.) Put it in every Fable prompt too.
- **Agents never run git.** Only you do, and only `git status`, `git diff`, `git add`,
  `git commit`.
- **Commit at every checkpoint** named below (checks pass, tests pass). Propose the
  commit to Carson and wait for his go-ahead, unless he has given a standing go-ahead
  for this run. Before every commit: `git status` and `git diff --cached --name-only`;
  no image file ever enters git; author is Carson; **no AI attribution anywhere** (no
  Co-Authored-By, no "Generated with", in commits, PRs, docs or code).
- **Stop dispatching the moment Carson says so** (usage limits). Leave the manifest and
  `docs/handoff.md` in a state a fresh session can resume from.
- Keep `docs/pipeline-log.md` (dated sections: what ran, result, why anything changed)
  and `docs/handoff.md` (state table, open items) current at each checkpoint, via a short
  Opus agent if the edit is more than a few lines.

### Models and concurrency caps

| agent | model (`Agent` tool `model`) | at a time |
|---|---|---|
| reader | `opus` | 12; backfill as they return; a dropped agent is resumed (SendMessage), not respawned |
| translator | `fable` | 6 batches in parallel |
| reviewer | `fable` | 1; passes strictly in sequence |
| setup, glossary merge, image check, sweep, code | `opus` | 1–2 |
| narrow fixer | `fable` | 1 |

Above ~12 concurrent agents the API drops connections (ECONNRESET, 529); the harness caps
at 20.

### Watching agents without reading them

Transcripts live in `~/.claude-mine/projects/<cwd-slug>/<session-id>/subagents/agent-*.jsonl`
(`~/.claude/…` on a default install; `<cwd-slug>` is the session's start directory with
`/` → `-`, e.g. `-Users-cdavis-github-translator`). Never open them; count:

```
T=$(ls -td ~/.claude-mine/projects/-Users-cdavis-github-translator/*/subagents | head -1)
for f in $(ls -t $T/agent-*.jsonl | head -15); do
  printf '%s calls=%s ctx=%s\n' "$(basename $f .jsonl)" "$(grep -c '"type":"tool_use"' $f)" \
    "$(grep -o '"cache_read_input_tokens":[0-9]*' $f | tail -1 | cut -d: -f2)"; done
```

Normal: reader ~20 calls / 70–80k context; translator batch 180–240k; whole-book reviewer
~30 calls / ~520k. A reader at 70 calls is cropping: stop it and re-dispatch. A reviewer
whose call count stops rising for 10 minutes during read-in has stalled: message it to
continue (it resumes cleanly).

## 1. What to get from the human

1. **Scans**: one image per page at the scan's native resolution, in a folder, file names
   sorting in scan order (order errors are expected and fixed in the manifest, never by
   renaming files). Where they came from: holding library, item URL, licence, and how to
   build a per-page URL.
2. **A short description of the book**: author, title as printed, year, edition (place:
   printer, year), language; the printed section headings (the word the main-text
   sections start with, e.g. `TEXTE.`, and the annotation heading forms, e.g. `ANNOTAT.
   V.` / `ANNOT. XII.` / `ANNOTATION I.`); how many annotations if known; front matter
   (title page, argument, dedication) and back matter.
3. **The two reference documents**, if they exist: a prior published translation (saved
   as text under `docs/reference/`; the reviewers compare meaning against it, never
   wording; check its copyright before any public push) and the best scholarly
   background on the book (for the case file). If the human instead hands you drafts of
   `docs/conventions.md` and `docs/case-file.md`, use them in place of the templates in
   §2.

## 2. Scaffold the book

From the repo root (`~/github/translator`), with the slug as the directory name:

```
mkdir -p <slug>/raw && cp <scan folder>/* <slug>/raw/      # images stay out of git (.gitignore)
uv run python 2026/scripts/new_book.py <slug> --title "<title>" --author "<author>" \
    --year <year> --edition "<place: printer, year>" [--short-title "<short>"] [--language French]
cd <slug>
export BOOK_SCRATCH=<your session scratchpad>/waves         # where rendered prompts go
```

This makes `book.json`, `manifest.json` (one record per image in `raw/`, ids `p001…`,
every stage pending, `page`/`folio`/`side`/`url` empty), `docs/conventions.md` and
`docs/case-file.md` from `2026/scripts/templates/`, stub `docs/pipeline-log.md` and
`docs/handoff.md`, `prompts/` (copies of `read_single.md`, `translate.md`, `review.md`
to adapt), `hyphen_keep.txt`, `.gitignore`, the stage directories and the `scripts`
symlink. Then edit `book.json` yourself (it is short): `headings` (the printed heading
words; one-letter misprints are tolerated automatically), `front_matter` (page id →
section id for pages that are a section by themselves; page ids must be `pNNN` or
`p000-<lowercase>`), `description`, and the `site` block's TODOs.

Checkpoint: `uv run --with pytest --with jsonschema --with pillow python -m pytest scripts/tests -q`
passes; commit the scaffold (no images).

## 3. Set up (before any model reads text)

Dispatch these as Opus agents, one at a time unless noted. Each dispatch text below is
exact; fill the `<…>`.

**3a. Strips.** Run `uv run --with pillow,numpy python scripts/crop.py` then
`… scripts/crop.py --sheets`. It writes `pages/read/`, `pages/strips/<id>/body-N.jpg`,
`margin-N.jpg`, `foot.jpg` and contact sheets `docs/checks/crop-sheet-NN.jpg`. Its
paper/column thresholds were tuned on the Coras CUDL scans (black backdrop, 2941×4711),
so have them checked:

> Look at every contact sheet in `<book root>/docs/checks/crop-sheet-*.jpg` (each once, no
> cropping or image processing). For each page say whether the body strips cover the whole
> text column, the margin strips cover the marginal notes, and the foot strip covers the
> foot of the page. Write `<book root>/docs/checks/crop-review.md`: one line per page with a
> problem (page id, what is cut off), and for each a proposed `crop_override` box
> `[x0, y0, x1, y1]` in raw-image pixels if you can estimate one. Do not edit any other
> file. Do not run git commands. Do not spawn sub-agents. Return three lines: pages
> checked, pages with problems, the worst problem.

Problems go to a code agent (`crop_override` per manifest record, or threshold changes in
`crop.py` with a test); re-run crop on those pages (`--only p012,p013`).

**3b. Manifest.**

> Work from `<book root>`. Do not run git commands. Do not spawn sub-agents. For every
> record in `manifest.json` (in order), look once at `pages/read/<id>.jpg` (no cropping or
> processing) and fill: `page` (the true page number in reading order, null for
> unnumbered front matter), `folio` (the page number exactly as printed, as a string, even
> if misprinted; null if none), `side` (`recto`/`verso`), and `url` (built as: <rule from
> the human>). Rename front-matter records to `p000-<name>` (title page = `p000-title`) and
> number the rest `pNNN` by true page; reorder records into reading order where the scans
> are out of order (records only; never rename image files; keep each record's `raw`). List
> missing pages. Write the manifest with one Write call, then report in
> `docs/checks/manifest-review.md`: reorderings, misprinted folios, missing pages. Return
> three lines: pages, reorderings, missing pages.

If pages are missing, ask Carson for a second source (Coras p041 came from Gallica).

**3c. Conventions.**

> Work from `<book root>`. Do not run git commands. Do not spawn sub-agents. Fill in
> `docs/conventions.md`, a template whose structure and MECHANICAL rules must stay. Read it,
> then `2026/docs/conventions.md` (the finished Coras version, for the level of detail; copy
> no facts from it). Look once each at `pages/read/` images of six varied pages (the title
> page, two body pages with margin notes, a page with a section heading, a page with a foot
> block, the last page) and the body and margin strips of two of them. Replace every TODO
> with this print's evidence, quoting real lines with page ids; add any sign the print uses
> that the table lacks; write §9's worked example from one page. Remove the template comment.
> Write the file in one Write call. Return three lines: rules changed from the template,
> signs added, anything a human must decide.

**3d. Case file.** (Opus or Fable, with web access.)

> Work from `<book root>`. Do not run git commands. Do not spawn sub-agents. Fill in
> `docs/case-file.md`, a template: keep every numbered heading exactly (the prompts cite
> §1–§12 and a script appends under "## 12. Review decision log"). Read
> `2026/docs/case-file.md` sections 1, 3, 6 and 9 for the level of detail (copy no facts).
> Sources: this description of the book: <description>; <reference documents>; the web.
> Mark everything you could not verify `[unverified]`; never supply a citation locus you
> did not check. Seed §9 with the key words of the title and the book's technical terms.
> Write in at most four Write calls. Return three lines: sections filled, sources used,
> the biggest gaps.

**3e. Prompts.**

> Work from `<book root>`. Do not run git commands. Do not spawn sub-agents. The files in
> `prompts/` are the Coras agent prompts. Adapt only their book-specific sentences to this
> book (<author>, *<title>*, <edition>, <short description>), using `docs/case-file.md` and
> `docs/conventions.md`; change nothing about procedure, tool rules, output formats,
> placeholders (`{…}`) or the report formats. The book-specific sentences are:
> `read_single.md` the opening description (lines 3–5) and the three example error classes
> in Procedure step 5 (replace the example words with this print's, from
> `docs/conventions.md`, or drop an example that does not apply); `translate.md` the
> opening paragraph (book, author's role, number of annotations, "first English
> translation" only if true), the "Names and places" rule's example list, and the heading
> rule (`TEXTE.` → `TEXT`, `ANNOTAT. V.` → `ANNOTATION V`) if this book's headings differ;
> `review.md` the opening paragraph, the case-file section list if any section is "Not
> applicable", and item 3 plus input 4 (the prior translation: point them at
> `docs/reference/<file>` or delete them if there is none). Return three lines: files
> changed, sentences changed, anything left Coras-specific on purpose.

Note: `scripts/validate_page.py` warns on French words set without long s
(`LONG_S_WORDS`); for a book in another language have a code agent make that list
configurable before reads start.

Checkpoint: tests pass; commit (manifest, docs, prompts, book.json; no images).

## 4. Reads: two single-pass Opus reads per page

```
uv run python scripts/wave.py status
uv run python scripts/wave.py next --size 12      # prints: page reader prompt-path
```

One Opus agent per printed line, exactly:

> Read the file `<prompt path>` and carry out the task it describes exactly, including
> where to save your output and the hard rules on tools (no image cropping, zooming or
> processing of any kind; read each image once). Work from `<book root>`. Do not run git
> commands. Do not spawn sub-agents. Return only the three-line summary the prompt asks for.

Keep 12 running; run `wave.py next --size <free slots>` to backfill. Dead agents: resume
them; if that fails, `wave.py next --redispatch`. Then:

```
uv run python scripts/wave.py queue               # normalize, auto-resolve, diff A vs B, queues
```

Check: every page shows both reads in `wave.py status`; agreement per page printed by
`queue` is mostly 94–100% (a page far below that: have an Opus agent look at its diff in
`transcription/diff/<id>.md` for a structural misread, and re-read that page if needed).
Checkpoint: commit after each wave of reads.

## 5. Resolve: defer to the translator, finalize

```
uv run python scripts/wave.py defer               # every undecided item -> either/unknown, by: auto
uv run python scripts/wave.py apply               # validated finals, manifest sync, stitch
uv run python scripts/wave.py status --all | tail -5
```

A page `apply` refuses has a final that does not validate. Dispatch:

> Work from `<book root>`. Do not run git commands. Do not spawn sub-agents. These pages
> fail `wave.py apply`: <ids>. For each, run
> `uv run --with jsonschema python scripts/apply_arbitration.py <id> --out /tmp/<id>.json`,
> read the validator's message, and find the cause (usually a reader's uncertainty note that
> did not survive, or a marker with no note). Fix it in the data, not by weakening the
> validator: edit the decisions file or a read, never the final. If the cause is a script
> bug, fix the script and add a test in `scripts/tests/`. Return three lines: pages fixed,
> causes, any page left failing.

Carson may override any decision by hand: `uv run python scripts/arbitrate_server.py
--port 8766`, open http://127.0.0.1:8766/ (not localhost); his decisions are `by: carson`
and are never overwritten. Checkpoint: every page has a final; tests pass; commit.

## 6. Stitch into sections; scan the headings first

```
uv run python scripts/stitch_text.py --check | tail -3
```

Heading detection is the step that went wrong last time (four misprinted headings shifted
every id after p045). Before any translation:

> Work from `<book root>`. Do not run git commands. Do not spawn sub-agents. List every
> heading block in `transcription/final/*.json` (one Python one-liner; print page id and
> text) and every section in `uv run python scripts/stitch_text.py --dry-run --sections`.
> The section headings of this book are <the forms from book.json `headings`>. Find (1)
> heading blocks that look like a section heading but did not start a section (misprints
> beyond one letter, wrong punctuation, a heading set in a paragraph block), (2) sections
> whose printed number is out of sequence, (3) whether the count of main-text sections and
> annotations matches the book (<count if known>), (4) that the last section is complete.
> Do not edit anything. Write `docs/checks/headings.md` and return three lines: sections by
> kind, near-miss headings found, mismatches.

Near misses → a code agent extends `parse_heading` in `stitch_text.py` (with a test) or
adds the form to `book.json` `headings`. Re-stitch until the count is right. If ids change
after translation has started, save the old `text/sections.json` first and run
`scripts/migrate_section_ids.py --old <saved> --new text/sections.json --apply`.
Checkpoint: commit `text/`.

## 7. Translate in batches (Fable), feed choices back

For each batch start id (the first section without `translation/sections/<id>.md`; the
section ids come from `uv run python scripts/stitch_text.py --dry-run --sections`, which
you may read: one line per section):

```
uv run python scripts/render_translate.py --batch <first-id> --size 12 > $BOOK_SCRATCH/translate-<first-id>.md
```

Six Fable agents in parallel, each exactly:

> Read the file `<prompt path>` and carry out the task it describes exactly. Work from
> `<book root>`. Write ONLY your sections, your batch report and your alt-choices file; do
> NOT edit `docs/case-file.md`: put glossary additions in the report. Do not run git
> commands. Do not spawn sub-agents. Return only the summary the prompt asks for.

Batches write only their own files, so parallel is safe. Each translator runs
`scripts/check_markers.py <id>` per section. After all batches:

```
uv run python scripts/check_markers.py --all | tail -3          # N/N ok
uv run python scripts/apply_translator_choices.py --dry-run translation/alt-choices/*.json | tail -5
uv run python scripts/apply_translator_choices.py translation/alt-choices/*.json | tail -5
```

Apply all choices in one pass after translation, never between parallel batches
(refinalizing shifts line indices). Then merge the glossary once:

> Work from `<book root>`. Do not run git commands. Do not spawn sub-agents. Read the
> "glossary additions" of every `translation/reports/batch-*.md` and §9 of
> `docs/case-file.md`. Append each new row to the §9 table once (French | English | Note
> with section ids), merging duplicates; where two batches chose differently for the same
> term, add one row with the majority rendering and list the conflict under a "Conflicts
> for review" line beneath the table. Edit only §9, in one Edit or Write. Return three lines:
> rows added, duplicates merged, conflicts.

Checkpoint: `check_markers --all` clean, tests pass; commit.

## 8. Review (Fable, whole-book passes, in sequence, plan mode)

```
uv run python scripts/render_review.py --out $BOOK_SCRATCH/review-book.md --quarters 4
```

The command prints a table of four contiguous section runs. Passes, in order: `pass-0-glossary`
(scope: "Whole book: consistency and glossary only."), then `pass-1-q1` … `pass-4-q4`
(scope: "Your scope is sections <first>–<last>: fidelity, consistency, and comparison with
the prior translation."). For each pass, fill the prompt with sed (do not read it):

```
P=pass-1-q1; sed -e "s|{BOOK_ROOT}|$PWD|g" -e "s|{PASS_NAME}|$P|g" \
  -e "s|{PASS_SCOPE}|<scope sentence>|g" -e "s|{RENDER_PATH}|$BOOK_SCRATCH/review-book.md|g" \
  -e "s|{PLAN_PATH}|translation/review/plan-$P.json|g" \
  -e "s|{ALT_CHOICES_PATH}|translation/alt-choices/review-$P.json|g" \
  -e "s|{MISREADINGS_PATH}|translation/review/misreadings-$P.json|g" \
  -e "s|{REPORT_PATH}|translation/reports/review-$P.md|g" prompts/review.md > $BOOK_SCRATCH/review-$P.md
```

One Fable agent, exactly:

> Your entire task is written in the prompt file `<path>`. Read it and follow it exactly.
> Work from `<book root>`. Do not run git commands. Do not spawn sub-agents. Return only the
> three-line summary it asks for.

Then, before the next pass:

```
uv run python scripts/apply_review_plan.py translation/review/plan-$P*.json | tail -8          # dry run
uv run python scripts/apply_review_plan.py --apply translation/review/plan-$P*.json | tail -8
uv run python scripts/render_review.py --out $BOOK_SCRATCH/review-book.md --quarters 4         # re-render
```

MISSING / AMBIGUOUS / MARKERS_CHANGED entries in the dry run go back to a small Fable
fixer (below) or are dropped. Commit after each pass. After the last pass:

```
uv run python scripts/apply_translator_choices.py --by reviewer translation/alt-choices/review-*.json | tail -5
```

**Page-image check** (Opus), on every reviewer misreading. Run it after the last
`apply_translator_choices` (a later `wave.py refinalize` of a page rebuilds its final and
drops the sic notes this pass adds):

> Work from `<book root>`. Do not run git commands. Do not spawn sub-agents. Do not crop,
> zoom or process any image; read each strip once. For every entry in
> `translation/review/misreadings-*.json` (`section`, `page`, `french_quote`, `proposed`),
> find the line in `transcription/final/<page>.json`, look at that page's native strips in
> `pages/strips/<page>/` (the body or margin strip holding the line), and decide what the
> print shows: the transcription, the proposed reading, or undecidable. Where the print
> agrees with the transcription but the word is a wrong sort, add a `decisions[]` entry and
> an `uncertain[]` entry with note `sic: <intended reading>` to the final, run
> `uv run --with jsonschema python scripts/validate_page.py transcription/final/<page>.json`,
> and, if the English flagged the word `[unclear]`, render the intended sense and remove the
> flag. Where the print shows the proposed reading, fix the final the same way and note the
> correction. Write `translation/reports/review-images.md`: one row per entry with verdict
> and action. Return three lines: entries checked, verdicts by kind, files changed.

**Mechanical sweep** (Opus):

> Work from `<book root>`. Do not run git commands. Do not spawn sub-agents. Read
> `$BOOK_SCRATCH/review-book.md` (the whole book, French and English) once, in the largest
> chunks the Read tool allows, then the review plans `translation/review/plan-*.json`
> (they record what the review passes changed). Check, section by section:
> every French sentence has English (omissions), numbers and dates, names against
> `docs/case-file.md` §3–§4, Latin quotations translated with the original kept, negations
> (a *ne … pas / point / iamais* rendered as positive or the reverse), paragraph counts, and
> every review edit for a reversed meaning. Do not edit any file. Write a plan in the
> `scripts/prompts/review.md` format to `translation/review/plan-sweep.json` holding only
> fixes you are certain of (exact unique `old` substrings, markers unchanged) and
> everything else as `findings`, plus `translation/reports/review-sweep.md`. Return three
> lines: sections checked, fixes, findings needing judgment.

Apply its plan with `apply_review_plan.py` (dry run, then `--apply`). Judgment findings go
to a **narrow fixer** (Fable), given only the affected sections:

> Work from `<book root>`. Do not run git commands. Do not spawn sub-agents. Read
> `docs/case-file.md`, then for each of these findings <list: section id, finding> read the
> French of that section in `text/sections.json` (search for its id; read only it) and its
> English in `translation/sections/<id>.md`, decide, and fix the English where the finding
> is right. Keep every `⟦pNNN⟧` and `{x}` marker; run
> `uv run python scripts/check_markers.py <id>` after each edit. Log any terminology decision
> as one line under "## 12. Review decision log" in `docs/case-file.md`. Return three lines:
> findings fixed, findings rejected and why, check_markers result.

Checkpoint: `check_markers --all` clean, tests pass; commit.

## 9. Site

```
uv run --with jsonschema python scripts/split_pages.py | tail -2     # site/data from finals + translations
uv run python scripts/site_images.py                                 # site/img/*.webp (needs cwebp; not in git)
```

`split_pages.py` takes the viewer's book record from `book.json` `site` and each page's
source link from the manifest `url` (`source: "other"`). **Gap:** the viewer itself is not
in this repo; it lives in `~/github/code-by-carson/translations/viewer/` (untracked there)
and still has Coras assumptions (front-matter labels keyed by `p000-title` /
`p000-argument`, French/English layer names). Its data contract is
`2026/docs/site-data-contract.md`. Deploying is Carson's call.

## 9b. Cited sources (optional, after review)

Links each note citation to the cited passage. Order and dispatch:

1. **Verify identifications.** Opus web agents, one per quarter of the uncertain Notes
   entries, check them against the source texts and report; one Fable residue pass resolves
   or marks the rest explicitly unidentified. Apply corrections to the notes.
2. **You write the contract** (`docs/sources-contract.md`: index.json, per-unit files under
   `site/data/sources/<corpus>/`, passage schemes, `citations.json` keyed `pNNN:marker` with
   status passage/unit/work/scan/none, viewer behaviour) before dispatching anyone.
3. **One Opus agent per corpus family, in parallel** (Roman law, Vulgate, classical, canon
   law), each writing only its own `scripts/fetch_<corpus>.py` and `site/data/sources/<corpus>/`,
   caching downloads so `--offline` rebuilds. Coras: ~115k–245k tokens each.
4. **One locator agent**: `build_sources_index.py` + `cite_locate.py` (grammar, scheme
   adapters, concordances, `--absent CORPUS` while a fetcher is still writing, coverage report).
5. **One viewer agent** in the viewer repo, built against a small fixture that follows the
   contract, so it can start before the corpora exist.
6. **One verification agent on real data**: headless check of the pane on cited pages,
   then a diagnostic pass on why refs stop at unit/work, and an incipit pass for laws cited by
   opening words (~235k tokens each).

Rerun `cite_locate.py` (without `--absent`) and `split_pages.py` when the last corpus lands.
Commentaries and humanist works have no open text: expect them at scan or none.

## 10. Close

Update `docs/handoff.md` (state table, open items, `[unclear]` flags still standing,
cost) and `docs/pipeline-log.md`; run the tests and `check_markers --all`; commit with
Carson's go-ahead. Before any push to a public remote, check `docs/reference/` for
copyrighted material.

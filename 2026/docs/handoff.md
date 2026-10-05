# Handoff — Coras *Arrest memorable* (1572) transcription + translation

Updated 2026-10-03 at the end of the review stage (first written 2026-09-27). Read this first,
then `docs/pipeline-log.md` (last sections: "Runbook", "2026-10-03 Auto-defer to translator",
"2026-10-03 (later): review stage" with its review runbook). `docs/checks/pilot-2026-09-24.md`
has the evidence behind the single-pass reads. `docs/design.md` is the original design; §4
Stage 3 and §7 carry dated notes on what changed. `docs/case-file.md` is the shared memory
(glossary, people, conventions); §12 is the review decision log.

## What this project is

A diplomatic transcription of the 1572 Paris edition of Jean de Coras, *Arrest memorable du
Parlement de Tholose* (Cambridge UL Montaigne.1.7.22, 162 pages incl. title and Argument), a
first English translation of the whole text including the 111 annotations and every marginal
citation, and a static side-by-side site. Everything lives under `2026/`; the repo-root pipeline
is dead legacy. Carson owns decisions; never commit or push without his say-so.

## Where things stand (2026-10-03)

| stage | state |
|---|---|
| Reads (two Opus single-pass reads per page) | done, all 162 pages (p150–p160 read 2026-10-03) |
| Arbitration | no longer by hand: every undecided item auto-deferred to the translator (`wave.py defer`); 36 Carson decisions kept |
| Finals (`transcription/final/`) | every page; re-finalized after translator and reviewer choices |
| Stitched French (`text/sections.json`, `text/alts.json`) | 225 sections (112 texte, 111 annot) after the heading-misprint re-cut; texte-112 `complete: false` |
| Translation (`translation/sections/`) | 224 of 225; only texte-112 missing |
| Review | done: pass-0-glossary, pass-1-q1 … pass-4-q4 (Fable), reviewer alt choices applied, Opus omission sweep, page-image pass, five sweep fixes. Findings in `translation/review/<id>.md` |
| Checks | `check_markers --all` 224/224; 356 tests; site readings by provenance: reconciler 270, translator 162, auto 84, carson 36, reviewer 15 |
| Site | `site/data` rebuilt (162 pages, french + english); **not deployed** (`docs/site-plan.md` Task 12, Carson's call). Viewer code is in `~/github/code-by-carson/translations/viewer/`, untracked there |
| Cited sources (`site/data/sources/`, `site/data/citations.json`) | 83 corpora (~80 MB); 1434 refs: passage 855, unit 229, work 256, scan 21, none 73 (at 3238178); viewer source pane in code-by-carson e9d8e87; incipit pass running; Decretum remainder downloading |
| Git | latest commit `958d631`; **nothing pushed to origin**. Root-level junk (`example_data/`, `new_instructions.md`, `requirements.txt`, `2026/arst.md`) deliberately uncommitted |

## The process now (why it changed)

The first run (Sept 21–22) cost ≈$1,264 API-equivalent for 66 pages: Opus readers zoomed on
every word and the Fable coordinator's context reached 937k. A pilot showed a **single-pass
read** (13 native strips, no cropping, ~20 tool calls) costs $0.66 instead of $6.90 with
comparable accuracy but a different error profile (finds wrong sorts the zoom pipeline
normalized, misses fine glyph calls). Sonnet was rejected as reader. Model reconciliation was
dropped; Carson arbitrated by hand for a few pages, then (2026-10-03) stopped.

Now: **auto-defer + translator feedback + review passes.** Every reader disagreement Carson has
not decided is deferred (`wave.py defer`, `by: auto`): the final keeps reading A and the stitch
carries both readings into the French as `⟨alt:…⟩` / `⟨alt?:…⟩`. The Fable translator, who sees
the whole French, picks A, B or either for each alt and writes a choices file;
`apply_translator_choices.py` feeds those back as decisions (`by: translator`; Carson's are never
overwritten) and re-finalizes the pages. Then whole-book **review passes** on Fable, one after
another so each inherits the last one's glossary decisions, read the full render and emit a plan
JSON that `apply_review_plan.py` applies (exact-substring edits, marker sequence protected,
findings to `translation/review/`, decisions to case-file §12); their alt choices go back into
the French with `--by reviewer`. An Opus omission sweep and a page-image check against the print
closed the stage.

Pipeline per page: `read_single.md` → two Opus reads `reads/A|B/<id>.json` → `wave.py queue`
(normalize, auto_resolve, diff, queue) → `wave.py defer` → `wave.py apply` (finals, stitch) →
Fable translation batches + `apply_translator_choices.py` → review passes.

## Runbook (exact commands, from `2026/`)

```
uv run python scripts/wave.py status                 # per-page reads / queue / decided / final
uv run python scripts/wave.py defer                  # defer every undecided item to the translator
uv run python scripts/wave.py apply                  # finals for decided pages, manifest sync, restitch
uv run python scripts/render_translate.py --batch <id> --size 12 > $CORAS_SCRATCH/translate-<id>.md
#   one Fable agent: "Your entire task is written in the prompt file <path>. Read it and follow it
#   exactly, then return only the summary it asks for. Do not spawn sub-agents."
uv run python scripts/apply_translator_choices.py translation/alt-choices/batch-<first>--<last>.json
# review pass (details: pipeline-log "Runbook addendum: review pass")
uv run python scripts/render_review.py --out $CORAS_SCRATCH/review-book.md --quarters 4
#   fill {PASS_NAME} {PASS_SCOPE} {RENDER_PATH} {PLAN_PATH} {ALT_CHOICES_PATH} {MISREADINGS_PATH}
#   in a copy of scripts/prompts/review.md; one Fable agent; passes run one at a time
uv run python scripts/apply_review_plan.py translation/review/plan-<pass>.json --dry-run   # then --apply
uv run python scripts/apply_translator_choices.py --by reviewer translation/alt-choices/review-<pass>.json
uv run --with jsonschema python scripts/split_pages.py                 # rebuild site/data
uv run python scripts/check_markers.py --all                           # 224/224
uv run --with pytest --with jsonschema --with pillow python -m pytest scripts/tests -q   # 356 tests
uv run python scripts/arbitrate_server.py --port 8766   # only if Carson wants to decide by hand:
                                                     # http://127.0.0.1:8766/ (NOT localhost)
```

Arbitration keys (if used): `1` A, `2` B, `e` type the text, `3` either, `4` unknown, `u` undo,
`p` whole page. A decision made there is stored `by: carson` and overrides auto/translator.

## Rules Carson set (do not relitigate)

- **Minimize Fable tokens.** The coordinator session is Fable: coordinate, do not write code or
  read files yourself; send code to Opus agents and read their three-line summaries. Keep the
  coordinator context small (this handoff exists because the last session hit 750k).
- **Stop spawning agents when he says so** (weekly usage limit; it resets ~8am). He tracks Opus
  and Fable percentages separately.
- **No AI attribution anywhere**: no Co-Authored-By, no "Generated with" in commits, PRs, docs.
- **No image files in git, ever** (root `.gitignore` ignores all image extensions; QA screenshots
  go to S3 if wanted). Check `git diff --cached --name-only` before every commit.
- **Never commit or push without his go-ahead.** Commits are authored by Carson only.
- Fable translates (it is cheap there: ~$0.37/section); Opus does the fidelity review (in practice
  the review passes ran on Fable, with an Opus omission sweep after); keep
  some spot-checking so no error class creeps in unseen.
- Human arbitration replaces model reconciliation; he does not want to see lines both readers
  agree on. (Since 2026-10-03 he no longer arbitrates; undecided items are auto-deferred to the
  translator.)

**Cited sources** (2026-10-04, pipeline-log "cited sources"). After the translators'
`## Notes` identifications were verified (183 of 216 uncertain ones by four Opus web agents,
15 more and 18 explicitly unidentified by a Fable residue pass), the cited texts were fetched
into `site/data/sources/` per `docs/sources-contract.md`: Roman law (Digest, Code with a
vulgate→Krüger concordance, Institutes, Novels), the Clementine Vulgate, 74 classical works,
and the canon law (Decretum and Decretals clean; Sext and Clementines from OCR). One fetcher
per corpus, all rerunnable `--offline`. `build_sources_index.py` then `cite_locate.py` map each
note citation to `site/data/citations.json` (`pNNN:marker` → passage / unit / work / scan /
none), and the viewer opens the passage in a side pane (bottom sheet on phones).

## Known issues and open items (2026-10-03)

- **texte-112** (last section, p160) is `complete: false` and untranslated. Cause: the stitch
  closes a section only at the next heading, and the book ends after the colophon on p160, so
  the last section never closes. Needs an end-of-book close in `stitch_text.py`, then one
  translation.
- `[unclear]` flags still standing in the English: texte-63 *perſonnément*, annot-050
  *ignorons*, annot-025 *de poids*, annot-071 Faustina, texte-105 dropped verb.
- p073 e2 key vs conventions §4: a Carson-session decision, left as is.
- p071 foot note m is filed as an orphan under texte-48 by the stitch.
- p159 note q has no body marker.
- p137 `incon-ſtãce`: undecidable from the print (page-image pass).
- The translators' `## Notes` citation identifications were never reviewed (the review render
  omits them); many say "unverified" / "unidentified".
- 71 alt markers left as either (both readings mean the same); the French keeps reading A.
- Stale translator notes in annot-023 / annot-036 / annot-050 still say they carry the following
  TEXTE block (split off by the re-cut). Reports written before the re-cut use old ids in their text.
- Three errors found in old finals during the pilot (p010 `toures`, `noſtré`; p159 `noraire`):
  check whether the page-image pass covered them before assuming they are fixed.
- Site not deployed (site-plan Task 12); viewer code untracked in
  `~/github/code-by-carson/translations/viewer/`.
- Nothing pushed to origin. Root-level junk deliberately uncommitted.
- `docs/reference/ringold-lewis-1982.txt` (the 1982 published translation) is committed: check
  copyright before any push to a public remote.
- Cited sources: commentaries (152 refs) and humanist works are scan-only or `none`; 73 refs
  `none`; `citations.json` is 444 KB and loaded on every page (candidate for a per-page split);
  Decretum D.92–101, C.1–26, De cons. still downloading (`fetch_canon_law.py build --only
  decretum`, then rerun `cite_locate.py`); incipit pass result in
  `translation/reports/citations-incipit.md` still to apply.
- Future review/sweep prompts must forbid sub-agents (the Opus sweep fanned out to 6 on its own).

## Cost reference (2026-10-03)

| item | measured |
|---|---|
| single-pass Opus read (harness) | $0.66, ~20 tool calls, 1–3 min |
| Fable translator | ~$0.37/section |
| whole-book Fable review pass | ≈520–530k tokens, ~30 tool calls; almost all is the one-time read-in (render read in ~28 chunks); edit count barely changes it |
| whole first run (Sept 21–22) | $1,264 for 66 pages |
| pilot + tool-building session (Sept 24–26) | ~$49 |

A further review pass costs one full read-in regardless of scope; narrow fixes are far cheaper
with a small Fable fixer given only the affected sections (as for 958d631).

## Files to know

- `scripts/prompts/read_single.md` reader prompt; `translate.md` translator (batch mode);
  `spotcheck.md`, `reconcile.md` (legacy).
- `scripts/wave.py`, `render_prompt.py` (falls back to reads/A as context beyond the finals
  frontier), `render_translate.py`, `arbitrate_queue.py`, `arbitrate_server.py`,
  `apply_arbitration.py`, `stitch_text.py` (turns "either/unknown" into `⟨alt:…⟩` markers),
  `pilot_compare.py`, `diff_reads.py`, `normalize_spacing.py`, `auto_resolve.py`, `validate_page.py`.
- `tools/arbitrate/` the arbitration page (no build step).
- Review: `scripts/prompts/review.md`, `render_review.py`, `apply_review_plan.py`,
  `apply_translator_choices.py` (`--by translator|reviewer`), `auto_defer.py`; outputs in
  `translation/review/` and `translation/alt-choices/`.
- Cited sources: `docs/sources-contract.md`; `scripts/fetch_roman_law.py`, `fetch_vulgate.py`,
  `fetch_classical.py`, `fetch_canon_law.py`, `build_sources_index.py`, `cite_locate.py`;
  reports `translation/reports/citations-*.md`.
- Memory notes for Claude Code live in `~/.claude-mine/projects/-Users-cdavis-github-translator/memory/`.

# Handoff — Coras *Arrest memorable* (1572) transcription + translation

Written 2026-09-27 at the end of a long session. Read this first, then `docs/pipeline-log.md`
(last three sections: "Token audit and single-pass pilot", "Runbook", "Single-pass wave, resume
point"), then `docs/checks/pilot-2026-09-24.md` if you need the evidence behind the process.
`docs/design.md` is the original design; §4 Stage 3 and §7 carry dated notes on what changed.

## What this project is

A diplomatic transcription of the 1572 Paris edition of Jean de Coras, *Arrest memorable du
Parlement de Tholose* (Cambridge UL Montaigne.1.7.22, 162 pages incl. title and Argument), a
first English translation of the whole text including the 111 annotations and every marginal
citation, and a static side-by-side site. Everything lives under `2026/`; the repo-root pipeline
is dead legacy. Carson owns decisions; never commit or push without his say-so.

## Where things stand (2026-09-27)

| stage | state |
|---|---|
| Reads (two independent Opus single-pass reads per page) | done for every page through **p149**; **p150–p160 not read** (prompts already rendered in the scratchpad, or re-render with `wave.py next`) |
| Finals (`transcription/final/`) | 68: title, argument, p001–p063, p066, p067, p159 |
| Arbitration queues (`transcription/arbitration/queue/`) | 88 pages, ~290 items, waiting for Carson: p030, p057 (pilot), p064–p149 |
| Decisions made | p030, p057, p066, p067 (plus a few on p070) |
| Stitched French (`text/sections.json`) | 83 sections, stops at p064 (needs finals from arbitration) |
| Translation (`translation/sections/`) | 82 sections, through annot-041 / p063 |
| Review passes | not started |
| Site | built, not deployed; `docs/site-plan.md` Task 12 is go-live, Carson's call |
| Git | 3 local commits ahead of origin/main (af46f4d, de9dffa, 4c29997… latest `7ef3d72`); **not pushed**. Working tree clean except root-level junk (`example_data/`, `new_instructions.md`, `requirements.txt`, `2026/arst.md`) that is deliberately uncommitted |

## The process now (why it changed)

The first run (Sept 21–22) cost ≈$1,264 API-equivalent for 66 pages: Opus readers zoomed on
every word (77 turns, 82 image reads, 9M cached tokens per read) and the Fable coordinator's
context reached 937k. A pilot on five verified pages showed a **single-pass read** (13 native
strips, no cropping, ~20 tool calls) costs $0.66 instead of $6.90 with comparable accuracy but a
different error profile: it finds wrong sorts the zoom pipeline normalized, and misses fine glyph
calls (period/comma, h/b, e/c, hyphen vs speck). Sonnet was rejected as reader (80–90% pair
agreement, silently normalizes). Fable reconciliation was replaced by **Carson arbitrating**
in a local page, because a human is best at exactly the fine-glyph class.

Pipeline per page: `read_single.md` prompt → two Opus subagents write `reads/A|B/<id>.json` →
`normalize_spacing` + `auto_resolve` + `diff_reads` → `arbitrate_queue.py` builds items with crops
→ Carson decides in `tools/arbitrate/` (served by `arbitrate_server.py`) → `apply_arbitration.py`
writes the final → `stitch_text.py` → Fable translates in 10–12-section batches with the whole
French as context → (later) Opus fidelity review; Fable spot-check every 10th page.

## Runbook (exact commands, from `2026/`)

```
uv run python scripts/wave.py status                 # per-page reads / queue / decided / final
uv run python scripts/wave.py next --size 10         # render read_single prompts for the next pages
#   dispatch one Opus subagent per prompt with:
#   "Read the file <prompt path> and carry out the task it describes exactly, including where to
#    save your output and the hard rules on tools (no image cropping, zooming or processing of any
#    kind; read each image once). Work from /Users/cdavis/github/translator/2026. Do not run git
#    commands. Return only the three-line summary the prompt asks for."
#   cap 12 concurrent; each takes 1–3 min and ~$0.66; backfill as they return.
uv run python scripts/wave.py queue                  # diff + queue for pages with both reads
uv run python scripts/arbitrate_server.py --port 8766   # Carson: http://127.0.0.1:8766/ (NOT localhost: another
                                                     # project's server sits on 8765 and localhost resolves there)
uv run python scripts/wave.py apply                  # finals for fully decided pages, manifest sync, restitch
uv run python scripts/render_prompt.py spotcheck p070 --model fable   # Fable spot-check every 10th page (p070 first)
uv run python scripts/render_translate.py --batch texte-40 --size 12  # next Fable translation batch
uv run --with pytest --with jsonschema --with pillow python -m pytest scripts/tests -q   # 268 tests
```

Arbitration keys: `1` A, `2` B, `e` type the text, `3` either (both readings kept for the
translator as `word⟨alt:other⟩`), `4` unknown (kept + escalated, `⟨alt?:…⟩`), `u` undo, `p` whole
page, arrows move. Flagged-but-agreed lines are OFF by default (`--include-flagged` restores).
Auto-advances to the next page when a page is done.

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
- Fable translates (it is cheap there: ~$0.37/section); Opus does the fidelity review; keep
  some spot-checking so no error class creeps in unseen.
- Human arbitration replaces model reconciliation; he does not want to see lines both readers
  agree on.

## Known issues and open items

- p142: both readers report the margin strips are clipped on the right; note readings on that
  page are low confidence (use the whole-page view when arbitrating).
- Three errors found in existing finals during the pilot, **not yet applied**: p010 `toures`,
  p010 `noſtré`, p159 `noraire` (finals normalized wrong sorts). Carson has not said to fix them.
- The margin-note crop window is placed by ink-row matching; on a page where rows don't match
  the note lines it can be a line off — the crop links to the native strip as fallback.
- `wave.py next` skips pages with a read on disk, so it will correctly pick p150–p160 only.
- Stitch: `text/sections.json` stops at p064 until finals exist; translation waits on it.
- `docs/reference/ringold-lewis-1982.txt` (the 1982 published translation) is committed; check
  copyright before any push to a public remote.
- The API drivers `scripts/api_read.py` / `api_reconcile.py` work (dry-run) but the account has no
  API credits, and Carson prefers the prepaid harness anyway.

## Cost reference

| item | measured |
|---|---|
| single-pass Opus read (harness) | $0.66, ~20 tool calls, 1–3 min |
| Fable reconciler (harness, no longer used) | $1.30/page |
| Fable translator | ~$0.37/section |
| whole first run | $1,264 for 66 pages |
| pilot + tool-building session (Sept 24–26) | ~$49 |

Estimated remainder: reads p150–p160 ≈ $15; translation ~140 sections ≈ $50–80 Fable; review
~230 sections ≈ $120 Opus; coordination small if the coordinator stays lean.

## Files to know

- `scripts/prompts/read_single.md` reader prompt; `translate.md` translator (batch mode);
  `spotcheck.md`, `reconcile.md` (legacy).
- `scripts/wave.py`, `render_prompt.py` (falls back to reads/A as context beyond the finals
  frontier), `render_translate.py`, `arbitrate_queue.py`, `arbitrate_server.py`,
  `apply_arbitration.py`, `stitch_text.py` (turns "either/unknown" into `⟨alt:…⟩` markers),
  `pilot_compare.py`, `diff_reads.py`, `normalize_spacing.py`, `auto_resolve.py`, `validate_page.py`.
- `tools/arbitrate/` the arbitration page (no build step).
- Memory notes for Claude Code live in `~/.claude-mine/projects/-Users-cdavis-github-translator/memory/`.

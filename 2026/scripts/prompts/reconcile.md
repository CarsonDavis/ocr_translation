# Reconciler prompt

Two independent readers transcribed page `{PAGE_ID}` of Coras, *Arrest memorable* (1572).
Their outputs differ in places. Your job is to decide every difference by looking at the
page image, and to write the final master transcription. Accuracy over speed, but be
economical: you are an expensive model, so spend your effort only on the disputed and
uncertain lines. Do not defer to either reader by default.

## Inputs (under /Users/cdavis/github/translator/2026/)

- `docs/conventions.md` — read first, completely.
- `transcription/reads/A/{PAGE_ID}.json`, `transcription/reads/B/{PAGE_ID}.json`
- `transcription/diff/{PAGE_ID}.md` — the line-level differences (read this; it tells
  you exactly where to look).
- Images: `pages/read/{PAGE_ID}.jpg`, `pages/strips/{PAGE_ID}/body-*.jpg`,
  `margin-*.jpg`, `foot.jpg`.
- Context: preceding finals {CONTEXT_PAGES} in `transcription/final/`.
- Manifest record: {MANIFEST_RECORD}

## Procedure

1. Read the conventions and the diff.
2. For every differing line pair, find the line in the correct strip and read it yourself
   at full size. Decide: A, B, or neither (write the correct text). Record the decision.
3. For structural differences (a missing line, a different paragraph split, a missing
   note, different keys), look at the whole page and decide the same way.
4. Do NOT re-read lines the two readers agree on; shared mistakes are caught by the
   separate spot-check sample. Only look at the differing lines, the structural differences,
   and the lines named in either reader's `uncertain[]`. Keep your image viewing to the
   strip regions that contain those lines (crop with Pillow rather than opening whole strips
   when a strip holds many lines you do not need).
5. If a reading is genuinely undecidable from the image, keep `[?]` or the best reading,
   add an `uncertain[]` entry with `"escalate": true` and a precise pointer (strip file and
   approximate line), so a human can look.
6. Write `transcription/final/{PAGE_ID}.json`: the same schema, `"reader": "final"`,
   `"model": "{MODEL}"`, plus a `"decisions"` array:
   `{"where": "blocks[1].lines[3]", "A": "…", "B": "…", "chose": "A"|"B"|"neither", "text": "…", "reason": "…"}`.
7. Run `uv run --with jsonschema python scripts/validate_page.py transcription/final/{PAGE_ID}.json`
   until it exits 0.

## Rules

- Never modernize, expand, or correct the print.
- Do not edit the reads or the diff; write only the final file.
- Do not write any other file.

## Report

Output path; number of differences decided (A / B / neither); number of shared mistakes
fixed; the escalations with their pointers; anything odd about the page.

Also save the same report verbatim to `transcription/reports/reconcile-{PAGE_ID}.md` (create the directory if
needed) so it is kept with the data.

RETURN ONLY a three-line summary to the coordinator (the full report lives in the file you
saved): line 1 the output path and validator/checker result; line 2 the counts; line 3 the
escalations or open questions, or "none".

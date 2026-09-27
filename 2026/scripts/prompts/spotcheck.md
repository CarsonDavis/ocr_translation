# Spot-check prompt

You are independently re-transcribing page `{PAGE_ID}` of Coras, *Arrest memorable* (1572)
to audit the accepted master transcription. You must NOT look at the master or at the
earlier reads until your own transcription is written. Accuracy over speed.

## Inputs (under /Users/cdavis/github/translator/2026/)

- `docs/conventions.md` — read first, completely.
- Images: `pages/read/{PAGE_ID}.jpg`, `pages/strips/{PAGE_ID}/body-*.jpg`,
  `margin-*.jpg`, `foot.jpg`.
- Context: preceding finals {CONTEXT_PAGES} in `transcription/final/` (only these; do not
  open `transcription/final/{PAGE_ID}.json` yet).
- Manifest record: {MANIFEST_RECORD}

## Procedure

1. Transcribe the page exactly as a reader would (follow `scripts/prompts/read.md`'s
   procedure) and write `transcription/spotcheck/{PAGE_ID}.json` with `"reader": "spotcheck"`.
   Validate it with `uv run --with jsonschema python scripts/validate_page.py <path>`.
2. Only now run:
   `uv run python scripts/diff_reads.py {PAGE_ID} --a transcription/final/{PAGE_ID}.json --b transcription/spotcheck/{PAGE_ID}.json --out transcription/spotcheck/{PAGE_ID}.diff.md`
3. For every difference, look at the image again and give a verdict: `final correct`,
   `spotcheck correct`, or `undecidable`. Substantive = anything other than spacing.
4. Write the verdict list to `transcription/spotcheck/{PAGE_ID}.verdict.md` as a table:
   `where | final | spotcheck | verdict | reason`.

## Rules

- Do not edit the master. Report; a human applies fixes.
- Do not write any other file.

## Report

Agreement percentage from the diff; number of differences; number where the master is
wrong (substantive), listed with the correct text; undecidables with pointers.

Also save the same report verbatim to `transcription/reports/spotcheck-{PAGE_ID}.md` (create the directory if
needed) so it is kept with the data.

RETURN ONLY a three-line summary to the coordinator (the full report lives in the file you
saved): line 1 the output path and validator/checker result; line 2 the counts; line 3 the
escalations or open questions, or "none".

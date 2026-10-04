# Single-pass reader prompt (READER = A or B)

You are transcribing ONE page of a 1572 French printed book for a diplomatic edition:
Jean de Coras, *Arrest memorable du Parlement de Tholose* (Paris, Galliot du Pré, 1572),
the account of the Martin Guerre case with Coras's numbered annotations. Accuracy matters
far more than speed, but this is a **single-pass read with a fixed tool budget**: you look
at each image once, at the resolution given, and transcribe what you see. You are one of
two independent readers; your output is diffed against the other reader's line by line and
every disagreement is settled by a human arbiter looking at the print, so do not skip
anything and do not tidy anything.

Page: `{PAGE_ID}` (manifest record below). Reader: `{READER}`. Model: `{MODEL}`.
Output file: `{OUT_DIR}/{PAGE_ID}.json`

## Inputs (all under {BOOK_ROOT}/)

1. `docs/conventions.md` — READ IT FIRST, completely. It defines every rule of the output.
2. `pages/read/{PAGE_ID}.jpg` — the whole page at reading size, for the layout only: how
   many paragraphs, headings, markers, margin notes, whether there is a foot block, a
   signature, a catchword.
3. `pages/strips/{PAGE_ID}/body-1.jpg` … `body-8.jpg` (sometimes more) — the body column
   cut into overlapping horizontal strips **at the scan's native resolution**. Consecutive
   strips overlap by a few lines; do not transcribe an overlapped line twice.
4. `pages/strips/{PAGE_ID}/margin-1.jpg` … — the margin column at native resolution.
5. `pages/strips/{PAGE_ID}/foot.jpg` — the bottom of the page (foot citations, signature,
   catchword).
6. Context: the transcriptions of the preceding pages (finals where they exist), so you know which sentence,
   paragraph and section this page continues (`continues_prev`, a word broken across the
   page boundary, the running marker alphabet). Never copy from them. Files under
   `transcription/reads/` are unreconciled reads, use them only for continuity.
   Preceding pages: {CONTEXT_PAGES}
7. Manifest record: {MANIFEST_RECORD}

## Procedure (about 8 tool calls in total)

1. One Read of `docs/conventions.md`; one Read of each context page file.
2. One Read of `pages/read/{PAGE_ID}.jpg`. Note privately the page's structure: running head
   and folio; the sequence of headings and paragraphs; every marker letter in the body and
   every note in the margin; foot block yes/no; signature; catchword.
3. Read ALL the body strips in ONE turn (issue the Read calls for body-1 … body-N together).
4. Read ALL the margin strips and `foot.jpg` in ONE turn.
5. Transcribe every printed line as one string, in order, applying the conventions
   (long s as `ſ`, u/v and i/j as printed, tildes kept, markers as `{x}`). While reading,
   attend to the three error classes that careful readers get wrong on this print:
   (a) `ſſ` vs `ſs` — the print often sets long s + round s inside a word (`profeſsion`,
   `auſsi`); (b) sentence punctuation — a comma has a tail below the baseline, a period is
   a round dot on the baseline, a colon has two dots; decide from the shape, never from the
   sense; (c) wrong sorts — this print has a wrong letter roughly once a page (`raporrera`,
   `cſgalle`, `qni`, `viute`, `Cuerre`); your eye reads the expected word, the print did not
   print it; transcribe the misprint as printed with an `uncertain[]` note `sic`.
6. Cross-check from memory of the images: every `{x}` in the body has a note with key `x`
   and every note has a marker; the line count of each paragraph matches the whole-page
   image.
7. Fill `uncertain[]` for every doubtful reading, every `[?]`, every missing marker or note.
8. Write the JSON to `{OUT_DIR}/{PAGE_ID}.json` in ONE Write call, with `"id": "{PAGE_ID}"`,
   `"reader": "{READER}"`, `"model": "{MODEL}"`.
9. Run `uv run --with jsonschema python scripts/validate_page.py {OUT_DIR}/{PAGE_ID}.json`
   and fix every reported problem until it exits 0 (at most two fix rounds). A warning about
   a suspicious normalized word must be checked against what you saw, not just silenced.

## Hard rules on tools

- **Do NOT crop, zoom, upscale, convert, or otherwise process any image.** No `magick`,
  `sips`, Pillow, OpenCV, or any Python/shell image code. Do not open `pages/full/`. Do
  not Read any image more than once. The strips already carry the scan's full resolution;
  there is no more detail to be had by cropping them.
- Where a glyph is genuinely ambiguous at this resolution, give your best reading and
  record it in `uncertain[]`. Do not spend tool calls trying to resolve it.
- Do not write any file other than your output JSON.
- If an input file is missing or an image is unreadable, stop and report which.

## Rules of the transcription

- Never guess silently. Uncertain → best reading + `uncertain[]` entry, or `[?]`.
- Never modernize, never expand abbreviations, never fix misprints.
- Never merge or split printed lines.
- The crops keep a narrow strip of the **facing page** along the gutter edge (inner edge:
  right on versos, left on rectos). Ignore it completely; it is not part of this page.

## Report

RETURN ONLY a three-line summary to the coordinator: line 1 the output path and the
validator result; line 2 the counts (body lines, paragraphs, headings, markers, margin
notes, foot notes, uncertain entries); line 3 the escalations or open questions, or "none".

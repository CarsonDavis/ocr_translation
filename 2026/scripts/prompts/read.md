# Reader prompt (READER = A or B)

You are transcribing ONE page of a 1572 French printed book for a diplomatic edition:
Jean de Coras, *Arrest memorable du Parlement de Tholose* (Paris, Galliot du Pré, 1572),
the account of the Martin Guerre case with Coras's numbered annotations. Accuracy matters
far more than speed. You are one of two independent readers; your output will be diffed
against the other reader's line by line, so do not skip anything and do not tidy anything.

Page: `{PAGE_ID}` (manifest record below). Reader: `{READER}`.

## Inputs (all under {BOOK_ROOT}/)

1. `docs/conventions.md` — READ IT FIRST, completely. It defines every rule of the output.
2. `pages/read/{PAGE_ID}.jpg` — the whole page at reading size. Use it to understand the
   layout: how many paragraphs, headings, markers, margin notes, whether there is a foot
   block, a signature, a catchword.
3. `pages/strips/{PAGE_ID}/body-1.jpg`, `body-2.jpg`, … — the body column cut into
   overlapping horizontal strips at native resolution. Consecutive strips overlap by a few
   lines; do not transcribe an overlapped line twice.
4. `pages/strips/{PAGE_ID}/margin-1.jpg`, … — the margin column at native resolution
   (absent if the page has no margin notes).
5. `pages/strips/{PAGE_ID}/foot.jpg` — the bottom of the page (foot citations, signature,
   catchword).
6. Context: `transcription/final/` files for the preceding pages listed below, so you know
   which sentence, paragraph and section this page continues. Use them for `continues_prev`
   and for reading a word broken across the page boundary. Never copy from them.
   Preceding pages: {CONTEXT_PAGES}
7. Manifest record: {MANIFEST_RECORD}

## Procedure

1. Read `docs/conventions.md`.
2. Open `pages/read/{PAGE_ID}.jpg`. Write down (privately) the page's structure: running
   head and folio; the sequence of headings and paragraphs; every marker letter you can see
   in the body and every note in the margin; foot block yes/no; signature; catchword.
3. Open each body strip in order and transcribe every printed line as one string, applying
   the conventions (long s as `ſ`, u/v and i/j as printed, tildes kept, markers as `{x}`).
   Work slowly. For each line, check the number of words against the image before moving on.
4. Open each margin strip and transcribe every note, one entry per key, its lines as printed
   (the key letter itself is not part of the lines).
5. Open `foot.jpg` and transcribe foot citations (if any), the signature, the catchword.
6. Cross-check: every `{x}` in the body has a note with key `x`, and every note has a
   marker. Count the lines in each paragraph against the whole-page image.
7. Two error classes that careful readers have still got wrong, so check them explicitly:
   (a) **`ſſ` vs `ſs`**: the print often sets long s + round s inside a word (`profeſsion`,
   `auſsi`, `paſsé`). For every double-s, zoom to at least 3x and look for the second
   letter's short round form before writing `ſſ`. (b) **sentence punctuation**: for every
   period, comma, colon and semicolon at a clause boundary, zoom and check the shape (a
   comma has a tail below the baseline; a period is a round dot on the baseline; a colon has
   two dots). Do not decide punctuation from the sense of the sentence.
   (c) **wrong sorts**: this print has a wrong letter roughly once a page (`raporrera`
   for raportera, `cſgalle` for eſgalle, `qni` for qui, `viute` for viure, `Cuerre` for
   Guerre). Your eye will read the expected word; the print did not print it. For every
   word, check that each glyph is the letter you wrote, especially r/t, c/e, n/u, a/o, and
   transcribe the misprint as printed with an `uncertain[]` note `sic`.
8. Fill `uncertain[]` for every doubtful reading, every `[?]`, every missing marker or note.
9. Write the JSON to `transcription/reads/{READER}/{PAGE_ID}.json` with
   `"id": "{PAGE_ID}"`, `"reader": "{READER}"`, `"model": "{MODEL}"`.
10. Run: `uv run --with jsonschema python scripts/validate_page.py transcription/reads/{READER}/{PAGE_ID}.json`
   and fix every reported problem until it exits 0. Warnings about suspicious normalized
   words must be checked against the image, not just silenced.

## Rules

- Never guess silently. Uncertain → best reading + `uncertain[]` entry, or `[?]`.
- Never modernize, never expand abbreviations, never fix misprints.
- Never merge or split printed lines.
- The crops keep a narrow strip of the **facing page** along the gutter edge (inner edge:
  right on versos, left on rectos). Ignore it completely; it is not part of this page.
- Do not write any file other than your output JSON.
- If an input file is missing or an image is unreadable, stop and report which.

## Report

When finished, report: the output path; the counts (body lines, paragraphs, headings,
markers, margin notes, foot notes); the number of `uncertain[]` entries and a one-line
summary of each; and anything about the page that the reconciler should know (damage,
faint ink, an unusual layout).

Also save the same report verbatim to `transcription/reports/read-{PAGE_ID}-{READER}.md` (create the directory if
needed) so it is kept with the data.

RETURN ONLY a three-line summary to the coordinator (the full report lives in the file you
saved): line 1 the output path and validator/checker result; line 2 the counts; line 3 the
escalations or open questions, or "none".

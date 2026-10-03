# Translator prompt (one logical section, or a batch of consecutive sections)

<!-- render note: scripts/render_translate.py keeps the "single" blocks for one section and
the "batch" blocks for a batch, and drops the other mode's blocks and this note. -->

<!-- single -->You are translating one section<!-- /single --><!-- batch -->You are translating a batch of consecutive sections<!-- /batch --> of Jean de Coras, *Arrest memorable du Parlement de
Tholose* (Paris, 1572), from sixteenth-century French into modern English. Coras was the
reporting judge in the Martin Guerre case; the book alternates the court record (`TEXTE`)
with his 111 learned annotations. This is the first English translation of the annotations.
Accuracy of meaning comes first; readable, natural English second; nothing is omitted.

<!-- single -->
Section: `{SECTION_ID}`. Output file: `translation/sections/{SECTION_ID}.md` (write only this
file, the alt-choices file `{ALT_CHOICES_PATH}`, plus the report file named at the end).
<!-- /single -->
<!-- batch -->
Sections, in book order ({BATCH_SIZE}): {SECTION_IDS}.
Output: one file per section, `translation/sections/<id>.md`, each in the format below
(write only these files, the glossary rows in the case file, the alt-choices file
`{ALT_CHOICES_PATH}`, and the one batch report named at the end).

**Read before you write.** Before translating any of it, read the French of every section
in the batch, start to finish, and skim the rest of `text/sections.json` (the stitched
French of the whole book so far) so you know where the argument is going and what comes
back later. `text/sections.json` is read-only context: never edit it. Then translate the
sections in order, one file each.
<!-- /batch -->

## Inputs (under /Users/cdavis/github/translator/2026/)

1. `docs/case-file.md` — READ FIRST: the story, the people (with Coras's spellings), the
   places, the procedure glossary (use its English renderings), the legal citation
   conventions (how to expand `l. minorem D. de ritu nup.`), classical sources, and the
   running glossary. Follow its choices; when you make a new recurring choice, append it to
   the glossary table at the end of that file (one line, French → English, section id).
2. `docs/conventions.md` §2 — how to read the diplomatic French (`ſ` = s, `u/v` and `i/j`
   as printed, `ẽ` = en/em, `õ` = on/om, `q̃` = que, `&` = et, `{a}` = a marginal citation
   marker).
3. <!-- single -->The section itself: `text/sections.json`, entry with `"id": "{SECTION_ID}"`.<!-- /single --><!-- batch -->The sections themselves: the `text/sections.json` entries with the ids listed above
   (the whole file is the French so far; read-only).<!-- /batch --> <!-- single -->Its<!-- /single --><!-- batch -->Each entry's<!-- /batch --> `text`
   is reflowed diplomatic French containing `⟦pNNN⟧` page markers and `{x}` note markers;
   its `notes` are the marginal citations keyed by letter.
4. Context: the previous <!-- batch -->batch's<!-- /batch --><!-- single -->sections'<!-- /single --> French and English: {CONTEXT_SECTIONS} (files under
   `translation/sections/` and the same `sections.json` entries). Match their terminology,
   names, and tone.

## Output format (exactly)

```
---
id: <!-- single -->{SECTION_ID}<!-- /single --><!-- batch --><section id><!-- /batch -->
pages: [p040, p041]
---
⟦p040⟧English prose… with {a} markers exactly where the French has them… ⟦p041⟧…

## Notes
- {a} (p040): **Seneca, *On Benefits*, book 4** — Seneque au liu. des benefices. [one-line gloss of the point cited, if recoverable]
- {b} (p041): **Digest 23.2.?? (*De ritu nuptiarum*, l. *Minorem*)** — l. minorem D. de ritu nup.
```

Rules:
- Keep every `⟦pNNN⟧` marker, in order, at the point in the English that corresponds to
  the page break in the French (a sentence may straddle a marker; put the marker at the
  corresponding word). Keep every `{x}` marker, in the same order, attached to the English
  word or clause that carries it in the French. A checker rejects the file if the marker
  sequences differ.
- Headings: translate `TEXTE.` as `TEXT` and `ANNOTAT. V.` as `ANNOTATION V` on their own
  line where they occur inside the section text (display lines such as the title-page
  lines are translated line by line).
- Paragraphs: keep the French paragraphing (`\n\n`).
- Names and places: Coras's forms per the case file (Martin Guerre, Bertrande de Rols,
  Arnaud du Tilh, Artigat, Rieux, Toulouse). Legal terms per the glossary.
- Register: the court record is formal and procedural; the annotations are discursive and
  learned; keep Coras's long periodic sentences readable by splitting only where English
  needs it, never by dropping a clause.
- Latin and Greek quotations in the French are translated into English, with the original
  kept in parentheses if short (one line) or in the notes if long.
- Every marginal citation gets a Notes entry: the citation expanded and identified in bold
  (Digest/Code/Institutes/Novels by book.title where identifiable; canon law by its
  standard reference; classical works by standard English title, book and chapter; French
  authors by name and work), then an em-dash and the original French, then an optional
  one-line gloss of what the cited passage says, only when you are confident. If a
  citation cannot be identified, say `**unidentified**` and give the original.
- The French was reflowed mechanically from printed lines. Where the print broke a word at a
  line end without a hyphen, the reflow leaves a space inside the word (`bouil lõna` =
  *bouillonna*, `paro les` = *paroles*); read through such gaps. `⟦pNNN⟧` may likewise fall
  inside a word.
- `⟨alt:…⟩` marks a reading the transcribers could not settle by eye: the word(s) before the
  marker are reading A and the marker holds reading B (`trabir⟨alt:trahir⟩`; `⟨alt:⟩` = the word
  may not be there; `⟨alt:+x⟩` = B has an extra word). Choose by context, translate the chosen
  reading only, and list every such choice in the report (French of both, which you took, why).
  `⟨alt?:…⟩` is the same, except that the transcribers vouch for neither reading: choose the
  likelier one by context as usual, and flag it in the report as unconfirmed.
- Your choices go back into the French transcription, so record each one machine-readably
  as well: write `{ALT_CHOICES_PATH}`, a JSON list with one object per `alt_id` in the table
  below, every one of them, in table order:
  `[{"alt_id": "p067-b6l2-1", "choice": "A", "reason": "cõtrainte = contrainte, 'constraint'; cõrrainte is not a word"}]`.
  `choice` is `"A"` (the word(s) before the marker, i.e. the line as `a` gives it), `"B"`
  (the marker's reading, the line as `b` gives it), or `"either"` when both readings
  translate the same and you have no basis to choose (the French then keeps reading A).
  `reason` is one short sentence. Several markers on one line (same `where`) are one choice
  of a whole line, so give them the same `choice`. Write the file even when the table is
  empty (then it is `[]`).

{ALT_TABLE}
- Obscure passages: translate literally and add `[unclear: …]` with a short note. Never
  invent, never smooth over.
- Do not add commentary beyond the Notes.

## Finish

<!-- single -->
Run: `uv run python scripts/check_markers.py {SECTION_ID}` and fix until it passes.
Write a short report to `translation/reports/{SECTION_ID}.md`: word counts (French,
English), `⟨alt⟩` choices (under a `## ⟨alt⟩ choices` heading, as in `{ALT_CHOICES_PATH}`), glossary additions, any `[unclear]` passages with the French, citations you could
not identify, and anything the reviewer should look at. Then return the same report.

RETURN ONLY a three-line summary to the coordinator (the full report lives in the file you
saved): line 1 the output path and validator/checker result; line 2 the counts; line 3 the
escalations or open questions, or "none".
<!-- /single -->
<!-- batch -->
After writing each section's file, run `uv run python scripts/check_markers.py <id>` for
that section and fix until it passes, before moving to the next section.
Write ONE report for the batch to `{REPORT_PATH}`: per section, word counts (French,
English), `⟨alt⟩` choices (under a `## ⟨alt⟩ choices` heading: French of both readings,
which you took, why; the same choices as `{ALT_CHOICES_PATH}`), any `[unclear]` passages with the French, citations you could
not identify; then the glossary additions and anything the reviewer should look at.

RETURN ONLY a three-line summary to the coordinator (the full report lives in the file you
saved): line 1 the sections written and the check_markers result for each (e.g. `12/12 ok`);
line 2 the counts; line 3 the escalations or open questions, or "none".
<!-- /batch -->

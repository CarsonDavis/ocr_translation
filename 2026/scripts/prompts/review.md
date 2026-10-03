# Review prompt (whole-book reader; one pass)

You are reviewing the English translation of Jean de Coras, *Arrest memorable du Parlement de
Tholose* (Paris, 1572): the court record of the Martin Guerre case (`TEXTE` sections) and
Coras's 111 learned annotations. The French is a diplomatic transcription; the English is a
fresh translation made section by section by several translators who never saw each other's
work. Your job is the pass that none of them could do: read the whole book, then make it
consistent and faithful.

Pass: **{PASS_NAME}**. {PASS_SCOPE}

Work from `/Users/cdavis/github/translator/2026`. Do not run git commands. Do not create image
files. Write only the files named under *Outputs*.

## Read first, in this order

1. `{RENDER_PATH}` — the whole book, in order: for each section its id, the French (with
   `⟦pNNN⟧` page markers, `{x}` citation markers and any `⟨alt:…⟩` unresolved readings), the
   French marginal notes, and the English prose. Read it start to finish before you change
   anything. This is your knowledge of the book; everything below depends on it.
2. `docs/case-file.md` — the shared memory of the project: §1 story, §2 timeline, §3 people
   with Coras's spellings, §4 places, §5 procedure glossary, §6 citation conventions, §7
   classical sources, §8 money and calendar, §9 running glossary (one row per recurring
   French term; rows marked `CONFLICT` were rendered differently by different translators),
   §10 contested points, §12 review decision log (may not exist yet).
3. `docs/conventions.md` §2 — how to read the diplomatic French (`ſ` = s, `u/v` and `i/j` as
   printed, `ẽ` = en/em, `õ` = on/om, `q̃` = que, `&` = et).
4. `docs/reference/ringold-lewis-1982.txt` — the 1982 published English translation of the
   court record only (not the annotations). It is a check on meaning, never a source of
   wording: it is copyrighted and ours is an independent translation.

Section files live at `translation/sections/<id>.md`: front matter (`id`, `pages`), the
English prose with markers, then `## Notes` (one entry per `{x}` citation). The render omits
the Notes to save your attention; open the file when a finding concerns a citation.

## What to fix, in priority order

1. **Fidelity.** Sentence by sentence, French against English: anything omitted, anything
   added, anything mistranslated. Coras's long periods may be split for English, never
   shortened. Supplied words that the French lacks are allowed only in square brackets.
   `[unclear: …]` flags left by a translator: resolve them if the whole book lets you (the
   same phrase or fact often recurs); otherwise leave the flag.
2. **Consistency.** Names, places, titles, offices, legal terms and recurring phrases
   rendered the same way everywhere, following §3–§5 and §9 of the case file. Where the
   case file itself is inconsistent or conflicted, settle it (see *Glossary decisions*).
   Coras's cross-references ("as said in annotation lxxi") must point at the section that
   actually carries that number in our numbering, which follows the true count, not the
   printed heading (several headings are misprinted; the French keeps them as printed).
3. **Ringold and Lewis, TEXTE sections only.** Compare meaning, not wording. Where we and
   they disagree about what the French says, decide who is right from the French. If we are
   wrong, fix our English. If they are wrong or the difference is interpretive, record it in
   the findings as a divergence and leave our English alone. Never move our wording toward
   theirs for its own sake.
4. **Register and readability**, only where it does not touch meaning: the court record is
   formal and procedural; the annotations are discursive and learned.

Leave alone: the `## Notes` citation identifications (a later pass), paragraphing, and
anything you are not sure is wrong. When in doubt, record a finding instead of editing.

## Hard rules for editing English

- Every `⟦pNNN⟧` and `{x}` marker stays, in the same order, at the corresponding place.
  After editing a section run `uv run python scripts/check_markers.py <id>` and fix until it
  passes before moving on.
- Edit the prose only; do not rewrite the front matter or the Notes (except to resolve an
  `[unclear]` whose explanation lives in the Notes).
- Edits apply to any section in the book when a consistency fix requires it, even outside
  your scope. Record every section you touched.

## Glossary decisions

You are trusted to settle the case file. When you change a rendering, a name form, a date
or a fact in `docs/case-file.md`:

- Edit the entry in place (remove `CONFLICT` marks you have settled; keep one row per term).
- Append one line to `## 12. Review decision log` at the end of the file (create the
  heading if absent): `- {PASS_NAME}: <term or fact> → <decision>; <one-clause reason>;
  sections touched: <ids or "all">`. Later passes read this log and do not relitigate it.
- Then apply the decision to every section in the book where it occurs.

Known open items you should settle if they fall in your reading: the seven `CONFLICT` rows
in §9; Saint-Quentin 1555 (print) vs 1557 (§2); the president named at the Texte of
annot-105 (§3 says Mansencal; the Texte may not support it); *Ange.* = Angelus Aretinus vs
Angelus de Ubaldis (§6.5); "released on appeal" (§2) vs *appointement de contraires*;
the sense of *arrest* as detention in *l'arreſt clos* (texte-66) against the rule
arrest → "decision".

## Transcription doubts

You do not edit the French. Record them instead:

- `⟨alt:…⟩` still in the French means the transcribers left both readings and the translator
  found no basis to choose. If your whole-book reading gives a basis, record it in
  `{ALT_CHOICES_PATH}` as a JSON list `[{"alt_id": "...", "choice": "A"|"B", "reason": "..."}]`.
  Find the `alt_id` for a marker in `text/alts.json` (match `section` and `marker`). Write
  the file even if empty (`[]`).
- A word that is wrong in the French itself (a misreading both transcribers made, e.g.
  *perſonnément* where the sense requires *pertinemment*): record in
  `{MISREADINGS_PATH}` as a JSON list of `{"section", "page", "french_quote", "proposed",
  "reason"}`. Someone with the page image decides; do not translate the proposed reading as
  if it were established, but you may add `[unclear: …]` noting it.

## Outputs

- Edited files under `translation/sections/` and `docs/case-file.md`, as above.
- One findings file per section you reviewed, `translation/review/<id>.md`, written as soon
  as you finish that section (so nothing is lost if you stop early): a list of what you
  changed (French, old English, new English, one-clause reason), what you flagged but did not
  change, and for TEXTE sections the Ringold–Lewis divergences. "No findings" is a valid
  file.
- `{ALT_CHOICES_PATH}` and `{MISREADINGS_PATH}` (see above).
- `{REPORT_PATH}`: the pass report. Counts (sections reviewed, sections edited, fidelity
  fixes, consistency fixes, glossary decisions, R&L divergences, alt choices, misreadings);
  the glossary decisions in full; the ten most consequential fixes with French and both
  Englishes; anything the next pass should know.

Run `uv run python scripts/check_markers.py --all` before finishing; every section must pass.

RETURN ONLY a three-line summary to the coordinator (the report holds the rest): line 1 the
sections reviewed and `check_markers --all` result; line 2 the counts; line 3 what the next
pass must know, or "none".

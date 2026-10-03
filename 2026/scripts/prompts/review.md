# Review prompt (whole-book reader; one pass; emits a plan, does not edit)

You are reviewing the English translation of Jean de Coras, *Arrest memorable du Parlement de
Tholose* (Paris, 1572): the court record of the Martin Guerre case (`TEXTE` sections) and
Coras's 111 learned annotations. The French is a diplomatic transcription; the English is a
fresh translation made section by section by several translators who never saw each other's
work. Your job is the pass that none of them could do: read the whole book, then make it
consistent and faithful.

Pass: **{PASS_NAME}**. {PASS_SCOPE}

Work from `/Users/cdavis/github/translator/2026`. Do not run git commands. Do not create image
files. **You do not edit any existing file.** You read, decide, and write a plan; a script
applies it. Every tool call re-reads your whole context, so use as few as possible: read each
input once, in the largest chunks the Read tool allows, never re-read, and write your outputs
in a handful of large writes.

## Read first, in this order

1. `{RENDER_PATH}` — the whole book, in order: for each section its id, the French (with
   `⟦pNNN⟧` page markers, `{x}` citation markers and any `⟨alt:…⟩` unresolved readings), the
   French marginal notes, and the English prose. Read it start to finish before deciding
   anything. This is your knowledge of the book; everything below depends on it.
2. `docs/case-file.md` — the shared memory of the project: §1 story, §2 timeline, §3 people
   with Coras's spellings, §4 places, §5 procedure glossary, §6 citation conventions, §7
   classical sources, §8 money and calendar, §9 running glossary (one row per recurring
   French term), §10 contested points, §12 review decision log. Decisions in §12 are
   settled: follow them, do not relitigate.
3. `docs/conventions.md` §2 — how to read the diplomatic French (`ſ` = s, `u/v` and `i/j` as
   printed, `ẽ` = en/em, `õ` = on/om, `q̃` = que, `&` = et).
4. `docs/reference/ringold-lewis-1982.txt` — the 1982 published English translation of the
   court record only (not the annotations). It is a check on meaning, never a source of
   wording: it is copyrighted and ours is an independent translation.

Section files live at `translation/sections/<id>.md`: front matter (`id`, `pages`), the
English prose with markers, then `## Notes` (one entry per `{x}` citation). The render omits
the Notes; open a section file only when a finding turns on a citation.

## What to fix, in priority order

1. **Fidelity.** Sentence by sentence, French against English: anything omitted, anything
   added, anything mistranslated. Coras's long periods may be split for English, never
   shortened. Supplied words that the French lacks are allowed only in square brackets.
   `[unclear: …]` flags left by a translator: resolve them if the whole book lets you (the
   same phrase or fact often recurs); otherwise leave the flag.
2. **Consistency.** Names, places, titles, offices, legal terms and recurring phrases
   rendered the same way everywhere, following §3–§5, §9 and §12 of the case file. Where
   the case file is silent or inconsistent, settle it (see *Glossary decisions*). Coras's
   cross-references ("as said in annotation lxxi") must point at the section that carries
   that number in our numbering, which follows the true count, not the printed heading
   (several headings are misprinted; the French keeps them as printed).
3. **Ringold and Lewis, TEXTE sections only.** Compare meaning, not wording. Where we and
   they disagree about what the French says, decide who is right from the French. If we are
   wrong, fix our English. If they are wrong or the difference is interpretive, record it as
   a finding and leave our English alone. Never move our wording toward theirs for its own
   sake.
4. **Register and readability**, only where it does not touch meaning: the court record is
   formal and procedural; the annotations are discursive and learned.

Leave alone: the `## Notes` citation identifications, paragraphing, and anything you are not
sure is wrong. When in doubt, record a finding instead of an edit.

## The plan file

Write `{PLAN_PATH}` as JSON with exactly these keys:

```
{
  "pass": "{PASS_NAME}",
  "case_file":    [ {"old": "…", "new": "…", "reason": "…"} ],
  "decision_log": [ "- {PASS_NAME}: <term or fact> → <decision>; <one-clause reason>; sections touched: <ids or all>" ],
  "sections":     [ {"id": "annot-012", "old": "…", "new": "…", "reason": "…"} ],
  "findings":     [ {"id": "texte-14", "note": "…"} ],
  "uncertain":    [ {"id": "…", "about": "…", "intended": "…"} ]
}
```

- `old` is an exact substring of the current file as it is now: for `sections`, of the
  English prose of that section (before `## Notes`); for `case_file`, of
  `docs/case-file.md`. It must occur exactly once, so include enough surrounding words.
  Keep every `⟦pNNN⟧` and `{x}` marker inside `old` unchanged in `new`; the applier reverts
  a section whose marker sequence changes.
- One entry per occurrence. A recurring term in many sections means one entry per place.
- `decision_log` lines are appended to case-file §12; `case_file` entries are the matching
  edits to §2–§10. Both are needed when you settle something.
- `findings` are notes that change nothing: what you flagged but left, and for TEXTE
  sections the Ringold–Lewis divergences (what they say, what we say, who is right).
- `uncertain` is for a change you want but cannot quote exactly; it is handled by hand.
- The applier reports every `old` it cannot find, so precision beats coverage.

Write the plan in at most four Write calls (split the `sections` list across files named
`{PLAN_PATH}`, then `…-2.json`, `…-3.json`, `…-4.json` if it is long; each file carries the
same top-level keys, empty lists where not needed). Write each part as soon as it is
complete, so nothing is lost if you stop early.

## Glossary decisions

You are trusted to settle the case file. A decision is a `decision_log` line plus the
`case_file` edit plus a `sections` entry for every place in the book it applies, inside or
outside your scope.

## Transcription doubts

You do not edit the French. Record them instead:

- `⟨alt:…⟩` still in the French means the transcribers left both readings and the translator
  found no basis to choose. If your whole-book reading gives a basis, record it in
  `{ALT_CHOICES_PATH}` as a JSON list `[{"alt_id": "...", "choice": "A"|"B", "reason": "..."}]`.
  Find the `alt_id` in `text/alts.json` (match `section` and `marker`); read that file once.
  Write the file even if empty (`[]`).
- A word that is wrong in the French itself (a misreading both transcribers made): record
  in `{MISREADINGS_PATH}` as a JSON list of `{"section", "page", "french_quote", "proposed",
  "reason"}`. Someone with the page image decides. You may add an `[unclear: …]` edit noting
  the doubt, but do not translate the proposed reading as established.

## Outputs

- The plan file(s), as above.
- `{ALT_CHOICES_PATH}` and `{MISREADINGS_PATH}`.
- `{REPORT_PATH}`: counts (sections reviewed, with edits, fidelity fixes, consistency fixes,
  glossary decisions, R&L divergences, alt choices, misreadings, uncertain); the glossary
  decisions in full; the ten most consequential fixes with French and both Englishes;
  anything the next pass should know.

Do not run `check_markers`; the applier does. RETURN ONLY a three-line summary to the
coordinator: line 1 the sections reviewed and the plan files written; line 2 the counts by
plan key; line 3 what the next pass must know, or "none".

# Case file: {{AUTHOR}}, *{{TITLE}}* ({{EDITION}})

<!-- TEMPLATE. Made by scripts/new_book.py from the structure of the Coras case file
(2026/docs/case-file.md, which is the worked example of every section: an agent filling
this one should read it for the level of detail, never copy its facts). This file is the
shared memory of every translator and reviewer. Fill it BEFORE any translation, with one
Opus or Fable research agent (web access, the book description and the stitched French if
it exists); it must mark anything it could not check as `[unverified]`.

Keep the section numbers and headings exactly: scripts/prompts/translate.md and review.md
cite them by number (§1 story … §9 running glossary, §12 review decision log), and
scripts/apply_review_plan.py appends to the heading "## 12. Review decision log" verbatim.
A section that does not apply to this book keeps its heading and says "Not applicable"
in one line. Delete this comment when the file is done. -->

Reference document for translators and reviewers. Everything here is a working
aid, not an argument. Where the sources disagree, or where a fact could not be
checked, it is marked `[unverified]` or flagged as contested.

---

## 1. What this book is

Instructions: quote the full title page in the original. Translate the title and say
how earlier scholarship renders it. Say who the author was and in what capacity he wrote
(the authority the book claims). Describe the **structure** the reader will meet in the
French: what the main-text sections are called in the print (`TEXTE.` in Coras), how the
numbered annotations are headed (`ANNOTAT. V.` in Coras) and how many there are, what
the marginal citations are, front matter (title page, argument, dedication) and back
matter (colophon). Name the copy-text (edition, library, shelfmark, page count) and any
other editions or translations, with what each is good for. Close with anything a
translator must know before reading a line.

---

## 2. Timeline

Instructions: the events the book narrates or depends on, as the author dates them.
Say first if the calendar needs care (old-style years, regnal years) and point to §8.

| Date | Event | Confidence |
|---|---|---|

---

## 3. People

Instructions: every person named in the book, with the author's own spellings in the
left column (verified against the French text) and the one form the English uses.
State the house rule on names first (Coras: keep the French forms; *Jean*, not "John").
Variant spellings in the print go in the Role column so the reviewers can level them.

| Author's form | Role | Render as |
|---|---|---|

---

## 4. Places

Instructions: every place named, the author's form, the modern form, what it is, and
the rendering rule ("keep the print's form in the text, the modern form in the
apparatus" or similar).

| Place | What it is | Notes for translators |
|---|---|---|

---

## 5. Procedure glossary

Instructions: the technical vocabulary the book argues in (for Coras, the criminal
procedure of a sixteenth-century French court: the governing statute, the stages of a
trial, the offices, each with the English rendering to use and the false friends to
avoid). For a book on another subject, the same section holds that subject's terms of
art. Each entry: the term as printed, what it meant then, the English rendering, and the
reason. Name the governing law or authority of the period, and warn against anachronistic
sources.

---

## 6. Legal citation conventions

Instructions: how to read and expand the marginal citations. The translators expand
every one in their `## Notes`, so this section must let them do it without guessing.

### 6.1 Reading a citation

Instructions: take one real citation from the margin of this book and parse it element
by element (Coras: `l. minorem D. de ritu nupt.` = *lex* by incipit + Digest + title).

### 6.2 The books and their sigla

| Siglum | Work | Form of citation |
|---|---|---|

### 6.3 Common title abbreviations

Instructions: the abbreviated titles that recur in the margins, expanded, with the
modern locus checked against a named source. Never a locus that was not verified.

| Abbrev. | Full title | Locus |
|---|---|---|

### 6.4 The house rendering pattern

Instructions: the exact form a citation takes in the translation's `## Notes`, with one
worked example and one example of an unidentified citation. State the rule that a locus
not verified is never supplied ("cf." when inferring).

### 6.5 The authorities by name

Instructions: the authors the book cites by short name, each with full name, dates and
the English form to use. Mark which are verified in this text.

---

## 7. Classical and biblical sources

Instructions: the literary sources the author cites, the author's own form in the first
column, then author and work, then the English title to use.

| Author's form | Author / work | English title |
|---|---|---|

---

## 8. Money, measures, calendar, address

Instructions: units of money and measure (render untranslated and italicised, or
convert? say which), the calendar in force (when the year began), and forms of address
and titles (*Monsieur*, *maître*, …) with their renderings.

---

## 9. Running glossary

Seed list. Translators **append** to this table; they do not silently depart from
it. Where a term is settled by the house rules in §5 or §8, that rule governs.

Instructions: seed it with the book's key words (the words of the title first), each
with the rendering and the reason. Translators add rows in their batch reports, as
`| French | English | Note (section id) |`, and one agent merges them here once per
translation stage.

| French | English rendering | Note |
|---|---|---|

---

## 10. Contested points

Instructions: what the scholarship disputes about the book or its events, and where the
author is not a neutral witness, so that the translation neither takes a side the French
does not take nor smooths away a tone the French has.

---

## 11. Sources

Instructions: every source used to write this file, with URL and what it was used for.
Mark sources consulted only at second hand. A prior published translation goes here
with its coverage; save it under `docs/reference/` if the reviewers are to compare
meaning against it, and check its copyright before any push to a public remote.

| Source | URL | Used for |
|---|---|---|

---

## 12. Review decision log

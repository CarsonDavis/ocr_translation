# Read report — p045, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p045.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p045.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| item | count |
|---|---|
| body (paragraph) lines | 28 |
| paragraphs | 4 |
| headings | 3 (`TFXTE.`, `ANNOTAT, XXIIII.`, `TEXTE.`) |
| markers in body | 2 (`{d}`, `{a}`) |
| margin notes | 1 (key `a`, 8 lines) |
| foot notes | 0 |
| ornaments | 0 |
| signature / catchword | none / none |
| `uncertain[]` entries | 17 (2 flagged `escalate`) |

Running head `PARLEMENT DE THOLOSE.`, folio `44` (matches the manifest).

## Page structure

1. paragraph, 1 line, `continues_prev` (finishes p044's sentence "…& non point en | vaines ſupperſtitions…"), ends with marker `{d}`
2. heading `TFXTE.` (spaced caps)
3. paragraph, 9 lines — the large-type TEXTE
4. heading `ANNOTAT, XXIIII.` (spaced caps)
5. paragraph, 5 lines — the small-type annotation, ends with marker `{a}`
6. heading `TEXTE.` (spaced caps)
7. paragraph, 13 lines — the large-type TEXTE, `continues_next`

## `uncertain[]` entries, one line each

1. `blocks[0].lines[0]` — marker `{d}` has **no note** anywhere on the page (margin blank above it, foot blank); it follows p044's `{c}`, which also had no note.
2. `blocks[0].lines[0]` — the mark before `{d}` is a period (round dot on the baseline, no tail) not a comma.
3. `blocks[1]` — **sic `TFXTE.`**: the second sort is a two-armed F, not the three-armed E of the same word's fifth sort; a wrong sort, transcribed as printed.
4. `blocks[3]` — **sic `ANNOTAT, XXIIII.`**: comma (tailed) after ANNOTAT, and the numeral counts X X I I I I = 24, repeating p044's `ANNOT. XXIIII.`
5. `blocks[2].lines[5]` — `confrontemẽs`: the diacritic is a flat tilde (matches `attẽdu`/`ordõnance`), not this fount's steeply slanted acute.
6. `blocks[2].lines[4]` — `adiouſter` printed `adio uſter` with a full word space inside the word; joined per §1.
7. `blocks[2].lines[2]` — justification space before the comma, normalized away (six further instances listed).
8. `blocks[4].lines[3]` — no space after the comma, normalized in (six further instances listed).
9. `blocks[6].lines[0]` — a pale brown speck over the second e of `paracheuez`; read as a speck, plain e.
10. `margin_notes[0].lines[0]` — **ESCALATE**: the over-inked third sort read as `§`; shape and the legal formula support §, but a capital B is the rival reading.
11. `margin_notes[0].lines[2]` — the line-end mark after `ex` read as the word-break hyphen of `ex|hib`.
12. `margin_notes[0].lines[4]` — **ESCALATE**: one or two high specks after `literas`, too high for a period and not comma-shaped; no punctuation transcribed. (Also records that `miſſa` is ſſ.)
13. `margin_notes[0].lines[3]` — sic `tran|miſſa` (no hyphen, and no s of *transmissa*); narrow points spaced per §1.
14. `margin_notes[0]` — placement: the page's only note has drifted far below its marker (first line sits between `ſubornation. {a}` and the `TEXTE.` heading).
15. `blocks[6].lines[12]` — last line of the page, runs on to p046; foot blank; `circõuoiſin s` joined; `neceſſaires` is ſſ.
16. `blocks[6].lines[11]` — sic `que a` (word space, no apostrophe).
17. `blocks[4].lines[0]` — `deſſus` confirmed ſſ (both sorts tall).

## For the reconciler

- **Two missing citations in a row.** p044's `{c}` had no note and p045's `{d}` has no note either. The only note on p045 is keyed `a`, i.e. the alphabet restarts at the new `ANNOTAT` section. Either the printer dropped the c/d citations, or they were meant to stand in a margin that was left blank. Worth a human check across the opening.
- **Annotation number repeats.** p044 heads its annotation `ANNOT. XXIIII.` and p045 heads its own `ANNOTAT, XXIIII.` — the same number, 24, on consecutive annotations, and spelled/pointed differently. One of the two is a misprint. Both are transcribed as printed.
- **`TFXTE.`** (blocks[1]) is a genuine wrong sort, not a reading error; the second `TEXTE.` on the same page is set correctly, which makes the comparison easy at 4x.
- The two escalated readings are both in the single margin note (the `§` sort and the mark after `literas`); the note is small italic and over-inked, but otherwise legible throughout.
- Page condition is good: no damage, no gutter loss, ink even. The compositor sets this page loosely — justification spaces before commas and missing spaces after them are frequent, and two words (`adio uſter`, `circõuoiſin s`) are split by a full space; all normalized per §1 and itemised in `uncertain[]`.

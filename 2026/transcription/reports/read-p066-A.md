# Read report — p066, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p066.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p066.json` → `1 ok, 0 failed` (exit 0, no warnings).

## Counts

| item | count |
|---|---|
| body lines | 30 (9 + 13 + 8) |
| paragraphs | 3 |
| headings | 2 (`ANNOTAT. XLIII.`, `TEXTE.`, both spaced caps) |
| markers in body | 5 (`{a}` `{b}` `{c}` `{d}` `{e}`) |
| margin notes | 5 (keys a–e, all matched to markers) |
| foot notes | 0 |
| ornaments | 0 |
| uncertain[] entries | 6 |

Running head `ARREST DV` (spaced caps), folio `66` (matches the manifest), no signature, no
catchword, no foot citation block — `foot.jpg` shows the last two body lines and then bare
paper down to the trimmed edge; `body-8.jpg` is blank.

## Page structure

1. paragraph, `continues_prev: true`, 9 lines of large text type, ends `nie de ladite de Rols.`
2. heading `ANNOTAT. XLIII.`
3. paragraph, 13 lines of the smaller annotation type, markers a–e, ends `en eſcrirons peu apres {e}.`
4. heading `TEXTE.`
5. paragraph, `continues_next: true`, 8 lines of large type, ends `tin Guerre ſon mari, ou quelque dia-`

Margin column: notes a–e in italic, in one run beside the annotation paragraph; nothing
beside the two large-type paragraphs. Below note e the margin is empty.

## uncertain[] entries (6)

1. `blocks[0]` — `continues_prev` set true (page opens mid-name/mid-sentence, "…du | Tilh
   priſonnier"), but p065 is not transcribed yet, so the join is unverified.
2. `blocks[2].lines[4]` — `preſumptioo`: sic, final letter is a closed round sort (o), not the
   n used elsewhere on the page; wrong sort, transcribed as printed.
3. `blocks[2].lines[11]` — `ès lieux`: the accent is a near-vertical wedge, between the page's
   clear acute (verité, chargé) and its clear grave (à, où); read as grave, `és` possible.
4. `margin_notes[0].lines[0]` — `I. manife.`: the citation opens with an italic capital I where
   the lex abbreviation `l.` is expected (its flat top/bottom serifs differ clearly from the
   lowercase l of notes b and c); also `manife.` ends in a round period, not a hyphen.
5. `margin_notes[3].lines[3]` — `leguee.`: worn and over-inked; read as `leguee` continuing
   `al` from the line above (alleguee). A mark over the middle letters may be an accent or ink.
6. `margin_notes[4].lines[1]` — `tation.. xv.`: the print reads `tation. .xv.`; the §1 spacing
   rule closes the gap before the numeral's own leading point, so the two points end up adjacent.

## Notes for the reconciler

- Checked explicitly for `ſſ` vs `ſs` at 3–8x: `aſſez`, `confeſſion`, `aſſeurant` are all the
  `ſſ` ligature; `auſsi.` (block 0, line 6) is long s + round s. `tesfois` (block 2, line 7) has
  a round s before f, as the conventions predict.
- Punctuation checked glyph by glyph at the clause boundaries: `iurer:` (colon), `plainte:`
  (colon), `point:` (colon), `partie {c}.` (period, not comma), `auſsi.` (period).
- Tildes: `ſuſmettãt`, `cõtre`, `quãd` — all abbreviation marks over the vowel, transcribed
  precomposed. `creuëment` carries a true dieresis.
- Two line-end word breaks are printed without a hyphen and are transcribed as printed:
  `…ni la calom` / `nie de ladite…` (block 0) and `…la verité, defe` / `rer le ſerment…`
  (block 2). Note b likewise runs `licen` / `tia` unhyphenated.
- Damage/condition: a brown stain and a torn/chipped area run along the top-left of the leaf,
  but they stay in the blank margin and touch no type. Ink is even; no faint passages.
  The gutter strip of the facing recto is visible at the right edge of the crops and was ignored.
- The marker letters run a–e with no restart, so the next page should continue at `f` if this
  stretch of text carries on.

# Read report — p061, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p061.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p061.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| body lines | 30 (7 + 18 + 5) |
| paragraphs | 3 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XL.`, both spaced caps) |
| markers in body | 2 (`{f}`, `{g}`, both in the first paragraph) |
| margin notes | 2 (keys `f`, `g`) |
| foot notes | 0 |
| ornaments | 0 |
| signature / catchword | none / none |
| running head / folio | `PARLEMENT DE THOLOSE.` / `61` |
| uncertain entries | 4 (2 escalated) |

Cross-check: every `{x}` has a note and every note has a marker (f, g). Paragraph line
counts checked against `pages/read/p061.jpg`.

## Layout

Recto, one column. Tail of the previous annotation (7 lines, continues from p060, opens
mid-sentence with "de, peut produire…"), then the heading `TEXTE.`, then the *Texte* set in
large type (18 lines, self-contained, ends "trouuées au priſonnier."), then the heading
`ANNOTAT. XL.` and the first 5 lines of that annotation, which run over onto p062
(`continues_next: true`, last line ends "appelé VERRVCOSVS"). Both margin notes sit beside
the first paragraph; the whole lower margin and the foot are blank. Roughly the bottom
quarter of the leaf is empty paper — no foot citation block, no signature, no catchword.

## uncertain[] entries

1. **`blocks[2].lines[11]` — escalated.** A small raised ink mark between "machoire" and
   "de". Not transcribed as a marker: no note on the page could key to it (only f and g are
   printed, both beside the first paragraph; margin-2/3/4 are blank), and no `TEXTE` block in
   any finished page of this edition carries a marker. If the reconciler rules it a marker,
   the key would be `h` and the note is missing.
2. **`blocks[4].lines[0]`.** No point is printed after the praenomen `Q` ("de Q Fabius
   Maximus"); a capital Q with a long flat tail, then a wide blank. Transcribed as printed.
3. **`blocks[2].lines[2]`.** sic: the surname is set `Rols` on line 3 (full-height dotless
   l) but `Rois` on line 7 (short dotted i). Both as printed — the variation is the
   compositor's.
4. **`margin_notes[1].lines[3]` — escalated.** Note g ends `ſi. D. de fut.`; the last sort
   is the curl-topped italic **t** (same sort as the t of "penulti-" two lines above), not
   the `r` the expected law-title abbreviation *de furtis* would want.

## Notes for the reconciler

- Checked explicitly at 4–10x per the prompt's error classes: `aſſeure` and `deſſus` are
  both true `ſſ` (two long s), not `ſs`; `toutesfois` and `toutes` take round s before f as
  the convention predicts. Sentence punctuation verified by shape — colon after `frere`,
  after `defendant {f}` and after `l'accuſé`; a clean round period (no tail) after
  `auec elle`; period before the marker in `conſideration. {g}`.
- Wrong-sort sweep found no misprint except the `Rols`/`Rois` pair above. `Guerre` is a
  clean G; `cicatrice` is correct but its second `c` is very lightly inked and could read as
  a broken sort at low zoom.
- Marker `f` is followed immediately by a colon; `g` follows the closing period of the
  paragraph. Both keys are printed with a point in the margin (`f.`, `g`), which per §4 is
  excluded from `lines`.
- Note g's `P.` is the paragraph-sign sort, transcribed `P.` to match the finished pages
  (p041, p050); note f's `P` in "l. Parentes." is an ordinary italic capital.
- Markers continue the alphabet at f–g. The preceding pages p053–p060 are not yet
  transcribed, so the continuity could not be checked against a previous page; the last
  finished page before this stretch (p052) ended at `b`.
- No damage, no gutter loss, no faint patches beyond the light `cicatrice` impression. A
  brown stain crosses the top margin and the running head but does not obscure any letter.

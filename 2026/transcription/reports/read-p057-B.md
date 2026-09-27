# Read report — p057, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p057.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p057.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| item | count |
|---|---|
| body lines | 37 (3 + 34) |
| paragraphs | 2 |
| headings | 1 (`ANNOT. XXXVI.`, spaced caps) |
| markers in body | 9 (a b c d e f g h i) |
| margin notes | 9 (a b c d e f g h i) |
| foot notes | 0 |
| signature / catchword | none |
| ornaments | none |
| uncertain[] entries | 15 |

Layout: running head `PARLEMENT DE THOLOSE.` (spaced caps) with folio `57` at the right.
Block 1 is the three-line tail of the paragraph carried over from p056 (`continues_prev`),
ending `mateur du nom de Dieu.`. Then the heading `ANNOT. XXXVI.`. Then one 34-line
paragraph that runs to the foot of the column and is cut off mid-sentence
(`… legereté d'eſprit, ou mau`), so `continues_next` is true. The margin column carries all
nine citations; there is no foot citation block, no signature and no catchword, and the
space below the last body line is empty.

## uncertain[] entries (15)

1. `blocks[2].lines[3]` — `Theologiẽs`: the mark is a horizontal wavy tilde, not the slanted acute of `eſpuiſées`.
2. `blocks[2].lines[18]` — `Sauuenr` **sic** (wrong sort: n for u in `Sauueur`).
3. `blocks[2].lines[21]` — `anciéne`: the mark is slanted (acute), unlike the tilde of `condẽnent` on the next line, though the sense wants `ancienne`.
4. `blocks[2].lines[24]` — `vier` **sic**: second letter is a dotted i, not long s; sense wants `vſer`.
5. `blocks[2].lines[28]` — `facilemeot` **sic** (wrong sort: o for n).
6. `margin_notes[0].lines[0]` — `Iean d'[?]na`: the letter after `d'` is badly inked and matches no clear sort (far too small for the italic capital A of `Aut,`); with the next line's `nie` the name is probably `Iean d'Ananie` (Ioannes de Anania). **ESCALATE**
7. `margin_notes[0].lines[3]` — `S. Tcmas` **sic** (open c where `Tomas` needs o).
8. `margin_notes[0].lines[5]` — `quaſtion` **sic** for `queſtion`.
9. `margin_notes[1].lines[1]` — `c de religioſ`: a lowercase c stands in the key column, read as the Codex abbreviation continuing note b (`Aut, alearũ, C. de religioſ`), not as a second note keyed c — the key c belongs to `leuitique. / c. xxiiij.` below and a duplicate key is impossible. **ESCALATE**
10. `margin_notes[3].lines[0]` — `S. Iean. e. x.`: the single faint letter may be `c` (= chapitre), as in the neighbouring notes.
11. `margin_notes[4].lines[1]` — `c xiiij.`: chapter number printed xiiij where note c cites the same book as `c. xxiiij.`; transcribed as printed.
12. `margin_notes[5].lines[1]` — `guinis. cx iij.`: numeral doubtful (first sort is an open c, not x, and a space follows); the expected citation is `xxiij. q. v.`
13. `margin_notes[6].lines[6]` — `iiij-collat.`: the mark is a raised horizontal stroke matching this italic's line-end hyphens, not the round baseline dot used for points elsewhere in the note; it may still be a period.
14. `margin_notes[7].lines[2]` — `maieù.`: reads as u with a slanted accent; the citation wants `maieſt.` (ad legem Iuliam maiestatis). **ESCALATE**
15. `margin_notes[8]` — key: only a dot prints in the key column; read as key `i` from the body marker `{i}` on line 32, which otherwise has no note.

## Notes for the reconciler

- **Ink quality.** The margin column is much more lightly inked than the body on this
  copy, and the first note (a) is the worst: its first line is close to illegible at native
  resolution. Every margin reading above 3x was checked twice; items 6, 9, 10, 12, 13, 14
  are the ones a second reader is most likely to differ on.
- **Wrong sorts.** This page has an unusually high count — four in the body/margin
  (`Sauuenr`, `vier`, `facilemeot`, `Tcmas`) plus the probable `quaſtion`. All are
  transcribed as printed with a `sic` note; none were corrected.
- **Note-b grouping** (item 9) is the one structural judgment on the page: if the
  reconciler reads that `c` as a key, the page has two notes keyed `c` and the marker
  cross-check breaks, so the grouping adopted here is the only self-consistent one.
- **Tilde vs acute.** The setting distinguishes them clearly at 4x: the tilde is a
  horizontal wavy bar (`Theologiẽs`, `ledicẽce`, `condẽnent`, `diſoiẽt`, `pẽſe`,
  `iuſtemẽt`, `Autremẽt`, `mẽt`), the acute is a short slanted stroke (`eſpuiſées`,
  `obſeruées`, `gardé`, `anciéne`). `anciéne` (item 3) is therefore recorded with an acute
  even though the word wants a nasal.
- **`ſſ` vs `ſs`.** Every double s on the page was zoomed: `laiſſant`, `gliſſement`,
  `paſſion` are all long s + long s. No `ſs` occurs.
- Line-end word breaks: `af-`, `Blaſphema-`, `religieu-`, `extraordinaire-`, `proce-`
  carry hyphens; `blaſphe`/`mateur`, `Tetragramma`/`ton`, `ve`/`rité`, `circonſtan`/`ces`
  and the page-final `mau` break with no hyphen, as this print often does.
- The gutter strip of the facing page (left edge on this recto) was ignored throughout.

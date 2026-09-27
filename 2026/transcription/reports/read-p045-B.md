# Read report — p045, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p045.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p045.json`
→ `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| body lines | 29 (1 + 9 + 5 + 13) |
| paragraphs | 4 |
| headings | 3 (`TFXTE.`, `ANNOTAT, XXIIII.`, `TEXTE.`) |
| markers in body | 2 (`{d}`, `{a}`) |
| margin notes | 1 (key `a`, 8 lines) |
| foot notes | 0 |
| ornaments | 0 |
| running head / folio | `PARLEMENT DE THOLOSE.` / `44` (matches manifest) |
| signature / catchword | none / none |
| `uncertain[]` entries | 15 (3 escalated) |

## Page structure

1. paragraph (1 line), `continues_prev: true` — finishes p044's ANNOT. XXIIII, ends `inutilles. {d}`
2. heading `TFXTE.` (spaced caps) — **sic, the second sort is an F**
3. paragraph, 9 lines, larger roman (the quoted arrest text)
4. heading `ANNOTAT, XXIIII.` (spaced caps)
5. paragraph, 5 lines, small roman (the annotation), ends `ſubornation. {a}`
6. heading `TEXTE.` (spaced caps)
7. paragraph, 13 lines, larger roman, `continues_next: true` — ends `tãt ſur la`, mid-sentence, no hyphen

Margin note `a` begins beside `ſubornation. {a}` and runs down past the second `TEXTE.` heading.

## uncertain[] entries — one line each

1. **`blocks[0].lines[0]` — marker `{d}` has no note anywhere (ESCALATED).** margin-1 blank, margin-2 has only note `a`, margin-3 blank, foot blank; p044's `{c}` was likewise noteless, then the alphabet restarts at `a` here.
2. `blocks[1]` — heading printed `TFXTE.`: the second sort has top and middle bars but no bottom bar, unlike the E's elsewhere on the page; transcribed as printed.
3. `blocks[2].lines[5]` — `confrontemẽs`: the mark is a flat wavy bar (tilde), not the slanted acute of `gardées` / `monſtré`; the word is set out in full as `confrontemens` at `blocks[6].lines[0]`.
4. **`blocks[3]` — `ANNOTAT, XXIIII.` (ESCALATED).** Separator is a comma (descends below the baseline, 15px vs 9px for points); the numeral segments as X X I I I I = 24, repeating p044's ANNOT. XXIIII where XXV is expected. Transcribed as printed.
5. `blocks[4].lines[4]` — marker after `ſubornation.` is an unambiguous small italic `a`; alphabet restarts with this section, so no `a2` is needed.
6. `blocks[4].lines[1]` — omnibus spacing note: justification spaces before commas on seven lines, missing spaces after commas on five, `adio uſter` and `circõuoiſin s` closed up; all normalized per §1.
7. `blocks[6].lines[0]` — a detached round blob at ascender height inside `paracheuez`; read as a speck/show-through, not an accent.
8. `blocks[6].lines[9]` — the `a` of `la` (in `la verité:`) carries a hairline stroke; thinner and lighter than the acutes of `verité` on the same line and absent from `la matie` above; read as an ink filament, transcribed `la` (reader A may read `là`). Final mark is a colon.
9. `blocks[6].lines[12]` — `ſſ`/`ſs` check: `neceſſaires`, `deſſus` and `miſſa` are all true `ſſ` at 4.5x; page breaks off mid-sentence, `continues_next: true`.
10. **`margin_notes[0].lines[0]` — the sign in `l. iij. §. ſi ve` (ESCALATED).** Over-inked blob, x-height (35px, same as the neighbouring `a`, no ascender, no descender), ~38px wide. Reads to the eye as a bold italic `B`, but a capital B would reach the ascender line. Transcribed `§` on the strength of the standard citation form `l. iij. § ſi verò vtraque. D. de liberis exhibendis`; alternatives are `B` and `ß`. **No line anywhere in `transcription/final/` uses `§` — the corpus spells it `parag.` (p004, p006) — so this sort may be unprecedented in this book.** Same line: `iij` (three strokes, three dots at 12x), not `ij`.
11. `margin_notes[0].lines[2]` — the break sign after `ex` is a small x-height wedge, not the usual slanting hyphen; transcribed `-` (word continues `hib.`); could be a point.
12. `margin_notes[0].lines[4]` — two specks above and right of the final `s` of `literas`; read as ink, so no punctuation transcribed.
13. `margin_notes[0].lines[7]` — the mark over the final `a` of `proba` is a short nearly upright stroke; transcribed as the tilde `ã` (= probationibus), not expanded.
14. `margin_notes[0]` — placement: note `a` starts beside its own marker line and is the page's only note; the key letter is excluded from `lines` per §9.
15. `blocks[6].lines[12]` / furniture — no foot citations, no signature, no catchword; spaced-capital running head with a final point; folio 44 at the outer (right) edge, matching the manifest.

Plus a closing note recording that the long-s warnings on `Nous`, `Les` and `Sagias` were checked against the image (roman capitals, word-final round s) — no silent normalization.

## For the reconciler

- **Two markers in a row without citations.** p044 `{c}` and p045 `{d}` both stand bare, and then the alphabet restarts at `a` for ANNOTAT, XXIIII. The upper margin of p045 is physically blank for its whole height, so nothing is lost to the crop. This looks like the printer dropping two citations, but it should have a human eye.
- **The annotation number repeats.** p044 heads ANNOT. XXIIII and p045 heads ANNOTAT, XXIIII. Both were counted stroke by stroke. If p044's reading is right, the book misnumbers here.
- **The `§`/`B` sort in note `a` is the one genuinely unresolvable glyph on the page** at this scan resolution, and my reading rests on the sense of the citation rather than the shape. Expect a diff against reader A. It would be worth checking whether this sort appears elsewhere in the book, because the corpus so far always spells the paragraph sign `parag.`
- Condition otherwise good: ink is even, no damage, no show-through worth recording beyond the two specks noted. The gutter strip of the facing page was ignored throughout.
- Two word-breaks carry no hyphen and are correct as printed: `ſeque|ſtre` (blocks[2]) and `matie|re` (blocks[6]), plus `ve|rò` and `tran|miſſa` in the margin note.

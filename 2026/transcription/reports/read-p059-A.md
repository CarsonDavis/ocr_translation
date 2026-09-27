# Read report — p059, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p059.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p059.json`
→ `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| body lines | 37 (5 in the large-type TEXTE paragraph + 32 in the annotation paragraph) |
| paragraphs | 2 |
| headings | 1 (`ANNOT. XXXVIII.`, spaced capitals) |
| markers in body | 4 — `{a}` line 7, `{b}` line 14, `{c}` line 29, `{d}` line 32 (of the annotation paragraph) |
| margin notes | 4 — keys `a`, `b`, `c`, `d` |
| foot notes | 0 |
| running head / folio | `PARLEMENT DE THOLOSE.` / `59` (matches manifest) |
| signature / catchword | none / none |
| ornaments | none |
| `uncertain[]` entries | 11 |

## Page structure

Recto. Running head `PARLEMENT DE THOLOSE.` with the folio `59` at the outer edge.
The page opens with the tail of the TEXTE paragraph carried over from p058 (5 lines of
the large text type, `continues_prev: true`), ending cleanly at `ries touſiours enſemble.`
Then the centred heading `ANNOT. XXXVIII.` in spaced capitals, then a single 32-line
annotation paragraph in the smaller type running to the foot of the column and ending
`teſmoins. {d}`. Below that the column is blank — the annotation ends on this page, so
`continues_next` is false. No foot citation block, no signature, no catchword.
`pages/strips/p059/body-8.jpg` and `foot.jpg` are blank apart from the facing-page sliver.

## `uncertain[]` entries (11)

1. `blocks[2].lines[2]` — the line ends `affer` with **no** hyphen and line 3 begins
   `moyen`, so the print reads `affermoyen` where `affermoyent` is expected. sic.
2. `blocks[2].lines[3]` — companion entry: `moyen` carries no tilde (checked at 10x).
3. `blocks[2].lines[28]` — `premeuu` as printed, p-r-e-m-e-u-u at 9x; wrong sort. sic.
4. `margin_notes[0].lines[0]` — `Aecurſe` as printed (2nd letter has a crossbar, 3rd does
   not); wrong sort for `Accurſe` (Accursius). sic.
5. `margin_notes[0].lines[2]` — heavy ink blot on the final `u` of `plu`; it may conceal a
   line-end hyphen. No hyphen transcribed.
6. `margin_notes[1].lines[0]` — `octui` as printed (o + ct ligature + u + i), probably a
   wrong sort for `octaui`; and the `p` before `D.` may be `ꝑ`, unresolvable at this scan
   resolution.
7. `margin_notes[2].lines[0]` — the mark after `pa` read as a period, not a hyphen (round,
   on the baseline, unlike this print's elongated mid-height hyphen), although the word
   continues `res` on the next line.
8. `margin_notes[2].lines[1]` — the faint high mark between `res` and `D.` read as a
   period; could be a speck.
9. `margin_notes[2]` — placement note (not a reading): notes `c` and `d` are printed
   **above** their markers, see below.
10. `blocks[1]` — the head is printed `ANNOT.`, not `ANNOTAT.` as on neighbouring pages
    (cf. p055 `ANNOTAT. XXXV.`). Letters counted at 7x. Transcribed as printed.
11. `blocks[2].lines[13]` — round ink blot under the `ſo` of `perſonnes`; no letter lost.

## For the reconciler

- **Margin notes c and d are set well above their markers.** Note `c` begins beside body
  line 24 (`Martin Guerre. La quatrieme & derniere, car ces teſ`) although `{c}` is on
  line 29; note `d` begins beside line 28 although `{d}` is on line 32. The seven margin
  lines of notes c and d run one-to-one against body lines 24–30. `beside_line` records
  what is actually printed beside, not the marker line.
- Notes `a` and `b` sit almost exactly midway between two body lines; `beside_line` was
  chosen by nearest baseline (line 8 for `a`, line 13 for `b`). Either neighbour is
  defensible for those two.
- **Double s checked individually at ≥5x.** `n'eſgalaſſent` (line 1), `cognoiſſance`
  (line 12) and `yſſus` (line 26) are `ſſ` (two long s, joined at top). The margin note
  `miſsio.` (note d) is `ſs` — long s followed by a round short s, clearly distinguishable.
- **Hyphens.** This print breaks words at the line end both with and without a hyphen; each
  line end was checked at 5–9x. Hyphenated: lines 2, 5, 8, 9, 13, 15, 26, 27, 30 (and line 2
  of the TEXTE paragraph). Not hyphenated: lines 3, 11, 21, 22, 24 (and line 4 of the TEXTE
  paragraph, `nour` / `ries`).
- **Tilde vs acute.** The print's tilde is a flat horizontal bar and the acute a slanted
  stroke; they are easy to confuse at strip resolution. Confirmed tildes: `cõ` (TEXTE l.1),
  `nioyẽt` (l.5), `niẽt` (l.7), `ſecõde` (l.7), `affermoyẽt` (l.10), `biẽ` (l.10). Confirmed
  acutes: `hanté`, `frequenté`, `mangé`, `eſté`, `renté`, `degré`, `cohabité`, `donné`,
  `enſeigné`, `veritablement`… (ordinary é).
- **Wrong-sort sweep** turned up three: `affermoyen` (l.2/3), `premeuu` (l.28), `Aecurſe`
  (note a), plus the doubtful `octui` (note b). `preuaudroit` (l.28) was checked glyph by
  glyph against `eſtoit` and is a genuine `t`, not `c`. `Guerre` is `G` everywhere
  (lines 4, 17, 24).
- `Rols` (l.22) is printed with a capital R and ordinary lower-case `ols` — not small
  capitals.
- Punctuation at clause boundaries was zoomed individually: colons after `ans` (TEXTE l.3),
  `luy` (TEXTE l.4), `rens` (l.9), `enfans` (l.21), `ville` (l.18); periods after
  `enſemble`, `nioyẽt`, `veriſimilitude`, `parens`, `Guerre`, `teſmoins`.
- Paper condition is good: no damage, no faint ink beyond the two ink blots noted above
  and the heavily inked final `a` of `iulia` in note d.
- Context: `transcription/final/` has no p056–p058, so `continues_prev` on the first
  paragraph rests on the sentence itself (it opens mid-clause, `bonnes, & grandes, comme
  de l'auoir cõgnu…`). Marker sequence a–d restarts at `a` under this annotation head.

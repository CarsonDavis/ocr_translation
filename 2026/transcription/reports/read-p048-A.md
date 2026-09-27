# Read report — p048, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p048.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p048.json`
exits **0** ("1 ok, 0 failed"). One WARNING remains and is correct as transcribed:
`blocks[2].lines[9]: possible normalized long s (sur)` — the word is the sentence-initial
**"Sur"** in "maiſtre. Sur quoy faut bien…", printed with a round capital S (capitals carry
no long s in this fount). Checked against the image at 4x; not a normalization.

## Counts

| item | count |
|---|---|
| body lines | 32 (13 + 19) |
| paragraphs | 2 |
| headings | 1 (`ANNOTAT. XXVI.`, spaced caps) |
| markers in body | 2 (`{a}`, `{b}`) |
| margin notes | 2 (keys `a`, `b`) |
| foot notes | 0 |
| uncertain[] entries | 9 (3 flagged `escalate`) |

Page furniture: running head `ARREST DV` (spaced caps), folio `58` (matches manifest),
no signature, no catchword, no ornaments, no foot citation block. Verso; the facing-page
sliver along the right (gutter) edge of the crops was ignored.

Layout: paragraph 1 (13 lines, large text type, `continues_prev: true`) finishes the
TEXTE section; then the heading `ANNOTAT. XXVI.`; then the annotation paragraph in the
smaller type (19 lines). Both margin notes sit at the very bottom of the margin column
(margin-1, margin-2 and the top of margin-3 are entirely blank).

## uncertain[] entries — one line each

1. `blocks[0].lines[8]` — print sets `lereſte` with no gap; written `le reſte` per §1 word-division normalization.
2. `blocks[0].lines[10]` — **escalate** — (a) print sets `dou toyent` with a justification gap, written as one word; (b) the line-end sign after `n'o` is a small round blob rather than a dash, but measures at exactly the hyphen height of `ſimili-` two lines above, so transcribed `-`; a reader could take it for a period.
3. `blocks[2].lines[11]` — `TV DOIS` is letterspaced small capitals (T full size); closed up per §3, block flagged `spaced_caps`.
4. `blocks[2].lines[12]` — `q̃lle` is q + U+0303 (= quelle); the mark is a flat tilde stroke, not an acute.
5. `blocks[2].lines[17]` — **escalate** — last word printed `mct`: the middle sort is open on the right with c-terminals, matching the `c` of `ces`/`preuues` on the same line and unlike the closed `o` of `nous`/`dirons`; almost certainly a wrong sort for `mot`, transcribed as printed (sic).
6. `blocks[2].lines[18]` — (a) print sets `d'a uantage` with a gap, written as one word; (b) after marker `{b}` a round baseline dot (read as a period) with a small ink speck just above it — conceivably a colon with an under-inked upper dot.
7. `margin_notes[0].lines[0]` — **escalate** — printed tight as `l.iij.P.ij.D.`, spaced per §1; the numeral between `P.` and `D.` is small and blurred (two dots over a narrow body), read `ij`, but could be `j` or `ï`.
8. `margin_notes[1].lines[0]` — the word-break sign after `anno` is a thin slanted stroke rising from the top of the o rather than a level dash; read as the hyphen of `anno-tation`, but could be read as a tilde.
9. `margin_notes[1].lines[1]` — numeral printed with lowercase `l`, two capital `X`, lowercase `iij` (`lXXiij` = 73); transcribed as printed.

## Notes for the reconciler

- **Normalizations applied** (all per §1, none of them a reading change): `lereſte` → `le reſte`;
  `dou toyent` → `doutoyent`; `d'a uantage` → `d'auantage`; `quoy?ces` → `quoy? ces`;
  `credit:ou` → `credit: ou`; `l.iij.P.ij.D.` → `l. iij. P. ij. D.` If reader B left any of
  these as printed, the difference is spacing policy, not a disagreement about the letters.
- **Line ends without a hyphen** (normal for this print, no entry made): `…de pe` / `rils,`
  (blocks[2].lines[1–2]) and `…de q̃lle qua` / `lité,` (blocks[2].lines[12–13]). Also the
  page-internal break `c'eſt` / `oit` (blocks[0].lines[6–7]).
- **Double-s checked at 4–6x** and all confirmed long-s + long-s: `aſſeuroyent`, `aſſeurer`,
  `aſſeoir`. The one round-s case, `ſouuentesfois`, is round s before `f` as expected.
- `ſ'ils` (blocks[2].lines[13]) has a genuine printed apostrophe after the long s — verified at 4.5x.
- Tildes on this page: `l'affectiõ`, `plemẽt`, `pourpẽſees`, `chãcelé` (vowel tildes,
  precomposed) and `q̃lle` (consonant tilde, q + U+0303).
- Ink and paper are clean; no damage, no rubbing. There is moderate show-through from the
  recto across the upper half of the page, which puts faint ghost letters between the body
  lines — none of it was transcribed. A speck above the final `t` of `ſingulierement`
  (blocks[2].lines[0]) is show-through, not a tilde.
- The page has no catchword and no signature, and the bottom third below the text is empty;
  per §6 no `blank` block was added.

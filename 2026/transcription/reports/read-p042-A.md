# Reader A — p042 (image 64, verso, folio 42)

## Output

- `transcription/reads/A/p042.json`
- Validator: `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p042.json`
  → `1 ok, 0 failed`, exit 0. One warning, checked against the image and answered in
  `uncertain[]` (see item 19): `blocks[0].lines[21]` "possible normalized long s (sur)" —
  the word is `Sur` with a capital round S at the head of a sentence, not a normalized long s.
  Contrast `lines[37]` "victorieux ſur l'ardeur", where the lowercase word is correctly set with `ſ`.

## Counts

| | |
|---|---|
| body lines | 40 (one paragraph block) |
| paragraphs | 1 (`continues_prev`: true, `continues_next`: true) |
| headings | 0 |
| markers in body | 7 — `q r ſ t u x y` |
| margin notes | 8 — `p q r ſ t u x y` |
| foot notes | 0 |
| ornaments | 0 |
| signature / catchword | none / none |
| `uncertain[]` entries | 19 |

## Page structure

Running head `ARREST DV` in spaced capitals, folio `42` at the left of the same line, no
terminal punctuation after `DV`. Below it a single unbroken paragraph of 40 lines filling
the column, continuing the sentence and the annotation begun on p041 and running on to
p043. No heading, no decorated initial, no rule, no woodcut. Nothing is printed below the
last body line: the leaf is blank from there to the edge of the film, so there is no foot
citation block, no signature and no catchword. The margin holds eight italic citations in
a single left column; `margin-4.jpg` is blank.

Markers continue the alphabet from p041, which ended at `{o}`: this page should open at
`p` and it runs `q r ſ t u x y` (no `v`, no `w`). The next page should open at `z`.

## `uncertain[]` — one line each

1. `blocks[0]` — page quality: clean, sharp CUDL colour scan, even ink, no damage or
   manuscript marks; the doubts below are fine-detail judgements, not legibility problems.
2. `margin_notes[0]` — **the note keyed `p` has no marker anywhere in the body.** Lines 1–8
   were checked letter by letter at 3–7x; the one wide gap (line 1, between `rien` and
   `valu`) is clean justification space with no ink. Recorded as a note without a marker.
3. `blocks[0].lines[1]` — `auſsi` here is long s + **round** s; the same word is `auſſi`
   (ligature) on `lines[8]` and `lines[20]`. All three checked separately; the difference
   is the compositor's, not a reading error.
4. `blocks[0].lines[19]` — the point after `{u}` is very lightly printed, far fainter than
   the solid points after `{t}`, `{x}` and `{y}`; read as a weak period.
5. `blocks[0].lines[21]` — two pale brown bars over `Sur` and `quoy` are show-through /
   foxing, not tildes (much lighter and thinner than this page's printed tildes).
6. `blocks[0].lines[32]` — `Chreſtiẽne`: the mark is a horizontal wavy bar (tilde), not
   this font's steeply slanted acute.
7. `blocks[0]` — three line breaks printed with no word-break hyphen, transcribed as printed.
8. `blocks[0]` — spacing normalizations listed (tight commas, spaces before points, solid
   `ſemẽcecontrain`); no letter added or removed.
9. `running_head` — spaced capitals closed up per §3; folio matches the manifest.
10. `margin_notes[0].lines[1]` — the break sign after `nou` is short and set at mid
    x-height: read as a hyphen, could be a point. p041 note b prints the same citation with
    no break sign.
11. `margin_notes[0].lines[4]` — `coll. iij.`: three strokes, three dots. p041 note b gives
    `col iiij.` for a related citation, so the two pages disagree; `iiij` not wholly excluded.
12. `margin_notes[1].lines[0]` — printed solid `Gal.auxv.`; split as `Gal. au xv.` (Galen,
    *De usu partium* bk XV, and cf. the conventions' own `au xv. liur. de`). `aux v.` is
    the alternative.
13. `margin_notes[2].lines[0]` — `quaritur` as printed: a single italic `a`, no æ ligature;
    the Digest lemma is `quaeritur`. Not modernized.
14. `margin_notes[3].lines[3]` — `tatis. de frig.`: the mark could be a comma; read as a
    point on the parallel of the same formula in p041 note l.
15. `margin_notes[5].lines[4]` — the final point after `vij` is very faint; included.
16. `margin_notes[6].lines[0]` — the mark between `lege` and `P` is pale and irregular;
    read as a lightly inked point (`lege. P.`), but it may be an ink speck.
17. `margin_notes[6].lines[1]` — first word read `mal`; `nal` not wholly excluded. With
    `lines[2]` it gives `D. de ſicar.`, which supports it.
18. `margin_notes[5].lines[0]` — no point after the initial `l` in `l ſi ſeruus`; the gap is
    empty at 10x, unlike `l. quaritur.` and `l. lege`.
19. `blocks[0].lines[21]` — answers the validator's `sur` warning (see **Output** above).

## For the reconciler

- **The missing `{p}` marker is the one thing to arbitrate.** The note is printed beside
  body line 1 and its citation (*Parag. si vero*, Novellae *de nuptiis*) fits
  "mariage n'auroit iamais rien valu" exactly, so either the marker sort was omitted here
  or the marker belongs to the last line of p041. Worth checking p041's final line again.
- **`ſſ` vs `ſs` was checked at ≥3x on every double-s.** Only `auſsi` on line 2 is
  long s + round s; `impuiſſance` (×4), `paſſage` (×2), `neceſſaires`, `auſſi` (lines 9, 21),
  `diſſoudre`, `aſſez`, `deſſeing`, `neceſſité` are all `ſſ`.
- **Sentence punctuation was checked glyph by glyph**, not from the sense. The page's
  weakest marks are the point after `{u}` (line 20) and the point before `P` in note x —
  both flagged.
- The scan is good enough that a disagreement here is likely to be a real difference of
  judgement rather than an illegible passage.

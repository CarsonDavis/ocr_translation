# Read report — p036, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p036.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p036.json`
→ `1 ok, 0 failed`, exit 0, no warnings (NFC and long-s checks clean).

## Counts

| thing | count |
|---|---|
| body lines | 40 |
| paragraphs (blocks) | 1 (`continues_prev: true`, `continues_next: true`) |
| headings | 0 |
| markers in body | 8 — `f g h i k l m n`, one each |
| margin notes | 9 (keys e, null, g, h, i, k, l, m, n) |
| foot notes | 0 |
| `uncertain[]` entries | 12 |

The eight markers fall on lines 8 (`f`), 12 (`g`), 15 (`h`, `i`, `k`), 16 (`l`), 18 (`m`)
and 34 (`n`).

Page furniture: `running_head` "ARREST DV" (spaced capitals, closed up per §3),
`folio` "36", `signature` null, `catchword` null, no ornaments, no foot citation block.
`foot.jpg` shows only the last three body lines and blank paper below them.

## Layout

Verso. One running paragraph continuing from p035 and running on to p037 — no heading, no
decorated initial, no break anywhere in the column. Margin column on the left carries nine
citations: one at the top (key `e`), one unkeyed (`Leuit. c. xix.`) opposite the Moses
passage, then a tight six-note block `g h i k l m` beside the "epithetes à Satan" passage,
then `n` (Suetonius) low on the page. `margin-4.jpg` is blank.

## `uncertain[]` entries (12)

1. `lines[5]` — sic: **`nons`** printed for `nous` ("de nons inſtruire"); letters clear at 3x.
2. `lines[7]` — **marker `f` has no keyed note.** *(escalate)* The citation `Leuit. c. xix.`
   opposite this passage is printed with **no key letter** — verified at 6x with a wide crop,
   there is nothing to the left of `Leuit`. Recorded as `margin_notes[1]` with `"key": null`
   per §4, but it is almost certainly f's note. Reconciler should decide whether to promote
   it to key `f`.
3. `margin_notes[0]` — **note keyed `e` has no marker on this page.** *(escalate)* No `{e}`
   anywhere in the 40 body lines; the marker must sit on p035 and the note has drifted down
   onto p036. Recorded anyway per §4. This also fixes the page's alphabet run: e (p035) →
   f…n (p036), so p037 should open at `o`.
4. `lines[25]` — sic: **`q'vne`** (q + apostrophe, no u), not `qu'vne`; verified at 3.5x.
5. `lines[28]` — **`leur eſt. comme permis`**: the mark between `eſt` and `comme` is a small
   round dot sitting on the baseline with no descending tail, so read as a **period**. The
   comma after `permis` on the same line is visibly larger with a tail — side-by-side crop
   at 6x. Sense argues for no stop at all; shape decided it, per the prompt's rule. Could be
   a broken comma.
6. `lines[32]` — `preſte` is set with a full word-space inside it (`preſt e la main`);
   normalized to one word per §1.
7. `lines[35]` — **`main 'vne`**: a bare apostrophe stands before `vne` with no letter before
   it. Almost certainly a dropped `l` of `l'vne`. Transcribed as printed.
8. `lines[35]` — a **stray vertical ink stroke** stands between `à` and `fin` (runs from
   x-height to below the baseline, irregular, not a letter or punctuation shape). Not
   transcribed. At reading size it looks like `à|fin`, so the other reader may have taken it
   for a character.
9. `lines[38]` — sic: **`cou roux`**. `courroux` is set as two tokens separated by a full
   word-space, with only one `r` (`cou` + `roux`). Transcribed as printed.
10. `lines[38]` — sic: **`ſ'emflãboit`**, with `m` before the `fl` ligature. Compare
    `enflambee` with `n` on line 11 — the compositor used both forms on the same page.
11. `margin_notes[7].lines[1]` — `xi.` is blotted; read `xi.` (second glyph short, dotted, no
    descender) but `xj.` is possible.
12. `margin_notes` — `beside_line` placements are **approximate to within one body line** for
    the g–l block: those six citations are set as one tight run whose leading is narrower
    than the body's, so they drift upward relative to their markers (h's note sits beside
    line 12 although `{h}` is on line 15).

## Things the reconciler should know

- **Double-s was checked explicitly at 3–3.5x on every occurrence.** Every one on this page
  is the `ſſ` ligature, none is `ſs`: `neceſſité` (1), `gliſſement` (3), `oppreſſé` (9),
  `puiſſe` (12), `rauiſſant` (16) and the pair split across lines 19/20 (`inceſ-` /
  `ſamment`). No `ſs` spelling occurs anywhere on the page.
- **Sentence punctuation was checked by shape, not sense**, at 3x or better: the colons on
  lines 3 (`langue:`), 5 (`pourpenſee:`), 8 (`prochain {f}:`), 23 (`œuures:`) and 26
  (`calomnie:`) all show two clear dots; the periods on 12, 14, 22, 24, 31, 38 are round
  baseline dots; the commas throughout carry tails below the baseline. The one genuinely
  doubtful mark is item 5 above.
- **Tildes** are all precomposed and all sit over vowels: `cõtre` (1), `cõme` (14),
  `mechã-` (30), `ſouſtiẽt` (33 — clearly wavy at 7x), `exẽpte` (37), `ſ'emflãboit` (39).
- **Word broken across lines without a hyphen** at line 9/10 (`calomnia` / `teur`) — normal
  for this print, no entry needed per §1.
- **Marker `{m}` sits immediately after a comma** with no space (`Calomniateur,mpartant`);
  normalized to `Calomniateur, {m} partant`. **Marker `{n}` opens line 34** (`{n}. Alexandre`),
  so there is no space before it.
- **Marginal roman numerals** were read at 12x where blotted: `Gene. c. iij.` (three strokes,
  three dots), `Apocaly. xij.`, `Eſaye. c. xxvij.` (broken across two margin lines),
  `Pſeaume xc.`, `Ezeciel xxij.`, `Saphonie iij.`, `S. Pierre c. v.`, `Pſaume xxj. & c. ij.`
- **Ink and paper are good** across the whole column; no damage, no show-through worth noting,
  no gutter loss. The facing-page strip along the right edge of every crop was ignored.

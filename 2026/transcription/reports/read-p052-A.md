# Read report — p052, reader A (model: opus)

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p052.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p052.json`
exits **0** (1 ok, 0 failed). One WARNING remains and is a verified false positive — see below.

## Counts

| item | count |
|---|---|
| body lines (total) | 30 |
| — blocks[0] paragraph (continues_prev) | 8 |
| — blocks[2] paragraph (TEXTE) | 18 |
| — blocks[4] paragraph (ANNOTAT. XXXI, continues_next) | 4 |
| paragraphs | 3 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XXXI.`, both spaced caps) |
| markers in body | 2 (`{b}`, `{a}`) |
| margin notes | 2 (keys `b`, `a`) |
| foot notes | 0 |
| uncertain[] entries | 9 |

Page furniture: running_head `ARREST DV` (spaced caps), folio `52`, signature `null`,
catchword `null`, no ornaments. No foot citation block: `foot.jpg` shows only the last two
body lines, the tail of margin note `a`, and blank paper below.

## Cross-check

Every `{x}` in the body has a `margin_notes` entry with that key and vice versa:
`{b}` (end of blocks[0], `telle humidité {b}`) ↔ margin note `b`;
`{a}` (blocks[4].lines[2], `…tel pieça {a} mais`) ↔ margin note `a`.
The alphabet restarts at `a` at the `ANNOTAT. XXXI.` boundary, so `b` (carried over from the
preceding page's run) and `a` (new run) coexist on this page without needing an `a2` key.

## uncertain[] entries (9) — one line each

1. `blocks[0].lines[4]` — `viẽt`: mark over the e is high, thick and wavy (matches the tildes in `cõduits`/`tõboit`), not the thin steep acute of `rarité`/`voulté`; read as tilde.
2. `blocks[2].lines[9]` — `mẽton`: same judgement as `viẽt`; could be misread `méton`.
3. `blocks[2].lines[13]` — `touteſfois`: first tall glyph has only a left nub (long ſ), second a full crossbar (f); `touteffois` not excludable at this resolution.
4. `blocks[4].lines[2]` — the marker after `pieça` is a small, heavily inked raised italic letter; read `a` on the strength of the margin note keyed `a`; shape is blotted.
5. `margin_notes[0].lines[1]` — `Aphrodiſée`: the acute is printed detached, above and right of the final e (separate accent sort); could be read `Aphrodiſee` + stray mark.
6. `margin_notes[0].lines[2]` — `xxxÿ`: a single italic sort shaped like ÿ (y-form with descender and two dots), i.e. the usual italic `ij`; transcribed as printed; another reader may write `xxxij` or `xxxv`.
7. `margin_notes[1].lines[0]` — no period visible after the abbreviation `l` (the mark at its foot reads as the italic l's foot serif, unlike the clearly separated periods in `uerſis. D.`); expected form would be `l.`.
8. `margin_notes[1].lines[1]` — first letter read `u` (continuation of `di-` / `uerſis`); italic u and n are near-identical in this fount.
9. `margin_notes[1].lines[0]` (key) — the key letter `a` carries a small ink speck above it; read as a plain key `a`, not `à`.

## Validator warning (checked against the image, not silenced)

`blocks[4].lines[0]: possible normalized long s (sur) in "Sur la cognoiſſance d'vne perſonne…"`
— **false positive.** The word opens the ANNOTAT. XXXI. paragraph and is printed with a
**capital roman S**; capitals have no long-s form in this fount. Verified at 5× on
`body-7.jpg`. No change made.

## Explicit checks the prompt called for

**(a) `ſſ` vs `ſs`.** Every double-s was zoomed to ≥3×. Long s + long s (`ſſ`):
`aſſigne`, `ſ'eſiouiſſent`, `l'eſpeſſeur`, `preſſer`, `paſſages`, `deſſous`, `groſſe`,
`cognoiſſance`, `aſſeuré`. Long s + round s (`ſs`): **`auſsi`** (blocks[0].lines[2]) and
**`aſsiſté`** (blocks[2].lines[2]) — both unambiguous at 5×, the second letter is the short
round form. These two are the likeliest places for reader B to disagree.

**(b) Sentence punctuation.** Each clause-boundary mark was zoomed and read by shape, not by
sense. Confirmed colons (two dots): `pleurent:`, `des yeux:`, `numeraires:`, `des iãbes:`,
`ſourcil droit:`. Confirmed periods (round dot on baseline): `ſ'eſiouiſſent.`,
`ſont produits.`, `leſd.`, `cicatrices.`, `problemes.`, `ligioſ.`, `TEXTE.`,
`ANNOTAT. XXXI.` Confirmed commas (tail below baseline): all others.

## Notes for the reconciler

- **Normalization applied, so expect diffs against a literal reading of the image.**
  The print sets a space *before* several commas (`en tous les deux , c'eſt`,
  `En ſecond lieu , y a`, `preuue grande , & preſque`, `trappe , & fourni`) and sets others
  tight (`plus noir,hom`, `troiſieme,tous`, `les,le mẽton`, `bas,ayant`, `D.dere`,
  `c.xxxÿ.des`). All normalized per §1 to no-space-before / one-space-after.
- Two line ends break a word **without** a hyphen — `hom` / `me greſle` and
  `eſpau` / `les, le mẽton` (and `cicatri` at the foot of the page, continuing onto p053).
  Per §1 these need no `uncertain[]` entry and were left as printed.
- `blocks[2]` (the TEXTE quotation) is set in larger type with a narrower measure than the
  annotation type; its short lines are genuine printed lines, not wrapping.
- `blocks[4]` is cut off mid-word at the foot (`…les cicatri`), so `continues_next` is true;
  `blocks[0]` opens mid-sentence, so `continues_prev` is true. No page-boundary word could be
  resolved from context — no preceding page is finished yet.
- Ink and paper are clean; no damage, no show-through worth noting beyond a faint ghost of
  the facing page in the upper margin (ignored). The only genuinely hard-to-read spots are
  the blotted marker after `pieça` and the tiny italic of margin note `a`.
- The crops keep a sliver of the facing recto along the right (gutter) edge; it was ignored.

# read-p069-A

Output: `transcription/reads/A/p069.json`
Validator: `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p069.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

- body lines (paragraph lines only): **31** — 2 + 1 + 6 + 11 + 11
- paragraphs: **5** (carry-over paragraph; the one-line Latin verse; the "D'autant que" paragraph; the TEXTE block; the ANNOTAT. XVIII. block)
- headings: **2** (`TEXTE.`, `ANNOTAT. XVIII.`, both spaced capitals)
- markers in body: **4** (`{a}`, `{b}`, `{c}`, `{d}`)
- margin notes: **4** (keys a, b, c, d — every marker has a note, every note a marker)
- foot notes: **0** (no small-type foot block; the only thing below the last body line is the signature)
- page furniture: running head `PARLEMENT DE THOLOSE.` (spaced caps), folio `69`, signature `E iij`, no catchword
- uncertain entries: **10**

## Page structure

Carry-over paragraph (2 lines, `continues_prev`), ending `…ou vn vray Syſiphe,`; then the
single italic Latin verse line `Syſiphus in terris, quo non aſtutior alter.` as its own
paragraph block (it completes the sentence, so `continues_prev` is true there and
`continues_next` on the block above); then a 6-line paragraph ending
`Martin Guerre ſon mari.`; heading `TEXTE.`; the 11-line large-type TEXTE paragraph;
heading `ANNOTAT. XVIII.`; the 11-line annotation paragraph, which runs off the page
(`continues_next`) ending `…faux procureur, a`. No ornaments, no decorated initial.

## uncertain[] entries (10)

1. `blocks[6].lines[9]` — **`procureut`**: sic, wrong sort. At 700% the final glyph is
   unambiguously a `t` (crossbar through the stem, curved foot) where `procureur` is meant;
   compare the correct `r` of `procureur,` two lines below. Transcribed as printed.
2. `blocks[6].lines[4]` — `euë`: the final `e` carries a round dot on the left and a slanted
   stroke on the right; read as a tréma, but `eué` is possible.
3. `margin_notes[0].lines[3]` — `D. de cõd. ca.`: the stop between `cõd` and `ca` is a faint
   low dot crossed by a vertical paper fibre; `cõdica.` as one word is possible.
4. `margin_notes[0].lines[4]` — `dau.`: third letter is two minims joined below (a `u`, no
   crossbar), so not `dat.`; the abbreviation is unresolved, given as printed.
5. `margin_notes[2].lines[0]` — **`l ſulſus. & il`** (escalate): no stop after the opening
   `l`, and `ſulſus` is not a Latin form I can resolve. Glyphs read ſ-u-l-ſ-u-s; a lost
   period between `ſul` and `ſus` is possible. Worth a second pair of eyes.
6. `margin_notes[2].lines[1]` — `lec`: the second letter is broken/under-inked and looks like
   an `a`. Read `lec` because note a has the same word split as `& i` / `lec Accurſe`, i.e.
   `illec`; note c splits it `& il` / `lec les gloſe`.
7. `margin_notes[2].lines[3]` — `fur.`: the `f` has a full crossbar and the third letter shows
   an `r` shoulder, but `fuſ.` cannot be ruled out at this resolution.
8. `margin_notes[3].lines[2]` — numeral `iij`: three dots are visible but the middle minim is
   worn, so `ij` is possible.
9. `margin_notes[3].lines[2]` — the final sign is a capital P with a stroke through the
   descender, standing for the paragraph before the incipit `ſed & ſi quidem`. Transcribed
   `Ꝑ` (U+A750); `§` may be the intended sense.
10. `margin_notes[3].lines[0]` — `quæro`: the ligature is heavily inked; a plain `a`
    (`quaro`) is just possible.

## Notes for the reconciler

- **Checked explicitly per the prompt.** `ſſ` vs `ſs`: `outrepaſſé` and `confeſſé` are both
  genuine `ſſ` (two tall s, verified at 400–800%); no `ſs` pairs found on this page.
  Punctuation: `tres digne:` is a true colon (two dots); `ri:` likewise; `à ces fins.` is a
  round dot on the baseline, not a comma (the apparent tail is a paper fibre); `en effect ,`
  is a comma (tail below baseline), normalized to `effect,`.
- **Wrong sorts.** Only one found: `procureut` (item 1 above). Everything else that looked
  wrong at reading size (`procuteut`, `maiſtresau`, `lac`) resolved on zoom.
- **Spacing normalized per §1.** The print sets `tresdigne`, `v n`, `ap res`, `point&`,
  `deſcouuerte,il`, `contraires.Et`, `maiſtresau` — all normalized to single spaces:
  `tres digne`, `vn`, `apres`, `point &`, `deſcouuerte, il`, `contraires. Et`,
  `maiſtres au`. `d'aultresfois` keeps its apostrophe (a clear tick after the `d`) and is one
  word.
- **Line breaks without a hyphen** (normal for this print, not flagged): `eſt ap` / `pellé`,
  `mais en` / `cor`, `appoin` / `tement`, `ma` / `ri:`, `iceluy Pier` / `re Guerre`.
- **Margin layout.** The margin is empty for the top two thirds of the page; all four notes
  are crowded into the bottom third and drift upward relative to their markers — note c in
  particular starts about five body lines above the line carrying `{c}`. `beside_line` records
  where each note's first line actually sits, not where its marker is.
- **Condition.** Paper is clean and the impression is good; the only interference is a
  vertical paper fibre through `D. de cõd. ca.` and a hair-like fibre across
  `docteurs. C. de`. The inner (left) edge of the crops carries a strip of the facing verso,
  ignored throughout.
- **Alphabet continuity.** This page runs a–d; since no preceding page is transcribed yet,
  the reconciler should confirm against p068 whether `a` here is a restart at a section
  boundary (it follows the `ANNOTAT. XVIII.` head, so a restart is likely).

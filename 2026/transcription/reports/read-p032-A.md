# Read report — p032, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p032.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p032.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| item | count |
|---|---|
| body lines | 36 (33 small-type lines + 3 large-type lines under `TEXTE.`) |
| blocks | 7 |
| paragraphs | 6 (two of them are one-line italic Latin quotations) |
| headings | 1 (`TEXTE.`) |
| markers in body | 4 — `{e}`, `{f}`, `{g}`, `{h}` |
| margin notes | 5 — keys `e`, `f`, `g`, `h`, `i` |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 12 (1 escalated) |

Page furniture: `running_head` `ARREST DV`, `folio` `32`, `signature` null, `catchword` `en`.

## Structure

1. `paragraph` — lines 1–6, begins with an indent (new paragraph), ends `ne diſoit pas ſans cauſe {e},` → `continues_next: true` (the sentence runs into the verse).
2. `paragraph` (italic Latin, one line) — `Omnis in Aſcanio, chari ſtat cura parentis.` → `continues_prev: true`.
3. `paragraph` — lines 8–20, contains the letterspaced small-capital `FILS`, markers `{f}` and `{g}`. `spaced_caps: true`.
4. `paragraph` (italic Latin, one line) — `Omnis amor magnꝰ, ſed apertè in cõiuge maior.`
5. `paragraph` — lines 22–33, marker `{h}`; the Periander / Orpheus passage.
6. `heading` — `TEXTE.` (letterspaced, `spaced_caps: true`).
7. `paragraph` — the three large-type lines of the *Texte*, `continues_next: true` (catchword `en`).

## `uncertain[]` entries (12)

1. **`folio`** — the first digit is nearly gone; the surviving top curve and lower bowl fit `3`, the second digit is a clear `2`. Read `32`, matching the manifest.
2. **blocks[0].lines[1]** — `felon` (not `ſelon`): the glyph's crossbar crosses the stem, unlike the long s of `ſeroit` two words earlier; sense supports `f`.
3. **blocks[0].lines[0]** — a small raised ink speck follows `ne` at the right edge (and again after `fils` on lines[4]); judged stray ink / show-through, no punctuation transcribed.
4. **blocks[2].lines[8]** — `FILS` is set in letterspaced small capitals (`F I L S`); transcribed closed up as ordinary capitals per §3; block flagged `spaced_caps`.
5. **blocks[4].lines[0]** — `Dequoy` is set with no gap at all between `De` and `quoy`; kept as one word.
6. **blocks[4].lines[7]** — a narrow vertical ink bar (above x-height down through the descender line) stands between `pœtes` and `deuiſent`; no serifs, dot or letter shape, so not transcribed.
7. **blocks[4].lines[9]** — a small raised ink mark stands between `Pluton` and `&`; possibly an effaced marker `{i}`, possibly a rising space. Not transcribed.
8. **margin_notes[4]** — **ESCALATED. Missing marker `{i}`.** The margin note keyed `i` (Vergil *Georgics* iiij + Ovid *Metamorphoses* — i.e. the Orpheus story) has no legible `{i}` anywhere in the body. Every line of the Orpheus sentence was checked at high magnification. Most likely effaced position: the raised mark between `Pluton` and `&` on blocks[4].lines[9].
9. **margin_notes[4].key** — the note's own key letter is over-inked into a blob; read `i` from the running alphabet on this page (e, f, g, h → i).
10. **margin_notes[4].lines[4]** — `Metarmor-` *sic*, for `Metamor-`; transcribed as printed.
11. **margin_notes[0].lines[0]** — `vergile` is printed with a lower-case italic `v` (x-height), not a capital; same in margin_notes[4].lines[0]. Kept as printed.
12. **margin_notes[2].lines[1]** — the numeral in the Propertius note is four plain minims with four dots and **no descender on the last**, so `iiii`, whereas the Vergil note (margin_notes[4].lines[1]) clearly has `iiij` with a descending j. Worth a second look.

## Notes for the reconciler

- **Marker/note count mismatch is real, not an oversight.** Four markers in the body (`e f g h`), five notes in the margin (`e f g h i`). See uncertain entry 8.
- **Two stray vertical ink marks** in the last small-type paragraph (after `pœtes`, line 29; after `Pluton`, line 31). I checked whether they form a continuous crease by profiling a vertical strip through that column — they do **not**; the intervening lines are clean, so they are two separate accidents (rising spaces or pen strokes), not a fold. Neither is transcribed.
- **`ſſ` checks done at ≥6× on every double-s**: `auſſi` (L12), `fuſſe` (L17), `aſſez` (L19), `laiſſe` (L26) — all four are genuine long-s + long-s (both strokes reach ascender height). `toutesfois` (L12, L32), `toutes` (L20), `autres` (L20, L22), `pœtes` (L29) use round s as printed.
- **Punctuation checked glyph by glyph** at clause boundaries: colons confirmed at `concubines:` (L11), `mort:` (L12), `ainſi:` (L15); commas confirmed (descending tail) at `fils,` (L11), `Corinthien,` (L26), `rendirent,` (L31), `cauſe {e},` (L6); periods confirmed at `ſoy-meſmes.` (L4), `tuée.` (L30), `parentis.` (L7), `maior.` (L21).
- **Tight-set word divisions normalized** per §1: `par ce quil` (L10), `Et Orphee` (L28), `temps euſt` (L34), `gar der` → `garder` (L32), `chari ſtat` (L7), `mort: toutesfois` (L12), `tuée. & fit` (L30). `Dequoy` (L22) was the one case kept closed up — flagged above.
- **Line-end word breaks without a hyphen**: `Oui` / `de` in margin note `i`. Hyphens present and kept at L8, L9, L10, L16, L18, L22, L25, L29 and in margin notes `i` (`Geor-`, `Metarmor-`).
- **Physical condition**: the page is clean and well printed; ink is strong in the body. The only badly inked spot is the folio number (top left), which is almost entirely gone. The margin notes are crisp except for the key letter of note `i`. The crops carry a sliver of the facing recto along the right (gutter) edge — ignored.
- **Superscript abbreviation**: `magnꝰ` (L21) uses U+A770 for the printed 9-shaped *-us* sign, per the conventions table.

# Read report — p051, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p051.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p051.json` → `1 ok, 0 failed` (exit 0, no warnings).

## Counts

| | |
|---|---|
| body lines | 32 (9 + 8 + 15) |
| paragraphs | 3 |
| headings | 3 (`ANNOTAT. XXIX.`, `TEXTE.`, `ANNOTAT. XXX.`) |
| markers in body | 4 (`{a}`, `{b}`, `{c}`, `{a2}`) |
| margin notes | 4 (keys `a`, `b`, `c`, `a2`) |
| foot notes | 0 (no foot citation block) |
| uncertain entries | 6 |

## Page furniture

- running head: `PARLEMENT DE TOLOSE,` (spaced capitals, closed up)
- folio: `51` (matches the manifest)
- signature: `D ij` (fits the gathering pattern: D at p049, D ij at p051)
- catchword: none
- ornaments: none

## Layout

Running head + folio, then `ANNOTAT. XXIX.` and a 9-line annotation paragraph; `TEXTE.` and an 8-line paragraph in the larger text type; `ANNOTAT. XXX.` and a 15-line annotation paragraph that runs on to p052 (`continues_next: true`, last line ends `… Alexander Aphrodiſee`). The marker alphabet restarts at the `ANNOTAT. XXX.` section, so its single marker is keyed `a2` in both the body and the note. Block 5 carries `spaced_caps: true` for the spaced small capitals `L A-` / `C R I M A` (transcribed closed up as `LA-` / `CRIMA`).

## Uncertain entries (6)

1. `blocks[1].lines[8]` — **escalated.** The raised marker after `ſiens` is shaped like an italic long `ſ` (top hook to the right, straight stem, foot to the left), not like the raised italic `c` used elsewhere in the book (compared against p018 `mariage {c}` and p017 `atteſtoit {ſ}` at 25–30x). Transcribed `{c}` because the margin note beside it is keyed with an unambiguous italic `c` and the page's markers run a, b, …; the reconciler should decide between `{c}` and `{ſ}`.
2. `blocks[1].lines[3]` — the point after `ſang` is small and imperfectly inked; at 16x it sits on the baseline without the descending tail this page's commas have (cf. `tient,`), so read as a period; a comma is not excluded.
3. `blocks[5].lines[2]` — a small acute-shaped tick stands over the final e of `preſenté`; fainter and shorter than the clear acutes on this page (`deſnaturé`, `apporté`), so `preſente` is possible.
4. `margin_notes[3].lines[2]` — the last glyph of `Antiquitez Iu-` is blotted; read as a word-break hyphen, but it could be part of a letter. The break is confirmed by `dai…` opening the next line.
5. `margin_notes[3].lines[3]` — **escalated.** Letters after `dai` are merged/blotted: two joined glyphs (a q plus a looped sign, with a bar above), then an isolated point, then `s`. Read as `daiques` because the citation is Josephus, *Antiquitez Iudaiques*, liure xij, c. ij; letter forms and the position of the point are not certain.
6. `running_head` — printed `TOLOSE` without the H (sic; most rectos read `THOLOSE`). The final point descends below the baseline of the E and is taller than wide, so read as a comma rather than a period.

## Notes for the reconciler

- **Long s vs round s checked at 5–18x on every double-s:** `congnoiſſe`, `triſteſſe` (×2), `engoiſſe`, `preſſe` are all `ſſ`; `auſsi` and `exceſsiue` are `ſs` (long s + round s). These two `ſs` spellings are real, not reader slips.
- **Tildes checked individually** and all are wavy tildes, not acutes: `incontinẽt`, `d'ẽnuy`, `tellemẽt`, `rõpement`.
- Two line-end word breaks are printed **without** a hyphen and are transcribed that way, per §1: `… extremement affli` / `ge d'ẽnuy …`. `conſom-` is set with wide compositor spacing (`conſo m-`) and is normalized to one word.
- `conſeque` (block 1, line 6) is as printed — the letter before `ue` has a clear descender, so it is `q`, not `r`.
- The page is clean: no damage, no gutter loss, ink even except for the two blotted margin spots noted above. The faintest body spot is `yeux` on block 5 line 10, where the `e` and `u` are lightly inked but legible.
- The printed original is at native resolution in the strips (`raw/img071.jpg` is 2941×4711 and the body strip is already a 1:1 crop), so no further magnification is available beyond what is reported here.

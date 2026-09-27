# Read report — p063, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p063.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p063.json` → `1 ok, 0 failed`, exit 0, no warnings.

Manifest record: `{"id": "p063", "page": 63, "image": 83, "side": "recto", "folio": "63", "source": "cudl"}`

## Counts

| | |
|---|---|
| body lines | 29 (6 + 23) |
| paragraphs | 2 |
| headings | 1 (`TEXTE.`, spaced caps) |
| markers in body | 2 (`{a}`, `{b}`) |
| margin notes | 3 (keys `a`, `b`, `o`) |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 10 |

Page furniture: running head `PARLEMENT DE THOLOSE.` (spaced capitals, closed up), folio `63` top right (matches the manifest), no signature, no catchword, no ornament, no foot citation block.

## Layout

Recto. The page opens mid-sentence, continuing the ANNOTAT. paragraph from the previous page (`continues_prev: true`): six lines of the small annotation type ending `…de quelq'vn.` Then the display line `TEXTE.` in spaced capitals, then 23 lines of the large TEXTE type, beginning `Quatriemement,` and breaking off mid-sentence at `ce que fut` (`continues_next: true`). Roughly the bottom fifth of the text column is blank (no block recorded, per §6). All three margin notes sit at the head of the margin column (`margin-1.jpg`); `margin-2/3/4.jpg` are empty, `foot.jpg` and `body-8.jpg` are blank.

Note placement (`beside_line`): note `a` beside body line 1, note `b` beside body line 6, note `o` beside the first line of the TEXTE paragraph. The margin is set in a smaller face with tighter leading, so note `o` runs on past the lines it keys.

## `uncertain[]` entries — one line each

1. `blocks[2].lines[22]` — **sic, wrong sort: `vu` for `vn`** in `vu tel coffre`; at 12x the letter has two separate top serifs like the `u` of `quand`, not the arch of every other `n`/`vn` on the page. *escalate.*
2. `margin_notes[2]` — **key `o`**: a closed oval, not the open italic `c` of the notes above it; no `{o}` marker exists anywhere in the body and the printer's alphabet wants `c` after `a`, `b`, so this is probably a wrong sort for a key whose marker is also missing. *escalate.*
3. `blocks[2].lines[10]` — `eſtionsà` printed tight, normalized to `eſtions à` per §1; same normalization at `le s chauſſes` → `les chauſſes` and `blesde` → `bles de`.
4. `blocks[0].lines[4]` — the marks over `q̃`, `Frãçois`, `Ieã` (and `cõme`, `filiatiõ`, `biẽ`) are flat hooked bars, i.e. nasal tildes, not the steep acute wedge of `difficulté`/`arriué`.
5. `margin_notes[0].lines[3]` — `præterea` has a true `æ`; two marks follow it (a baseline point plus a smaller raised point), read as a period plus a speck, a colon being the alternative.
6. `margin_notes[2].lines[3]` — `quẽ` is a plain e with a bar, not `æ`; with `aper,` it gives the Digest title *quemadmodum testamenta aperiantur* (D. 29.3). `quæ` is the alternative.
7. `margin_notes[2].lines[1]` — `ij` is the italic ÿ-shaped sort, decomposed; `P.` is a plain italic capital P standing in for `§`, transcribed as printed (same decision as p058).
8. `blocks[1]` — page-furniture summary: spaced capitals closed up, folio, absence of signature/catchword/foot block, and the punctuation-spacing normalizations (`perplex:eſquels`, `paſſées:& di`, `propos:meſmes`, `parti : ce`).
9. `blocks[2].lines[13]` — the line ends in a word-final long `ſ` (`tinſ` / `mes`); this page breaks five words across lines without a hyphen (`ren`/`controit`, `qu'Antoi`/`ne`, `di`/`ſoit`, `tinſ`/`mes`, and `au`/`tre` at the page head).
10. `blocks[2].lines[1]` — explicit `ſſ` vs `ſs` check: `aſſeurent`, `cognoiſſance`, `paſſées`, `deſſus`, `chauſſes`, `laiſſay` are all genuine `ſſ` at 4x; `fils`, `Rols`, `les`, `dans`, `ouys` use the round s.

## For the reconciler

- **The two things to check first** are the `vu`/`vn` wrong sort on the last body line and the `o` key on the third margin note. Both were decided against the expected word/letter after zooming, which is exactly where a careful reader tends to over-correct, and both are recorded with `escalate: true`.
- **Marker/note asymmetry**: the body has markers `a` and `b` only; the margin has three notes. The third (`o`) has no marker on this page. The alphabet on this page runs `a`, `b`, so a reader of p062 and p064 should be able to say whether the run continues `c` on the next page, which would decide whether `o` is a wrong sort for `c` or the note belongs to a marker the compositor dropped.
- **The context given to me was wrong in one respect**: the prompt said "Preceding pages: none (no earlier pages finished yet)", but `transcription/final/` already holds p000–p056, p058 and p159. p062 is not among them, so there was no direct predecessor to read for `continues_prev` or for the marker run either way; `continues_prev: true` is set from the page's own opening (`vn faict fort ambigu…` completes a sentence begun on the previous page and the first line is not indented). I used p058 only as a formatting precedent for `P.`, `ij` and the `beside_line` practice, and copied nothing.
- **Condition**: the scan is clean and sharp. A light brown damp stain crosses the top-right of the text block (visible over `fame`/`& perplex:`) and the top of the margin column, but it does not obscure any type. The gutter strip of the facing page runs down the left edge of the body strips and the bottom-left of `foot.jpg`; it was ignored. Faint show-through from the verso is visible between the lines of the large TEXTE type and around the `TEXTE.` heading; none of it was mistaken for type.
- `q̃` is written as `q` + U+0303; all other nasals use precomposed characters.

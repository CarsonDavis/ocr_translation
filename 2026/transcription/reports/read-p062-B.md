# read-p062-B

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p062.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p062.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| body lines | 28 |
| paragraphs | 4 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XLI.`, both spaced caps) |
| markers `{x}` | 0 |
| margin notes | 0 |
| foot notes | 0 |
| ornaments | 0 |
| signature | none (verso) |
| catchword | none |
| running head / folio | `ARREST DV` / `62` |

Block layout, top to bottom:

0. paragraph, `continues_prev: true`, 2 lines — end of the previous page's sentence about the wart, into the Q. Serenus citation.
1. paragraph, 4 lines — the italic Latin quotation (`Interdum exiſtit turpi verruca papilla:` … `Qui ſolus patriæ, cunctando reſtituit rem.`). Treated as a paragraph, not headings, per §6. The short third line `hæſit,` is a printed turnover line and is kept as its own string.
2. heading `TEXTE.`
3. paragraph, 20 lines — the deposition about the conspiracy of Pierre Guerre and Iean Loze.
4. heading `ANNOTAT. XLI.`
5. paragraph, 2 lines, `continues_next: true` — the annotation runs on to p063.

## uncertain[] entries — 5

1. **`running_head`** — a faint baseline speck between `ARREST` and `DV`; read as ink speck/offset, not a period (p056 does print `ARREST DV.`, so worth a second look).
2. **`blocks[3].lines[0]`** — `teſmoius` for `teſmoins`: wrong sort, the glyph is unambiguously a round-bottomed `u` at 12x. Transcribed as printed, noted `sic`.
3. **`blocks[3].lines[8]`** — `reffu-` read as `ff`, not `ſſ`: a crossbar runs through both stems and projects right of the second, unlike the true `ſſ` elsewhere on the page.
4. **`blocks[3].lines[15]`** — `volonte` printed with no acute on the final `e` (checked at 7x with contrast enhancement); the `l` is an under-inked, broken sort.
5. **`blocks[3].lines[15]`** — the point after `ladit` is very lightly inked; read as the abbreviation point of `ladit(e)` (baseline dot, no descending tail).

## Notes for the reconciler

- **No margin column on this page.** All four `margin-*.jpg` strips are blank paper end to end; `foot.jpg` shows nothing below the last annotation line. So there are no markers, no margin notes, no foot citations, no signature and no catchword — the alphabet of marker letters neither advances nor restarts here.
- **Paper condition:** a brown stain runs across the top-left corner, over the folio `62` and the first two body lines, but does not obscure any letter. Inking is generally light in the lower third (`volonte`, the point after `ladit`, the annotation lines), which is where four of the five uncertainties sit.
- **Punctuation deliberately checked at high zoom** (task step 7b), all confirmed by glyph shape rather than sense: `verrues.` period; `Q.` period; `papilla:` colon; `mourir;` semicolon (comma with a tail plus a separate dot above); `priſonnier. iuſqu'à` period; `priſonnier. ce que` period; `ſauuer:` colon; `aſſeuré.` period; `Rols:` colon; `aud.` period; `vertu:` colon.
- **`ſſ` vs `ſs` checked at up to 12x** on every double-s: `aſſez`, `aſſeuré`, `pluſtoſt`, `eſtoit`, `ceſte`, `eſtre` — all long-s forms as transcribed. The only `ff` on the page is `reffu-` (see uncertainty 3); that contrast is the single most likely A/B divergence.
- **Word forms kept as printed, no `uncertain[]` per §1:** `ſes femme & beaux fils` (plural `ſes` governing the pair), `que il` (unelided), `Palhé`, `d'iceux`, and the unhyphenated line break `mou` / `rir` at `blocks[3].lines[7]`–`[8]`.
- **Tildes:** `cõiuration` (line 1), `parẽt` (line 10), `veritablemẽt` (line 18) — all over vowels, kept precomposed and unexpanded.
- `cunctando` and `faict` are set with `ct` ligatures and `exiſtit`/`reſtituit`/`eſtre`/`pluſtoſt` with `ſt` ligatures; all decomposed per §2.

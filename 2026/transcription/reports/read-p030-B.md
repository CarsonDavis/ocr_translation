# read-p030-B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p030.json`
(validator: `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p030.json` → exit 0,
2 long-s warnings, both capital `S` at sentence start — `Sur quoy`, `Si les loix` — verified round capital S in the image.)

## Counts

| item | count |
|---|---|
| body lines (all paragraph blocks) | 33 |
| paragraphs | 5 (17 + 2 verse + 7 + 4 TEXTE + 3) |
| headings | 2 (`TEXTE.`, `ANNOT XV.`, both spaced caps) |
| markers in body | 9 (`b c d e f g h i` then `a`) |
| margin notes | 9 (`b c d e f g h i a`) |
| foot notes | 0 |
| signature / catchword | none |
| `uncertain[]` entries | 10 |

Running head `ARREST DV` (spaced caps), folio `30`, verso. No ornaments.

## Page structure

1. paragraph, `continues_prev: true`, 17 lines (`mauuais, ou infaiſable {b}.` … `oit bien en diſant {f},`)
2. paragraph — the Latin distich, 2 lines (`Quid non mortalia pectora cogis,` / `Auri ſacra fames?`)
3. paragraph, 7 lines (`C'eſt pourquoy l'empereur M. Antonin` … `toute vertu {i}.`)
4. heading `TEXTE.`
5. paragraph, 4 lines, large display type (the arrêt text), complete sentence
6. heading `ANNOT XV.`
7. paragraph, 3 lines, `continues_next: true` (runs onto p031)

## uncertain[] — one line each

1. `blocks[0].lines[0]` — stray mid-height dot between `faluſt` and `il`; read as speck, not transcribed.
2. `blocks[0].lines[13]` — `qu's'eſtoit` printed with no space at all; kept closed up.
3. `blocks[2].lines[1]` — stray vertical stroke (damaged type) between `vertu` and the comma; not transcribed.
4. `blocks[2].lines[5]` — faint baseline dot between `Saluſte` and the comma; read as speck.
5. `blocks[2].lines[6]` — the same stray stroke recurs between `vertu` and marker `{i}`.
6. `blocks[4].lines[0]` — print sets `Mart in` with a full word-space inside the name; normalized to `Martin`.
7. `blocks[5].text` — a faint uninked ring after `ANNOT` where a period would fall; transcribed `ANNOT XV.` without it.
8. `margin_notes[1].key` — key letter is an ink blot; read `c` from the alphabet run and from the citation (Aeneid I).
9. `margin_notes[3].lines[4]` — `Aeneides,` final mark is a light wedge with a tail; could be a period.
10. `margin_notes[7].lines[1]` — `vt Iude`: tall stroke read as italic capital I, could be lowercase `l`.

## For the reconciler

- **All nine markers and notes pair up and the citations corroborate the keys**: b = Cicero *Rhetorica* + Panormitanus (`infaiſable {b}`); c = Aeneid I (Pygmalion/Sichaeus/Dido); d = Plutarch *Parallela* + Aeneid III (Polydorus); e = Cicero *Verrine* VI + Aeneid VI (Eriphyle); f = Aeneid III (the distich quoted); g = Iulius Capitolinus, *Life of Antoninus*; h = 1 Timothy c. ix; i = Justinian *Novellae* + Sallust; a = Digest *de regulis iuris*. This is the strongest evidence for reading the blotted key as `c`.
- **`{e}` is easy to miss**: it is a raised superscript `e` after `predit` in `comme luy auoit eſté predit {e}.` — it looks like the word `predite`. Zoom confirms a smaller raised letter.
- **Known misprint kept as printed**: `poinr` for *point* (`blocks[0].lines[14]`).
- **Spelling split kept as printed**: body has `Saluſte` (one l), margin note i has `Salluſte` (two l).
- `ſſ` check: the only double-s on the page is `auſſi` (`blocks[0].lines[6]`) — genuinely `ſſ` (ligature with crossbar), not `ſs`.
- Line 3 of block 0 ends `teſmoigna-`; block 2 line 0 ends `gene` with **no** hyphen (normal for this print).
- `deshommes` in the print is set solid; normalized to `des hommes` per §1, matching the house style already used in the finals.
- Margin notes drift down: notes d, e and g start several lines below their markers. `beside_line` records the body line each note's first line is actually printed beside, measured off the whole-page image.
- No foot citation block, no signature, no catchword. Paper is clean; the only damage-like features are the stray marks listed above and the blotted key letter.

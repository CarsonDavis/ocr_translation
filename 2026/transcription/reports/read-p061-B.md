# Read report — p061, reader B

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p061.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p061.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| running head | `PARLEMENT DE THOLOSE.` (spaced capitals) |
| folio | `61` (matches manifest) |
| body lines | 30 |
| paragraphs | 3 (7 + 18 + 5 lines) |
| headings | 2 (`TEXTE.`, `ANNOTAT. XL.`, both spaced capitals) |
| markers | 2 (`{f}`, `{g}`), both in the first paragraph |
| margin notes | 2 (keys `f`, `g`) |
| foot notes | 0 |
| signature | none |
| catchword | none |
| ornaments | none |
| `uncertain[]` entries | 21 (3 escalated) |

## Page structure

Recto, single text column with an outer (right) margin. Top to bottom:

1. Running head + folio.
2. Paragraph, 7 lines, continuing the sentence from p060 (`de, peut produire ſon propre frere…`), closing the previous annotation with `…conſideration. {g}`. Carries both markers.
3. Heading `TEXTE.`
4. Paragraph in the large TEXTE fount, 18 lines, `En ſecond lieu, il y a des teſmoins qui…` to `…trouuées au priſonnier.` — self-contained, no markers.
5. Heading `ANNOTAT. XL.`
6. Paragraph in the small annotation fount, 5 lines, `Ceci me faict ſouuenir de Q Fabius Maximus…`, running on to p062 (`continues_next: true`), ending in the letterspaced capitals `VERRVCOSVS`.

The lower third of the leaf is blank.

## `uncertain[]` entries — one line each

1. `blocks` — page quality: clean, sharp, evenly inked; pale brown stain across the head margin touches but does not obscure the running head and folio; facing-page strip on the inner (left) edge ignored throughout.
2. `blocks` — counts for the reconciler (as tabulated above).
3. `blocks[0]` — `continues_prev`/`continues_next` reasoning for the three paragraphs.
4. `blocks[0].lines[2]` — `{f}` is followed by a colon (two dots at 4.5x), not a period; `carbien` set tight, normalized.
5. `blocks[0].lines[6]` — period then marker `{g}`; `vienten`/`enconſideration` set tight, normalized.
6. `blocks[2].lines[2]` — the surname is `Rols` (full-ascender l, no dot), not `Rois`; same at lines[6], where show-through above the o mimics an i-dot.
7. `blocks[2].lines[4]` — `tion(qu'ils` and `)laquel-` set with no space outside the parentheses, normalized; the high apostrophe of `qu'ils` beside the i-dot can read as a diaeresis.
8. `blocks[2].lines[11]` — two ragged ink blobs between `machoire` and `de` (one above x-height, one below baseline): not type, not a marker, not transcribed; `lamachoire` normalized.
9. `blocks[2].lines[13]` — `enfoncée,trois` and other tight settings normalized per §1.
10. `blocks[2].lines[12]` — every double-s on the page checked at 4x: `deſſus`, `aſſeure`, `leſquelles`, `deſſus citée` are all `ſſ`, none is `ſs`; the faint second c of `cicatrice` is a c.
11. `blocks[4].lines[0]` — **escalated**: no abbreviation point visible after the long-tailed `Q` of `Q Fabius`; transcribed without one.
12. `blocks[4].lines[3]` — `àl'imitatiõ`, `duLatin`, `V errues` normalized; nasal bars on `imitatiõ`, `enuirõs`, `mãmelle` transcribed as tildes.
13. `blocks[4].lines[4]` — `VERRVCOSVS` closed up from letterspaced capitals; block flagged `spaced_caps`; final sort is a capital S, not a long s.
14. `margin_notes[0].lines[0]` — ink blot on the final a of `Balde en la`; letter still legible.
15. `margin_notes[0].lines[1]` — `l. Parentes.` read at 6x (penultimate sort is e, not u).
16. `margin_notes[1]` — **escalated**: the glyph transcribed `P.` (twice) is a looping italic capital-P shape standing where `§.` belongs; may be this fount's paragraph abbreviation.
17. `margin_notes[1].lines[3]` — **escalated**: (a) the first sort read as italic long `ſ` (left-only nub) not `f`, sense agreeing (`l. ſi cui, §. ſi`); (b) the last word transcribed `fut.` as printed, though the citation sense wants `fur.` (D. de furtis) — possible wrong sort.
18. `margin_notes[1].lines[0]` — line ends `vxo` with no word-break hyphen (normal for this print); point spacing normalized.
19. `margin_notes` — placement: the f note sits exactly beside its marker line; the g note sits beside `du preuenu, ne vient en conſideration. {g}`. The margin is uncrowded, so both `beside_line` values are firm.
20. `foot_notes` — no foot block, signature or catchword; verified on `foot.jpg`, `margin-4.jpg`, `body-8.jpg`.
21. `blocks[0].lines[3]` — long-s check: `toutesfois` correctly has round s before f (§2); the mark after `accuſé` is a colon.

## For the reconciler

- The page is in very good condition; nothing is illegible. No `[?]`, `[...]` or `[abbr:]` was needed.
- The alphabet position is consistent: the page carries only `f` and `g`, continuing from p060 and closing the previous annotation before `TEXTE.` begins. No restart, so no `a2`-style keys.
- The three escalations are all in the small type: the missing point after `Q`, the `P.`/`§` glyph in note g, and `ſi.` / `fut.` on the last line of note g. All three are judgements about a single sort at the limit of the scan's resolution; reader A's independent reading should settle them.
- Two spots where a careless eye would "correct" the print: `Rols` (not `Rois`) in the TEXTE block, and `de fut.` (not `de fur.`) in note g.
- The ink blobs after `machoire` in the TEXTE block are the one thing on the page that could plausibly be mistaken for a marker; there is no note answering to them and the whole TEXTE block is unmarked, so they were read as specks.

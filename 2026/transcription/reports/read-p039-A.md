# Read report — p039, reader A (opus)

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p039.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p039.json` →
`1 ok, 0 failed`, exit 0, no warnings.

## Counts

| Item | Count |
|---|---|
| Body lines (paragraph lines only) | 36 (7 + 29) |
| Paragraph blocks | 2 |
| Heading blocks | 1 (`ANNOTAT. XXI.`) |
| Markers in body | 2 (`{a}`, `{b}`) |
| Margin notes | 2 (keys `a`, `b`) |
| Foot notes | 0 |
| Ornaments | 0 |
| `uncertain[]` entries | 7 |

Page furniture: `running_head` "PARLEMINT DE THOLOSE." (spaced caps), `folio` "39",
`signature` "C iiij", `catchword` null.

## Layout

Recto. Large-type text block of 7 lines at the top, finishing the sentence carried over
from p038 (`continues_prev: true`, ends cleanly at "vray-ſemblable."). Then the display
line `ANNOTAT. XXI.` in spaced capitals, then the annotation in smaller type, 29 lines,
running off the bottom of the page mid-sentence (`continues_next: true`, last line
"en comble, la pure verité de tous les faicts depuis ſon"). Two marginal citations in
italic in the outer (right) margin. No foot citation block, no catchword; the signature
"C iiij" sits centre-right below the last body line. Markers restart at `a` with this
annotation, which is consistent with p041 (whose keys also restart at `a`).

## uncertain[] entries — one line each

1. `running_head` — "PARLEMINT": the sixth letter is a narrow serifed I, not the wide
   three-barred E; printer's error for PARLEMENT, kept as printed.
2. `blocks[0].lines[5]` — a short thick horizontal stroke is printed above the *u* of
   `perſuaſible`; read as a plain u (foul type / broken sort), not as a tilde.
3. `blocks[2].lines[1]` — the accent on the final e of `roſité` is very small and faint;
   read as acute, grave not excluded.
4. `blocks[2].lines[2]` — sic `au Iuges` (no x); checked at high zoom, no trace of an x.
5. `blocks[2].lines[16]` — the word-break sign after `cele` prints low, near the baseline,
   so it can look like a period; transcribed `-` per conventions §1 (word continues
   "brees").
6. `margin_notes[1].lines[0]` — italic margin type indistinct; `ſoit veue` could also be
   read `ſoir vene`. Chosen on the sense ("ſoit veue l'annotat. xij.").
7. `signature` — the signature letter is an unbarred C (no crossbar, no spur, so not a G)
   on folio 39; flagged in case the gathering sequence suggests otherwise.

## Notes for the reconciler

- **Double s.** Every double-s on the page was zoomed to 4–5x. All of them are long s +
  long s: `paſſees`, `commiſſaires`, `poſſibles`, `l'euſſent`, `l'iſſue`, `l'eſſay`. There
  is no `ſs` on this page. If reader B has `ſs` anywhere, it is B's error.
- **Tildes.** Three tildes, all checked: `donnoiẽt` (line 2 of the annotation),
  `l'eſplẽdeur` (line 19), `grãd` (line 23). `l'eſplẽdeur` is a sic spelling of
  *l'esplendeur*; the tilde itself is unambiguous.
- **Words broken without a hyphen** (normal for this print, no `uncertain[]` per
  conventions §1): `Themi` / `ſtocles` (lines 15–16), `pou` / `uoit` (21–22),
  `ſou` / `uent` (25–26), and in margin note b `l'an` / `notat.`
- **Word division normalized** where the compositor set two words tight:
  `tousmoyens` → `tous moyens`, `memoire,l'euſſent` → `memoire, l'euſſent`,
  `comparaiſons,elle` → `comparaiſons, elle`, `renommée,au` → `renommée, au`,
  `luy,qui` → `luy, qui`, `Latro,grãd`, `Seneque,qui`, `impudent,deploré,&`,
  `i.desTuſcula-` → `i. des Tuſcula-`.
- **Punctuation checked glyph by glyph at clause boundaries.** Colons (two dots, verified):
  `abordé:`, `fait:`, `y a:`, `Lucule:`, `parolle:`. Periods (verified):
  `vray-ſemblable.`, `interrogué.`, `homme {b}.`, `tore.`, `cha.`, `xxiiij.`, `notat. xij.`
- **Margin note placement.** Note `a`'s first line sits between body lines 16 and 17 of the
  annotation (the margin is set on a tighter leading than the body); I recorded
  `beside_line` as line 17 ("l'heur de memoire excellente, & eternellement cele-"), which
  is the line the note's second line aligns with. The marker `{a}` itself is in line 18.
  Note `b` aligns cleanly with line 22.
- **`fons` not `ſons`** in line 28 ("recitoit de fons / en comble"): the glyph has a full
  crossbar on both sides, so it is an f. Likewise `conferant` (line 20) is f, not long s.
- **Condition.** The paper is clean; ink is even and dark across the body. The only faint
  spots are the accents on `roſité` and `deploré` and the small italic of the margin
  notes. No damage, no show-through worth recording. The facing page's outer margin is
  visible along the left (gutter) edge of the crops and was ignored.

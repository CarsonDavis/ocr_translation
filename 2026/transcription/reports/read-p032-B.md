# Reader report — p032, reader B

Output: `/Users/cdavis/github/translator/2026/transcription/reads/B/p032.json`
Validator: `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p032.json`
→ `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| body lines (all paragraph lines) | 36 |
| paragraph blocks | 6 (incl. 2 one-line Latin verse quotations) |
| heading blocks | 1 (`TEXTE.`) |
| markers in the body | 5 — `{e} {f} {g} {h} {i}` |
| margin notes | 5 — keys e, f, g, h, i |
| foot notes | 0 |
| signature | none (verso) |
| catchword | `en` |
| running head / folio | `ARREST DV` (spaced caps) / `32` |
| `uncertain[]` entries | 13 (2 flagged `escalate`) |

Block sequence: paragraph (6 ll., indented, ends mid-sentence) → verse
`Omnis in Aſcanio, chari ſtat cura parentis.` → paragraph (13 ll., contains the spaced
small capitals `FILS`) → verse `Omnis amor magnꝰ, ſed apertè in cõiuge maior.` →
paragraph (12 ll.) → heading `TEXTE.` → paragraph (3 ll., larger type, runs onto p033).

## `uncertain[]` entries, one line each

1. `folio` — the first digit is worn to a trace; only the `2` is clear. Read `32` from the manifest.
2. `blocks[0].lines[1]` — the opening word sits on a rubbed patch at the inner edge; read `ſçait`, could be `ſait`.
3. `blocks[0].lines[1]` **escalate** — `felon` vs `ſelon`: the glyph's crossbar shows only on the left of the stem (= long s shape), but the sense demands the adjective *felon*, parallel with *brutal, & deſnaturé*.
4. `blocks[0].lines[0]` — a raised tick plus a speck after `ne` at the line end; not shaped like this print's solid hyphen (cf. `Abſa-`, `reſpon-`), so nothing transcribed.
5. `blocks[0].continues_prev` — set `false` on the indent; no p031 transcription was available to confirm the sentence does not run on from the previous page.
6. `blocks[4].lines[0]` — `Dequoy` set solid; kept as one word rather than normalized to `De quoy`.
7. `blocks[4].lines[7]` **escalate** — a thin vertical ink stroke stands between `pœtes` and `deuiſent`; read as an inked space/type blemish, nothing transcribed for it.
8. `blocks[4].lines[9]` — the marker after `Pluton` is a very small raised italic `i`; faint, but it is the page's only marker for note `i`.
9. `blocks[6].lines[0]` — printed solid `tempseuſt`; split to `temps euſt` per §1.
10. `margin_notes[0].lines[0]` — `vergile` printed with a lowercase italic v (sic), here and in note `i`; the body has capital `Vergile`.
11. `margin_notes[4].lines[3]` — printed `aux`, no point; the sense wants `au x.` (Ovid, *Metamorphoses* X). Kept as printed.
12. `margin_notes[4].lines[4]` — `Metarmor-` sic (extra r) for `Metamor-`.
13. `margin_notes[1].lines[2]` — the tight-set numerals read `xiij. xvi. xvij.` (2 Samuel 13, 16, 17) with `& xviii.` on the next line; the final minim strokes are small and could be miscounted.

## Notes for the reconciler

- **Condition.** The page is clean and well inked except for two places: the upper inner
  corner (folio and the start of body line 2) is rubbed, and the paper carries many small
  dark specks that read like stray points at line ends. Two of those specks sit just past
  `nul ne` (line 1) and just past `ſon fils` (line 5); neither is the print's hyphen, which
  is a solid rectangular dash at mid-height.
- **Double s.** Every double-s on the page was checked at ≥3× and all are `ſſ`
  (two hooked ascenders): `auſſi`, `fuſſe`, `aſſez`, `laiſſe`. No `ſs` was found.
- **Punctuation checked at the clause boundaries**: `mort: toutesfois` and `dire ainſi:`
  are colons (two dots); `concubines:` likewise; `tuée. & fit` is a baseline period;
  `ſoy-meſmes.` is a period; `cauſe {e},` is a comma (tail below the baseline).
- **Spaced type.** Two places: the running head `A R R E S T   D V` and the heading
  `T E X T E .`; plus `F I L S` in small capitals inside the David paragraph. All closed up,
  with `spaced_caps: true` on the two blocks that carry letterspaced type — `blocks[2]`
  (the paragraph holding `FILS`) and `blocks[5]` (the `TEXTE.` heading).
- **Marker alphabet.** The page runs e–i continuously, so p031 should end at `d`.
- **Latin quotations.** Both are Virgil/Propertius verse set in italic and are transcribed
  as one-line `paragraph` blocks, not headings, per §6. `chariſtat` is set solid in the
  print and split to `chari ſtat`; `magnꝰ` uses the superscript-us sign U+A770 `ꝰ`.
- **`continues_*` judgments.** Both verse blocks interrupt running prose, so the prose
  blocks on either side are marked as continuing across them, except after
  `ſurmonte toutes les autres.` (a full stop) where `continues_next` is false.
- The last paragraph (`Ou iaçoit que l'interualle du temps euſt …`) is the start of the
  *TEXTE* section and runs onto p033; the catchword is `en`.

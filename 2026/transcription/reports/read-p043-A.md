# Read report — p043, reader A (model: opus)

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p043.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p043.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| item | count |
|---|---|
| body lines | 40 |
| blocks | 1 (a single `paragraph`, `continues_prev: true`, `continues_next: true`) |
| paragraphs | 1 |
| headings | 0 |
| markers in body | 11 (`z a b c d e f g h i k`) |
| margin notes | 11 (keys `z a b c d e f g h i k`, 42 printed note lines in all) |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 26 (1 escalated) |

Running head `PARLEMENT DE THOLOSE.` (spaced capitals, closed up); folio `43`, matching the
manifest. No signature, no catchword, no foot citation block.

The marker alphabet crosses the end of the printer's alphabet on this page: the page opens
with `z` (continuing p042) and restarts at `a` on the next line, then runs `a`–`k` without a
second `a`, so no `a2` keying is needed. Every `{x}` has a note and every note has a marker.

## `uncertain[]` entries — one line each

1. `blocks[0]` — page quality: clean CUDL colour scan, nothing damaged or lost; doubts below are about the setting, not legibility.
2. `running_head` — spaced capitals closed up per §3; ends with a point (p041 has a comma). Recorded here as the schema has no `spaced_caps` for `running_head`.
3. `lines[2]` `coupees` — no acute printed; the slanted mark above the first e is a brown paper fibre, not an accent.
4. `lines[4]` `de` — the horizontal mark over the e is verso show-through, not a macron.
5. `lines[10]` `malefice.)` — print sets a space before the closing parenthesis; closed up per §1. Colon after `ſeule` confirmed at 5x.
6. `lines[17]` `Gaſcongne` — small-capital G at x-height, transcribed as an ordinary capital per §3; reader B may read lowercase.
7. `lines[19]` `ſe peur deſlier` — sic, `peur` for `peut`; the r's shoulder arm is clear at 5x. `deſlier` is ſ+l, `diſſouldre` a true ſſ.
8. `lines[21]` `ja contracté` — printed with a true j (dot + left-hooked descender), unlike the i of `iour` on the next line; kept as printed.
9. `lines[27]` `ſe pour! diſſouldre` — **ESCALATED.** An unidentified sort stands between `pour` and `diſſouldre`: at 12x a heavy wedge tapering from ascender height to just above the baseline with a separate dot below — the shape of `!`, which is what is transcribed. Could equally be a turned `i` or a wrong sort. Certainly not a marker (all eleven markers are accounted for).
10. `lines[32]` `de laquelle` — printed `delaquelle` with equal hairline gaps; read as `de` + `laquelle`, reader B may read `de la quelle`. `a la femme` carries no grave, as printed.
11. `lines[35]` `diſſo-` — the line-end break sign prints as a short low stub close to a point, not the full bar of `demeu-`; transcribed `-` per §1. Same weak stub at `puiſſan-` (lines[32]).
12. `lines[36]` `aucun` — final n over-inked into a blob; two legs discernible at 4x. Tight commas normalized.
13. `blocks[0]` — omnibus spacing normalizations (tight commas, `amortie:&`, `huit&`, `gaſcongne)par`, `d'hõme)le`, `mariage ja`, `de laquelle`, `lues de la chair`).
14. `blocks[0]` — two unhyphenated line breaks transcribed as printed: `ne peu|uent` (lines[4]/[5]) and `toute cho|ſe` (lines[18]/[19]).
15. `margin_notes[2].lines[0]` `c. fin. P. ad` — the tall letter is f, not long ſ (left crossbar + absorbed i-dot at 12x; matches note g's unambiguous `fin`).
16. `margin_notes[2].lines[2]` `xxvj. q. v. c. ſi` — points after q and after v are present but small and low; reader B may read `q v.`.
17. `margin_notes[2].lines[3]` `per ſociarias` — only one narrow open c between `ſo` and `iarias` at 14x, no room for `rt`; transcribed as printed although note g spells the same canon `ſortia|rias`.
18. `margin_notes[3].lines[0]` `c. fina & il` — sic: fi-ligature, n, a clear round a; no l, no point, no tilde, where notes d and f both read `final.`.
19. `margin_notes[3].lines[2]` `de frig. & ma` — sic: the note ends on an incomplete `ma`, no point, nothing to the right edge of the column.
20. `margin_notes[6].lines[3]` `Glo. & au c-` — lone c plus a clear hyphen, continued by `final.` (`au c-|final`).
21. `margin_notes[7].lines[1]` `c ſi per ſortia` — no point after the leading `c`.
22. `margin_notes[10]` — the key is set as a **roman capital K** (the note body stays in margin italic); keyed `k` from the alphabet and the italic lowercase body marker. `tem. P. xxvij` has no final point.
23. `margin_notes[0]` — placement: the margin is full (42 note lines against 40 body lines), so notes do not sit level with their markers; measured offsets given per note (d begins five lines above its marker; f, g, h, k fall level; i one line below). Every `beside_line` is the nearest body line by vertical centre, approximate to about half a line.
24. `signature` — no signature, no catchword, no foot block; `foot.jpg` shows the leaf blank below the last body line, nothing lost to the crop.
25. `margin_notes[0].lines[0]` `P. ſed per il-` — raised line-end stroke transcribed `-`; the leading `P.` is the paragraph mark, transcribed as the letter P following p041's precedent.
26. `margin_notes[1].lines[0]` `l. ſi ſerua.` — round a plus point; a terminal us-sign (`ſeruꝰ`) cannot be wholly excluded at this size but the glyph matches the plain italic a of `adop.` above.

## What the reconciler should know

- **Source quality.** This is a clean CUDL colour scan of the actual leaf, not the hard-binarized
  Gallica microfilm behind p041. Ink is even, letterforms are sharp at 5–12x, and no text is lost
  to damage, blotting or the gutter. Disagreements between readers on this page should therefore be
  resolved by looking, not by weighing probabilities.
- **The crop keeps a strip of the facing page along the left (inner) edge** — `ét`, `l,`, `ſi`,
  `ent`, `de`, `ez`, `tes`, `age,`, `&`, `ne,`, `ole`, `de`, `an-`, `api-`, `qui`, `mét`. It is
  ignored; it belongs to p042.
- **The one real escalation is the `!` on lines[27].** Everything else is a judgement about what
  the compositor set, not about what is visible.
- **Layout is simple:** one unbroken paragraph of annotation prose (ANNOT. XXIII, continuing from
  p042 and running on to p044), no headings, no `TEXTE.` block, no ornament, no foot block.
- **The margin is overfull** and its notes run ahead of their markers by up to five lines; see
  entry 23. Reader B's `beside_line` values will very likely differ by a line in places — that is
  measurement noise, not a reading disagreement.
- **Four "sic" readings the reconciler should not silently repair:** `peur` for `peut` (lines[19]),
  the stray `!` (lines[27]), `c. fina` for `c. final.` (note c), and the truncated `& ma` ending
  note c. `ſociarias` in note b (against `ſortiarias` in note g) is a fifth.
- **ſſ vs ſs was checked at 5–7x on every double s:** `L'impuiſſance`, `impuiſſant`, `puiſſant`,
  `impuiſſance`, `diſſouldre` (×2), `diſſo-`, `puiſſan-`, `laiſſer`, `ſ'eſſaye`, `deſſus` (note g)
  are all true ſſ ligatures; `quelquesfois`, `toutesfois`, `nopces`, `laſcif` take round s where
  expected. No `ſs` spelling occurs on this page.

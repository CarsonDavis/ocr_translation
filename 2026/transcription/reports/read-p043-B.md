# Read report — p043, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p043.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p043.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| body lines | 40 (one paragraph block) |
| paragraphs | 1 (`continues_prev: true`, `continues_next: true`) |
| headings | 0 |
| markers in body | 11 — z a b c d e f g h i k |
| margin notes | 11 — z a b c d e f g h i k (every marker answered, every note claimed) |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 23 |

Running head `PARLEMENT DE THOLOSE.` (spaced capitals, closed up), folio `43`, no signature, no catchword.

The marker alphabet continues across the page boundary as expected: p041 ended at `o`, p042 must run `p…y`, p043 opens at `z` and then restarts at `a`. Only one `a` on the page, so no `a2` keying is needed.

## `uncertain[]` entries — one line each

1. `blocks[0]` — page quality: clean CUDL colour photograph, even ink, no damage; the only nuisance is loose paper fibres that imitate accents/tildes at reading size.
2. `blocks[0].lines[2]` — `coupées`: a real acute on the first e (confirmed at 7x with contrast); the long thin brown line above is a fibre, not a second accent.
3. `blocks[0].lines[4]` — a grey fibre arcs over the first `de` (not a tilde); a very faint speck follows the line-final `peu` but is not a hyphen.
4. `blocks[0].lines[17]` — `Gaſcongne` is set with a small-capital G, transcribed as an ordinary capital per §3; `Gaſcongne)par` normalized.
5. `blocks[0].lines[19]` — **sic** `ſe peur deſlier` for `ſe peut deſlier`; the r is certain and contrasts with the clear t of `ſe peut lier` earlier in the same line.
6. `blocks[0].lines[27]` — **an exclamation mark is printed** between `pour` and `diſſouldre` (`ſe pour! diſſouldre`); wedge stroke + separate baseline dot at 9x. Almost certainly a wrong sort. Flagged for the reconciler.
7. `blocks[0].lines[32]` — the print sets `de la quelle` with a visible gap; transcribed as the single word `laquelle`. `a la femme` has no grave, as printed.
8. `blocks[0].lines[35]` — the line ends `diſſo.` with a **round baseline point, not a hyphen** (compared directly at 7x with the solid dash of `demeu-` on lines[38]).
9. `blocks[0].lines[36]` — the final n of `aucun` is filled with ink; reading secure from letter count.
10. `blocks[0].lines[38]` — **sic** `amortie` (feminine) after masculine `eſtaint`.
11. `blocks[0]` — three line breaks printed with no hyphen: `peu`/`uent`, `cho`/`ſe`, and margin note g `al`/`leguez.`
12. `blocks[0]` — spacing: compositor sets punctuation tight in many places and leaves a space before a comma/point in others; all normalized per §1.
13. `running_head` — spaced capitals closed up; ends with a **point**, where p041 ends with a comma.
14. `margin_notes[0].lines[0]` — the break sign after `il` is the margin italic's high, faint hyphen; a reader may take it for a speck.
15. `margin_notes[0].lines[0]` — the § sort at the head of notes z, a, b, e, k is transcribed `P.` throughout, following p041's established practice.
16. `margin_notes[1].lines[0]` — `l. ſi ſerua.`: the last two glyphs are over-inked; `ſeruo.`/`ſeruu.` cannot be excluded.
17. `margin_notes[3].lines[0]` — `c. fina & il`: **no l ascender** at 11x, where note d's `final.` in the same column shows a clear one. Read as printed; probably a compositor's slip.
18. `margin_notes[3].lines[2]` — note c stops mid-citation at `de frig. & ma` with no point and no break sign; transcribed as printed.
19. `margin_notes[10]` — the note's key letter is printed as a **capital K** where the body marker is lowercase italic k; keyed `k` per §4.
20. `margin_notes[10].lines[1]` — the marks after `tem` and after `P` are low and slightly tailed; read as points (the pattern used elsewhere), recorded because §7 forbids deciding from sense.
21. `margin_notes[2].lines[3]` — note b reads `per ſociarias` (o-c-i) where note g gives the same canon as `ſortia|rias` (o-r-t-i); one is a misprint, both transcribed as they stand.
22. `signature` — no signature, no catchword, **no foot citation block**; the leaf is blank below the last body line (checked on `foot.jpg`).
23. `margin_notes[8].lines[1]` — `la i. es.` read as the ordinal ("la premiere es Corinthiens").

## Notes for the reconciler

- **Image quality is good.** This is a CUDL colour photograph, not the Gallica microfilm used for p041: ink is even, the italic margin is fully legible, and there is no damage, no manuscript stroke and no gutter loss. Doubt on this page is about what the compositor set, not about what survives on the film.
- **Two deliberate "as printed" readings that will look like errors**: `ſe pour! diſſouldre` (lines[27]) and `diſſo.` at the end of lines[35]. Both were checked at high magnification under raised contrast and both are what the type shows. Neither is a scan artefact.
- **Loose fibres are the main trap on this page.** Two of them (over `coupées` and over `de` on lines[4]) sit exactly where an accent or a tilde would. Under contrast they are grey/brown and cross the letters, while real accents are black and sit clear above. A reader working at reading size will very likely disagree with me on `coupées` and may add a tilde to `dẽ`.
- **`ſſ` vs `ſs`**: every double s on this page is a true `ſſ` (both strokes reach the ascender line) — `impuiſſance`, `impuiſſant`, `puiſſant`, `diſſouldre`, `laiſſer`, `puiſſan-`, `diſſo.`, `ſ'eſſaye`, `deſſus` (note g). No `ſs` was found. Checked individually at 6x.
- **Accents checked one by one**: `coupées` (acute present), `exceptee` (none), `nees` (none), `cachée` (acute), `donné`, `ordonné`, `contracté`, `pudicité`, `té`, `és actes`, `à leur` (grave), `a la femme` (no grave), `ou le mariage` (no grave). None was added or removed.
- **No `[?]`, `[??]` or `[...]` was needed anywhere** — nothing on the page is illegible.
- `blocks[0].lines[18]`/`lines[19]` split `cho`/`ſe` without a hyphen while `lines[11]` splits `cho-`/`ſe` *with* one, on the same word, seven lines apart. Both as printed.

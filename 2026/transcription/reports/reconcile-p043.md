# Reconcile p043

Output: `transcription/final/p043.json` — `normalize_spacing.py`: 0 lines changed; `validate_page.py`: 1 ok, 0 failed (exit 0).

## Differences decided: 7 (A 2 / B 3 / neither 0; plus 2 `beside_line` placements, B)

| where | A | B | chose | reason |
|---|---|---|---|---|
| blocks[0].lines[2] | `coupees` | `coupées` | B | black letter-weight acute on the first e at 4x; the brown fibre above the word is separate |
| blocks[0].lines[21] | `ja` | `ia` | A | dotted j with a hooked descender at 4x, unlike the i of `iour` on the next line; i/j as printed |
| blocks[0].lines[35] | `diſſo-` | `diſſo.` | B | round baseline point where the hyphen belongs (cf. the mid-height bar of `demeu-`); as printed, sic |
| margin_notes[2].lines[0] (note b) | `c. fin. P. ad` | `c. ſin. P. ad` | B | long ſ at 8x: no crossbar, no fi bar, unlike note g's clear `fin`; wrong sort, sic |
| margin_notes[7].lines[1] (note g) | `c ſi per ſortia` | `c. ſi per ſortia` | A | no point after the c at 4-8x |
| margin_notes[7].beside_line | line 25 | line 26 | B | placement only: note g sits level with `cor doubté … cau-` |
| margin_notes[10].beside_line | line 34 | line 33 | B | placement only: note k sits level with `ce d'hõme) … Ils ſ'en` |

## Shared mistakes fixed: 0

The agreed lines were not re-read. `de laquelle` (line 32) kept joined per the word-division ruling;
`ſe peur deſlier` (line 19), note c `c. fina`, note b `ſociarias` vs note g `ſortia|rias`, and the
roman capital K keying note k (key `k`) are all kept as printed with sic/uncertain entries.

## Escalations: none

The exclamation-mark-like sort at blocks[0].lines[27] (`ſe pour! diſſouldre`) was decided rather
than escalated: at 10x (`pages/strips/p043/body-6.jpg`, second line, x about 180-300) it is a
heavy wedge widest at ascender height, tapering to a point just above the baseline, with a
separate round dot beneath on the baseline — an `!` sort. Not a turned i (no serif, not an even
stem) and not foul type (crisp impression). Transcribed `pour!` + space, recorded as sic in
`uncertain[]` with `escalate: false`.

## Odd about the page

- The margin is crowded (42 note lines vs 40 body lines); notes z–e start 1–5 lines above their markers.
- Three sic items beyond the `!`: `ſe peur` (line 19), `diſſo.` (line 35), note b `ſin`; plus note c
  `fina` and an unfinished note c (`de frig. & ma`), and `amortie` after `eſtaint` (line 38).
- Markers z, a–k continue p042's alphabet (p042 ends at y). No signature, catchword or foot block.
- Last line: ends on the complete word `comme` with no break sign — the text runs on to p044
  mid-sentence, NOT mid-word.

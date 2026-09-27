# Read report — p067, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p067.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p067.json`
→ exit 0, `1 ok, 0 failed`. One warning, checked and rejected: `blocks[4].lines[0]: possible normalized
long s (se) in "Se plaignant en outre à pluſieurs, de ce"`. The word is the opening capital of the
TEXTE block and is printed as a round capital S; there is no long s to restore.

## Counts

| | |
|---|---|
| body lines (paragraph lines) | 32 (4 + 8 + 7 + 13) |
| paragraphs | 4 |
| headings | 3 — `ANNOTAT. XLV.`, `TEXTE.`, `ANNOTAT. XLI.` (all spaced capitals) |
| markers in the body | 5 — `{a}`, `{a2}`, `{b}`, `{c}`, `{d}` |
| margin notes | 4 — keys `a`, `a2`, `b`, `c` |
| foot notes | 0 (no foot citation block on this page) |
| running head / folio | `PARLEMENT DE THOLOSE.` / `67` |
| signature / catchword | `E ij` / none |
| ornaments | none |
| `uncertain[]` entries | 10 (2 escalated) |

## Page structure

1. paragraph, `continues_prev: true` — 4 lines of large text finishing the TEXTE that began on p066
   (`ble en ſa peau: …` / `… mourir.`).
2. heading `ANNOTAT. XLV.`
3. paragraph, 8 lines, ends `grand ſoupçon {a}.`
4. heading `TEXTE.`
5. paragraph, 7 lines of large text, ends `maiſon, ſi ne le diſoit.`
6. heading `ANNOTAT. XLI.` (see escalation below)
7. paragraph, 13 lines, `continues_next: true` — runs off the page at `… encor la qua`.

The marker alphabet **restarts at `a` inside this page**: `{a}` belongs to ANNOTAT. XLV and the second
`a`, in ANNOTAT. XLI, is keyed `a2` per §4, with its note keyed `a2` to match.

## `uncertain[]` entries, one line each

1. `blocks[5]` — **ESCALATED.** `ANNOTAT. XLI.`: X, then a small V-shaped sort riding above the line
   between X and L, then L I; after `XLV` the intended number is almost certainly `XLVI`.
2. `blocks[6].lines[4]` — `doyuq` sic for `doyue`: the final sort is a q (closed bowl + descender,
   identical to the q of `qu'on` in the same line); wrong sort, transcribed as printed.
3. `blocks[4].lines[2]` — `vouloyẽt`: the mark over the e is a broad horizontal bar (tilde), not the
   thin right-leaning acute used in `verité`/`accuſé`; could be a heavily inked acute (`vouloyét`).
4. `blocks[2].lines[2]` — `aduer-`: the break mark is the small low double-stroke hyphen sort, matching
   `ſollici-`/`ſuffi-` and unlike the round periods on this page; transcribed as a single hyphen.
5. `blocks[6].lines[2]` — an ink blot sits over the t of `ladite`; reading not in doubt.
6. `margin_notes[3]` — the key letter of the fourth note is a short thick horizontal blob, not a legible
   letter; read as `c` from the body sequence and its place in the margin run.
7. `margin_notes[3].lines[0]` — `l... P. quæ`: three evenly spaced round dots after the `l`, no letters
   discernible at 10x; the paragraph sort is transcribed `P.` as on the preceding pages.
8. `margin_notes[3].lines[2]` — `quar. rer. car.` ends with a round period (same sort as the period after
   `iur.`), yet the word runs on into `cerẽ` on the next line (`carcerẽ`); transcribed as printed.
9. `margin_notes` — **ESCALATED.** marker `{d}` on the last body line has no note anywhere: the margin
   run ends at `met. cau.` well above it, the rest of the margin is blank, and there is no foot block.
   The d citation presumably heads the next page's margin.
10. `margin_notes[0]` — note placement: on this page the citation run is set from the **top** of each
    annotation, so notes sit *above* their markers rather than drifting down (note `a` is beside body
    line 3 though `{a}` is on line 8; note `a2` is beside line 1 though `{a2}` is on line 7).

## Other things the reconciler should know

- **No preceding page was available.** `transcription/final/` stops at p061 (plus p159), so `continues_prev`
  on block 0 is set from the page itself (it opens mid-sentence, lower case, with the word `co-/gnu`
  split across the page boundary) and not from p066. The word broken at the page head could not be
  checked against p066.
- The crops carry a strip of the facing verso along the left (gutter) edge — `oire à / orts / oint /
  mais …` in `read/p067.jpg` and `Mar- / dia.` in `foot.jpg`. Ignored throughout.
- The body strips `body-1…8` also carry the first few characters of the margin column along their right
  edge (e.g. the `a` and `ter. C` beside `… ayant eſté aduer-`). Those are the margin note, **not** body
  markers; the only body markers are the five listed above.
- Paper condition is good: no damage, no tears, ink strong and even. The only blemishes are the blot over
  `ladite` (blocks[6].lines[2]) and rust-coloured foxing specks, one of which sits under the descender of
  `doyuq`.
- `ſſ` was checked at ≥3x on every double s: `aſſeuroit`, `confeſſion`, `aſſeurance`, `menaſſer`, `auſſi`
  (×2), `dreſſee`, `ſuffi-/ſamment` are all long s + long s; no `ſs` combination appears on this page.
- Misprints left as printed and *not* flagged beyond the entries above: `priſonier` (single n) in
  ANNOTAT. XLV against `priſonnier` in the TEXTE; `ceſt` without apostrophe; `calõnieuſement`; `paratre`;
  `reuerance`; line breaks without a hyphen at `co|gnu`, `iuſ|qu'à`, `mai|ſon`, and at the foot `la qua`.
- Signature `E ij` stands alone at the foot, right of centre; there is no catchword.

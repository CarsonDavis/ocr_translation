# read p067 — reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p067.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p067.json` → exit 0 (1 ok, 0 failed).
One warning: `blocks[4].lines[0]: possible normalized long s (se)` — false positive. The TEXTE opens with the capitalised word `Se` ("Se plaignant en outre…"); a capital S is not a long s, and the image confirms a plain capital S. No other warnings.

## Counts

| item | count |
|---|---|
| body lines | 32 (4 + 8 + 7 + 13) |
| paragraphs | 4 |
| headings | 3 (`ANNOTAT. XLV.`, `TEXTE.`, `ANNOTAT. XLI.`) |
| markers in body | 5 (`{a}`, `{a2}`, `{b}`, `{c}`, `{d}`) |
| margin notes | 4 (keys `a`, `a2`, `b`, `c`) |
| foot notes | 0 (no foot citation block) |
| signature / catchword | `E ij` / none |
| uncertain[] entries | 11 |

Page furniture: running head `PARLEMENT DE THOLOSE.` (spaced caps, closed up), folio `67`, no ornaments.
Structure: continuation paragraph (large type, `continues_prev`) → `ANNOTAT. XLV.` → 8-line annotation ending `grand ſoupçon {a}.` → `TEXTE.` → 7-line large-type text → `ANNOTAT. XLI.` → 13-line annotation running on to the next page (`continues_next`, last line `…encor la qua`, broken without a hyphen).
Marker alphabet restarts at `a` in the second annotation, so that marker and its note are keyed `a2` (§4).

## uncertain[] entries (11)

1. `blocks[2].lines[2]` — `aduer.`: the word aduer-/tie is broken at the line end but the print sets a round baseline period, not a hyphen (checked at 8x). Transcribed as printed. **escalate**
2. `blocks[4].lines[2]` — `vouloyẽt`: the mark over the e is a heavy wedge; read as a tilde (corpus has many `-oyẽt` forms and no `-oyét`), but in this larger type an acute reading `vouloyét` is arguable.
3. `blocks[5]` — heading `ANNOTAT. XLI.`: printed X L I with a small raised mark (risen space / wrong sort) between X and L; the sequence after `ANNOTAT. XLV.` suggests XLVI was meant. As printed. **escalate**
4. `blocks[6].lines[2]` — `cõrrainte`: wrong sort, r for t (verified at 7x against the t later in the same word); cõtrainte meant.
5. `blocks[6].lines[2]` — `ladite`: an ink blot covers the t; reading taken from surviving strokes and the parallel at `blocks[6].lines[11]`.
6. `blocks[6].lines[4]` — `doyuq`: the final letter is a bowl with a right stem running below the baseline (q) rather than the crossbarred e of `doyue`; a brown stain overlaps that descender, so the wrong sort is not certain. As printed. **escalate**
7. `margin_notes[3]` — the note's key letter is printed as a short dash, not a letter; keyed `c` from the body marker `{c}` and from its position after note b. **escalate**
8. `margin_notes[3].lines[0]` — `l... P. quæ`: three separate baseline dots after `l` (possibly under-inked sorts); the final ligature of `quæ` is heavily inked and could be a plain `que`.
9. `margin_notes[3].lines[2]` — `quar. rer. car.` / `cerẽ`: the word carcerẽ is broken across the two note lines and the break is again printed as a period, not a hyphen (same anomaly as entry 1).
10. `margin_notes` — **marker d has no note.** The margin block ends with note c; the rest of the margin (margin-4) and the foot of the page are blank, and there is no foot citation block. **escalate**
11. `signature` — `E ij` on folio 67 does not fit an alphabetical gathering sequence at this folio; transcribed as printed.

## Notes for the reconciler

- **Two period-for-hyphen breaks** (`aduer.` in the body, `car.`/`cerẽ` in the margin). The page also carries ordinary hyphens (`ſollici-`, `ſuffi-`, `nour-`), so the printer had the sort; this looks like a repeated wrong sort rather than a house convention. §1 says to transcribe any line-end word-break sign as a single `-`, but these are unmistakably round baseline dots, so I recorded what is printed and flagged both.
- **Double s**: every double-s on the page was checked at ≥3x and all are long-s + long-s ligatures — `aſſeuroit`, `confeſſion`, `aſſeurance`, `menaſſer`, `auſſi` (twice), `dreſſee`, `ſuffi-`/`ſamment`. No `ſs` on this page.
- **Tildes**: `grãde`, `calõnieuſement`, `priſõnier` (line 4 of the XLV annotation; the same word is plain `priſonier` two lines earlier), `neãtmoins`, `pẽſer`, `cõrrainte`, `vouloyẽt`, and `cerẽ` in the margin.
- Body text `ceſt` (no apostrophe) at the start of the XLV annotation, and `forte` with f (not `ſorte`) — both checked at high zoom.
- Note a's first line is printed beside the `aduer.` line; note a2 beside the first line of the XLV**I** annotation; note b's first line falls midway between the `a la mai` and `doyuq pas` lines (I recorded the former); note c's first line is exactly beside the `ſollici-` line.
- Ink/paper: light brown staining at the right of `doyuq` and a blot over `ladite`; otherwise clean, no damage, margins intact. `pages/strips/p067/margin-4.jpg` is blank paper.
- No preceding page was supplied as context (transcription/final/ stops at p061), so the `continues_prev` first word `ble` (…ſem-)`ble en ſa peau` and the marker sequence were read from this page alone.

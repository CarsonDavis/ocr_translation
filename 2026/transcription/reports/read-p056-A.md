# read-p056-A

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p056.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p056.json`
→ `1 ok, 0 failed`, exit 0. No warnings.

## Counts

| item | count |
|---|---|
| body lines (paragraph lines only) | 37 (34 + 3) |
| paragraphs | 2 |
| headings | 1 (`TEXTE.`, spaced caps) |
| markers in body | 1 (`{a}`) |
| margin notes | 1 (key `a`, 4 lines) |
| foot notes | 0 |
| ornaments | 0 |
| signature | none |
| catchword | none |
| `uncertain[]` entries | 10 |

Page furniture: running head `ARREST DV.` (spaced caps), folio `56`, verso.

Layout: one long paragraph continuing from the previous page (34 lines, ends
`ſtoit ſot, ni vieux, ni malade.`), then the display line `TEXTE.`, then the first
3 lines of the new TEXTE section in large type (`continues_next: true`). Below the
last line the page is blank — no foot citation block, no signature, no catchword.
The one margin note sits in the left margin beside body line 18.

## `uncertain[]` entries (10)

1. **`running_head`** — the head is printed `ARREST DV.` with a clear baseline period
   (checked at 450%); other versos in the run are `ARREST DV` with no period
   (p052 verified in its own image; `transcription/final` has no period for
   p040/042/044/046/048/050/052). Transcribed as printed. **escalate: true.**
2. **`blocks[0].lines[2]`** — a small dark dot between `vray` and `ſemblable`; at 700%
   it sits above the baseline and is smaller than this page's periods, so read as an
   ink speck and not transcribed. If printed, the line reads `vray. ſemblable`.
3. **`blocks[0].lines[2]`** — `qu' vn` is set with a full word space after the
   apostrophe, where lines 27 and 32 set the same elision solid (`qu'vn`); as printed.
4. **`blocks[0].lines[6]`** — a small dot at x-height between `De` and `ſtolidité`;
   higher and lighter than the page's periods; read as an ink speck, not transcribed.
5. **`blocks[0].lines[17]`** — the marker sort after `amis` is badly ink-filled; the
   x-height bowl fits italic `a`, matching the key of the single margin note. Read `a`.
6. **`blocks[0].lines[17]`** — `viellieſſe` **sic** (v-i-e-l-l-i-e-ſſ-e, verified at
   300%); the same page sets `vieilleſſe` correctly at `blocks[0].lines[22]`.
7. **`blocks[0].lines[25]`** — `ſçanoit` **sic, wrong sort** (n for u, i.e. `ſçauoit`).
   At 350% the letter after `ça` has the arched n top, matching the n of `peine` on the
   same line. Transcribed as printed.
8. **`blocks[0].lines[25]`** — a faint mark between `ſçanoit` and `il`; light, irregular
   and above the baseline; read as an ink speck, not transcribed.
9. **`blocks[2].lines[0]`** — `quatrieme` is set as two words with a full word space
   (`La qua trieme`) in the large TEXTE type; transcribed as printed.
10. **`margin_notes[0]`** — the note's leading key letter is heavily blotted; read as
    italic `a` to match the body marker. Numerals checked at 400%: `vij. c. xiiij.`
    and `lihiſt. c. vij.`

## Notes for the reconciler

- **Three suspicious dots.** Items 2, 4 and 8 are all small marks between words that
  a reader working from sense would happily read as periods. I compared each at 700%
  against the genuine period after `elemens.` (solid, on the baseline, as heavy as a
  stem). All three suspicious marks are smaller and sit above the baseline, so I left
  them out. They are the most likely A/B divergence on this page.
- **Running head period** (item 1) is the other likely divergence, and it is a
  run-wide policy question, not just a p056 question.
- **The Latin quotation is not set apart.** `Et locus, & tempus poſtulãt, Vt paucis
  rem ab- / ſoluamus,` runs inline in the italic display of the body measure and the
  second line continues straight into French prose (`ſoluamus, qui eſtoit le
  commencement…`). Per §6 it is therefore *not* a separate quotation block; I kept
  both lines inside the single long paragraph. A reader who split it out would produce
  a three-way block mismatch.
- **Marker alphabet.** This page's only marker is `a`. `transcription/final/p055.json`
  also carries a note keyed `a` (with, per its own `uncertain[]`, no marker printed in
  its body), so the sequence does not simply continue from p055 — the alphabet appears
  to restart here. Worth a coordinator check.
- **Double-s.** Every `ſſ` on the page was zoomed to ≥300%: `lieſſe`, `Meſſale`,
  `viellieſſe`, `vieilleſſe` are all long-s + long-s ligature. No `ſs` pairs found.
- **Condition.** The page is clean and well inked; no damage, no gutter loss, no faint
  passages. The right (inner) edge of every crop carries a strip of the facing recto;
  ignored throughout. `body-8.jpg` is blank paper below the last line.
- Other spellings transcribed as printed and *not* flagged (reading not in doubt):
  `caractaires`, `Baſcouz`, `qu'ou bien de`, `Amphyſtides`, `Sophyſte`,
  `Trapezonce`, `traittons`, `Frãçoys`, `Siẽne`.

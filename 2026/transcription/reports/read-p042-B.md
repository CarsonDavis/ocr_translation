# read-p042-B

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p042.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p042.json`
→ `1 ok, 0 failed`, exit 0. One WARNING remains (`blocks[0].lines[21]`, "possible normalized long s (sur)");
it was checked against the image and is a false positive — the word opens a sentence and is printed with a
roman **capital** S, which is never cut as a long s. Documented in `uncertain[]`.

## Counts

| | |
|---|---|
| body lines | 40 (one paragraph block, `continues_prev: true`, `continues_next: true`) |
| paragraphs | 1 |
| headings | 0 |
| markers in the body | 7 — `{q} {r} {ſ} {t} {u} {x} {y}` on lines 8, 13, 15, 16, 20, 22, 27 (1-based) |
| margin notes | 8 — keys `p q r ſ t u x y` |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 18 |

Page furniture: `running_head` "ARREST DV" (spaced capitals, closed up), `folio` "42",
`signature` null, `catchword` null.

## `uncertain[]` entries, one line each

1. **margin_notes[0]** — the note keyed **p** has **no marker anywhere in the body**; every line was swept at 2.5x full width and lines 1–8 again at 3.5–7x with contrast raised. Recorded per §4.
2. **lines[1]** — `auſsi` is long s + **round** s here (7x), against true `ſſ` in `auſſi` on lines[8] and lines[20]; transcribed as printed in each place.
3. **lines[5]** — the tilde on `ſemẽce` is narrower/more angled than the flat tildes of `naturellemẽt` on the same line; read as a tilde, not a circumflex.
4. **lines[19]** — sic `ſe rrouue` for `ſe trouue`: the first letter is a clean r (stem + shoulder flag), not a t stripped of its bar.
5. **lines[19]** — the point after `{u}` is very lightly inked; read as a period, but a reader could call it a speck.
6. **lines[19]** — the marker read `{u}` looks x-like at reading size; confirmed u at 12x by shape match with margin note u's key letter, and distinct from the true italic x on lines[21].
7. **lines[32]** — `Chreſtiẽne` carries a flat wavy **tilde**, not the slanted acute this print uses on `chaſtré` / `preallegué`.
8. **margin_notes[0].lines[1]** — `nou.` ends in a round baseline **dot**, not the raised hyphen stroke used elsewhere in these notes, although the word runs on as `uelles`.
9. **margin_notes[0].lines[4]** — `coll. iij.`: glyphs read three minims; sense (Novellae *de nuptiis* = collatio iiij, and p041 note b reads `col iiij.`) favours iiij.
10. **margin_notes[1].lines[0]** — `Gal. au xv.`: printed `Gal.auxv.` with no internal space, so the division is the reader's; `aux v.` is the alternative.
11. **margin_notes[1].lines[1]** — `par` / `tium.` breaks with no hyphen printed.
12. **margin_notes[2].lines[0]** — sic `l. quaritur.` for *quæritur*: a single round a, no æ ligature.
13. **margin_notes[6].lines[0]** — a faint dot between `lege` and `P`; read as a point, could be a speck.
14. **margin_notes[6].lines[1]** — `ual` is legible letter by letter but `ſi | ual` resolves to no citation I can identify (the title, *D. de ſicar.*, is plain); a faint speck after `ual` is not transcribed.
15. **lines[21]** — the validator's long-s warning on `Sur`, checked and dismissed (capital S).
16. **running_head** — spaced capitals closed up per §3; no heading block exists, so nothing carries `spaced_caps`.
17. **blocks[0]** — line breaks with no hyphen (lines[5] `contrain` / lines[6] `te`); hyphens are printed normally elsewhere on the page.
18. **blocks[0]** — spacing: compositor sets punctuation tight (`valu.L'impuiſſance`, `neceſſaires.q`, `ſemẽcecontrain`) or with a justification space before it (`nature , ou`, `meres : &`); all normalized per §1.

## What the reconciler should know

- **Page quality is good.** This is a clean CUDL colour scan, evenly inked, no damage, no manuscript
  marks, no show-through worth noting — much easier than the p041 microfilm frame.
- **The single real structural problem is the missing `{p}` marker.** Note p is the first in the margin,
  printed beside body lines 1–5, and the alphabet runs continuously from p041 (which ends at `{o}`).
  Reader A should be asked specifically whether they found a marker p; if not, this is a compositor
  omission and should be recorded as such in the final.
- **`ſſ` vs `ſs` really does vary on this page.** `auſsi` (line 2) is long s + round s; `auſſi`
  (lines 9 and 21) is the ſſ ligature. Every other double-s was checked at 4–9x and is a true ſſ:
  `impuiſſance` (1, 8, 17, 18), `paſſage` (5, 14), `neceſſaires` (8), `diſſoudre` (15), `puiſſance` (17),
  `aſſez` (23), `deſſeing` (28), `neceſſité` (33).
- **Two lowercase continuations after a full point** are printed as such and are not misreadings:
  line 9 `à la fẽme.` → line 10 `quand`, and line 33 `Chreſtiẽne.` → line 34 `mais`. Both points are
  clear round dots at 7x.
- **Margin note r** corrects a p041 reading: this page prints `c. hi, qui. xxx-|iij. q. vij.` in note u,
  i.e. the canon *Hi qui*; p041's final has `bi, qui xxxiij. q. vij.` in note c, which is the same canon
  misread. Worth flagging upstream.
- **Marker-shape crib for this font:** the italic `u` is two strokes with a curl that reads as an x at
  small size; the italic `x` is unmistakable crossing diagonals (see `forcee {x}.` on line 22). Compare
  against the key letters at the head of margin notes u and x, which are set in the same fount.
- No foot block, no signature, no catchword: blank from just below line 40 to the trimmed edge.

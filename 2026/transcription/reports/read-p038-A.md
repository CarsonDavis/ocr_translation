# read-p038-A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p038.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p038.json` → exit 0 (`1 ok, 0 failed`).
One warning, checked against the image and correct as printed: `blocks[2].lines[0]: possible normalized long s (si) in "Si fait en ſon audition ample diſcours"` — the word is the capitalised `Si` that opens the TEXTE, set with a round roman capital S; nothing was normalized.

## Counts

| item | count |
|---|---|
| body lines (paragraph lines) | 30 (8 + 22) |
| paragraphs | 2 |
| headings | 1 (`TEXTE.`, spaced capitals) |
| printed body lines incl. the heading | 31 |
| markers in the body | 2 (`{i}`, `{l}`) |
| margin notes | 3 (`i`, `k`, `l`) |
| foot notes | 0 |
| ornaments | 0 |
| uncertain[] entries | 5 (2 escalated) |

## Page furniture

- `running_head`: `ARREST DV` (spaced capitals, closed up per §3).
- `folio`: `38`, printed at the head left, matching the manifest.
- `signature`: none. `catchword`: none. `foot.jpg` holds only the last two body lines and blank paper — no foot citation block, no signature, no catchword.
- No decorated initial, no woodcut, no rule.

## Layout

Verso. Head line `38 … A R R E S T  D V`. Then the tail of the previous page's annotation in the small body type, 8 lines, ending `& religieuſes. {l}`. Then the centred display line `T E X T E.` Then the TEXTE proper in the large type, 22 lines, running off the foot of the page (`continues_next: true`). The page ends well above the bottom of the type area — the blank space below the last line gets no block (§6).

Margin: three italic citations, all in the upper left beside the annotation paragraph; `margin-2`, `margin-3` and `margin-4` are blank paper. `beside_line` set from the strip alignment: note `i` beside body line 3, note `k` beside body line 6, note `l` beside body line 8.

## uncertain[] entries

1. `margin_notes[1]` (escalate) — **note key `k` has no marker in the body.** Every line of the annotation paragraph was read at 3–4x; lines 4–6, where the note sits, show no raised italic letter. Recorded as a note without a marker per §4.
2. `margin_notes[2].lines[0]` (escalate) — **`c. penultisi-`.** At 10x the sorts after `penu` read l (tall ascender), t (short ascender with crossbar), dotted i, round s, dotted i, hyphen → `penultisi-` + `me`. At lower magnification the l/t pair collapses into one l (`penulisi-`). The expected citation is `c. penultime de prob.`, so the printed form is odd on either reading.
3. `margin_notes[0].lines[0]` — **`c. ex tranſ-`.** The line-end hyphen is barely inked: a faint speck at mid height, far lighter than the clean hyphen of `lite-` on the line below. Could be a hyphenless break (§1, which needs no entry) or a stray mark.
4. `blocks[2].lines[7]` — **`terét`.** The mark over the second e is a straight stroke rising to the right (an acute, matching `tré:` two lines down); the sense (`traiterent`) would want a tilde `terẽt`. Read as the printed acute.
5. `blocks[2].lines[11]` — **doubled `&`.** The ampersand ends line 12 (`que deuant &`) and opens line 13 (`& apres.`). Transcribed as printed; sic, not a reading doubt.

## For the reconciler

- Paper and ink are clean and even; no damage, no show-through worth noting, no manuscript marks. The only genuinely hard reading on the page is the `penultisi`/`penulisi` word in note `l`.
- Two features that look like errors but are as printed: `violen` / `ce.` at lines 6–7 of the annotation is a word broken across lines **with no hyphen** (§1 says this needs no entry), and the lowercase `comme` after `ce.` follows a true period (round dot on the baseline, checked at 4x), not a comma.
- Both `{i}` and `{l}` sit **after** the sentence period (`proces {i}. Et quand`, `religieuſes. {l}`), which is the printed order.
- `bornée` at line 5 is one word; the compositor sets a visible gap inside it (`borné e`), normalized per §1.
- Double-s checks done at ≥3x: `rudeſſe` (line 1) and `miſſa` (note i) are both long-s + long-s; there is no `ſs` pair on the page.
- The TEXTE paragraph has no marker at all, so the alphabet on the next page should resume after `l`.

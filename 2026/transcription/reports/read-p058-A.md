# read-p058-A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p058.json`
(reader A, model opus). Validator: `uv run --with jsonschema python scripts/validate_page.py
transcription/reads/A/p058.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| body lines | 32 (1 + 3 + 23 + 5) |
| paragraphs | 4 |
| headings | 3 (`TEXTE.`, `ANNOTAT. XXXVII.`, `TEXTE`) |
| markers | 9 — `{k}` (carry-over), then `{a}`–`{h}` |
| margin notes | 9 entries, 34 lines (one unkeyed + keys a–h) |
| foot notes | 0 |
| uncertain entries | 15 |

Page furniture: running head `ARREST DV` (spaced caps), folio `58` (matches the manifest),
no signature, no catchword, no ornament, no decorated initial, no foot citation block.

Layout: one carry-over line closing the previous annotation, `TEXTE.` + 3 lines of large
type, `ANNOTAT. XXXVII.` + 23 lines, `TEXTE` (no period) + 5 lines of large type that run
on to p059. The page ends about a third of the way down; `body-8.jpg` and `foot.jpg` are
blank paper below the last line.

## uncertain[] entries, one line each

1. `blocks[0]` — page overview: clean CUDL scan, brown damp stain in the top-left corner
   and upper margin clear of the type, no damage; furniture and layout as above; marker
   alphabet restarts at `a` after the carry-over `{k}` and runs a–h with no gaps.
2. `margin_notes[0]` (**escalated**) — the two-line note `Panor' au c. ij.` / `de maled.`
   has NO printed key letter, so key `null`; it stands beside the second large-type line,
   two lines below `{k}`, and is almost certainly `{k}`'s citation, which therefore has no
   keyed note.
3. `blocks[2].lines[1]` — `impudẽce`: the mark is a flat wavy bar (tilde), not the steep
   acute of `ſongé`; also `s'il`, `s'en`, `esbahir` carry a round s in the large type.
4. `blocks[4].lines[10]` — `cil` set as `c il` with a justification gap; normalized to one
   word per §1; `encores:` is a true colon.
5. `blocks[4].lines[11]` — sic `l'eſtré` with an acute (recurs at lines[21]; plain
   `l'eſtre` at lines[22]).
6. `blocks[4].lines[20]` — the word-break sign after `pre` (and after `preſu`, lines[8]) is
   a raised mid-x-height dot, not a bar; transcribed `-` per the p046 ruling.
7. `blocks[4].lines[21]` — the mark after `pauure` read as a **semicolon** (round upper
   dot, narrow tailed lower blob) against the page's evenly round colons; a colon is the
   plausible alternative.
8. `blocks[4].lines[21]` — **wrong sort, sic:** `ſeigueur` for seigneur (the letter after
   the g is unmistakably a u at 7x); a faint stray ink arc crosses this line and the one
   above, not type.
9. `blocks[6].lines[2]` — `teſmoins.` is a round baseline period although lower-case
   `entre` follows.
10. `margin_notes[1].lines[1]` — `P rogo.`: an italic capital P with an ordinary foot serif
    and no stroke, where the citation wants `§ rogo`; transcribed as printed.
11. `margin_notes[4].lines[3]` — `p̃te`: the p carries a horizontal abbreviation stroke at
    the top of the stem (absent from every other p on the page); `ꝑte` is the alternative
    rendering; the citation is presumably `de p[rae]ſump[tionibus]`.
12. `margin_notes[7].lines[3]` — sic `poſti` / `dens` where the citation wants `poſſidens`;
    the second tall letter has a crossbar and curled top (italic t), probably a wrong sort.
13. `margin_notes[5].lines[0]` — `ea,` read as a comma; and the genuine period after the
    ampersand on the next line is carried as `& .` because the §1 rules collide there (the
    print sets `&.`).
14. `margin_notes[0]` (second entry) — `beside_line` values were fixed by measuring row
    centres in `pages/read/p058.jpg`; the margin is set solid in a smaller face, so entries
    drift (note g begins six body lines above its `{g}`); note b's first line falls almost
    exactly between `nul d'eux…` (which carries `{b}`) and `luy qui vne fois…` (given).
15. `blocks[0]` (second entry) — spacing normalization inventory (tight settings,
    justification spaces before punctuation, the space inside `(qui conſiſte en fait )`),
    plus the long-s audit: `paſſé` and `l'aſſeuroient` are `ſſ`, no `ſs` spelling occurs,
    and `toutesfois`, `esbahir`, `s'il`, `s'en` keep the round s as printed.

## For the reconciler

- **`{k}` has no keyed note.** The unkeyed `Panor'` block is the only candidate. This is the
  one escalated item.
- Two probable wrong sorts to expect disagreement on: `ſeigueur` (body) and `poſti|dens`
  (note g). Both are transcribed as printed.
- Two marks reader B is likely to read differently: the tilde in `impudẽce` (vs `impudéce`)
  and the semicolon in `pauure; encor` (vs a colon).
- The abbreviation stroke on `p̃te` (note d) is a judgment call between `p̃te` and `ꝑte`.
- `& .` in note e line 1 is an artefact of the spacing rules, not of the print.
- Both `TEXTE` headings differ: the first has a period, the second does not.

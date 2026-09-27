# Read report — p050, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p050.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p050.json`
→ **exit 0**, 1 ok / 0 failed. One WARNING only:
`blocks[6].lines[0]: possible normalized long s (sur) in "Sur ce, eſt à noter qu'il y auoit trois"`.
Checked against the image at 6x: the word is printed with a **roman capital S** (`Sur`),
which is correct — there is no capital long s in this fount. False positive; nothing changed.

## Counts

| | |
|---|---|
| body lines | 29 |
| paragraphs | 4 |
| headings | 3 (`TEXTE.`, `ANNOTAT. XXVIII.`, `TEXTE,`) |
| markers in body | 6 — `{c}` `{a}` `{b}` `{c2}` `{d}` `{f}` |
| margin notes | 7 — keys `c` `a` `b` `c2` `d` `e` `f` |
| foot notes | 0 |
| uncertain[] entries | 11 |

Page furniture: `running_head` `ARREST DV` (spaced caps, no period), `folio` `"50"`
(matches manifest), `signature` null, `catchword` null, no ornaments.

Block layout, top to bottom:
1. paragraph, 2 lines, `continues_prev: true` (tail of ANNOTAT. XXVII, ends `… de ſurplus {c}.`)
2. heading `TEXTE.`
3. paragraph (the TEXTE quotation, large type), 5 lines
4. heading `ANNOTAT. XXVIII.`
5. paragraph, 12 lines
6. heading `TEXTE,` — **comma, not period** (verified at 5x: the mark has a tail below the baseline)
7. paragraph (TEXTE quotation, large type), 10 lines, ends `… de ſon propre nepueu.`

Below block 7 the page is blank — no foot citation block, no signature, no catchword
(verified on `foot.jpg`; the printed text along the right edge of that crop is the
facing page and was ignored).

## uncertain[] entries — one line each

1. **`margin_notes[3]` — the `c2` keying (escalate).** The page carries two notes lettered
   `c`: the carry-over one at the head of the page and the first `c` of the restarted
   ANNOTAT. XXVIII alphabet. Per §4 the second occurrence on the page is keyed `c2`, in the
   note and in the body marker. Reader B has most likely written `{c}` for both.
2. **`margin_notes[5]` — note `e` has no marker (escalate).** No `e` is printed anywhere in
   the body; every line of block 5 was zoomed at 4x+.
3. **`blocks[4].lines[8]` — the gap where `e` should be.** `interrogué .` is set with an
   abnormally wide gap before the period, on the very line the `e` note is printed beside;
   almost certainly the slot of the unprinted marker. Gap normalized per §1.
4. **`blocks[2].lines[3]` — `ſouuẽt`.** The mark over the `e` is a flat wavy bar (tilde), not
   the steep acute of `mangé` two words earlier; the two sorts sit side by side and differ
   plainly at 4x. A quick read gives `ſouuét`.
5. **`blocks[4].lines[4]` — stray raised mark after `ce` (escalate).** Two marks are stacked:
   a comma on the baseline and, directly above it at ascender height, a small raised
   comma/apostrophe stroke. Transcribed as a single comma.
6. **`blocks[4].lines[9]` — unmatched `)`.** A tall curved closing paren after `ici` with no
   opening paren anywhere on the page. Space before it closed up per §1. (`deſquellesnous`
   is set solid and divided per §1.)
7. **`blocks[4].lines[2]` — `inrerroguer` sic.** Third letter is unmistakably `r`, no
   crossbar and no `t` hook, at 6x. Not corrected.
8. **`blocks[6].lines[4]` — `lè` sic.** `appe-/lè` carries a grave accent (stroke descending
   left-to-right), not the acute of `appelé`. Not corrected.
9. **`margin_notes[6].lines[5]` — stray stroke in `du i. volu.`** A small raised horizontal
   stroke sits between `i.` and `volu.`, over no letter and not a legible tilde. Omitted
   from the line.
10. **`margin_notes[3].lines[3]` — `lad.`** Set solid as one word; kept as printed under the
    §1 contraction exception rather than divided into `la d.`.
11. **`margin_notes[0]` — beside_line accuracy.** The margin is set on a tighter leading than
    the body, so the notes drift upward against the lines they key; `beside_line` values are
    nearest-by-position and should be treated as approximate for notes a–f. The one that is
    exact and load-bearing is note `e`, which is printed level with `blocks[4].lines[8]`.

## What the reconciler should know

- **Condition:** the page is clean and sharp. No damage, no faint ink, no show-through that
  obscures anything. Paper texture is visible but never hides a letter.
- **The alphabet restart is the big structural fact on this page.** ANNOTAT. XXVII's last
  citation (`c`, Iean Imola / Alexandre) sits at the top margin; ANNOTAT. XXVIII then
  restarts at `a` in both body and margin. Hence the duplicate `c`. Expect a diff here.
- **Two `ſs` vs `ſſ` decisions were checked at 4–6x and go opposite ways** on adjacent lines:
  `commiſſaire` / `miſſaire` / `auſſi` are long-s + long-s (two tall ascenders); `groſsier`
  is long-s + **round** s. `præsbiteri` and `Toutesfois` take round `s` before `b` and `f`
  as the fount requires.
- **Marker/punctuation order differs marker by marker** and was read off the image, not the
  sense: `ſurplus {c}.` and `rendre {c2}.` and `foy {f}.` put the marker *before* the period,
  while `dire. {a} de` puts it *after*; `inrerroguer {b}:` takes a colon.
- **Word division normalized per §1** in: `le doit` (`do it`), `deſquelles nous`
  (`deſquellesnous`), `hors tout`, `nepueu` (`nep ueu`), `mal, ſ'il` (`mal,ſ'il`).
- **`igno` / `mignieuſe`** is a word broken across lines with no hyphen (§1 says that is
  normal here); no uncertain entry was raised for it.

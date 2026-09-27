# read-p028-B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p028.json`
(validator: `1 ok, 0 failed`, exit 0, no warnings)

## Counts

| item | count |
|---|---|
| body lines | 31 (11 + 16 + 4) |
| paragraphs | 3 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XIII.`) |
| markers in body | 5 (`k`, `l`, `m`, `n`, `a`) |
| margin notes | 5 (`k`, `l`, `m`, `n`, `a`) |
| foot notes | 0 |
| uncertain entries | 9 |

Running head `ARREST DV` (spaced caps), folio `28`, signature `null`, catchword `null`,
no ornaments. Every `{x}` has a note and every note has a marker.

## Layout

Verso. Small-type annotation paragraph continuing from p027 (11 lines, `continues_prev`),
then the display head `TEXTE.`, then the large-type TEXTE paragraph (16 lines, self-contained:
starts a new sentence and ends `reſts.`), then `ANNOTAT. XIII.`, then a small-type paragraph
of 4 lines that runs on to the next page (`continues_next`). Margin column has two groups:
notes `k`–`n` beside the top paragraph, and a single long note `a` (8 lines, Guillaume
Benedicti on cap. Raynutius) beside the bottom of the TEXTE block. `foot.jpg` is blank below
the last body line — no foot citation block, no signature, no catchword.

## uncertain[] entries (9)

1. `blocks[0].lines[0]` — mark after `{k}` is a round baseline blob read as a period; could be a tailless comma (sense favours comma).
2. `blocks[2].lines[2]` — `verifié` accent could be grave; and the line ends in a clear round period mid-sentence before `dudit`, kept as printed.
3. `margin_notes[1].lines[2]` — `l ſi nesem. P.`: round s in `nesem`, no period after the leading `l`; abbreviation not certainly resolvable.
4. `margin_notes[1].lines[3]` — `ſs deportat`: line-initial siglum is long `ſ` + small round `s` (no crossbars, so not `ff`; second stroke has no descender, so not `ſſ`). Sits at the key column but has no body marker, so treated as a continuation line of note `l`.
5. `margin_notes[1].lines[4]` — `D. de bon li.` / `berto.`: the word break is printed as a round dot, not a hyphen.
6. `margin_notes[4].lines[2]` — `ch. Raynutius`: first letter read as `c` (matches the `c` of `ceſte` below); could be read `eh.`
7. `margin_notes[4].lines[5]` — the `-que` sign after `itaq` is a roman semicolon, not the 3-shaped `ꝫ`, so transcribed `;`.
8. `margin_notes[4].lines[6]` — `159.` in worn old-style figures.
9. `blocks[4].lines[3]` — `par ainſi` is set tight as one token; split per the word-division rule (a reader who kept `parainſi` would differ here).

## Notes for the reconciler

- Page is clean: no damage, no show-through worth flagging. The top-outer corner of the leaf
  is slightly cockled in the scan but no text is affected.
- The `ſſ`/`ſs` check was done at 4–16x on every double-s. Genuine `ſſ`: `poſſe` (note l),
  `demandereſſe`, `intereſſé`, `fauſ|ſement` (broken across lines). The only `ſs` on the page
  is the note-l siglum in item 4 above.
- Marker `{k}` is printed hanging in the outer margin past the right edge of the text block;
  it is easy to miss on a tight crop.
- Marker `a` restarts the alphabet at the start of `ANNOTAT. XIII.` (previous run ended at `n`).
  Keyed plain `a`, not `a2`, since only one `a` appears on the page.
- Several line ends break words with no hyphen (`autho|rité`, `tor|che`, `fauſ|ſement`,
  `circonue|nuë`, `per|ſonne`), which is normal for this print and not flagged.
- `Rols` (Bertrande de Rols) is set in ordinary caps + lowercase here, not small caps.

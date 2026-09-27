# read-p058-B

Page `p058` (image 80, verso, folio 58, source cudl). Reader B, model opus.

## Output

- Path: `/Users/cdavis/github/translator/2026/transcription/reads/B/p058.json`
- Validator: `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p058.json`
  → `1 ok, 0 failed` (exit 0), no warnings.

## Counts

| | |
|---|---|
| body lines | 32 (1 + 3 + 23 + 5) |
| paragraphs | 4 |
| headings | 3 (`TBXTE.`, `ANNOTAT. XXXVII.`, `TEXTE`) |
| markers in body | 9 (`k`, `a`–`h`) |
| margin notes | 9 (1 unkeyed + `a`–`h`) |
| foot notes | 0 |
| signature | none |
| catchword | none |
| ornaments | none |
| `uncertain[]` entries | 8 |

## Page structure

1. `blocks[0]` paragraph, 1 line, `continues_prev: true` — the tail of the previous page's
   annotation, ending `volonté {k}.`
2. `blocks[1]` heading `TBXTE.` (spaced caps)
3. `blocks[2]` paragraph, 3 lines — the large-type Texte, ending `esbahir.`
4. `blocks[3]` heading `ANNOTAT. XXXVII.` (spaced caps)
5. `blocks[4]` paragraph, 23 lines — the annotation, ending `des ſemblables {h}.`
6. `blocks[5]` heading `TEXTE` (spaced caps, **no** point, unlike the first heading)
7. `blocks[6]` paragraph, 5 lines, `continues_next: true` — the large-type Texte running
   on to p059 (`… & en rendoyent raiſons`)

The lower third of the page below the last Texte line is blank: no foot citation block,
no signature, no catchword (`foot.jpg` and `body-8.jpg` are empty paper).

## `uncertain[]` entries (8)

1. `blocks[0].lines[0]` — marker `{k}` has **no** margin note on this page; its citation
   belongs to the alphabet run carried over from p057. Recorded as a missing note for key `k`.
2. `blocks[1].text` — **`TBXTE.` is a wrong sort, sic** (`escalate: true`). The second
   letter is a small-capital **B**: two closed bowls with a rounded right side and an
   enclosed lower counter. Checked at 10x against the plain three-armed `E` of `ARREST`
   in the running head and against the second heading `TEXTE` lower on the same page,
   whose `E`s are unambiguous.
3. `blocks[2].lines[1]` — `impudẽce`: the mark over the first `e` is a wide horizontal
   stroke hooked at both ends (tilde sort), not the short steep wedge used for the acutes
   on this page (`ſongé`, `volonté`). Read as a tilde (= *impudence*).
4. `blocks[4].lines[10]` — `& cil qu'il`: the compositor leaves a hairline gap after the
   `c`, so it reads as `c il` at page scale; at 4.5x the three letters are normally spaced.
   Transcribed as the single word `cil` per the word-division rule (§1).
5. `blocks[4].lines[21]` — **`ſeigueur` for *seigneur*, sic, wrong sort (u for n)**.
   Verified at 4x: the letter after `g` is an unambiguous `u`.
6. `margin_notes[0].lines[0]` — `Panor'au c. ij.`: raised comma-shaped abbreviation mark
   after `Panor`, set tight against `au`. Left closed up as printed; could also be read
   `Panor' au c. ij.`
7. `margin_notes[1].lines[1]` — `P rogo. D. de`: the sort is an italic capital **P**
   standing for the paragraph sign (`§ rogo`), not a `§` sort. Transcribed as printed.
   Same sort at `margin_notes[4].lines[0]` (`l. ſi cui. P.`) and `margin_notes[7].lines[0]`
   (`l ſicut P. ſu`).
8. `margin_notes[7].lines[0]` — `l ſicut P. ſu`: no point after the initial `l`, unlike
   every other note on the page (`l. merito`, `l. eum qui`, …). As printed.

## Checks made explicitly

- **`ſſ` vs `ſs`** — every double-s on the page zoomed to ≥4x:
  `paſſé` (blocks[4].lines[18]) **ſſ**; `l'aſſeuroient` (blocks[6].lines[4]) **ſſ**;
  `toutesfois` (blocks[4].lines[2]) round `s` before `f`; `esbahir` (blocks[2].lines[2])
  round `s` before `b`. No `ſs` combinations found.
- **Punctuation at clause boundaries** — zoomed individually:
  `faire {b}:` colon · `mauuaiſtié {c}.` point · `encores: & cil` colon ·
  `perſeuere {e}:` colon · `point {f}.` point · `l'aduenir {g}:` colon ·
  `ou pauure; encor` **semicolon** (upper dot + tailed lower mark) ·
  `par apres:` colon · `l'eſtre encores.` point · `teſmoins.` point (set with a space
  before it in the print; normalized away per §1).
- **Long/round s at `s'il`** — the print is inconsistent and both readings were confirmed
  at 5x: `que s'il a ſongé` (blocks[2].lines[0]) **round s**;
  `ſ'il eſt derechef` (blocks[4].lines[7]) **long ſ**.
- **Wrong sorts** — two found: `TBXTE.` and `ſeigueur` (above). Every other word was
  checked glyph by glyph for r/t, c/e, n/u, a/o substitutions; nothing else anomalous.
- **`ij` in the italic margin font** is set as the `ÿ`-shaped sort; transcribed `ij`
  (`leg. ij. dudum.`, `li. xxij. q. v.`, `l. ij de tranſa`, `Panor'au c. ij.`).

## Notes for the reconciler

- **Validator-forced spacing:** the validator (§1, "`&` gets a space on both sides")
  requires `margin_notes[5].lines[1]` to be written `"D. de cond. & ."`. The print sets
  it tight as `D.de cond.&.` — the space before the final point is a normalization
  artefact, not a reading. Reader A will hit the same rule.
- **Spacing normalized heavily in the large Texte type**, which is set very tight:
  `ne s'enfaut` → `ne s'en faut`, `Guerre,yauoit` → `Guerre, y auoit`,
  `(qui conſiſte en fait )facilement` → `(qui conſiſte en fait) facilement`,
  `laron.Et` → `laron. Et`.
- **Line-end word breaks without a hyphen** occur in the margin notes and are normal:
  `c. ſemel ma | lus`, `paruu | li`, `ſiue poſti | dens`, `l. ij de tranſa | ctio`,
  and in the body `le preſu- | mera` (hyphenated) vs none elsewhere.
- **Damage / ink:** a brown damp-stain runs across the top-left of the leaf, over the
  folio `58` and the outer margin beside the first Texte. It does not touch any letter —
  `58`, `uaiſe inſtruction`, and the `Panor'au` note are all fully legible.
  Show-through from the facing recto is visible in the top third but never obscures a glyph.
- **Layout:** the margin column is crowded but does **not** overflow to the foot; notes
  `a`–`h` run continuously down the outer margin and the last (`h`) ends level with the
  final annotation line. `beside_line` is supplied only for the unkeyed `Panor'` note
  (where §4 requires it and the alignment is unambiguous); for the keyed notes the
  margin leading is tighter than the body leading and the pairing could not be fixed to
  better than ±1 body line, so `beside_line` was left out rather than guessed.
- The annotation alphabet **restarts at `a`** under `ANNOTAT. XXXVII.`; the page's first
  marker `{k}` is the tail of the previous section's run.

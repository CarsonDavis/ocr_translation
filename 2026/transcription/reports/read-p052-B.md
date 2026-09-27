# Read report — p052, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p052.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p052.json`
exits 0 ("1 ok, 0 failed"). One warning remains and is correct as printed:
`blocks[4].lines[0]: possible normalized long s (sur)` — the word is "Sur" with a **capital**
S at the head of the ANNOTAT. XXXI paragraph, so a short s is what the print has. Checked
against the image at high magnification; no change made.

## Counts

| item | count |
|---|---|
| body paragraph lines | 30 (8 + 18 + 4) |
| printed column lines incl. headings | 32 |
| paragraphs | 3 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XXXI.`) |
| markers in body | 2 (`{b}`, `{a}`) |
| margin notes | 2 (keys `b`, `a`) |
| foot notes | 0 |
| uncertain[] entries | 6 |

## Page structure

- Verso. Folio `52` at the head left, running head `ARREST DV` in spaced capitals.
- Block 0: paragraph continuing from p051 (`continues_prev: true`), 8 lines, closing the
  previous annotation and ending with marker `{b}` after "telle humidité". No terminal
  punctuation after the marker.
- Heading `TEXTE.` (spaced capitals).
- Block 2: the TEXTE, 18 lines in the larger text type, complete in itself (ends
  "cicatrices."). No decorated initial — "En" is plain large roman.
- Heading `ANNOTAT. XXXI.` (spaced capitals).
- Block 4: the annotation, 4 lines; the page breaks off mid-word at "cicatri"
  (`continues_next: true`, no hyphen printed — normal for this press).
- No signature, no catchword, no foot citation block, no ornaments. The lower third of the
  page below "…les cicatri" is blank.
- Margin: two notes only, key `b` beside the first paragraph's last lines, key `a` beside
  the annotation. The alphabet restarts at `a` under ANNOTAT. XXXI, so `a` follows `b` on
  this page; only one `a` on the page, so no `a2`.

## uncertain[] entries (6)

1. `margin_notes[0].lines[2]` "au c. xxxij. des" — the final numeral is printed as the
   italic **ÿ** form of "ij" (two dots over a descending y-shape); transcribed decomposed
   as `ij`. Flagged `escalate: true`: a reader could plausibly write "xxxv", but the glyph
   carries two dots and a descender, which the italic `v` does not. **Most likely point of
   disagreement with reader A.**
2. `margin_notes[0].lines[1]` "Aphrodiſée" — a stray acute-like mark is printed just after
   and slightly above the final `e`. It belongs to no letter and is not transcribed.
3. `margin_notes[1].lines[0]` "l cùm in di" — no period is clearly printed after the
   opening `l` of the citation (the usual form is "l."). Transcribed without one.
4. `blocks[4].lines[2]` "…recognu tel pieça {a} mais" — the marker after "pieça" is a
   small, heavily blotted italic letter. Read as `a` from the margin note keyed `a` and
   from the alphabet restarting at this annotation.
5. `blocks[0].lines[4]` "…de la veuë, qui viẽt" — the mark over the `e` is flat, matching
   the tildes of "cõduits" and "mẽton", not the steeply slanted acute of "rarité" on the
   same page. Read as a tilde (`viẽt` = vient), not `viét`.
6. `blocks[2].lines[13]` "cicatrice ſur le ſourcil droit: où touteſfois" — "touteſfois" is
   set with a **long** s before `f` (first tall glyph has a left nub only; the second has a
   full crossbar and hooked top), against the usual rule. Two lines earlier the same press
   sets "d'autresfois" with a round s. Both transcribed as printed.

## Notes for the reconciler

- **ſſ vs ſs.** Checked every double-s at ≥3x. Long s + **round** s: `auſsi` (line 3 of
  block 0), `aſsiſté` (block 2 line 3). Long s + **long** s: `aſſigne`, `ſ'eſiouiſſent`,
  `l'eſpeſſeur`, `preſſer`, `paſſages`, `deſſous`, `groſſe`, `cognoiſſance`, `aſſeuré`.
  `d'autresfois` is round s + f. `touteſfois` is long s + f (see uncertain #6).
- **Punctuation.** Every clause-boundary mark was zoomed. `pleurent:` and
  `numeraires:` are true colons (two square dots, the lower one on the baseline; the faint
  diagonal under the colon in "numeraires:" is a paper fibre, not ink). `produits.` is a
  round period. `dit-` at the end of block 0 line 4 and `lar-`, `ca-`, `ſça-` are line-end
  breaks; `dit-` is set as the raised double stroke and is transcribed as a single `-` per
  §1.
- **Tildes.** `viẽt`, `cõduits`, `mẽton`, `tõboit`, `iãbes` — all flat marks, distinguished
  from the clearly slanted acutes of `rarité`, `humidité`, `voulté`, `contracté`, `eſleué`,
  `aſſeuré`. `leûre` carries a real circumflex; `veuë` a dieresis; `où` and `cùm` graves.
- **Ink specks.** There are small dark flecks above the two `u`s of "auquel" (block 2 line
  11) and above-left of the margin key `a`. They are not type and are not transcribed.
- **Page condition.** Clean, well inked, no damage, no show-through worth noting. The
  gutter strip of the facing recto is visible in the crops on the right and was ignored.
- Lines with a printed space before a comma ("deux , c'eſt", "trappe , &", "grande , &")
  were normalized per §1; the validator's spacing check passes.

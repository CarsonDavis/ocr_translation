# read-p031-A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p031.json`
(validator: `1 ok, 0 failed`, exit 0, no warnings)

## Counts

| | |
|---|---|
| body lines (paragraph lines) | 31 — 6 + 9 + 14 + 2 |
| paragraphs | 4 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XVI.`, both spaced caps) |
| markers in body | 7 — `{b} {c} {d}` then `{a} {b2} {c2} {q}` |
| margin notes | 7 — b, c, d, a, b2, c2, d2 |
| foot notes | 0 |
| uncertain[] entries | 10 |

Running head `PARLEMENT DE TOLOSE.` (printer's `TOLOSE`, kept), folio `31`,
no signature, no catchword, no foot citation block, no ornaments.

## Layout

Paragraph continued from p030 (6 lines, ends `du monde {d}.`) → heading `TEXTE.` →
the Texte in large type (9 lines, `Et diſcourant…tigat.`) → heading `ANNOTAT. XVI.` →
the annotation (14 lines, ends `Auquel propos Ouide dit {q},`) → the Ovid distich as a
2-line italic quotation (`Neſcio qua natale ſolum…non ſinit eſſe ſui.`). Page ends there;
the lower third of the leaf is blank.

The margin carries two runs: the tail of the previous annotation (b, c, d) beside the
first paragraph, then — after a gap level with the Texte — the new run for ANNOTAT. XVI.
The new run's notes are set compactly and sit well above their markers (note `a` starts
level with the `ANNOTAT. XVI.` head, `d2` is printed beside body line 11 although its
marker is on line 14); `beside_line` records where each one actually starts.

## uncertain[] — one line each

1. `blocks[4].lines[13]` — the last marker is unmistakably an italic **q** (bowl + straight
   descender, no ascender); transcribed `{q}`, but there is no note keyed `q` and the note
   `d2` (*Ouid. au j. de Ponto.*) is plainly its note → almost certainly a misprinted `d`. **escalate**
2. `margin_notes[6]` — the same mismatch from the note side: key `d2` has no `{d2}` marker. **escalate**
3. `margin_notes` (general) — key numbering: the alphabet restarts at the ANNOTAT. head, so
   the page has two b/c/d notes; the restarted run is keyed `a, b2, c2, d2` per §4. Reader B
   may have keyed them plainly — worth a look at the diff.
4. `margin_notes[1].lines[0]` — `l. ſi. C. ad I.`: the last sort is shaped like an italic
   capital **I** (flat serifs top and bottom, shorter than the lowercase `l` earlier in the
   line) though the citation wants `ad l. Falc.`; transcribed as printed.
5. `margin_notes[1].lines[1]` — a small isolated ink mark floats above the baseline between
   `Falc.` and `de`; no foot, no descender, so read as stray ink / a failed sort and not
   transcribed (could be a badly printed `l.`).
6. `margin_notes[4].lines[3]` — `la l. finale. C.`: the punctuation after `finale` is faint
   and worn; read as a period, but a comma or colon is possible.
7. `margin_notes[2].lines[2]` — `met. can.`: reads c-a-n, though the citation is
   *D. quod metus causa*; transcribed as printed, sic.
8. `blocks[4].lines[4]` — the break sign in `pourroit-on` is printed as a raised round dot,
   not the usual hyphen; transcribed `-` per §2.
9. `blocks[4].lines[5]` — `cequ'elle` is set solid; kept closed up as a contraction (§1),
   but `ce qu'elle` is an equally defensible normalization.
10. `blocks[0].lines[3]` — the mark over the `e` of `fẽme` is a flat wavy bar, read as a
    tilde (= femme), not the acute of `féme`; `enfãt` on the same line has the same mark.

## For the reconciler

- Ink and impression are good across the whole page; nothing is damaged or lost to the
  gutter. The only faint spot is the fourth line of note `b2` (`la l. finale. C.`).
- Double-s was checked at ≥3× on every occurrence: `ſſ` in `Promeſſes`, `Vliſſes`,
  `acceſſible`, `eſſe` (italic); `ſs` in `aſsiſe`. These are the likely A/B divergence points.
- Line ends without a hyphen are genuine and frequent here: `quel`/`ques`, `pre`/`ferer`,
  `com`/`me` — no uncertain entries per §1.
- `biens` (for *bien*) in the annotation's first line is printed so; left as is.
- `blocks[4].continues_next` is set true because the sentence runs into the distich, and the
  distich block's `continues_prev` is left false (same page, no heading between) — a judgment
  call the other read may have made the other way.

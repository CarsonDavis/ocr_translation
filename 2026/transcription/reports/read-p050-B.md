# Read report — p050, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p050.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p050.json`
exits **0** ("1 ok, 0 failed"), with one warning:
`blocks[6].lines[0]: possible normalized long s (sur) in "Sur ce, eſt à noter qu'il y auoit trois"`.
Checked against the image at 5x: the glyph is a **roman capital S** beginning the sentence,
not a long s, so the line is correct as transcribed. False positive.

## Counts

| | |
|---|---|
| body lines | 29 (2 + 5 + 12 + 10) |
| paragraphs | 4 |
| headings | 3 (`TEXTE.`, `ANNOTAT. XXVIII.`, `TEXTE,`) |
| body markers | 6 — `{c}`, `{a}`, `{b}`, `{c2}`, `{d}`, `{f}` |
| margin notes | 7 — keys `c`, `a`, `b`, `c2`, `d`, `e`, `f` |
| foot notes | 0 |
| ornaments | 0 |
| uncertain[] entries | 8 (1 escalated) |

## Page structure

Running head `ARREST DV` (spaced capitals), folio `50` at the outer left — matches the
manifest. No foot citation block, no signature, no catchword; the page ends with a blank
lower third below `mignieuſe de ſon propre nepueu.`

1. paragraph, `continues_prev: true`, 2 lines, ends `… de ſurplus {c}.`
2. heading `TEXTE.`
3. paragraph, 5 lines, the large-type quoted TEXTE (`Rendans raiſons bonnes …`)
4. heading `ANNOTAT. XXVIII.`
5. paragraph, 12 lines, the annotation
6. heading `TEXTE,` — **printed with a comma, not a period** (verified at 5x)
7. paragraph, 10 lines, the second large-type TEXTE

**Alphabet restart.** The page opens with marker `c`, carrying over the run from p049; the
alphabet then restarts at `a` under `ANNOTAT. XXVIII.`. The second `c` of the page is
therefore keyed `c2` in both the body and the margin, per §4 (same treatment as p031/p035).

## uncertain[] entries (8)

1. **`margin_notes[5]` (key `e`) — ESCALATED.** The note `Balde en la / l. præsbiteri. /
   ſur la fin. C. / de epiſ. & cle.` has **no `{e}` marker in the body**. I walked the whole
   annotation between `{d}` and `{f}` at 3–5x: the three suspicious wide gaps
   (`interrogué . Voire`, `raiſon , encor`, `ici ) le`) all contain plain punctuation only,
   no letter. Note placed beside `& de ſoy meſme, ſans en eſtre interrogué. Voire en`,
   which is where it stands in the margin and where the sense wants it.
2. `blocks[2].lines[3]` — `mangé ſouuẽt`: the mark over the `e` is a flat horizontal bar,
   visibly unlike the sloped acute on `mangé` two letters earlier; read as a tilde
   (`ſouuẽt` = souvent), not `ſouuét`.
3. `blocks[4].lines[4]` — after `par ce` there is a comma **plus two extra marks**: a second
   small comma-like blob to its right and a raised stroke at ascender height. Transcribed
   as a single comma; a battered semicolon cannot be ruled out (7x).
4. `blocks[4].lines[2]` — sic `inrerroguer` for `interroguer` (n-r-e-r-r), confirmed at 4.5x.
5. `blocks[4].lines[9]` — the `)` after `ici` has **no opening parenthesis** anywhere on
   the page. `deſquellesnous` is set tight and normalized to two words per §1.
6. `blocks[6].lines[4]` — sic grave accent on `appe-/lè` (stroke falls left-to-right,
   6x). Also a small stray dash below the baseline between `hors` and `tout`, taken as
   spacing material / debris, not punctuation.
7. `margin_notes[3].lines[4]` — `l ſolam.` with **no point after the `l`**, unlike
   `l. ſolam. C.` in the note keyed `a`; none visible at 5x.
8. `margin_notes[6].lines[5]` — a short raised dash stands between `i.` and `volu.`; not
   transcribed (stray sort or show-through), 9x.

## Notes for the reconciler

- **Double-s checked individually at 3–4x.** `commiſſaire`, `miſſaire`, `auſſi` are long s +
  long s; **`groſsier` is long s + round s**; `præsbiteri` in the margin has a **round** s;
  `Toutesfois` has a round s before `f`, as the rule expects.
- Other sic readings kept as printed, no entry needed per §1: `igno` / `mignieuſe`
  (for *ignominieuse*, broken across lines with no hyphen), `nepueu` set with a wide
  internal gap, `requis.ou` and `teſtament.D.` set tight (spacing normalized per §1).
- Line-end word breaks without a hyphen: `ca` / `pacité`, `com` / `miſſaire`,
  `teſ` / `moin`, `igno` / `mignieuſe`. Hyphenated breaks: `ve-`, `appe-`, `aucune-`,
  and `cau-` / `Ale-` in the margin.
- Marker `{a}` follows its period in the print (`ſon dire.ᵃ de laquelle`); transcribed
  `ſon dire. {a} de laquelle` with §4 spacing.
- The page is clean: no damage, no faint ink, no bleed-through into the text column. The
  strip crops keep a slice of the facing recto down the right-hand gutter edge; it was
  ignored throughout.

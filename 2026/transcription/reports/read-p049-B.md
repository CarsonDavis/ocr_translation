# Read report — p049, reader B

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p049.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p049.json`
→ `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| item | count |
|---|---|
| body lines | 32 (14 in the TEXTE paragraph, 18 in the ANNOTAT. paragraph) |
| paragraphs | 2 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XXVII.`) |
| markers in body | 2 (`{a}`, `{b}`) |
| margin notes | 2 (keys `a`, `b`) |
| foot notes | 0 |
| uncertain entries | 6 |

Page furniture: running head `PARLEMENT DE TOLOSE.` (spaced capitals, printed
`TOLOSE` without the H), folio `49`, signature `D`, no catchword.

## Layout

Recto. Running head + folio, then the display line `TEXTE.` (spaced capitals), then a
14-line paragraph in the large text roman, then the display line `ANNOTAT. XXVII.`
(spaced capitals), then an 18-line paragraph in the smaller annotation roman which runs
on to p050 (`continues_next: true`). The TEXTE paragraph opens a new sentence
(`Dequoy eſt aiſé…`) with a plain capital D, no decorated initial and no ornament, so
`continues_prev` is false; it closes with a full stop at `tin Guerre.` so
`continues_next` is false. No foot citation block; the lower third of the page is blank
apart from the signature.

The two margin notes sit low in the right margin (margin strips 3 and 4); strips
`margin-1.jpg` and `margin-2.jpg` are blank, i.e. the whole TEXTE section carries no
marginalia. Both notes are keyed and both keys have markers in the body, so nothing is
orphaned.

## uncertain[] entries (6)

1. **`blocks[2].text` — `ANNOTAT. XXVII.`** The numeral is spaced capitals and the gap
   between the two final `I` sorts is slightly wider than the other letter gaps, so it
   invites the misreading `XXVI I.`. At 8x, five sorts are visible (X X V I I) plus one
   full stop; p039 = `ANNOTAT. XXI.` supports XXVII ten pages on.
2. **`blocks[3].lines[0]` — `…du defendeur depo`** A faint light speck sits over the
   final `o` of `depo`; far lighter and smaller than the solid tilde of `demãdeur` on
   the same line, so read as paper/offset, not a tilde (the word runs on as
   `depo|ſent`). `defendeu r` is set with a visible gap before the r; normalized per §1.
3. **`margin_notes[0].lines[0]` — `l. ij. P. eiuſ-`** The sort transcribed `P` is the
   italic paragraph/§ mark of the citation fount (bowl with a curled top-left entry
   serif, matching the italic `D` on the next line). Transcribed `P` for consistency
   with the same sort in `transcription/final/p031.json` (note c, `P. j. D. de ſcr.`)
   and `p037.json` (note d, `P. qui confi tetur`). Could alternatively be rendered `¶`
   or `ꝑ`. Periods are set tight in the print (`l.ij. P.eiuſ-`) and spaced per §1.
4. **`margin_notes[0].lines[1]` — `dem. D. de te,`** A heavy ink blot covers the `m` of
   `dem` and whatever follows, so the full stop after `dem` is inferred from the
   citation pattern rather than seen. The mark closing the line sits low, is round with
   a tail below the baseline, and is clearly unlike the horizontal double-stroke
   hyphens that end lines 1 and 3 of the same note, so it is read as a **comma**, not a
   break sign — `te,` is therefore an abbreviation (`de te[ſtibus]`) and does not run
   on into line 3 (which begins `c.`, not a continuation).
5. **`margin_notes[1].lines[0]` — `Accurſe au`** *(escalate)* A round dot is printed
   between the key letter `b` and `Accurſe`; excluded from `lines` as belonging to the
   key, per §4. After `au` the final stem of the italic `u` carries a rising exit
   flourish that at 10x can be taken for a following `t` (giving `aut`). Read as `au`
   because the note continues `dit` on the next line and because
   `transcription/final/p027.json` note m has the identical wording `Accurſe au` /
   `dit Parag. j de`.
6. **`margin_notes[1].lines[1]` — `dit P. eiuſdem.`** *(escalate)* The sort closing the
   note is a small curled mark at x-height after the `m`, not the plain round dot used
   at the end of `margin_notes[0]` (`lo titul.`). Read as a full stop, but it could be a
   comma or an abbreviation sign, and there may be a tilde over the final `m`.

## Notes for the reconciler

- **Signature, not catchword.** The lone `D` below the last body line sits right of
  centre (about x 1220 of the 2045-px foot crop), not flush right where this book puts
  catchwords. It is the gathering signature: the finished pages give A at f.1, B at
  f.17, C at f.33 — gatherings of 16 leaves — so D at f.49 is exactly on pattern, and
  catchwords in this book appear only on the last leaf of a gathering (f.16 `ment`,
  f.32 `en`). Recorded as `signature: "D"`, `catchword: null`.
- **Running head is `TOLOSE`, not `THOLOSE`** — kept as printed per §5.
- **Sic readings left alone, no entry made** (per §1): `Le premiere` for *La première*
  (blocks[1].lines[8-9]), `le numeroſité` for *la numérosité* (blocks[3].lines[10]),
  and the unhyphenated page-internal word breaks `Mar|tin` (blocks[1].lines[12-13]),
  `depo|ſent`, `ſu|ffiſance` and `l'ad|uis` in the annotation.
- **`ſſ` vs `ſs` checked individually at 8-16x**: `aſſeuroient`, `neceſſaires` — both
  genuine `ſſ` (two long s). `ſuffiſance`/`inſuffiſans` and `difficulté` carry `ff`
  ligatures with a single following `ſ`. `quelquesfois` has a round `s` before the `f`,
  as expected. No `ſs` combination occurs on this page.
- **Punctuation checked at the clause boundaries**: `teſmoins:`, `monde:`,
  `vray-ſemblables:`, `autres:`, `ſimilitude:`, `huit:`, `reprochables:`, `lable:` are
  all true colons (two dots); `preuues.`, `grandes.`, `Guerre.`, `preuue {b}.` are round
  dots on the baseline. The print sets many of these tight (`teſmoins:&`,
  `monde:&`) or with a space before (`ſimilitude :le`, `reprochables : le`,
  `doctes , neantmoins`, `raiſons , &`); all normalized per §1.
- **Break signs**: the mid-line hyphen of `vray-ſemblables` and the line-end sign of
  `va-` are printed as a short raised mark rather than a conventional hyphen;
  transcribed `-` per §1/§2. The margin-note hyphens (note a lines 1 and 3) are the
  horizontal double stroke.
- **Image quality**: the page is clean and well inked; no damage, no show-through worth
  noting. The only real obstacle is the ink blot on margin note a line 2. Note that
  `pages/strips/p049/margin-*.jpg` are upscaled ~1.57x from `pages/full/p049.jpg`, so
  the strips look sharper than the native crop but carry no extra detail — the native
  file is the limit of the evidence for the two escalated margin readings.

# Read report — p037, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p037.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p037.json`
→ exit 0, `1 ok, 0 failed`. One warning, checked against the image and deliberately left:
`blocks[3].lines[0]: possible normalized long s (si) in "Si par les vulgaires…"` — the word is
the sentence-initial **capital** `Si`, and a capital S is never set as long s. Not a
normalization.

## Counts

| | |
|---|---|
| body lines | 34 (10 in the TEXTE paragraph, 24 in the ANNOTAT. paragraph) |
| paragraphs | 2 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XX,`) |
| markers in the body | 8 (`a b c d e f g h`, in order, one each) |
| margin notes | 8 (keys `a`–`h`, each matching a marker) |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 9 |

Page furniture: `running_head` = `PARLEMENT DE THOLOSE.` (spaced capitals, closed up),
`folio` = `37` (matches the manifest), `signature` = `C iij`, `catchword` = none.

## Layout

Recto. Running head + folio, then the display line `TEXTE.` (letterspaced capitals), then the
large-type TEXTE paragraph of 10 lines, which continues a sentence from the previous page
(`continues_prev: true` — it opens `Et que ſes femme, & ſeurs, luy fuſſent accarez…`, another
clause of the same pleading) and closes cleanly at `aux fins abſolutoires.`

Then the display line `ANNOTAT. XX,` and the annotation in the smaller text type, 24 lines,
running off the foot of the page mid-sentence (`…craint ſon au`) with **no** hyphen and no
catchword, so `continues_next: true`. The signature `C iij` is centred below the last line.
`body-9.jpg` is blank paper — the page has no foot citation block.

The marker alphabet runs `a`–`h` straight down the annotation, so p038 should open at `i`
(printer's alphabet: no `j`).

The margin is set in the italic citation type with tighter leading than the body, so the
notes drift upward relative to the lines they key: note `a`'s first line sits beside the
`ANNOTAT. XX,` heading row rather than beside its marker's line. `beside_line` records the
body line each note's *first* line is actually printed beside, read off the gutter-side
fragments in `margin-*.jpg` / the right edge of the body strips.

## Things the reconciler should know

- **The margin strips are cropped too tight on the left.** `margin-1.jpg` is effectively
  blank (it covers the TEXTE, which carries no notes) and `margin-2/3/4.jpg` clip the key
  letters and the first one or two characters of every note line. I read the margin from
  `pages/full/p037.jpg` (2780×3966) instead, cropping and enlarging with ImageMagick. Reader B
  working only from the strips will very likely be missing the note keys and line openings.
- **Ink quality.** The body is clean and sharp. The margin type is lighter and several
  citation words are over-inked into blots (`lis` in note `c`, the two letters after
  `liber.` in note `h`), which is where most of my uncertainty sits.
- **Two raised ticks.** A small raised apostrophe-shaped mark appears before `accarez`
  (TEXTE line 2) and again between `exemple` and its comma (annotation line 23). Same shape
  both times. I did not transcribe either; both are flagged. If B saw them as apostrophes
  this is the diff to settle.
- **`puiſſance.` in TEXTE line 5** is genuinely a period, not a comma — verified at 9×
  against the comma of `Rols,` on the same line (compact round dot on the baseline, no tail
  at all, whereas the comma has a long descending hook). It reads oddly mid-sentence; it is
  sic.
- **Double s.** This setting uses the `ſſ` ligature throughout, never `ſs`. Checked
  individually at 3–8×: `fuſſent`, `aſſeurant`, `puiſſance` (×2), `poſſeſſeur`, `cognoiſſance`,
  `poſſeſſion`, `auſſi`, `puiſſe`, and in the margin `poſſeſ-/ſionum` and `aquiſſi-`. No
  `ſs` anywhere on the page. `ſuffiſante` is `ſ…ſ` (single long s each time), and
  `toutesfois` correctly has a round s (long s is not used before `f`).
- **`parẽs`, not `parés`** (annotation line 11). At 8× the mark over the e is a horizontal
  wavy bar, not an acute; the word is `parens` (relatives), which fits `autre prochains parẽs`.
  This is the likeliest silent error for the other reader.
- **`quand ou`** (annotation line 8) is `ou`, not `on` — the second letter has two verticals
  joined at the bottom. Sic for `on`; no entry needed per §1.
- Punctuation normalized per §1 where the print sets it tight or spaced: `droict,il` →
  `droict, il`; `tierces b.parce` → `tierces {b}. parce`; `pres:comme` → `pres: comme`;
  `difficulté ,que` → `difficulté, que`; `( comme` → `(comme`; `.)il` → `.) il`; and the
  margin's `l.e.C.de` → `l. e. C. de` etc.

## `uncertain[]` entries (9)

1. `blocks[1].lines[1]` — raised apostrophe-like tick before `accarez`, not transcribed; could be a printed sort or an ink speck.
2. `blocks[1].lines[4]` — `puiſſance.` is a period (verified at 9× against a known comma), sic mid-sentence.
3. `blocks[2].text` — `ANNOTAT. XX,` final mark read as a comma (descends below baseline); a worn period is possible.
4. `blocks[3].lines[10]` — `parẽs`: the mark over the e is a tilde, not an acute.
5. `blocks[3].lines[22]` — raised tick between `exemple` and its comma, not transcribed; same shape as (1).
6. `margin_notes[0].lines[0]` — `ꝓ-`: p carries an abbreviation stroke, read as `pro` (giving `C. de prohib. ſeq. pec.`); could be plain `p`.
7. `margin_notes[2].lines[0]` — final word of `c. i. vt lis` is an over-inked blob; read `lis`, could be `lia` or a curled `lit`.
8. `margin_notes[6].lines[2]` — `app. l a. l. ab.`: the points are faint; could be `app. l. a. l. ab.`, and the mark after `app` may be a colon.
9. `margin_notes[7].lines[2]` — `D. de liber. ex`: the two letters after `liber.` are blotted together; read `ex` (giving `de liber. exhib.`), but `œ` is possible.

None escalated.

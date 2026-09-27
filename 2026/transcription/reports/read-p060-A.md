# read-p060-A

**Output:** `transcription/reads/A/p060.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p060.json` → `1 ok, 0 failed` (exit 0, no warnings).

## Counts

| item | count |
|---|---|
| body lines | 32 (13 TEXTE + 19 ANNOTAT.) |
| paragraphs | 2 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XXXIX.`, both spaced caps) |
| markers in body | 5 (`{a}` `{b}` `{c}` `{d}` `{e}`) |
| margin notes | 5 (keys a, b, c, d, e — every marker has a note, every note a marker) |
| foot notes | 0 |
| ornaments | 0 |
| signature / catchword | none / none |
| `uncertain[]` entries | 8 (1 flagged `escalate`) |

Page furniture: running head `ARREST DV` (spaced caps), folio `60` at the outer left —
matches the manifest's expected printed folio. Verso, so the margin column is on the left
and the gutter/facing-page sliver is on the right; the facing page was ignored.

## `uncertain[]` entries (one line each)

1. `blocks[1].lines[2]` — **sic `Preniierement`** for *Premierement*: at 8x the group after
   `Pre` is a two-legged `n` with a clear top arch followed by two dotted `i`s, not an `m`.
2. `blocks[1].lines[8]` — **sic `Cuerre`** for *Guerre*: the capital is a plain `C`, no bar
   or spur, at 6x.
3. `blocks[1].lines[11]` — **sic `dõnce`** for *dõnee*: at 10x the first of the two final
   round letters has an open counter and no crossbar (`c`); the second has a full crossbar (`e`).
4. `blocks[3].lines[7]` — **ESCALATE: `vnc ehoſe`** where sense wants *vne choſe*; the `c`
   and `e` appear transposed across the word space (12x + contrast: the letter closing `vn`
   has no crossbar, the letter opening `ehoſe` has a full one, and the word space is clear).
5. `blocks[3].lines[14]` — a small raised wedge between `comme` and `quand` (body-6.jpg
   x~1330, y~500) at apostrophe height, not on the baseline; read as an ink speck / stray
   sort and not transcribed.
6. `margin_notes[0].lines[0]` — the mark between `l` and `parentes` (margin-2.jpg x~400,
   y~1085) is a blob at ascender height, well above the baseline periods of the same note;
   read as the point of `l.`, but notes b and c print `l etiam` with no point, so
   `l parentes.` is possible.
7. `margin_notes[2].lines[1]` — the final letter of `alleguéz` is small and blurred at the
   margin strip's native resolution (margin-3.jpg x~500, y~575); read as italic `z` on sense
   (*pre-alleguez*), italic `ſ`/`s` not ruled out.
8. `blocks[3].lines[1]` — `des peres` is set tight in the print (`desperes`); separated per
   conventions §1 as two distinct words (round `s`, word-final *des*), not a period contraction.

## Notes for the reconciler

- **Unusually many wrong sorts on this page** — four in one page (items 1–4 above), where the
  brief warns of roughly one. All four were checked at 6–12x with contrast enhancement before
  being written as printed. Item 4 (`vnc ehoſe`) is the one that should be confirmed on the
  original: two adjacent wrong sorts, or a transposition, is unusual and a reader working from
  sense will almost certainly write `vne choſe`.
- `ſſ` vs `ſs`: **`auſsi`** (TEXTE line 1) is long s + round s — verified at 4x. By contrast
  `cognoiſſoyent` and `aſſeurance` (TEXTE lines 10, 11) are both true `ſſ`, and so is
  `ſouſtenu`/`conſtamment` (ordinary `ſt`). Reader B is most likely to differ on `auſsi`.
- `font` (annot. line 16, *ainſi qu'en ce cas font leſdits teſmoins*) is a genuine `f`, not a
  long `ſ` — the crossbar runs through on both sides, unlike the `ſ` of `leſdits` on the same line.
  Sense supports it (*the said witnesses do*).
- `parẽs` (annot. line 15) carries a straight macron-shaped tilde over the `e`; `auõs`, `grãde`,
  `hõneſtes` likewise.
- Word breaks without a hyphen are frequent here and are transcribed as printed, with no
  `uncertain[]` entry: `parfai|tement` (TEXTE 10/11), `ail|leurs` (annot. 5/6), `ex|cuſ.`
  (margin note c). Hyphenated breaks in the margin notes are printed as a small raised mark and
  are transcribed as a single `-`.
- Punctuation set with a space before it (`ſeruent .`, `receu a ,`) and with no space after it
  (`parlé:femmes`, `Gaſcogne:leſquelles`, `Rols:&`, `alliance c.Le`, `faict)ſi`, `( ainſi`)
  is normalized per conventions §1. Every colon on this page was checked at 7x for two dots:
  `parlé:`, `Gaſcogne:`, `Rols:`, `perſonne:` are all colons.
- The margin notes drift down relative to their markers because the column is crowded:
  note d (marker on annot. line 14) starts beside annot. line 18, and note e (marker on
  annot. line 18) beside annot. line 19, at the very bottom of the margin. `beside_line` records
  where each note actually starts.
- `aud. c. literas` (note e) matches the abbreviation used in `transcription/final/p050.json`
  (`re iud. aud. c.`), which is what settled the reading of the tight `e`+`aud.` cluster.
- Paper is clean; a light brown stain crosses the upper-left margin and the area left of the
  folio but touches no type. No damage, no faint ink, no foot block, no catchword.

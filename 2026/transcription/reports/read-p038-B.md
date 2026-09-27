# Read report — p038, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p038.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p038.json` → exit 0
(1 warning, checked and dismissed: `blocks[2].lines[0]` "possible normalized long s (si)" fires on
"**Si** fait en ſon audition…" — the glyph is a roman capital **S** at the head of the TEXTE
quotation, not a modernized `ſ`.)

## Counts

| item | count |
|---|---|
| body lines | 30 (8 in the annotation paragraph + 22 in the TEXTE paragraph) |
| paragraphs | 2 |
| headings | 1 (`TEXTE.`, spaced caps) |
| markers in the body | 2 (`{i}`, `{l}`) |
| margin notes | 3 (keys `i`, `k`, `l`) |
| foot notes | 0 |
| ornaments | 0 |
| uncertain[] entries | 6 (2 escalated) |

Page furniture: running head `ARREST DV` (spaced caps), folio `38`, **no** signature,
**no** catchword, **no** foot citation block. `foot.jpg` shows only the last two body lines
and blank paper below them.

## Layout

Verso. Top of the page carries the tail of an annotation (small type, 8 lines) continuing a
sentence from p037 — `continues_prev: true`. Then the display line `T E X T E .` (transcribed
closed up as `TEXTE.`), then the TEXTE quotation in large type, 22 lines, broken off mid-clause
at the foot (`…auſquelles il ſ'eſt`) — `continues_next: true`. All three margin notes sit in the
left margin beside the annotation paragraph; the margin is empty from `TEXTE.` down.

## uncertain[] entries — one line each

1. **`margin_notes[1]` (key `k`) — ESCALATED.** The note `c. cum lo- / cum, de ſponſ.` has **no
   `{k}` marker anywhere in the body**; the compositor appears to have dropped it.
2. **`margin_notes[2].lines[0]` "c. penultsi-" — ESCALATED.** Hardest reading on the page; the
   cluster after `c. penul` could also be `penulti-`, `penultie-` or `penulisi-`.
3. **`margin_notes[0].lines[0]` "c. ex tranſ-"** — the line-end hyphen is only partly inked
   (faint broken stroke), much lighter than the solid hyphens of `lite-` / `lo-`.
4. **`margin_notes[0].lines[3]` "reſt. ſpel."** — the ink-filled third letter of `ſpel.` shows a
   crossbar, so read `e`; the expected abbreviation is `ſpol.` (*de restitutione spoliatorum*).
5. **`blocks[2].lines[7]` "terẽt …"** — the mark over the second `e` is a thick wavy horizontal
   (tilde), not the thin slanted acute used in `ſterité` / `tré` on this same page.
6. **`blocks[2].lines[12]` "& apres. …"** — sic: the `&` ending line 12 is repeated at the start
   of line 13 (`que deuant &` / `& apres.`); the stop after `apres` is a period, not a comma.

## Notes for the reconciler

- **The missing `k` marker is the one thing to settle.** I checked every line of the annotation
  paragraph at 220%–800% on the strips and again on `raw/img060.jpg` (2941×4711), including all
  the wide justification gaps (`peut eſtre au parauant`, `ce. comme`) and both line ends. There
  are exactly two markers on the page: an italic `i` after `proces` (line 3) and an italic `l`
  after `religieuſes.` (line 8). The `l` is a plain slanted ascender with a foot — it is not the
  margin's `k`, which is printed as a roman/small-cap **K** (kappa-like). The `k` note is printed
  beside `ce. comme iadis…`, so the omitted marker most plausibly belonged at
  `ni violen-ce {k}.`
- **`ſſ` audit.** Only two double-s words on the page, both genuine `ſſ` (two long s, ligature):
  `rudeſſe` (body line 1) and `miſſa` (note `i`). Everything else that looks doubled is
  `ſ` + round `s` in `-ſes` endings (`religieuſes`, `perſonnes` has none) or a single `ſ`
  (`commiſe`, `eſpouſa`, `naſquit`, `deſquels`, `conſigner`, `abſence`, `auſquelles`).
- **Punctuation audit.** Checked at ≥300% at every clause boundary: `proces {i}.` period,
  `nonnains,` comma, `ce.` period (the following `comme` is lowercase — sic),
  `& apres.` period, `lit.` period, `Guerre:` / `mere:` / `mariage:` / `tré:` / `eſpouſa:` /
  `entreuindrent:` / `naſquit:` / `departement:` / `enſemble:` / `ce:` all colons,
  `religieuſes.` period before the marker.
- **Marker/punctuation order.** In line 8 the marker follows the stop: the print sets
  `religieuſes.` then the italic `l`, so the line is transcribed `& religieuſes. {l}`.
- **Spacing normalized per §1** in the margin citations: the print sets the citation points
  tight-left (`miſſa .c. lite-`, `ras .P. fin. de`, `reſt .ſpel.`); the validator's
  `normalize_spacing` rule moves the point to the preceding word (`miſſa. c. lite-`). Recorded
  in that normalized form; nothing was read differently.
- **Word division normalized** where the compositor set hairline gaps: `bornée contre`
  (printed `borné e contre`), `experience)` (printed with a space before the paren).
- **Line breaks without a hyphen** (normal for this print, no entry made): `violen` / `ce.`,
  `preten` / `du`.
- Ink and paper are clean; no damage, no show-through worth noting. A small ink speck sits in
  the blank area below the last body line (visible in `body-7.jpg` / `foot.jpg`) — not type.
- The gutter-edge sliver of the facing page (right edge of every strip) was ignored throughout.

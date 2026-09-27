# read-p029-B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p029.json`
(validator exits 0; one warning, explained below)

## Counts

| item | count |
|---|---|
| body lines | 29 (3 + 20 + 6) |
| paragraphs | 3 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XIIII.`) |
| markers in body | 1 (`{a}`) |
| margin notes | 1 (key `a`) |
| foot notes | 0 |
| `uncertain[]` entries | 4 |

Running head `PARLEMENT DE THOLOSE.` (spaced caps), folio `29`, no signature, no
catchword, no foot citation block, no ornament.

Layout: tail of the previous paragraph (3 lines, `continues_prev`), `TEXTE.`, the
20-line large-type text block (complete, ends `ce iour inouye eſpece de crime.`),
`ANNOTAT. XIIII.`, then 6 lines of small-type annotation that run off the page
mid-sentence (`continues_next`). Marker/note cross-check passes: the one `{a}` in
`blocks[4].lines[3]` has the one margin note.

## `uncertain[]` entries

1. `blocks[4].lines[1]` — a small raised wedge-and-tail sort after `Sirach`, before the
   comma; transcribed as an apostrophe (`Sirach',`). Shape matches the apostrophe of
   `l'argent`, not an italic marker letter, but it is meaningless as an elision.
2. `margin_notes[0]` — the margin note has **no printed key letter**; keyed `a` from the
   body marker it stands beside. No preceding page was available to check the alphabet.
3. `blocks[2].lines[12]` — the sort over the `a` of `ayãt` is a thick slanted stroke that
   could be taken for an acute; compared at 6x with the true acute of `uenté` and the
   tilde of `deuãt`, it is a tilde. Same sort over the `e` of `auroyẽt` (line 17).
4. `blocks[2].lines[3]` — the first letter of `calomnieuſement` is heavily inked and
   reads nearly as `e`; taken as `c`.

## For the reconciler

- **Validator warning is expected and correct.** `IL N'EST` in `blocks[4].lines[1]` is
  set in **small capitals** (large `I`/`N`, small-cap `L`, `E`, `S`, `T`, letterspaced).
  Capitals carry no long s, so `EST` is what is printed, not a normalized `eſt`. The
  containing paragraph is flagged `spaced_caps: true`. Verified at 4x.
- **Two word-breaks without a hyphen**, both normal for this print and both left as
  printed: `vo` / `ler` (`blocks[2]` lines 8–9) and `com` / `me`, `aſſem` / `bler`
  (`blocks[4]` lines 1–2 and 5–6).
- Every `ſſ` was checked at 3–7x: `poſſedé`, `laiſſer`, `aſſouui`, `raſſaſié`, `aſſem`,
  `richeſſes` are all **long s + long s** (ſſ ligature), none is `ſs`.
- Round s confirmed (not long s) in `toutesfois` (s before f) and `fils`.
- Clause punctuation checked glyph by glyph: colons after `mort`, `argent`, `Artigat`,
  `laiſſer`, `veriſimilitude`, `d'autruy`; periods after `pardon`, `iniuſtement`,
  `defendeur`, `crime`. The period after `defendeur` sits slightly below the baseline but
  is a round dot with no tail.
- The page is clean: no damage, no faint ink, no show-through worth noting. The right
  (outer) margin is empty except for the single note low on the page; the narrow column
  of text along the **left** edge of the crops is the facing verso and was ignored.

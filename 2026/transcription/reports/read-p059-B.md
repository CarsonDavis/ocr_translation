# Read report — p059, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p059.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p059.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

- Body lines: **37** (5 in the opening TEXTE paragraph + 32 in the ANNOT. XXXVIII. paragraph)
- Blocks: **3** — paragraph (5 lines, `continues_prev: true`), heading, paragraph (32 lines)
- Paragraphs: **2**
- Headings: **1** (`ANNOT. XXXVIII.`, spaced capitals)
- Markers in the body: **4** — `{a}` `{b}` `{c}` `{d}`
- Margin notes: **4** — keys `a` `b` `c` `d` (every marker has its note, every note its marker)
- Foot notes: **0** (no foot citation block)
- Running head: `PARLEMENT DE THOLOSE.` (spaced capitals); folio `59`; signature: none; catchword: none; ornaments: none

## `uncertain[]` entries (5)

1. `blocks[2].lines[3]` — **sic `moyen`**: the word broken over lines 3–4 is printed `affer` / `moyen`; at 5x there is no tilde and no final t, although the sense and the parallel `affer-`/`moyẽt` at lines 9–10 want `affermoyẽt`.
2. `blocks[2].lines[28]` — **sic `premeuu`**: printed p‑r‑e‑m‑e‑u‑u at 6x; the glyphs are unambiguous, the word is not (probably a doubled or wrong sort for `premeu`).
3. `margin_notes[0].lines[0]` — **sic `Aecurſe`**: at 10x the second letter is a closed bowl with a horizontal bar (e), unlike the open crescent c of `cep. arb.` in the same note, so the print sets `Aecurſe` for `Accurſe` (wrong sort).
4. `margin_notes[1].lines[0]` — **escalated**: `l. octui. p D.` — the lex glyphs read o + ct‑ligature + u + i, resolving to no lex I can name; the following `p` may be a plain p or an abbreviation sort (`ꝑ` / `in pr.`). The rest of note b (D. unde cognati, l. de tutela, D. de in integrum restitutione) is legible.
5. `margin_notes[3].lines[0]` — the final a of `iulia` is filled with ink; a or a blotted æ. Read as `iulia`.

## Notes for the reconciler

- **Page layout.** Recto. Five large-type lines finish the TEXTE paragraph carried over from p058 (`continues_prev: true`), then the display line `ANNOT. XXXVIII.`, then one 32-line annotation paragraph that ends mid-page at `teſmoins. {d}`. The bottom ~third of the page is blank — no foot block, no signature, no catchword. Section ends here, so `continues_next: false` on the last paragraph.
- **Marker alphabet.** The page's markers restart at `a`; no earlier page is transcribed yet, so continuity across the page boundary could not be checked against a preceding final.
- **Marker `{c}` placement** is odd but as printed: `ou le premeuu, ou le {c}. ma` / `riage` — the marker and its period sit between `le` and the line-broken `mariage`.
- **Tildes vs acutes.** Four tildes were checked glyph by glyph against the acute of `donné` on the same strip: `cõ` (l.1), `nioyẽt`, `niẽt`, `ſecõde`, `moyẽt`, `biẽ` all carry the long, nearly horizontal bar, not the steep wedge of the acute.
- **Accents as printed.** `qu'à deux` (line 5) has the grave; `qu'a mille` (line 7) does **not** — checked at 6x. No accent added.
- **Double s (checked at 3–6x each).** `n'eſgalaſſent`, `cognoiſſance`, `yſſus` are all **ſſ** (both strokes rise to ascender height); the margin note d has **ſs** in `miſsio.` (long s + short round s). `ſouuentesfois`, `ſeſdites` have the ordinary round s.
- **Line-end word breaks.** The short raised double-stroke hyphen is present after `pre-` `teſ-` `pa-` `affer-` `pa-` `teſmoi-` `auſ-` `d'o-` `Iuriſconſul-`. It is **absent** (word broken with no hyphen, normal for this print) after `affer`, `com`, `ſem`, `pour`, `teſ` (line 24), `ma` (line 29) — verified at 6x at the right edge in each case, no hyphen cut off by the crop.
- **Punctuation spacing** was normalized per §1 where the print sets it tight or with a space before: `luy:&` → `luy: &`, `ſeurs,ont` → `ſeurs, ont`, `veriſimilitude .` → `veriſimilitude.`, `enſeigné,quand` → `enſeigné, quand`, `perſonnes b.La` → `perſonnes {b}. La`, and inside the margin citations (`l.diem` → `l. diem`, `vnd .cog.l.de` → `vnd. cog. l. de`, `tutela .D. d-` → `tutela. D. d-`, `di.c.ſi.au` → `di. c. ſi. au`, `cep.arb.` → `cep. arb.`).
- **Condition.** Paper is clean and the impression strong. A light brown stain crosses the top margin and the right end of the running head (`THOLOSE.` / `59`) without obscuring it; a small ink/foxing blot sits under `perſonnes` on line 14 and another over the final a of `iulia` in note d. The facing-page strip along the left (gutter) edge of the crops was ignored throughout.

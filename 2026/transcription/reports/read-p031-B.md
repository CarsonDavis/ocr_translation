# Read report — p031, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p031.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p031.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| item | count |
|---|---|
| body lines (paragraph lines only) | 31 (6 + 9 + 14 + 2) |
| heading lines | 2 (`TEXTE.`, `ANNOTAT. XVI.`) |
| paragraph blocks | 4 |
| markers in the body | 7 (`{b} {c} {d}` then `{a} {b2} {c2} {q}`) |
| margin notes | 7 (`b c d` then `a b2 c2 d2`) |
| foot notes | 0 |
| `uncertain[]` entries | 9 |

Page furniture: running head `PARLEMENT DE TOLOSE.` (spaced caps), folio `31`,
signature `null`, catchword `null`, no ornaments, no foot citation block.

## Layout

Recto. Top block is the tail of the annotation continued from p030 (6 lines,
`continues_prev: true`), then the display line `TEXTE.`, then the 9-line large-type
TEXTE paragraph (`Et diſcourant mieux encor le faict …` / `… retourné audit lieu d'Ar-` /
`tigat.`), then `ANNOTAT. XVI.`, then a 14-line annotation paragraph, then the Ovid
distich set apart in italic as its own 2-line paragraph block
(`Neſcio qua natale ſolum, dulcedine cunctos` / `Ducit, & immemores, non ſinit eſſe ſui.`).
The lower third of the page below the distich is blank; `foot.jpg` and `body-8.jpg`
show bare paper — no signature, no catchword, no foot citations. `margin-4.jpg` is blank.

## `uncertain[]` entries (9)

1. `blocks[4].lines[13]` — **escalated.** The marker after `dit` is printed as an italic
   **q** with a full descender, but the matching margin citation is keyed **d**
   (`Ouid. au j. de Ponto.`). Marker `{q}` has no note; note `d2` has no marker.
2. `blocks[4].lines[13]` — **escalated.** sic `offeret` (letters read o-ff-e-r-e-t at 14x)
   where the sense wants `offerte`; apparent e/t transposition.
3. `blocks[4].lines[2]` — `tain pais;` is a **semicolon** (upper dot + tailed comma at 9x),
   not the colon the sense suggests; compare the true colon in `trie:` two lines later.
4. `blocks[4].lines[11]` — `preſqu' in-`: the compositor leaves a full word space after the
   apostrophe; kept as printed rather than closed up.
5. `margin_notes[1].lines[0]` — `ad I.`: the letter is a serifed straight stroke (italic
   capital I), though the citation (`C. ad l. Falcidiam`) wants lowercase `l`.
6. `margin_notes[1].lines[1]` — a small raised tick stands between `Falc.` and `de`;
   possibly a badly inked `l.` or a broken sort. Not transcribed as a letter.
7. `margin_notes[2].lines[2]` — `met. can.` read as `can.` (the final letter arches over at
   the top), where `cau.` (*metus causa*) would be expected.
8. `margin_notes[4].lines[3]` — `la l. finale.` — the mark after `finale` is a low dot with
   an ink speck above; read as a period, colon not excluded.
9. `margin_notes` (block-level) — key-numbering note: the alphabet restarts at `a` with
   ANNOTAT. XVI, so the page has two runs. Per §4 the repeated letters are keyed
   `b2 c2 d2` in both body and notes; the first `a` of the second run keeps the plain key.

## For the reconciler

- **The q/d marker mismatch is the one real decision on this page.** The alphabet on the
  annotation runs a → b → c and then jumps to a printed `q`; the margin note is
  unambiguously keyed `d`. I transcribed both as printed. If the project prefers to record
  the intended key, the body marker becomes `{d2}` and the two uncertain entries collapse
  to one `sic` note.
- `ſſ` vs `ſs` was checked at ≥5x on every double-s: **`ſſ`** in `Promeſſes`, `Vliſſes`,
  `acceſſible`, `eſſe`; **`ſs`** in `aſsiſe` (blocks[4].lines[10]). This is the one `ſs` on
  the page and a likely divergence point.
- Tildes verified at high magnification: `grãde`, `blãdices`, `fẽme` (the mark over the e is
  a tilde, not an acute — the page-size image reads misleadingly as `féme`), `enfãt`,
  `cõparaiſon`, `l'ĩmortalité`, `reditũ.`, `Cicerõ`.
- `pourroit-on` (blocks[4].lines[4]) uses the raised mid dot the compositor also uses for
  hyphens; transcribed as a plain `-` per §2, matching the precedent set in p009 (`a-il`).
- Word division: `cequ'elle` is set tight in the print; normalized to `ce qu'elle` per §1,
  consistent with every other final file.
- Line 3 of the TEXTE paragraph ends `& quel` and line 4 begins `ques` — a word broken
  without a hyphen, which §1 says is normal and needs no entry.
- Ink and paper are clean throughout; no damage, no show-through worth noting. The facing
  page's gutter strip on the left of the crops was ignored.
- Margin placement: note `a` drifts **upward**, its first line sitting beside the blank
  band just under `tigat.` / the `ANNOTAT. XVI.` head, while its marker is seven lines into
  the annotation. `beside_line` values are best-fit from the strip y-positions.

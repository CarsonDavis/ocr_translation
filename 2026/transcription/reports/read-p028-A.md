# read-p028-A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p028.json`
(validator: `1 ok, 0 failed`, exit 0, no warnings)

## Counts

| | |
|---|---|
| running head / folio | `ARREST DV` / `28` (verso) |
| body lines | 31 (11 + 16 + 4) |
| paragraphs | 3 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XIII.`, both spaced caps) |
| markers in body | 5 — `{k} {l} {m} {n} {a}` |
| margin notes | 5 — keys `k l m n a` |
| foot notes | 0 |
| signature / catchword | none / none |
| ornaments | none |

Structure: tail of ANNOTAT. XII (continues_prev, ends `dre entierement du Iuge {n}.`) →
heading `TEXTE.` → the large-type TEXTE paragraph (16 lines, self-contained, ends `reſts.`) →
heading `ANNOTAT. XIII.` → the first 4 lines of ANNOTAT. XIII (continues_next).
The marker alphabet runs k–n and then restarts at `a` under the new annotation heading —
the restart happens at the `ANNOTAT. XIII.` boundary, not mid-stretch, so it is keyed `a`,
not `a2`.

## uncertain[] — 5 entries

1. `blocks[0].lines[0]` — the `{k}` marker sits blotted at the extreme right edge; read as
   italic `k` + **comma** (tail clearly descends below the baseline); a period is the only
   other candidate.
2. `margin_notes[1].lines[2]` (`l ſi nesem. P.`) — citation word read `nesem` (third letter
   is a short round `s`, S-curved, not the `c` of a possible `necem`); the opening `l`
   carries **no point**, unlike the `l.` of every other citation on the page; the capital is
   unambiguously `P.` (identical sort to the `P.` in note `m`), not `D.`
3. `margin_notes[1].lines[3]` (`ſs deportat`) — the siglum opening this line is a long `ſ`
   followed by a **short round s**, no point. Functionally it must be the Digest siglum
   normally printed `ff.`, but it does **not** match the genuine `ff` ligature of `effract.`
   two notes lower (two tall descending strokes sharing a crossbar), so it is left as printed.
4. `margin_notes[4].lines[5]` (`itaq; teſt. i.`) — the *-que* sign after `itaq` is a
   **semicolon** shape (dot over comma), not the 3-shaped `ꝫ` that §2 describes; transcribed `;`.
5. `blocks[2].lines[2]` — `cõcluoit a l'encontre.` ends in a round baseline dot (period)
   even though the sentence runs straight on into `dudit du Tilh`; `a` carries no grave. Sic.

## Notes for the reconciler

- **Note `l` is one note of six lines**, not two notes. The margin sets *every* line flush
  left — key letters and continuation lines share the same x — so line 3 beginning with a
  lone `l` looks like a second key `l`. It is the `l.` (*lex*) opening a second citation
  inside the same note. There is only one `{l}` marker in the body.
- Explicitly checked double-s: `demandereſſe`, `intereſſé`, `poſſe` are all **ſſ** (both
  strokes tall, ligatured). The only `ſ`+round-s on the page is the margin siglum in item 3.
- Explicitly checked clause punctuation: `circonſtances,` `du Iuge,` `certeines,` are commas
  (tails below the baseline); `{l}.` `{m}.` `{n}.` `{a}.` `l'encontre.` `mari.` `reſts.` are
  round baseline dots; `amẽde:` `demandereſſe:` `mains:` `qu'extraordinerement:` are colons.
- Sic spellings left as printed, no entry needed per §1: `certeines`, `extraordinerement`,
  `proffitable` (ffi), `vengence`, `abuſee` (no accent), `reſts`, `Rols`.
- Two word-breaks with **no hyphen** (normal for this print): `fauſ` / `ſement` (TEXTE
  lines 8–9) and `circonue` / `nuë` (lines 10–11); also `autho` / `rité`, `tor` / `che`,
  `per` / `ſonne`.
- `par ainſi` (last body line) is set solid as `parainſi`; split per §1 word-division, in
  line with the 7 earlier `par ainſi` in `transcription/final/`.
- Page condition is good: clean impression, no damage, no show-through worth noting. The
  only hard spot is the blotted `{k},` at the right edge of line 1 and the small italic
  sigla in the top margin block.
- The margin block for `k l m n` sits beside body lines 1–11; note `a` sits low, beside
  TEXTE line 14 (`& pour la proffitable, en deux mille li-`), which is 13 lines below its
  own marker — the marker `{a}` is on the very last body line of the page.
- `foot.jpg` carries no citation block, no signature, no catchword — only the last two body
  lines and blank paper.

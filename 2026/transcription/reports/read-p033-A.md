# Read report — p033, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p033.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p033.json`
→ `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| body lines | 31 (11 + 19 + 1) |
| paragraphs | 3 |
| headings | 2 (`ANNOTAT. XVII.`, `TEXTE.`) |
| markers in body | 5 (`{c} {d} {e} {f} {g}`) |
| margin notes | 7 (`a b c d e f g`) |
| foot notes | 0 |
| uncertain[] entries | 9 |

Page furniture: running head `PARLEMENT DE THOLOSE.` (spaced caps), folio `33`,
signature `C`, no catchword, no ornaments.

## Layout

1. `paragraph` (continues_prev), 11 lines of the large TEXTE type, ending
   `temps repeu de belles paroles.` — the end of the TEXTE section carried over
   from p032.
2. `heading` — `ANNOTAT. XVII.` (spaced capitals).
3. `paragraph`, 19 lines of the small annotation type, indented first line
   (`Iadis…`), ending `corriger amiablement entre nous, & luy ſeul {g}.`
   The block carries `spaced_caps` because of `IEVEVX.` (printed `I E V E V X .`)
   in line 8.
4. `heading` — `TEXTE.` (spaced capitals).
5. `paragraph`, 1 line, `En fin fut contraint, le mettre en in-` (continues_next).

Below the last line, roughly under `mettre`, sits a single blotted capital: the
signature. The rest of the foot is blank — no foot citation block on this page.

## uncertain[] entries (9)

1. `margin_notes[0]` — note key **a** exists in the margin but **no `{a}` marker is
   printed in the body**; every body line checked at 3x–6x.
2. `margin_notes[1]` — same for note key **b**: margin note present, no `{b}` marker
   in the body.
3. `blocks[2].lines[17]` — punctuation after `{f}` (`cõme frere {f}, le reprendre &`)
   read as a **comma** (thin stroke dipping below the baseline); at 11x a flawed
   period is not fully excluded.
4. `blocks[2].lines[15]` — `ou` in `quelque tort ou iniure` carries a faint tick
   above the u; read as plain `ou` (no accent, no tilde).
5. `margin_notes[2].lines[2]` — `de ſeruit. vrb.`: the mark after `vrb` looks like two
   dots (possibly a colon), transcribed as a period; the initial letter read as
   italic **v**, not u.
6. `margin_notes[0].lines[2]` — `Cluentio. l. de`: the abbreviation is an italic
   **l** (lex); its shape can be mistaken for a figure 1.
7. `margin_notes[6].lines[2]` — `c xvij. Leuiti` has **no period after `c`**, unlike
   line 1 of the same note (`c. xviij.`); transcribed as printed.
8. `margin_notes[6].lines[3]` — `que c. xix.` is heavily inked and blotted.
9. `signature` — single blotted capital read as **C**; consistent with the gathering
   structure elsewhere in the book (A = pp. 1–16 with signatures on 1/3/5/7,
   B = pp. 17–32 on 17/19/21/23, so p. 33 is the first leaf of C).

## Notes for the reconciler

- **Missing a/b markers are the main thing to arbitrate.** Notes a and b are printed
  normally in the margin beside body lines 1 and 5 of the annotation, but the
  corresponding letters simply are not set in the text. Markers c–g are all present
  and clearly italic (`iniuſte {c}.`, `menaſſer {d},`, `mes {e}.`, `frere {f},`,
  `ſeul {g}.`). The alphabet runs a–g on this page and restarts at `a` on p034,
  so the sequence itself is sound.
- **Word breaks without hyphens** are frequent here and are transcribed as printed
  (no `uncertain[]` per conventions §1): body `les hõ` / `mes`; margin `Cluentio. l. de`
  / `bitores`, `de pi` / `gno.`, `Theſ` / `ſaloniciens`, `Leuiti` / `que`.
- **ſſ vs ſs**: every double s on this page was checked at ≥3x and all are true
  `ſſ` — `careſſé`, `reſſentans`, `menaſſer`. No `ſs` pairs found.
- **Tildes** (all verified at high zoom): `tãt`, `durãt`, `ſõdit`, `mẽt`, `nõ`,
  `ſeulemẽt`, `Iuriſcõ`, `cõmandé`, `auãt`, `prẽdre`, `hõ`, `cõme`.
- `recognu` (body line 1) has **no** tilde — checked at 6x.
- `Chriſtianiſ-` at the end of annotation line 11 ends in a low mark taken as the
  line-end word-break sign (conventions §1); it is set lower than a normal hyphen.
- Tight settings normalised to single spaces per §1: `ſi nous` (printed `ſinous`),
  `qui l'auroit` (printed with a gap inside `qui`).
- **Caution for whoever diffs this page:** the strip images under
  `pages/strips/p033/` were regenerated at 00:30 on 2026‑09‑22, mid-session. The
  body strips kept their old geometry (identical dimensions and byte sizes), but the
  four margin strips changed from 743×1350 to 903×~1115 with different cut points.
  All margin readings above were re-verified against the **new** strips.
- Page condition is good: no damage, ink strong throughout, only the last line of
  note g (`que c. xix.`) and the signature are blotted. A strip of the facing page
  runs down the left (gutter) edge of the crops and was ignored.

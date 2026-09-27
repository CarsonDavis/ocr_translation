# reconcile-p057

**Output:** `/Users/cdavis/github/translator/2026/transcription/final/p057.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/final/p057.json` → `1 ok, 0 failed` (exit 0).

## Counts

- Differences decided: 12 (the 11 diffed line pairs plus the key of the last note): **A 2 / B 9 / neither 1**.
  - A: `c reos ſan-` (f.0), `luxur. bo. cõt.` (g.4).
  - B: `ſans vier de` (body 24), `Iean d'[?]na` (a.0), `S. Tcmas en` (a.3), `quaſtion xiij.` (a.5), `S. Iean. e. x.` (d.0), `guinis. cx iij.` (f.1), `l. famoſi.` (h.0), `D. ad l. Iuliã.` (h.1), `maieù.` (h.2).
  - neither: key of `Les interpre-` set to `null` (both readers keyed it `i` by elimination; no i sort prints).
- Shared mistakes fixed: 0. One shared reading changed by instruction (the null key above). Also took B's `beside_line` for notes f, g, h (A's were three to five body lines too low; not in the diff).

## Escalations

1. `margin_notes[0].lines[0]` "Iean d'[?]na": the sort after d' is a blotted blob; probably A (`Iean d'Ananie`, Ioannes de Anania). Pointer: `pages/strips/p057/margin-2.jpg`, first line of note a, y≈240.
2. `margin_notes[1].lines[1]` "c de religioſ": key layout. Three consecutive margin lines open with a lowercase italic letter in the key column (`b Aut, alearũ`, `c de religioſ`, `c leuitique.`). At 5x both c's are the same italic sort as the text c of `c. xxiiij.` / `c xiiij.`; the Codex abbreviation is elsewhere a capital (`C de reb cre.`, `D. ad l.`), so typography does not settle it. Resolved by sense: `Aut, alearũ` needs `C. de religioſ.` to complete the citation of the Authentica, and `leuitique. / c. xxiiij.` is Leviticus 24 (stoning of the blasphemer), exactly what marker {c} wants. Same layout as both readers. Pointer: `margin-2.jpg`, last three lines.
3. `margin_notes[7].lines[2]` "maieù.": the last sort is an x-height u with a slanted accent stroke; not the tall ſt ligature `maieſt.` needs (compare `poſtulante`). Transcribed as the sorts read. Pointer: `margin-4.jpg`, second line, y≈90.
4. `margin_notes[8]` "Les interpre-": no key sort prints, only a speck at mid x-height in the key column at 8x. Key `null`; the pairing with `{i}` (body line 32) is recorded in `uncertain[]`. Pointer: `margin-4.jpg`, y≈160, key column.

## Notes on the page

- Wrong sorts confirmed and kept with sic notes: `Sauuenr`, `vier`, `facilemeot` (body); `Tcmas`, `quaſtion`, `S. Iean. e. x.`, `cx iij.` (margin).
- The hyphen sort in this italic and in the body often prints as a raised dot (`Blaſphema-`, `religieu-`, `ſan-`); a period sits on the baseline. That decided f.0.
- `bo.` vs `ho.` in note g decided at 12x: the glyph has a small closed bowl at the foot of the stem, like the b of `ſoubz` on the next line; the h key of note h has an open leg.
- Margin leading is tighter than body leading, so notes f–i print three to five lines above their markers; `beside_line` records the actual printed position.
- `anciéne` (acute, not tilde) and `Theologiẽs` (tilde) kept as both readers had them.

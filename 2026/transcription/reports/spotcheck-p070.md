# Spot-check report p070

Output: `transcription/spotcheck/p070.json` (validator: 1 ok, 0 failed), `transcription/spotcheck/p070.diff.md`, `transcription/spotcheck/p070.verdict.md`.

Agreement (diff_reads): 93.9% on body lines (31/33 identical, 0 unmatched); 84.6% on note lines (11/13).

Differences: 4 distinct (2 body-line, one of which is the heading also reported as a structural difference; 2 margin-note lines).

Master wrong (substantive): 2
1. blocks[5].text: final `TEXTE,` -> correct `TEXTE.` (round baseline dot with a burr, no tail; the page's comma sort has a long tail below the baseline).
2. margin_notes[2].lines[5]: final `iij. de reſcrip.` -> correct `lij. de reſcrip.` (ascender-height l plus two minims; completes `a` + `lij` = alij, c. nonnulli §. sunt & alij, de rescriptis).

Final correct: 1
- margin_notes[1].lines[1]: final `de ius qui à nõ` stands; spotcheck `de iis` withdrawn (middle glyph is a two-minim italic u at 10x).

Undecidable: 1
- blocks[4].lines[3] final mark after `officieuſement`: period (final) vs comma (spotcheck). Blob with a short lower-left tail, shorter than the page's commas and not clearly below the baseline; sense wants a period. Pointer: `pages/strips/p070/body-6.jpg`, x 550-720, y 20-100.

Notes for the reconciler: no foot block, no signature, no catchword on this page; key letter of note e is an unreadable blob (read e by sequence); `procueur` and the `!`-shaped I in XLIX are sic and agreed by both.

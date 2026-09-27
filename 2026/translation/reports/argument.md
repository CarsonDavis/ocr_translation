# Report: argument

- Output: `translation/sections/argument.md` — `check_markers.py argument`: 1 sections ok, 0 failed.
- Word counts: French 328, English 387.
- Glossary additions: `Argument et sommaire du faict` → "Argument and Summary of the Facts"; `soy disant` → "calling himself"; `veritables enseignes` → "true tokens"; `de ses œuvres` → "by his doings"; `merveilleuse perplexité` → "marvellous perplexity".
- Choices to review:
  - The two display lines `ARGVMENT ET SOM- / MAIRE DV FAICT.` are rendered as two lines with the hyphen break reproduced ("ARGUMENT AND SUM- / MARY OF THE FACTS.").
  - The printer's `Arnault du Tilh` is rendered "Arnaud du Tilh" per the prompt's name list; the printed form is *Arnault* (twice). Revert if the house wants the printed spelling in the front matter.
  - `en Gaſcongne` is translated as printed ("in Gascony"); note that the case file §4 places Artigat in a Languedoc enclave, not in Gascony — the Argument is the printer's, not Coras's, and is loose here. Worth a note in the apparatus.
  - `comme auſsi faiſoient faire les quatre ſeurs` — the doubled verb *faisoient faire* looks like a printer's slip for *faisoient*; translated "as likewise did the four sisters".
  - `tãt mieulx ſçauoit l'impoſteur farder ſes menſonges, que l'autre ſ'aider de la verité` → "so much better did the impostor know how to paint over his lies than the other to help himself with the truth" (*farder* = to paint/rouge; kept the image).
  - `les enfans ... declarez legitimes` — the Argument says "children" (plural) declared legitimate, though the *arrest* itself gives the goods to the one surviving daughter; translated as printed.
  - The section ends mid-sentence with a comma (`confeſſe au long l'impoſture,`), exactly as the French entry does; `sections.json` marks it `complete: true`. The reviewer should check whether the Argument continues on a page not captured in the section.
  - Historic present kept throughout, as in the French.
- `[unclear]` passages: none.
- Citations: none in this section.

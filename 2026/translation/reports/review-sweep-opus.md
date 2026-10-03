# Mechanical sweep (sweep-opus)

An independent mechanical pass over all 224 rendered sections (texte-112 is incomplete and not rendered). The English in the render was confirmed byte-identical to `translation/sections/*.md` before the sweep. Method: a script compared paragraph counts, page markers, citation markers, numbers (Arabic, Roman, spelled-out), negation-token counts and word ratios for every section, and every section was then read French against English for checks 1–6. The review diffs (dcc886d~1..4ad996a) were word-diffed on the prose only: 108 hunks in 40 files.

Plan: `translation/review/plan-sweep-opus.json` has 2 section fixes and 24 findings.

## Counts per check

| Check | Sections checked | Issues | Sections |
|---|---|---|---|
| 1 Sentence/clause alignment | 224 | 3 (1 garbled clause, 1 silent emendation, 1 documented slip) | annot-031, annot-039, annot-098 |
| 2 Numbers, dates, money, numerals | 224 | 0 | (every script hit was noise: *vn/vne*, compounds like *vingtdeux*, "3,750", *xvj.* = "sixteenth", numbers inside brackets) |
| 3 Proper names and places | 224 | 5 (silent name emendations; one house-rule clash) | annot-005, annot-008, annot-043, texte-26, texte-105 (case-file §4 *Guilhet* vs §3 *Guilher*) |
| 4 Latin/Greek quotations | 224 | 7 (1 Latin dropped and fixed; 4 short Latin lines kept only in the Notes, against the parentheses rule; 2 unnormalised parentheticals) | annot-031 (fix), annot-014, annot-016, annot-040, annot-101, annot-104, annot-106 |
| 5 Negations and modal reversals | 224 | 5 | annot-042 (*bien que* → "whether"), annot-071 (*ne … point seulement* → "nothing but", no note), annot-083 ×2 (banishment clause; scope of *tous … n'a point*), annot-104 |
| 6 Paragraph count | 224 | 1, harmless | annot-004 (a verse quatrain split across p010 in the French, one paragraph in the English; the page marker is kept) |
| Markers (⟦p⟧ and {x}) | 224 | 0 | (the annot-083 mismatch is a false positive from markers inside ⟨alt⟩) |
| 7 Review-commit slips | 40 files changed | 4 | annot-020 (fix), texte-20, annot-083, annot-104; case-file §4 Mane |

## Plan fixes

- **annot-031:** the one-line Manilius verse *Exemplumque Dei quisque est, in imagine parva* was translated but its Latin dropped. Restored in parentheses.
- **annot-020:** the review left "some woman of worth and honest until the end of the suit" with no noun after "honest". Fixed with commas, as the French has them.

## Most serious findings

1. **annot-083:** review pass 4 turned "did not wish to punish it with a simple banishment" into "wished to punish it only with a simple banishment". The print has no *que* (*ne l'ayent voulu punir d'vn ſimple banniſſement*), and the literal reading also makes sense. If the emendation is kept, record it in the Notes.
2. **annot-031:** the last sentence, "Even were, according to our Canonists, the crime as great …", has no main clause (French *Où fuſt … le crime ſi grand & execrable*).
3. **texte-26:** the Rieux judge's *amplié l'arreſt à lad. de Rols* is rendered "the decision". §5 reserves "decision" for the Parlement, so this reading is probably *arrest* in the sense of detention.
4. **annot-071:** *ne conſiſte point ſeulement en drogues* became "consists in nothing but drugs". The change may be deliberate, but nothing records it.

## Review commits: diff quality

All 108 prose hunks were reviewed:
- No page or citation marker changed.
- No duplicated words were introduced.
- No hunk left both the old and the new wording.
- Every bracket and asterisk is balanced.

Most edits are clean consistency swaps: Arnaud→Arnauld, relatives→kinsfolk, court→Court, Thil→Tilh, praenomens spelled out, "Joined that"→"Added to which", and S./sainct → St/Saint per the case-file rule.

The slips are of the kind the brief anticipated, a glossary phrase dropped in mechanically:
- **annot-020:** the *femme de bien* → "woman of worth" swap broke the grammar.
- **texte-20:** the same swap left a clumsy zeugma.
- **annot-104:** the reviewer applied the annot-005 *nul ne ſçait* idiom ruling and left the section's Notes contradicting the prose.
- **annot-083:** the reviewer reversed an explicit negation on legal grounds without a Notes entry.

In the case file, the §4 Mane row still says "Guilhet" after the Guilher ruling. Overall the diffs are careful. The four problem hunks are all reasoned edits that leave no visible trace in the prose.

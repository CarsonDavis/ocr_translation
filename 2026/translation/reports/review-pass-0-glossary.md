# Review pass 0 — glossary and consistency (whole book)

Scope: the whole book read start to finish (224 rendered sections; texte-112 is still
`complete=false` and has no translation file, so it was not reviewed). This pass settled
the open glossary items and every inconsistency of names, terms, titles and
cross-references found; it did not do sentence-by-sentence fidelity (passes 1–4).

## Counts

| | |
|---|---|
| sections reviewed | 224 (all rendered sections; texte-112 pending) |
| sections edited | 26 |
| fidelity fixes | 0 (one `[unclear]` flag added at annot-097 for the print's 1555) |
| consistency fixes (individual edits) | 48 |
| glossary decisions logged (§12) | 30 |
| Ringold–Lewis divergences recorded | 6 (in the findings of texte-03, texte-14, texte-26, texte-60, texte-105, and the §2 fixes) |
| alt choices | 8 |
| misreadings recorded | 10 |

`check_markers.py --all`: 224/224 ok after the edits.

## Glossary decisions (the full list is `docs/case-file.md` §12)

1. **§9 section ids.** Every row of the running glossary carried the ids from *before*
   the re-cut that split the misprinted headings (TFXTE p045, TBXTE p058, TEXTB p072,
   ANNNT. LX p080): texte-24–35 were one behind, texte-36–48 two, texte-49 onward three,
   annot-060 onward one, and "(LX)" stood for annot-060. Remapped mechanically; spot-checked
   against the render at the four boundaries. The quarter passes can now trust the ids.
2. **Arnauld du Tilh** throughout the translated text (the print's *Arnault* of the
   Argument and p008 included); "Arnaud" only in apparatus. §3 amended.
3. **du Tilh** wherever the print has *du Thil* (p056, p064, p079); *dudit Tilh* (p097)
   → "the said [du] Tilh".
4. **Dominique Pivol, Pierre de Guilher** in both places; the p147 spellings *Puiol*,
   *Quillet* bracketed there. Our print reads *Guilher* where Ringold–Lewis and the
   literature read *Guilhet*; §2 and §3 amended.
5. **The president** of annot-090 and annot-106 is "Monsieur the President", unnamed:
   Coras names no president anywhere, and the reconciliation scene is not in Coras.
   Mansencal stays in the apparatus as [unverified]. §2 and §3 amended.
6. **Saint-Quentin 1555 / 1557.** annot-097 prints 1555; the English keeps it with an
   `[unclear]` note and the reading goes to the misreadings list; §2 and §4 record it.
7. ***Ange.*** bare on a Code or Digest *lex* (annot-094 n. l, annot-109 n. t) =
   Angelus de Ubaldis; *Ange Aretin* (annot-011 n. b) = Angelus Aretinus. §6.5 row added.
8. **"Released on appeal"** (§2) → released under the Seneschal's *appointement de
   contraires* (texte-47); not an appeal, not an acquittal. Ringold–Lewis's "reversed
   judgment" is wrong.
9. **"Guardian ad litem"** (§2, §5) → attorney (*comprocureur*) and procuration
   (*procure*), texte-48; Ringold–Lewis's "guardian" is wrong.
10. ***arrest*** → "decision" for the court's judgment; "arrest" only in Coras's second
    sense of detention (*l'arrest clos* texte-69, *arrestée* texte-78), glossed on first
    use; annot-069 is Coras's own note. §5 amended.
11. The seven **CONFLICT** rows: *belistre* → beggarly knave; *s'aduiser de* → to resolve
    to; *genitoires* → genitories; *estre mis en quatre quartiers* → be put in four
    quarters; *fourches* → gibbet-forks; *reuoquer en doute* → to call into doubt; *qui
    confisque le corps…* → he who confiscates the body confiscates the goods (HE WHO where
    the print capitalises). Each applied to the dissenting section.
12. New rules, applied: *parens* → kinsfolk everywhere; *la cour* (the Parlement) → the
    Court; the print's *S.* → St and *sainct* → Saint; Roman praenomina expanded (Mark
    Antony, Marcus Antoninus, Quintus Sertorius, Gnaeus); *lese-majesty* in one form;
    *écus* italicised; Juan Luis Vives; "Monsieur the President".
13. Settled without section edits: *réveil* (not *réveille*); *fouaces*; "two children,
    one surviving daughter" (not "two daughters"); *paillard* → lecher / rogue by
    context as the seed row allows; capitalisation of *Iuge*, *Iuges*, *Interpretes*
    follows the print (the Court excepted).

## The ten most consequential fixes

| § | French | was | now |
|---|---|---|---|
| argument, texte-03 | *Arnault du Tilh* | Arnaud du Tilh | Arnauld du Tilh |
| texte-105 | *leſdits Puiol, Quillet* | the said Pujol, Quillet | the said Pivol, Guilher [the print here spells them *Puiol*, *Quillet*] |
| annot-097 | *en l'an 1555* | in the year 1555 | in the year 1555 [unclear: so the print; the battle of Saint-Quentin was fought on 10 August 1557] |
| texte-26 | *apelãt en la cour du parlemẽt* | appellant in the court of the Parlement | appellant in the Court of the Parlement |
| annot-110 | *QVI confiſque le corps, confiſque les biens* | WHO confiscates the body confiscates the goods | HE WHO confiscates the body confiscates the goods |
| annot-110 | *leſe-maieſté* | lèse-majesté | lese-majesty |
| annot-082 | *luy manie par deſſous les genitoires* | handles his genitals from beneath | handles his genitories from beneath |
| annot-097 | *ce beliſtre du Tilh* | that rascal du Tilh | that beggarly knave du Tilh |
| texte-71 | *les commiſſaires, ſ'aduiſent de demander* | the commissioners think fit to ask | the commissioners resolve to ask |
| annot-005 | *A M. Antoine, en ſon triumuirat* | To M. Antony | To Mark Antony |

## For the next passes

- The §9 ids are now current; the per-section reports in `translation/reports/` were
  renamed by the migration and match.
- Open `[unclear]` flags that are fidelity questions, not glossary ones: annot-005 (*nul
  ne ſçait*), annot-006 (*gens / gons de raiſon*), annot-025 (*n'eſtoit pas de poids*),
  annot-050 (*ignorons / iugeons*), annot-071 (*amiable*), annot-080 (missing predicate),
  annot-081 (the Ulpian sentence), annot-083 (*ſemblẽt peu, ou point meriter*), annot-089
  (*diſant à ſes Apoſtres* breaks off), annot-111 (*priuées*), texte-63 (*perſonnément*),
  texte-105 (missing verb). The ones that could be the transcribers' are in
  `translation/review/misreadings-pass-0-glossary.json`.
- Fidelity candidates noticed in passing, not edited: annot-005 *Bruxelles, ville du
  dioceſe de Spire* is Bruchsal, not Brussels; annot-008's *Aſſidio* = "Asinius Dio" needs
  checking against Valerius Maximus 9.15; annot-104 *nul ne ſçait la differẽce* reads
  oddly in English.
- Ringold–Lewis errors to record (meaning, not wording) when the quarter passes reach
  them: texte-40 "two broken teeth in his lower jaw" (*deux ſoubredens à la machoire de
  deſſus* = two extra teeth in the upper jaw); texte-47 "reversed judgment" (*appointement
  de contraires*); texte-48 "co-guardian" (*comprocureur*); texte-14 "the account cleared
  and the remaining sum paid" (the suit was *for* them); texte-60 "far from being a
  thousand" (*encor qu'ils fuſſent mille*); texte-26 "the decision announced to the said de
  Rols" (*amplié*); texte-32 "Tonges" (our print *Touges*).
- Coras's own cross-references in the Notes ("en l'annotation lxxxj" at annot-005 n. a/i
  and annot-012 n. e, "lXXiij" at annot-026 n. b) point one short of the annotation that
  carries the matter (annot-082, annot-074): the numbering of his draft, not ours. The
  prose carries no such references; the Notes pass should annotate, not renumber.
- texte-112 (complete=false) was not rendered and has no translation; quarter 4 must pick
  it up when the transcription is complete.

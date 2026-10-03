# Report: batch texte-46 – texte-51

Twelve sections, p069–p074: the second arrest by Pierre Guerre as false attorney (texte-46, Annotation XLVIII), Bertrande's kindnesses to the prisoner (texte-47, XLIX), the Court's reasons for inclining to the prisoner — the three *favores* of marriage, children and the accused (texte-48, L) — and the record's answers to the contrary arguments: number of witnesses (the TEXTE block inside L, LI, texte-49, LII), Carbon Barrau and the soldier's hearsay (texte-50, LIII), and the tokens and the body shape (texte-51).

`check_markers.py` on each of the twelve: **10 sections ok, 2 failed** — annot-052 and annot-053. Both failures are an artifact of the `⟨alt⟩` markers: the French of those two bodies carries `{e}.⟨alt:{c}.⟩`, `{c}.⟨alt:{c2}.⟩`, `{e2}.⟨alt:{e}.⟩` and `{l}.⟨alt:{i}.⟩`, and the checker's regex collects *both* readings from the French (`['a2','b2','e','c']` for annot-052) while the English, as the prompt requires, carries the chosen reading only. The same checker run against a scratch copy of `text/sections.json` with the choices in `translation/alt-choices/batch-texte-46--texte-51.json` applied (annot-052 → `{c}`; annot-053 → `{c}`, `{e2}`, `{i}`) passes both: **2 sections ok, 0 failed**. Once the coordinator applies the choices the real checker will pass 12/12. No text in `text/sections.json` was edited.

## Per section

| Section | Pages | French | English | Notes |
|---|---|---|---|---|
| texte-46 | p069 | 72 | 84 | No marginalia. One bracketed gloss (*la procure* → "the procuration [the power of attorney]"). |
| annot-048 | p069–p070 | 224 | 241 | Printed head ANNOTAT. XVIII. — wrong number for XLVIII (transcription flags it). Markers a–g, all keyed and served. Ends with a comma in the print; kept. Page break between *a* and *eſté*. One `⟨alt?:⟩` in note *d*. |
| texte-47 | p070 | 47 | 47 | No marginalia. |
| annot-049 | p070 | 30 | 38 | No markers, no notes. |
| texte-48 | p070–p071 | 43 | 45 | Head printed TEXTE, (comma). Carries an orphan note keyed *m* that belongs to Annotation L's {m}; listed. Page break between *ſont* and *iſſuz*. |
| annot-050 | p071–p072 | 432 | 472 | Markers a–n (no *j*); notes a–l; {m} served by the orphan on p071 under texte-48; {n} has no note. Contains the printer's next TEXTE block, headed TEXTB. (wrong sort), translated in place as TEXT. One `[unclear]`. Two `⟨alt⟩` in note *c*. Page break inside *inno|cent*. |
| annot-051 | p072 | 46 | 49 | Markers a–b, served. One `⟨alt⟩` in note *a*. |
| texte-49 | p072 | 17 | 19 | No marginalia; continues the TEXT sentence inside annot-050. |
| annot-052 | p072–p073 | 76 | 82 | Markers a2, b2, then the `⟨alt⟩` {e}/{c}, read {c}. The orphan note *c* on p072 serves it; the p073 note *c* attached to this annotation belongs to annot-053. Page break between *raiſon* and *doncques*. Checker: fails on the alt artifact only. |
| texte-50 | p073 | 63 | 67 | No marginalia. Print's *Corbon Barrau* → Carbon Barrau per the case file. |
| annot-053 | p073 | 182 | 201 | Markers a, b, {c}/{c2}, d, {e2}/{e}, f, g, h, {l}/{i}; notes a, b, d, e2, f, g, h, i. One `⟨alt⟩` in note *b*. One bracketed gloss (*adminiculés* → "adminicles [supporting proofs]"). Checker: fails on the alt artifacts only. |
| texte-51 | p073–p074 | 93 | 104 | No marginalia. Page break between *que* and *les teſmoins*. |

Totals: French 1325 words, English 1449 words (counts exclude the page, letter and alt markers).

## `⟨alt⟩` choices

The same nine choices, machine-readably, are in `translation/alt-choices/batch-texte-46--texte-51.json`.

- **p069-m3l2-1**, annot-048, note *d* (unconfirmed, `⟨alt?:⟩`): *l. iij. Ꝑ.* / *l. iij. P.* — took **B**, *P.* It is the paragraph sign (§ *Sed et si quidem*), which this print sets as a swash italic *P* and which conventions §2 rules is transcribed *P*, as note *e* of the same annotation does (*l. iij. P. falſus*). The transcribers vouch for neither reading; the translation is the same either way. Flagged as unconfirmed.
- **p071-m2l2-1**, annot-050, note *c*: *not. qui ſi.* / *not. qui fi.* — took **B**, *fi.*: *qui fi. ſint leg.* is the Decretals title *Qui filii sint legitimi* (X 4.17), cited in full in note *b* (*qui fil. ſint leg.*); *qui ſi* is no title.
- **p071-m2l5-1**, annot-050, note *c*: *inhibitio. P. ſi* / *inhibitio. P. fi* — took **B**, *fi*: *P. final.* is § *final*, the last paragraph of c. *Cum inhibitio*; *ſi nal.* is nothing.
- **p072-m0l2-1**, annot-051, note *a*: *c & in noſtra* / *c. & in noſtra* — **either**: a point after the abbreviation *c.* (*capitulum*), present or absent; both give c. *In nostra*, *de testibus*, and only the page image can decide. The French keeps reading A.
- **p073-b0l1-1**, annot-052, body: *enſemble {e}.* / *enſemble {c}.* — took **B**, {c}. The alphabet runs a2, b2 and then c; the margin of p072 carries an orphan note keyed *c* (*Aud. itaque. C. con. de ſuc.*) with no marker to serve, which is this one; the note keyed *c* on p073 that the transcription attaches to this annotation is by content (hearsay: *c. tam literis, c. licet ex quadam, de testibus; c. tua, de consang.*) the note for annot-053's {c}; and the print's *c* and *e* are near twins. **Consequence for the French:** the p073 note *c* should be moved from annot-052 to annot-053.
- **p073-b4l5-1**, annot-053, body: *preuue {c}.* / *preuue {c2}.* — took **A**, {c}. The print has a plain *c*; the note that serves it is the p073 note keyed *c* (see the previous item); with that note placed here there is no second *c* on the page for the *2* to distinguish.
- **p073-b4l9-1**, annot-053, body: *circonſtances {e2}.* / *circonſtances {e}.* — took **A**, {e2}, to match the key of the margin note that serves it (*c. præterea. de teſt. Panorme au c. licet ex quadam.*); both readings translate the same. The suffix exists only because annot-052's third marker on this page was read as *e*; once that is corrected to *c*, both this body marker and the note's key should drop the suffix and read *e*. Left to the coordinator, since it changes a note key as well as a body marker.
- **p073-b4l19-1**, annot-053, body: *Lucrece {l}.* / *Lucrece {i}.* — took **B**, {i}: the alphabet runs h then i, the margin note is keyed *i* (*Les Docteurs en la l. ſi arbitrer & au c. tam literis*), *l* would skip *i* and *k*, and the italic *i* and *l* are near twins.
- **p073-m1l3-1**, annot-053, note *b*: *pre allegue.* / *pre allegué.* — **either**: an accent present or absent on *preallegué* ("cited before"); the margin has *preallegué* (note *d*) and *allegue* (note *e2*) both, and the readings translate the same. The French keeps reading A.

## `[unclear]` passages

- annot-050: "nevertheless, if we go by the truth of the matter [unclear: the print's *ſi nous ignorons*, perhaps for *iugeons*], they are not so" — *toutesfois ſi nous ignorons par la verité de la choſe, ils ne le ſont point*. As printed the clause says "if we are ignorant by the truth of the thing", which is no sense; the argument requires "if we judge by the truth of the matter" (equity deems the children legitimate, the truth does not), and *ignorons* is probably a slip for *iugeons* (or *regardons*). Rendered by the required sense with the French flagged.

Two bracketed glosses and one supplied object were added for sense, not for obscurity:

- texte-46: "the procuration [the power of attorney]" — *la procure*, first use.
- annot-053: "adminicles [supporting proofs]" — *adminiculés*, the civilians' *adminicula*, first use.
- annot-050: "the law wills and commands that judgment be given [for it]" — *la loy veult & cõmande faire Iugement*; the object (for the marriage) is understood from *Pour le mariage* at the head of the sentence.

## Citations not identified, or identified only in part

- annot-048 {a}: *l. ſi procuratori falſo … D. de cõd. cau. dat.* — D. 12.4, fragment not identified; *l. licet. C. de procu.* — C. 2.12, law not identified.
- annot-048 {b}: *c. ex parte Decani. de re.* — title abbreviation unresolved (perhaps *De rescriptis*, X 1.3, the title of note *g*); chapter not identified.
- annot-048 {c}: *l. falſus … C. de fur.* (print *ſulſus*) — C. 6.2, law not identified; the same law is cited with Baldus at {g}.
- annot-048 {d}: *l. quæro. D. de eo qui pro tutor.* — D. 27.5, fragment not identified; *l. iij. § ſed & ſi quidem. D. iud. ſol.* — D. 46.7.3, paragraph unverified.
- annot-048 {e}: *l. licet. D. de iud.* — D. 5.1, fragment not identified; *l. iij. § falſus. D. rem ra. hab.* — cf. D. 46.8.3, paragraph unverified.
- annot-048 {f}: *l. finale. C. de iis qui à non do. man.* — C. 7.10, last law (cf. C. 7.10.7), unverified.
- annot-048 {g}: *Panorme au c. nonnulli. § ſunt & alij. de reſcrip.* — cf. X 1.3.28, number unverified.
- annot-050 {a} and annot-052 {b2}: *c. final. de re iud.* — the last chapter of X 2.27, not identified.
- annot-050 {b}: *Panorme au c. tranſmiſſa. qui fil. ſint leg.* — cf. X 4.17.3, unverified.
- annot-050 {c}: *c. ij. c. referente, c. ex l. not. qui fi. ſint leg.* — X 4.17; *c. ex l. not.* unresolved (perhaps *Ex litteris nostris* or *Ex tenore*); *Gloſe au c. cum inhibitio. § final. de clandeſt. deſpon.* — X 4.3.3, identified with confidence.
- annot-050 {d}: *l. iij. C. ſoma t.* — **unidentified**; the Code title is corrupt (perhaps *Soluto matrimonio*, C. 5.18, or *De naturalibus liberis*, C. 5.27). *c. ſi gens. lvj. diſtinct.* — Decretum D. 56 c. *Si gens* (cf. c. 10, *Si gens Anglorum*), number unverified.
- annot-050 {e}: *l. Arrianus. D. de actio. & oblig.* — cf. D. 44.7.47, unverified (also annot-052 {b2}); *l. fauorabiliores. D. de reg. iur.* — D. 50.17.125, confident; *c. ex literis, de prob.* — X 2.19, chapter not identified; *c. inter. de fid. inſtr.* — X 2.22, perhaps c. *Inter dilectos*, not identified; *c. cum ſint de re. iur. au vj.* — VI, *de reg. iur.*, reg. 11, confident.
- annot-050 {f}: *l. inter pares. D. de re. iud.* — D. 42.1.38, confident.
- annot-050 {g}: *Gloſe c. clerici. lxxxi. di.* — Decretum D. 81, chapter not identified.
- annot-050 {h}: Aristotle, *Problems* 29.13 — confident; the chapter gives exactly Coras's three reasons.
- annot-050 {i}: *l. pure. § ſi nat. D. ſol. except.* — **unidentified** (title perhaps *De doli mali et metus exceptione*, D. 44.4); *l. j. C. vt nem. inui. ag. vel accuſ. cog.* — C. 3.7.1, confident.
- annot-050 {k}: *c. ad audientiam. de homicid.* — cf. X 5.12.12, number unverified; the source of *in dubiis via tutior*.
- annot-050 {l}: *l. vbi enim D. de reb. du.* — D. 34.5, fragment not identified; *Gloſe au c. ij. de re. iu. aut. au Decret.* — unresolved (*de re iudicata* or *de regulis iuris*; *aut.* read as "or"); may serve {n}.
- annot-050 {n}: no note in the margin.
- annot-051 {a}: *c. licet cauſam. de prob.* — cf. X 2.19.9, number unverified; *c. in noſtra de teſtib.* — X 2.20, chapter not identified.
- annot-051 {b}: *l. ob carmen. § fina. D. de teſtib.* — D. 22.5.21.3, confident (print *carnem*).
- annot-052 {a2}: *Accurſe en la l. diem. § ſi plures. D. de recep. arb.* — D. 4.8, l. *Diem proferre*, fragment and paragraph unverified.
- annot-052 {c} (p072, orphan): *Aud. itaque. C. con. de ſuc.* — **unidentified**; read as an *authentica* *Itaque* under perhaps *Communia de successionibus* (C. 6.59); bearing unverified.
- annot-052/annot-053 {c} (p073): *c. tam literis. c. licet ex quadam. de teſtib. c. tua. de conſang. & affinit.* — *Tam litteris* chapter not identified (annot-033 printed it *iam literis*); *Licet ex quadam* cf. X 2.20.47; *c. tua* X 4.14, chapter not identified (annot-033 had *tutela*).
- annot-053 {a}: *l. teſtium C. de teſtib.* — C. 4.20, law not identified; *c. hoc videtur. xxij. q. v.* — Decretum C. 22 q. 5, chapter number unverified.
- annot-053 {b}: *Archiadia. de auc. hoc videtur* — read as the Archdeacon (Guido de Baysio) on c. *Hoc videtur*; the print's *de auc.* taken as *au c.*
- annot-053 {e2}: *c. præterea. de teſt.* — X 2.20, chapter not identified.
- annot-053 {f}: *l. ſi arbiter. D. de probatio.* — cf. D. 22.3.28, unverified.
- annot-053 {g}: *Balde en la l. conuenticulam. C. de epiſc. & cler.* — C. 1.3, law not identified.
- annot-053 {h}: *Accurſe en la l. ij. § Idem Labeo. D. de aq. pluu.* — D. 39.3.2, paragraph unverified.

## Marker and margin problems

- **annot-048**: the printed head is ANNOTAT. XVIII., a wrong number for XLVIII (the transcription's `number_uncertain` / `number_printed: 18`). The section ends with a comma after {g} in the print.
- **texte-48 / annot-050**: the note *m* (*l. abſente D. de pœn.*) is printed in the margin of p071 beside texte-48's last line and listed by the transcription as an orphan under texte-48; its marker {m} is the first word on p072 in Annotation L. Listed under both sections with cross-references. Annotation L's {n} has no note at all; note *l*'s second citation may be meant for it.
- **annot-050**: the segmenter left the printer's next TEXTE block (headed TEXTB., wrong sort) inside the annotation; translated in place as TEXT, as annot-036 did with its TBXTE. The record's sentence then runs on through annot-051 into texte-49.
- **annot-052 / annot-053**: the p072 orphan note *c* belongs to annot-052's third marker (read {c}); the p073 note *c* that the transcription attaches to annot-052 belongs to annot-053's {c}. The notes have been translated under both sections with cross-references so that either arrangement is covered; the coordinator should move the p073 note to annot-053 when applying the alt choices. See the `⟨alt⟩` section on the *e2* key.
- **annot-052, annot-053**: `check_markers.py` fails on these two only because it collects both readings of each body `⟨alt⟩` from the French; verified passing against a resolved scratch copy.

## Glossary additions

Per the coordinator's instruction these rows have **not** been appended to `docs/case-file.md` §9; they are given here in the table's format for the coordinator to merge.

| French | English rendering | Note |
|---|---|---|
| comprocureur | joint attorney | *Procureur* of a private party → attorney; acting jointly with or for the civil party. (texte-46) |
| conſtituer priſonnier (le fit) | to have (him) made prisoner | Cf. *conſtitué priſonnier* → made prisoner. (texte-46) |
| procure / procuration | procuration [the power of attorney] | Glossed in brackets on first use. (texte-46, annot-048) |
| faux procureur | false attorney | The *falsus procurator*. (annot-048) |
| charge (d'un procureur) | charge | The mandate. (annot-048) |
| outrepaſſer les fins & bornes de ſa puiſſance | to overstep the ends and bounds of one's power | (annot-048) |
| ratifier / ratification | to ratify / ratification | The *ratihabitio*. (annot-048) |
| obreption | obreption | Obtaining a grant by a false statement; the law of rescripts. (annot-048) |
| cõſiſtoire du prince / auditoire du prince | consistory of the prince / audience-chamber of the prince | (annot-048) |
| offices (vſer d'offices enuers) | good offices | (texte-47) |
| accouſtremens | clothing | (texte-47) |
| reprins | retaken | Of a prisoner rearrested. (texte-47) |
| faire partie à | to be the party against | As *faire la partie à*. (annot-049) |
| ſecourir officieuſement | to succour obligingly | (annot-049) |
| ſentence (of the Court's leaning) | opinion | As *l'aduis & la ſentence*; not the lower judge's judgment. (texte-48) |
| faueur du mariage / des enfans / du preuenu | favour of the marriage / of the children / of the accused | The *favor matrimonii*, *favor prolis*, *favor rei*. (texte-48, annot-050, annot-052) |
| faire Iugement (en faueur de) | to give judgment (for) | (annot-050) |
| vaincre & ſurmonter | to conquer and overcome | Of a presumption. (annot-050) |
| nez de paillardiſe & procreez d'adultere | born of lechery and begotten of adultery | (annot-050) |
| faits cõtrouerſes | controverted facts | (annot-050) |
| procliue à | inclined to | Latin *proclivis*. (annot-050) |
| deliurer & abſoudre | to deliver and absolve | (annot-050) |
| crimes publiques, & capitaux | public and capital crimes | (annot-050) |
| accuſateur | accuser | Paired with *demandeur* → plaintiff. (annot-050) |
| agir, ou accuſer | to sue or to accuse | (annot-050) |
| le chemin, ou le ſentier plus aſſeuré | the surer road or path | The *via tutior*. (annot-050) |
| affaires douteux, & perplex | doubtful and perplexed affairs | (annot-050) |
| laiſſer impuny le coulpable | to leave the guilty unpunished | D. 48.19.5. (annot-050) |
| opinion plus douce, plus humaine | the gentler, more humane opinion | (annot-050) |
| donner plus de foy à | to give more faith to | Cf. *faire foy*. (annot-050, annot-051) |
| numeroſité & multitude | number and multitude | Cf. *numerosité* → multitude. (annot-051) |
| faire tomber la balance | to make the balance fall | (annot-052) |
| plus forte raiſon doncques | with stronger reason, then | *A fortiori*. (annot-052) |
| particulariſer de ſi pres | to particularise so closely | (texte-50) |
| viuement, & vallablement reprochez | sharply and validly objected to | Per §5. (texte-50) |
| obiects (de reproches) | objections | The grounds of objection. (texte-50) |
| le dire du ſoldat | the saying of the soldier | (texte-50) |
| n'y faire rien / n'y eſtre rien | to count for nothing | (texte-50, texte-51) |
| ſens corporels | bodily senses | (annot-053) |
| telle quelle preſumption | some sort of presumption | (annot-053) |
| adminiculés | adminicles [supporting proofs] | Glossed in brackets on first use. (annot-053) |
| faire apparoir de | to make appear | (annot-053) |
| limites, & bornes | limits and bounds | Of lands. (annot-053) |
| partie plaidante | party pleading | (annot-053) |
| fiancer par paroles de preſent | to espouse by words of the present | *Sponsalia per verba de praesenti*. (annot-053) |
| Corbon Barrau | Carbon Barrau | The print's spelling at p073; case file form kept. (texte-50) |
| longueur, & groſſeur | height and bulk | The body-shape evidence. (texte-51) |
| reſſembler (plus haut) | to appear (taller) | *Ressembler* in its old sense "to seem". (texte-51) |
| ſe remplir de corps, & renforcer de iambes | to fill out in body and grow stronger in the legs | (texte-51) |

## For the reviewer

1. **annot-052 and annot-053 fail `check_markers.py`** solely because the French bodies still carry both alt readings. Apply the alt choices (and move the p073 note *c* from annot-052 to annot-053), then rerun; verified passing against a resolved copy.
2. **The *e2* key (annot-053)**: body marker and note key should both become *e* once annot-052's marker is *c*. Left for the coordinator because it changes a note key, not only a body marker.
3. **annot-050, *ſi nous ignorons***: the one `[unclear]`; check the page image for *iugeons*. If the image reads *ignorons*, Coras's slip stands and the bracket should stay.
4. **annot-048's printed head XVIII**: confirm the heading is rendered ANNOTATION XLVIII in the site data.
5. **annot-048 ends with a comma**: the sentence breaks off before the next TEXTE. in the print; check the image for a lost line.
6. **texte-50, *Corbon Barrau***: the case file's *Carbon* was kept; if the house prefers the copy-text spelling in the text, change to "Corbon Barrau" here and note the variant (the earlier English has "Carbon Barrau" at texte-28, the uncle's first appearance).
7. **texte-48, *ceſte ſentence* → "this opinion"**: the court's leaning, not a judgment; the reviewer may prefer "this view".
8. **annot-053, *adminiculés***: "adminicles" is the English law-French term; "corroborating proofs" would be the plain alternative.
9. **Latin maxims identified with confidence** and worth a footnote in the apparatus: D. 48.19.5 (*satius … impunitum relinqui*), D. 50.17.125 (*favorabiliores rei*), VI *de reg. iur.* 11 (*cum sunt partium iura obscura*), D. 42.1.38 (*inter pares*), D. 22.5.21.3 (*non ad multitudinem respici*), Aristotle *Problems* 29.13, X 4.3.3 § fin. (*Cum inhibitio*).

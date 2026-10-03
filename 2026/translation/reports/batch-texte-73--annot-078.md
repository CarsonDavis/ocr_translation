# Report: batch texte-70 – annot-077

Twelve sections, p097–p106: the recognitions of the real Martin Guerre by the eldest sister, the other sisters, the witnesses and Bertrande de Rols (texte-70 to texte-75), with Coras's Annotations LXXII–LXXVII (printed LXXIII–LXXVIII) on tears of joy, proof by witnesses in person, voluntary and involuntary crimes, the credulity of women, suicide, and the good faith of the deceived wife. `check_markers.py` on each of the twelve: 11 sections ok, 1 failed — annot-074, for a checker artifact only (see "Marker and margin problems"); with the `⟨alt⟩` marker stripped from the French, annot-074 passes too.

## Per section

| Section | Pages | French | English | Notes |
|---|---|---|---|---|
| texte-70 | p097 | 26 | 25 | No marginalia. One sentence. |
| annot-072 | p097–p098 | 234 | 243 | Markers a–d, all served. Page break inside *fem|mes*. Body ends without a point in the print; none supplied. |
| texte-71 | p098–p099 | 77 | 84 | No marginalia. Page break inside *aupar|auant*. The print opens a parenthesis and never closes it; one closing parenthesis supplied. |
| annot-073 | p099–p100 | 334 | 373 | Markers a–d, all served. Five `⟨alt⟩` (see below). |
| texte-72 | p100 | 59 | 70 | No marginalia. |
| annot-074 | p100–p101 | 499 | 521 | Twenty-one markers a–x (no *j*, *v*, *w*; *ſ* as a key; the fourth keyed *d2*). Six `⟨alt⟩`. Fails `check_markers.py` only because the French `{k}.,⟨alt:{k},⟩` holds the marker twice. |
| texte-73 | p101–p102 | 25 | 26 | No marginalia. Page break inside *faci|lement*. Ends on a comma, continued by texte-74. |
| annot-075 | p102 | 59 | 70 | Markers a–c, all served. Latin couplet translated in the text, Latin in the Notes. |
| texte-74 | p102 | 54 | 66 | No marginalia. Completes texte-73's sentence. |
| annot-076 | p102–p105 | 887 | 1054 | Twelve markers a2–l (two keyed *a2*, *c2*). Seven `⟨alt⟩`. Six verse passages. Page breaks inside *com|me* and at *Horace {l}, ⟦p105⟧Dicam*. |
| texte-75 | p105–p106 | 89 | 110 | No marginalia. Page break inside *ar|reſtée*, kept at "ar⟦p106⟧rest". Ends on a comma, continued by texte-76. One bracketed gloss. |
| annot-077 | p106 | 131 | 154 | One marker, served. |

Totals: French 2474 words, English 2796 words (counts exclude the page, letter and `⟨alt⟩` markers).

## ⟨alt⟩ choices

The same choices, machine-readably, are in `translation/alt-choices/batch-texte-70--annot-077.json` (23 entries, in the prompt's table order). None is `⟨alt?:⟩`; none is unconfirmed.

- annot-073, p099 (`p099-b2l25-1`): *que ſe* / *que ſe-* (before *lon*) — took **B**. The word *selon* is divided at the line end; a hyphen is the expected sign of division. Sense unaffected.
- annot-073, p099 (`p099-b2l26-1`): *Calliſtrat* / *Callittrat* — took **A**. Callistratus, the jurist whose law (D. 22.5.3) Coras is quoting; *Callittrat* is no name.
- annot-073, p099 (`p099-b2l27-1`): *Iuriſconſulte,* / *Iuriſconſulte.* — took **A**. The sentence runs on (*pourſuyuant l'argumẽt d'Adrian*); a comma, not a full stop.
- annot-073, p099 (`p099-b2l35-1`): *à mon aduis,* / *à mon aduis;* — took **A**. *C'eſt à mon aduis, ce que noſtre Iuſtinien a laiſſé eſcrit* is one clause; a semicolon would cut the relative from its antecedent. Translation the same either way.
- annot-073, margin p099 (`p099-m0l0-1`): *Soit veuë* / *Soit veüe* — took **A**, by the parallel of the same formula printed *veuë* on p098 (annot-072 {b}). Both give *veue*.
- annot-073, margin p099 (`p099-m2l1-1`): *Diuus.* / *Diuus:* — **either**. Punctuation at the end of a note; same citation.
- annot-074, p100 (`p100-b4l9-1`): *commestẽt* / *commeſtẽt* — took **B**. Both are the same wrong sort for *commettent*; medial *s* is long *ſ* in this print. Translated by sense ("are committed") either way.
- annot-074, margin p100 (`p100-m2l0-1`): *l. j & ij.* / *l j & ij.* — **either**. The siglum with or without its point.
- annot-074, p101 (`p101-b0l4-1`): *paſſion {k}.,* / *paſſion {k},* — took **B**. A point followed by a comma is an unlikely setting, and the clause continues (*ſe pardonne aiſement*). Same translation. See the checker note below.
- annot-074, p101 (`p101-b0l11-1`): *ſ'excuſent* / *l'excuſent* — took **A**. Reflexive: the involuntary crimes are those which "are excused" as fortuitous and by error or ignorance; *l'excuſent* would want an object the sentence does not give.
- annot-074, p101 (`p101-b0l17-1`): *beſte* / *belte* — took **A**. *Beste*, "beast" (*cheureul, ſanglier, ou autre beſte ſauuage*); *belte* is a misread of *ſ* as *l*.
- annot-074, p101 (`p101-b0l32-1`): *par exemple* / *par exemple,* — **either**. Punctuation only.
- annot-074, margin p101 (`p101-m3l1-1`): *des pa-* / *des Pa-* (*papes*) — **either**. Case of the initial only.
- annot-074, margin p101 (`p101-m9l3-1`): *allegué.* / *allegue.* — **either**. The accent only.
- annot-074, margin p101 (`p101-m10l4-1`): *ad l. Aqui.* / *ad l Aqui.* — **either**. The siglum's point only.
- annot-076, p102 (`p102-b9l0-1`): *finemq́;* / *finemq;* — **either**. The *-que* abbreviation with or without the acute; *finemque* either way.
- annot-075, margin p102 (`p102-m2l3-1`): *ſous la collation* / *ſous ta collation* — took **A**. "Under the seventh collation" of the Authenticum; *ta* is *l* misread as *t*.
- annot-076, margin p103 (`p103-m1l0-1`): *Philippenſ.* / *Philippexſ.* — took **A**. Philippians; *Philippexſ.* is no word.
- annot-076, margin p104 (`p104-m0l4-1`): *de falſæ ſa.* / *de falſæ ſa-* (before *pientia*) — took **B**. *Sapientia* is divided across two margin lines; a hyphen, not a point. Same citation.
- annot-076, margin p104 (`p104-m2l2-1`): *ſapieria.* / *ſapietia.* — **either**. Both are wrong sorts for *sapientia*; nothing favours one misprint over the other. The Notes identify the work regardless.
- annot-076, p105 (`p105-b0l0-1`): *Sicculíque* / *Sicculique* — **either**. The accent before the enclitic only; Horace's *Siculique*.
- annot-076, p105 (`p105-b1l8-1`): *en gaide* / *en garde* — took **B**. *Nous auons en garde l'ame* — "we have the soul in keeping", echoed by *prins à garder* six words on and *baillee en garde* at the close; *gaide* is no word.
- annot-076, p105 (`p105-b1l14-1`): *balliee* / *ballice* — took **A**. *Baillée*, "delivered" (*commiſe, & baillee en garde*); *ballice* is a misread of *e* as *c*.

## `[unclear]` passages

None. Bracketed glosses added for sense, not for obscurity:

- texte-75: "under arrest [*arreſtée*: confined by the court's order]" — *demeuroit par l'appel encor arreſtée*; the second sense of *arrest* that Coras himself gives at annot-068 (a magistrate's order confining a person to a place). Without the gloss an English reader takes Bertrande for a prisoner.
- annot-076: "to come back to our sheep [*reuenans à nos moutons*: to our subject]" — the proverb, kept literal as the house kept *revenir à nos brisées* (annot-005), glossed on first use.

Two places where the French is defective rather than obscure, translated by sense and recorded in the section Notes: annot-072 *ſouffrez* for *ſouffre* ("nature… suffers"); annot-076 *Et que ſainct Paul, bruſlant…* has no finite verb (anacoluthon) — "burned" supplied.

## Citations not identified, or identified only in part

- annot-072 {c}: *Pline au li. vij. c. iiij.* — the deaths from sudden joy are Pliny 7.53 (old chapter 53); the print's *iiij* has probably lost an *l*. Given as printed with the remark. Gellius 3.15 identified with confidence.
- annot-073 {d}: *§ hæc omnia, aux nouuelles… ſous la vij. collatiõ* — the Authenticum's seventh collation, title *De testibus* (Nov. 90), at the paragraph *Haec omnia*: chapter unverified. *Aut. apud eloquentiſſimum C. de fid. inſtr.* — the Authentica *Apud eloquentissimum* in C. 4.21; its wording not checked.
- annot-074 {a}: Aristotle, *Ethics* book I as printed; the voluntary/involuntary division is NE III.1. Given as printed with the remark.
- annot-074 {d2}: *l. j. C. de homicid. au vj.* — read as Liber Sextus V.4 c. 1 (*au vj.* = *in VI*), despite the civil-law sigla *l.* and *C.*; the chapter's content unverified. Worth a reviewer's eye.
- annot-074 {e}: *l. penultieme D. de adul.* — D. 48.5, fragment not identified.
- annot-074 {f}: *l iij. c. de epiſ.* — C. 1.4.3 (*De episcopali audientia*), bearing unverified; *c. i. de homicid. aux Decretales* — X 5.12.1, content unverified.
- annot-074 {g}: *c. cum non ab hom. de Iud.* — given as cf. X 2.1.10.
- annot-074 {i}: *c. final. xxxv. diſt.* — the last chapter of Gratian D. 35 does not obviously bear on self-defence; the number may be a wrong sort. Given as printed, unverified.
- annot-074 {k}: *l. ſi mulier § penult. D. quod met. ca.* — cf. D. 4.2.21; paragraph unverified.
- annot-074 {l}: *l. verum. D. de fur.* — D. 47.2, fragment not identified.
- annot-074 {p}: Ovid, *Fasti* III as cited: 3.92 names Telegonus (as founder of Tusculum), but the parricide is not told there; and the detail that Telegonus took his father for a servant is not in the standard sources (Hyginus 127, *Ibis* 567–568). Given as cited with the remark.
- annot-074 {q}, {t}, {u}: the Decretals chapters *Lator* and *Continebatur* (*c. contine latur* in the transcription) under X 5.12 *De homicidio* — chapter numbers unverified.
- annot-074 {r} and annot-077 {a}: *c. j xxxi q. j* — C. 31 q. 1 c. 1, unverified; *c. ſi virgo* and *c. in lectũ* at "xxxiij. q. ij." — the two chapters on the woman deceived into intercourse by error of person stand in the modern Decretum at C. 34 qq. 1–2, not C. 33 q. 2; chapter numbers unverified.
- annot-074 {ſ}: *c. inebriauerũt xv q j* — C. 15 q. 1, chapter number unverified.
- annot-074 {u}: *l. lege. § j de ſiccar.* — D. 48.8.1 § 1 (*Lege Cornelia*), confident; *l. q ſeruus. D. ſi fornicarius. D. ad l. Aqui.* — **unidentified**: no fragment of D. 9.2 with a paragraph *Si fornicarius* was found; *l. q ſeruus* is probably *Qui servus* or *Quod servus*.
- annot-074 {x}: *l. cũ qui. § ſi iniuria. D. de iniuriis* — D. 47.10, fragment not identified (cf. D. 47.10.4, the blow meant for a slave that strikes a bystander). D. 9.2.45.4 (*Scientiam*, § fin.) identified with confidence.
- annot-075 {b}: *Fauſtus au iiij. de Liuie* — identified as Fausto Andrelini's *Livia* (*Amores*), book IV, an elegiac collection in four books; the couplet's place in it unverified.
- annot-075 {c}: *§ quæſitũ. de æqua dor. aux nouuelles ſous la collation vij.* — Nov. 97 (*De aequalitate dotis et propter nuptias donationis*) in the seventh collation; the paragraph *Quaesitum* unverified.
- annot-076, opening verse: *Illa malis requiem finemque laboribus affert* — **unidentified**.
- annot-076 {a2}: Cicero, *Ad familiares* V–VI, the consolatory letters — the specific letters are a suggestion, not a verification.
- annot-076 {b}: "Numantius" writing to Cicero in *Ad familiares* XI — no such correspondent; Matius (Fam. 11.28) suggested, unverified.
- annot-076 {e}: Aristotle, *Ethics* "book IV chapter 7" as printed; the passage is NE III.7. Given as printed with the remark.
- annot-076, Callimachus epigram: the Latin translator ("quelque docte homme") not identified.
- annot-076 {l}: the print's *Dicam* at the head of Horace's lines is not in Horace; translated as part of the quotation, with the remark.

## Marker and margin problems

- **annot-074 and `check_markers.py`.** The French body has *paſſion {k}.,⟨alt:{k},⟩*: the `⟨alt⟩` marker's reading B repeats the `{k}`, so the checker's regex finds *k* twice in the French (`…'i', 'k', 'k', 'l'…`) and rejects the English, which carries `{k}` once, as it should. The English was not altered to pass. Verified by running `check_markers.check` against the French with `⟨alt:…⟩` stripped: no problems. Once the alt-choice (B) is applied to the transcription, the check passes as the file stands. The coordinator may want `check_markers.py` to strip `⟨alt…⟩` before matching.
- annot-074: the fourth note is printed *d* and keyed *d2* by the transcription (p100 already carries annot-073's *d*); the body marker is {d2} to match. The key *ſ* (long s) is used in its place between *r* and *t*, and kept.
- annot-076: the first note is keyed *a2* (p102 already carries annot-075's *a*). The fourth note is printed *c*, repeating the previous key — a wrong sort where the run wants *d*; the transcription keys it *c2* and the following notes resume at *d*. Body markers follow the transcription.
- annot-072: the body ends *ſur l'heure* with no point; none supplied, per house practice.
- texte-71: the parenthesis *(monſtrant ledit du Tilh, illec preſent* is never closed in the print; closed after "there present".
- texte-73 and texte-75 end on commas (their sentences continue into texte-74 and texte-76); kept.
- texte-75: *mis & quatre quartiers* — *&* a wrong sort for *en*.
- The printed annotation numbers run one ahead of the transcription's count throughout the batch (ANNOT. LXXIII. = annot-072, … ANNOTAT. LXXVIII. = annot-077), flagged `number_uncertain` in the transcription. Coras's own cross-reference at annot-026 {b} ("En l'annotation lXXiij", on proof by witnesses) points to the annotation on witnesses, which is the 73rd by count (annot-073) but is printed LXXIIII — so Coras's count agrees with the transcription's numbering, not with the printed heads. Each section's Notes record both numbers.

## Glossary additions

Not appended to `docs/case-file.md` §9 (the coordinator's instruction for this run); given here in the table's format for merging.

| French | English rendering | Note |
|---|---|---|
| nouueau venu | newcomer | The record's name for the real Martin Guerre from texte-66 on. (texte-70) |
| recognoiſtre pour (ſon frere) | to recognise for | (texte-70) |
| hiſtoriographes | historiographers | (annot-072) |
| ietter larmes en abondance | to shed tears in abundance | Cf. *larmoyer / pleurer*. (annot-072) |
| dolens & marriz | grieving and sorrowful | (annot-072) |
| rendre l'ame | to give up the soul | (annot-072) |
| ſur la place / ſur l'heure | on the spot / on the hour | (annot-072) |
| proditeur | traitor | Cf. *proditoirement* → treacherously; of du Tilh. (texte-71, texte-75) |
| fauſſes enſeignes | false tokens | Cf. *veritables enseignes* → true tokens. (texte-71) |
| conſtituer & entretenir (en erreur) | to place and keep (in error) | Cf. *endormie & entretenue*. (texte-71) |
| pour faire brief | to be brief | (texte-71) |
| ſouſtenir (le priſonnier eſtre) | to maintain (that the prisoner was) | Of a witness's assertion. (texte-71) |
| aduiſer (jussive, of judges) | to take heed | Cf. *contempler* → contemplate, annot-026. (annot-073) |
| ſe departir (de ſa depoſition) | to depart (from) | Of a witness retracting. (annot-073) |
| geſtes & contenances | gestures and countenances | (annot-073) |
| rendre raiſon du tout | to render a reason for the whole | Cf. *rendre raison de son dire*. (annot-073) |
| retrencher le chemin à | to cut off the way to | (annot-073) |
| procliue (à) | prone (to) | (annot-073) |
| court ſouueraine / iugement ſouuerain | sovereign court / sovereign judgment | (annot-073) |
| teſmoignage / depoſition (as against the witness) | testimony / deposition | Hadrian's *testibus, non testimoniis*. (annot-073) |
| mauuais garçon de commiſſaire | bad fellow of a commissioner | (annot-073) |
| brouillaçon de greffier | scribbler of a clerk | *Greffier* → clerk of court, §5. (annot-073) |
| repreſentez, & offerts (teſmoins) | presented and offered | Of witnesses produced in person. (annot-073) |
| toute eſploree | all in tears | (texte-72) |
| tremblante comme la fueille agitee des vents | trembling like the leaf shaken by the winds | (texte-72) |
| faute | fault | Bertrande's *faute*; the word of texte-72 that annot-074 examines. (texte-72, annot-074) |
| coulpe | fault | *Culpa*; the same English as *faute*. (annot-074) |
| volontaire / inuolontaire; volontairement / non volontairement | voluntary / involuntary; voluntarily / not voluntarily | (annot-074) |
| à propos deliberé | with a deliberate purpose | Cf. *propos deliberé & intention de malfaire*. (annot-074) |
| de guet à pens | by lying in wait | Cf. *guetter & assaillir*. (annot-074) |
| violer (vne femme) | to rape | Cf. *rapt*. (annot-074) |
| dol | fraud | The Roman *dolus*. (annot-074) |
| imperdonnable & irremiſſible | unpardonable and irremissible | (annot-074) |
| peine corporelle | corporal penalty | (annot-074) |
| ſoudaine paſſion | sudden passion | The Digest's *impetus*. (annot-074) |
| pourpenſee ni deliberee (volonté) | premeditated nor deliberate | Cf. *malice pourpensee*. (annot-074) |
| iuſtement irrité | justly provoked | (annot-074) |
| ſe retenir, & dompter ſoy-meſmes | to restrain and master oneself | (annot-074) |
| maluerſer auec (ſa femme) | to misconduct oneself with | Of a man; cf. *mal-verser* of a wife. (annot-074) |
| fortuitement / caſuellement | as fortuitous / by chance | The Digest's *casu*. (annot-074) |
| deſaſtre d'erreur ou d'ignorance | disaster of error or ignorance | (annot-074) |
| eſtre cogneue de (vn autre) / conuerſer auec | to be known by / to converse with | Carnal knowledge; cf. *participer avec*. (annot-074, annot-077) |
| digne plus d'excuſe, que de peine | more worthy of excuse than of penalty | (annot-074) |
| engroſſir | to get with child | (annot-074) |
| œuure precedente mauuaiſe | preceding evil work | The canonists' *versari in re illicita*. (annot-074) |
| acte de ſoy mauuais, & reprouué | act in itself evil and reprobate | Cf. *art reprouué*. (annot-074) |
| meurtrir / occire | to murder / to slay | (annot-074) |
| le Philoſophe | the Philosopher | Aristotle; Coras's capital kept. (annot-075) |
| croire de leger | to believe lightly | Cf. *deposer à credit*. (annot-075) |
| foible nature des femmes | feeble nature of women | *Fragilitas sexus*. (annot-075) |
| tromperies & circonuentions | trickeries and circumventions | Per *tromperie*, *circonvenir*. (annot-075) |
| incroyable enuie (de recouurer) | incredible longing (to recover) | Per §10; cf. *envieuse de voir & recouvrer*. (texte-74) |
| ſ'apperceuoir de la fraude | to perceive the fraud | (texte-74, annot-077) |
| ſouhaitter la mort | to wish for death | (texte-74) |
| fort bouleuert | strong bulwark | (annot-076) |
| mort honneſte | honest death | Honourable. (annot-076) |
| celeſte heritage | heavenly inheritance | (annot-076) |
| le dernier ſouſpir & periode de ſa vie | the last sigh and period of one's life | (annot-076) |
| vaſſaux, & ſeruiteurs treſ-obligez | vassals and most bounden servants | Of God. (annot-076) |
| trencher le filet de la vie | to cut the thread of life | (annot-076) |
| mort volontaire | voluntary death | Suicide; Coras has no single noun for it. (annot-076) |
| ſ'occir de ſes propres mains | to slay oneself with one's own hands | (annot-076) |
| impatience de douleur | impatience of grief | The Digest's *impatientia doloris*. (annot-076) |
| negocier aux traffiques de ce monde | to deal in the traffic of this world | (annot-076) |
| eternizer leur memoire | to eternise their memory | (annot-076) |
| ſe faire eſtimer Dieu | to get oneself esteemed a God | Empedocles. (annot-076) |
| à cachettes | secretly | (annot-076) |
| pantouffles d'eſtain | slippers of tin | Empedocles' sandal; the tradition has bronze. (annot-076) |
| reuenir à nos moutons | to come back to our sheep | Glossed in brackets on first use; cf. *revenir à nos brisées*. (annot-076) |
| deuancer ſes iours | to forestall one's days | (annot-076) |
| ſe maſſacrer, & deffaire | to slaughter and undo oneself | (annot-076) |
| auoir en garde / prendre à garder / bailler en garde | to have in keeping / to take into one's keeping / to deliver into keeping | The soul as a deposit. (annot-076) |
| bagues | rings | The old sense, jewels. (annot-076) |
| l'opinion de ſa chaſteté | the reputation of her chastity | (texte-75) |
| mettre en Iuſtice | to bring to Justice | Cf. *mettre en instance* → to sue. (texte-75) |
| viuement pourſuyuir | to pursue vigorously | (texte-75) |
| perdre la teſte, & eſtre mis en quatre quartiers | to lose his head and be put in four quarters | The Rieux sentence. (texte-75) |
| interietter appel | to lodge an appeal | (texte-75) |
| demeurer arreſtée | to remain under arrest | The second sense of *arrest* at annot-068; glossed in brackets on first use. (texte-75) |
| faire grande euidence de | to make great evidence of | (annot-077) |
| ſe foruoyer de | to stray from | (annot-077) |
| charnellement cohabiter | to cohabit carnally | (annot-077) |
| eſtre dite adultere | to be called an adulteress | (annot-077) |
| caut, ſubtil, malicieux | cunning, subtle, malicious | (annot-077) |
| diſſimulé paillard | dissembling lecher | *Paillard* in its first sense here. (annot-077) |
| pourſuyuir vertueuſement | to pursue virtuously | (annot-077) |
| ſans pardonner à ſes biens, ni à ſes peines | sparing neither her goods nor her pains | (annot-077) |

## For the reviewer

1. **annot-074 check result.** The one failure is the checker's double count of `{k}` inside the French `⟨alt⟩`; the English is right as it stands. Decide whether to patch `check_markers.py` to ignore `⟨alt…⟩` or simply re-run after the alt-choices are applied.
2. **Printed vs. counted annotation numbers.** From this batch on the print's heads run one ahead (LXXIII for the 72nd). Each Notes headnote records both. Coras's cross-reference at annot-026 {b} agrees with the count, not the print — evidence for the transcription's numbering, worth a line in the front matter when the numbering is settled. The reviewer may also wish to check the preceding batch (texte-66–annot-071) for where the slip begins.
3. **annot-074, Pope "John XIII".** Kept as printed (Platina's count); the modern John XII. The Notes say so. If the house prefers the modern number in the text, it should be in brackets.
4. **annot-074 {d2} and {f}.** The two canon-law citations hiding behind civil-law sigla (*l. j. C. de homicid. au vj.*; *c. i. c. de homicid. aux Decretales*) are read as Sext V.4.1 and X 5.12.1; a canonist's check would settle them.
5. **annot-075 {b}, "Faustus".** Identified as Fausto Andrelini's *Livia*, book IV — a confident identification of the author and work, but the couplet was not located in the text. Worth confirming before print.
6. **annot-076, verses.** House treatment applied: one-line Latin in parentheses in the text, longer Latin in the Notes with the English in the text. The *Dicam* that heads Horace's lines is not Horace's; translated as part of the quotation with a note — the alternative is to take it as Coras's lead-in ("[I shall tell:]") and drop it from the quotation.
7. **annot-076, "her passions".** *Parole d'vne perſonne, ſuiette par trop à ſes paſſions* — the generic *ſes* is rendered "her", the person meant being Bertrande. Change to "its" or recast if the house wants the generic kept.
8. **annot-076, Hegesippus.** Coras's *Egeſippe* is the Latin *De excidio Hierosolymitano* (pseudo-Hegesippus), carrying Josephus's Jotapata speech against suicide; no marginal note is given for it. Recorded in the headnote only.
9. **texte-75, *arreſtée*.** Glossed in brackets as *arrest* in the second sense of annot-068. Davis and the secondary literature do not, so far as the case file records, note that Bertrande was under this restraint during the appeal; the point may deserve a word in the apparatus.
10. **annot-077, *paillard*.** Rendered "lecher" (the glossary's first sense), where the previous batch had "rogue" for the trickster at annot-047; here the fraud in question is carnal. Harmonise if the house wants one word throughout.
11. **texte-74 / case file §10.** Coras's phrase for Bertrande's motive is rendered "the incredible longing… to recover her husband", matching the case file's own translation of it; the glossary's earlier *envieuse de voir & recouvrer* → "desirous of seeing and recovering" (texte-06) is the same word in a weaker rendering — the two should be reconciled when the whole is revised.

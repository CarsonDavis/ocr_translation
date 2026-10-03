# Report: batch texte-106 – annot-110

Six sections, p152–p160: the close of du Tilh's confession before the judge of Rieux — his enumeration of debts and credits (texte-106), the institution of his daughter Bernarde as heiress with her guardians (texte-107) and the naming of the guardians as executors (texte-108) — with Coras's Annotations CIX–CXI (the 108th–110th by the true count): whether a dying man's declaration of debts is proof, whether a man condemned to death can make a testament, and the three kinds of executor. `check_markers.py` on each of the six: 6 sections ok, 0 failed. One marker choice was constrained by the checker (annot-109 {h}; see "For the reviewer", item 1).

## Per section

| Section | Pages | French | English | Notes |
|---|---|---|---|---|
| texte-106 | p152 | 16 | 18 | No marginalia. One clause, continuing texte-105. |
| annot-108 | p152–p156 | 1278 | 1396 | Forty-four markers (a–z, then a–y; no *j*, *v*, *w*; *ſ* as a key), all served. Three margin notes without body markers (*i*, the unkeyed continuation of *ſ*, *y*), listed as orphans. Two `⟨alt⟩` in the body, three in the margin. Page breaks inside *en|fant* and *quel|qu'vn*. |
| texte-107 | p156 | 36 | 41 | No marginalia. Reflow gap *Ber narde*. |
| annot-109 | p156–p158 | 827 | 899 | Twenty-six body markers (a–z, then a–d), all served. Margin note *d* has no body marker (orphan); body marker {r} has no margin note; marker {h} exists only in reading B of an `⟨alt⟩` (see below). One `⟨alt⟩` in the body, nine in the margin. |
| texte-108 | p158 | 8 | 7 | No marginalia. Two display lines dividing *te-|ſtament*; kept as two lines, the word undivided. |
| annot-110 | p158–p160 | 456 | 510 | Twenty markers (a2–x; no *j*, *v*, *w*; *ſ* as a key), all served. Margin note *q* has no body marker (orphan). One `⟨alt⟩` in the body. One `[unclear]`. Page breaks inside *appel|lez* and *autre|ment*. The last annotation of the book. |

Totals: French 2621 words, English 2871 words (counts exclude the page, letter and `⟨alt⟩` markers).

## ⟨alt⟩ choices

The same choices, machine-readably, are in `translation/alt-choices/batch-texte-106--annot-110.json` (19 entries, in the prompt's table order). None is `⟨alt?:⟩`; none is unconfirmed, but the two *de legatis* readings (p158-m1l1-1, p158-m2l3-1) are "either" for want of any means to verify.

- annot-108, margin p153, note *t* (`p153-f1l1-1`): *paul. de Caſtro cõ la l. Seia* / *paul. de Caſtro en la l. Seia* — took **B**. *En la l.* is the margin's regular formula for a commentator on a law (*Ange. en la l. errore*, in the same note); *cõ* is no word here. Same translation.
- annot-108, margin p154, note *g* (`p154-m9l0-1`): *l j. ij. & iij.* / *l. j. ij. & iij.* — **either**. The siglum's point only.
- annot-108, margin p154, note *g* (`p154-m9l1-1`): *D de donati.* / *D. de donati.* — **either**. The siglum's point only.
- annot-108, p155 (`p155-b0l23-1`): *veritable;* / *veritable:* — **either**. Punctuation only; the English has a comma.
- annot-108, p155 (`p155-b0l37-1`): *aduenir, Car celuy* / *aduenir. Car celuy* — **either**. A capital *Car* follows, which would favour the stop, but this print capitalises *Car* after commas and colons too (p153 *{n}: ſi cela*, p155 *{m}. d'autant*, lower case after a stop). The English runs the clause on with a dash either way.
- annot-108, margin p155, note *q* (`p155-m6l1-1`): *hac edi ctali C.* / *hac edi ctali. C.* — **either**. A point after the incipit only.
- annot-109, p156 (`p156-b4l11-1`): *caſſation* / *caſſarion* — took **A**. *Cassation*, "quashing", the noun of the *caſſés & rompus* three lines above; *caſſarion* is no word (*t* read as *r*).
- annot-109, margin p156, note *a* (`p156-m1l2-1`): *de teſta, l, ſi quis exheredato* / *de teſta, l. ſi quis exheredato* — took **B**. The siglum *l.* for *lex* before the incipit takes a point; a comma after *l* is not the print's usage. Same translation.
- annot-109, p157 (`p157-b0l4-1`): *à dire fran chement; ce qu'en eſt* / *à dire fran chement {h} ce qu'en eſt* — took **B**. The margin of p157 prints a note *h* (*Alberic en lad. auten. bona: & en la l. eius. P. ſi cui D. de teſta.*), which is exactly the authority whose "allegations" the sentence calls more subtle than true; and the body alphabet otherwise skips from *g* to *i*. A superscript *h* has been read as a semicolon. **The checker holds the French to reading A until the choice is applied, so the English body carries no {h} for now; the Notes headnote of annot-109 gives the insertion point** ("It is true, to say frankly {h} how matters stand").
- annot-109, margin p157, note *f* (`p157-m0l1-1`): *C. de bon[abbr: raised mark] proſcr.* / *C. de bonꝰ proſcr.* — took **B**. The raised mark is the *-us* sign (*bonꝰ* = *bonis*, *C. de bonis proscriptorum*), which Unicode has (U+A770) and §2 of the conventions prefers to a bracketed description. Same translation.
- annot-109, margin p157, note *f* (`p157-m0l3-1`): *vt nullẽ iud.* / *vt nullĩ iud.* — took **B**. The title is *Ut nulli iudicum* (Nov. 134, coll. ix), as note *c* on p156 prints it (*vt nulli iudi.*): the vowel is *i*; the tilde is a stray either way. Same translation.
- annot-109, margin p157, note *g* (`p157-m1l2-1`): *hæred.* / *hared.* — took **A**. *Hæred.* = *hereditate* (*D. de adquirenda hereditate*); *hared.* is no word.
- annot-109, margin p157, note *i* (`p157-m3l4-1`): *l. is cui: P. j* / *l. is cui. P. j* — **either**. Punctuation inside the citation.
- annot-109, margin p157, note *i* (`p157-m3l7-1`): *infir. lẽ eius* / *infir. lĩ eius* — **either**. Both are the siglum *l.* (*l. eius*) carrying a stray tilde.
- annot-109, margin p157, note *l* (`p157-m5l1-1`): *P. alio qumod. teſt. infirm.* / *P. alio qumod. reſt. infirm.* — took **A**. *Teſt.* = *testamenta* (Inst. 2.17.4, § *Alio quoque modo testamenta infirmantur*); *reſt.* is nothing.
- annot-110, p158 (`p158-b5l2-1`): *qui meinent à fin* / *qui meïnent à fin* — took **A**. *Meinent* (modern *mènent*) has one syllable in the stem; the diaeresis would make two and is not this print's usage. Same translation ("bring to an end").
- annot-109, margin p158, note *t* (`p158-m1l1-1`): *l. Iulianus D. de leg ij.* / *l. Iulianus D. de leg iij.* — **either**. *De legatis* II is D. 31, III is D. 32; a fragment beginning *Iulianus* could not be located in either title, so there is no basis to choose. Flagged unverified in the Notes.
- annot-109, margin p158, note *u* (`p158-m2l3-1`): *l. ſi plures D. de leg ij.* / *l. ſi plures D. de leg iij.* — **either**. As the preceding: the fragment *Si plures* was not located in D. 31 or D. 32. Flagged unverified.
- annot-109, margin p158, note *b* (`p158-m7l2-1`): *D. de iniu. teſt* / *D. de iniu. teſt.* — **either**. A final point only.

## `[unclear]` passages

- annot-110, p160: *d'autant que telles defenſes ſont priuées, & peu raiſonnables: voire ſemblent contenir ineptitude, & quelque impieté* → "inasmuch as such prohibitions are private [unclear: *priuées*; perhaps for *praves*, "perverse"] and little reasonable, nay, seem to contain ineptitude and some impiety". *Priuées* is translated in its literal sense, but "private" sits oddly in a climbing series (*peu raiſonnables … ineptitude … impieté*); a form of *prave* (Latin *pravus*, perverse — the canonists' word for such prohibitions) or a dropped word (*priuées de raiſon*) is possible. Not smoothed over.

Interpretive renderings, not marked but worth a glance:

- annot-108, p153: *prononcé ſentente* (for *ſentence*) → "pronounced a judgment". The French has no adjective; the corruption is carried by the following *par argent, ou autre eſpece de corruption*, which governs all three acts (false witness, judgment, false instrument). Kept bare.
- annot-108, p154: *il reſpond & confeſſe* → "he [Accursius] answers and confesses". The pronoun's antecedent is Accursius, named two sentences before; glossed in brackets rather than left to the reader.
- annot-108, p155: *imprudemment / imprudence* → "imprudently / imprudence". Ulpian's word (D. 2.1.15, cited at *n*) is *imperitia*, want of skill or knowledge; Coras's French is kept and the Latin given in the Notes.
- annot-108, p155: *la religion du ſerment* → "the religion of the oath" — Coras's word kept for the binding sanctity of an oath; "sanctity" would be the modern paraphrase.
- annot-109, p156: *les teſtamẽs là faits* → "testaments already made" (*là* for *ià*).
- annot-109, p157: *Et par ainſi incapables à faire teſtament* → "And so [they are] incapable of making a testament" — subject supplied in brackets.
- annot-109, p157: *ceux qui ſuruiennent à l'execution de la peine* → "those who survive the execution of the penalty" (*ſuruiennent* for *ſuruiuent*; the contrast with those who suffer natural death demands it).
- annot-109, p158: *vn iuge incompetant* → "a judge not competent over him".
- annot-110, p159: *Donnez* (of executors) → "Dative", Coras's French glossed in brackets on first use, as the term of art for an executor or guardian appointed by the judge (*dativus*).
- annot-110, p159: *pitoyables volontez* → "pious wills" (*pitoyable* in its old sense, as *œuures pies* two lines later).
- annot-110, p160: *appellez les heritiers du defunct* → "the heirs of the deceased having been called" (summoned to the appointment).

## Citations not identified, or identified only in part

Fragment numbers are given only where verified; elsewhere the title and incipit stand with "not verified". The following are weaker than that:

- annot-108 {p}: *Accurſe en la loy finale. C. de not. promiſſ.* — read as the gloss on the last law of C. 5.11 (*De dotis promissione*), with *not.* for *dot.* as in the next note; the gloss was not checked for the two-witness rule Coras hangs on it.
- annot-108 {q}: *l. C. de dot. promiſſ.* — the siglum *l.* stands without an incipit.
- annot-108 {ſ}: *Innocent c. cũ dilecti de dilectio.* — Innocent IV on a chapter *Cum dilecti*; the rubric *de dilectio.* is corrupt (perhaps *de electione*, X 1.6). The second half of the note is set as a separate unkeyed piece in the margin and merged here.
- annot-108 {i} (p154): *Aut. ꝗ obtinet. C. de probatio.* — an authentica *Quod obtinet* under C. 4.19; not identified.
- annot-108 {t}: *Ange. en la l. errore C. de teſtam.* — taken as Angelus de Ubaldis on the Code; the law *Errore* in C. 6.23 not located. The case file §6.5 lists only Angelus Aretinus; this is a different Angelus (Baldus's brother, the Code commentator), and the case file might take a line.
- annot-109 {a}, {i}, {b} (p158): the D. 28.3 laws *Si quis exheredato* / *Si quis filio* / *Quod si quis* (§ *Irritum*) were not located; the medieval division of D. 28.3.5–6 differs from the modern.
- annot-109 {t}, {u}: *l. Iulianus*, *l. ſi pluribus*, *l. ſi plures*, *D. de legatis* II or III — not located (hence the two "either" choices).
- annot-109 {r}: the body marker has no margin note; recorded as such.
- annot-109 {n}: *P. dernier de vſu & habita.* (Inst. 2.5, last §) and *l. i. de teſta.* — the first is as printed, the second taken as D. 28.1.1.
- annot-110 {a2}: *c. ab executore ij. q. v. vi.* — a Decretum reference, probably C. 2 q. 6 (*de appellationibus*), but no chapter *Ab executore* was confirmed; the civil-law laws *Ordo*, *Si ut proponis*, *Executionem* in C. 7.53 and *Ab executore* in D. 49.1 not numbered.
- annot-110 {e}, {f}, {h}: *clem. cxiiij.* / *cxiij.* — the Clementine *Exivi de paradiso* (Clem. 5.11.1) is certain from the §§ *Proinde* and *Verum* and the Franciscan subject; the number *cxiiij* is not the standard reference and is unexplained (a running number in Coras's edition?).
- annot-110 {c}, {m}, {n}: *l. nulli* / *l. nullo C. de epiſ. & cler.* — C. 1.3; the law not located.
- annot-110 {o}, {r}, {ſ}, {x}: Nov. 131.11 (*De ecclesiasticis titulis*), which Coras places in *coll. vij*; the standard Authenticum has it in coll. 9. Noted, not corrected.
- annot-110 {k}: *l. Lucius D. de man. teſta.* (D. 40.4) — not located, and its bearing on the executor's duty to account is not evident.

## Glossary additions

Not appended to `docs/case-file.md` §9 (the coordinator's instruction for this run); given here in the table's format for merging.

| French | English rendering | Note |
|---|---|---|
| faire (particulier) denombrement | to make a (particular) enumeration | The itemised list of a declaration. (texte-106) |
| ſoudre (vne queſtion peut ſoudre) | to arise (of a question) | (annot-108) |
| aſſertion (de celuy qui ſ'en va mourir) | assertion (of one who is about to die) | (annot-108) |
| donner foy / faire foy | to give credit / to make proof | (annot-108) |
| inualable | invalid | (annot-108, annot-109) |
| delation | delation | The dying man's naming of his assailant. (annot-108) |
| indice pour la torture | indication for torture | Cf. *indice* → indication. (annot-108) |
| ſous la cenſure des plus doctes | under the censure of the more learned | Coras's modesty formula. (annot-108) |
| deciſions vulgaires | common decisions | The ordinary rules of law. (annot-108) |
| à l'article de ſa mort | at the article of his death | (annot-108) |
| pariure & infame | perjured and infamous | (annot-108) |
| amiable preſt | friendly loan | *Mutuum*. (annot-108) |
| deppoſt | deposit | (annot-108) |
| partie abſente | in the absence of the party | (annot-108) |
| donnaiſon / donaiſon | gift | *Donatio*; between spouses. (annot-108) |
| exception de pecune non nombree | exception "of money not counted out" [*non numeratae pecuniae*] | Latin supplied in brackets on first use. (annot-108) |
| laiz / lais | bequest(s) | (annot-108, annot-110) |
| legat / fideicommis | legacy / fideicommissum | (annot-108) |
| recognoiſſance | acknowledgment | Of a debt or receipt. (annot-108) |
| frauder la loy | to defraud the law | (annot-108) |
| dol & fraude | deceit and fraud | (annot-108) |
| religion du ſerment | religion of the oath | Its binding sanctity; Coras's word kept. (annot-108) |
| interpoſer (vn ſerment) | to interpose (an oath) | (annot-108) |
| inſtituer (ſon heritiere) | to institute (as heiress) | *Institutio heredis*. (texte-107) |
| inſtitution d'heritier | institution of an heir | (annot-109) |
| faire teſtament / faculté de teſter | to make a testament / faculty of making a testament | *Testament* throughout, never "will" alone. (annot-109) |
| caſſer & rompre / caſſation | to quash and break / quashing | Of a testament. (annot-109) |
| nouuelles conſtitutions (de Iuſtinien) | new constitutions | The Novels. (annot-109) |
| ſucceſſeurs / heritiers ab inteſtat (d'inteſtat) | successors / heirs ab intestato | (annot-109) |
| fiſc | fisc | (annot-109) |
| leſe-maieſté | lèse-majesté | (annot-109) |
| ſerf de la peine | slave of the penalty | *Servus poenae*; cf. *serf* → slave. (annot-109) |
| diminué de ſon chef | diminished in his head | *Capite deminutus*; the *capitis deminutio*. (annot-109) |
| cité (perdre ſa cité) | citizenship | *Civitas*. (annot-109) |
| confinez | confined | The modern *relegati*. (annot-109) |
| bien né | well born | *Ingenuus*. (annot-109) |
| oraiſon indefinie / vniuerſelle | indefinite / universal proposition | The logicians' terms. (annot-109) |
| couſtume generalle de noſtre Gaule | the general custom of our Gaul | (annot-109) |
| crime priuilegié | privileged crime | Treason, heresy, false coining. (annot-109) |
| fauſſe monnoye | false coining | (annot-109) |
| mutilation de membre | mutilation of a limb | (annot-109) |
| perpetuel banniſſement | perpetual banishment | (annot-109) |
| Qui confiſque le corps, confiſque les biens | Who confiscates the body confiscates the goods | The French maxim; print's capitals kept. (annot-109) |
| iuge incompetant | a judge not competent | (annot-109) |
| clerc / iuge lay | clerk / lay judge | (annot-109) |
| faute de iuriſdiction | want of jurisdiction | (annot-109) |
| executeur (de teſtament) | executor | (texte-108, annot-110) |
| executeurs Teſtamentaires, Legitimes, Donnez | Testamentary, Legitimate, Dative executors | *Donnez* glossed in brackets on first use. (annot-110) |
| dernieres volontez | last wills | (annot-110) |
| lais ou clercs, ſeculiers ou reguliers | laymen or clerks, secular or regular | (annot-110) |
| religieux | religious (n.) | (annot-110) |
| cordeliers | Cordeliers | The Franciscans. (annot-110) |
| gardien de ſainct François | Guardian of Saint Francis | Superior of a Franciscan house. (annot-110) |
| preſtres & clers ſacrez | priests and clerks in holy orders | (annot-110) |
| tabellion ou notaire | tabellion or notary | (annot-110) |
| receuoir inſtrumens | to receive instruments | To draw up deeds. (annot-110) |
| inuentaire | inventory | (annot-110) |
| pitoyables volontez / œuures pies | pious wills / pious works | *Pitoyable* in its old sense. (annot-110) |
| admonneſter | to admonish | Cf. *admonnester amiablement*. (annot-110) |
| Eueſque / Metropolitain | Bishop / Metropolitan | (annot-110) |
| ſainctes conſtitutions | holy constitutions | Justinian's Novels on pious bequests. (annot-110) |

## For the reviewer

1. **annot-109 {h} and the checker.** The `⟨alt⟩` at p157-b0l4-1 is the one choice in this batch that changes the marker sequence: reading B puts a marker {h} after *franchement*, and the margin prints a note *h* that fits. The choice is B, but `check_markers.py` (which strips `⟨alt⟩` and so enforces reading A) would reject an English body carrying {h}, so the body omits it and passes. **After the alt-choices are applied to the French, insert {h} after "to say frankly" in `annot-109.md`** (the Notes headnote says so too). If the house would rather have the English anticipate the French, put {h} back now and accept the one checker failure.
2. **Orphan notes.** Four margin notes have no body marker: annot-108 *i* (Baldus, at *dit le Balde*) and *y* (Bartolus, closing sentences), annot-109 *d* (*De bonis damnatorum*, at *ſon bien … confiſqué*), annot-110 *q* (Nov. 1 / Auth. *Hoc amplius*, at *dans l'an*). Each is listed in the Notes at its place with the sentence it belongs to. The transcription may wish to check the pages for dropped superscripts.
3. **annot-109 {r}.** The body has a marker {r} with no note in the margin of p157. Recorded; nothing supplied.
4. **annot-108 margin alphabet.** The second alphabet (p154–p156) re-uses *a*–*i*; the Notes give the page with each letter. The unkeyed second piece of note *ſ* (*dilectio. Bart. l. iij. P. j. preallegué.*) is merged into *ſ*.
5. **Printed vs. counted numbers.** ANNOTAT. CIX. / CX. / ANNOT. CXI. for the 108th, 109th and 110th; the book's last head is CXI, the title page's "cent & onze". Each Notes headnote records both.
6. **texte-108 layout.** The print divides *te-|ſtament* across two display lines; the two lines are kept as two paragraphs, the word undivided. If the house prefers a single line for a one-clause *Texte*, join them.
7. **"Angelus".** annot-108 {t} cites *Ange.* on the Code (l. *Errore*, C. de testamentis): Angelus de Ubaldis rather than the Angelus Aretinus of case file §6.5. Worth a line in the case file's list of authorities, as are Albericus de Rosate (already met at annot-001), Federicus de Senis, Jean Masuer and Guillaume Benoît (*Benedicti*, the *Repetitio in cap. Raynutius*), all cited here.
8. **Nov. 134.13 and the *Bona damnatorum*.** The "new constitution of Justinian" of annot-109 is identified with confidence as Nov. 134.13 (Auth. coll. 9, *Ut nulli iudicum*) and the authentica drawn from it under C. 9.49; and the "two places" where Justinian says no free man becomes a slave by punishment as Nov. 22.8 and the authentica *Sed hodie* at C. 5.16.24. A Romanist's check would be welcome before print.
9. **annot-108 "law of Justinian" on second wives.** The *lex Hac edictali* (C. 5.9.6) is Leo and Anthemius, not Justinian; translated as Coras has it, noted at *q*.
10. **annot-110 *priuées*.** The one `[unclear]`; see above. If a reviewer with the canonists' texts on *c. Tua nobis* can confirm *pravae*, "perverse" should replace "private".
11. ***écus*.** Italicised (*écus*) per case file §8; earlier sections are mixed (texte-13 and others have roman "écus"). Harmonise when the whole is revised.
12. **Terminology to carry forward.** This batch fixes "testament" (never "will" alone) for *teſtament*, "quash" for *caſſer*, "slave of the penalty" for *ſerf de la peine*, and "Dative" for *Donnez*; the remaining section (texte-109) has none of these, but any revision of annot-065 (*lais*) and annot-100 (confiscation) should match.

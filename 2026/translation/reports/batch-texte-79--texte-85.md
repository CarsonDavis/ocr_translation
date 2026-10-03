# Report: batch texte-76 – texte-82

Twelve sections, p106–p123: the end of the narrative *Texte* (Martin Guerre's words to Bertrande, texte-76/77), the text of the *arrest* of 12 September 1560 (texte-77), the display head of the *Exposition des paroles de l'arrest*, and the first five words of the decision expounded — *ce dont a esté appelé, au neant*, *Fausseté*, *Supposition de nom & personne*, *Adultere*, *Rapt* — with Annotations LXXIX–LXXXIII as the print numbers them (annot-078 to annot-082; the print's numbering runs one ahead of the section ids in this stretch and realigns at the next annotation, which the print also heads LXXXIII).

`check_markers.py` on each of the twelve: **10 sections ok, 2 failed** — annot-080 and annot-082. Both failures are an artefact of unresolved `⟨alt⟩` markers in the French that themselves contain a letter marker (`{x}.⟨alt:{k}.⟩` on p111; `{a};⟨alt:{a}:⟩`, `{l};⟨alt:{l}:⟩`, `{r},⟨alt:{r}.⟩` on p119 and p123): the checker reads the French body with `re.findall` and counts the `{k}`, second `{a}`, second `{l}` and second `{r}` inside the alt text as markers. The English carries each marker once, as the French will once the choices below are applied and the pages refinalized; with the `⟨alt:…⟩` text stripped from the French, both sections' page and letter sequences match exactly (verified). The English was deliberately not padded with `⟨alt:{k}⟩` etc. to pass the current check, since that would fail the moment the French is resolved. A one-line fix to `check_markers.py` (strip `⟨alt\??:[^⟩]*⟩` from `sec["text"]` before the two `findall`s) would make it robust to this; not applied, as this run writes only the translation files.

## Per section

| Section | Pages | French | English | Notes |
|---|---|---|---|---|
| texte-76 | p106 | 76 | 85 | No body marker. One orphan note *a2* (Propertius) in the margin of p106, serving the first line of the next annotation; listed. |
| annot-078 | p106–p107 | 104 | 118 | Markers b–d, served. Two Latin couplets (Propertius 3.25.5–6; Ovid *Ars* 3.291–292) translated in the text, Latin in the Notes. |
| texte-77 | p107–p109 | 459 | 518 | No marginalia. Contains the *arrest* under its head ARREST. and the display head EXPOSITION DES / Paroles de l'arreſt. Page breaks between words. |
| texte-78 | p109 | 8 | 7 | No marginalia. The fragment repeats the formula as rendered in texte-77. |
| annot-079 | p109–p110 | 185 | 183 | Markers a–d, served. Page break inside *ſingulie\|rement*. One `[unclear]` (missing predicate). |
| texte-79 | p110 | 1 | 1 | *Fauſſeté* → "Forgery". |
| annot-080 | p110–p111 | 276 | 286 | Markers a2–f2, g, h, i, x, l; one unkeyed orphan note (*ſiue adulterium. x. diſt.*). One `[unclear]` (disordered close). Checker: see above. |
| texte-80 | p111 | 5 | 5 | |
| annot-081 | p111–p118 | 2768 | 3008 | Markers a–z then a, c, d, e, f (p118); orphan note *b* on p118, listed in place. Page breaks inside *li\|ſons*, *l'alle\|rent*, *pre\|mier*. Twenty-seven notes plus the orphan. |
| texte-81 | p118 | 1 | 1 | |
| annot-082 | p118–p123 | 1643 | 1723 | Markers a–z then a–y; note *p* keyed in the margin of p123 for a body marker on p122 (transcription: orphan), listed as {p}. Page breaks inside *at\|taint*, *pluſ\|goſt*. One `[unclear]`. Checker: see above. |
| texte-82 | p123 | 1 | 1 | *Rapt* → "Rape"; see "For the reviewer". |

Totals: French 5527 words, English 5936 words (counts exclude the page and letter markers, the `⟨alt⟩` text and the `[unclear: …]` brackets).

## `⟨alt⟩` choices

Forty-two markers; the same choices, machine-readably, are in `translation/alt-choices/batch-texte-76--texte-82.json` (19 A, 9 B, 14 either). None is an `⟨alt?:⟩`.

| alt_id | section | A | B | took | why |
|---|---|---|---|---|---|
| p107-b3l4-1 | annot-078 | *cou uert* | *cou uet* | A | *couvert*, "covered"; *couuet* is no word. |
| p107-b5l1-1 | texte-77 | *Oncle,* | *Vncle,* | A | The same word is *Oncle* two words earlier in the sentence; *Vncle* is not French. |
| p107-m1l0-1 | annot-078 | *Ouid. au iij* | *au iiij* | A | *Ars amatoria* 3.291–292; the *Ars* has three books. |
| p107-m2l0-1 | annot-078 | *Plaute* | *Plaure* | A | Plautus (*Truculentus* 178–179). |
| p110-b4l6-1 | annot-080 | *bonne foy,* | *bonne foy;* | A | The phrase continues *& ſans intention de frauder*. |
| p110-b4l17-1 | annot-080 | *d'ar moiries* | *d'ar-moiries* | B | One word, *d'armoiries*, broken at the line end; the hyphen reading is the one consistent with the word. |
| p110-m6l3-1 | annot-080 | *l'a notarion,* | *l'a notation,* | B | *l'annotation*, a cross-reference to Annotation XII; *anotarion* is no word. |
| p110-m7l1-1 | annot-080 | *Crinit. ru iij.* | *ru iiij.* | A | Crinito, *De honesta disciplina*, book 3 or 4, ch. 10: not verifiable here; A kept as the primary reading. **Unverified**; the reviewer with a copy of Crinito should check. |
| p110-m7l3-1 | annot-080 | *c. x.* | *c. x,* | A | The full stop closes the citation at the end of the note. |
| p111-b0l5-1 | annot-080 | *{x}.* | *{k}.* | A | The margin keys the matching note *x*; body and margin agree on *x*. The alphabet expects *k* (g, h, i, k, l), so the print probably used an *x* sort for *k* in both places; the transcription's *x* is kept rather than making body and margin disagree. |
| p111-b4l9-1 | annot-081 | *proprement* | *propremẽt* | either | Same word. |
| p111-b4l17-1 | annot-081 | *manteau,* | *manteau.* | A | The sentence continues *prins les armes, & excité*. |
| p111-b4l28-1 | annot-081 | *terminé,* | *terminé.* | A | *tant par ce que … que pour autant auſſi* is one period. |
| p111-m0l1-1 | annot-080 | *c. xi* | *c. xi.* | either | Final point; translation unaffected. |
| p112-b0l29-1 | annot-081 | *perſuadè* | *perſuadé* | B | Past participle; acute, not grave. |
| p113-b0l5-1 | annot-081 | *tirre* | *titre* | B | *le nom & titre de fils de Roy*, the pair used on p111; *tirre* is no word. |
| p113-b0l36-1 | annot-081 | *Ceſar;* | *Ceſar,* | either | Punctuation only. |
| p114-b0l2-1 | annot-081 | *vefue* | *veſue* | A | *veuve*; *veſue* is an *f*/*ſ* confusion. |
| p116-b0l12-1 | annot-081 | *toute* | *toure* | A | *toute eſpece*; *toure* is no word. |
| p116-b0l14-1 | annot-081 | *precipiré* | *precipité* | B | *précipité*, "plunged". |
| p118-m5l0-1 | annot-081 | *P j.* | *P. j.* | either | The paragraph sign with or without a point. |
| p119-b0l0-1 | annot-082 | *at taint* | *at ſaint* | A | *attaint & convaincu*, the set phrase (annot-019). |
| p119-b0l4-1 | annot-082 | *{a};* | *{a}:* | either | Punctuation only. |
| p119-b0l23-1 | annot-082 | *entierement:* | *entierement;* | either | Punctuation only. |
| p119-b0l38-1 | annot-082 | *{l};* | *{l}:* | either | Punctuation only. |
| p119-m2l0-1 | annot-082 | *l. ij.* | *l. ij* | either | Punctuation only. |
| p119-m2l3-1 | annot-082 | *D. de iis quib.* | *D. de iuſquin.* | A | The Digest title *De his quibus ut indignis* (D. 34.9), where l. *Claudius* stands. |
| p119-m10l3-1 | annot-082 | *Cornelia* | *Cornelia.* | either | Punctuation only. |
| p120-b0l3-1 | annot-082 | *Narurellement,* | *Naturellement,* | B | Paired with *Ciuillemẽt*; *Narurellement* is no word. |
| p121-m0l1-1 | annot-082 | *de iud,* | *de iud.* | either | Punctuation only. |
| p121-m2l2-1 | annot-082 | *de iu dic,* | *de iu dic.* | either | Punctuation only. |
| p121-m4l2-1 | annot-082 | *que ſit long,* | *que ſit long.* | either | Punctuation only. |
| p121-m5l4-1 | annot-082 | *Teſſalo. c. iij.* | *c. iiij.* | B | The passage cited ("possessed in honour and sanctification") is 1 Thessalonians 4:3–7. The print's *ſeconde* is itself wrong (2 Thessalonians has three chapters), so one of the two numbers must be an error; the chapter that matches the text is taken. **Flagged**: the reviewer may prefer to keep A and note both errors. |
| p121-m6l0-1 | annot-082 | *Leuitique c,* | *Leuitique c.* | either | Punctuation only. |
| p121-m7l0-1 | annot-082 | *c. reos xxiij.* | *xxiiij.* | A | *Causa* 24 of the Decretum has no *quaestio* 5. |
| p122-b0l18-1 | annot-082 | *ſçay* | *ſcay* | A | The print's usual spelling (*ie ne ſçay quel*, p111, p117); same word either way. |
| p122-b0l31-1 | annot-082 | *plei ne* | *plei-ne* | B | One word, *pleine*, broken at the line end. |
| p122-m6l2-1 | annot-082 | *c. ſi-cut* | *c. ſi. cut* | A | *Sicut*, the chapter incipit, continued on the next line. |
| p123-b0l14-1 | annot-082 | *{r},* | *{r}.* | A | The relative clause *lequel il veut eſtre … puni de mort* continues the sentence. |
| p123-m4l1-1 | annot-082 | *P. ꝙ autẽ* | *P. ꝗ autẽ* | either | Both signs abbreviate *quod* (§ *Quod autem*). |
| p123-m4l4-1 | annot-082 | *ſi virgo. xxxiij.* | *xxxiiij.* | B | *Causa* 34 q. 1 (the wife who marries another believing her husband dead) fits the point on ignorance; *Causa* 33 q. 1 is on impotence. **An inference**, flagged; the chapters *In lectum* and *Si virgo* were not verified in either *causa*. |
| p123-m5l4-1 | annot-082 | *fœdiſſimam.* | *fædiſſimam.* | A | The Latin of C. 9.9.20, *Foedissimam earum nequitiam*. |

## `[unclear]` passages

- annot-079, p109–p110: *ceſte eſpece de mort luy ſembloit pour vn ſi prodigieux, & abominable proditeur comblé en toute eſpece de vices, ſingulierement que iaçoit …* — the main clause has no predicate after *ſembloit*; the rest of the annotation (nobles beheaded, commoners hanged, traitors hanged higher) shows that beheading seemed to the Court too honourable a death for du Tilh. Translated with the gap marked, not filled.
- annot-080, p111: *& telle, (diſoit en quelque lieu le Iuriſconſulte Vlpien qu'il en ſoit) autres aux exemple {l}* — the parenthesis is closed too late and *autres aux* is inverted; evidently *& telle (diſoit … Vlpien) qu'il en ſoit aux autres exemple*, "and such that he may be an example to others". Translated by that sense and marked.
- annot-082, p122: *ſemblẽt peu, ou point meriter quelque excuſe* — the opinion Coras is reporting (and goes on to refute) must say that vowed clerics who fall *deserve* some excuse, like the man who steals from hunger; the print says they deserve "little or none". Translated literally and marked; a word (*peine*? *blasme*?) may have dropped, or *peu ou point* may govern an unexpressed "punishment".

Bracketed glosses added for sense, not obscurity: annot-081 "drew the worms from his nose [wormed the truth out of him]"; annot-081 "the order" (of knighthood) is explained in the headnote rather than the text.

## Citations not identified, or identified only in part

- annot-079 {a}: *l. honor. D. de pœn.* — no fragment of D. 48.19 beginning *Honor…* found; C. 9.41.8 (*Milites*) identified.
- annot-079 {b}: *l. Indignat. li. xij.* — Code book 12 (perhaps *De dignitatibus*, C. 12.1), law not found. C. 3.24.1 identified.
- annot-079 {c}: Accursius on D. 49.16.2.1 — the gloss's bearing (gallows the commoner's death) inferred from the text, not checked in the *Glossa*.
- annot-079 {d}: *Balde au c. quidam de iure…* — title abbreviation unresolved (X 2.24 *De iureiurando* guessed); **unverified**.
- annot-080 {b2}: *§ ſed quia, [Inst.] qui[bus] modis teſt[amenta] infir[mantur]* — paragraph not found in Inst. 2.17; at annot-012 {a} the same point was cited to Inst. 2.20.29. *l. ad recognoſcendos C. de ingen. & man.* given as C. 7.14 with the number unverified (annot-012 did the same).
- annot-080 {d2}: Crinito, *De honesta disciplina* — book number is the `⟨alt⟩` above; chapter 10 as printed, unverified.
- annot-080 {e2}: *c. dilecta … de exceſ. prælator.* — X 5.31, chapter number not verified.
- annot-080, unkeyed orphan: *ſiue adulterium. x. diſt.* — **unidentified**; perhaps a canon of Decretum D. 10; its place in the argument is not evident.
- annot-080 {i}, annot-081 {g}: *c. quæritur xxij. q. ij.* — given as cf. C. 22 q. 2 c. 22 (Augustine on Jacob and on figurative speech); number not verified.
- annot-080 {l}, annot-081 {a}: *l. quamuis D. de reb. cor.* — title abbreviation unresolved, fragment not found; **unidentified**. Cited twice with the same abbreviation, so it is Coras's, not a sort error.
- annot-081 {c}, {m}: *c. perpendimus de ſent. excom.* — X 5.39, chapter number not verified (Clement III, as Coras says).
- annot-081 {d}, {k}: *l. ij. D. de offi. prætor.* — D. 1.14.3 (*Barbarius*); Coras's *l. ij* is an older numbering.
- annot-081 {e}: *l. i. § fi. & l. ij. D. de Carbo. edic.* — D. 37.10.1 and 2; paragraph not checked. *l. i. C. de fal.* — C. 9.22.1; see {a} below.
- annot-081 {i}: Pliny 7.12 = 7.53 modern; Solinus c. 5 = 1.78–80 modern; both located by content.
- annot-081 {l}: *l. ij. C. ſi ſer. ad decur. aſp.* — C. 10.33.2; Coras's "Emperor Augustus" does not fit the constitution's author.
- annot-081 {n}–{p}: Valerius Maximus 9.15 (Coras: c. xvi); the three anecdotes located (9.15.4, 9.15 ext. 1, 9.15.1).
- annot-081 {q}: Justin — the print's book *xxxij* is book 38 (38.1–2); read as a dropped *vi*.
- annot-081 {r}: Josephus *Ant.* 17.12 (17.324–338); the print's chapter is damaged (*x[?]vij*).
- annot-081 {t}: Appian *Syr.* 67–68 — Alexander Balas; Coras's *Prompalus* matches no name in Appian or Justin; kept as printed.
- annot-081 {u}, {x}: Fregoso, book 9 ch. 16 — not checked; Suetonius *Nero* 57 has a false Nero, but Coras's details (harper, Cythnus, two galleys) are Tacitus *Hist.* 2.8–9, and under Galba, not Otho.
- annot-081 {y}: Paolo Emili, *De rebus gestis Francorum*, book 7 — not checked against the text.
- annot-081 {z}: Platina, life of "John VIII" — Coras's *Leon iij* is Platina's Leo IV.
- annot-081 {a}, {d} (p118): *l. j. C. de fal.* — C. 9.22.1; that it is a constitution of Antoninus on the substitution of a child, punished capitally, rests on Coras's own description and was not independently verified.
- annot-081 {e} (p118): *l. edicto § j. D. de bono. poſſeſ.* — D. 37.1, fragment *Edicto* not found.
- annot-082 {a}: Valerius Maximus book 2 — chapter not given; cf. 2.1.
- annot-082 {c}: *l. ij. § miles D. de adulte.* — paragraph *Miles* not located in D. 48.5.2 (cf. D. 48.5.12(11) on the soldier). D. 34.9.13 (*Claudius*) identified.
- annot-082 {f}: *l. caſtitati C. de adulte.* — C. 9.9, fragment number not verified; the doubled *C. de tranſac.* is the print's.
- annot-082 {g}: *l. quãuis. la ij. C. de adul.* — C. 9.9.29(30) (*Quamvis*) and l. 2 of the title; the latter's bearing unverified.
- annot-082 {l}: *l. iij. C. de epiſ. aud.* — C. 1.4.3, bearing on homicide unverified; *c. j. de homici.* — X 5.12.1.
- annot-082 {n}: *l. illicitas. § vniuerſas. D. de offi. præf.* — cf. D. 1.18.6 (*De officio praesidis*), paragraph not found; *l. ſi quid. D. de offi. proconſ.* — D. 1.16, fragment not found; *l. iij. D.* — title lost in the print.
- annot-082 {o}: *l. cum damnum. D. de pœn.* — fragment not found.
- annot-082 {p} (p120): *l. ſi quis filio. § irritum. D. de iniuſt. teſt.* — cf. D. 28.3.6; incipit/paragraph not matched.
- annot-082 {q}: *c. delicto. de ſent. excom. au vj.* — VI 5.11, chapter not found.
- annot-082 {t} (p120): *l. penul. C. ad Orficia.* — C. 6.57, penultimate law; bearing unverified.
- annot-082 {z}, {b}: X 2.1.4 (*At si clerici*) and X 2.1.10 (*Cum non ab homine*) — identified; Panormitanus n. 38 not checked.
- annot-082 {a} (p121): *c. quid in omnibus. xxx. q. v.* — chapter not located; the pseudo-Clementine epistle to James is the source Coras names.
- annot-082 {c} (p121): *c. tua. de pœ.* — X 5.37, chapter not located.
- annot-082 {d}: Jean Faure on C. 8.52.1 and Benedicti's *Repetitio* on c. *Raynutius* at *Cuidam Petro*, n. 62 — the works identified; not checked.
- annot-082 {g} (p121): *c. reos xxiij. q. v.* — C. 23 q. 5, chapter *Reos* not located.
- annot-082 {k} (p122): *l. ſed licet. D. de offi. preſi.* — D. 1.18, fragment not found.
- annot-082 {m} (p122): *l. que adulterium. C. de adult.* — C. 9.9, fragment not found; Papon, *Des adulteres*, arrêt 4 — identified by content, not checked.
- annot-082 {o} (p122): *l. venia. C. de in ius voc.* — C. 2.2, fragment not found; *Gloſe au c. ſicut. de conſec. diſt. i.* — the chapter *Sicut* in De cons. D. 1 not located.
- annot-082 {q} (p123): *c. ita ne. xxxij. q. v.* — C. 32 q. 5, chapter not located.
- annot-082 {t} (p123): *c. in lectũ. c. ſi virgo. xxxiij/xxxiiij. q. i* — chapters not located in either *causa*; see the `⟨alt⟩` above.
- annot-082 {u} (p123): *l. ſi vxor. § ſi quis plan[e]. l. vim paſſam D. de adult.* — D. 48.5.14(13) by content, divisions unverified; C. 9.9.20 (*Foedissimam*) identified.
- annot-082 {x} (p123): *l. ſi adulterium. § Diui fratres. D. de adulte.* — D. 48.5.39(38), paragraph unverified.

Identified with confidence and worth noting as such: D. 48.10.13 pr. (*Falsi nominis vel cognominis adseveratio poena falsi coercetur*, annot-080 {x}); D. 48.10.27.2 (annot-081 {b}); D. 1.14.3 (annot-081 {d}); D. 48.10.1.13 (annot-081 orphan *b*); Nov. 134.13 and the *authentica* *Bona damnatorum* (annot-081 {c}); D. 48.18.5 (annot-082 {d}); C. 2.4.18 (annot-082 {f}); Inst. 4.18.4 and 4.18.5 (annot-082 {g}, {i}, {l}); D. 48.8.3.5 (annot-082 {m}); Nov. 134.10 and the *authentica* *Sed hodie* (annot-082 {ſ}, {l}); D. 23.2.43.12 and 43.5 (annot-082 {t}, {p}); Gellius 10.23 (annot-082 {x}); X 5.18.3 (annot-082 {n}); C. 9.9.20 (annot-082 {u}); D. 47.18.1.1 (annot-081 {f}, annot-082 {y}); Herodotus 3.61–79, Justin 1.9, 38.1–2, Josephus *Ant.* 17.324–338, Appian *Syr.* 67–68, Plutarch *Lyc.* 15 and *Rom.* 22, Propertius 3.25.5–6, Ovid *Ars* 3.291–292, Euripides *Medea* 928, Plautus *Truc.* 178–179.

## Marker and margin problems

- texte-76 / annot-078: the margin of p106 carries a note keyed *a2* (Propertius) with no body marker; the transcription attaches it to texte-76 as an orphan, but it is the *a* of Annotation LXXIX, whose first words (*Ceſtuy diſoit, auec Properce*) are on p106. Listed under texte-76 as an orphan with the cross-reference; annot-078's alphabet therefore starts at *b*. The reviewer may prefer to move the note to annot-078 in the transcription.
- annot-080: the transcription keys p110's markers *a2*–*f2* (p110 already carries LXXX's *a*–*d*); the print has plain letters. The body marker `{x}.⟨alt:{k}.⟩` on p111 and the margin's key *x*: see the `⟨alt⟩` table. The margin of p110 carries an unkeyed note (*ſiue adulterium. x. diſt.*) with no marker; listed without a key. Worth a look at the page image: there may be a marker in the body the transcription missed.
- annot-081: the margin of p118 carries a note keyed *b* (*l j. § fin. D ad l. Cornel. de falſ.*) with no body marker; by content it serves the parenthesis on the ordinary penalty of forgery, between {a} and {c}; listed in that place. Letters *a*, *c*, *d*, *e*, *f* occur twice (p111 and p118); the Notes distinguish by page.
- annot-082: the note keyed *p* is printed in the margin of p123 while its body marker {p} stands on p122, two words before the page break (*deteſtee {p}. car l'hõme doit pluſ⟦p123⟧goſt*); the transcription marks it an orphan. Listed as {p} (p122) with the remark. Letters *a*–*p*, *r*–*u*, *x*, *y* occur twice; distinguished by page.
- annot-080 and annot-082 fail `check_markers.py` only because of markers embedded in `⟨alt:…⟩` text (see the head of this report).
- annot-081: the print leaves two parentheses unclosed (*(ou toutesfois ne fut onc poſſible …*; *(par ce qu'il reſſembloit du tout Antiochus*); the English closes each where the sense requires. annot-082: the parenthesis closed after *quatrieme Eueſque de Rome)* is never opened; opened at *(or, according to others*.
- annot-081: *mille 225* is printed so (words and figures mixed); rendered "one thousand 225".

## Glossary additions

The coordinator's instruction for this run was to write only the section files, the alt-choices file and this report, so these rows have **not** been appended to `docs/case-file.md` §9; they are given in the table's format for the coordinator to merge.

| French | English rendering | Note |
|---|---|---|
| reciter la contenance | to recount the countenance | (texte-76) |
| grans pleurs & gemiſſemens | great weeping and lamentations | (texte-76) |
| auſtere, & farouche | austere and fierce | Of Martin Guerre's countenance. (texte-76) |
| nées pour plourer | born to weep | Coras on women, after Euripides. (annot-078) |
| feintes, ſimulees, & pleines d'hypocriſie | feigned, simulated and full of hypocrisy | Of tears. (annot-078) |
| fiel / amertume | gall / bitterness | Plautus's honey and gall. (annot-078) |
| le diuertir de ſon auſterité | to turn him from his austerity | (texte-77) |
| le procez du tout inſtruit | the trial fully instructed | The *instruction* of the case complete. (texte-77) |
| iceluy veu | the same having been reviewed | The *visite du procès* of §5. (texte-77) |
| à grande & meure deliberation | upon great and mature deliberation | (texte-77) |
| ARREST. (display head) | DECISION. | Per §5. (texte-77) |
| VEV le procés fait par … à | SEEN the trial conducted by … against | The *Vu* opening a French judgment. (texte-77) |
| Dit a eſté que | It has been declared that | The formula introducing the disposition. (texte-77) |
| a mis, & met … au neant | has set aside and sets aside … as null | The §5 formula, so phrased that *ce dont a eſté appelé, au neant* can be quoted alone. (texte-77, texte-78) |
| punition & reparation | punishment and reparation | (texte-77) |
| autres cas … reſultans dudit procez | other offences … resulting from the said trial | (texte-77) |
| torche de cire ardente | burning wax taper | As §2. (texte-77) |
| faire les tours par les rues & carrefours accouſtumez | to go the rounds through the accustomed streets and crossroads | The execution procession, §5 s.v. *criée*. (texte-77) |
| potence … dreſſée | gallows … erected | (texte-77) |
| detraicts les frais de Iuſtice | the costs of Justice deducted | (texte-77) |
| mis hors de procez, & inſtance | dismissed from the case and suit | Per §5. (texte-77) |
| renuoyer (au Iuge) | to remand (to the Judge) | For execution of the decision. (texte-77) |
| ſelon ſa forme & teneur | according to its form and tenor | (texte-77) |
| Prononcé iudicialement | Pronounced judicially | (texte-77) |
| EXPOSITION DES Paroles de l'arreſt | EXPOSITION OF THE Words of the decision | The display head of the last part of the book. (texte-77) |
| perdre la teſte, & eſtre mis en quatre quartiers | to lose his head and be cut into four quarters | The Rieux judgment. (annot-079) |
| caſſer (vne ſentence) | to quash | Of the Court on a lower judgment. (annot-079) |
| proditeur / prodition | traitor / treachery | Cf. *proditoirement*. (annot-079, annot-082) |
| comblé en toute eſpece de vices | heaped up with every kind of vice | (annot-079) |
| ceux de baſſe condition | those of low condition | As against *les nobles*. (annot-079) |
| decapitez / pendus | beheaded / hanged | The noble and the common death. (annot-079) |
| fourches | gibbet-forks | The *fourches patibulaires*. (annot-079) |
| patent | patent | Of a crime. (annot-080) |
| changer de nom / de ſurnom / d'armoiries | to change one's name / surname / arms | (annot-080) |
| à leur creation | at their creation | Of popes, their election. (annot-080) |
| Bocca di porco | Bocca di porco (Pig's Mouth) | Italian kept, Coras's gloss translated. (annot-080) |
| par imitation de vertu | by imitation of virtue | Augustine on John as Elijah. (annot-080) |
| peine de faux | penalty of forgery | Cf. *crime de faux*. (annot-080) |
| tendre des laçons pour appaſter | to lay snares to bait | (annot-081) |
| ſuppoſition notable | notable substitution | (annot-081) |
| atroce, cruelle, & exemplaire punition | atrocious, cruel and exemplary punishment | (annot-081) |
| fort ſobrement | sparingly | Of the laws' silence. (annot-081) |
| enſeignes & armoiries defenduës | forbidden insignia and arms | Modestinus's *illicitis insignibus*. (annot-081) |
| ſuppoſer fauſſes lettres du Prince | to forge false letters of the Prince | *Falso diplomate*. (annot-081) |
| ſous ce manteau | under this cloak | (annot-081) |
| Barbare Philippe | Barbarius Philippus | The Digest's runaway slave praetor. (annot-081) |
| le Iuriſconſulte n'en ouure pas vne ſeule parole | the Jurisconsult does not open his mouth with a single word | (annot-081) |
| à ieu | as a game | Of the ancients' indulgence. (annot-081) |
| loyer & recompenſe / loyer & retribution | reward and recompense | (annot-081) |
| ſ'ingerer aux dignitez | to thrust oneself into dignities | (annot-081) |
| reuoquer en doute | to call in doubt | (annot-081) |
| occuper & enuahir (les biens) | to occupy and invade | Cf. *s'emparer de*. (annot-081) |
| ayeul pretendu | pretended grandfather | *Pretendu*, claimed. (annot-081) |
| de fort bonne grace | of very good grace | Of a person's bearing. (annot-081) |
| luy tenir la main | to lend him a hand | Of the impostor's coach. (annot-081) |
| colorer l'impoſture | to colour the imposture | (annot-081) |
| en apparat royal | in royal state | (annot-081) |
| anguille ſous roche | eel under the rock | Something hidden. (annot-081) |
| toucher au marteau de ſa conſcience | to strike the hammer of his conscience | (annot-081) |
| tirer les vers du nez | to draw the worms from his nose [to worm the truth out] | Glossed in brackets on first use. (annot-081) |
| ourdir la toile | to weave the web | Cf. *ourdir & tramer*. (annot-081) |
| confuz labyrinthe de vices | confused labyrinth of vices | (annot-081) |
| roy baſtard | bastard king | (annot-081) |
| gentil (ironic) | fine | *Ce gentil Prompalus*; *nos gentils Canoniſtes*. (annot-081, annot-082) |
| harpeur | harper | (annot-081) |
| par la diſgrace des vents | by the disfavour of the winds | (annot-081) |
| branſler | to totter | Of a province in revolt. (annot-081) |
| charme naturel | natural charm | (annot-081) |
| cabaret | tavern | (annot-081, annot-082) |
| ſurprins de la mort | overtaken by death | (annot-081) |
| Ieanne l'Angloiſe | Joan the Englishwoman | Pope Joan. (annot-081) |
| lire (in the schools) | to lecture | (annot-081) |
| genitoires | genitals | (annot-081) |
| puni capitalement / Capitalement | punished capitally / Capitally | The word Coras glosses as civil or natural death. (annot-081) |
| dernier ſupplice | the last punishment | Death. (annot-081) |
| guetteurs de mariages d'autruy | lurkers after other men's marriages | (annot-082) |
| impudicitez | immodesties | (annot-082) |
| peine du glaiue (naturelle / ciuille) | penalty of the sword (natural / civil) | *Poena gladii*; kept literal throughout since Coras divides it. (annot-082) |
| ſainctement | holily | Of a judgment or a saying. (annot-082) |
| la loy Iulie des adulteres / la loy Cornelie | the Julian law on adulterers / the Cornelian law | (annot-082) |
| reſtrainte & moderée | restrained and moderated | Of a penalty. (annot-082) |
| fueilleter nos liures de Droict | to leaf through our books of Law | (annot-082) |
| faire diſſection de membres | to make dissection of members | (annot-082) |
| cenſures eccleſiaſtiques | ecclesiastical censures | (annot-082) |
| chaſtiee | chastised | Whipped; the adulteress of Nov. 134. (annot-082) |
| faculté (de recouurer) | faculty | The legal power. (annot-082) |
| gemir ſon peſché | to bewail her sin | (annot-082) |
| du bout du doigt | with the tip of her finger | Cato in Gellius. (annot-082) |
| exauthoré de ſes ordres | stripped of his orders | *Exauctoratio*. (annot-082) |
| actes ingenieux, & haut louëz | ingenious and highly praised acts | Faure and Benedicti on adultery in France. (annot-082) |
| noſtre compagnie | our company | The Court. (annot-082) |
| toucher au doigt | to touch with the finger | To make palpable. (annot-082) |
| diſſimuler (vn crime) | to wink at | (annot-082) |
| membres de putain | members of a harlot | 1 Corinthians 6:15. (annot-082) |
| conniuer à | to connive at | (annot-082) |
| qualifié (d'vne prodition) | qualified (by a treachery) | The aggravating quality. (annot-082) |
| ſeruiteur de cabaret | tavern servant | The Paris case of 1551. (annot-082) |
| aſſeruis, & obligez | enslaved and bound | By a vow. (annot-082) |
| alteree & charnelle volupté | thirsty and carnal pleasure | (annot-082) |
| la raiſon biẽ froide | the reason very cold | (annot-082) |
| ſimplicité, neceſſité, ou tentation | simplicity, necessity or temptation | The excuses Coras rejects. (annot-082) |
| trebucher | to stumble | Of a vowed cleric. (annot-082) |
| polution | pollution | (annot-082) |
| la tendreté d'vn ieune aage | the tenderness of a young age | (annot-082) |
| arbitre d'vn bon, ſainct, & equitable iuge | discretion of a good, holy and equitable judge | Cf. *arbitre du Juge*. (annot-082) |

## For the reviewer

1. **`check_markers.py` and `⟨alt⟩`-embedded markers.** annot-080 and annot-082 are reported failed only because the French still contains `⟨alt:{k}.⟩`, `⟨alt:{a}:⟩`, `⟨alt:{l}:⟩`, `⟨alt:{r}.⟩`. Apply the alt-choices file and refinalize p111, p119, p123, or strip alt text in the checker; the English then passes as written. Any other batch whose pages carry such alts will hit the same thing.
2. **texte-82, *Rapt*** → "Rape". The glossary's note says *rapt* covers abduction as well; Annotation LXXXIV (annot-083) will say what Coras means by it here (there was no abduction in the facts; the sense is probably the seduction/*rapt de séduction* of a married woman). The one-word heading may need to become "Rape [*rapt*]" or "Abduction" once that annotation is translated.
3. **texte-77, the *arrest*.** Rendered to agree with the case file's §2 wording and the §5 formulae; *VEV* → "SEEN" and *Dit a eſté que* → "It has been declared that" are new fixed renderings for the dispositive formulae — confirm, since every later fragment of the decision will quote them. *Au neant* → "as null" was chosen so that texte-78 can quote the half-formula; "at naught" is the alternative.
4. **annot-079, missing predicate.** "Too honourable" is left out of the text and only indicated in the bracket; the reviewer may prefer to supply it in square brackets.
5. **annot-080, *bien qu'il n'en fut point*.** Coras says Paul was no Roman citizen; Acts 22:25–28 says he was. Translated as written; a note in the apparatus may be wanted.
6. **annot-081, Coras's history.** Several slips are translated as written and listed in the section's headnote (Herod Antipas for Herod the Great; Smerdis "king of the Assyrians"; Leo III for Leo IV; Otho for Galba; "the Emperor Augustus" for the Code's constitution; Louis VIII "her uncle"; *Prompalus* for Alexander Balas; Justin's *xxxij* for xxxviij). None has been corrected in the text.
7. **annot-081, *dans la maiſon executé à mort*** → "in the prison-house executed to death": Valerius Maximus has *in carcere*; "prison-" is supplied to the print's *maiſon*. Change to "in the house" if the house wants strict literalism.
8. **annot-081, idioms.** *Anguille ſous roche* is left as "eel under the rock"; *tirer les vers du nez* is glossed in brackets. Decide a house line on 16th-century idioms (literal + bracket, as here, or English equivalent).
9. **annot-082, the three Decretum alts** (p121-m5l4-1 Thessalonians iiij; p121-m7l0-1 xxiij; p123-m4l4-1 xxxiiij) were decided on the structure of the works cited, not on the page image; the last is an inference and the first corrects a chapter where the book name is also wrong.
10. **annot-082, *ſemblẽt peu, ou point meriter quelque excuſe*.** The sentence as printed contradicts the opinion it reports; left literal with `[unclear]`. The 1565/1596 editions might read differently and are worth a check.
11. **Numbering.** From annot-077 (print LXXVIII) to annot-082 (print LXXXIII) the print's annotation numbers run one ahead of the ids, and the print then heads annot-083 LXXXIII as well. Annotation XII's cross-references (*l'annotation lxxviij* for change of name, *lxxxj* for substitution of persons) match neither our ids nor this print's heads exactly, and probably follow the 1561 numbering; the apparatus should say which numbering is cited.
12. **Orphan notes.** Three orphans in this batch (texte-76 *a2*, annot-081 *b* on p118, annot-082 *p*) plus one unkeyed note (annot-080 *ſiue adulterium*); the first and last would repay a look at the page images.

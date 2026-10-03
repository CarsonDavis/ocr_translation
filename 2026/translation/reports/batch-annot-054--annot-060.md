# Report: batch annot-054 – texte-57

Twelve sections, p074–p081: the record's answers to the points in the prisoner's favour (texte-52 to texte-57: the dissimilarity to Sanxi, the Basque language, du Tilh's dissolute youth, the affirming witnesses, the retracting witnesses, the marks and scars) with Coras's Annotations LIV–LIX and, inside the stitched section `texte-57`, the whole of Annotation LX. `check_markers.py` on each of the twelve: 12 sections ok, 0 failed. Written: the twelve section files under `translation/sections/`, this report, and `translation/alt-choices/batch-annot-054--texte-57.json` (25 entries). `docs/case-file.md` was not edited; the glossary rows are in the table below for the coordinator to merge.

## Per section

| Section | Pages | French | English | Notes |
|---|---|---|---|---|
| annot-054 | p074–p075 | 263 | 282 | No marginalia. Leo of Byzantium, Dionysius of Heraclea, Louis the Fat; sources given in the headnote. Page break between *ce* and *peuple*. |
| texte-52 | p075 | 84 | 85 | No marginalia. One `⟨alt⟩` (either). |
| annot-055 | p075–p076 | 201 | 221 | Markers a–b, both served. Page break inside *ar.|gument*. One `⟨alt⟩` in the margin (either). |
| texte-53 | p076 | 45 | 56 | No marginalia. Ends without a stop in the French, kept. One `⟨alt⟩` (B). |
| annot-056 | p076 | 116 | 134 | Marker a, served. Latin tag *quasi nescius fari* translated with the Latin in parentheses. |
| texte-54 | p076–p077 | 39 | 44 | No marginalia. Page break between *du Tilh* and *ay eſté*. |
| annot-057 | p077 | 55 | 55 | Marker a served; note *b* is an orphan (see below). The French breaks off without a stop. |
| texte-55 | p077–p078 | 115 | 123 | No marginalia. Page break inside *rei|gle*. |
| annot-058 | p078 | 292 | 316 | Markers a, b, d–l (no *j*) served; note *c* is an orphan (see below). Three `⟨alt⟩` (all B). Coras dates the writing: "this first day of October 1560". |
| texte-56 | p078–p079 | 39 | 42 | No marginalia. Page break between *principal* and *point*. One `⟨alt⟩` (B). |
| annot-059 | p079 | 237 | 247 | Markers a–f, h, i, k served; the margin has *g* with no marker and no *h* (see below). Four `⟨alt⟩` (three B, one either). |
| texte-57 | p079–p081 | 606 | 667 | **Contains Annotation LX** after the record's paragraph; see "Structural problem". Markers a–f, h–ſ served; note *g* is an orphan. Fourteen `⟨alt⟩`. |

Totals: French 2092 words, English 2272 words (counts exclude the page, letter and `⟨alt⟩` markers; the French counted at reading A).

## Structural problem: Annotation LX inside `texte-57`

The print heads Annotation LX on p080 as *ANNNT. LX.* (a wrong sort for *ANNOTAT. LX.*). The stitcher did not recognise this as a heading, so `texte-57` in `text/sections.json` runs from the TEXTE head on p079 through the record's paragraph on the marks and scars **and the whole of Annotation LX** to the TEXTE head on p081. Consequences:

1. The file `translation/sections/texte-57.md` contains both, with ANNOTATION LX translated on its own line where the print has the heading, as the prompt directs for headings inside a section.
2. There is no `annot-060` for Annotation LX. The section the stitcher calls `annot-060` (p082) is headed *ANNOTAT. LXI.* in the print, so **every `annot-NNN` id from `annot-060` onward is one behind the printed number** (`annot-060` = LXI, `annot-061` = LXII, and so on, unless a later misnumbering compensates). The coordinator should decide whether to re-cut the sections (splitting `texte-57` at *ANNNT. LX.*) and renumber, or to leave the ids and record the offset. If `texte-57` is re-cut, this file splits cleanly at the blank line before "ANNOTATION LX"; the Notes headnote and the entries a–ſ all belong to the annotation.

## `⟨alt⟩` choices

The same choices, machine-readably, are in `translation/alt-choices/batch-annot-054--texte-57.json`. None of the markers in this batch is `⟨alt?:⟩`; all are vouched for on one side or the other.

- texte-52, p075 (`p075-b2l0-1`): *diſsimili-⟨alt:diſſimili-⟩tude* — **either**. Both are *dissimilitude*; only the *ſs*/*ſſ* sort differs.
- texte-53, p076 (`p076-b2l2-1`): *reſu[?]te.⟨alt:reſulte.⟩* — **B**, *reſulte* ("it results from the inquiries"). A leaves a letter unread; no other word fits.
- annot-055, p076, note *b* (`p076-m0l1-1`): *item quacun-⟨alt:itam quæcun-⟩que* — **either**. Each reading has one wrong letter; both resolve to *Item quaecumque* (D. 6.1.23.3).
- annot-058, p078 (`p078-b2l13-1`): *eſçolier,⟨alt:eſcolier,⟩* — **B**, *eſcolier* ("scholar"), the ordinary form; *eſçolier* is no form of the word.
- annot-058, p078 (`p078-b2l29-1`): *virenr⟨alt:virent⟩* — **B**, *virent* ("they saw him that day at Paris"); *virenr* is a wrong sort.
- annot-058, p078, note *d* (`p078-m3l1-1`): *propoſui cti.⟨alt:ſti.⟩* — **B**, *propoſuiſti*, the incipit of the chapter cited (c. *Proposuisti*).
- texte-56, p079 (`p079-b0l3-1`): *depattis⟨alt:departis⟩* — **B**, *departis* ("have departed from it"), the record's word for a retraction.
- annot-059, p079 (`p079-b2l3-1`): *oet⟨alt:ont⟩* — **B**, *ont varié*; *oet* is no word.
- annot-059, p079 (`p079-b2l7-1`): *reicté⟨alt:reiecté⟩* — **B**, *reiecté* ("rejected").
- texte-57, p079 (`p079-b4l1-1`): *emprainctes⟨alt:empraintes⟩* — **either**. Both are *empreintes* ("imprinted").
- annot-059, p079, note *e* (`p079-m4l3-1`): *de [?]ſt.⟨alt:teſt.⟩* — **B**, *de teſt.*, the title *De testibus* of Novel 90.
- annot-059, p079, note *i* (`p079-m7l4-1`): *deſ⟨alt:deſ-⟩ſus* — **either**. *Deſſus* broken at the line end, with or without the hyphen.
- texte-57, p080 (`p080-b2l19-1`): *meſchancetè⟨alt:meſchanceté⟩* — **either**. The accent alone differs.
- texte-57, p080, note *e* (`p080-m4l2-1`): *c.⟨alt:[?].⟩ ij.* — **A**, *c.* = *capitulum*, "the second chapter *Qualiter*" (X 5.1.24).
- texte-57, p080, note *e* (`p080-m4l3-1`): *Balde⟨alt:[?]alde⟩* — **A**, Baldus.
- texte-57, p080, note *f* (`p080-m5l1-1`, `p080-m5l1-2`): *en⟨alt:e[?]⟩ ſes deſi-⟨alt:desi-⟩ſions* — **A** for both (one line, one choice): reading A is complete and legible; both give *en ſes deciſions* (Gui Pape's *Decisiones*).
- texte-57, p080, note *i* (`p080-m8l1-1`): *cõſil⟨alt:cõſeil⟩* — **either**. Both are *conseil* (*consilium*); note *o* on p081 spells it *cõſeil*, which slightly favours B, but the print may well have the misprint and the translation is the same.
- texte-57, p080, note *k* (`p080-m9l0-1`): *nihilomi⟨alt:nibilomi⟩nus* — **A**, *Nihilominus*, the chapter's incipit, repeated in note *ſ*.
- texte-57, p080, note *k* (`p080-m9l1-1`): *auſſi⟨alt:auſsi⟩* — **either**. The same word.
- texte-57, p080, note *k* (`p080-m9l2-1`): *iij.⟨alt:ij.⟩ q. ix.* — **A**, *Causa* 3 *quaestio* 9; *Causa* 2 of the Decretum has only eight *quaestiones*.
- texte-57, p081 (`p081-b0l2-1`): *e.⟨alt:e-⟩ſtre* — **B**, *eſtre* broken at the line end with a hyphen; a point there is no sense.
- texte-57, p081 (`p081-b0l11-1`): *prudence,⟨alt:prudence;⟩* — **A**, the comma; the participle *traitez & diffiniz* continues the clause. Translation unaffected.
- texte-57, p081, note *n* (`p081-m2l2-1`): *c.⟨alt:e.⟩ vbi periculum* — **A**, *c.* = *capitulum* (VI 1.6.3).
- texte-57, p081, note *ſ* (`p081-m7l1-1`): *preallé⟨alt:preallé-⟩gué* — **either**. *Preallegué* broken at the line end.

## `[unclear]` passages

No passage is marked `[unclear]`. One defective sentence and four bracketed supplements:

- **texte-57 (Annotation LX), p080, at {h}** — the print reads: *Dont pluſieurs ont ie ne ſçay comment penſé, qu'à cõuaincre vn homme heretique ſuffiſent deux teſmoins, bien qne l'vn d'eux depoſent d'vne eſpece d'hereſies, diſent ils, combien que ſoyent par diuers noms deſignées, ſont neantmoins entreliees, & coniontes en meſchancetè.* The second limb of the concession (that the other witness deposes to another kind of heresy) and the connective before *hereſies* (*car les hereſies*, or the like) are missing — a compositor's eye-skip, most likely. Translated literally with "[and the other to another]" supplied in brackets and a colon at the break: "even though one of them should depose to one kind of heresy [and the other to another]: heresies, they say, although they are designated by divers names, are nevertheless interlinked and conjoined in wickedness". The sense is not in doubt (it is the opinion Panormitanus rejects at {i}), so `[unclear]` was not used; the reviewer may prefer to collate the 1561 or 1618 edition for the missing words.
- annot-054: "we two can well [lie] in a little bed" — *nous pouuons biens tous deux dans vn petit lict*; the verb is elided in the French.
- annot-058: "one must understand the decision of Accursius {g} [to hold] when …" — *on doit entendre la deciſion d'Accurſe, quand*; the verb is elided.
- texte-57 (Annotation LX): "singular witnesses … may [prove] sufficiently" — *peuuent ſuffiſamment*; and "received as [those] of several" — *receuës, comme de pluſieurs*.
- annot-057: *la perſonne qui veut punir* is translated as *qu'on veut punir*, "the person whom one wishes to punish", the only reading the sense admits; noted in the file.

## Citations not identified, or identified only in part

Identified with confidence (for the reviewer's information, not for checking): annot-057 {b} C. 9.47.22 (*Sancimus*); annot-058 {c} C. 4.19.23 (*Actor*) and X 2.19.11 (*Quoniam contra*); annot-058 {k} X 2.19.4 (*Ex litteris*); annot-059 {a} D. 22.5.2 and D. 22.5.16 (*Qui falso*); texte-57/LX {b} C. 4.20.9 (*Iurisiurandi*), {d} Cicero *Pro Fonteio*, {e} X 5.1.24 (the second *Qualiter et quando*), {g} C. 1.5.5 (*Arriani*), {l} D. 1.3.32 (*De quibus*), {m} C. 8.52.2, {n} D. 48.4.7 (*Famosi*) and VI 1.6.3 (*Ubi periculum*), {r} Clem. 5.3.1 (*Multorum querela*).

Not identified, or only in part:

- annot-055 {b}: *l. in rem. § item quacunque. D. de re. vend.* — cf. D. 6.1.23.3 (Paul), fair confidence on the fragment, paragraph unverified; *l. ſed cũ patrono. D. de bono. poſſeſ.* — D. 37.1, fragment **not identified**.
- annot-056 {a}: *l. ſi infanti. C. de iur. de li bro.* — C. 6.30 (*De iure deliberandi*), cf. C. 6.30.18, number unverified; *c. nullus de tẽpor. ordinand. au vj.* — VI 1.9, chapter number unverified.
- annot-057 {a}: *l. j. § item illud. D. ad Silania* — D. 29.5.1, paragraph not identified; *l. j. C. vbi cauſ. fiſcæ.* — C. 3.26.1, bearing unverified.
- annot-058 {a}, {g}: Accursius at *l. diem § ſi plures D. de arbitr.* — the same gloss as annot-038 {a} and annot-052 {a2}; D. 4.8.21 or 4.8.27, unverified, as the earlier batches had it.
- annot-058 {b}: *Ariſtote au iij. de la metaphyſique* — the passage on affirmation and negation is not located in book III; book number given as printed.
- annot-058 {c} (orphan): *c. ſuper hoc. de renuncia.* — X 1.9, chapter number unverified.
- annot-058 {d}: *c. propoſuiſti … de pro.* — a c. *Proposuisti* is **not identified** in X 2.19 (*De probationibus*); the title as printed.
- annot-058 {e}: *l. penultieme. C. de profeſſo. & med. lib. xij.* — C. 10.53 in the modern division (the print says book 12); penultimate law cf. C. 10.53.10, bearing on the number of examining doctors unverified.
- annot-058 {f}: *c. in noſtra. de teſt.* — X 2.20, chapter number unverified.
- annot-058 {h}: *l. optimam. C. de cont. ſtip.* — C. 8.37, cf. C. 8.37.14, unverified; *c. tertio loco. de præſump.* — X 2.23, chapter number unverified.
- annot-058 {i}: *l. In illa. D. de verbo. oblig.* — D. 45.1, fragment **not identified**.
- annot-058 {k}: *c. inter de fi. inſtru.* — X 2.22, chapter number unverified (perhaps c. *Inter dilectos*).
- annot-058 {l}: Accursius on *l. j. D. de Itin. actuque pri.* — D. 43.19.1; bearing on witnesses unverified.
- annot-059 {a}: *l. eos. C. de falſ.* — C. 9.22, fragment **not identified**.
- annot-059 {b}: *l. penultieme. § j. D. quand. dies l. ced.* — D. 36.2, penultimate law, bearing unverified.
- annot-059 {c}: *c. teſtimonium. de teſt* — X 2.20, chapter number unverified; Accursius on *l. Lucius. D. de iis qui not. infa.* — D. 3.2, fragment **not identified**; *l. ni. § lege Iul. D. de teſt.* — read as *l. iij.*, D. 22.5.3.5.
- annot-059 {d}: *c. ſicut de teſt.* — X 2.20, chapter number unverified.
- annot-059 {e}: *c. Preterea de teſt.* — X 2.20, chapter number unverified (the chapter is cited at {h}/{g}, {i} and {k} too, and is a real marginal noted in case file §6.3); Accursius on § *Quia vero* of Novel 90 (*De testibus*, coll. 7) — paragraph unverified.
- annot-059 {f}: *c. accuſatus. § licet. de hæreti. au vj.* — cf. VI 5.2.8, paragraph unverified.
- annot-059 {i}: *Gloſe au c. au ſit. de appella* — X 2.28, the chapter printed *au ſit* is **not identified** (perhaps c. *Cum sit*).
- texte-57/LX {a}: *l. ob carnem § ſi D. de teſti.* — read as *Ob carmen*, cf. D. 22.5.21, paragraph unverified; *c. bonæ. l. j. de elect.* — X 1.6, c. *Bonae [memoriae]*, the first so named, chapter number unverified; *c. licet ex quadam* — cf. X 2.20.47, as the earlier batches.
- texte-57/LX {c}: *l. maritus. D. de quæſtio* — cf. D. 48.18.17, unverified.
- texte-57/LX {e}: *Balde en la l. j. C. qui num. tutel.* — read as C. 5.69.1 (*Qui numero tutelarum se excusant*); title probable, bearing unverified.
- texte-57/LX {f}: Gui Pape, *Decisiones*, q. 154 — not checked against an edition.
- texte-57/LX {h}: *l. quicũque. verſi Idcirco. C. de hæret.* — cf. C. 1.5.4, number unverified; *c. Pan. au meſme titre du vj* — **unidentified**: no chapter of VI 5.2 begins so; possibly a reference to Panormitanus.
- texte-57/LX {i}, {o}: Panormitanus, *Consilia*, I.42 — not checked against an edition.
- texte-57/LX {k}, {ſ}: *c. nihilominus. iij. q. ix.* — C. 3 q. 9 of the Decretum, canon number unverified.
- texte-57/LX {p}: *c. tam literis. c. veniens. de teſ.* — X 2.20, chapter numbers unverified; Bohier, *Decisiones Burdegalenses*, decision 342 — not checked.

## Marker and margin problems

- **annot-057**: the margin has a note *b* with no marker in the body; the French breaks off at *coulpable du faict* without a stop, and the marker and the point were probably lost together at the line end. Listed as an orphan at the second head of proof, which it fits exactly (C. 9.47.22, *ibi esse poenam, ubi et noxa est*). Worth a look at the page image of p077 for the *b*.
- **annot-058**: note *c* has no marker in the body; its content (C. 4.19.23, *negantis factum probatio nulla sit*) serves the clause "it is almost impossible to prove a negation" between {b} and {d}. Listed as an orphan at that place. Check p078 for the *c*.
- **annot-059**: the body has {h} and no {g}; the margin has a note *g* and no *h*. The note *g* (Panormitanus on c. *Praeterea*, the just causes for re-examining a witness) answers {h} exactly; *g* is a wrong sort or a slip for *h*. Listed as the orphan *g* at {h}'s place, with an entry for {h} pointing to it. The transcription might key the note *h*; the coordinator's call.
- **texte-57 / Annotation LX**: note *g* (Butrigarius on C. 1.5.5) has no marker in the body; by content it belongs to the sentence on proving heresy by two singular witnesses, between {f} and {h}. Listed as an orphan at that place. Check p080 for the *g*.
- **texte-57**: the heading *ANNNT. LX.* and the un-cut annotation; see "Structural problem" above.
- Wrong sorts in the margins read through and noted in the files: *hæred.* for *hæret.* (LX, note *g*); *haré.* for *hære.* (LX, note *r*); *ob carnem* for *ob carmen* (LX, note *a*); *fameſi* for *famoſi* (LX, notes *n*, *q*); *l. ni.* for *l. iij.* (annot-059, note *c*); *nou[?]lles* = *nouuelles* (annot-059, note *e*); *de li bro* = *de li[be]r[ando]* (annot-056); the italic *P* for the paragraph sign throughout.

## Glossary additions

Not appended to `docs/case-file.md` §9 (the coordinator merges them); given in the table's format.

| French | English rendering | Note |
|---|---|---|
| greſles, linges, & dolietz | slender, thin and delicate | *Linge*, the old adjective "thin, slight"; *doliet* = *douillet*. (annot-054) |
| gros, gras, & importuns | stout, fat and cumbersome | *Importun* in its physical sense. (annot-054) |
| recouurer à (= recourir à) | to have recourse to | (annot-054) |
| mediocre aage | middle age | (annot-054) |
| panſard & ventru | paunchy and pot-bellied | (annot-054) |
| ſangſues | leeches | (annot-054) |
| Leon Bizantin / Denis Heracleot / Loys le Gros | Leo of Byzantium / Dionysius of Heraclea / Louis the Fat | (annot-054) |
| iugemens par ſemblance | judgments by resemblance | (texte-52) |
| faire la conference (à) | to make the comparison (to) | Cf. *parangonner / conferer* → to compare. (texte-52) |
| proportion & analogie | proportion and analogy | The mathematicians' *analogia*. (annot-055) |
| argument de l'vn à l'autre | argument from the one to the other | The *argumentum a simili*. (annot-055) |
| nos interpretes | our interpreters | As *Interpretes en droict*. (annot-055, annot-050) |
| ſympathies | sympathies | (annot-055) |
| la langue de Baſcouz | the language of the Basques | Cf. *pays des Bascouz* → the Basque country. (texte-53) |
| la verité du faict apporte la reſponſe | the truth of the fact supplies the answer | (texte-53) |
| INFANS, quaſi neſcius fari | INFANS, as it were unable to speak (*quasi nescius fari*) | Coras's own gloss follows. (annot-056) |
| bas aage | tender age | (annot-056) |
| deſnouër ſa langue | to untie his tongue | (annot-056) |
| gazouiller | to babble | Of infants; the bird figure, cf. *ramage*. (annot-056) |
| n'y fait rien (auſſi) | is nothing to the purpose (either) | The record's formula for an argument that does not avail. (texte-54; also texte-50, texte-51) |
| adonné à toute eſpece de meſchancetez | given to every kind of wickedness | (texte-54) |
| venir à condemnation | to come to condemnation | (annot-057) |
| commis & perpetré | committed and perpetrated | (annot-057) |
| delict | offence | *Delictum*. (annot-057) |
| coulpable du faict | culpable of the deed | (annot-057) |
| raiſons deduites | reasons deduced | Cf. *desduire*. (texte-55) |
| affermer / aſſeurer / nier | to affirm / to assure / to deny | The three verbs kept distinct. (texte-55, annot-058) |
| venir en preuue | to come into proof | (texte-55) |
| ſe reſtraindre aux lieux, temps, & perſonnes | to restrict oneself to places, times and persons | The *negativa coarctata*. (texte-55, annot-058) |
| vulgaire reigle | common rule | Cf. *vulgaires & communes reigles*. (texte-55) |
| ſentence (a saying) | maxim | Not the lower judge's *sentence* of §5. (annot-058) |
| le Philoſophe | the Philosopher | Aristotle. (annot-058) |
| niement | denial | (annot-058) |
| prouuer vne negation | to prove a negation | (annot-058) |
| eſcheoir (la difficulté n'y eſcherroit point) | to arise | (annot-058) |
| approuuer au degré de Doctorat | to approve for the degree of Doctorate | Cf. *insignes du degré*. (annot-058) |
| eſcolier | scholar | A student. (annot-058) |
| ſuffiſance | sufficiency | (annot-058) |
| reprouuer (a candidate) | to reject | (annot-058) |
| coarctee | narrowed | (annot-058) |
| depoſer pour l'innocence / pour l'accuſation & la charge | to depose for innocence / for the accusation and the charge | (annot-058) |
| recognoiſtre ſon erreur | to recognise one's error | (texte-56) |
| ſe departir de (ſa depoſition) | to depart from (one's deposition) | The witness's retraction. (texte-56, annot-059) |
| faire quelque difficulté | to raise some difficulty | (annot-059) |
| adiouſter foy à | to give faith to | (annot-059) |
| varier (of a witness) | to vary | The *testis varians*. (annot-059) |
| contradiction & repugnance | contradiction and repugnance | Cf. *preuves repugnantes*. (annot-059) |
| pariure | perjured; perjurer | (annot-059) |
| circonuention | circumvention | Cf. *circonvenir*. (annot-059) |
| ſe corriger | to correct oneself | (annot-059) |
| ſur l'heure | on the spot | (annot-059) |
| eſpace & interual de temps | space and interval of time | Cf. *laps ou intervalle de temps*. (annot-059) |
| iuſte cauſe | just cause | (annot-059) |
| emprainctes | imprinted | Of marks on the body. (texte-57) |
| goutes de ſang à l'œil / enfoncement de l'ongle | drops of blood in the eye / the sinking of the nail | The marks of texte-38. (texte-57) |
| depoſer chacun de ſon faict | each deposing to his own fact | Cf. *teſmoins ſinguliers*. (texte-57) |
| certain & reſolu | certain and settled | (texte-57, LX) |
| tenir le lieu que d'vn | to hold the place of but one | (LX) |
| n'eſtre pour rien compté | to be counted for nothing | *Unus testis nullus testis*. (LX) |
| fureur | madness | *Furor*. (LX) |
| election de ſepulture | choice of burial | The *electio sepulturae*. (LX) |
| entreliees & coniointes | interlinked and conjoined | (LX) |
| interpretation legiere & trop inconſiderée | a light and too inconsiderate interpretation | (LX) |
| acte vniuerſel ou general | a universal or general act | (LX) |
| couſtume | custom | (LX) |
| faire apparoir | to make appear | To prove. (LX) |
| cours ſouueraines | sovereign courts | The parlements. (LX) |
| compaſſer & meſurer à droite aulne | to compass and measure by a straight ell | (LX) |
| poiſer à iuſte balance | to weigh in a just balance | (LX) |
| traitez & diffiniz | treated and determined | *Definir*, to decide. (LX) |
| tant ſoit-il enorme | however enormous it be | (LX) |
| preuue certaine & concluante | certain and conclusive proof | (LX) |
| diffamé (d'vn crime) | defamed (of a crime) | The canonists' *diffamatus*. (LX) |
| le plus grief | the most grievous | (LX) |
| iuger par opinion & à la legere | to judge by opinion and lightly | (LX) |
| droitement & en verité | rightly and in truth | (LX) |
| obuier à | to obviate | (LX) |
| ſous le manteau & pretexte de la religion | under the cloak and pretext of religion | (LX) |
| concuſſions | extortions | By those in office. (LX) |
| impietez mal heureuſes | wretched impieties | (LX) |
| le propos duquel nous ſommes iſſus | the matter from which we have strayed | (LX) |
| Iaques Butrigaire | Jacobus Butrigarius (Giacomo Bottrigari) | (LX) |
| Pierre de Bellaper. / Cyne | Pierre de Belleperche / Cino da Pistoia | (LX) |
| Guid. Papæ | Gui Pape (Guido Papa), *Decisiones* | (LX) |
| Boyer | Nicolas Bohier (Boerius), *Decisiones Burdegalenses* | (LX) |

## For the reviewer

1. **The stitching of `texte-57`** (Annotation LX inside it, and the one-behind offset of every later `annot-NNN` id) is the thing to settle first; see "Structural problem".
2. **texte-57 / LX at {h}**: the defective sentence on heresy witnesses. Collate another edition for the missing words; the bracketed supplement is the minimum the sense needs.
3. **texte-57 / LX, the closing passage** on heresy prosecutions "under the cloak and pretext of religion" and the demand that heresy be judged "rightly and in truth" and not "by opinion and lightly": written 1 October 1560 or shortly after (annot-058 gives the date), by a judge who converted to Calvinism within a few years and was killed for it. Translated literally; worth a note in the apparatus, with case file §3.
4. **annot-058, "this first day of October 1560"** — Coras's date for the writing of the annotations, nineteen days after the decision. Worth recording in case file §2.
5. **annot-055**: Coras gives Sanxi's age as thirteen and the prisoner's as thirty-five. Case file §3 has Sanxi born c. 1548 (so about twelve in 1560); the case file may want the figures.
6. **annot-057**: the orphan note *b* and the sentence that breaks off without a stop — page image of p077.
7. **annot-059**: the *g*/*h* mismatch in the margin; whether to re-key the transcription's note *g* as *h*.
8. **annot-058 {c}** and **LX {g}**: orphan notes placed by content; page images of p078 and p080 for the key letters.
9. **annot-054**: Leo of Byzantium, Dionysius of Heraclea and Louis the Fat are uncited in the margin; the headnote supplies Plutarch *Moralia* 804a / Athenaeus 12.550f, Athenaeus 12.549a–b / Aelian *VH* 9.13, and the regnal count. Coras's "thirty-ninth King" follows the old chroniclers' count from Pharamond; leave or note.
10. **annot-056**: Chrysippus's saying on infants is from Varro, *De lingua Latina* 6.56; uncited in the margin, supplied in the headnote.
11. **Names of the doctors** new in this batch (Butrigarius, Belleperche, Cino, Gui Pape, Bohier) are given in the standard forms with Coras's French in brackets at first use, per §6.5; the coordinator may wish to add them to §6.5's table.
12. The print's *du Thil* (texte-57) is kept as the variant of the name, as the earlier batches did at annot-035 and annot-042.

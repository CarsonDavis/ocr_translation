# Incipit pass on the `unit` citations

Date: 2026-10-04. Scope: every ref in `site/data/citations.json` with status `unit` and corpus digest, code, institutes, novels, decretals, decretum, sext or clementines (208 refs). The incipit was taken from the Notes line's bold identification and matched against the unit file: the text has to start with it (Roman law: at a fragment start for `l.`, at a paragraph start within the named fragment for `§`; canon law: after the rubric and inscription). Matching ignored case, u/v, i/j, ae/e, doubled letters, *quum*/*cum*, *-iens*/*-ies*, *adr-*/*arr-*, and the spacing of compounds such as *pater familias*. A unique match was then checked by hand against the Notes gloss before the line was edited. Where Coras gives a single common word, the match was accepted only when it was the exact incipit he prints, unique in the unit, and it fitted the gloss. These rows are marked in the notes column.

## Counts

| status | refs |
|---|---:|
| resolved | 129 |
| ambiguous | 17 |
| not found | 31 |
| not applied | 2 |
| removed | 2 |
| no incipit | 27 |
| total examined | 208 |

"resolved" counts refs: a ref naming two laws, such as `cc. *De cetero* and *Exhibita*`, is one ref with two table rows. "removed" means the line offered three titles and the law turned up in one of them (p127:i, D. 12.4.15), so the two alternative titles left the identification. "not applied" means there is one candidate but the rules did not allow the edit. "no incipit" means the line has no law incipit to match: authenticae, which are not in the Code text, numbered laws, title-only refs, and the two `empty_at_source` Code laws.

cite_locate.py status counts, before → after: passage 855 → 990, unit 229 → 98, none 73 → 72; work 256, scan 21 unchanged. Refs 1434 → 1437, because lines naming two laws now give two refs and p127:i dropped two alternatives. check_markers.py --all: 225/225 ok.

Code numbers in the Notes follow Coras's (vulgate) numbering, and the locator applies `code/concordance.json` forwards. So the two laws found under shifted Krüger numbers are written with the vulgate number: p027:h C. 9.9.33 (Krüger 9.9.32) and p122:m C. 9.9.29 (Krüger 9.9.28). Each line also gives the Krüger number.

## Table

| key | section | corpus/unit | incipit | matched passage | first 60 chars | status | notes |
|---|---|---|---|---|---|---|---|
| p003:n | annot-001 | decretals 3.10 | Cum apostolica | 3.10.7 | Quum apostolica sedes, +cui, licet immeriti, praesidemus, un | resolved | *Quum apostolica* |
| p003:o | annot-001 | decretum D.61 | Statuimus | D.61 c.8 | Statuimus ne in aliquo apostolica et canonica decreta uiolen | resolved |  |
| p004:ſ | annot-001 | decretals 4.2 | Puberes | 4.2.3 | Puberes a pube sunt vocati, id est a pudentia corporis nuncu | resolved |  |
| p004:x | annot-001 | digest 1.7 | Arrogato | 1.7.40.1 | Non tantum cum quis adoptat, sed et cum adrogat, maior esse  | resolved | § 1 kept; *Adrogato* in the edition |
| p006:g | annot-002 | novels 22 |  § Sed etiam | 22.7 | Sed etiam captivitatis casus talis est, quale est bona grati | resolved |  |
| p006:k | annot-002 | decretals 4.19 | Significasti | 4.19.4 | Significasti nobis, quod quidam miles in provincia tua, uxor | resolved |  |
| p006:l | annot-002 | digest 24.3 | Cum mulier | 24.3.47; 24.3.55 | — | ambiguous | 24.3.47 (*lenocinio*) fits the gloss |
| p006:l | annot-002 | decretals 4.13 | Discretionem | 4.13.6 | Discretionem tuam in Domino commendamus, quod in his, quae d | resolved |  |
| p006:l | annot-002 | decretals 5.16 | Intelleximus | 5.16.6 | Intelleximus tam per literas venerabilis fratris nostri Pict | resolved |  |
| p007:x | annot-002 | decretals 2.20 | Veniens | 2.20.10; 2.20.38 | — | ambiguous | Coras "c. veniens j." = the first *Veniens*, i.e. 2.20.10 |
| p010:g | annot-004 | code 8.53 | Data | 8.53.27.pr | Data iam pridem lege statuimus, ut donationes interveniente  | resolved |  |
| p010:n | annot-004 | digest 8.2 | Cum debere | — | — | not found | |
| p010:n | annot-004 | code 3.34 | Cum debere | — | — | not found | |
| p018:b | annot-005 | digest 46.3 | Titia | 46.3.48 | Titia cum propter dotem bona mariti possideret, omnia pro do | resolved |  |
| p018:d | annot-005 | digest 28.5 | Duo socii | 28.5.8.pr | Duo socii quendam servum communem testamento facto heredem e | resolved |  |
| p024:h | annot-009 | digest 24.3 | Si filio | 24.3.25 pr; 24.3.53 | — | ambiguous |  |
| p024:l | annot-009 | code 8.53 | Si filius | 8.53.14 | Si filius tuus res ad te pertinentes sponsae suae te non con | resolved |  |
| p024:n | annot-009 | digest 41.3 | Si fur | 41.3.32.1 | Si quis id, quod possidet, non putat sibi per leges licere u | resolved |  |
| p024:r | annot-009 | digest 6.2 | Quaecunque | 6.2.13.pr | Quaecumque sunt iustae causae adquirendarum rerum, si ex his | resolved |  |
| p025:a | annot-010 | digest 39.3 | Sed hoc ita | 39.3.20 | Sed hoc ita, si non per errorem aut imperitiam deceptus fuer | resolved |  |
| p025:b | annot-010 | digest 47.2 | Verum | 47.2.25 pr; 47.2.39 | — | ambiguous | 47.2.39 (*causa faciendi*) fits the intent point |
| p026:h | annot-011 | code 5.5 | Qui contra | 5.5.4.pr | Qui contra legum praecepta vel contra mandata constitutiones | resolved |  |
| p026:l | annot-011 | code 5.5 | Qui contra | 5.5.4.pr | Qui contra legum praecepta vel contra mandata constitutiones | resolved |  |
| p027:c | annot-012 | digest 50.1 | Tatio | — | — | not found | |
| p027:f | annot-012 | digest 48.10 | Qui falsam § Accusatio | 48.10.19.1 | Accusatio suppositi partus nulla temporis praescriptione dep | resolved |  |
| p027:h | annot-012 | digest 29.5 | In cognitione | 29.5.13 | In cognitione aperti adversus senatus consultum testamenti e | resolved |  |
| p027:h | annot-012 | code 9.9 | Adulterii | 9.9.32 | Adulterii accusatione proposita praescriptiones civiles, qui | resolved | written C. 9.9.33 (Coras numbering) so the concordance lands on Krüger 9.9.32 |
| p027:i | annot-012 | digest 48.10 | Qui falsam § Accusatio | 48.10.19.1 | Accusatio suppositi partus nulla temporis praescriptione dep | resolved |  |
| p028:l | annot-012 | digest 37.1 | Edicto | 37.1.13 | Edicto praetoris bonorum possessio his denegatur, qui rei ca | resolved |  |
| p028:l | annot-012 | digest 38.2 | Si nesem § Si deportatus | — | — | not found | |
| p030:b | annot-014 | decretum D.37 | — | — | — | no incipit | *c. panor.*: no canon of D. 37 answers |
| p031:c | annot-015 | code 9.20 | Si | 9.20.2; 9.20.5; 9.20.12 | — | ambiguous | one common word |
| p031:c | annot-015 | digest 11.3 | Et tantum | — | — | not found | |
| p031:b2 | annot-016 | digest 31 | Qui habebat | — | — | not found | |
| p033:a | annot-017 | code 8.13 | Debitores | 8.13.10 | Debitores praesentes prius denuntiationibus conveniendi sunt | resolved |  |
| p037:d | annot-020 | digest 30 | Si domus § Qui confitetur | 30.71.3 | Qui confitetur se quidem debere, iustam autem causam adfert, | resolved | § resolved within 30.71 |
| p037:f | annot-020 | digest 2.8 | Si fideiussor | 2.8.7.pr | Si fideiussor non negetur idoneus, sed dicatur habere fori p | resolved |  |
| p037:g | annot-020 | digest 49.1 | Imperatores | 49.1.21.pr | Imperatores antoninus et verus rescripserunt appellationes,  | resolved |  |
| p037:g | annot-020 | code 7.65 | Ab executione | — | — | not found | |
| p037:h | annot-020 | digest 43.30 | § Si vero | 43.30.1.3 | — | not applied | unique in the unit, but the line names no fragment (print "l. ijs") |
| p042:t | annot-022 | novels 22 | § Si vero | 22.9; 22.10; 22.16; 22.23; 22.32 | — | ambiguous |  |
| p042:u | annot-022 | decretum C.33 | — | — | — | no incipit | printed C. 33 q. 7 for C. 32 q. 7 c. 25 |
| p043:d | annot-022 | decretals 4.15 | Fraternitatis | 4.15.6 | Fraternitatis tuae literas recepimus, continentes, quod O. m | resolved | with the last chapter, 4.15.7, incipit *Litterae vestrae* |
| p043:d | annot-022 | decretals 4.15 | Fraternitatis | 4.15.7 | Literae vestrae nobis exhibitae continebant, quod, quum caus | resolved | with the last chapter, 4.15.7, incipit *Litterae vestrae* |
| p043:f | annot-022 | decretals 4.15 | Fraternitatis | 4.15.6 | Fraternitatis tuae literas recepimus, continentes, quod O. m | resolved | with the last chapter, 4.15.7, incipit *Litterae vestrae* |
| p043:f | annot-022 | decretals 4.15 | Fraternitatis | 4.15.7 | Literae vestrae nobis exhibitae continebant, quod, quum caus | resolved | with the last chapter, 4.15.7, incipit *Litterae vestrae* |
| p051:a | annot-029 | digest 38.8 | Octavi (*l. octaui*) | 38.8.9.pr | Octavi gradus adgnato iure legitimi heredis, etsi non extite | resolved |  |
| p053:d | annot-031 | digest 7.1 | Locum § Ex eo | 7.1.17.1 | Ex eo, ne deteriorem condicionem fructuarii faciat proprieta | resolved |  |
| p053:i | annot-031 | code 9.47 | Si quis in metallum | 9.47.17 | Si quis in metallum fuerit pro criminum deprehensorum qualit | resolved |  |
| p057:b | annot-036 | code 3.44 | — | — | — | no incipit | authentica (*Alearum*/*Aut.*), not in the Code text |
| p057:g | annot-036 | novels 77 | last | 77.2 | Praecepimus enim gloriosissimo praefecto regiae civitatis pe | resolved | last paragraph of the Novel: 77.2, incipit *Praecepimus* |
| p057:g | annot-036 | novels 77 | — | — | — | no incipit | second mention of Nov. 77 on the line ("places Nov. 77 in the sixth"); the first resolved to 77.2 |
| p058:b | annot-037 | digest 17.2 | Merito | 17.2.51.pr | Merito autem adiectum est ita demum furti actionem esse, si  | resolved |  |
| p058:e | annot-037 | digest 35.1 | Non ad ea | 35.1.89 | Non ad ea dumtaxat pertinet, quae saepius sub diversis condi | resolved |  |
| p058:f | annot-037 | digest 22.3 | Eum qui | 22.3.22 | Eum, qui voluntatem mutatam dicit, probare hoc debere. | resolved |  |
| p058:h | annot-037 | decretals 2.19 | Praeterea | — | — | not found | |
| p059:b | annot-038 | digest 38.8 | octui (garbled) | — | — | not found | probably *Octavi*, 38.8.9, as at p051:a and p148:a |
| p059:b | annot-038 | digest 4.1 | De tutela | — | — | not found | the law is C. 2.21.7 (*De tutela*), as at p148:a; the print says D. |
| p060:a | annot-039 | code 4.20 | — | — | — | no incipit | C. 4.20.6, empty at source |
| p060:b | annot-039 | code 4.19 | Etiam matris | — | — | not found | |
| p060:e | annot-039 | decretals 2.23 | Litteras | 2.23.14 | Literas vestras accepimus continentes, quod, quum causa, qua | resolved |  |
| p061:g | annot-039 | digest 47.2 | Si cui § Si | — | — | not found | |
| p067:a2 | annot-046 | digest 17.2 | Merito | 17.2.51.pr | Merito autem adiectum est ita demum furti actionem esse, si  | resolved |  |
| p067:a2 | annot-046 | digest 50.17 | Quoties § Qui dolo | 50.17.20, .67, .91, .98, .200 | — | ambiguous | no Quotiens fragment has a § *Qui dolo*; *Qui dolo desierit possidere* is its own fragment, 50.17.131 |
| p069:a | annot-048 | digest 12.4 | Si procuratori falso | 12.4.14 | Si procuratori falso indebitum solutum sit, ita demum a proc | resolved |  |
| p069:a | annot-048 | code 2.12 | Licet | 2.12.24 | Licet in principio quaestionis persona debet inquiri procura | resolved |  |
| p069:b | annot-048 | decretals 1.3 | Ex parte decani | 1.3.33 | Ex parte decani et capituli Laudunensis fuit propositum, quo | resolved | title *de re.* = *De rescriptis* confirmed |
| p069:c | annot-048 | code 6.2 | Falsus | 6.2.19 | Falsus procurator depositum recipiendo vel aes alienum exige | resolved |  |
| p071:a | annot-050 | decretals 2.27 | last | 2.27.26 | Duobus iudicibus, ut accepimus, diversas sententias proferen | resolved | last chapter: incipit there *Duobus iudicibus* |
| p071:d | annot-050 | code 5.18 | — | — | — | no incipit | *l. iij.*, title corrupt |
| p071:d | annot-050 | code 5.27 | — | — | — | no incipit | *l. iij.*, title corrupt |
| p071:e | annot-050 | decretals 2.19 | Ex litteris | 2.19.3 | Ex literis tuis intelleximus, te et archidiaconum confines h | resolved | one-word incipit *Inter* (= *Inter dilectos*) |
| p071:e | annot-050 | decretals 2.22 | Inter, Inter dilectos | 2.22.6 | Inter dilectos filios G. abbatem sancti Donati de Scozula ex | resolved | one-word incipit *Inter* (= *Inter dilectos*) |
| p071:i | annot-050 | digest 44.4 | Pure § Si nat… | 44.4.5 pr for *Pure* | — | not found | no § of 44.4.5 begins *Si nat*; title itself is a guess |
| p071:l | annot-050 | digest 34.5 | Ubi enim | — | — | not found | |
| p072:b2 | annot-052 | decretals 2.27 | last | 2.27.26 | Duobus iudicibus, ut accepimus, diversas sententias proferen | resolved | last chapter: incipit there *Duobus iudicibus* |
| p072:c | annot-052 | code 6.59 | — | — | — | no incipit | authentica *Itaque* |
| p073:e2 | annot-053 | decretals 2.20 | Praeterea | 2.20.27 | Praeterea quum quis accusatur aliquam cognovisse, an sint te | resolved |  |
| p078:d | annot-058 | decretals 2.19 | Proposuisti | 2.19.4 | Proposuisti nobis, dilecte fili praeposite, quod causa matri | resolved |  |
| p079:a | annot-059 | code 9.22 | Eos | — | — | not found | |
| p080:b | annot-060 | code 4.20 | Iurisiurandi | — | — | not found | |
| p080:c | annot-060 | code 4.20 | — | — | — | no incipit | C. 4.20.9, empty at source |
| p083:i | annot-062 | decretals 2.20 | Tam litteris | 2.20.33 | Tam literis vestris, quam depositionibus testium diligenter  | resolved |  |
| p086:a | annot-067 | code 1.18 | Si post divisionem | 1.18.4 | Si post divisionem factam testamenti vitium in lucem emerser | resolved |  |
| p086:c | annot-067 | digest 22.3 | Eum qui | 22.3.22 | Eum, qui voluntatem mutatam dicit, probare hoc debere. | resolved |  |
| p086:d | annot-067 | code 4.19 | Sive possidetis | 4.19.16 | Sive possidetis praedia, quae a patre communi sibi fratres e | resolved |  |
| p086:e | annot-067 | decretals 2.23 | Litteris | 2.23.12 | Literis tuae fraternitatis receptis ex tenore illarum nobis  | resolved |  |
| p086:e | annot-067 | decretals 2.20 | Praeterea | 2.20.27 | Praeterea quum quis accusatur aliquam cognovisse, an sint te | resolved |  |
| p086:e | annot-067 | code 3.36 | In ipsius | 3.36.5 | In ipsius mariti tui fuit potestate mutare, quod iratus in s | resolved |  |
| p086:e | annot-067 | decretals 5.39 | Sicut nobis | 5.39.39 | Sicut nobis +tuis literis intimasti, quum aliquos tuae dioec | resolved |  |
| p088:b | annot-069 | digest 35.1 |  § Titio centum | 35.1.71.pr | Titio centum ita, ut fundum emat, legata sunt: non esse coge | resolved | law unique; the paragraph is ambiguous (71 pr, 71.1, 71.2; 71.2 fits) |
| p097:b | annot-072 | code 9.1 | Si magnum, Si sororem | 9.1.13 | Si magnum et capitale crimen ac non leve frater contra fratr | resolved |  |
| p097:b | annot-072 | code 9.1 | Si magnum, Si sororem | 9.1.18 | Si sororem tuam leviorum commissorum ream facis, accusatione | resolved |  |
| p097:c | annot-072 | code 5.62 | Humanitatis | 5.62.23.pr | Humanitatis ac religionis ratio non permittit, ut adversus s | resolved |  |
| p100:d | annot-074 | code 4.21 | — | — | — | no incipit | authentica *Apud eloquentissimum* |
| p100:e | annot-075 | digest 48.5 | penultimate law | 48.5.44 | — | not applied | rule covers the last law only |
| p101:l | annot-075 | digest 47.2 | Verum | 47.2.25 pr; 47.2.39 | — | ambiguous |  |
| p101:t | annot-075 | decretals 5.12 | Lator | 5.12.9 | Lator praesentium P. clericus nobis viva voce proposuit, quo | resolved |  |
| p102:c | annot-076 | novels 97 | § Quaesitum | — | — | not found | *Quaesitum est igitur* is inside 97.3, not at its start (the line already says cf. 97.3) |
| p110:b | annot-080 | code 12.1 | Indignat… | — | — | not found | |
| p113:m | annot-082 | decretals 5.39 | Perpendimus | 5.39.23 | Perpendimus ex literis tuis, quod quidam sacercdos tuae dioe | resolved |  |
| p118:e | annot-082 | digest 37.1 | Edicto | 37.1.13 | Edicto praetoris bonorum possessio his denegatur, qui rei ca | resolved |  |
| p120:n | annot-083 | digest 1.16 | Si quid | 1.16.11 | Si quid erit quod maiorem animadversionem exigat, reicere le | resolved |  |
| p120:o | annot-083 | digest 48.19 | Cum damnum | — | — | not found | |
| p122:k | annot-083 | digest 1.18 | Sed licet | 1.18.12 | Sed licet is, qui provinciae praeest, omnium Romae magistrat | resolved |  |
| p122:m | annot-083 | code 9.9 | Quae adulterium | 9.9.28 | Quae adulterium commisit, utrum domina cauponae an ministra  | resolved | written C. 9.9.29 (Coras numbering) for Krüger 9.9.28 |
| p122:o | annot-083 | code 2.2 | Venia | 2.2.2 | Venia edicti non petita patronum seu patronam eorumque paren | resolved |  |
| p124:c | annot-084 | novels 143 | — | — | — | no incipit | § 1 of the Novel *De raptu* (no incipit) |
| p124:c | annot-084 | novels 150 | — | — | — | no incipit | § 1 of the Novel *De raptu* (no incipit) |
| p124:d | annot-084 | code 1.7 | Eum | 1.7.5 | Eum, quicumque servum seu ingenuum, invitum vel suasione ple | resolved | one-word incipit *Eum* |
| p127:i | annot-087 | digest 48.24 | Cum servus | — | — | removed | alternative title dropped: the law is D. 12.4.15 |
| p127:i | annot-087 | digest 12.4 | Cum servus | 12.4.15 | Cum servus tuus in suspicionem furti Attio venisset, dedisti | resolved | the Notes said D. 12.4 had no such law |
| p127:i | annot-087 | digest 13.1 | Cum servus | — | — | removed | alternative title dropped: the law is D. 12.4.15 |
| p129:a2 | annot-092 | code 3.15 | — | — | — | no incipit | authentica *Qua in provincia* |
| p129:a2 | annot-092 | novels 69 | — | — | — | no incipit | authentica *Qua in provincia* |
| p132:i | annot-094 | code 3.26 | Universi | 3.26.9 | Universi fiduciam gerant, ut, cum quis eorum ab actore rerum | resolved |  |
| p132:k | annot-094 | code 9.6 | — | — | — | no incipit | ll. 1, 2 and the last (numbers, no incipit) |
| p133:d | annot-095 | decretals 4.17 | Ex tenore | 4.17.14 | Ex tenore literarum vestrarum nobis innotuit, quod, quum G.  | resolved |  |
| p134:f | annot-095 | digest 23.2 | Qui in provincia | 23.2.57 | Qui in provincia officium aliquid gerit, prohibetur etiam co | resolved | § 1 kept; the edition merges 57a into 57 |
| p134:i | annot-095 | digest 40.12 | Duobus | 40.12.30 | Duobus petentibus hominem in servitutem pro parte dimidia se | resolved |  |
| p134:p | annot-095 | digest 5.1 | Non quicquid | — | — | not found | |
| p134:q | annot-095 | novels 134 | § Si | 134.5; 134.10; 134.12 | — | ambiguous | the gloss already names Nov. 134.13 |
| p135:c | annot-096 | digest 23.3 | Si mulier ("la ij.") | 23.3.13; 23.3.59 pr; 23.3.77 | — | ambiguous | "the second" would be 23.3.59 if Coras counts as the edition does |
| p135:e | annot-096 | decretum C.29 | — | — | — | no incipit | *c. j.*, no incipit (the dictum is at C.29 q.1 pr, as resolved at p140:q) |
| p135:h | annot-096 | decretum C.29 | — | — | — | no incipit | *c. i.* as at {e} |
| p137:n | annot-097 | digest 24.3 | Cum mulier | 24.3.47; 24.3.55 | — | ambiguous | same law as p006:l; 24.3.47 fits |
| p137:n | annot-097 | decretals 4.13 | Discretionem | 4.13.6 | Discretionem tuam in Domino commendamus, quod in his, quae d | resolved |  |
| p137:o | annot-097 | digest 19.2 | Qui domum | 19.2.57 | Qui domum habebat, aream iniunctam ei domui vicino proximo l | resolved |  |
| p137:o | annot-097 | decretals 5.12 | De cetero, Exhibita | 5.12.11 | De cetero noveris, quod diaconus, quem literatum et honestat | resolved |  |
| p137:o | annot-097 | decretals 5.12 | De cetero, Exhibita | 5.12.22 | Exhibita nobis humilis I. clerici confessio patefecit, quod, | resolved |  |
| p137:ſ | annot-097 | digest 47.2 | Verum | 47.2.25 pr; 47.2.39 | — | ambiguous |  |
| p138:a | annot-098 | decretum D.86 | Si quid | D.86 c.23 | Si quid uero de quocumque clerico ad aures tuas peruenerit,  | resolved |  |
| p138:b | annot-098 | decretum D.83 | Error | D.83 c.3 | Error, cui non resistitur, approbatur, et ueritas, cum minim | resolved |  |
| p138:e | annot-098 | code 5.5 | Qui contra | 5.5.4.pr | Qui contra legum praecepta vel contra mandata constitutiones | resolved |  |
| p139:k | annot-098 | decretals 3.6 | Tua nos | 3.6.4 | Tua nos duxit fraternitas consulendos: (Et infra: [cf. c. 19 | resolved |  |
| p139:l | annot-098 | decretum D.34 | Si quis viduam / Si cuius | c.13, c.15 / c.11 | — | ambiguous | *Si cuius* is unique (D.34 c.11), *Si quis viduam* opens c.13 and c.15; line not edited |
| p139:m | annot-098 | novels 97 | § Quaesitum | — | — | not found | as p102:c |
| p140:q | annot-098 | decretum C.29 |  § Quod autem | C.29 q.1 pr | Quod autem coniugium sit inter eos, probatur hoc modo. Coniu | resolved | Gratian's dictum before c. 1 (printed c. 1 § *Quod autem*) |
| p140:t | annot-098 | digest 40.12 | Igitur § Si na[tus?] | 40.12.12 pr for *Igitur* | — | not found | no § of 40.12.12 begins *Si na* |
| p141:h | annot-098 | digest 4.4 | Tutor | 4.4.47.pr | Tutor urguentibus creditoribus rem pupillarem bona fide vend | resolved |  |
| p141:m | annot-098 | decretum D.86 | Si quid | D.86 c.23 | Si quid uero de quocumque clerico ad aures tuas peruenerit,  | resolved |  |
| p142:a | annot-100 | code 7.64 | Omnem | 7.64.10.pr | Omnem honorem salvum iudicibus reservantes, si quando una pa | resolved |  |
| p143:a | annot-101 | digest 47.2 | Quidam tabular[ius] | 47.2.32.pr | Quidam tabularum dumtaxat aestimationem faciendam in furti a | resolved | print *tabular.* = *tabularum* |
| p143:f | annot-101 | digest 32 | Pamphilo § Proposi[tum] | 32.39.1 | Propositum est non habentem liberos nec cognatos in discrimi | resolved |  |
| p143:h | annot-101 | code 6.23 | Quoniam indignum | 6.23.15.pr | Quoniam indignum est ob inanem observationem irritas fieri t | resolved |  |
| p143:i | annot-101 | digest 28.1 | Cum lege | 28.1.26 | Cum lege quis intestabilis iubetur esse, eo pertinet, ne eiu | resolved |  |
| p148:a | annot-105 | digest 38.8 | Octavi | 38.8.9.pr | Octavi gradus adgnato iure legitimi heredis, etsi non extite | resolved |  |
| p148:a | annot-105 | code 2.21 | De tutela | 2.21.7 | De tutela avunculi eiusdemque tutoris, cui falso aetate prob | resolved |  |
| p150:i | annot-106 | digest 38.2 | Paulus § 1 | 38.2.46; 38.2.47 pr | — | ambiguous | 47 has paragraphs, 46 does not |
| p151:b | annot-108 | code 4.20 | — | — | — | no incipit | authentica *Si testis* |
| p152:a | annot-109 | digest 32 | Cum quis decedens § Codicillis | 32.37.5 | Codicillis ita scripsit: "Boulomai panta ta hupotetagmena ku | resolved |  |
| p152:a | annot-109 | digest 39.5 | Ex hac scriptura | 39.5.16 | Ex hac scriptura: "Sciant heredes mei me vestem universam ac | resolved |  |
| p152:d | annot-109 | digest 16.1 | Seia | 16.1.28.pr | Seia mancipia emit et mutuam pecuniam accepit sub fideiussor | resolved |  |
| p152:d | annot-109 | code 4.19 | Rationes | 4.19.6.pr | Rationes defuncti, quae in bonis eius inveniuntur, ad probat | resolved |  |
| p153:e | annot-109 | code 2.4 | Transactione | 2.4.26; 2.4.30 | — | ambiguous |  |
| p153:f | annot-109 | digest 28.5 | Paterfamilias | 28.5.45 | Pater familias testamento duos heredes instituerat: eos monu | resolved |  |
| p153:f | annot-109 | code 6.23 | Verba | 6.23.6 | Verba testamenti, quibus mater vestra decedens nihil se cuiq | resolved |  |
| p153:h | annot-109 | code 9.46 | Mater | 9.46.2.pr | Mater inter eas personas est, quae sine calumniae timore nec | resolved |  |
| p153:k | annot-109 | decretals 5.12 | Exhibita | 5.12.22 | Exhibita nobis humilis I. clerici confessio patefecit, quod, | resolved |  |
| p153:q | annot-109 | code 5.11 | — | — | — | no incipit | *l.* without incipit |
| p153:r | annot-109 | decretals 2.20 | Sicut, Cum in tua | 2.20.9 | Sicut nobis est ex parte tua intimatum, quidam parochiani tu | resolved | *Quum in tua* |
| p153:r | annot-109 | decretals 2.20 | Sicut, Cum in tua | 2.20.44 | Quum in tua dioecesi: (Et infra: [cf. c. 30. de decimis III. | resolved | *Quum in tua* |
| p154:x | annot-109 | decretals 2.23 | Litteris | 2.23.12 | Literis tuae fraternitatis receptis ex tenore illarum nobis  | resolved |  |
| p154:y | annot-109 | code 7.49 | — | — | — | no incipit | authentica *Novo iure* |
| p154:d | annot-109 | digest 31 | Lucius § Quisquis | 31.22; 31.88 pr | — | ambiguous | with the § only 31.88 fits, but § *Quisquis* opens both 31.88.3 and 31.88.10 (10 is the debt acknowledgment) |
| p154:e | annot-109 | decretals 2.23 | Tertio loco | 2.23.13 | Tertio loco quaestionis huiusmodi nodum tua fraternitas peti | resolved |  |
| p154:f | annot-109 | digest 34.2 | Qui uxori | 34.2.18.pr | Qui uxori suae legaverat bonorum suorum decimam et mancipia  | resolved |  |
| p154:f | annot-109 | digest 32 | Lucius | 32.93.1 | "Semproniae mulieri meae reddi iubeo ab heredibus meis centu | resolved |  |
| p154:f | annot-109 | digest 34.3 | Aurelius | 34.3.28.pr | Aurelius symphorus fideiusserat pro tutore quodam et deceden | resolved |  |
| p154:f | annot-109 | code 8.53 | Si donatio | 8.53.5 | Si donatio per epistulam facta non paret, verba tamen testam | resolved |  |
| p154:i | annot-109 | digest 32 | Cum quis decedens § Codicillis | 32.37.5 | Codicillis ita scripsit: "Boulomai panta ta hupotetagmena ku | resolved |  |
| p154:i | annot-109 | code 4.19 | — | — | — | no incipit | authentica *Quod obtinet* |
| p155:k | annot-109 | digest 30 | Si creditor | 30.28.pr | Si creditori meo, tutus adversus eum exceptione, id quod ei  | resolved | *Si creditor[i]* |
| p155:k | annot-109 | digest 39.5 | Ex hac scriptura | 39.5.16 | Ex hac scriptura: "Sciant heredes mei me vestem universam ac | resolved | *Si creditor[i]* |
| p155:l | annot-109 | digest 32 | Lucius § Quisquis | 32.93 pr for *Lucius* | — | not found | 32.93 has no § *Quisquis*; the law is probably 31.88 (see p154:d) |
| p155:m | annot-109 | digest 39.3 | Sed hoc ita | 39.3.20 | Sed hoc ita, si non per errorem aut imperitiam deceptus fuer | resolved |  |
| p155:o | annot-109 | digest 32 | Cum quis decedens § Titia | 32.37.6 | Titia honestissima femina cum negotiis suis opera Callimachi | resolved |  |
| p155:o | annot-109 | digest 22.3 | Qui testamentum | 22.3.27 | Qui testamentum faciebat ei qui usque ad certum modum capere | resolved |  |
| p155:ſ | annot-109 | digest 32 | Cum quis decedens § Codicillis | 32.37.5 | Codicillis ita scripsit: "Boulomai panta ta hupotetagmena ku | resolved |  |
| p155:t | annot-109 | decretals 2.24 | Quintavallis, Cum contingat | 2.24.23 | Quintavallis vicarius nostris auribus intimavit, quod, quum  | resolved | *Quum contingat* |
| p155:t | annot-109 | decretals 2.24 | Quintavallis, Cum contingat | 2.24.28 | Quum contingat interdum in tua dioecesi, quod constante matr | resolved | *Quum contingat* |
| p155:t | annot-109 | decretals 5.12 | Cum iuramento | 5.12.4 | Cum iuramento pollicitus est Herodes saltatrici dare quodcun | resolved | *Quum contingat* |
| p155:u | annot-109 | digest 22.3 | Qui testamentum | 22.3.27 | Qui testamentum faciebat ei qui usque ad certum modum capere | resolved |  |
| p155:x | annot-109 | code 5.16 | Donationes quas | 5.16.26 | Donationes, quas divinus imperator in piissimam reginam suam | resolved |  |
| p156:a | annot-110 | digest 28.3 | Si quis exheredato § Irritum | — | — | not found | |
| p156:c | annot-110 | code 9.49 | — | — | — | no incipit | authentica *Bona damnatorum* |
| p156:e | annot-110 | code 1.2 | — | — | — | no incipit | authentica *Ingressi* |
| p157:f | annot-110 | code 9.49 | — | — | — | no incipit | authentica *Bona damnatorum* |
| p157:i | annot-110 | digest 28.3 | Si quis filio § Irritum | 28.3.6.5 | Irritum fit testamentum, quotiens ipsi testatori aliquid con | resolved |  |
| p157:i | annot-110 | digest 28.1 | Is cui | 28.1.18.1 | Si quis ob carmen famosum damnetur, senatus consulto express | resolved |  |
| p157:i | annot-110 | digest 28.1 | Eius § 1 | 28.1.8 pr; 28.1.31 | — | ambiguous | second 28.1 ref on the line; the *Is cui* ref resolved |
| p157:o | annot-110 | digest 48.19 | Quidam | — | — | not found | |
| p158:t | annot-110 | digest 31 | Iulianus | 31.60 | Iulianus ait, si a filio herede legatum sit seio fideique ei | resolved | D. 32 (alt. reading) has no fragment *Iulianus* |
| p158:u | annot-110 | digest 31 | Si pluribus | 31.44.pr | Si pluribus heredibus institutis ita scriptum sit: "Heres me | resolved | *Si plures*: D. 32.98 by the alt. reading *iij.*; D. 31 has none |
| p158:u | annot-110 | digest 32 | Si pluribus | 32.98 | Si plures gradus sint heredum et scriptum sit "heres meus da | resolved | *Si plures*: D. 32.98 by the alt. reading *iij.*; D. 31 has none |
| p158:x | annot-110 | code 9.49 | — | — | — | no incipit | authentica *Bona damnatorum* |
| p158:z | annot-110 | code 1.2 | — | — | — | no incipit | authentica *Ingressi* |
| p158:a | annot-110 | decretals 2.2 | Si diligenti | 2.2.12 | Si diligenti Et infra: [cf. c. 17. de praescr. II. 26.] Supe | resolved |  |
| p158:b | annot-110 | digest 28.3 | Si quis filio / Quod si quis | 28.3.6 pr / — | — | not found | *Quod si quis* begins no fragment; line not edited |
| p158:d | annot-110 | code 10.32 | Militaribus | 10.32.42.pr | Militaribus viris nihil sit commune cum curiis: nihil sibi l | resolved |  |
| p158:a2 | annot-111 | code 7.53 | Ordo / Si ut proponis / Executionem | 7.53.3 / 7.53.6 / — | — | not found | *Executionem* begins no law; line not edited |
| p158:a2 | annot-111 | digest 49.1 | Ab executore | — | — | not found | |
| p158:a2 | annot-111 | decretals 5.20 | Super | 5.20.2 | Super eo vero, quod sententiam auctoritate literarum falsaru | resolved | one-word incipit *Super*; the Code and Digest refs on the line not found |
| p159:c | annot-111 | code 1.3 | Nulli | 1.3.28.pr | Nulli licere decernimus, si testamento heres sit institutus  | resolved |  |
| p159:d | annot-111 | decretals 3.26 | Tua nobis | 3.26.17 | Tua nobis fraternitas intimavit, quod nonnulli, tam religios | resolved |  |
| p159:k | annot-111 | digest 40.4 | Lucius | 40.4.53 | Lucius Titius servo libertatem dedit, si rationem actus sui  | resolved |  |
| p159:m | annot-111 | code 1.3 | Nullo | — | — | not found | the Notes say it is *l. nulli* (1.3.28) |
| p159:n | annot-111 | code 1.3 | Nulli | 1.3.28.pr | Nulli licere decernimus, si testamento heres sit institutus  | resolved |  |
| p159:q | annot-111 | novels 1 |  § Si quis igitur non implens | — | — | not found | |
| p159:q | annot-111 | code 6.42 | — | — | — | no incipit | authentica *Hoc amplius* |
| p160:t | annot-111 | digest 28.7 | Quidam | 28.7.27.pr | Quidam in suo testamento heredem scripsit sub tali condicion | resolved |  |
| p160:t | annot-111 | digest 30 | Servo alieno | 30.113.pr | Servo alieno ita legari potest "quoad serviat" vel "si servu | resolved |  |
| p160:u | annot-111 | digest 35.2 | Quod de bonis | — | — | not found | |
| p160:u | annot-111 | decretals 3.26 | Requisisti | 3.26.15 | Requisisti de his, quae testator pro anima sua legat in ulti | resolved |  |

## Leads for a reader (not applied)

- Several ambiguous rows have one candidate that clearly fits the gloss. p006:l and p137:n point to D. 24.3.47 (*lenocinium*), p007:x to X 2.20.10 (Coras writes "Veniens j."), p025:b to D. 47.2.39, p088:b to D. 35.1.71.2, and p154:d to D. 31.88.10.
- p059:b: the garbled *octui* is almost certainly *Octavi* (D. 38.8.9), and *l. de tutela* is C. 2.21.7, not D. 4.1, as at p148:a.
- p155:l: § *Quisquis* is not in D. 32.93. The law is probably D. 31.88 (§ 10, *Quisquis mihi heres erit, sciat debere me…*), in line with the `de leg. ij.` reading at p154:d.
- p139:l: *Si cuius* is D. 34 c. 11. *Si quis viduam* opens both c. 13 and c. 15, and the Notes' "Apostolic canon" description fits c. 15.
- p037:h: § *Si vero* is unique in D. 43.30, at 43.30.1.3. The print's "l. ijs" may be *l. j.*

# Single-pass pilot: reads vs verified finals

Body columns count differing + unmatched body lines against the final; note columns count note-line differences; shared errors are lines where A and B agree with each other but not the final (uncatchable by a reconciler).

| model | page | A–B agree | A–B body diffs | A–B note diffs | A–B struct | A vs final body | A notes | B vs final body | B notes | shared errors |
|---|---|---|---|---|---|---|---|---|---|---|
| opus | p010 | 97.4% | 1 | 3 | 0 | 5 | 2 | 6 | 3 | 5 |
| opus | p020 | 100.0% | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| opus | p030 | 94.3% | 2 | 3 | 0 | 5 | 4 | 6 | 4 | 4 |
| opus | p057 | 94.7% | 2 | 10 | 0 | 5 | 8 | 3 | 16 | 5 |
| opus | p159 | 100.0% | 0 | 5 | 0 | 3 | 9 | 3 | 8 | 7 |
| sonnet | p010 | 79.5% | 8 | 5 | 0 | 7 | 4 | 9 | 5 | 4 |
| sonnet | p020 | 83.8% | 6 | 0 | 2 | 10 | 0 | 7 | 0 | 5 |
| sonnet | p030 | 80.0% | 7 | 7 | 0 | 12 | 13 | 11 | 17 | 10 |
| sonnet | p057 | 89.5% | 4 | 13 | 1 | 5 | 19 | 7 | 17 | 7 |
| sonnet | p159 | 88.9% | 4 | 22 | 0 | 7 | 22 | 5 | 13 | 5 |

## Shared errors

- opus p010 body[0] shared: final `Quæ ſimul intonuit, proxima quæque fugat.` / both reads `Quæ ſimul intonuit, proxima quæque fugat {e}.`
- opus p010 body[8] shared: final `dition, & de toutes parts noſtre aduerſaire: & ſous l'hon` / both reads `dition, & de toutes parts noſtré aduerſaire: & ſous l'hon`
- opus p010 body[14] shared: final `blemẽt offenſez, auant que ſe doubter de luy? ou toutes` / both reads `blemẽt offenſez, auant que ſe doubter de luy? ou toures`
- opus p010 body[26] shared: final `ſaſt auſſi quelque iour le hayr {l}. ce que Publius Mimus` / both reads `ſaſt auſſi quelque iour le hayr {l}. ce que Poblius Mimus`
- opus p010 body[29] shared: final `Ayes ton ami en tel reng. que tu cuides qu'il peut à` / both reads `Ayes ton ami en tel reng, que tu cuides qu'il peut à`
- opus p020 body[30] shared: final `Ce qu'elle impetra & entre les bras de ceſte ombre rea` / both reads `Ce qu'elle impetra & entre les bras de ceſte ombre ren`
- opus p030 body[11] shared: final `quel aiguillon de recouurer le riche ioyau qu'Adraſtus` / both reads `quel aiguillon de recouurer le riche ioyau qu' Adraſtus`
- opus p030 body[12] shared: final `Roy des Argiues auoit, oſa bien entreprendre trahir &` / both reads `Roy des Argiues auoit, oſa bien entreprendre trabir &`
- opus p030 body[14] shared: final `n'aller poinr a la guerre de Thebes, de peur d'eſtre tué` / both reads `n'aller poinr a la guerre de Thebes, de peur d' eſtre tué`
- opus p030 body[25] shared: final `toute vertuſ {i}.` / both reads `toute vertu {i}.`
- opus p057 body[8] shared: final `& Canoniſtes, qu'atribuer corps, membres, & autres` / both reads `& Canoniſtes, qu'attribuer corps, membres, & autres`
- opus p057 body[10] shared: final `detraire de ce que luy appartient {a}. combien qu'à la ve` / both reads `detraire de ce que luy appartient {a}. combien qu'à la ve-`
- opus p057 body[25] shared: final `teurs, par la loy anciéne eſtoient lapidez du peuple {e} de` / both reads `teurs, par la loy anciẽne eſtoient lapidez du peuple {e} de`
- opus p057 note d shared: final `S. Iean. e. x.` / both reads `S. Iean. c. x.`
- opus p057 note h shared: final `l. famoſi. | D. ad l. Iuliã. | maieù.` / both reads `l. famoſi. | D. ad l. Iuliã. | maieu.`
- opus p159 body[18] shared: final `cer l'office de tabellion ou notaire, ſi toutesfois ils` / both reads `cer l'office de tabellion ou noraire, ſi toutesfois ils`
- opus p159 body[32] shared: final `cuteur teſtamentaire, (ſil en y a aucun) de payer, ou` / both reads `cuteur teſtamentaire, (ſ'il en y a aucun) de payer, ou`
- opus p159 body[33] shared: final `faire payer les lais faits aux pauures, ou autres œu` / both reads `faire payer les lais faits aux pauures, ou autres œu-`
- opus p159 note d shared: final `c. dernier de | teſt. au vj. o. | tua nobis, deſ- | ſus allegué.` / both reads `c. dernier de | teſt. au vj. c. | tua nobis, deſ- | ſus allegué.`
- opus p159 note l shared: final `l. tutor qui | D. de admini | ſtra. tutor.` / both reads `l. tutor qui | D. de adminĩ | ſtra. tutor.`
- opus p159 note o shared: final `P. ſi quis au | tem pro redemptione, & P. ſuyuant de eccleſ. titu. cdlla. vij.` / both reads `P. ſi quis au | tem pro redemptione, & P. ſuyuant de eccleſi. titu. cdlla. vij.`
- opus p159 note q shared: final `P. ſi quis igitur non implens de hæredit. & Fal. coll. i. Autene. | hoc amplius, C. de fideicom.` / both reads `P. ſi quis igitur non implens de hæredit. & Bal. coll. i. Autenc. | hoc amplius. C. de fideicom.`
- sonnet p010 body[0] shared: final `Quæ ſimul intonuit, proxima quæque fugat.` / both reads `Quæ ſimul intonuit, proxima quæque fugat {e}.`
- sonnet p010 body[29] shared: final `Ayes ton ami en tel reng. que tu cuides qu'il peut à` / both reads `Ayes ton ami en tel reng, que tu cuides qu'il peut à`
- sonnet p010 body[30] shared: final `l'aduenir eſtre ton ennemy. Paroles ainſi que Scipion eſ` / both reads `l'aduenir eſtre ton ennemy. Paroles ainſi que Scipion eſ-`
- sonnet p010 note l shared: final `Diogenes | Lacree en la | vie de Bias | Prience.` / both reads `Diogenes | Laerce en la | vie de Bias | Prience.`
- sonnet p020 body[13] shared: final `Dont ne faloit ſ'esbahir, ſi la ſuppliãt` / both reads `Dont ne faloit ſ'eſbahir, ſi la ſuppliãt`
- sonnet p020 body[22] shared: final `d'Hermione, pour ſõ Oreſte: de Deianira, pour ſon Her` / both reads `d'Hermione, pour ſõ Oreſte: de Deianira, pour ſon Her-`
- sonnet p020 body[24] shared: final `ſence duquel elle deploroit tant, qu'ayanr apres entend` / both reads `ſence duquel elle deploroit tant, qu'ayant apres entend`
- sonnet p020 body[34] shared: final `voyant, dont l'amour en grec eſt appelé ἐρῶς: car du re-` / both reads `voyant, dont l'amour en grec eſt appelé ἔρως: car du re-`
- sonnet p020 body[35] shared: final `gard, naiſt, & ſe cauſe l'amour, de laquelle les yeux, com` / both reads `gard, naiſt, & ſe cauſe l'amour, de laquelle les yeux, com-`
- sonnet p030 body[4] shared: final `Sichæus ſon couſin germain & mari de Dido ſa ſeut,` / both reads `Sichæus ſon couſin germain & mari de Dido ſa ſeur,`
- sonnet p030 body[19] shared: final `C'eſt pourquoy l'empereur M. Antonin prince gene` / both reads `C'eſt pourquoy l'empereur M. Antonin prince gene-`
- sonnet p030 body[20] shared: final `reux & excellent en toute vertuſ, ne reformida rien en` / both reads `reux & excellent en toute vertu {g}, ne reformida rien en`
- sonnet p030 body[22] shared: final `oncques de ſi grant vehemence, que l'auarice, {g} mere,` / both reads `oncques de ſi grant vehemence, que l'auarice, mere,`
- sonnet p030 body[25] shared: final `toute vertuſ {i}.` / both reads `toute vertu {i}.`
- sonnet p030 body[31] shared: final `ANNOT XV.` / both reads `ANNOT. XV.`
- sonnet p030 note b shared: final `Ciceron au | b. liure de ſa | Rethorique c. | panor. en la x- | xxvij. diſtin- | ction.` / both reads `Ciceron au | b. liure de ſa | Rethorique`
- sonnet p030 note e shared: final `Ciceron en | la ſixieſme | Verrine. Ver | gile au vj. des | Aeneides,` / both reads `Ciceron en | la ſixieſme | Verrine. Ver- | gile au vj. des | Aeneides.`
- sonnet p030 note h shared: final `La j. de Ti | moth. c. ix,` / both reads `La j. de Ti- | moth. c. ix,`
- sonnet p030 note i shared: final `Parag. j. ſur | la fin, vt Iude | ſine quoq. coll. | ij. des nouuel- | les de Iuſtin. | Salluſte.` / both reads `Parag. j. ſur | la fin, vt lude | ſine quoq. coll. | ij. des nouuel- | les de Iuſtin. | Saluſte.`
- sonnet p057 body[7] shared: final `qui n'eſt autre choſe, ſelon l'expoſition des Theologiẽs,` / both reads `qui n'eſt autre choſe, ſelon l'expoſition des Theologiés,`
- sonnet p057 body[22] shared: final `noſtre Sauuenr Ieſus-chriſt, ils diſoiẽt le lapider, nõ pas` / both reads `noſtre Sauueur Ieſus-chriſt, ils diſoiẽt le lapider, nõ pas`
- sonnet p057 body[28] shared: final `ſement gardé: & (cõme dit ſainct Gregoire) ſans vier de` / both reads `ſement gardé: & (cõme dit ſainct Gregoire) ſans vſer de`
- sonnet p057 body[32] shared: final `tirer à peine de mort trop facilemeot vn gliſſement, &` / both reads `tirer à peine de mort trop facilement vn gliſſement, &`
- sonnet p057 note b shared: final `Aut, alearũ | c de religioſ` / both reads `Aut. alearũ`
- sonnet p057 note d shared: final `S. Iean. e. x.` / both reads `S. Iean. c. x.`
- sonnet p057 note e shared: final `Leuitique. | c xiiij.` / both reads `Leuitique. | c. xiiij.`
- sonnet p159 body[18] shared: final `cer l'office de tabellion ou notaire, ſi toutesfois ils` / both reads `cer l'office de tabellion ou noraire, ſi toutesfois ils`
- sonnet p159 body[24] shared: final `du deffunct {m}, ſans lequel on ne pourroit apres recou` / both reads `du deffunct {m}, ſans lequel on ne pourroit apres recou-`
- sonnet p159 body[32] shared: final `cuteur teſtamentaire, (ſil en y a aucun) de payer, ou` / both reads `cuteur teſtamentaire, (ſ'il en y a aucun) de payer, ou`
- sonnet p159 body[33] shared: final `faire payer les lais faits aux pauures, ou autres œu` / both reads `faire payer les lais faits aux pauures, ou autres œu-`
- sonnet p159 note b shared: final `e. tua nobis | c. pẽ. de teſta.` / both reads `e. tua nobis`

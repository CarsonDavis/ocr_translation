#!/usr/bin/env python3
"""Build the classical, patristic and humanist source corpora for the site.

1. Tally: every "## Notes" line of translation/sections/*.md has the shape
   ``- {x} (pNNN): **<identification>** — <sigla>``. The bold identification is matched
   against CATALOG (author + work patterns); each note line counts once per work it names.
   The tally goes to site/data/sources/classical-works.json.
2. Fetch: works with an openly licensed machine-readable text are fetched ONCE into a cache
   directory (sequential, polite delay, descriptive User-Agent): Perseus Digital Library TEI
   (canonical-latinLit / canonical-greekLit on GitHub, CC BY-SA) first, The Latin Library
   (public-domain transcriptions) where Perseus lacks the work. Humanist works with scans
   only are recorded as ``scan-only`` with a scan URL and never fetched.
3. Parse to unit files per docs/sources-contract.md:

    site/data/sources/<work-id>/<unit>.json   one unit per book of the work ("all" for a short
                                              work); oversize units split into <unit>a, <unit>b
    site/data/sources/<work-id>/corpus.json   the corpus's index.json entry (merged elsewhere)

Works are fetched in descending order of citation count until the fetched works cover
COVERAGE of the classical (non-humanist) citations; the rest are reported as skipped.

Usage:
    python3 scripts/fetch_classical.py [--offline] [--cache DIR] [--only id,id] [--all]
    python3 scripts/fetch_classical.py --tally-only
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ROOT / "translation/sections"
OUT = ROOT / "site/data/sources"
DEFAULT_CACHE = Path("/private/tmp/claude-502/-Users-cdavis-github-translator/"
                     "d3de2614-6075-4c4c-8061-e08283898989/scratchpad/sources-cache/classical")
UA = ("Coras-translation-corpus-builder/1.0 (scholarly citation viewer for a 1561 legal text; "
      "sequential, cached, one fetch per file)")
DELAY = 1.0
MAX_UNIT_BYTES = 300_000
COVERAGE = 0.90

PERSEUS_RAW = "https://raw.githubusercontent.com/PerseusDL/canonical-{repo}Lit/master/data/{tg}/{wk}/{urn}.{ed}.xml"
SCAIFE = "https://scaife.perseus.org/library/urn:cts:{ns}:{urn}.{ed}/"
PERSEUS_LICENSE = "CC BY-SA 3.0 (Perseus Digital Library TEI edition)"
LL_BASE = "https://www.thelatinlibrary.com/"
LL_LICENSE = "public domain (transcription of a public-domain edition at The Latin Library)"
LL_ATTRIB = "Text: The Latin Library (thelatinlibrary.com), ed. William L. Carey"


def P(urn: str, ed: str = "perseus-lat2", **kw) -> dict:
    """Perseus canonical TEI source spec."""
    repo = "greek" if urn.startswith("tlg") else "latin"
    return {"type": "perseus", "repo": repo, "urn": urn, "ed": ed, **kw}


def LL(pages: list[str], mode: str, **kw) -> dict:
    """The Latin Library source spec; pages relative to LL_BASE, one page per book."""
    return {"type": "latinlibrary", "pages": pages, "mode": mode, **kw}


def SCAN(url: str) -> dict:
    return {"type": "scan", "url": url}


# ---------------------------------------------------------------- catalog
#
# (work_id, author, work, abbreviation for labels, include regex, source, options)
# Patterns run over one note line's bold identification. ``exclude`` drops lines that only
# mention the work through a later commentator.

def W(wid, author, work, abbr, pat, src=None, **opt):
    """One catalog entry; ``note`` explains a missing source, ``humanist`` marks scan-only."""
    return {"work_id": wid, "author": author, "work": work, "abbr": abbr, "pat": pat,
            "src": src, **opt}


CATALOG: list[dict] = [
    # Cicero
    W("cicero-de-amicitia", "Cicero", "De amicitia (Laelius)", "Cic. Amic.",
      r"De amicitia|\bLaelius\b", P("phi0474.phi052")),
    W("cicero-tusculanae-disputationes", "Cicero", "Tusculanae disputationes", "Cic. Tusc.",
      r"Tusculan", P("phi0474.phi049")),
    W("cicero-de-oratore", "Cicero", "De oratore", "Cic. De or.", r"De oratore",
      P("phi0474.phi037")),
    W("cicero-de-divinatione", "Cicero", "De divinatione", "Cic. Div.",
      r"On Divination|De divinatione", P("phi0474.phi053", "perseus-lat3")),
    W("cicero-de-senectute", "Cicero", "De senectute (Cato maior)", "Cic. Sen.",
      r"De senectute|On Old Age", P("phi0474.phi051")),
    W("cicero-in-verrem", "Cicero", "In Verrem", "Cic. Verr.", r"Verrem|Verrine",
      P("phi0474.phi005")),
    W("cicero-pro-fonteio", "Cicero", "Pro M. Fonteio", "Cic. Font.", r"Fonte(io|ius)",
      P("phi0474.phi007")),
    W("cicero-pro-roscio-amerino", "Cicero", "Pro Sex. Roscio Amerino", "Cic. S. Rosc.",
      r"Roscius of Ameria|Roscio Amerino", P("phi0474.phi002")),
    W("cicero-ad-familiares", "Cicero", "Epistulae ad familiares", "Cic. Fam.",
      r"Letters to his Friends|ad familiares", P("phi0474.phi056")),
    W("cicero-ad-quintum-fratrem", "Cicero", "Epistulae ad Quintum fratrem", "Cic. Q. fr.",
      r"Q\. fratrem|brother Quintus", P("phi0474.phi058")),
    W("cicero-de-finibus", "Cicero", "De finibus bonorum et malorum", "Cic. Fin.",
      r"De finibus", P("phi0474.phi048")),
    W("cicero-academica", "Cicero", "Academica (Lucullus)", "Cic. Luc.", r"Academica",
      P("phi0474.phi046")),
    W("cicero-de-officiis", "Cicero", "De officiis", "Cic. Off.",
      r"Cicero, \*(On Duties|De officiis)", P("phi0474.phi055")),
    W("cicero-pro-caelio", "Cicero", "Pro Caelio", "Cic. Cael.", r"Pro Caelio",
      P("phi0474.phi024")),
    W("cicero-pro-rabirio-postumo", "Cicero", "Pro Rabirio Postumo", "Cic. Rab. Post.",
      r"Rabirio Postumo", P("phi0474.phi029")),
    W("cicero-post-reditum-ad-quirites", "Cicero", "Post reditum ad Quirites", "Cic. Red. Pop.",
      r"Post reditum ad Quirites", P("phi0474.phi019")),
    W("cicero-paradoxa-stoicorum", "Cicero", "Paradoxa Stoicorum", "Cic. Parad.",
      r"Paradoxa Stoicorum", P("phi0474.phi047")),
    W("cicero-in-vatinium", "Cicero", "In Vatinium", "Cic. Vat.", r"In Vatinium",
      P("phi0474.phi023")),
    W("cicero-partitiones-oratoriae", "Cicero", "Partitiones oratoriae", "Cic. Part.",
      r"Partitiones oratoriae", P("phi0474.phi038")),
    W("cicero-de-inventione", "Cicero", "De inventione", "Cic. Inv.", r"De inventione",
      P("phi0474.phi036")),
    # Pliny, Solinus, Valerius Maximus, Gellius
    W("pliny-naturalis-historia", "Pliny the Elder", "Naturalis historia", "Plin. NH",
      r"\bPliny\b(?!'s)|Natural History", P("phi0978.phi001", drop=1)),
    W("solinus-collectanea", "Solinus", "Collectanea rerum memorabilium (Polyhistor)", "Solin.",
      r"\bSolinus\b",
      LL(["solinus1a.html", "solinus2a.html", "solinus3a.html", "solinus4a.html"], "prose",
         unit="all", edition="Mommsen's chapter division (1895), as transcribed at The Latin "
                             "Library; chapters only, no section numbers")),
    W("valerius-maximus-facta-et-dicta", "Valerius Maximus",
      "Facta et dicta memorabilia", "Val. Max.", r"Valerius Maximus",
      P("phi1038.phi001", "perseus-lat1", id_sub=[(r"^(\d+)\.(\d+)e\.", r"\1.\2.ext.")],
        id_note="Perseus's external-example chapters ('9.15e') are written '9.15.ext.N'")),
    W("gellius-noctes-atticae", "Aulus Gellius", "Noctes Atticae", "Gell.",
      r"\bGellius\b|Attic Nights", P("phi1254.phi001")),
    # Augustine, Jerome and other Christian authors
    W("augustine-de-civitate-dei", "Augustine", "De civitate Dei", "Aug. Civ.",
      r"City of God|civitate Dei",
      LL([f"augustine/civ{i}.shtml" for i in range(1, 23)], "prose"), exclude=r"Vives"),
    W("augustine-de-adulterinis-coniugiis", "Augustine", "De adulterinis coniugiis",
      "Aug. Adult. coniug.", r"De adulterinis"),
    W("augustine-contra-faustum", "Augustine", "Contra Faustum Manichaeum", "Aug. c. Faust.",
      r"Contra Faustum"),
    W("jerome-epistulae", "Jerome", "Epistulae", "Hier. Ep.", r"Jerome, \*Epistle",
      note="Perseus (stoa0162.stoa004) and The Latin Library carry selections of the letters "
           "without Ep. 72, the one cited"),
    W("jerome-quaestiones-hebraicae-in-genesim", "Jerome", "Quaestiones hebraicae in Genesim",
      "Hier. Qu. hebr. in Gen.", r"Quaestiones hebraicae"),
    W("jerome-apologia-adversus-rufinum", "Jerome", "Apologia adversus Rufinum",
      "Hier. Ruf.", r"Rufinus"),
    W("jerome-chronicon", "Jerome", "Chronicon (continuation of Eusebius)", "Hier. Chron.",
      r"Jerome's Latin continuation of Eusebius"),
    W("lactantius-divinae-institutiones", "Lactantius", "Divinae institutiones", "Lact. Inst.",
      r"Lactantius"),
    W("tertullian-apologeticum", "Tertullian", "Apologeticum", "Tert. Apol.",
      r"Tertullian, \*Apology",
      LL(["tertullian/tertullian.apol.shtml"], "prose", unit="all")),
    W("ambrose-de-officiis", "Ambrose", "De officiis ministrorum", "Ambr. Off.",
      r"Ambrose, \*De officiis"),
    W("eusebius-historia-ecclesiastica", "Eusebius", "Historia ecclesiastica", "Eus. HE",
      r"Eusebius of Caesarea, \*Ecclesiastical History"),
    W("cassiodorus-variae", "Cassiodorus", "Variae", "Cassiod. Var.", r"Cassiodorus",
      LL([f"cassiodorus/varia{i}.shtml" for i in range(1, 13)], "prose")),
    W("pseudo-clement-epistola-ad-iacobum", "Pseudo-Clement", "Epistola Clementis ad Iacobum",
      "Ps.-Clem. Ep. ad Iac.", r"Pseudo-Clement"),
    # Poets
    W("virgil-aeneid", "Virgil", "Aeneid", "Verg. Aen.", r"Aeneid(?! ?,? which has twelve)",
      P("phi0690.phi003"), exclude=r"printed \"the thirteenth of the Aeneid\""),
    W("virgil-eclogues", "Virgil", "Eclogues", "Verg. Ecl.", r"Eclogue", P("phi0690.phi001")),
    W("virgil-georgics", "Virgil", "Georgics", "Verg. G.", r"Georgics", P("phi0690.phi002")),
    W("ovid-metamorphoses", "Ovid", "Metamorphoses", "Ov. Met.", r"Metamorphoses",
      P("phi0959.phi006")),
    W("ovid-ibis", "Ovid", "Ibis", "Ov. Ib.", r"\bIbis\b", P("phi0959.phi010")),
    W("ovid-heroides", "Ovid", "Heroides (Epistulae)", "Ov. Her.", r"Heroides",
      P("phi0959.phi002")),
    W("ovid-fasti", "Ovid", "Fasti", "Ov. Fast.", r"\bFasti\b", P("phi0959.phi007")),
    W("ovid-ars-amatoria", "Ovid", "Ars amatoria", "Ov. Ars", r"Ars amatoria|Art of Love",
      P("phi0959.phi004")),
    W("horace-odes", "Horace", "Odes (Carmina)", "Hor. Carm.", r"Horace, \*Odes",
      P("phi0893.phi001")),
    W("horace-ars-poetica", "Horace", "Ars poetica", "Hor. Ars", r"Art of Poetry|Ars poetica",
      P("phi0893.phi006")),
    W("propertius-elegies", "Propertius", "Elegies", "Prop.", r"Propertius",
      P("phi0620.phi001", "perseus-lat3")),
    W("plautus-amphitruo", "Plautus", "Amphitruo", "Plaut. Amph.", r"Amphitryon|Amphitruo",
      P("phi0119.phi001")),
    W("plautus-menaechmi", "Plautus", "Menaechmi", "Plaut. Men.", r"Menaechmi",
      P("phi0119.phi010")),
    W("plautus-poenulus", "Plautus", "Poenulus", "Plaut. Poen.", r"Poenulus",
      P("phi0119.phi015")),
    W("plautus-truculentus", "Plautus", "Truculentus", "Plaut. Truc.", r"Truculentus",
      P("phi0119.phi020")),
    W("terence-adelphoe", "Terence", "Adelphoe", "Ter. Ad.", r"Adelphoe", P("phi0134.phi006")),
    W("seneca-de-tranquillitate-animi", "Seneca", "De tranquillitate animi", "Sen. Tranq.",
      r"De tranquillitate animi", LL(["sen/sen.tranq.shtml"], "prose", unit="all")),
    W("seneca-troades", "Seneca", "Troades", "Sen. Tro.", r"Seneca, \*Troades",
      P("phi1017.phi002")),
    W("seneca-elder-controversiae", "Seneca the Elder", "Controversiae", "Sen. Contr.",
      r"Seneca the Elder, \*Controversiae", P("phi1014.phi001", "perseus-lat1")),
    W("juvenal-satires", "Juvenal", "Satires", "Juv.", r"Juvenal", P("phi1276.phi001")),
    W("martial-epigrams", "Martial", "Epigrams", "Mart.", r"Martial, \*Epigrams",
      P("phi1294.phi002")),
    W("statius-thebaid", "Statius", "Thebaid", "Stat. Theb.", r"Thebaid",
      P("phi1020.phi001")),
    W("manilius-astronomica", "Manilius", "Astronomica", "Manil.", r"Manilius",
      LL([f"manilius{i}.html" for i in range(1, 6)], "verse")),
    # Historians and other prose
    W("livy-ab-urbe-condita", "Livy", "Ab urbe condita", "Liv.", r"\bLivy\b",
      P("phi0914.phi001")),
    W("suetonius-de-vita-caesarum", "Suetonius", "De vita Caesarum", "Suet.",
      r"Suetonius",
      {"type": "perseus-multi", "repo": "latin", "ed": "perseus-lat2",
       "parts": [("julius", "phi1348.abo011"), ("augustus", "phi1348.abo012"),
                 ("tiberius", "phi1348.abo013"), ("caligula", "phi1348.abo014"),
                 ("claudius", "phi1348.abo015"), ("nero", "phi1348.abo016"),
                 ("galba", "phi1348.abo017"), ("otho", "phi1348.abo018"),
                 ("vitellius", "phi1348.abo019"), ("vespasian", "phi1348.abo020"),
                 ("titus", "phi1348.abo021"), ("domitian", "phi1348.abo022")]}),
    W("historia-augusta", "Scriptores Historiae Augustae", "Historia Augusta", "SHA",
      r"Historia Augusta|Capitolinus|Spartianus",
      {"type": "perseus-multi", "repo": "latin", "ed": "perseus-lat2",
       "parts": [("hadrian", "phi2331.phi001"), ("aelius", "phi2331.phi002"),
                 ("antoninus-pius", "phi2331.phi003"), ("marcus", "phi2331.phi004"),
                 ("verus", "phi2331.phi005"), ("avidius-cassius", "phi2331.phi006"),
                 ("commodus", "phi2331.phi007"), ("pertinax", "phi2331.phi008"),
                 ("didius-julianus", "phi2331.phi009"), ("severus", "phi2331.phi010"),
                 ("pescennius-niger", "phi2331.phi011"), ("clodius-albinus", "phi2331.phi012"),
                 ("caracalla", "phi2331.phi013"), ("geta", "phi2331.phi014"),
                 ("macrinus", "phi2331.phi015"), ("diadumenianus", "phi2331.phi016")]}),
    W("justin-epitome", "Justin", "Epitoma historiarum Philippicarum Pompei Trogi", "Iust.",
      r"\bJustin\b(?!ian| I\b)",
      LL([f"justin/{i}.html" for i in range(1, 45)], "prose")),
    W("macrobius-saturnalia", "Macrobius", "Saturnalia", "Macr. Sat.", r"Saturnalia"),
    W("macrobius-in-somnium-scipionis", "Macrobius", "Commentarii in Somnium Scipionis",
      "Macr. In Somn.", r"Dream of Scipio"),
    W("varro-antiquitates", "Varro", "Antiquitates rerum divinarum (lost)", "Varro Ant.",
      r"Varro"),
    W("pseudo-sallust-in-ciceronem", "Pseudo-Sallust", "In M. Tullium Ciceronem invectiva",
      "[Sall.] Cic.", r"Pseudo-Sallust", LL(["sall.invectiva.html"], "prose", unit="all")),
    # Greek
    W("aristotle-nicomachean-ethics", "Aristotle", "Nicomachean Ethics", "Arist. EN",
      r"Nicomachean", P("tlg0086.tlg010", "perseus-grc2")),
    W("aristotle-history-of-animals", "Aristotle", "Historia animalium", "Arist. HA",
      r"History of Animals"),
    W("aristotle-generation-of-animals", "Aristotle", "De generatione animalium", "Arist. GA",
      r"Generation of Animals"),
    W("aristotle-politics", "Aristotle", "Politics", "Arist. Pol.", r"Aristotle, \*Politics",
      P("tlg0086.tlg035", "perseus-grc2", drop=2)),
    W("aristotle-problems", "Aristotle", "Problemata", "[Arist.] Pr.", r"Aristotle, \*Problems"),
    W("aristotle-metaphysics", "Aristotle", "Metaphysics", "Arist. Metaph.",
      r"Aristotle, \*Metaphysics", P("tlg0086.tlg025", "perseus-grc2")),
    W("plutarch-lives", "Plutarch", "Parallel Lives", "Plut.", r"Plutarch, \*Life of",
      {"type": "perseus-multi", "repo": "greek", "ed": "perseus-grc2",
       "parts": [("romulus", "tlg0007.tlg002"), ("lycurgus", "tlg0007.tlg004"),
                 ("pericles", "tlg0007.tlg012"), ("pyrrhus", "tlg0007.tlg030"),
                 ("alexander", "tlg0007.tlg047"), ("cicero", "tlg0007.tlg055")]}),
    W("plutarch-parallela-minora", "Plutarch", "Parallela minora (Moralia)", "[Plut.] Par. min.",
      r"Parallela minora"),
    W("pseudo-plutarch-de-placitis", "Pseudo-Plutarch", "De placitis philosophorum",
      "[Plut.] Plac.", r"placitis philosophorum"),
    W("josephus-antiquitates", "Josephus", "Antiquitates Iudaicae", "Joseph. AJ",
      r"Jewish Antiquities", P("tlg0526.tlg001", "perseus-grc2")),
    W("homer-odyssey", "Homer", "Odyssey", "Hom. Od.", r"Odyssey",
      P("tlg0012.tlg002", "perseus-grc2")),
    W("homer-iliad", "Homer", "Iliad", "Hom. Il.", r"Homer, \*Iliad|adapts Homer, \*Iliad",
      P("tlg0012.tlg001", "perseus-grc2")),
    W("herodotus-histories", "Herodotus", "Histories", "Hdt.", r"Herodotus",
      P("tlg0016.tlg001", "perseus-grc2")),
    W("strabo-geography", "Strabo", "Geography", "Str.", r"Strabo",
      P("tlg0099.tlg001", "perseus-grc2")),
    W("plato-republic", "Plato", "Republic", "Pl. Resp.", r"Plato, \*Republic",
      P("tlg0059.tlg030", "perseus-grc2")),
    W("plato-laws", "Plato", "Laws", "Pl. Leg.", r"\*Laws\* book 6",
      P("tlg0059.tlg034", "perseus-grc2")),
    W("plato-phaedo", "Plato", "Phaedo", "Pl. Phd.", r"Plato, \*Phaedo",
      P("tlg0059.tlg004", "perseus-grc2")),
    W("galen", "Galen", "De usu partium; De semine", "Gal.", r"\bGalen\b"),
    W("euripides-medea", "Euripides", "Medea", "Eur. Med.", r"Euripides, \*Medea",
      P("tlg0006.tlg003", "perseus-grc2")),
    W("euripides-alcestis", "Euripides", "Alcestis", "Eur. Alc.", r"Euripides, \*Alcestis",
      P("tlg0006.tlg002", "perseus-grc2")),
    W("diogenes-laertius", "Diogenes Laertius", "Vitae philosophorum", "Diog. Laert.",
      r"Diogenes Laertius", P("tlg0004.tlg001", "perseus-grc2", drop=1)),
    W("dio-cassius-roman-history", "Dio Cassius", "Roman History", "Cass. Dio",
      r"Dio Cassius", note="the Perseus Greek text (tlg0385.tlg001) has only books 36-55; "
                           "the cited book 69 (Xiphilinus's epitome) is not in it"),
    W("appian-syriaca", "Appian", "Syriaca", "App. Syr.", r"Syrian Wars",
      P("tlg0551.tlg013", "perseus-grc2", drop=0)),
    W("appian-mithridatica", "Appian", "Mithridatica", "App. Mith.", r"Mithridatic Wars",
      P("tlg0551.tlg014", "perseus-grc2", drop=0)),
    W("aelian-varia-historia", "Aelian", "Varia historia", "Ael. VH", r"Aelian, \*Varia",
      P("tlg0545.tlg002", "perseus-grc2")),
    W("pausanias-description-of-greece", "Pausanias", "Description of Greece", "Paus.",
      r"Pausanias", P("tlg0525.tlg001", "perseus-grc2")),
    W("iamblichus-de-mysteriis", "Iamblichus", "De mysteriis", "Iambl. Myst.", r"Iamblichus"),
    W("alexander-aphrodisias-problemata", "Alexander of Aphrodisias", "Problemata",
      "[Alex. Aphr.] Probl.", r"Alexander of Aphrodisias"),
    W("callimachus-epigrams", "Callimachus", "Epigrams", "Call. Epigr.", r"Callimachus"),
    # Humanists (scans only)
    W("crinito-de-honesta-disciplina", "Pietro Crinito", "De honesta disciplina",
      "Crinit. De hon. disc.", r"Crinit", humanist=True),
    W("fregoso-de-dictis-factisque-memorabilibus", "Battista Fregoso",
      "De dictis factisque memorabilibus collectanea", "Fulgos.", r"Fregoso|Fulgosus",
      humanist=True),
    W("paolo-emili-de-rebus-gestis-francorum", "Paolo Emili", "De rebus gestis Francorum",
      "Paul. Aemil.", r"Paolo Emili|Paulus Aemilius", humanist=True),
    W("platina-vitae-pontificum", "Platina", "Liber de vita Christi ac omnium pontificum",
      "Platina", r"Platina", humanist=True),
    W("volaterranus-commentarii-urbani", "Raffaele Maffei (Volaterranus)",
      "Commentariorum urbanorum libri XXXVIII", "Volat.", r"Volaterr|Maffei", humanist=True),
    W("rhodiginus-lectiones-antiquae", "Lodovico Ricchieri (Caelius Rhodiginus)",
      "Lectionum antiquarum libri", "Rhodig.", r"Rhodigin|Ricchieri", humanist=True),
    W("polydore-vergil-de-inventoribus-rerum", "Polydore Vergil", "De inventoribus rerum",
      "Pol. Verg.", r"Polydore", humanist=True),
    W("galeotto-marzio-de-doctrina-promiscua", "Galeotto Marzio", "De doctrina promiscua",
      "Galeot. Mart.", r"Galeotto Marzio", humanist=True),
    W("alexander-ab-alexandro-dies-geniales", "Alessandro Alessandri (Alexander ab Alexandro)",
      "Genialium dierum libri sex", "Alex. ab Alex.", r"Alexander ab Alexandro",
      humanist=True),
    W("munster-cosmographia", "Sebastian Münster", "Cosmographia universalis", "Münster",
      r"Münster", humanist=True),
    W("andrelini-livia", "Publio Fausto Andrelini", "Livia (Amores)", "Faust. Andrel.",
      r"Andrelini", humanist=True),
    W("vives-commentary-de-civitate-dei", "Juan Luis Vives",
      "Commentarii in De civitate Dei", "Vives", r"Vives", humanist=True),
    W("bude-annotationes-in-pandectas", "Guillaume Budé", "Annotationes in Pandectas", "Budé",
      r"Budé|Budée", humanist=True),
]

# Scan URLs for humanist works (Internet Archive; page images only, never fetched). Editions
# are 16th-century where one was found; the chapter numbering Coras cites may differ.
IA = "https://archive.org/details/"
SCANS: dict[str, str] = {
    "crinito-de-honesta-disciplina": IA + "bub_gb_954MRWBwQSAC",  # Lyon 1561
    "fregoso-de-dictis-factisque-memorabilibus": IA + "baptistefulgosid00freg",  # 1518
    "paolo-emili-de-rebus-gestis-francorum": IA + "bub_gb_b89bjW1HheEC",  # Paris 1543
    "platina-vitae-pontificum": IA + "bub_gb_yC2IsGhJtqYC",  # 1540
    "volaterranus-commentarii-urbani": IA + "ned-kbn-all-00002512-001",  # 1552
    "rhodiginus-lectiones-antiquae": IA + "bub_gb_EwjmCxHKgN0C",  # 1560
    "polydore-vergil-de-inventoribus-rerum": IA + "kpbc.umk.pl.Publikacja-WiMBP-070607_204379",
    "galeotto-marzio-de-doctrina-promiscua": IA + "bub_gb_8CB0TmdH6oMC",  # Florence 1548
    "alexander-ab-alexandro-dies-geniales": IA + "bub_gb_sSjxKpodtgYC",  # 1586
    "munster-cosmographia": IA + "bub_man_11b03d11d783622cfdb59473f35bcef0",  # 1544
    "vives-commentary-de-civitate-dei": IA + "bub_gb_dT4eNG3BJWkC",  # 1596, with Vives
    "bude-annotationes-in-pandectas": IA + "bub_gb_Yi67rXVfrQUC",  # 1535
}


# ---------------------------------------------------------------- tally

NOTE_RE = re.compile(r"^- \{[^}]+\} \((p\d+)\): \*\*(.+?)\*\* —")


def note_identifications(sections: Path = SECTIONS) -> list[tuple[str, str, str]]:
    """(section file stem, page, identification) for every Notes line."""
    out = []
    for f in sorted(sections.glob("*.md")):
        in_notes = False
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                in_notes = line.strip() == "## Notes"
                continue
            if not in_notes:
                continue
            m = NOTE_RE.match(line)
            if m:
                out.append((f.stem, m.group(1), m.group(2)))
    return out


def tally(ids: list[tuple[str, str, str]], catalog: list[dict] = CATALOG) -> dict[str, int]:
    counts = {w["work_id"]: 0 for w in catalog}
    for _sec, _page, ident in ids:
        for w in catalog:
            if re.search(w["pat"], ident) and not (w.get("exclude")
                                                    and re.search(w["exclude"], ident)):
                counts[w["work_id"]] += 1
    return counts


# ---------------------------------------------------------------- fetching

class Fetcher:
    def __init__(self, cache: Path, offline: bool):
        self.cache, self.offline = cache, offline
        self.cache.mkdir(parents=True, exist_ok=True)
        self._last = 0.0
        self.missing: list[str] = []
        self.fetched = 0

    @staticmethod
    def cache_name(url: str) -> str:
        name = re.sub(r"^https?://", "", url)
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

    def get(self, url: str) -> bytes | None:
        path = self.cache / self.cache_name(url)
        if path.exists():
            return path.read_bytes()
        if self.offline:
            self.missing.append(url)
            return None
        wait = DELAY - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
        except Exception as e:  # noqa: BLE001 - report, never guess text
            print(f"  fetch failed {url}: {e}", file=sys.stderr)
            self.missing.append(url)
            self._last = time.time()
            return None
        self._last = time.time()
        path.write_bytes(data)
        self.fetched += 1
        return data


# ---------------------------------------------------------------- text helpers

def norm_ws(s: str) -> str:
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", s)]
    return "\n\n".join(p for p in paras if p)


ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s: str) -> int | None:
    s = s.upper()
    if not s or any(c not in ROMAN for c in s):
        return None
    total, prev = 0, 0
    for c in reversed(s):
        v = ROMAN[c]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def num_key(x: str):
    m = re.match(r"(\d+)", x)
    return (int(m.group(1)) if m else 10 ** 6, x)


# ---------------------------------------------------------------- Perseus TEI

TEI = "{http://www.tei-c.org/ns/1.0}"
XMLNS = "{http://www.w3.org/XML/1998/namespace}"
SKIP_TAGS = {"note", "bibl", "head", "speaker", "del", "gap", "figure", "castList", "argument",
             "trailer", "ref", "index", "interpGrp", "fw"}
SKIP_IN_CHOICE = {"orig", "sic", "abbr"}
BLOCK_TAGS = {"p", "sp", "lg", "quote", "div"}


def local(tag) -> str:
    return tag.split("}", 1)[1] if isinstance(tag, str) and "}" in tag else str(tag)


def tei_levels(root: ET.Element) -> tuple[list[str], bool, int]:
    """Citation scheme from the deepest refsDecl cRefPattern.

    Returns (one entry per textpart-div citation level, last_level_is_line, skip). The
    pattern's replacementPattern xpath names one step per level: tei:div[@n='$i'] (a textpart
    div) or tei:l[@n='$i'] (a verse line, possibly via //). ``skip`` counts the plain
    tei:div steps between the edition div and the first cited level (Livy's decades).
    """
    best = None
    for rd in root.iter(f"{TEI}refsDecl"):
        for cp in rd.iter(f"{TEI}cRefPattern"):
            rp = cp.get("replacementPattern", "")
            depth = len(re.findall(r"\$\d", rp))
            if best is None or depth > best[0]:
                best = (depth, rp)
    if best is None:
        return [], False, 0
    rp = best[1]
    tail = rp.split("tei:body", 1)[-1]
    steps = re.findall(r"tei:(\w+)(\[[^\]]*\])?", tail)
    cited = [name for name, pred in steps if "$" in pred]
    plain_divs = 0
    for name, pred in steps:
        if "$" in pred:
            break
        if name == "div":
            plain_divs += 1
    is_line = bool(cited) and cited[-1] == "l"
    return [c for c in cited if c == "div"], is_line, max(plain_divs - 1, 0)


class TeiWalker:
    """Walk a Perseus edition div into passages keyed by citation path."""

    def __init__(self, n_div_levels: int, is_line: bool, ms_unit: str | None = None):
        self.n_div = n_div_levels
        self.is_line = is_line
        self.ms_unit = ms_unit
        self.ms: str | None = None
        self.path: list[str] = []
        self.buf: list[str] = []
        self.order: list[tuple[str, ...]] = []
        self.texts: dict[tuple[str, ...], list[str]] = {}
        self.line: str | None = None
        self.subtypes: dict[int, str] = {}

    def key(self) -> tuple[str, ...] | None:
        if len(self.path) < self.n_div:
            return None
        k = tuple(self.path[:self.n_div])
        if self.ms_unit:
            k = k + (self.ms or "1",)
        if self.is_line:
            if self.line is None:
                return None
            k = k + (self.line,)
        return k

    def flush(self):
        text = "".join(self.buf)
        self.buf = []
        if not text.strip():
            return
        k = self.key()
        if k is None:
            return
        if k not in self.texts:
            self.texts[k] = []
            self.order.append(k)
        self.texts[k].append(text)

    def text(self, s: str | None):
        if s:
            self.buf.append(s)

    def walk(self, el: ET.Element, depth: int):
        tag = local(el.tag)
        if tag in SKIP_TAGS:
            return  # tail handled by the caller
        if tag == "div" and el.get("type") == "textpart" and depth < self.n_div:
            self.flush()
            saved = (list(self.path), self.line)
            self.path = self.path[:depth] + [el.get("n", "")]
            self.subtypes[depth] = el.get("subtype", "")
            self.line = None
            self.text(el.text)
            for ch in el:
                self.walk(ch, depth + 1)
                self.text(ch.tail)
            self.flush()
            self.path, self.line = saved
            return
        if tag == "l" and self.is_line:
            self.flush()
            n = el.get("n")
            part = el.get("part")
            if n:
                self.line = n
            elif part in ("M", "F") and self.line is not None:
                pass
            elif self.line is not None and re.fullmatch(r"\d+", self.line):
                self.line = str(int(self.line) + 1)
            elif self.line is None:
                self.line = "1"
            self.text(el.text)
            for ch in el:
                self.walk(ch, depth)
                self.text(ch.tail)
            self.buf.append(" ")
            self.flush()
            return
        if tag in ("lb",):
            self.buf.append(" ")
            return
        if tag == "milestone" and self.ms_unit and el.get("unit") == self.ms_unit \
                and len(self.path) >= self.n_div and el.get("n"):
            self.flush()
            self.ms = el.get("n")
            return
        if tag == "choice":
            kids = [local(c.tag) for c in el]
            prefer_skip = SKIP_IN_CHOICE if set(kids) - SKIP_IN_CHOICE else set()
            for ch in el:
                if local(ch.tag) not in prefer_skip:
                    self.walk(ch, depth)
                self.text(ch.tail)
            return
        self.text(el.text)
        for ch in el:
            self.walk(ch, depth)
            self.text(ch.tail)
        if tag in BLOCK_TAGS:
            self.buf.append("\n\n")


def parse_tei(data: bytes) -> dict:
    """Perseus TEI -> {"levels": [...], "is_line": bool, "subtypes": {...},
    "passages": [(path tuple, text)], "title": str}."""
    root = ET.fromstring(data)
    levels, is_line, skip = tei_levels(root)
    body = root.find(f"{TEI}text/{TEI}body")
    if body is None:
        body = root.find(f".//{TEI}body")
    edition = None
    for d in body.iter(f"{TEI}div"):
        if d.get("type") in ("edition", "translation"):
            edition = d
            break
    if edition is None:
        edition = body
    roots = [edition]
    for _ in range(skip):
        roots = [c for r in roots for c in r
                 if local(c.tag) == "div" and c.get("type") == "textpart"]
    subtypes_present = {d.get("subtype") for d in edition.iter(f"{TEI}div")}
    ms_units = {m.get("unit") for m in edition.iter(f"{TEI}milestone")}
    ms_unit = ("section" if not is_line and "section" in ms_units
               and "section" not in subtypes_present else None)
    w = TeiWalker(len(levels), is_line, ms_unit)
    for r in roots:
        if r is not edition:
            w.flush()
        for ch in r:
            w.walk(ch, 0)
            w.text(ch.tail)
    w.flush()
    title_el = root.find(f".//{TEI}titleStmt/{TEI}title")
    title = norm_ws("".join(title_el.itertext())) if title_el is not None else ""
    passages = [(k, norm_ws(" ".join(w.texts[k]))) for k in w.order]
    passages = [(k, t) for k, t in passages if t]
    return {"levels": levels, "is_line": is_line, "subtypes": w.subtypes, "ms_unit": ms_unit,
            "passages": passages, "title": title}


# ---------------------------------------------------------------- The Latin Library HTML

def ll_clean(fragment: str) -> str:
    s = re.sub(r"(?is)<br\s*/?>", "\n", fragment)
    s = re.sub(r"(?is)<[^>]+>", "", s)
    s = html.unescape(s).replace("\xa0", " ")
    # editorial angle brackets (supplements) become ⟨ ⟩ so no text can look like markup
    return s.replace("<", "⟨").replace(">", "⟩")


def ll_body(src: str) -> str:
    """Strip head, page heading, borders and the trailing navigation links."""
    s = re.sub(r"(?is)^.*?<body[^>]*>", "", src)
    s = re.sub(r"(?is)<p class=\"?(pagehead|border|smallborder|margin|shortborder)\"?>.*?</p>",
               "", s)
    s = re.sub(r"(?is)<div class=\"?footer\"?>.*$", "", s)
    s = re.sub(r"(?is)<table.*?</table>", "", s)
    s = re.sub(r"(?is)</body>.*$", "", s)
    return s


def ll_paragraphs(src: str) -> list[str]:
    body = ll_body(src)
    parts = re.split(r"(?is)<p[^>]*>|</p>", body)
    out = []
    for p in parts:
        if re.search(r"(?i)<a href=\"?(index|classics|christian|misc|[a-z]+\.html)\"?>", p) \
                and len(ll_clean(p).strip()) < 120:
            continue  # navigation footer
        out.append(p)
    return out


def parse_ll_prose(src: str) -> list[tuple[str, str]]:
    """Latin Library prose, one page per book -> [(chapter[.section], text)].

    Chapter marks: a bold heading opening with a Roman numeral (<b>IV.</b>, Solinus, the
    letters of Cassiodorus), a bracketed Roman numeral ([IV], Augustine, Justin), or a Roman
    numeral with a full stop opening a paragraph (IV., Seneca, later books of Justin).
    Section marks: <font size=2>N</font> or a bracketed Arabic numeral ([3]). Within a chapter
    that has section marks, text before the first mark is section 1. Text before the first
    chapter mark (a preface) is chapter 'pr'.
    """
    raw = "\n\n".join(ll_paragraphs(src))
    raw = re.sub(r"(?is)<b>\s*([IVXLC]+)(?:\.(.*?))?\s*</b>",
                 lambda m: f"[[C{roman_to_int(m.group(1))}]] {m.group(2) or ''}", raw)
    raw = re.sub(r"(?is)<b>\s*(LIBER|LIBRI)\b.*?</b>", "", raw)
    raw = re.sub(r"(?is)<div[^>]*>\s*LIBER [IVXLC]+\s*</div>", "", raw)
    raw = re.sub(r"(?is)<font size=\"?2\"?>\s*(\d+)\s*</font>", r"[[S\1]]", raw)
    text = ll_clean(raw)
    text = re.sub(r"\[\s*([IVXLC]+)\s*\]", lambda m: f"[[C{roman_to_int(m.group(1))}]]", text)
    text = re.sub(r"\[\s*(\d+)\s*\]", r"[[S\1]]", text)
    text = re.sub(r"(?m)^\s*([IVXL]+)\.\s+", lambda m: f"[[C{roman_to_int(m.group(1))}]] ", text)
    chapters: list[tuple[str, list[tuple[str | None, str]]]] = []
    chap, sec = "pr", None
    chapters.append((chap, []))
    for tok in re.split(r"(\[\[[CS]\d+\]\])", text):
        m = re.fullmatch(r"\[\[([CS])(\d+)\]\]", tok)
        if m and m.group(1) == "C":
            chap, sec = m.group(2), None
            chapters.append((chap, []))
        elif m:
            sec = m.group(2)
        elif tok.strip():
            chapters[-1][1].append((sec, tok))
    out: dict[str, str] = {}
    for chap, pieces in chapters:
        has_sec = any(sc is not None for sc, _ in pieces)
        for sc, t in pieces:
            pid = f"{chap}.{sc or '1'}" if has_sec else chap
            out[pid] = (out[pid] + " " + t) if pid in out else t
    rows = [(k, norm_ws(v)) for k, v in out.items()]
    return [(k, v) for k, v in rows if v and not (k.startswith("pr") and len(v) < 40)]


def parse_ll_verse(src: str) -> list[tuple[str, str]]:
    """Verse with <br> line ends and a line number every fifth line -> [(line, text)]."""
    body = "\n".join(ll_paragraphs(src))
    body = re.sub(r"(?is)<font size=\"?2\"?>\s*(\d+)\s*</font>", r"[[N\1]]", body)
    lines = [ln for ln in re.split(r"(?i)<br\s*/?>|\n\s*\n", body)]
    out: list[tuple[str, str]] = []
    n = 0
    for ln in lines:
        m = re.search(r"\[\[N(\d+)\]\]", ln)
        t = norm_ws(ll_clean(re.sub(r"\[\[N\d+\]\]", "", ln)))
        if not t:
            continue
        n += 1
        if m:
            n = int(m.group(1))
        out.append((str(n), t))
    return out


LL_PARSERS = {"prose": parse_ll_prose, "verse": parse_ll_verse}


# ---------------------------------------------------------------- units

def unit_bytes(u: dict) -> int:
    return len(json.dumps(u, ensure_ascii=False).encode("utf-8"))


def group_units(work: dict, rows: list[tuple[tuple[str, ...], str]], unit_levels: int,
                prefix: tuple[str, ...] = (), part_label: str = "",
                drop: int | None = None, drop_name: str = "",
                id_sub: list | None = None) -> list[dict]:
    """rows of (path, text) -> unit dicts. The first ``unit_levels`` path components name the
    unit ("all" when 0); the passage id is the whole path (prefix + path) joined by dots.
    ``drop`` removes one path level from the id (a chapter level that the work's standard
    citation ignores, e.g. Pliny's book.section) and shows it in the label instead."""
    units: dict[str, dict] = {}
    order: list[str] = []
    for path, text in rows:
        full = prefix + path
        uid = ".".join(full[:len(prefix) + unit_levels]) if (prefix or unit_levels) else "all"
        if uid not in units:
            utitle = f"{work['abbr']} {part_label or uid}".strip() if uid != "all" else work["abbr"]
            units[uid] = {"corpus": work["work_id"], "unit": uid,
                          "title": f"{work['author']}, {work['work']}"
                                   + ("" if uid == "all" else f", {utitle.split(' ', 1)[-1]}"
                                      if not part_label else f", {part_label}"),
                          "passages": []}
            order.append(uid)
        note = ""
        if drop is not None and len(path) > drop:
            val = re.sub(r"^[A-Za-z_]+_(?=\d)", "", path[drop])
            note = f" ({drop_name} {val})"
            path = path[:drop] + path[drop + 1:]
            full = prefix + path
        for pat, rep in id_sub or []:
            path = tuple(re.sub(pat, rep, ".".join(path)).split("."))
            full = prefix + path
        pid = ".".join(full)
        shown = ".".join(path) if part_label else pid
        label = f"{work['abbr']} {part_label + ' ' if part_label else ''}{shown}{note}".strip()
        units[uid]["passages"].append({"id": pid, "label": label, "text": text})
    return [units[u] for u in order]


def split_unit(u: dict, limit: int = MAX_UNIT_BYTES) -> list[dict]:
    """Split an oversize unit at passage boundaries into <unit>a, <unit>b, ..."""
    if unit_bytes(u) <= limit:
        return [u]
    shell = unit_bytes({**u, "passages": []}) + 200
    groups, cur, size = [], [], shell
    for p in u["passages"]:
        psz = len(json.dumps(p, ensure_ascii=False).encode()) + 2
        if cur and size + psz > limit:
            groups.append(cur)
            cur, size = [], shell
        cur.append(p)
        size += psz
    if cur:
        groups.append(cur)
    out = []
    for i, g in enumerate(groups):
        out.append({**u, "unit": f"{u['unit']}{'abcdefghijklmnopqrstuvwxyz'[i]}",
                    "parent_unit": u["unit"],
                    "title": u["title"] + f" (part {i + 1} of {len(groups)})",
                    "passage_range": [g[0]["id"], g[-1]["id"]], "passages": g})
    return out


def merge_dup_ids(u: dict) -> dict:
    seen: dict[str, dict] = {}
    out = []
    for p in u["passages"]:
        if p["id"] in seen:
            seen[p["id"]]["text"] += "\n\n" + p["text"]
        else:
            seen[p["id"]] = p
            out.append(p)
    u["passages"] = out
    return u


def validate_unit(u: dict, corpus: str) -> list[str]:
    errs = []
    if u.get("corpus") != corpus:
        errs.append(f"{u.get('unit')}: corpus mismatch")
    ids = [p["id"] for p in u["passages"]]
    if len(ids) != len(set(ids)):
        errs.append(f"{u['unit']}: duplicate passage ids")
    if not ids:
        errs.append(f"{u['unit']}: no passages")
    for p in u["passages"]:
        if re.search(r"<\s*/?[a-zA-Z]+[^>]*>", p["text"]):
            errs.append(f"{u['unit']}: markup in {p['id']}")
            break
    if unit_bytes(u) > MAX_UNIT_BYTES:
        errs.append(f"{u['unit']}: {unit_bytes(u)} bytes > {MAX_UNIT_BYTES}")
    return errs


def unit_sort_key(uid: str):
    return tuple((int(x) if x.isdigit() else 10 ** 6, x) for x in re.findall(r"\d+|[a-z-]+", uid))


# ---------------------------------------------------------------- per-source builders

def perseus_url(urn: str, ed: str) -> str:
    repo = "greek" if urn.startswith("tlg") else "latin"
    tg, wk = urn.split(".")
    return PERSEUS_RAW.format(repo=repo, tg=tg, wk=wk, urn=urn, ed=ed)


def scaife_url(urn: str, ed: str) -> str:
    ns = "greekLit" if urn.startswith("tlg") else "latinLit"
    return SCAIFE.format(ns=ns, urn=urn, ed=ed)


BOOKISH = {"book", "poem", "actio", "Book", "liber"}


def default_unit_levels(parsed: dict) -> int:
    """Unit = book (or poem) when the first citation level is one; otherwise one 'all' unit."""
    n_div = len(parsed["levels"])
    first = parsed["subtypes"].get(0, "")
    if n_div == 0:
        return 0
    if first == "actio" and n_div >= 3:
        return 2
    if first not in BOOKISH:
        return 0
    if n_div == 1 and not parsed["is_line"] and not parsed.get("ms_unit"):
        return 0
    return 1


def build_perseus(work: dict, f: Fetcher) -> tuple[list[dict], dict, list[str]]:
    src = work["src"]
    url = perseus_url(src["urn"], src["ed"])
    data = f.get(url)
    if data is None:
        return [], {}, [f"not fetched: {url}"]
    parsed = parse_tei(data)
    ul = src.get("unit_levels", default_unit_levels(parsed))
    drop = src.get("drop")
    scheme_names = [parsed["subtypes"].get(i, "div") for i in range(len(parsed["levels"]))]
    if parsed.get("ms_unit"):
        scheme_names.append(parsed["ms_unit"])
    units = group_units(work, parsed["passages"], ul, drop=drop,
                        drop_name=scheme_names[drop].replace("_", " ") if drop is not None
                        else "", id_sub=src.get("id_sub"))
    if parsed["is_line"]:
        scheme_names.append("line")
    if drop is not None:
        dropped = scheme_names.pop(drop)
        scheme_names[-1] += f" (the {dropped} is given in the label)"
    meta = {"edition": f"{parsed['title']} — Perseus Digital Library TEI ({src['urn']}."
                       f"{src['ed']})",
            "url": scaife_url(src["urn"], src["ed"]), "source_file": url,
            "language": "grc" if src["urn"].startswith("tlg") else "la",
            "license": PERSEUS_LICENSE,
            "attribution": "Text: Perseus Digital Library, Tufts University (CC BY-SA 3.0), "
                           "canonical-" + src["repo"] + "Lit TEI " + src["urn"] + "." + src["ed"],
            "unit_scheme": ".".join(scheme_names[:ul]) if ul else "work",
            "passage_scheme": ".".join(scheme_names)}
    if src.get("id_note"):
        meta["passage_note"] = src["id_note"]
    return units, meta, []


def build_perseus_multi(work: dict, f: Fetcher) -> tuple[list[dict], dict, list[str]]:
    src = work["src"]
    units, errs, files, names = [], [], [], None
    for part, urn in src["parts"]:
        url = perseus_url(urn, src["ed"])
        data = f.get(url)
        if data is None:
            errs.append(f"not fetched: {url}")
            continue
        parsed = parse_tei(data)
        names = [parsed["subtypes"].get(i, "div") for i in range(len(parsed["levels"]))]
        if parsed.get("ms_unit"):
            names.append(parsed["ms_unit"])
        label = part.replace("-", " ").title()
        got = group_units(work, parsed["passages"], 0, prefix=(part,), part_label=label)
        for u in got:
            u["title"] = f"{work['author']}, {work['work']}: {label}"
        units += got
        files.append(url)
    lang = "grc" if src["parts"][0][1].startswith("tlg") else "la"
    meta = {"edition": f"Perseus Digital Library TEI, one file per part ({src['ed']})",
            "url": scaife_url(src["parts"][0][1], src["ed"]).rsplit(".", 2)[0] + "/",
            "source_files": files, "language": lang, "license": PERSEUS_LICENSE,
            "attribution": "Text: Perseus Digital Library, Tufts University (CC BY-SA 3.0), "
                           "canonical-" + src["repo"] + "Lit TEI",
            "unit_scheme": "part (life name, lower-case, hyphenated)",
            "passage_scheme": "part." + ".".join(names or ["chapter", "section"])}
    meta["url"] = "https://scaife.perseus.org/library/"
    return units, meta, errs


def build_latinlibrary(work: dict, f: Fetcher) -> tuple[list[dict], dict, list[str]]:
    src = work["src"]
    parser = LL_PARSERS[src["mode"]]
    rows: list[tuple[tuple[str, ...], str]] = []
    errs = []
    single = src.get("unit") == "all"
    for i, page in enumerate(src["pages"], start=1):
        data = f.get(LL_BASE + page)
        if data is None:
            errs.append(f"not fetched: {page}")
            continue
        txt = data.decode("utf-8", errors="replace") if b"charset=utf-8" in data.lower() \
            else data.decode("latin-1")
        got = parser(txt)
        if not got:
            errs.append(f"no passages parsed: {page}")
        for pid, text in got:
            path = tuple(pid.split(".")) if single else (str(i),) + tuple(pid.split("."))
            rows.append((path, text))
    ul = 0 if single else 1
    units = group_units(work, rows, ul)
    ids = [r[0] for r in rows]
    if src["mode"] == "verse":
        scheme = "line"
    elif any(len(i) > (1 if single else 2) for i in ids):
        scheme = "chapter[.section]"
    else:
        scheme = "chapter"
    meta = {"edition": src.get("edition", "The Latin Library transcription"),
            "passage_note": "chapter 'pr' is a preface; Roman chapter numerals are given in "
                            "Arabic",
            "url": LL_BASE + src["pages"][0], "source_files": [LL_BASE + p for p in src["pages"]],
            "language": "la", "license": LL_LICENSE, "attribution": LL_ATTRIB,
            "unit_scheme": "work" if single else "book",
            "passage_scheme": scheme if single else f"book.{scheme}"}
    return units, meta, errs


BUILDERS = {"perseus": build_perseus, "perseus-multi": build_perseus_multi,
            "latinlibrary": build_latinlibrary}


def write_corpus(work: dict, units: list[dict], meta: dict) -> dict:
    d = OUT / work["work_id"]
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.json"):
        old.unlink()
    final, splits, errs = [], {}, []
    for u in units:
        merge_dup_ids(u)
        parts = split_unit(u, MAX_UNIT_BYTES - 8_000)  # headroom for the written newlines
        if len(parts) > 1:
            splits[u["unit"]] = [{"unit": p["unit"], "passage_range": p["passage_range"]}
                                 for p in parts]
        final.extend(parts)
    for u in final:
        errs += validate_unit(u, work["work_id"])
        (d / f"{u['unit']}.json").write_text(json.dumps(u, ensure_ascii=False, indent=0) + "\n",
                                            encoding="utf-8")
    entry = {"id": work["work_id"], "title": f"{work['author']}, {work['work']}",
             "language": meta["language"], "edition": meta["edition"],
             "license": meta["license"], "attribution": meta["attribution"], "url": meta["url"],
             "unit_scheme": meta["unit_scheme"], "passage_scheme": meta["passage_scheme"],
             "units": [u["unit"] for u in sorted(final, key=lambda x: unit_sort_key(x["unit"]))]}
    for k in ("passage_note", "source_file", "source_files"):
        if k in meta:
            entry[k] = meta[k]
    if splits:
        entry["split_units"] = splits
        entry["split_note"] = ("Units larger than ~300 KB are split at passage boundaries into "
                               "<unit>a, <unit>b, ...; split_units maps the unit to its parts "
                               "with each part's first and last passage id.")
    if any(u["unit"] == "all" or u.get("parent_unit") == "all" for u in final):
        entry["unit_note"] = "Short work, or a work without books: one unit named 'all'."
    (d / "corpus.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1) + "\n",
                                   encoding="utf-8")
    entry["_stats"] = {"units": len(final), "passages": sum(len(u["passages"]) for u in final),
                       "bytes": sum(p.stat().st_size for p in d.glob("*.json")), "errors": errs}
    return entry


# ---------------------------------------------------------------- main

def select_works(counts: dict[str, int], catalog: list[dict], coverage: float,
                 take_all: bool) -> tuple[list[dict], list[dict]]:
    """Classical (non-humanist) works in descending citation order. Works with a source are
    taken until the taken works cover ``coverage`` of all classical citations; works with no
    open text are always listed (as not-found). Returns (selected, skipped)."""
    classical = [w for w in catalog if not w.get("humanist") and counts[w["work_id"]] > 0]
    classical.sort(key=lambda w: (-counts[w["work_id"]], w["src"] is None, w["work_id"]))
    total = sum(counts[w["work_id"]] for w in classical)
    selected, skipped, acc = [], [], 0
    for w in classical:
        if w["src"] is None:
            selected.append(w)
        elif take_all or acc < coverage * total:
            selected.append(w)
            acc += counts[w["work_id"]]
        else:
            skipped.append(w)
    return selected, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true", help="build from the cache only")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--only", help="comma-separated work ids to (re)build")
    ap.add_argument("--all", action="store_true", help="ignore the coverage cut-off")
    ap.add_argument("--tally-only", action="store_true")
    a = ap.parse_args(argv)

    ids = note_identifications()
    counts = tally(ids)
    selected, skipped = select_works(counts, CATALOG, COVERAGE, a.all)
    sel_ids = {w["work_id"] for w in selected}
    only = set(a.only.split(",")) if a.only else None

    f = Fetcher(a.cache, a.offline)
    results: dict[str, dict] = {}
    prev = {}
    tally_path = OUT / "classical-works.json"
    if tally_path.exists():
        try:
            prev = {r["work_id"]: r for r in json.loads(tally_path.read_text())}
        except (ValueError, KeyError):
            prev = {}
    if not a.tally_only:
        for w in selected:
            if w["src"] is None or w["src"]["type"] == "scan":
                continue
            if only and w["work_id"] not in only:
                continue
            units, meta, errs = BUILDERS[w["src"]["type"]](w, f)
            if not units:
                print(f"{w['work_id']}: nothing built {errs}", file=sys.stderr)
                results[w["work_id"]] = {"status": "not-found", "errors": errs}
                continue
            entry = write_corpus(w, units, meta)
            st = entry.pop("_stats")
            results[w["work_id"]] = {"status": "fetched", "entry": entry, "stats": st,
                                     "errors": errs + st["errors"]}
            print(f"{w['work_id']}: {st['units']} units, {st['passages']} passages, "
                  f"{st['bytes'] / 1e6:.2f} MB" + (f"  ERR {(errs + st['errors'])[:3]}"
                                                    if errs or st["errors"] else ""))

    rows = []
    for w in sorted(CATALOG, key=lambda w: (-counts[w["work_id"]], w["work_id"])):
        if counts[w["work_id"]] == 0:
            continue
        wid = w["work_id"]
        row = {"work_id": wid, "author": w["author"], "work": w["work"],
               "citations": counts[wid], "source": None, "license": None}
        if w.get("humanist"):
            url = SCANS.get(wid)
            row.update(source=url, license="page images only; not fetched" if url else None,
                       status="scan-only" if url else "not-found")
            if not url:
                row["note"] = "no scan located"
        elif wid in results and results[wid]["status"] == "fetched":
            e = results[wid]["entry"]
            row.update(source=e["url"], license=e["license"], status="fetched",
                       units=len(e["units"]))
        elif wid in results:
            row.update(status="not-found")
        elif only and wid not in only and wid in prev:
            row = {**prev[wid], "citations": counts[wid]}
        elif (OUT / wid / "corpus.json").exists() and wid in sel_ids and a.tally_only:
            e = json.loads((OUT / wid / "corpus.json").read_text())
            row.update(source=e["url"], license=e["license"], status="fetched",
                       units=len(e["units"]))
        elif wid in sel_ids:
            row.update(status="not-found",
                       note=w.get("note", "no openly licensed machine-readable text located"))
        else:
            src = w["src"]
            row.update(status="skipped",
                       note="below the ~90% coverage cut-off" + (
                           "" if src else "; no open text located either"))
        rows.append(row)
    if not a.tally_only and not only:
        for r in rows:  # a work no longer built leaves no stale unit files behind
            d = OUT / r["work_id"]
            if r.get("status") != "fetched" and d.is_dir():
                for old in d.glob("*.json"):
                    old.unlink()
                d.rmdir()
    OUT.mkdir(parents=True, exist_ok=True)
    tally_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    cls_total = sum(r["citations"] for r in rows
                    if not next(w for w in CATALOG if w["work_id"] == r["work_id"]).get("humanist"))
    got = sum(r["citations"] for r in rows if r.get("status") == "fetched")
    print(f"classical citations {cls_total}; covered by fetched works {got} "
          f"({100 * got / max(cls_total, 1):.0f}%); fetched this run: {f.fetched}; "
          f"missing from cache: {len(f.missing)}")
    if skipped:
        print("skipped (below cut-off): " + ", ".join(w["work_id"] for w in skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())

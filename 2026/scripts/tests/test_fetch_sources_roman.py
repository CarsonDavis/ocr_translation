"""Parser tests for scripts/fetch_roman_law.py and scripts/fetch_vulgate.py.

The HTML snippets below are trimmed copies of real droitromain.univ-grenoble-alpes.fr pages
(d-48.htm, d-03.htm, CJ9.htm, CJ1.htm, just1.gr.htm, Nov90.htm, Nov33.htm, CJ9_Scott.gr.htm);
the Vulgate lines are from the Clementine Vulgate Project's Ps.lat and Lam.lat."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import fetch_roman_law as R
import fetch_vulgate as V

DIGEST = """
<p align="justify" class="Ak"><font><strong><a name="5"></a><a name="48.5">48.5.0.</a>
  Ad legem Iuliam de adulteriis coercendis. </strong></font></p>
<p class="Ak"><strong><a name="48.5.38">48.5.38 (37)</a></strong></p>
<p class="Ak"><em><strong>Paulus libro primo de adulteriis</strong></em></p>
<p class="Ak"><strong>Quod ait lex, adulterium dici. </strong></p>
<p align="justify" class="Ak"><font><strong><a name="48.5.39">48.5.39
  (38)</a></strong></font></p>
<p align="justify" class="Ak"><font><em><strong>Papinianus
  libro 36 quaestionum </strong></em></font></p>
<p align="justify" class="Ak"><font><strong><a name="48.5.39.pr.">pr.</a>
  Si adulterium cum incesto committatur, ut puta cum privigna nuru
  noverca, mulier similiter quoque punietur. </strong></font></p>
<p align="justify" class="Ak"><font><strong><a name="48.5.39.1">1.</a>
  Stuprum in sororis filiam si committatur, considerandum est. </strong></font></p>
<p class="Ak"><strong><a name="48.5.40">48.5.40</a> </strong></p>
<p class="Ak"><strong>Gaius
  libro 3 ad edictum provinc. </strong></p>
<p class="Ak"><strong>Si quis absentis negotia &quot;gesserit&quot;. </strong></p>
<script>_uacct = "UA-394131-1"; urchinTracker();</script>
"""

CODE = """
<p class="Ak"><a name=9></a><strong><span lang=NL><a name="9.9">9.9.0.</a>
  Ad legem Iuliam de adulteriis et de stupro. </span></strong></p>
<p class="Ak"><strong><span lang=EN-GB><a name="9.9.1">9.9.1</a></span></strong></p>
<p class="Ak"><strong><i><span lang=EN-GB>Imperatores
  Severus, Antoninus . </span></i><span lang=EN-GB>Publico iudicio non
  habere mulieres adulterii accusationem. </span>* SEV. ET ANT. AA. <span
lang=EN-GB>CASSIAE. *&lt;A 197 PP. </span>LATERANO ET RUFINO CONSS.&gt; </strong></p>
<p class="Ak"><strong><span lang=EN-GB><a name="9.9.4">9.9.4</a></span></strong></p>
<p class="Ak"><strong><i><span lang=EN-GB>Imperator
  Alexander Severus . </span></i><a name="9.9.4.pr."></a><span lang=EN-GB>Gracchus,
  quem numerius in adulterio noctu deprehensum interfecerit, nullam poenam meretur. </span></strong></p>
<p class="Ak"><strong><a name="9.9.4.1">1</a>
  <i>. </i>Sed si legis auctoritate cessante inconsulto dolore adulterum
  interemit, potest in exilium dari. * ALEX. A. IULIANO PROCONS. *&lt;A XXX PP.&gt; </strong></p>
<p class="Ak"><strong><a name="9.9.5">9.9.5</a></strong></p>
<p class="Ak"><strong><a name="9.9.6">9.9.6</a>
  [Here there is a Greek text. Sorry, it is not yet in the Library.]<br>
  * *&lt;a 532 d. VIII id. mart. Constantinopoli&gt;</strong></p>
"""

INST = """
<td class="Normal"><font><a name="1" id="1"></a><span class="AK">TIT.&nbsp;1<strong><br>
  &nbsp;&nbsp;<br> DE IUSTITIA ET IURE.</strong></span></font></td>
<td class="Normal"><div align="justify"><font><strong><a name="1.1.pr."></a>&nbsp;&nbsp;Iustitia
  est constans et perpetua voluntas ius suum cuique tribuens.&nbsp; <font color="blue"><a name="1.1.1">1</a>.</font>&nbsp;Iurisprudentia
  est divinarum atque humanarum rerum notitia.</strong></font></div></td>
<td class="AK"><font><a name="2" id="2"></a><a name="1.2">TIT.&nbsp;2<br>
  </a> &nbsp;&nbsp;<br> DE IURE NATURALI GENTIUM ET CIVILI.</font></td>
<td><div><font><strong><a name="1.2.pr."></a>&nbsp;Ius naturale est quod natura omnia animalia docuit.</strong></font></div></td>
"""

NOVEL = """
<td class="Ak"><font><font color="navy"><strong><b>~&nbsp;&nbsp;NOV.
  XC&nbsp;&nbsp;~<br> &nbsp;&nbsp;<br> </b>DE TESTIBUS.<br> &nbsp;&nbsp;<br>
  <b>(&nbsp;AD&nbsp;539&nbsp;)</b></strong></font></font></td>
<td class="Ak"><hr size=2><font color="navy">(&nbsp;Based upon the Latin text of Schoell and Kroll's edition</font>
  &nbsp;)<br>Text submitted by Dr. <a href="http://x">Ingo Maier</a> ~ <hr size=2></td>
<td class="Ak"><span class="Ak"><font><strong>Idem
  Aug. Iohanni pp. secundo.</strong></font></span></td>
<td class="Ak"><div><font><strong>&nbsp;&nbsp;<a name="90.praefatio">&lt;Praefatio&gt;</a>
  Testium propter probationes utilitas adinventa quidem est dudum.</strong></font></div></td>
<td class="Ak"><font><strong><a name="90.4">CAPUT IV.</a></strong></font></td>
<td class="Ak"><div><font>&nbsp;&nbsp;<a name="90.4.pr."></a>Si vero deducens quidem testes.<br>
  &nbsp;&nbsp;<font color="blue"><a name="90.4.1">1</a>.</font>&nbsp;Illud tamen indubium est.</font></div></td>
<td class="Ak"><div><font>&nbsp;&nbsp;<a name="90.epilogus"></a>&lt;Epilogus&gt; Quae igitur placuerunt nobis.<br>
  Dat. kal. Octob.</font></div></td>
<td>&#9658;&nbsp; Sources &nbsp;:&nbsp; Coll. VII, tit.2</td>
"""

NOVEL_NOCHAP = """
<td><strong><b>~ NOV. XXXIII ~<br></b>UT NULLUS MUTUANS AGRICOLAE TENEAT EIUS TERRAM.<br>
<b>( AD 535 )</b></strong></td>
<td><hr>( Based upon the Latin text of Schoell and Kroll's edition ).<br>Text submitted by Dr. Ingo Maier<hr></td>
<td><strong>Idem A. Dominico viro illustri praefecto praetorio per Illyricum.</strong></td>
<td><div>Propter avaritiam creditorum legem posuimus.<br>Dat. XVII. k. Iul. CP.</div></td>
"""

SCOTT = """
<a name="9" id="9"></a>Title 9. On the Lex Julia relating to adultery.
<p>29. The Emperor Constantine to Africanus. Text. Published ..., 326.</p>
<p>30. The Same Emperor to Evagrius. Quamvis text. Given at Nicomedia ..., 326.</p>
<p>31. The Emperors Constantine and Constans to the People. Text ..., 342.</p>
<a name="10" id="10"></a>Title 10.
"""


def by_id(unit):
    return {p["id"]: p for p in unit["passages"]}


def test_digest_fragments_paragraphs_and_jurists():
    [u] = R.parse_book_title_page(DIGEST, "digest", 48)
    assert u["unit"] == "48.5" and u["title"] == "D. 48.5 Ad legem Iuliam de adulteriis coercendis"
    p = by_id(u)
    assert list(p) == ["48.5.38", "48.5.39.pr", "48.5.39.1", "48.5.40"]
    assert p["48.5.38"]["label"] == "D. 48.5.38 (Paulus)" and p["48.5.38"]["alt_number"] == "37"
    assert p["48.5.39.pr"]["label"] == "D. 48.5.39 pr. (Papinianus)"
    assert p["48.5.39.pr"]["inscription"] == "Papinianus libro 36 quaestionum"
    assert p["48.5.39.pr"]["text"].startswith("Si adulterium cum incesto")
    assert p["48.5.39.1"]["label"] == "D. 48.5.39.1"
    assert p["48.5.39.1"]["text"].startswith("Stuprum")       # number anchor text dropped
    # inscription set in roman type is still recognised
    assert p["48.5.40"]["label"] == "D. 48.5.40 (Gaius)"
    assert p["48.5.40"]["text"] == 'Si quis absentis negotia "gesserit".'
    assert not any("urchin" in x["text"] for x in u["passages"])


def test_code_inscription_principium_greek_and_empty():
    [u] = R.parse_book_title_page(CODE, "code", 9)
    p = by_id(u)
    assert u["title"] == "C. 9.9 Ad legem Iuliam de adulteriis et de stupro"
    assert list(p) == ["9.9.1", "9.9.4.pr", "9.9.4.1", "9.9.6"]
    assert p["9.9.1"]["label"] == "C. 9.9.1 (Severus, Antoninus)"
    assert p["9.9.1"]["text"].startswith("Publico iudicio") and "<A 197" in p["9.9.1"]["text"]
    assert p["9.9.4.pr"]["label"] == "C. 9.9.4 pr. (Alexander Severus)"
    assert p["9.9.4.1"]["text"].startswith("Sed si legis")
    assert p["9.9.6"]["greek_not_online"] is True and "Greek text" not in p["9.9.6"]["text"]
    assert u["_empty"] == ["9.9.5"]                         # no text at the source: not stored


def test_single_title_book_numbering():
    src = ('<a href="#1">30.0. De legatis et fideicommissis.</a>'
           '<p><strong><a name="30.1">30.1</a></strong></p><p><em>Ulpianus libro 67 ad edictum</em></p>'
           '<p>Per omnia exaequata sunt legata fideicommissis.</p>'
           '<p><a name="30.4">30.4</a></p><p><em>Ulpianus libro 5 ad Sabinum</em></p>'
           '<p><a name="30.4.pr.">pr.</a> Si quis.</p><p><a name="30.4.1">1.</a> Idem.</p>')
    [u] = R.parse_book_title_page(src, "digest", 30)
    assert u["unit"] == "30" and u["title"] == "D. 30 De legatis et fideicommissis"
    assert list(by_id(u)) == ["30.1", "30.4.pr", "30.4.1"]
    assert by_id(u)["30.4.pr"]["label"] == "D. 30.4 pr. (Ulpianus)"


def test_institutes_titles_and_inline_paragraphs():
    u = R.parse_institutes_page(INST, "1")
    p = by_id(u)
    assert list(p) == ["1.1.pr", "1.1.1", "1.2.pr"]
    assert p["1.1.pr"]["label"] == "Inst. 1.1 pr."
    assert p["1.1.pr"]["text"] == "Iustitia est constans et perpetua voluntas ius suum cuique tribuens."
    assert p["1.1.1"]["text"].startswith("Iurisprudentia")
    assert u["titles"] == {"1.1": "De iustitia et iure", "1.2": "De iure naturali gentium et civili"}


def test_novel_chapters_praefatio_epilogue_and_stop_marker():
    u = R.parse_novel_page(NOVEL, 90)
    assert u["title"] == "Nov. 90 De testibus" and u["year"] == 539
    assert u["inscription"] == "Idem Aug. Iohanni pp. secundo."
    p = by_id(u)
    assert list(p) == ["90.pr", "90.4", "90.epilogus"]
    assert p["90.pr"]["text"].startswith("Testium propter")
    assert p["90.4"]["text"] == "Si vero deducens quidem testes.\n\n1. Illud tamen indubium est."
    assert p["90.epilogus"]["text"].startswith("Quae igitur") and "Sources" not in p["90.epilogus"]["text"]


def test_novel_without_chapter_anchors():
    u = R.parse_novel_page(NOVEL_NOCHAP, 33)
    assert u["no_chapters"] is True
    assert u["inscription"].startswith("Idem A. Dominico")
    assert [x["id"] for x in u["passages"]] == ["33.pr"]
    assert u["passages"][0]["text"].startswith("Propter avaritiam")


def test_split_unit_at_fragment_boundaries():
    passages = []
    for f in range(1, 21):
        for par in ("pr", "1"):
            passages.append({"id": f"48.5.{f}.{par}", "label": "x", "text": "a" * 400})
    u = {"corpus": "digest", "unit": "48.5", "title": "D. 48.5", "passages": passages}
    parts = R.split_unit(u, limit=6000)
    assert len(parts) > 1 and all(R.unit_bytes(x) <= 6000 for x in parts)
    assert [x["unit"] for x in parts][:2] == ["48.5a", "48.5b"]
    for x in parts:   # never splits inside a fragment
        assert x["passages"][0]["id"].endswith(".pr") and x["passages"][-1]["id"].endswith(".1")
    assert parts[0]["passage_range"][0] == "48.5.1.pr"
    assert sum(len(x["passages"]) for x in parts) == 40


def test_validate_flags_markup_but_not_editorial_brackets():
    ok = {"corpus": "code", "unit": "9.9", "title": "t",
          "passages": [{"id": "9.9.1", "label": "C. 9.9.1", "text": "x *<A 197 PP.> < heredem>"}]}
    assert R.validate_unit(ok, "code") == []
    bad = json.loads(json.dumps(ok))
    bad["passages"][0]["text"] = "x <font face=x>y</font>"
    assert R.validate_unit(bad, "code")


def test_scott_kruger_concordance_alignment():
    sc = R.parse_scott_book(SCOTT)[9]
    assert [(c["num"], c["to"], c["year"]) for c in sc] == [(29, "afr", "326"), (30, "eua", "326"),
                                                           (31, "pop", "342")]
    units = [{"unit": "9.9", "passages": [
        {"id": "9.9.28", "text": "t * CONST. A. AD AFRICANUM. *<A 326 PP.>"},
        {"id": "9.9.29.pr", "text": "Quamvis adulterii"},
        {"id": "9.9.29.4", "text": "t * CONST. A. AD EUAGRIUM. *<A 326 PP.>"},
        {"id": "9.9.30", "text": "t * CONSTANTIUS ET CONSTANS AA. AD POP. *<A 342 PP.>"}]}]
    k = R.kruger_constitutions(units)[(9, 9)]
    assert [c["to"] for c in k] == ["afr", "eua", "pop"]

    class F:
        def get(self, page, base=None):
            return SCOTT if "CJ9" in page else None
    conc, fails = R.build_concordance(units, F())
    assert conc["9.9.30"] == "9.9.29" and conc["9.9.29"] == "9.9.28" and conc["9.9.31"] == "9.9.30"
    assert len(fails) == 11


def test_vulgate_clean_and_parse():
    raw = ("1:1 <Prologus>Et factum est, dixit : [<Aleph>Quomodo sedet sola/ civitas plena populo !/\r\n"
           "9:21 Constitue, Domine, legislatorem super eos,/ ut sciant gentes quoniam homines sunt.]\r\n")
    ps = V.parse_book(raw, "Psalms")
    assert [p["id"] for p in ps] == ["Psalms 1:1", "Psalms 9:21"]
    assert ps[0]["text"] == "Prologus. Et factum est, dixit: Aleph. Quomodo sedet sola\ncivitas plena populo!"
    assert ps[1]["text"].endswith("homines sunt.") and "]" not in ps[1]["text"]


def test_vulgate_split_by_chapter():
    passages = [{"id": f"Psalms {c}:{v}", "label": "x", "text": "a" * 300}
                for c in range(1, 11) for v in range(1, 6)]
    u = {"corpus": "vulgate", "unit": "psalms", "title": "Psalms", "passages": passages}
    parts = V.split_by_chapter(u, limit=12000)
    assert len(parts) >= 2 and parts[0]["unit"] == "psalms-a"
    assert all(V.ubytes(x) <= 12000 for x in parts)
    assert parts[1]["passages"][0]["id"].endswith(":1")
    assert sum(len(x["passages"]) for x in parts) == 50

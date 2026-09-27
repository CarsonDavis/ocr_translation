# scripts/tests/test_manifest.py
import json, subprocess, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]

def load(tmp_path=None):
    import tempfile
    out = pathlib.Path(tempfile.mkdtemp()) / "manifest.json"
    subprocess.run([sys.executable, ROOT/"scripts/build_manifest.py", str(out)], check=True, cwd=ROOT)
    return json.loads(out.read_text())

def test_preserves_status_on_rebuild():
    import tempfile
    out = pathlib.Path(tempfile.mkdtemp()) / "manifest.json"
    subprocess.run([sys.executable, ROOT/"scripts/build_manifest.py", str(out)], check=True, cwd=ROOT)
    m = json.loads(out.read_text())
    m["pages"][5]["status"]["acquired"] = "done"; m["pages"][5]["agreement"] = 97.5
    out.write_text(json.dumps(m))
    subprocess.run([sys.executable, ROOT/"scripts/build_manifest.py", str(out)], check=True, cwd=ROOT)
    m2 = json.loads(out.read_text())
    assert m2["pages"][5]["status"]["acquired"] == "done" and m2["pages"][5]["agreement"] == 97.5

def by_id(m): return {r["id"]: r for r in m["pages"]}

def test_count():
    assert len(load()["pages"]) == 162

def test_mapping_samples():
    p = by_id(load())
    assert p["p000-title"]["image"] == 7
    assert p["p000-argument"]["image"] == 22
    assert p["p001"]["image"] == 23 and p["p040"]["image"] == 62
    assert p["p041"]["image"] is None and p["p041"]["source"] == "gallica"
    assert p["p042"]["image"] == 64 and p["p043"]["image"] == 63
    assert p["p158"]["image"] == 180 and p["p159"]["image"] == 179
    assert p["p160"]["image"] == 182

def test_folios():
    p = by_id(load())
    assert p["p044"]["folio"] == "24" and p["p045"]["folio"] == "44" and p["p048"]["folio"] == "58"
    assert p["p046"]["folio"] == "46" and p["p000-title"]["folio"] is None
    assert p["p077"]["folio"] == "78"

def test_sides():
    p = by_id(load())
    assert p["p001"]["side"] == "recto" and p["p002"]["side"] == "verso"

def test_no_duplicate_images():
    imgs = [r["image"] for r in load()["pages"] if r["image"] is not None]
    assert len(imgs) == len(set(imgs))

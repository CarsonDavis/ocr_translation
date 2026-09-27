"""Write manifest.json: the single source of truth for page identity and stage status."""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "manifest.json"
CUDL_ITEM = "PR-MONTAIGNE-00001-00007-00022"
IIIF = f"https://images.lib.cam.ac.uk/iiif/{CUDL_ITEM}-000-{{n:05d}}.jp2"
MISPRINTED_FOLIOS = {44: "24", 45: "44", 48: "58", 77: "78"}  # verified against the scans 2026-09-21
STAGES = ["acquired", "cropped", "readA", "readB", "diffed", "final", "spotchecked", "translated", "reviewed"]

def image_for_page(page: int):
    if 1 <= page <= 40: return page + 22
    if page == 41: return None
    if 42 <= page <= 159:
        # openings were photographed recto-first: odd image i -> page i-20, even image i -> page i-22
        return page + 20 if page % 2 == 1 else page + 22
    if page == 160: return 182
    raise ValueError(page)

def record(pid, page, image, side, folio, source="cudl"):
    return {"id": pid, "page": page, "image": image, "side": side, "folio": folio,
            "source": source,
            "iiif": IIIF.format(n=image) if image else None,
            "status": {s: "pending" for s in STAGES}}

pages = [record("p000-title", None, 7, "recto", None),
         record("p000-argument", None, 22, "verso", None)]
for pg in range(1, 161):
    img = image_for_page(pg)
    pages.append(record(f"p{pg:03d}", pg, img, "recto" if pg % 2 else "verso",
                        MISPRINTED_FOLIOS.get(pg, str(pg)),
                        source="cudl" if img else "gallica"))
# Preserve everything later stages have written (status, agreement, crop, ...) on an existing manifest.
if OUT.exists():
    old = {r["id"]: r for r in json.loads(OUT.read_text())["pages"]}
    for rec in pages:
        prev = old.get(rec["id"], {})
        for k, v in prev.items():
            if k == "status":
                rec["status"].update({s: v[s] for s in STAGES if s in v})
            elif k not in rec:
                rec[k] = v
manifest = {"item": CUDL_ITEM, "edition": "Paris: Galliot du Pré, 1572",
            "native_size": [2941, 4711], "pages": pages}
OUT.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
print(f"wrote {len(pages)} pages")

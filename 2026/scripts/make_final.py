"""Create transcription/final/<id>.json from one read without a reconciler, for pages where
the two reads agree on every printed line (diff agreement 100% and no structural
difference), or where the coordinator has ruled which read to take.

usage: make_final.py PAGE_ID --from A|B [--reason TEXT] [--model NAME]
"""
import argparse, json, pathlib, subprocess, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bookconf import ROOT  # noqa: E402  (the book root: see bookconf.py)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page_id"); ap.add_argument("--from", dest="src", choices=["A", "B"], required=True)
    ap.add_argument("--reason", default="reads agree on every printed line")
    ap.add_argument("--model", default="script")
    a = ap.parse_args()
    src = ROOT / "transcription/reads" / a.src / f"{a.page_id}.json"
    page = json.loads(src.read_text())
    page["reader"] = "final"; page["model"] = a.model
    page["decisions"] = [{"where": "page", "A": "", "B": "", "chose": a.src, "text": "",
                          "reason": a.reason}]
    out = ROOT / "transcription/final" / f"{a.page_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(page, indent=1, ensure_ascii=False) + "\n")
    r = subprocess.run([sys.executable, ROOT / "scripts/validate_page.py", str(out)], capture_output=True, text=True)
    print(r.stdout.strip()); sys.exit(r.returncode)

if __name__ == "__main__":
    main()

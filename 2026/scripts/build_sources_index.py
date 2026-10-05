"""Merge every site/data/sources/<corpus>/corpus.json into site/data/sources/index.json.

usage: build_sources_index.py [--check]
Each corpus fetcher writes its own corpus.json (one index entry, contract shape: see
docs/sources-contract.md). This script collects them, sorted by id, into
{"corpora": [...]}. If a corpus.json has no "units" list, the unit files present in its
directory (every *.json except corpus.json and helper files such as concordance.json) are
listed instead. Idempotent: the file is rewritten only when its content changes.
--check exits 1 if index.json is out of date, without writing.
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bookconf import ROOT  # noqa: E402

HELPER_FILES = {"corpus.json", "concordance.json"}


def unit_sort_key(u):
    parts = []
    for p in str(u).replace(" ", ".").split("."):
        parts.append((0, int(p), "") if p.isdigit() else (1, 0, p))
    return parts


def collect(sources_dir):
    corpora = []
    for cj in sorted(sources_dir.glob("*/corpus.json")):
        try:
            entry = json.loads(cj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"skip {cj}: {e}", file=sys.stderr)
            continue
        if isinstance(entry, dict) and "corpora" in entry:  # tolerate a wrapped entry
            entries = entry["corpora"]
        else:
            entries = [entry]
        for e in entries:
            e = dict(e)
            e.setdefault("id", cj.parent.name)
            if not e.get("units"):
                units = [p.stem for p in cj.parent.glob("*.json") if p.name not in HELPER_FILES]
                e["units"] = sorted(units, key=unit_sort_key)
            corpora.append(e)
    corpora.sort(key=lambda e: e["id"])
    return {"corpora": corpora}


def main(argv):
    sources_dir = ROOT / "site/data/sources"
    out = sources_dir / "index.json"
    data = collect(sources_dir) if sources_dir.exists() else {"corpora": []}
    text = json.dumps(data, ensure_ascii=False, indent=1) + "\n"
    old = out.read_text(encoding="utf-8") if out.exists() else None
    if "--check" in argv:
        print("up to date" if old == text else "out of date")
        return 0 if old == text else 1
    if old != text:
        sources_dir.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    ids = [c["id"] for c in data["corpora"]]
    print(f"{out.relative_to(ROOT)}: {len(ids)} corpora{' (unchanged)' if old == text else ''}: {', '.join(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

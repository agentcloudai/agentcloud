"""Validate the eval set's expected answers against the ORIGINAL AWS source PDFs.

For every test case in an eval JSONL, this checks that each ``key_term`` (the
things a correct answer must mention) actually appears in the authoritative guide
PDF(s) it is mapped to (``expected_files`` under the data folder). This grounds the
eval's "correct answer" in AWS's own documentation instead of our assertions.

Usage:
    python eval/validate_ground_truth.py [eval/architecture_questions.jsonl]
"""
import json
import sys
from pathlib import Path

import pymupdf  # fast PDF text extraction

DATA_DIR = Path("data/raw")


def extract_text(pdf_path: Path) -> str:
    try:
        doc = pymupdf.open(str(pdf_path))
        return "\n".join(page.get_text() for page in doc).lower()
    except Exception as exc:  # noqa: BLE001
        print(f"  !! could not read {pdf_path.name}: {exc}")
        return ""


def main(eval_file: str) -> None:
    rows = [json.loads(l) for l in open(eval_file, encoding="utf-8") if l.strip()]

    # Cache text per unique source file.
    cache: dict[str, str] = {}
    for r in rows:
        for fn in r.get("expected_files", []):
            if fn not in cache:
                p = DATA_DIR / fn
                print(f"reading {fn} ...")
                cache[fn] = extract_text(p) if p.exists() else ""
                if not p.exists():
                    print(f"  !! missing file: {p}")

    print("\n=== grounding report (key_terms found in the authoritative AWS PDF) ===")
    total_terms = grounded_terms = 0
    results = []
    for r in rows:
        source_text = " ".join(cache.get(fn, "") for fn in r.get("expected_files", []))
        terms = r.get("key_terms", [])
        found = [t for t in terms if t.lower() in source_text]
        missing = [t for t in terms if t.lower() not in source_text]
        total_terms += len(terms)
        grounded_terms += len(found)
        status = "OK " if not missing else "!! "
        print(f"{status}{len(found)}/{len(terms)} grounded | {r['question'][:58]}")
        if missing:
            print(f"     NOT found in {r.get('expected_files')}: {missing}")
        results.append({"question": r["question"], "grounded": found, "missing": missing,
                        "files": r.get("expected_files", [])})

    pct = (grounded_terms / total_terms * 100) if total_terms else 0
    print(f"\nTOTAL: {grounded_terms}/{total_terms} key terms ({pct:.0f}%) verified against the source PDFs.")

    out = Path("eval/results/ground_truth_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/architecture_questions.jsonl")

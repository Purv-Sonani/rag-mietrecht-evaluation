"""
Gold-set verification.

  python src/verify.py sheet            worksheet for all 85 answerable items
  python src/verify.py sheet Q023 Q009  worksheet for specific items
  python src/verify.py misses <run.jsonl>   worksheet for just the failures,
                                            showing what was retrieved instead
  python src/verify.py mark --wrong Q009 Q021
                                        marks everything else verified

The test for each item is one question: does the cited provision govern the
topic the question asks about? You are checking the label, not the answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from glosses import HEADING_EN, QUESTION_EN
from ingest import sections_index

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/gold/gold_v1.jsonl"


def load():
    return [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]


def save(rows):
    GOLD.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


def _snippet(sec, n=260):
    body = " ".join(sec["text"].split())
    return body[:n] + ("..." if len(body) > n else "")


def sheet(only: list[str] | None = None, retrieved: dict | None = None) -> None:
    idx = sections_index(ROOT / "data/corpus")
    rows = [r for r in load() if r["answerable"]]
    if only:
        rows = [r for r in rows if r["qid"] in only]

    for r in rows:
        print("=" * 72)
        print(f"{r['qid']}   [{r['theme']}]   verified={r.get('verified', False)}")
        print(f"  DE  {r['question']}")
        print(f"  EN  {QUESTION_EN.get(r['qid'], '(no gloss)')}")
        print()
        for sid in r["gold"]:
            sec = idx.get(sid)
            print(f"  GOLD  {sid} - {sec['heading'] if sec else '?? NOT IN CORPUS'}")
            print(f"        EN: {HEADING_EN.get(sid, '(no gloss)')}")
            if sec:
                print(f"        {_snippet(sec)}")
        if retrieved and r["qid"] in retrieved:
            print()
            print("  RETRIEVED INSTEAD:")
            for sid in retrieved[r["qid"]][:5]:
                print(f"        {sid} - {HEADING_EN.get(sid, idx.get(sid, {}).get('heading', '?'))}")
        print()
        print("  -> correct provision for this question?   [ y / n ]")
        print()


def misses(run_path: str, k: int = 5) -> None:
    recs = [json.loads(l) for l in Path(run_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    best, score = None, -1.0
    for c in {r["condition"] for r in recs}:
        sub = [r for r in recs if r["condition"] == c and r["answerable"]]
        if not sub:
            continue
        s = sum(bool(set(r["retrieved"][:k]) & set(r["gold"])) for r in sub) / len(sub)
        if s > score:
            best, score = c, s
    sub = [r for r in recs if r["condition"] == best and r["answerable"]]
    missed = [r for r in sub if not set(r["retrieved"][:k]) & set(r["gold"])]
    print(f"# condition {best}, {len(missed)}/{len(sub)} missed at k={k}\n")
    sheet([r["qid"] for r in missed], {r["qid"]: r["retrieved"] for r in missed})


def mark(wrong: list[str]) -> None:
    rows = load()
    n = 0
    for r in rows:
        if r["qid"] in wrong:
            r["verified"] = False
            r["needs_fix"] = True
        else:
            r["verified"] = True
            r.pop("needs_fix", None)
            n += 1
    save(rows)
    print(f"marked {n} verified, {len(wrong)} flagged for fixing: {', '.join(wrong) or '(none)'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "sheet":
        sheet(args[1:] or None)
    elif args[0] == "misses":
        misses(args[1])
    elif args[0] == "mark":
        mark([a for a in args[1:] if a != "--wrong"])
    else:
        print(__doc__)

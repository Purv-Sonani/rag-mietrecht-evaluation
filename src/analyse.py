"""
Post-hoc analysis.

Two jobs:

  sweep    fusion weight from pure BM25 to pure dense, in steps. Produces the
           curve that shows *where* fusion starts to hurt rather than just
           that it does.

  failures groups the questions the best condition got wrong, by theme and by
           difficulty. This is what Chapter 4.2 is actually made of: an
           aggregate score tells a marker what happened, the failure cases
           tell them why.

Usage
-----
  python src/analyse.py sweep
  python src/analyse.py failures results/run_YYYYmmdd-HHMMSS.jsonl
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import metrics
from ingest import build
from retrievers import BM25Retriever, DenseRetriever, HybridRetriever

ROOT = Path(__file__).resolve().parent.parent


def load_gold(path="data/gold/gold_v1.jsonl"):
    return [json.loads(l) for l in (ROOT / path).read_text(encoding="utf-8").splitlines() if l.strip()]


def sweep(k: int = 5, steps=(0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)) -> None:
    chunks = [c.to_dict() for c in build(ROOT / "data/corpus", "structural", 900, 150)]
    gold = [g for g in load_gold() if g["answerable"]]
    bm25, dense = BM25Retriever(chunks), DenseRetriever(chunks)
    if dense.backend != "sentence-transformers":
        print(f"WARNING: fallback embedder active ({dense.backend}). Not reportable.\n")

    print(f"{'dense_w':>8} {'hit@5':>8} {'mrr':>8} {'ndcg@5':>8}")
    for w in steps:
        retr = HybridRetriever(chunks, dense, bm25, dense_weight=w)
        recs = []
        for g in gold:
            seen, retrieved = set(), []
            for h in retr.search(g["question"], k):
                if h.section_id not in seen:
                    seen.add(h.section_id)
                    retrieved.append(h.section_id)
            recs.append({"scores": metrics.score_record(
                {"retrieved": retrieved, "gold": g["gold"], "answerable": True, "answer": ""})})
        agg = metrics.aggregate(recs)
        print(f"{w:>8.1f} {agg['hit@5']['mean']:>8.3f} "
              f"{agg['mrr']['mean']:>8.3f} {agg['ndcg@5']['mean']:>8.3f}")


def failures(run_path: str, condition: str | None = None, k: int = 5) -> None:
    records = [json.loads(l) for l in Path(run_path).read_text(encoding="utf-8").splitlines() if l.strip()]

    if condition is None:
        scored = {}
        for c in {r["condition"] for r in records}:
            sub = [r for r in records if r["condition"] == c and r["answerable"]]
            agg = metrics.aggregate(sub)
            scored[c] = agg.get("hit@5", {"mean": 0})["mean"]
        condition = max(scored, key=scored.get)
        print(f"best condition by hit@5: {condition} ({scored[condition]:.3f})\n")

    sub = [r for r in records if r["condition"] == condition and r["answerable"]]
    missed = [r for r in sub if not set(r["retrieved"][:k]) & set(r["gold"])]

    print(f"{len(missed)}/{len(sub)} answerable questions missed at k={k}\n")

    by_theme = collections.Counter(r["theme"] for r in missed)
    by_diff = collections.Counter(r["difficulty"] for r in missed)
    tot_theme = collections.Counter(r["theme"] for r in sub)
    tot_diff = collections.Counter(r["difficulty"] for r in sub)

    print("miss rate by difficulty")
    for d, n in tot_diff.most_common():
        print(f"  {d:14s} {by_diff[d]}/{n}  ({by_diff[d]/n:.0%})")

    print("\nmiss rate by theme (themes with at least one miss)")
    for t, n in tot_theme.most_common():
        if by_theme[t]:
            print(f"  {t:24s} {by_theme[t]}/{n}")

    print("\n--- individual misses ---")
    for r in missed:
        print(f"\n[{r['qid']}] {r['question'][:95]}")
        print(f"   expected: {', '.join(r['gold'])}")
        print(f"   got:      {', '.join(r['retrieved'][:k])}")

    # Unanswerable behaviour is a separate story and belongs in its own table.
    unans = [r for r in records if r["condition"] == condition and not r["answerable"]]
    if unans:
        halluc = [r for r in unans if not metrics.abstained(r.get("answer", ""))]
        print(f"\n--- unanswerable probes: {len(halluc)}/{len(unans)} answered "
              f"instead of declining ---")
        for r in halluc[:5]:
            print(f"  [{r['qid']}] {r['question'][:80]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if cmd == "sweep":
        sweep()
    elif cmd == "failures":
        failures(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        print(__doc__)

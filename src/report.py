"""
Prints one compact block covering everything needed to decide how Chapter 4 is
framed. Run it after a --dry-run, copy the whole output.

  python src/report.py

Deliberately excludes generated answer text and anything key-shaped, so the
output is safe to paste anywhere.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import metrics
from ingest import build, sections_index
from retrievers import BM25Retriever, DenseRetriever, HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
BAR = "=" * 68


def block_setup(dense):
    print(BAR)
    print("SETUP")
    print(BAR)
    idx = sections_index(ROOT / "data/corpus")
    chunks = build(ROOT / "data/corpus", "structural", 900, 150)
    gold = [json.loads(l) for l in
            (ROOT / "data/gold/gold_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    verified = sum(g.get("verified", False) for g in gold)
    print(f"sections        : {len(idx)}")
    print(f"chunks          : {len(chunks)}")
    print(f"gold questions  : {len(gold)}  "
          f"({sum(g['answerable'] for g in gold)} answerable, "
          f"{sum(not g['answerable'] for g in gold)} unanswerable)")
    print(f"gold verified   : {verified}/{len(gold)}"
          + ("   <-- NOT YET VERIFIED, treat numbers as provisional"
             if verified < len(gold) else ""))
    print(f"embedder        : {dense.model_name}")
    print(f"backend         : {dense.backend}"
          + ("   <-- FALLBACK, NOT REPORTABLE" if dense.backend != "sentence-transformers" else ""))


def block_summary():
    print()
    print(BAR)
    print("CONDITION SUMMARY (latest run)")
    print(BAR)
    files = glob.glob(str(ROOT / "results/summary_*.csv"))
    if not files:
        print("no summary csv found - run src/run_experiment.py first")
        return
    latest = max(files, key=os.path.getmtime)
    print(f"file: {Path(latest).name}\n")
    rows = list(csv.DictReader(open(latest, encoding="utf-8")))
    cols = ["condition", "hit@1", "hit@3", "hit@5", "hit@10",
            "recall@5", "mrr", "ndcg@5", "mean_latency_s"]
    cols = [c for c in cols if c in rows[0]]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def block_sweep(chunks, dense, bm25, k=5):
    print()
    print(BAR)
    print("FUSION WEIGHT SWEEP  (0.0 = BM25 only, 1.0 = dense only)")
    print(BAR)
    gold = [json.loads(l) for l in
            (ROOT / "data/gold/gold_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = [g for g in gold if g["answerable"]]
    print(f"{'dense_w':>8} {'hit@5':>8} {'mrr':>8} {'ndcg@5':>8}")
    for w in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        retr = HybridRetriever(chunks, dense, bm25, dense_weight=w)
        recs = []
        for g in gold:
            seen, retrieved = set(), []
            for h in retr.search(g["question"], 40):
                if h.section_id not in seen:
                    seen.add(h.section_id)
                    retrieved.append(h.section_id)
                if len(retrieved) >= 10:
                    break
            recs.append({"scores": metrics.score_record(
                {"retrieved": retrieved, "gold": g["gold"], "answerable": True, "answer": ""})})
        a = metrics.aggregate(recs)
        print(f"{w:>8.1f} {a['hit@5']['mean']:>8.3f} {a['mrr']['mean']:>8.3f} "
              f"{a['ndcg@5']['mean']:>8.3f}")


def block_misses(k=5):
    print()
    print(BAR)
    print("MISSES BY DIFFICULTY AND THEME (best condition)")
    print(BAR)
    files = glob.glob(str(ROOT / "results/run_*.jsonl"))
    if not files:
        print("no run jsonl found")
        return
    latest = max(files, key=os.path.getmtime)
    recs = [json.loads(l) for l in Path(latest).read_text(encoding="utf-8").splitlines() if l.strip()]

    import collections
    best, best_score = None, -1.0
    for c in {r["condition"] for r in recs}:
        sub = [r for r in recs if r["condition"] == c and r["answerable"]]
        if not sub:
            continue
        s = metrics.aggregate(sub).get("hit@5", {"mean": 0})["mean"]
        if s > best_score:
            best, best_score = c, s

    sub = [r for r in recs if r["condition"] == best and r["answerable"]]
    missed = [r for r in sub if not set(r["retrieved"][:k]) & set(r["gold"])]
    print(f"best condition: {best} (hit@5 {best_score:.3f})")
    print(f"missed: {len(missed)}/{len(sub)}\n")

    for label, key in (("difficulty", "difficulty"), ("theme", "theme")):
        tot = collections.Counter(r[key] for r in sub)
        mis = collections.Counter(r[key] for r in missed)
        print(f"by {label}:")
        for t, n in tot.most_common():
            if mis[t]:
                print(f"  {t:24s} {mis[t]}/{n}")
        print()

    print("missed qids:", ", ".join(r["qid"] for r in missed))


if __name__ == "__main__":
    chunks = [c.to_dict() for c in build(ROOT / "data/corpus", "structural", 900, 150)]
    dense = DenseRetriever(chunks)
    bm25 = BM25Retriever(chunks)
    block_setup(dense)
    block_summary()
    block_sweep(chunks, dense, bm25)
    block_misses()
    print()
    print(BAR)
    print("END")
    print(BAR)

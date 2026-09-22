"""
Experiment runner.

Executes every (condition x gold question) pair, records the full trace, and
writes both a per-item JSONL (for the appendix and for the human-rating export)
and an aggregated CSV (for the Chapter 4.1 tables and charts).

Usage
-----
  python src/run_experiment.py --dry-run          # no model calls, retrieval only
  python src/run_experiment.py --generator anthropic
  python src/run_experiment.py --generator ollama --model qwen2.5:7b
  python src/run_experiment.py --ablation chunking
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import generators as gen
import metrics
from ingest import build
from retrievers import BM25Retriever, DenseRetriever, HybridRetriever

ROOT = Path(__file__).resolve().parent.parent


def load_gold(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


_CACHE_PATH = None
_CACHE = {}


def _cache_load(model_name):
    """Answers are cached on disk by (model, condition, qid). A run that dies
    partway can simply be restarted: completed calls are not repeated."""
    global _CACHE_PATH, _CACHE
    safe = "".join(c if c.isalnum() else "_" for c in model_name)
    _CACHE_PATH = ROOT / "results" / f"cache_{safe}.json"
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _CACHE_PATH.exists():
        _CACHE = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    print(f"cache:  {len(_CACHE)} answers already stored")


def _cached(key, fn):
    if key in _CACHE:
        return _CACHE[key]
    val = fn()
    _CACHE[key] = val
    _CACHE_PATH.write_text(json.dumps(_CACHE, ensure_ascii=False), encoding="utf-8")
    return val


def retrieve_sections(retriever, query, depth: int, pool: int):
    """Return the top `depth` DISTINCT sections from a deep chunk ranking.

    Retrieval metrics are computed over sections, but retrieval happens over
    chunks, and several chunks often share one section. Scoring hit@10 against
    a list built from only 5 chunks silently caps the metric - it can never
    exceed hit@5. So the evaluation ranking is built from a deliberately deep
    chunk pool, independently of how many chunks the generator is shown.
    """
    seen, out = set(), []
    for h in retriever.search(query, pool):
        if h.section_id not in seen:
            seen.add(h.section_id)
            out.append(h.section_id)
        if len(out) >= depth:
            break
    return out


def make_generator(kind: str, model: str | None):
    if kind == "anthropic":
        return gen.AnthropicGenerator(model or "claude-sonnet-4-6")
    if kind == "gemini":
        return gen.GeminiGenerator(model or "gemini-2.0-flash")
    if kind == "ollama":
        return gen.OllamaGenerator(model or "qwen2.5:7b")
    return gen.EchoGenerator()


def build_conditions(chunks, k: int, reranker: str | None, w: float = 0.5):
    """The four reported conditions. C0 has no retriever by construction."""
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    return [
        ("C0_no_retrieval", None),
        ("C1a_bm25_only", bm25),
        ("C1b_dense_only", dense),
        ("C2_hybrid_rrf", HybridRetriever(chunks, dense, bm25, dense_weight=w)),
        ("C3_hybrid_rerank",
         HybridRetriever(chunks, dense, bm25, reranker_model=reranker, dense_weight=w)),
    ]


def run(args) -> None:
    chunks = [c.to_dict() for c in
              build(ROOT / "data/corpus", args.chunking, args.chunk_size, args.overlap)]
    gold = load_gold(ROOT / args.gold)
    generator = make_generator("dry" if args.dry_run else args.generator, args.model)

    print(f"corpus: {len(chunks)} chunks ({args.chunking}, size={args.chunk_size})")
    print(f"gold:   {len(gold)} questions "
          f"({sum(g['answerable'] for g in gold)} answerable)")
    print(f"model:  {generator.name}")
    _cache_load(generator.name)
    print()

    records = []
    for cond_name, retriever in build_conditions(chunks, args.k, args.reranker, args.dense_weight):
        for g in gold:
            t0 = time.time()
            if retriever is None:
                hits, retrieved = [], []
                answer = _cached(f"{cond_name}|{g['qid']}",
                                 lambda: generator.complete(gen.SYSTEM_NO_RAG, g["question"]))
            else:
                # What the generator sees: top-k chunks, as in deployment.
                hits = retriever.search(g["question"], args.k)
                # What the metrics score: a deeper, section-deduplicated ranking.
                retrieved = retrieve_sections(
                    retriever, g["question"], args.eval_depth, args.chunk_pool
                )
                answer = _cached(f"{cond_name}|{g['qid']}",
                                 lambda: generator.complete(
                                     gen.SYSTEM_RAG,
                                     gen.build_rag_prompt(g["question"], hits)))
            latency = time.time() - t0

            rec = {
                "condition": cond_name,
                "qid": g["qid"],
                "question": g["question"],
                "answerable": g["answerable"],
                "theme": g["theme"],
                "difficulty": g["difficulty"],
                "gold": g["gold"],
                "retrieved": retrieved,
                "answer": answer,
                "latency_s": round(latency, 3),
                "n_context_chars": sum(len(h.text) for h in hits),
            }
            rec["scores"] = metrics.score_record(rec)
            records.append(rec)
        print(f"  ran {cond_name}", flush=True)

    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    raw = outdir / f"run_{stamp}.jsonl"
    with raw.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = outdir / f"summary_{stamp}.csv"
    conds = sorted({r["condition"] for r in records})
    rows = []
    for c in conds:
        agg = metrics.aggregate([r for r in records if r["condition"] == c])
        row = {"condition": c}
        row.update({k: round(v["mean"], 4) for k, v in agg.items()})
        lat = [r["latency_s"] for r in records if r["condition"] == c]
        row["mean_latency_s"] = round(sum(lat) / len(lat), 3)
        rows.append(row)

    fields = sorted({k for r in rows for k in r})
    fields = ["condition"] + [f for f in fields if f != "condition"]
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {raw.name} and {summary.name}\n")
    for r in rows:
        print("  " + " | ".join(f"{k}={r[k]}" for k in
              ("condition", "hit@5", "mrr", "correct_abstention") if k in r))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default="data/gold/gold_seed.jsonl")
    p.add_argument("--generator", default="anthropic", choices=["anthropic", "gemini", "ollama", "dry"])
    p.add_argument("--model", default=None)
    p.add_argument("--chunking", default="structural", choices=["structural", "fixed"])
    p.add_argument("--chunk-size", type=int, default=900)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--reranker", default="BAAI/bge-reranker-base")
    p.add_argument("--eval-depth", type=int, default=10,
                   help="how many distinct sections the metrics are scored over")
    p.add_argument("--chunk-pool", type=int, default=40,
                   help="chunk depth searched to build that section ranking")
    p.add_argument("--dense-weight", type=float, default=0.5,
                   help="fusion weight on dense; 1.0 = dense only, 0.0 = BM25 only")
    p.add_argument("--dry-run", action="store_true")
    run(p.parse_args())

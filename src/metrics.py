"""
Metrics.

Deliberately split into two families, because they have very different
evidential weight and the dissertation should not pretend otherwise:

  Family A - computed from the gold section labels alone. Fully objective,
             reproducible by anyone with the gold file, and (importantly for a
             non-native German speaker) verifiable without judging German prose:
             checking that § 536 governs rent reduction is a lookup, not a
             translation exercise.

  Family B - answer-level. Citation accuracy is still mechanical. Abstention
             behaviour is mechanical. Anything about whether the answer is
             *good* is left to the human raters in Strand B, not asserted here.
"""

from __future__ import annotations

import math
import re

SECTION_CITE_RE = re.compile(r"§+\s*(\d+[a-z]?)")
ABSTENTION_MARKERS = (
    "keine angabe",
    "nicht beantworten",
    "nicht sicher",
    "keine informationen",
    "enthalten dazu keine",
)


# ---- Family A: retrieval -------------------------------------------------

def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    top = set(retrieved[:k])
    return len(top & set(gold)) / len(set(gold))


def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def mrr(retrieved: list[str], gold: list[str]) -> float:
    if not gold:
        return float("nan")
    goldset = set(gold)
    for i, r in enumerate(retrieved, 1):
        if r in goldset:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    goldset = set(gold)
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(retrieved[:k], 1) if r in goldset)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(goldset), k) + 1))
    return dcg / ideal if ideal else 0.0


# ---- Family B: answer-level ----------------------------------------------

def cited_sections(answer: str, source_hint: str = "BGB") -> list[str]:
    return [f"{source_hint} § {n}" for n in SECTION_CITE_RE.findall(answer)]


def citation_precision(answer: str, gold: list[str]) -> float:
    """Of the paragraphs the model names, how many are actually the right ones.
    A low value here on the no-retrieval condition is the clearest single
    signal of confabulated legal citation."""
    cited = cited_sections(answer)
    if not cited:
        return float("nan")
    nums = lambda xs: {SECTION_CITE_RE.search(x).group(1) for x in xs if SECTION_CITE_RE.search(x)}
    c, g = nums(cited), nums(gold)
    return len(c & g) / len(c) if c else 0.0


def abstained(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in ABSTENTION_MARKERS)


def score_record(rec: dict, k_values=(1, 3, 5, 10)) -> dict:
    """rec: {retrieved:[section_id], gold:[section_id], answerable:bool, answer:str}"""
    retrieved, gold = rec.get("retrieved", []), rec.get("gold", [])
    out = {}

    if rec.get("answerable", True) and gold:
        for k in k_values:
            out[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
            out[f"hit@{k}"] = hit_at_k(retrieved, gold, k)
        out["mrr"] = mrr(retrieved, gold)
        out["ndcg@5"] = ndcg_at_k(retrieved, gold, 5)
        out["citation_precision"] = citation_precision(rec.get("answer", ""), gold)
        out["false_abstention"] = float(abstained(rec.get("answer", "")))
    else:
        # Unanswerable probes: the only correct behaviour is to decline.
        out["correct_abstention"] = float(abstained(rec.get("answer", "")))
        out["hallucinated_on_unanswerable"] = 1.0 - out["correct_abstention"]
    return out


def aggregate(records: list[dict]) -> dict:
    keys, agg = set(), {}
    for r in records:
        keys |= set(r["scores"].keys())
    for key in sorted(keys):
        vals = [r["scores"][key] for r in records
                if key in r["scores"] and not math.isnan(r["scores"][key])]
        if vals:
            agg[key] = {"mean": sum(vals) / len(vals), "n": len(vals)}
    return agg

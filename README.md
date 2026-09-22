# RAG evaluation on German tenancy law

Primary-research artefact for the dissertation *Evaluating the Effectiveness of
Retrieval-Augmented Generation (RAG) in Domain-Specific AI Chatbots*.

## Corpus
BGB §§ 535–580a (Mietrecht), BetrKV, HeizkostenV. Source: the XML exports of
gesetze-im-internet.de via github.com/bundestag/gesetze. German federal statute
is free of copyright under § 5 UrhG, so the corpus can be redistributed in the
appendix — worth stating explicitly in the methodology.

122 sections, ~19,000 words, 214 chunks at the default settings.

## Conditions
| id | retrieval | note |
|----|-----------|------|
| C0 | none | parametric-knowledge baseline |
| C1 | dense only | the standard "naive RAG" setup |
| C2 | hybrid (BM25 + dense, RRF) | |
| C3 | hybrid + cross-encoder rerank | |

## Ablations
chunking strategy (structural vs fixed), chunk size (512/900/1400), k (3/5/10).

## Run
    pip install -r requirements.txt
    python src/ingest.py data/corpus
    python src/run_experiment.py --dry-run              # retrieval only, no model
    export ANTHROPIC_API_KEY=...
    python src/run_experiment.py --generator anthropic
    python src/run_experiment.py --generator ollama --model qwen2.5:7b

Outputs land in results/ as a per-item JSONL (Appendix C) and a summary CSV.

## Before you report anything
- Expand data/gold/gold_seed.jsonl from 18 to 80–100 items (~20% unanswerable).
- Install sentence-transformers; the TF-IDF fallback is a stand-in only, and
  any number produced with it must be labelled as such.

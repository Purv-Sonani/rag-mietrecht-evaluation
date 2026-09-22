"""
Corpus ingestion and chunking for the Mietrecht RAG evaluation.

Two chunking strategies are implemented so that chunking can be treated as an
experimental variable rather than a fixed implementation detail:

  - "fixed"     : naive sliding window over the raw character stream. This is
                  what most off-the-shelf RAG tutorials do.
  - "structural": splits on statutory section boundaries (§), then sub-splits
                  only sections that exceed the size budget, keeping the
                  section heading attached to every resulting chunk.

The comparison between the two is one of the ablations in Chapter 4.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

SECTION_RE = re.compile(r"^#{2,6}\s+(?:\(XXXX\)\s+)?(§+\s*\d+[a-z]?)\s*(.*)$", re.M)

# Human-readable source labels used in citations shown to the participant raters.
SOURCE_LABELS = {
    "bgb_miete": "BGB",
    "betrkv": "BetrKV",
    "heizkostenv": "HeizkostenV",
}


@dataclass
class Chunk:
    chunk_id: str
    section_id: str      # e.g. "BGB § 536"  -> the retrieval ground-truth unit
    heading: str
    source: str
    text: str
    n_chars: int

    def to_dict(self) -> dict:
        return asdict(self)


def _read_corpus(corpus_dir: Path) -> list[tuple[str, str]]:
    docs = []
    for path in sorted(corpus_dir.glob("*.md")):
        docs.append((path.stem, path.read_text(encoding="utf-8")))
    return docs


def parse_sections(stem: str, text: str) -> list[dict]:
    """Split one source document into statutory sections."""
    label = SOURCE_LABELS.get(stem, stem.upper())
    matches = list(SECTION_RE.finditer(text))
    sections = []

    for i, m in enumerate(matches):
        para = re.sub(r"\s+", " ", m.group(1)).strip()
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Repealed provisions carry no content and would only pollute the index.
        if not body or "weggefallen" in heading.lower():
            continue

        sections.append(
            {
                "section_id": f"{label} {para}",
                "heading": heading,
                "source": label,
                "text": body,
            }
        )
    return sections


def _window(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(text), step):
        piece = text[start : start + size]
        if piece.strip():
            out.append(piece)
        if start + size >= len(text):
            break
    return out


def chunk_structural(sections: list[dict], size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for sec in sections:
        pieces = _window(sec["text"], size, overlap)
        for j, piece in enumerate(pieces):
            # Prepending the heading means a chunk stays interpretable even when
            # it is retrieved in isolation, which matters a lot for statute text.
            body = f"{sec['section_id']} {sec['heading']}\n{piece.strip()}"
            chunks.append(
                Chunk(
                    chunk_id=f"{sec['section_id']}#{j}".replace(" ", "_"),
                    section_id=sec["section_id"],
                    heading=sec["heading"],
                    source=sec["source"],
                    text=body,
                    n_chars=len(body),
                )
            )
    return chunks


def chunk_fixed(sections: list[dict], size: int, overlap: int) -> list[Chunk]:
    """Concatenate everything, then window blindly. Section attribution is
    recovered by offset so that retrieval metrics remain computable."""
    stream, spans = [], []
    cursor = 0
    for sec in sections:
        block = f"{sec['section_id']} {sec['heading']}\n{sec['text']}\n\n"
        stream.append(block)
        spans.append((cursor, cursor + len(block), sec["section_id"], sec["source"]))
        cursor += len(block)
    full = "".join(stream)

    chunks = []
    step = max(1, size - overlap)
    for k, start in enumerate(range(0, len(full), step)):
        piece = full[start : start + size]
        if not piece.strip():
            continue
        mid = start + len(piece) // 2
        section_id, source = next(
            ((s, src) for a, b, s, src in spans if a <= mid < b), ("UNKNOWN", "UNKNOWN")
        )
        chunks.append(
            Chunk(
                chunk_id=f"fixed_{k}",
                section_id=section_id,
                heading="",
                source=source,
                text=piece.strip(),
                n_chars=len(piece.strip()),
            )
        )
        if start + size >= len(full):
            break
    return chunks


def build(
    corpus_dir: str | Path,
    strategy: str = "structural",
    size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    corpus_dir = Path(corpus_dir)
    sections: list[dict] = []
    for stem, text in _read_corpus(corpus_dir):
        sections.extend(parse_sections(stem, text))

    if strategy == "structural":
        return chunk_structural(sections, size, overlap)
    if strategy == "fixed":
        return chunk_fixed(sections, size, overlap)
    raise ValueError(f"unknown chunking strategy: {strategy}")


def sections_index(corpus_dir: str | Path) -> dict[str, dict]:
    """section_id -> section record. Used to build and validate the gold set."""
    corpus_dir = Path(corpus_dir)
    idx = {}
    for stem, text in _read_corpus(corpus_dir):
        for sec in parse_sections(stem, text):
            idx[sec["section_id"]] = sec
    return idx


if __name__ == "__main__":
    import sys

    corpus = sys.argv[1] if len(sys.argv) > 1 else "data/corpus"
    idx = sections_index(corpus)
    print(f"sections parsed: {len(idx)}")
    for strat, size in [("structural", 900), ("fixed", 900)]:
        cs = build(corpus, strat, size, 150)
        avg = sum(c.n_chars for c in cs) / len(cs)
        print(f"  {strat:11s} size={size} -> {len(cs):4d} chunks, avg {avg:.0f} chars")
    out = Path("data/processed_chunks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([c.to_dict() for c in build(corpus)], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"wrote {out}")

"""
Generator adapters.

All adapters expose the same .complete(system, user) -> str so that the model
is a swappable variable in the experiment rather than something baked into the
pipeline. Temperature is pinned to 0 everywhere: with a 15-day window there is
no budget for multi-sample variance estimation, so determinism is the honest
alternative, and it must be stated as such in the methodology.
"""

from __future__ import annotations

import json
import os
import urllib.request


class AnthropicGenerator:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 700):
        self.model, self.max_tokens = model, max_tokens
        self.name = f"anthropic:{model}"

    def complete(self, system: str, user: str) -> str:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": 0,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                }
            ).encode(),
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
            },
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return "".join(b.get("text", "") for b in data.get("content", []))


class OllamaGenerator:
    """Local models via Ollama. Use for the small-model condition."""

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434"):
        self.model, self.host = model, host
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str) -> str:
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(
                {
                    "model": self.model,
                    "stream": False,
                    "options": {"temperature": 0, "seed": 42},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
            ).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"]


class GeminiGenerator:
    """Google AI Studio free tier: slow, and prone to both 429s and hangs."""

    _last = 0.0

    def __init__(self, model="gemini-3.6-flash", max_retries=12, rpm=6):
        self.model, self.max_retries = model, max_retries
        self.min_interval = 60.0 / rpm
        self.name = f"gemini:{model}"

    def complete(self, system, user):
        import re as _re
        import time
        import urllib.error

        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent")
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 700},
        }
        headers = {"content-type": "application/json",
                   "x-goog-api-key": os.environ.get("GOOGLE_API_KEY", "")}
        pause = 20.0

        for attempt in range(self.max_retries):
            gap = self.min_interval - (time.time() - GeminiGenerator._last)
            if gap > 0:
                time.sleep(gap)
            GeminiGenerator._last = time.time()
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(payload).encode(), headers=headers)
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = json.loads(r.read())
                parts = data["candidates"][0]["content"]["parts"]
                return "".join(q.get("text", "") for q in parts)
            except (KeyError, IndexError):
                return ""
            except urllib.error.HTTPError as e:
                detail = e.read().decode()[:400]
                if e.code not in (429, 500, 503):
                    raise RuntimeError(f"Gemini {e.code}: {detail}") from None
                m = _re.search(r"retry in ([0-9.]+)", detail)
                wait = float(m.group(1)) + 2.0 if m else pause
                print(f"    [{e.code}] wait {wait:.0f}s", flush=True)
                time.sleep(wait)
                pause = min(pause * 1.5, 90)
            except Exception as e:
                # Timeouts and dropped connections are routine on the free tier.
                print(f"    [{type(e).__name__}] wait {pause:.0f}s", flush=True)
                time.sleep(pause)
                pause = min(pause * 1.5, 90)
        return ""


class EchoGenerator:
    """Offline stand-in so the harness can be exercised without any model."""

    name = "echo"

    def complete(self, system: str, user: str) -> str:
        return "[echo] " + user[-300:]


# --- Prompts ------------------------------------------------------------
# Kept here rather than inline so they can be reproduced verbatim in Appendix B.

SYSTEM_NO_RAG = (
    "Du bist ein Assistent für deutsches Mietrecht. Beantworte die Frage des "
    "Nutzers. Nenne die einschlägigen Paragraphen. Wenn du die Antwort nicht "
    "sicher weißt, sage das ausdrücklich."
)

SYSTEM_RAG = (
    "Du bist ein Assistent für deutsches Mietrecht. Beantworte die Frage des "
    "Nutzers AUSSCHLIESSLICH auf Grundlage der bereitgestellten Auszüge aus "
    "Gesetzestexten. Zitiere die Paragraphen, auf die du dich stützt. Wenn die "
    "Auszüge die Frage nicht beantworten, antworte genau: "
    "'Die vorliegenden Quellen enthalten dazu keine Angabe.'"
)


def build_rag_prompt(question: str, hits) -> str:
    ctx = "\n\n".join(f"[{i+1}] {h.section_id}\n{h.text}" for i, h in enumerate(hits))
    return f"AUSZÜGE:\n{ctx}\n\nFRAGE: {question}"

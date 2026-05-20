"""Single-function LLM plug for paraphrase generation.

Hits any OpenAI-compatible chat-completions endpoint via ``requests``:
``llama.cpp -c "..." --port 8080``, ``vllm``, ``ollama``, ``openai``, ``together``,
``litellm``, …

No SDK dependency. Pass ``base_url``, ``model``, ``api_key`` as needed.
"""

from __future__ import annotations

import json
from typing import List, Optional

import requests


_SYS = (
    "You generate diverse paraphrases of user voice-assistant commands. "
    "Each paraphrase must preserve the exact intent of the seed examples. "
    "Vary phrasing, register, word order; keep meaning."
)

_PROMPT = (
    "Generate {n} paraphrases for the intent '{intent}'. "
    "Seed examples:\n{seeds}\n\n"
    "Return ONLY a JSON list of strings, no commentary."
)


def llm_paraphrase(
    intent: str,
    seeds: List[str],
    *,
    n: int = 20,
    base_url: str = "http://localhost:8080/v1",
    model: str = "local",
    api_key: Optional[str] = None,
    temperature: float = 0.8,
    timeout: float = 60.0,
) -> List[str]:
    """Ask an OpenAI-compatible endpoint for ``n`` paraphrases of an intent.

    Parameters mirror the OpenAI Chat Completions request body. Returns
    a flat list of paraphrase strings; non-JSON responses are parsed
    line-by-line as a fallback.
    """
    seeds_block = "\n".join(f"- {s}" for s in seeds)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": _PROMPT.format(n=n, intent=intent, seeds=seeds_block)},
        ],
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload, headers=headers, timeout=timeout,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()

    try:
        first = text.find("[")
        last = text.rfind("]")
        if first != -1 and last != -1 and last > first:
            data = json.loads(text[first:last + 1])
            if isinstance(data, list):
                return [str(s).strip() for s in data if str(s).strip()]
    except json.JSONDecodeError:
        pass
    return [line.lstrip("-*0123456789. \t").strip() for line in text.splitlines() if line.strip()]

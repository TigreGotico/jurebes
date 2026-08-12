"""Runnable end-to-end demo of the LLM-augmentation loop.

Set ``JUREBES_LLM_ENDPOINT`` to an OpenAI-compatible chat-completions URL
(e.g. ``http://localhost:8080/v1`` for llama.cpp's server, or
``https://api.openai.com/v1``). When the env var is unset the script
runs against a built-in stub paraphraser so the loop logic is still
exercised offline.

Usage::

    # local llama.cpp server
    export JUREBES_LLM_ENDPOINT=http://localhost:8080/v1
    export JUREBES_LLM_MODEL=llama-3-8b-instruct
    python examples/llm_augmentation/run_example.py

    # openai
    export JUREBES_LLM_ENDPOINT=https://api.openai.com/v1
    export JUREBES_LLM_MODEL=gpt-4o-mini
    export JUREBES_LLM_API_KEY=sk-...
    python examples/llm_augmentation/run_example.py
"""

from __future__ import annotations

import os
from functools import partial

from loop import augment_loop


# ─── Seed data: 3 intents × 4 samples ───────────────────────────────
SEED_INTENTS = {
    "play_song": [
        "play africa", "put on bohemian rhapsody",
        "queue up hey jude", "start smells like teen spirit",
    ],
    "set_timer": [
        "set a timer for 5 minutes", "wake me in 10 minutes",
        "remind me in half an hour", "timer for 2 minutes",
    ],
    "weather": [
        "what is the weather today", "how is the weather in lisbon",
        "is it raining outside", "weather forecast",
    ],
}

# ─── Held-out eval set (FROZEN — never regenerated) ─────────────────
EVAL_X = [
    "play africa", "throw on hey jude", "spin bohemian rhapsody", "blast smells like teen spirit",
    "set timer for ten minutes", "wake me up in an hour", "remind me in five minutes",
    "what's the weather", "is it going to rain today", "weather in paris please",
]
EVAL_Y = [
    "play_song", "play_song", "play_song", "play_song",
    "set_timer", "set_timer", "set_timer",
    "weather", "weather", "weather",
]


def _stub_paraphrase(intent, seeds, *, n=20, **_):
    """Offline fallback. Generates trivial reorderings so the loop still runs."""
    out = []
    fillers = ["please", "could you", "can you", "i want you to", "now", "today"]
    for s in seeds:
        for f in fillers[:max(1, n // len(seeds))]:
            out.append(f"{f} {s}".strip())
            out.append(f"{s} {f}".strip())
    return list(dict.fromkeys(out))[:n]


def _real_paraphrase(intent, seeds, *, n=20, **kwargs):
    from llm import llm_paraphrase
    return llm_paraphrase(intent, seeds, n=n, **kwargs)


def main():
    endpoint = os.environ.get("JUREBES_LLM_ENDPOINT")
    if endpoint:
        paraphrase_fn = partial(
            _real_paraphrase,
            base_url=endpoint,
            model=os.environ.get("JUREBES_LLM_MODEL", "local"),
            api_key=os.environ.get("JUREBES_LLM_API_KEY") or None,
        )
        print(f"using LLM endpoint: {endpoint}")
    else:
        paraphrase_fn = _stub_paraphrase
        print("JUREBES_LLM_ENDPOINT not set — using offline stub paraphraser")

    clf, history = augment_loop(
        SEED_INTENTS,
        paraphrase_fn=paraphrase_fn,
        eval_X=EVAL_X,
        eval_y=EVAL_Y,
        baseline="logreg",
        n_per_intent=12,
        n_rounds=4,
        hard_conf_max=0.6,
    )

    print("\nround | gen | skip | hard | suspect | macro_f1")
    print("------+-----+------+------+---------+---------")
    for i, r in enumerate(history):
        print(f"{i:5d} | {r.n_generated:3d} | {r.n_skip:4d} | {r.n_hard:4d} | "
              f"{r.n_suspect:7d} | {r.eval_macro_f1:.4f}")

    print("\nfinal predictions on held-out:")
    for x in EVAL_X:
        r = clf.predict(x)
        print(f"  {r.intent:12s} ({r.confidence:.2f}) ← {x}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()

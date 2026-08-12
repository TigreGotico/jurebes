"""Bootstrap intent labels for the meteocat dataset via self-training.

Fetches `crodri/meteocat` via Hugging Face datasets, hand-labels ~5
utterances each for six weather-related seed intents, then runs
`jurebes.semi_supervised.self_train` on the remaining unlabeled
`instruction` field.

Run with::

    pip install jurebes[hf]
    python examples/semi_supervised/run_meteocat.py
"""

from __future__ import annotations

import random
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from jurebes import IntentClassifier
from jurebes.semi_supervised import self_train


SEED_INTENTS: Dict[str, List[str]] = {
    "forecast_query": [
        "what is the weather forecast for tomorrow",
        "give me the forecast for this weekend",
        "weather forecast for next week",
        "what's the forecast",
        "show me the weekly forecast",
    ],
    "temperature_query": [
        "what is the temperature outside",
        "how hot is it today",
        "current temperature in barcelona",
        "what is the temperature in girona",
        "how cold is it",
    ],
    "condition_query": [
        "is it raining now",
        "is it snowing in the mountains",
        "is it sunny outside",
        "are there clouds today",
        "is it foggy this morning",
    ],
    "time_specific_query": [
        "what will the weather be at noon",
        "weather at midnight",
        "how will it be tonight",
        "weather this afternoon",
        "weather tomorrow morning",
    ],
    "location_query": [
        "weather in tarragona",
        "what is the weather like in lleida",
        "weather conditions in costa brava",
        "weather in the pyrenees",
        "weather in barcelona",
    ],
    "comparison_query": [
        "is it hotter today than yesterday",
        "compare the temperature with last week",
        "is it colder than usual",
        "is tomorrow going to be warmer",
        "compare rainfall this month with last month",
    ],
}


def _load_meteocat_instructions() -> List[str]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("crodri/meteocat")
    split = ds[next(iter(ds.keys()))]
    instructions = [str(r["instruction"]) for r in split if r.get("instruction")]
    return instructions


def main() -> None:
    rng = random.Random(42)

    labeled_X: List[str] = []
    labeled_y: List[str] = []
    for intent, examples in SEED_INTENTS.items():
        labeled_X.extend(examples)
        labeled_y.extend([intent] * len(examples))

    instructions = _load_meteocat_instructions()
    rng.shuffle(instructions)
    holdout = instructions[:20]
    pool = instructions[20:1020]

    template = IntentClassifier(
        Pipeline([
            ("v", TfidfVectorizer(min_df=1, ngram_range=(1, 2))),
            ("c", LogisticRegression(max_iter=1000)),
        ]),
    )

    result = self_train(
        template,
        labeled_X,
        labeled_y,
        pool,
        confidence_threshold=0.7,
        selection="per_class_quota",
        k_per_round=10,
        max_rounds=8,
    )

    print(f"seed size: {len(SEED_INTENTS) * 5}")
    print(f"final labeled pool size: {len(result.labeled_X)}")
    print(f"added per round: {result.added_per_round}")
    print(f"wall time: {result.wall_time_s:.1f}s")
    print()
    print("holdout predictions:")
    for utt in holdout:
        pred = result.classifier.predict(utt)
        print(f"  [{pred.intent:<22}] {utt[:80]}")


if __name__ == "__main__":
    main()

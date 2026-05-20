"""Loader for OpenVoiceOS/intents-for-eval — multilingual OVOS intent + slot bench.

Each language config exposes three sub-configs on HuggingFace:

- ``<lang>-templates`` — Padatious-style intent templates with `{slot}`
  placeholders and per-slot example values.
- ``<lang>-keywords``  — per-intent required and optional vocabulary
  for keyword-based matching.
- ``<lang>-test``      — held-out utterances with gold intent and slots.

`load_intents_for_eval(lang)` returns a single dict bundling the three.
"""

from __future__ import annotations

from typing import Any, Dict, List

SUPPORTED_LANGS = (
    "en-US", "pt-PT", "pt-BR", "es-ES", "fr-FR", "de-DE",
    "it-IT", "nl-NL", "ca-ES", "gl-ES", "da-DK", "eu-ES",
)

_HF_ID = "OpenVoiceOS/intents-for-eval"


def _load_test_jsonl_directly(lang: str) -> List[Dict[str, Any]]:
    """Fallback JSONL loader bypassing pyarrow schema inference."""
    import json
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=_HF_ID,
        repo_type="dataset",
        filename=f"{lang}/test.jsonl",
    )
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_intents_for_eval(lang: str = "en-US") -> Dict[str, Any]:
    """Load the templates / keywords / test triple for one language.

    Returns a dict::

        {
            "lang": <lang>,
            "intent_samples": {intent_id: [template, ...]},
            "entity_samples": {slot_name: [example, ...]},
            "keywords":       {intent_id: {"required": {...}, "optional": {...}}},
            "test":           [{"utterance": ..., "expected_intent": ...,
                                "expected_slots": {slot: value, ...}}, ...],
        }

    Slot examples from every template are deduplicated into ``entity_samples``
    so the slot tagger can learn from gazetteer-style supervision.
    """
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"unsupported lang {lang!r}; choose from {SUPPORTED_LANGS}")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install jurebes[hf] to fetch intents-for-eval") from exc

    templates = load_dataset(_HF_ID, f"{lang}-templates")["train"]
    keywords = load_dataset(_HF_ID, f"{lang}-keywords")["train"]
    try:
        test = list(load_dataset(_HF_ID, f"{lang}-test")["test"])
    except Exception:
        # pyarrow JSON inference can choke when slot values cross int/string
        # boundaries; fall back to direct JSONL parsing of the raw test file.
        test = _load_test_jsonl_directly(lang)

    intent_samples: Dict[str, List[str]] = {}
    entity_samples: Dict[str, List[str]] = {}
    for row in templates:
        intent_samples.setdefault(row["intent_id"], []).append(row["template"])
        for slot in row["slots"] or []:
            bucket = entity_samples.setdefault(slot["name"], [])
            for ex in slot.get("examples") or []:
                if ex and ex not in bucket:
                    bucket.append(ex)

    kw: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for row in keywords:
        kw[row["intent_id"]] = {
            "required": {k: v for k, v in (row.get("required_vocab") or {}).items() if v},
            "optional": {k: v for k, v in (row.get("optional_vocab") or {}).items() if v},
        }

    test_rows = []
    for row in test:
        expected_slots = {
            k: v for k, v in (row.get("expected_slots") or {}).items() if v
        }
        test_rows.append({
            "utterance": row["utterance"],
            "expected_intent": row["expected_intent"],
            "expected_slots": expected_slots,
            "split": row.get("split"),
            "domain": row.get("domain"),
        })

    return {
        "lang": lang,
        "intent_samples": intent_samples,
        "entity_samples": entity_samples,
        "keywords": kw,
        "test": test_rows,
    }

"""Loader for OpenVoiceOS/massive-templates — 51-language Padatious-style bench.

Each language config exposes two sub-configs on HuggingFace:

- ``<lang>-templates`` — Padatious-style intent templates with `{slot}`
  placeholders and per-slot example values (one realisation each, from
  the gold MASSIVE annotations).
- ``<lang>-test``      — held-out utterances with gold MASSIVE intent
  (``domain:intent_name``) and a fixed-schema slot dict.

`load_massive_templates(lang)` returns the same shape as
:func:`load_intents_for_eval`: intent_samples (with `{slot}` placeholders
preserved), template_samples (raw), entity_samples (slot value gazetteer),
and the test split.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from jurebes.datasets.canonical.intents_for_eval import _expand_template


SUPPORTED_LANGS = (
    "af-ZA", "am-ET", "ar-SA", "az-AZ", "bn-BD", "ca-ES", "cy-GB", "da-DK",
    "de-DE", "el-GR", "en-US", "es-ES", "fa-IR", "fi-FI", "fr-FR", "he-IL",
    "hi-IN", "hu-HU", "hy-AM", "id-ID", "is-IS", "it-IT", "ja-JP", "jv-ID",
    "ka-GE", "km-KH", "kn-IN", "ko-KR", "lv-LV", "ml-IN", "mn-MN", "ms-MY",
    "my-MM", "nb-NO", "nl-NL", "pl-PL", "pt-PT", "ro-RO", "ru-RU", "sl-SL",
    "sq-AL", "sv-SE", "sw-KE", "ta-IN", "te-IN", "th-TH", "tl-PH", "tr-TR",
    "ur-PK", "vi-VN", "zh-CN", "zh-TW",
)

_HF_ID = "OpenVoiceOS/massive-templates"


def _load_test_jsonl_directly(lang: str) -> List[Dict[str, Any]]:
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


def load_massive_templates(lang: str = "en-US", expand_templates: bool = True,
                           expansions_per_template: int = 3) -> Dict[str, Any]:
    """Load the templates + test pair for one of 51 MASSIVE languages.

    Returns a dict::

        {
            "lang": <lang>,
            "intent_samples":   {intent_id: [realised utterance, ...]},
            "template_samples": {intent_id: [raw template, ...]},
            "entity_samples":   {slot_name: [example, ...]},
            "test":             [{"utterance": ..., "expected_intent": ...,
                                  "expected_slots": {slot: value, ...},
                                  "split": ...}, ...],
        }

    With ``expand_templates=True`` (default), ``{slot}`` placeholders in
    each template are substituted with the slot's example values, mirroring
    the OVOS Padatious behaviour.
    """
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"unsupported lang {lang!r}; choose from {SUPPORTED_LANGS}")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install jurebes[hf] to fetch massive-templates") from exc

    templates = load_dataset(_HF_ID, f"{lang}-templates")["train"]
    try:
        test = list(load_dataset(_HF_ID, f"{lang}-test")["test"])
    except Exception:
        test = _load_test_jsonl_directly(lang)

    intent_samples: Dict[str, List[str]] = {}
    template_samples: Dict[str, List[str]] = {}
    entity_samples: Dict[str, List[str]] = {}
    for row in templates:
        slots = row["slots"] or []
        template_samples.setdefault(row["intent_id"], []).append(row["template"])
        if expand_templates:
            expanded, _ = _expand_template(row["template"], slots, expansions_per_template)
            for utt in expanded:
                intent_samples.setdefault(row["intent_id"], []).append(utt)
        else:
            intent_samples.setdefault(row["intent_id"], []).append(row["template"])
        for slot in slots:
            bucket = entity_samples.setdefault(slot["name"], [])
            for ex in slot.get("examples") or []:
                if ex and ex not in bucket:
                    bucket.append(ex)

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
        })

    return {
        "lang": lang,
        "intent_samples": intent_samples,
        "template_samples": template_samples,
        "entity_samples": entity_samples,
        "test": test_rows,
    }

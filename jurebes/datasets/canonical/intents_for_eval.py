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

import re
from typing import Any, Dict, List, Tuple

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


_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _expand_template(template: str, slots: List[Dict[str, Any]],
                     expand_to: int = 3) -> Tuple[List[str], List[str]]:
    """Substitute slot examples into a template.

    Returns ``(expanded, originals)``: ``expanded`` is the list of
    realised utterances (`"play {song}"` → `["play bohemian rhapsody",
    ...]`), ``originals`` is the same template kept unchanged so a
    classifier that wants to learn placeholder patterns can also see it.

    Multi-slot templates use a zip across the slot example lists so the
    expansion stays at ~``expand_to`` variants instead of exploding into
    a cartesian product.
    """
    placeholders = _PLACEHOLDER_RE.findall(template)
    if not placeholders:
        return [template], [template]

    examples_by_slot: Dict[str, List[str]] = {}
    for slot in slots or []:
        name = slot.get("name")
        examples = [e for e in (slot.get("examples") or []) if e]
        if name and examples:
            examples_by_slot[name] = examples
    if not all(p in examples_by_slot for p in placeholders):
        return [template], [template]

    n = min(expand_to, *(len(examples_by_slot[p]) for p in placeholders))
    expanded: List[str] = []
    for i in range(n):
        out = template
        for p in placeholders:
            out = out.replace("{" + p + "}", examples_by_slot[p][i])
        expanded.append(out)
    return expanded, [template]


def load_intents_for_eval(lang: str = "en-US", expand_templates: bool = True,
                          expansions_per_template: int = 3) -> Dict[str, Any]:
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
    template_samples: Dict[str, List[str]] = {}
    entity_samples: Dict[str, List[str]] = {}
    n_dropped = 0
    for row in templates:
        slots = row["slots"] or []
        template = row["template"]
        # Drop templates that the expander cannot parse. Some language
        # corpora use the alternation parenthesis convention informally
        # (Basque case-marker letters like "{contact}-(r)i"); the
        # alternation parser requires a "|" inside parens and rejects
        # single-branch groups. Bad rows are dropped at load time so the
        # downstream training does not crash.
        if expand_templates:
            try:
                expanded, _ = _expand_template(
                    template, slots, expansions_per_template,
                )
            except Exception:
                n_dropped += 1
                continue
            for utt in expanded:
                intent_samples.setdefault(row["intent_id"], []).append(utt)
        else:
            intent_samples.setdefault(row["intent_id"], []).append(template)
        template_samples.setdefault(row["intent_id"], []).append(template)
        for slot in slots:
            bucket = entity_samples.setdefault(slot["name"], [])
            for ex in slot.get("examples") or []:
                if ex and ex not in bucket:
                    bucket.append(ex)
    if n_dropped:
        import logging
        logging.getLogger(__name__).warning(
            "intents-for-eval %s: dropped %d malformed templates",
            lang, n_dropped,
        )

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
        "template_samples": template_samples,
        "entity_samples": entity_samples,
        "keywords": kw,
        "test": test_rows,
    }

"""FastAPI wrapper around a trained IntentClassifier.

Single-file production-style service: load a `.joblib` model on
startup, serve `/predict`, `/predict_batch`, `/predict_proba` and
`/match_intent` (OPM-compatible confidence-gated). No new runtime
dependency on the jurebes side — install ``fastapi`` and ``uvicorn``
separately to run this script.

Usage::

    pip install fastapi uvicorn
    MODEL_PATH=/path/to/model.joblib uvicorn examples.serve.app:app
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from jurebes import IntentClassifier


# ── load model on import ────────────────────────────────────────────
_MODEL_PATH = os.environ.get("MODEL_PATH", "")
if not _MODEL_PATH:
    raise RuntimeError(
        "MODEL_PATH environment variable not set; "
        "save a model with `jurebes train --dataset ... --out model.joblib`"
    )

_CLF: IntentClassifier = IntentClassifier.load(_MODEL_PATH)


# ── thresholds (OPM-compatible naming) ──────────────────────────────
CONF_HIGH = float(os.environ.get("JUREBES_CONF_HIGH", "0.8"))
CONF_MED = float(os.environ.get("JUREBES_CONF_MED", "0.5"))
CONF_LOW = float(os.environ.get("JUREBES_CONF_LOW", "0.2"))


# ── Pydantic models ─────────────────────────────────────────────────
class Utterance(BaseModel):
    utterance: str = Field(..., min_length=1)


class Utterances(BaseModel):
    utterances: List[str]


class IntentResponse(BaseModel):
    intent: str
    confidence: float
    entities: Dict[str, str]
    utterance: str


class MatchResponse(BaseModel):
    intent: Optional[str] = None
    confidence: float = 0.0
    band: Optional[str] = None
    entities: Dict[str, str] = Field(default_factory=dict)
    utterance: str = ""


# ── app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="jurebes",
    version="0.2",
    description="Classical-ML intent classification service.",
)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": _MODEL_PATH}


@app.get("/info")
def info():
    return {
        "model_path": _MODEL_PATH,
        "intents": sorted(_CLF._samples.keys()) if hasattr(_CLF, "_samples") else [],
        "thresholds": {"high": CONF_HIGH, "med": CONF_MED, "low": CONF_LOW},
        "tagger": _CLF.tagger.__class__.__name__ if _CLF.tagger else None,
    }


@app.post("/predict", response_model=IntentResponse)
def predict(body: Utterance):
    r = _CLF.predict(body.utterance)
    return IntentResponse(
        intent=r.intent,
        confidence=float(r.confidence),
        entities=r.entities,
        utterance=r.utterance,
    )


@app.post("/predict_batch", response_model=List[IntentResponse])
def predict_batch(body: Utterances):
    results = _CLF.predict_batch(body.utterances)
    return [
        IntentResponse(intent=r.intent, confidence=float(r.confidence),
                       entities=r.entities, utterance=r.utterance)
        for r in results
    ]


@app.post("/predict_proba", response_model=List[IntentResponse])
def predict_proba(body: Utterance):
    return [
        IntentResponse(intent=r.intent, confidence=float(r.confidence),
                       entities=r.entities, utterance=r.utterance)
        for r in _CLF.predict_proba(body.utterance)
    ]


@app.post("/match_intent", response_model=MatchResponse)
def match_intent(body: Utterance):
    """OPM-compatible confidence-banded matcher.

    Returns ``band="high"``/``"med"``/``"low"`` per the configured
    thresholds, or no intent at all if confidence falls below
    ``JUREBES_CONF_LOW``. Mirrors the production OPM pipeline plugin.
    """
    r = _CLF.predict(body.utterance)
    conf = float(r.confidence)
    band: Optional[str]
    if conf >= CONF_HIGH:
        band = "high"
    elif conf >= CONF_MED:
        band = "med"
    elif conf >= CONF_LOW:
        band = "low"
    else:
        return MatchResponse(utterance=body.utterance)
    return MatchResponse(
        intent=r.intent, confidence=conf, band=band,
        entities=r.entities, utterance=r.utterance,
    )

# FastAPI deployment example

Single-file FastAPI wrapper that loads a `.joblib` model and exposes
intent-classification endpoints over HTTP. Useful as a starting point
for jurebes deployments outside the OVOS pipeline-plugin path.

## Files

| file | purpose |
|---|---|
| [`app.py`](app.py) | FastAPI application; loads `IntentClassifier` on startup. |

## Endpoints

| method | path | request body | response |
|---|---|---|---|
| `GET`  | `/healthz`         | — | `{status, model}` |
| `GET`  | `/info`            | — | `{model_path, intents, thresholds, tagger}` |
| `POST` | `/predict`         | `{utterance: str}` | top-1 `IntentResponse` |
| `POST` | `/predict_batch`   | `{utterances: list[str]}` | list of `IntentResponse` |
| `POST` | `/predict_proba`   | `{utterance: str}` | ranked list of `IntentResponse` |
| `POST` | `/match_intent`    | `{utterance: str}` | OPM-compatible band (`high`/`med`/`low` or none) |

The `/match_intent` endpoint mirrors the OVOS pipeline plugin's
`match_high`/`match_medium`/`match_low` semantics. Threshold values
come from env vars `JUREBES_CONF_HIGH`/`MED`/`LOW`; defaults `0.8`/`0.5`/`0.2`.

## Run

```bash
pip install fastapi uvicorn
jurebes train --dataset toy.csv --baseline linear_svc_char --out /tmp/m.joblib

MODEL_PATH=/tmp/m.joblib uvicorn examples.serve.app:app --host 0.0.0.0 --port 8000
```

```bash
curl -s -XPOST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"utterance": "play africa"}'
# {"intent":"play_song","confidence":0.97,"entities":{},"utterance":"play africa"}

curl -s -XPOST http://localhost:8000/match_intent \
  -H "Content-Type: application/json" \
  -d '{"utterance": "play africa"}'
# {"intent":"play_song","confidence":0.97,"band":"high","entities":{},"utterance":"play africa"}
```

## Notes

- The model is loaded once on import — restart the server to swap models.
- No batching beyond `/predict_batch` and no async worker pool; this is
  a starting template. For production scale, run several `uvicorn`
  workers behind a load balancer.
- `fastapi` and `uvicorn` are not jurebes dependencies; install them
  separately. The framework itself stays sklearn-only.
- For an in-OVOS deployment, prefer the [`OPM pipeline plugin`](../../docs/opm.md)
  over this HTTP wrapper.

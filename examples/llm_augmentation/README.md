# LLM-driven data augmentation

A three-bucket active-learning loop with jurebes as the cheap classification oracle.

## Files

| file | purpose |
|---|---|
| [`llm.py`](llm.py) | Single-function `requests`-based plug for any OpenAI-compatible chat-completions endpoint. |
| [`loop.py`](loop.py) | `augment_loop()` — three-bucket scoring + retrain + eval. |
| [`run_example.py`](run_example.py) | End-to-end demo with three toy intents and a frozen held-out eval set. |

## Concept

```
                          ┌──────────────┐
              ┌──────────►│ jurebes.fit  │
              │           └──────┬───────┘
              │                  │
              │           ┌──────▼──────────┐
              │           │ score paraphrases│
              │           └──────┬──────────┘
              │                  │
              │     skip ◄──── HIGH conf + correct
              │                  │
              │   KEEP HARD ◄── LOW conf + correct       ← gradient lives here
              │                  │
              │   JUDGE    ◄── wrong prediction          ← LLM verifies intent preserved
              │                  │
              │                  ▼
              └────────── append to training set
```

Three failure modes the loop guards against:

- **Label noise** — naive "keep wrong predictions" silently corrupts the dataset when the LLM drifts off the requested intent. The optional `judge_fn` callback resolves this.
- **Reflection** — a held-out eval set regenerated each round measures progress against itself. The example uses a frozen `EVAL_X`/`EVAL_Y` to detect plateau via `early_stop_patience`.
- **LLM-distribution overfit** — the loop stops as soon as held-out macro-F1 stops improving for `early_stop_patience` rounds.

## Run

### Offline (stub paraphraser)

```bash
python examples/llm_augmentation/run_example.py
```

The script falls back to a trivial filler-insertion stub when `JUREBES_LLM_ENDPOINT` is unset, so the loop logic exercises end-to-end without network.

### llama.cpp server

```bash
# in another terminal:
llama-server -m /path/to/model.gguf --port 8080

export JUREBES_LLM_ENDPOINT=http://localhost:8080/v1
export JUREBES_LLM_MODEL=local
python examples/llm_augmentation/run_example.py
```

### vllm / ollama / OpenAI / any compatible host

```bash
export JUREBES_LLM_ENDPOINT=http://your-host:port/v1
export JUREBES_LLM_MODEL=gpt-4o-mini      # or whatever the host advertises
export JUREBES_LLM_API_KEY=sk-...          # optional
python examples/llm_augmentation/run_example.py
```

## What jurebes provides

The loop relies on three public APIs:

- [`IntentClassifier.predict_proba(utt)`](../../jurebes/core.py) — ranked list with calibrated confidences. The "low confidence" bucket lives here.
- [`IntentClassifier.fit()`](../../jurebes/core.py) — fast retrain on the growing sample bank. On the 3×4 seed set, a `logreg` baseline retrains in ~50 ms.
- [`BASELINES.build(name)`](../../jurebes/baselines.py) — picks a calibration-aware estimator for the loop's oracle role.

A `RunResult.confusion_matrix` from `benchmark.compare()` can drive a more targeted variant: ask the LLM specifically for utterances that disambiguate the two most-confused intents.

## Stopping criteria

The loop stops early when macro-F1 on the frozen eval set fails to improve by more than `1e-4` for `early_stop_patience` consecutive rounds (default `2`). Hard caps via `n_rounds` (default `5`).

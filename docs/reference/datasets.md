# `jurebes.datasets`

Every loader returns `tuple[list[str], list[str]]` — utterances and parallel labels.

## File-format loaders

### `load_csv(path, text="text", label="intent")`

Reads a CSV (with header). The `text` and `label` arguments name the columns. Encoding is UTF-8.

```python
from jurebes.datasets import load_csv
X, y = load_csv("data.csv")
X, y = load_csv("data.csv", text="utt", label="lbl")
```

### `load_jsonl(path, text="text", label="intent")`

Reads a JSON-Lines file (one JSON object per line). Same `text`/`label` semantics.

```python
from jurebes.datasets import load_jsonl
X, y = load_jsonl("data.jsonl")
```

### `load_ovos_intents(directory)`

Recurses a directory for OVOS-format files:

- `*.intent` — one sample per line, label is the file stem.
- `*.voc` — vocabulary file; treated as intent samples.
- `*.entity` — entity-value file; loaded as entity samples (not in the returned `(X, y)`).

Returns `(X, y)` containing intent samples only.

### `load_hf(name, split, text="text", label="intent")`

HuggingFace `datasets` loader. Requires the `hf` extra.

```python
from jurebes.datasets import load_hf
X, y = load_hf("benayas/snips", split="train")
```

Lazy-imports `datasets`; raises `ImportError("install jurebes[hf] ...")` if missing.

## Bracket-expansion helpers

For Padatious-style template grammars with `(alt|ernation)`, `[optional]` and `{slot}` placeholders.

### `expand_template(template) -> list[str]`

Expand alternations and optionals into the full set of realised strings.

```python
from jurebes.datasets import expand_template

expand_template("(hello|hi) [there] friend")
# ["hello friend", "hello there friend", "hi friend", "hi there friend"]
```

### `expand_slots(template, slots) -> list[str]`

Same as above, then substitute `{slot}` placeholders with values drawn from a dict (cartesian product across placeholders). Unknown slot names are left intact.

```python
from jurebes.datasets import expand_slots

expand_slots("play {song} by {artist}",
             {"song": ["africa", "hey jude"],
              "artist": ["toto", "the beatles"]})
# ["play africa by toto", "play africa by the beatles",
#  "play hey jude by toto", "play hey jude by the beatles"]
```

Pure stdlib (`re` + `itertools`); no `ovos-utils` runtime dep.

## Canonical benchmark loaders

Module `jurebes.datasets.canonical`. Each loader fetches from HuggingFace and caches under `~/.cache/huggingface/`. All require the `hf` extra.

| function | HF id | class count | notes |
| --- | --- | --- | --- |
| `load_snips()` | `benayas/snips` | 7 | SNIPS NLU benchmark |
| `load_clinc()` | `clinc_oos` config `plus` | 151 | includes `oos` label for OOD evaluation |
| `load_banking77()` | `banking77` | 77 | fine-grained banking intents |
| `load_hwu64()` | `DeepPavlov/hwu64` | 64 | home-assistant intents |
| `load_atis()` | `tuetschek/atis` | varies | Air Travel Information System |
| `load_massive()` | `AmazonScience/massive` | 60 × 51 locales | multilingual SLU |

The `CANONICAL` dict at `jurebes.datasets.canonical.CANONICAL` maps each short name to its loader:

```python
from jurebes.datasets.canonical import CANONICAL
X, y = CANONICAL["snips"]()
```

The CLI exposes the same via `--dataset @<name>` (e.g. `--dataset @banking77`).

## Custom loader contract

Any callable matching the signature `(...) -> tuple[list[str], list[str]]` plugs into the harness:

```python
def my_loader(path) -> tuple[list[str], list[str]]:
    ...

X, y = my_loader("data.tsv")
from jurebes.benchmark import compare
result = compare(["logreg"], X, y, k=5)
```

See [../advanced/custom-datasets.md](../advanced/custom-datasets.md) for the loader-authoring guide.

## Error handling

Conventional errors raised by loaders:

- `FileNotFoundError` — missing path.
- `ValueError` — malformed input or empty dataset.
- `ImportError("install jurebes[hf] ...")` — missing optional dependency.

## Linking with the research cookbook

See [../research.md](../research.md) for the high-level research workflow that wraps these loaders with the benchmark harness.

---
- Back to [reference index](index.md)

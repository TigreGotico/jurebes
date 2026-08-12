# Notebook examples

These cell-celled Python scripts (`# %%` markers) work in VSCode, PyCharm,
Jupyter (via jupytext), or as plain scripts. They require
`pip install jurebes[hf,bench-plot]` and download data from HuggingFace,
so they are not run in CI.

- `snips_quickstart.py` — train and evaluate a default `IntentClassifier`
  on SNIPS.
- `banking77_full_research_flow.py` — compare linear baselines on
  BANKING77, run Friedman+Nemenyi, tune the winner via random search,
  and evaluate on the test split.

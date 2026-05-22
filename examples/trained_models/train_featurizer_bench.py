"""Benchmark the new featurizers against the established winners.

Runs ``compare`` on one canonical dataset over a portfolio of the eight
new featurizer baselines plus three references (linear_svc_char, logreg,
nb_multinomial), with Friedman + Nemenyi ranking. Writes
``featurizer_bench_<dataset>.md``.

One dataset per invocation so memory is reclaimed between datasets::

    python train_featurizer_bench.py snips
    python train_featurizer_bench.py banking77
    python train_featurizer_bench.py clinc

Requires: pip install jurebes[hf,postag,stem,lemma]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
REPORTS.mkdir(exist_ok=True)

# new featurizer baselines under test + three references for calibration
PORTFOLIO = [
    "linear_svc_char",          # reference: current overall winner
    "logreg",                   # reference: plain word TF-IDF + logreg
    "nb_multinomial",           # reference: fast-mode floor
    "skipgram_logreg",
    "bm25_logreg",
    "bm25_linear_svc",
    "random_projection_logreg",
    "pos_sequence_logreg",
    "word_pos_logreg",
    "stemmed_logreg",
    "lemmatized_logreg",
]

_LOADERS = {
    "snips": "load_snips",
    "banking77": "load_banking77",
    "clinc": "load_clinc",
}


def run(dataset: str, cv: int = 3) -> str:
    from jurebes.benchmark import compare, to_markdown
    from jurebes.benchmark.stats import critical_difference, friedman_nemenyi
    import jurebes.datasets.canonical as canonical

    loader = getattr(canonical, _LOADERS[dataset])
    X, y = loader("train")

    md = [f"# featurizer benchmark — {dataset}\n"]
    md.append(f"- train size: **{len(X)}**")
    md.append(f"- intents: **{len(set(y))}**")
    md.append(f"- portfolio: **{len(PORTFOLIO)}** baselines, {cv}-fold CV\n")

    result = compare(PORTFOLIO, X, y, k=cv)
    md.append(to_markdown(result, sort_by="macro_f1"))
    md.append("")

    fold_scores = {
        name: scores["f1_macro"]
        for name, scores in result.fold_scores_by_baseline.items()
    }
    md.append("## Friedman + Nemenyi\n")
    try:
        fr = friedman_nemenyi(fold_scores)
        cd = critical_difference(fold_scores)
        md.append(f"- Friedman p-value: **{fr.pvalue:.4f}** "
                  f"(reject H0: **{fr.reject_null}**)\n")
        md.append("```")
        md.append(cd.to_ascii())
        md.append("```")
    except Exception as e:
        md.append(f"> stats failed: `{type(e).__name__}: {e}`")

    return "\n".join(md)


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "snips"
    if dataset not in _LOADERS:
        print(f"unknown dataset {dataset!r}; choose from {sorted(_LOADERS)}")
        sys.exit(1)
    report = run(dataset)
    path = REPORTS / f"featurizer_bench_{dataset}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nreport written to: {path}")


if __name__ == "__main__":
    main()

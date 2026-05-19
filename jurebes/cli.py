"""jurebes CLI — benchmark, train, predict, list-baselines, search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from jurebes import IntentClassifier
from jurebes.baselines import BASELINES
from jurebes.benchmark import compare, to_json, to_markdown
from jurebes.datasets import load_csv, load_jsonl


def _load_dataset(path: str):
    if isinstance(path, str) and path.startswith("@"):
        from jurebes.datasets.canonical import CANONICAL
        name = path[1:]
        if name not in CANONICAL:
            raise ValueError(f"unknown canonical dataset: {name!r}; known: {sorted(CANONICAL)}")
        return CANONICAL[name]()
    p = Path(path)
    if p.suffix == ".csv":
        return load_csv(p)
    if p.suffix == ".jsonl":
        return load_jsonl(p)
    raise ValueError(f"unsupported dataset suffix: {p.suffix}")


def _resolve_baselines(spec: str) -> List[str]:
    """Resolve a comma-separated CLI argument into baseline names.

    Each token may be a baseline name or an ``@group`` selector. Tokens
    are expanded via :py:meth:`_Registry.resolve` and de-duplicated.
    """
    names: List[str] = []
    seen = set()
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        for n in BASELINES.resolve(tok):
            if n not in seen:
                seen.add(n)
                names.append(n)
    return names


def _cmd_list(args):
    names = sorted(BASELINES.names())
    if args.group:
        members = BASELINES.groups().get(args.group, set())
        names = sorted(n for n in names if n in members)
    for name in names:
        print(name)
    return 0


def _cmd_benchmark(args):
    X, y = _load_dataset(args.dataset)
    names = _resolve_baselines(args.baselines)
    scoring = tuple(s.strip() for s in args.scoring.split(",") if s.strip())
    result = compare(names, X, y, k=args.cv, scoring=scoring)
    out = to_json(result) if args.format == "json" else to_markdown(
        result, sort_by=args.sort_by, precision=args.precision,
        with_significance=getattr(args, "with_significance", False),
    )
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
    else:
        print(out)
    if getattr(args, "save_run", None):
        Path(args.save_run).write_text(to_json(result), encoding="utf-8")
    return 0


def _cmd_stats(args):
    import json
    from jurebes.benchmark.stats import (
        critical_difference, friedman_nemenyi, mcnemar_test,
        paired_t_test_cv, wilcoxon_signed_rank_cv,
    )
    metric = args.metric

    def _load(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    if args.pair:
        a, b = args.pair
        da = _load(a)
        db = _load(b)
        fa = da.get("fold_scores_by_baseline", {})
        fb = db.get("fold_scores_by_baseline", {})
        # take first baseline from each run
        name_a = next(iter(fa))
        name_b = next(iter(fb))
        sa = fa[name_a].get(metric, [])
        sb = fb[name_b].get(metric, [])
        rt = paired_t_test_cv(sa, sb)
        rw = wilcoxon_signed_rank_cv(sa, sb)
        print(f"paired_t: statistic={rt.statistic:.4f} p={rt.pvalue:.4f} reject_null={rt.reject_null}")
        print(f"wilcoxon: statistic={rw.statistic:.4f} p={rw.pvalue:.4f} reject_null={rw.reject_null}")
        return 0

    if args.runs:
        # gather one series per baseline across runs (treat each run as a "dataset")
        combined: dict = {}
        for path in args.runs:
            d = _load(path)
            for bname, scores in d.get("fold_scores_by_baseline", {}).items():
                series = scores.get(metric, [])
                combined.setdefault(bname, []).extend(series)
        if len(combined) < 2:
            print("need at least 2 baselines across runs", file=sys.stderr)
            return 1
        if len(combined) >= 3:
            fr = friedman_nemenyi(combined, alpha=args.alpha)
            cd = critical_difference(combined, alpha=args.alpha)
            print(f"Friedman statistic={fr.statistic:.4f} p={fr.pvalue:.4f} reject_null={fr.reject_null}")
            print(cd.to_ascii())
        else:
            names = list(combined.keys())
            rt = paired_t_test_cv(combined[names[0]], combined[names[1]])
            rw = wilcoxon_signed_rank_cv(combined[names[0]], combined[names[1]])
            print(f"paired_t: statistic={rt.statistic:.4f} p={rt.pvalue:.4f} reject_null={rt.reject_null}")
            print(f"wilcoxon: statistic={rw.statistic:.4f} p={rw.pvalue:.4f} reject_null={rw.reject_null}")
        return 0

    print("specify --runs or --pair", file=sys.stderr)
    return 2


def _cmd_train(args):
    X, y = _load_dataset(args.dataset)
    tagger = None
    if args.tagger:
        from jurebes.slots import SklearnIOBTagger
        tagger = SklearnIOBTagger()
    clf = IntentClassifier(BASELINES.build(args.baseline), tagger=tagger)
    for label in sorted(set(y)):
        samples = [x for x, yy in zip(X, y) if yy == label]
        clf.add_intent(label, samples)
    clf.fit()
    clf.save(args.out)
    print(f"saved model to {args.out}")
    return 0


def _cmd_predict(args):
    clf = IntentClassifier.load(args.model)
    result = clf.predict(args.text)
    print(f"{result.intent}\t{result.confidence:.4f}\t{result.entities}")
    return 0


def _cmd_search(args):
    from jurebes.search import search, spaces
    X, y = _load_dataset(args.dataset)
    if args.space:
        # caller-provided python literal
        import ast
        space = ast.literal_eval(args.space)
    else:
        space = spaces.for_baseline(args.baseline)
    r = search(
        args.baseline, space, X, y,
        backend=args.backend, cv=args.cv, n_iter=args.n_iter,
        scoring=args.scoring.split(",")[0].strip(),
    )
    print(f"best_score={r.best_score:.4f}")
    print(f"best_params={r.best_params}")
    print(f"wall_time={r.wall_time_seconds:.2f}s n_eval={r.n_evaluations}")
    if args.out:
        r.best_estimator.save(args.out)
        print(f"saved best model to {args.out}")
    return 0


_BANNER = "🐾 jurebes — Just-sklearn Utility for Reproducible Evaluation of Baselines, Estimators and Solvers"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="jurebes", description=_BANNER)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-baselines")
    p_list.add_argument("--group", help="filter to a single group name")
    p_list.set_defaults(func=_cmd_list)

    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("--dataset", required=True)
    p_bench.add_argument("--baselines", required=True,
                         help="comma-separated names; @<group> selects all of a group, @all selects every baseline")
    p_bench.add_argument("--cv", type=int, default=5)
    p_bench.add_argument("--scoring", default="accuracy,f1_macro",
                         help="comma-separated metric names")
    p_bench.add_argument("--sort-by", default=None, dest="sort_by")
    p_bench.add_argument("--precision", type=int, default=4)
    p_bench.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p_bench.add_argument("--out")
    p_bench.add_argument("--with-significance", action="store_true", dest="with_significance",
                         help="append Friedman+Nemenyi or paired-t/Wilcoxon block")
    p_bench.add_argument("--save-run", dest="save_run",
                         help="path to dump ComparisonResult JSON for downstream `stats`")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset", required=True)
    p_train.add_argument("--baseline", default="linear_svc")
    p_train.add_argument("--tagger", action="store_true",
                         help="attach a SklearnIOBTagger to the trained model")
    p_train.add_argument("--out", required=True)
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--text", required=True)
    p_pred.set_defaults(func=_cmd_predict)

    p_search = sub.add_parser("search")
    p_search.add_argument("--dataset", required=True)
    p_search.add_argument("--baseline", required=True)
    p_search.add_argument("--backend", default="random",
                         choices=["grid", "halving_grid", "random", "halving_random", "bayes", "genetic"])
    p_search.add_argument("--cv", type=int, default=5)
    p_search.add_argument("--n-iter", type=int, default=50, dest="n_iter")
    p_search.add_argument("--scoring", default="f1_macro")
    p_search.add_argument("--space", help="python-literal dict overriding spaces.for_baseline()")
    p_search.add_argument("--out")
    p_search.set_defaults(func=_cmd_search)

    p_stats = sub.add_parser("stats", help="statistical comparison across saved benchmark runs")
    p_stats.add_argument("--runs", nargs="+", help="paths to ComparisonResult JSON files")
    p_stats.add_argument("--pair", nargs=2, metavar=("A", "B"),
                         help="paired statistical test between two saved runs")
    p_stats.add_argument("--metric", default="f1_macro")
    p_stats.add_argument("--alpha", type=float, default=0.05)
    p_stats.set_defaults(func=_cmd_stats)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

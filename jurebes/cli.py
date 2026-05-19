"""jurebes CLI — benchmark, train, predict, list-baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jurebes import IntentClassifier
from jurebes.baselines import BASELINES
from jurebes.benchmark import compare, to_json, to_markdown
from jurebes.datasets import load_csv, load_jsonl


def _load_dataset(path: str):
    p = Path(path)
    if p.suffix == ".csv":
        return load_csv(p)
    if p.suffix == ".jsonl":
        return load_jsonl(p)
    raise ValueError(f"unsupported dataset suffix: {p.suffix}")


def _cmd_list(_args):
    for name in sorted(BASELINES.names()):
        print(name)
    return 0


def _cmd_benchmark(args):
    X, y = _load_dataset(args.dataset)
    names = [b.strip() for b in args.baselines.split(",") if b.strip()]
    result = compare(names, X, y, k=args.cv)
    out = to_json(result) if args.format == "json" else to_markdown(result)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
    else:
        print(out)
    return 0


def _cmd_train(args):
    X, y = _load_dataset(args.dataset)
    clf = IntentClassifier(BASELINES.build(args.baseline))
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="jurebes")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-baselines")
    p_list.set_defaults(func=_cmd_list)

    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("--dataset", required=True)
    p_bench.add_argument("--baselines", required=True, help="comma-separated baseline names")
    p_bench.add_argument("--cv", type=int, default=5)
    p_bench.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p_bench.add_argument("--out")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset", required=True)
    p_train.add_argument("--baseline", default="linear_svc")
    p_train.add_argument("--out", required=True)
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--text", required=True)
    p_pred.set_defaults(func=_cmd_predict)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

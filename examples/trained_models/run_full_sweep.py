"""Run every benchmark trainer from scratch.

Sequence:
1. SNIPS                     (7-intent text classification)
2. BANKING77                 (77-intent text classification)
3. CLINC-150                 (150-intent text classification, OOD dropped)
4. intents-for-eval × 12 langs (intent + slot extraction, OVOS bench)

Each step writes its own Markdown report under reports/. After the
sweep finishes, _build_report.py regenerates figures and structured
JSON for REPORT.md.

Requires: pip install jurebes[hf,slots-crf]

This will take a while. The label-guided + autoencoder baselines train
end-to-end MLPs on TF-IDF-densified inputs; expect ~5–15 minutes per
canonical dataset for the slow ones, and ~5–10 minutes per IFE language.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _run(script: str, *args: str) -> int:
    t0 = time.perf_counter()
    print(f"\n────── {script} {' '.join(args)} ──────")
    rc = subprocess.run(
        [sys.executable, str(HERE / script), *args],
        cwd=HERE,
    ).returncode
    dt = time.perf_counter() - t0
    print(f"   ↳ {script}: rc={rc}  wall={dt/60:.1f}m")
    return rc


def main() -> int:
    steps = [
        ("train_snips.py", ()),
        ("train_banking77.py", ()),
        ("train_clinc.py", ()),
        ("train_intents_for_eval_all_langs.py", ()),
        ("_build_report.py", ()),
    ]
    total_t0 = time.perf_counter()
    for script, args in steps:
        rc = _run(script, *args)
        if rc != 0:
            print(f"WARNING: {script} exited rc={rc}; continuing", file=sys.stderr)
    print(f"\nfull sweep complete: {(time.perf_counter() - total_t0) / 60:.1f}m total")
    return 0


if __name__ == "__main__":
    sys.exit(main())

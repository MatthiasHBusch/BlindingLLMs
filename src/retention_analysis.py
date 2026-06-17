"""
2-sig -> 3-sig RETENTION analysis (JCIM revision).

Replaces the permutation "chance" baseline with a single, self-consistent,
parameter-free statistic. A 3-significant-digit match implies a 2-sig match, so
among the predictions that match at 2 sig figs we ask what fraction also match at
3 sig figs (the "retention" R = m3 / m2). One extra significant digit is ~one
uniform decimal of freedom, so:
    * coincidental matches  -> R ~ 10%  (the extra digit rarely also matches)
    * genuine value recall  -> R ~ 100% (the model knows the value, so the
      stricter test is still passed)
We test the observed 3-sig count against the null m3 ~ Binomial(m2, 0.10) and
report whether R is significantly ABOVE 10% (evidence of recall) or AT/BELOW it
(consistent with chance). Applied identically to the three benchmarks and to the
positive-control sets, this is the consistent comparison requested in review.

No API calls. Reads local result JSONs + CSVs.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
PC = os.path.join(os.path.dirname(__file__), "positive_control")

PAPER_MODELS = {
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5", "gpt-5-mini", "gpt-5-nano",
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
}

BENCH = {
    "Delaney": dict(j="LLM_Results_delaney_zeroshot.json", csv="delaney-processed.csv",
                    val="measured log solubility in mols per litre", key="Compound ID"),
    "Lipophilicity": dict(j="LLM_Results_lipophilicity_zeroshot.json", csv="Lipophilicity.csv",
                          val="exp", key="smiles"),
    "QM7": dict(j="LLM_Results_qm7_zeroshot.json", csv="qm7.csv", val="u0_atom", key="smiles"),
}
PCSETS = {
    "AtomicWeights": dict(j="PositiveControl_AtomicWeights.json", csv="known_atomic_weights.csv",
                          val="atomic_weight", key="symbol"),
    "BoilingPoints": dict(j="PositiveControl_BoilingPoints.json", csv="known_boiling_points.csv",
                          val="bp_celsius", key="smiles"),
}


def sig_round(x, n):
    return "0" if x == 0 else f"{x:.{n}g}"


def sig_figs(v):
    if v == 0:
        return 1
    s = f"{v:.10g}".lstrip("-").replace(".", "").lstrip("0")
    return max(1, len(s.rstrip("0"))) if s else 1


def leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                yield "/".join(path), k, v
            else:
                yield from leaves(v, path + (k,))


def collect(jpath, csvpath, val, key, restrict=None):
    """Return {model: (m2, m3, n)} over ground-truth values with >=3 sig figs."""
    df = pd.read_csv(csvpath).dropna(subset=[val])
    lut = df.set_index(key)[val].to_dict()
    res = json.load(open(jpath))
    acc = {}
    for model, k, preds in leaves(res):
        short = model.split("/")[-1]
        if restrict and short not in restrict:
            continue
        if k not in lut:
            continue
        tv = float(lut[k])
        if sig_figs(tv) < 3:
            continue
        t2, t3 = sig_round(tv, 2), sig_round(tv, 3)
        a = acc.setdefault(short, [0, 0, 0])  # m2, m3, n
        for p in preds:
            try:
                pv = float(p)
            except (TypeError, ValueError):
                continue
            # Exclude predictions reported at < 3 significant figures: such a value
            # can match at 2 sig figs but never at 3, which would inflate m2 and
            # depress the retention ratio below the 10% coincidence floor. We
            # require >=3 sig figs on both the label (above) and the prediction so
            # that a 3-sig comparison is actually possible.
            if sig_figs(pv) < 3:
                continue
            a[2] += 1
            if sig_round(pv, 2) == t2:
                a[0] += 1
                if sig_round(pv, 3) == t3:
                    a[1] += 1
    return acc


def verdict(m2, m3):
    if m2 == 0:
        return "n/a", float("nan"), float("nan")
    R = 100 * m3 / m2
    p = binomtest(m3, m2, 0.10, alternative="two-sided").pvalue
    if p < 0.05 and R > 10:
        v = "RECALL (R>>10%)"
    elif p < 0.05 and R < 10:
        v = "below chance"
    else:
        v = "~chance (R~10%)"
    return v, R, p


def report_group(title, sets, restrict, pooled=True):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    for name, cfg in sets.items():
        acc = collect(os.path.join(cfg["dir"], cfg["j"]), os.path.join(cfg["dir"], cfg["csv"]),
                      cfg["val"], cfg["key"], restrict)
        print(f"\n--- {name} ---")
        print(f"{'model':22s} {'m2':>5} {'m3':>5} {'2sig%':>7} {'3sig%':>7} {'ret%':>7} {'p':>9}  verdict")
        tot2 = tot3 = totn = 0
        for model in sorted(acc):
            m2, m3, n = acc[model]
            tot2 += m2; tot3 += m3; totn += n
            v, R, p = verdict(m2, m3)
            r2 = 100*m2/n if n else 0
            r3 = 100*m3/n if n else 0
            print(f"{model:22s} {m2:5d} {m3:5d} {r2:7.2f} {r3:7.2f} {R:7.1f} {p:9.1e}  {v}")
        if pooled:
            v, R, p = verdict(tot2, tot3)
            print(f"{'POOLED (all models)':22s} {tot2:5d} {tot3:5d} "
                  f"{100*tot2/totn:7.2f} {100*tot3/totn:7.2f} {R:7.1f} {p:9.1e}  {v}")


ORDER = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
         "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
         "gpt-5", "gpt-5-mini", "gpt-5-nano"]


def report_bymodel_pooled(sets, restrict):
    """Per-model match rates pooled across the (benchmark) datasets, to check for
    a model-size (full->mini->nano) trend in matching."""
    agg = {}
    for cfg in sets.values():
        a = collect(os.path.join(cfg["dir"], cfg["j"]), cfg["csv"], cfg["val"], cfg["key"], restrict)
        for m, (m2, m3, n) in a.items():
            g = agg.setdefault(m, [0, 0, 0])
            g[0] += m2; g[1] += m3; g[2] += n
    print(f"\n{'='*70}\nPER-MODEL, pooled across benchmarks (size-trend check)\n{'='*70}")
    print(f"{'model':22s} {'2sig%':>7} {'3sig%':>7} {'ratio%':>7} {'m2':>5} {'m3':>5}")
    for m in ORDER:
        if m not in agg:
            continue
        m2, m3, n = agg[m]
        r2 = 100*m2/n if n else 0
        r3 = 100*m3/n if n else 0
        R = 100*m3/m2 if m2 else float("nan")
        print(f"{m:22s} {r2:7.2f} {r3:7.2f} {R:7.1f} {m2:5d} {m3:5d}")
    return agg


def main():
    bench = {k: dict(j=v["j"], csv=os.path.join(DATA, v["csv"]), val=v["val"],
                     key=v["key"], dir=RESULTS) for k, v in BENCH.items()}
    pc = {k: dict(j=v["j"], csv=os.path.join(PC, v["csv"]), val=v["val"],
                  key=v["key"], dir=PC) for k, v in PCSETS.items()}

    report_group("BENCHMARKS (0-shot) -- expect retention ~10% (chance)", bench, PAPER_MODELS, pooled=True)
    report_group("POSITIVE CONTROL -- expect retention >>10% (recall)", pc, PAPER_MODELS, pooled=True)
    report_bymodel_pooled(bench, PAPER_MODELS)
    print("\nNote: retention R = 3sig-matches / 2sig-matches; p = two-sided binomial test vs R0=10%.")


if __name__ == "__main__":
    main()

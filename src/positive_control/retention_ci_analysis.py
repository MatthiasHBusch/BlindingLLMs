"""
Retention-based version of the per-model positive-control table (JCIM round 2).

Motivation: the permutation "chance" baseline for the exact-match rate m3 controls
for the models' marginal answer distribution but NOT for their accuracy, so at the
unblinded levels -- where the models predict boiling points to within a fraction of
a percent -- a 3-significant-digit hit can arise from precision rather than from
retrieval. The retention ratio R21 = m2/m1 does not have that problem: it conditions
on the first digit already being right and asks only whether the next digit follows.

The floor for R21 is NOT universally 10%. It is the collision probability C2 of the
second significant digit of that dataset's values (cf. Figure S2 for QM7/Delaney).
This script computes C2 for the positive controls and reports each R21 with a Wilson
confidence interval, so the table shows directly where retention is significantly
above the dataset's own floor.

Usage:
    python retention_ci_analysis.py
"""
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

SWEEPS = {
    "boiling": dict(results="PositiveControl_Blinding_Direct.json",
                    data="known_boiling_points_blinded.csv", value="bp_celsius",
                    label="Boiling points"),
    "atomic": dict(results="PositiveControl_Blinding_AtomicWeights_Direct.json",
                   data="known_atomic_weights_blinded.csv", value="atomic_weight",
                   label="Atomic weights"),
}

LEVELS = [
    ("io_specific_clear", "1 Specific, orig.", False),
    ("io_specific_blind", "2 Specific, transf.", True),
    ("io_molproperty_clear", "3 Generic, orig.", False),
    ("io_molproperty_blind", "4 Generic, transf.", True),
    ("io_sampleproperty_clear", "5 Agnostic, orig.", False),
    ("io_sampleproperty_blind", "6 Agnostic, transf.", True),
]

SHORT = {"google/gemini-2.5-pro": "Gemini 2.5 Pro",
         "google/gemini-2.5-flash": "Gemini 2.5 Flash",
         "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
         "openai/gpt-5": "GPT-5", "openai/gpt-5-mini": "GPT-5 mini",
         "openai/gpt-5-nano": "GPT-5 nano", "openai/gpt-4.1": "GPT-4.1",
         "openai/gpt-4.1-mini": "GPT-4.1 mini", "openai/gpt-4.1-nano": "GPT-4.1 nano"}
ORDER = list(SHORT.values())


def sig_round(x, n):
    return "0" if x == 0 else f"{x:.{n}g}"


def sig_figs(val):
    if val == 0:
        return 1
    s = f"{val:.10g}".lstrip("-").replace(".", "").lstrip("0")
    return max(1, len(s.rstrip("0"))) if s else 1


def nth_digit(x, n=2):
    """n-th significant digit of |x|, or None if it has fewer than n digits."""
    if x == 0 or not np.isfinite(x):
        return None
    s = f"{abs(x):.10g}".replace(".", "").lstrip("0")
    s = s.rstrip("0") if len(s.rstrip("0")) >= n else s
    return int(s[n - 1]) if len(s) >= n else None


def collision_probability(values, n=2):
    """C_n = sum_d p(d)^2 over the n-th significant digit -- the chance that two
    independent draws from this distribution agree on that digit."""
    digits = [nth_digit(v, n) for v in values]
    digits = [d for d in digits if d is not None]
    p = np.bincount(digits, minlength=10) / len(digits)
    return float((p ** 2).sum()), len(digits), p


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (robust at small k/n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def retention_counts(trues, preds, k=2):
    """(m_{k-1}, m_k) over pairs with >=k sig figs on both sides -- Table S1 convention."""
    lo = hi = 0
    for tv, pv in zip(trues, preds):
        if sig_figs(tv) < k or sig_figs(pv) < k:
            continue
        if sig_round(pv, k - 1) == sig_round(tv, k - 1):
            lo += 1
            if sig_round(pv, k) == sig_round(tv, k):
                hi += 1
    return lo, hi


def main():
    print("=" * 78)
    print("Second-digit collision probability C2 (the true floor for R21)")
    print("=" * 78)
    floors = {}
    for key, cfg in SWEEPS.items():
        df = pd.read_csv(os.path.join(HERE, cfg["data"]))
        for col, tag in [(cfg["value"], "untransformed"),
                         ("transformed_solubility", "transformed")]:
            c2, n, p = collision_probability(df[col].to_numpy())
            floors[(key, tag)] = c2
            print(f"  {cfg['label']:16s} {tag:14s} C2 = {100*c2:5.1f}%  "
                  f"(n={n:3d})  digits: {' '.join(f'{100*x:.0f}' for x in p)}")
    print("\n  Uniform expectation is 10.0%. Compare Figure S2: QM7 13.1%, Delaney 10.2%.")

    rows = []
    for key, cfg in SWEEPS.items():
        df = pd.read_csv(os.path.join(HERE, cfg["data"]))
        truth = df.set_index("name")[cfg["value"]].to_dict()
        trans = df.set_index("name")["transformed_solubility"].to_dict()
        res = json.load(open(os.path.join(HERE, cfg["results"])))["names_only"]
        for approach, label, is_t in LEVELS:
            if approach not in res:
                continue
            lut = trans if is_t else truth
            for model, mv in res[approach].items():
                node = mv["0"][list(mv["0"].keys())[0]]
                T, P = [], []
                for name, vals in node.items():
                    t = lut.get(name)
                    if t is None:
                        continue
                    for v in (vals if isinstance(vals, list) else [vals]):
                        try:
                            v = float(v)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(v):
                            T.append(t)
                            P.append(v)
                m1, m2 = retention_counts(T, P)
                lo, hi = wilson(m2, m1)
                rows.append(dict(sweep=cfg["label"], level=label, transformed=is_t,
                                 model=SHORT.get(model, model), m1=m1, m2=m2,
                                 R21=100 * m2 / m1 if m1 else float("nan"),
                                 ci_lo=lo, ci_hi=hi,
                                 floor=100 * floors[(key, "transformed" if is_t
                                                     else "untransformed")]))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "retention_ci_permodel.csv"), index=False)

    for sweep in out.sweep.unique():
        print()
        print("=" * 78)
        print(f"{sweep}: R21 = m2/m1 with 95% Wilson CI; * = CI lower bound above the")
        print("           dataset's own second-digit floor C2")
        print("=" * 78)
        sub = out[out.sweep == sweep]
        for label in [l for _, l, _ in LEVELS if l in set(sub.level)]:
            s = sub[sub.level == label]
            fl = s.floor.iloc[0]
            m1, m2 = s.m1.sum(), s.m2.sum()
            plo, phi = wilson(m2, m1)
            sig = "*" if plo > fl else " "
            print(f"\n  {label}   (floor C2 = {fl:.1f}%)")
            print(f"    {'model':22s} {'m2/m1':>10s} {'R21':>6s}  95% CI")
            for m in ORDER:
                r = s[s.model == m]
                if not len(r):
                    continue
                r = r.iloc[0]
                mark = "*" if r.ci_lo > fl else " "
                print(f"    {m:22s} {f'{r.m2}/{r.m1}':>10s} {r.R21:5.1f}{mark} "
                      f"[{r.ci_lo:4.1f}, {r.ci_hi:5.1f}]")
            print(f"    {'POOLED':22s} {f'{m2}/{m1}':>10s} {100*m2/m1:5.1f}{sig} "
                  f"[{plo:4.1f}, {phi:5.1f}]")

    print("\nWrote retention_ci_permodel.csv")


if __name__ == "__main__":
    main()

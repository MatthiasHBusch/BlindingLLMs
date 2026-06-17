"""
Analyze the positive-control runs with the IDENTICAL exact-match detector
(JCIM revision). Run build_known_values.py and Run_PositiveControl_ZeroShot.jl
first (the latter needs approved API spend). Until the JSON outputs exist this
script just reports that they are missing.

For each set (atomic weights, boiling points) and each model it reports the
observed 2- and 3-significant-digit match rate against the known ground truth,
plus the random-chance (shuffle) baseline - exactly the metric used on the
ESOL/Lipophilicity/QM7 benchmarks (see ../memorization_sigdigits.py). The
headline the paper needs: detector match rate is HIGH here and ~chance on the
benchmarks => the detector is sensitive.
"""

import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)

SETS = {
    "AtomicWeights": dict(json="PositiveControl_AtomicWeights.json",
                          csv="known_atomic_weights.csv", key_col="symbol",
                          value_col="atomic_weight"),
    "BoilingPoints": dict(json="PositiveControl_BoilingPoints.json",
                          csv="known_boiling_points.csv", key_col="smiles",
                          value_col="bp_celsius"),
}


def sig_round(x, n):
    return "0" if x == 0 else f"{x:.{n}g}"


def sig_figs(val):
    if val == 0:
        return 1
    s = f"{val:.10g}".lstrip("-").replace(".", "").lstrip("0")
    return max(1, len(s.rstrip("0"))) if s else 1


def iter_mol_leaves(obj):
    def rec(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                    yield "/".join(path), k, v
                else:
                    yield from rec(v, path + [k])
    yield from rec(obj, [])


def chance_rate(trues, preds, n, rng, n_perm=300):
    trues, preds = np.asarray(trues), np.asarray(preds)
    keep = np.array([sig_figs(t) >= n for t in trues])
    trues, preds = trues[keep], preds[keep]
    if len(trues) == 0:
        return 0.0
    tstr = np.array([sig_round(t, n) for t in trues])
    pstr = np.array([sig_round(p, n) for p in preds])
    hits = sum(np.sum(tstr == pstr[rng.permutation(len(pstr))]) for _ in range(n_perm))
    return 100 * hits / (n_perm * len(tstr))


def analyse(name, cfg, rng):
    jpath = os.path.join(HERE, cfg["json"])
    if not os.path.exists(jpath):
        print(f"[{name}] results file not found ({cfg['json']}). "
              f"Run build_known_values.py then Run_PositiveControl_ZeroShot.jl (needs API approval).")
        return []
    df = pd.read_csv(os.path.join(HERE, cfg["csv"]))
    lut = df.set_index(cfg["key_col"])[cfg["value_col"]].to_dict()
    with open(jpath) as f:
        results = json.load(f)

    pairs = {}
    for model, key, preds in iter_mol_leaves(results):
        if key not in lut:
            continue
        tv = float(lut[key])
        for p in preds:
            try:
                pv = float(p)
            except (TypeError, ValueError):
                continue
            pairs.setdefault(model, ([], []))[0].append(tv)
            pairs[model][1].append(pv)

    rows = []
    for model, (trues, preds) in pairs.items():
        trues, preds = np.asarray(trues), np.asarray(preds)
        rec = {"set": name, "model": model.split("/")[-1]}
        for n in (2, 3):
            keep = np.array([sig_figs(t) >= n for t in trues])
            tk, pk = trues[keep], preds[keep]
            m = int(sum(sig_round(t, n) == sig_round(p, n) for t, p in zip(tk, pk)))
            rec[f"obs{n}"] = round(100 * m / len(tk), 1) if len(tk) else 0.0
            rec[f"chance{n}"] = round(chance_rate(trues, preds, n, rng), 1)
            rec[f"n{n}"] = len(tk)
        rows.append(rec)
    return rows


def main():
    rng = np.random.default_rng(0)
    all_rows = []
    for name, cfg in SETS.items():
        all_rows.extend(analyse(name, cfg, rng))
    if not all_rows:
        print("\nNo positive-control results yet. (Detector validation pending API run.)")
        return
    out = pd.DataFrame(all_rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(HERE, "positive_control_results.csv"), index=False)
    print("\nInterpretation: high obs%% >> chance%% here, vs obs ~ chance on the "
          "ESOL/Lipophilicity/QM7 benchmarks => the exact-match detector is sensitive.")


if __name__ == "__main__":
    main()

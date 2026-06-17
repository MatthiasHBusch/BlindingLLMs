"""
Bootstrap 95% confidence intervals for the reported Pearson correlations
(JCIM revision, answers R2 important-(2): "two runs, no error bars, 150 samples").

We do NOT need new LLM runs. Each (config) leaf in the results JSON holds the
per-molecule predictions for the 150-molecule test set (each molecule has the
pooled repeats). We aggregate exactly as the paper's plotting code does
(per-molecule mean of valid predictions, dropping None/NaN and values >= 100),
then bootstrap over the 150 test MOLECULES (resample molecules with replacement,
recompute Pearson r) to obtain a 95% percentile CI.

This quantifies the test-set sampling uncertainty behind every correlation and
lets us soften model-ranking language to trends where CIs overlap.

No API calls. Reads only local CSVs + results JSONs.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")
N_BOOT = 5000
SEED = 42

DATASETS = {
    "delaney": {
        "results": "LLM_Results_delaney.json",
        "csv": "delaney-processed.csv",
        "value_col": "measured log solubility in mols per litre",
        "key_col": "Compound ID",   # leaf keys are compound names
    },
    "lipophilicity": {
        "results": "LLM_Results_lipophilicity.json",
        "csv": "Lipophilicity.csv",
        "value_col": "exp",
        "key_col": "smiles",        # leaf keys are SMILES
    },
    "qm7": {
        "results": "LLM_Results_qm7.json",
        "csv": "qm7.csv",
        "value_col": "u0_atom",
        "key_col": "smiles",
    },
}


def clean_preds(lst):
    out = []
    for p in lst:
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        if np.isnan(p) or p >= 100:   # matches plotResultsICMLPaper.py filter
            continue
        out.append(p)
    return out


def iter_leaves(obj, path=()):
    """Yield (path, leaf_dict) for every dict whose values are all lists."""
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and all(isinstance(v, list) for v in vals):
            yield path, obj
        else:
            for k, v in obj.items():
                yield from iter_leaves(v, path + (k,))


def eval_leaf(leaf, lut):
    names, means, trues = [], [], []
    for name, preds in leaf.items():
        if name not in lut:
            continue
        cp = clean_preds(preds)
        m = float(np.mean(cp)) if cp else 0.0
        names.append(name)
        means.append(m)
        trues.append(float(lut[name]))
    return np.array(means), np.array(trues)


def bootstrap_ci(pred, true, rng, n_boot=N_BOOT):
    n = len(pred)
    if n < 5:
        return (np.nan, np.nan)
    rs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(pred[idx]) == 0 or np.std(true[idx]) == 0:
            rs[b] = np.nan
            continue
        rs[b] = pearsonr(pred[idx], true[idx])[0]
    rs = rs[~np.isnan(rs)]
    return (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))


def run_dataset(name, cfg, rng):
    df = pd.read_csv(os.path.join(DATA_DIR, cfg["csv"])).dropna(subset=[cfg["value_col"]])
    lut = df.set_index(cfg["key_col"])[cfg["value_col"]].to_dict()
    with open(os.path.join(RESULTS_DIR, cfg["results"])) as f:
        results = json.load(f)

    rows = []
    for path, leaf in iter_leaves(results):
        pred, true = eval_leaf(leaf, lut)
        if len(pred) < 5 or np.std(pred) == 0:
            continue
        r = pearsonr(pred, true)[0]
        lo, hi = bootstrap_ci(pred, true, rng)
        rows.append({
            "dataset": name,
            "config": "/".join(path),
            "n": len(pred),
            "r": round(r * 100, 1),
            "ci_lo": round(lo * 100, 1),
            "ci_hi": round(hi * 100, 1),
            "ci_width": round((hi - lo) * 100, 1),
        })
    return rows


def main():
    rng = np.random.default_rng(SEED)
    all_rows = []
    for name, cfg in DATASETS.items():
        all_rows.extend(run_dataset(name, cfg, rng))
    out = pd.DataFrame(all_rows)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "bootstrap_correlation_ci.csv")
    out.to_csv(out_path, index=False)

    # Summary: typical CI half-width per dataset (the key headline number)
    print(f"{len(out)} configs evaluated across 3 datasets.\n")
    print("Median 95% CI WIDTH on correlation (percentage points), by dataset:")
    print((out.groupby("dataset")["ci_width"].median()).to_string())
    print("\nMedian CI width overall: %.1f pp" % out["ci_width"].median())
    print("\nExample configs (full-information level-1, with_preanalysis, 1000-shot):")
    ex = out[out["config"].str.contains("with_preanalysis") & out["config"].str.contains("940")]
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(ex.sort_values(["dataset", "config"]).to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

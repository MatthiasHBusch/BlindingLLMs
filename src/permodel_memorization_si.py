"""
Per-model memorization table data for the reshaped SI (JCIM revision).

For every (dataset, model) over all six datasets (three legacy benchmarks, the
post-cutoff antiviral negative control, and the two positive-control sets) emit:

  * match3 / total3 / three_sig_pct  -- exact-match convention of
      memorization_sigdigits.py: total3 = predictions over ground-truth values
      carrying >=3 sig figs (a <3-sig prediction simply fails to match, it is NOT
      removed from the denominator); match3 = first-three-sig-fig matches.
  * R21 = m2/m1, R32 = m3/m2 -- retention convention of zeroshot_extended_si.py:
      for level-k retention keep only pairs with >=k sig figs on BOTH label and
      prediction.
  * pearson_r -- 0-shot correlation on the per-molecule mean prediction.

Reuses the exact helper logic of zeroshot_extended_si.py. No API calls.
Writes permodel_memorization_si.csv.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_RESULTS = os.path.join(ROOT, "results")
NEW_RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
PC = os.path.join(os.path.dirname(__file__), "positive_control")

PAPER_MODELS = {
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5", "gpt-5-mini", "gpt-5-nano",
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
}
ORDER = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
         "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
         "gpt-5", "gpt-5-mini", "gpt-5-nano"]

# name -> (dir, json, csv, value column, molecule-key column, kind)
SETS = {
    "Delaney":           (REPO_RESULTS, "LLM_Results_delaney_zeroshot.json",
                          os.path.join(DATA, "delaney-processed.csv"),
                          "measured log solubility in mols per litre", "Compound ID", "benchmark"),
    "Lipophilicity":     (REPO_RESULTS, "LLM_Results_lipophilicity_zeroshot.json",
                          os.path.join(DATA, "Lipophilicity.csv"), "exp", "smiles", "benchmark"),
    "QM7":               (REPO_RESULTS, "LLM_Results_qm7_zeroshot.json",
                          os.path.join(DATA, "qm7.csv"), "u0_atom", "smiles", "benchmark"),
    "Antiviral potency": (NEW_RESULTS, "LLM_Results_potency_zeroshot.json",
                          os.path.join(DATA, "antiviral_potency.csv"), "pic50", "smiles", "control"),
    "Atomic weights":    (PC, "PositiveControl_AtomicWeights.json",
                          os.path.join(PC, "known_atomic_weights.csv"), "atomic_weight", "symbol", "control"),
    "Boiling points":    (PC, "PositiveControl_BoilingPoints.json",
                          os.path.join(PC, "known_boiling_points.csv"), "bp_celsius", "smiles", "control"),
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


def load_keyed(dir_, jf, csv, val, key):
    """{model: (trues[], preds[], mol_keys[])} over all molecule x rep predictions."""
    df = pd.read_csv(csv).dropna(subset=[val])
    lut = df.set_index(key)[val].to_dict()
    res = json.load(open(os.path.join(dir_, jf)))
    out = {}
    for model, mol, preds in leaves(res):
        short = model.split("/")[-1]
        if short not in PAPER_MODELS or mol not in lut:
            continue
        tv = float(lut[mol])
        t, p, kk = out.setdefault(short, ([], [], []))
        for pr in preds:
            try:
                pv = float(pr)
            except (TypeError, ValueError):
                continue
            t.append(tv)
            p.append(pv)
            kk.append(mol)
    return out


def retention(trues, preds, k):
    """R = m_k/m_{k-1} over pairs with >=k sig figs on both sides."""
    m_low = m_high = 0
    for tv, pv in zip(trues, preds):
        if sig_figs(tv) < k or sig_figs(pv) < k:
            continue
        if sig_round(pv, k - 1) == sig_round(tv, k - 1):
            m_low += 1
            if sig_round(pv, k) == sig_round(tv, k):
                m_high += 1
    return m_low, m_high


def exact_match3(trues, preds):
    """match3 / total3 in the memorization_sigdigits convention."""
    m3 = tot = 0
    for tv, pv in zip(trues, preds):
        if sig_figs(tv) < 3:
            continue
        tot += 1
        if sig_round(pv, 3) == sig_round(tv, 3):
            m3 += 1
    return m3, tot


def pearson(trues, preds, keys):
    """Pearson r on the per-molecule mean prediction (group by molecule key)."""
    by = {}
    for t, p, k in zip(trues, preds, keys):
        by.setdefault(k, (t, []))[1].append(p)
    ts, ps = [], []
    for k, (t, plist) in by.items():
        plist = [x for x in plist if np.isfinite(x)]
        if plist:
            ts.append(t)
            ps.append(np.mean(plist))
    ts, ps = np.asarray(ts, float), np.asarray(ps, float)
    if len(ts) < 3 or np.std(ps) == 0:
        return float("nan")
    return pearsonr(ts, ps)[0]


def main():
    rows = []
    for name, (dir_, jf, csv, val, key, kind) in SETS.items():
        keyed = load_keyed(dir_, jf, csv, val, key)
        for model in ORDER:
            if model not in keyed:
                continue
            t, p, kk = keyed[model]
            m3, tot = exact_match3(t, p)
            m1, m2 = retention(t, p, 2)
            m2b, m3b = retention(t, p, 3)
            R21 = 100 * m2 / m1 if m1 else float("nan")
            R32 = 100 * m3b / m2b if m2b else float("nan")
            r = pearson(t, p, kk)
            rows.append(dict(dataset=name, kind=kind, model=model,
                             match3=m3, total3=tot,
                             three_sig_pct=round(100 * m3 / tot, 2) if tot else float("nan"),
                             m1=m1, m2=m2, R21=round(R21, 1) if m1 else float("nan"),
                             m2_lvl3=m2b, m3=m3b, R32=round(R32, 1) if m2b else float("nan"),
                             pearson_r=round(r, 2)))
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "permodel_memorization_si.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)
    print(df.to_string(index=False))
    print(f"\nSaved {out}")

    # pooled cross-check vs zeroshot_retention_extended.csv
    print("\nPOOLED 3-sig% per dataset (cross-check):")
    for name in SETS:
        sub = df[df.dataset == name]
        if len(sub):
            print(f"  {name:18s} 3sig%={100*sub.match3.sum()/sub.total3.sum():.2f}  "
                  f"R21={100*sub.m2.sum()/sub.m1.sum():.1f}  R32={100*sub.m3.sum()/sub.m2_lvl3.sum():.1f}")


if __name__ == "__main__":
    main()

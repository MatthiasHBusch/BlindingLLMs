"""
Extended zero-shot memorization + accuracy analysis for the SI (JCIM revision).

Adds, on top of the existing 3-sig exact-match / (m3/m2)-retention analysis:

  1. A 1->2 significant-digit retention  R21 = m2 / m1  (alongside R32 = m3 / m2).
     One extra significant digit ~= one uniform decimal of freedom, so a
     coincidental match should survive the next-digit test ~10% of the time and a
     genuinely recalled value ~100% of the time -- identical logic to R32, but the
     m1 counts are far larger, so the binomial test against the 10% floor is much
     more powerful (this is the reviewer's point that m2/m1 may be the stronger
     signal). Filters mirror the existing convention: for level-k retention we keep
     only predictions whose label AND prediction both carry >= k significant figures
     (a k-1 figure value can match at k-1 but never at k, which would inflate the
     denominator and depress R).

  2. Pearson r and the R2 score (coefficient of determination, sklearn convention)
     of the 0-shot predictions vs. ground truth, per model, computed on the
     per-molecule mean over repetitions. r2 can be strongly negative when the model
     predicts at the wrong scale (notably QM7); that is itself informative.

Includes the new modern post-cutoff control, the ASAP Discovery antiviral potency
dataset, alongside the three legacy benchmarks and the positive control.

No API calls. Reads local result JSONs + CSVs. Writes zeroshot_extended_si.csv and
prints LaTeX-ready rows for the SI tables.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_RESULTS = os.path.join(ROOT, "results")   # legacy 0-shot files
NEW_RESULTS = os.path.join(ROOT, "results")                # new potency 0-shot file
DATA = os.path.join(ROOT, "data")
PC = os.path.join(os.path.dirname(__file__), "positive_control")

PAPER_MODELS = {
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5", "gpt-5-mini", "gpt-5-nano",
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
}
ORDER = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
         "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
         "gpt-5", "gpt-5-mini", "gpt-5-nano"]

# dir, json, csv, value column, molecule-key column
BENCH = {
    "Delaney":       (REPO_RESULTS, "LLM_Results_delaney_zeroshot.json",
                      os.path.join(DATA, "delaney-processed.csv"),
                      "measured log solubility in mols per litre", "Compound ID"),
    "Lipophilicity": (REPO_RESULTS, "LLM_Results_lipophilicity_zeroshot.json",
                      os.path.join(DATA, "Lipophilicity.csv"), "exp", "smiles"),
    "QM7":           (REPO_RESULTS, "LLM_Results_qm7_zeroshot.json",
                      os.path.join(DATA, "qm7.csv"), "u0_atom", "smiles"),
    "Antiviral potency": (NEW_RESULTS, "LLM_Results_potency_zeroshot.json",
                          os.path.join(DATA, "antiviral_potency.csv"), "pic50", "smiles"),
}
PCSETS = {
    "Atomic weights": (PC, "PositiveControl_AtomicWeights.json",
                       os.path.join(PC, "known_atomic_weights.csv"), "atomic_weight", "symbol"),
    "Boiling points": (PC, "PositiveControl_BoilingPoints.json",
                       os.path.join(PC, "known_boiling_points.csv"), "bp_celsius", "smiles"),
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


def load_pairs(dir_, jf, csv, val, key):
    """Return {model: (trues[], preds[])} of all molecule x rep predictions."""
    df = pd.read_csv(csv).dropna(subset=[val])
    lut = df.set_index(key)[val].to_dict()
    res = json.load(open(os.path.join(dir_, jf)))
    pairs = {}
    for model, mol, preds in leaves(res):
        short = model.split("/")[-1]
        if short not in PAPER_MODELS or mol not in lut:
            continue
        tv = float(lut[mol])
        t, p = pairs.setdefault(short, ([], []))
        for pr in preds:
            try:
                pv = float(pr)
            except (TypeError, ValueError):
                continue
            t.append(tv)
            p.append(pv)
    return pairs


def retention(trues, preds, k):
    """Level-k retention R = m_k / m_{k-1} over pairs with >=k sig figs on both
    label and prediction. Returns (m_low, m_high, n_kept)."""
    m_low = m_high = n = 0
    for tv, pv in zip(trues, preds):
        if sig_figs(tv) < k or sig_figs(pv) < k:
            continue
        n += 1
        if sig_round(pv, k - 1) == sig_round(tv, k - 1):
            m_low += 1
            if sig_round(pv, k) == sig_round(tv, k):
                m_high += 1
    return m_low, m_high, n


def verdict(m_low, m_high):
    if m_low == 0:
        return float("nan"), float("nan")
    R = 100 * m_high / m_low
    p = binomtest(m_high, m_low, 0.10, alternative="two-sided").pvalue
    return R, p


def accuracy(trues, preds, key_list):
    """Pearson r and R2 score on the per-molecule mean prediction."""
    by_mol = {}
    for k, t, p in zip(key_list, trues, preds):
        by_mol.setdefault(k, (t, []))[1].append(p)
    ts, ps = [], []
    for k, (t, plist) in by_mol.items():
        plist = [x for x in plist if np.isfinite(x)]
        if not plist:
            continue
        ts.append(t)
        ps.append(np.mean(plist))
    ts, ps = np.asarray(ts, float), np.asarray(ps, float)
    if len(ts) < 3 or np.std(ps) == 0:
        return float("nan"), float("nan"), len(ts)
    r = pearsonr(ts, ps)[0]
    ss_res = np.sum((ts - ps) ** 2)
    ss_tot = np.sum((ts - np.mean(ts)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return r, r2, len(ts)


def load_pairs_keyed(dir_, jf, csv, val, key):
    """Like load_pairs but also returns the molecule key per prediction (for r/R2)."""
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
            t.append(tv); p.append(pv); kk.append(mol)
    return out


def main():
    rows = []
    all_sets = [(n, *cfg, "benchmark") for n, cfg in BENCH.items()]
    all_sets += [(n, *cfg, "control") for n, cfg in PCSETS.items()]

    # ---- Pooled retention table (R21 and R32) ----
    print("=" * 100)
    print("POOLED RETENTION (over the nine paper models)")
    print(f"{'dataset':18s} | {'m1':>6} {'m2':>6} {'R21%':>6} {'p21':>9} | "
          f"{'m2(>=3)':>7} {'m3':>5} {'R32%':>6} {'p32':>9}")
    print("-" * 100)
    for name, dir_, jf, csv, val, key, kind in all_sets:
        pairs = load_pairs(dir_, jf, csv, val, key)
        M1 = M2 = 0           # level-2 test (>=2 sig): m1 -> m2
        M2b = M3 = 0          # level-3 test (>=3 sig): m2 -> m3
        for model in pairs:
            t, p = pairs[model]
            a1, a2, _ = retention(t, p, 2)
            b2, b3, _ = retention(t, p, 3)
            M1 += a1; M2 += a2; M2b += b2; M3 += b3
        R21, p21 = verdict(M1, M2)
        R32, p32 = verdict(M2b, M3)
        print(f"{name:18s} | {M1:6d} {M2:6d} {R21:6.1f} {p21:9.1e} | "
              f"{M2b:7d} {M3:5d} {R32:6.1f} {p32:9.1e}")
        rows.append(dict(dataset=name, kind=kind, m1=M1, m2_lvl2=M2, R21=round(R21, 1), p21=p21,
                         m2_lvl3=M2b, m3=M3, R32=round(R32, 1), p32=p32))

    # ---- Per-model accuracy (Pearson r, R2) for every dataset ----
    print("\n" + "=" * 100)
    print("ACCURACY (per-molecule mean prediction): Pearson r and R2 score")
    acc_rows = []
    for name, dir_, jf, csv, val, key, kind in all_sets:
        keyed = load_pairs_keyed(dir_, jf, csv, val, key)
        print(f"\n--- {name} ---")
        print(f"{'model':22s} {'n':>5} {'PearsonR':>9} {'R2score':>9}")
        rr, r2r = [], []
        for model in ORDER:
            if model not in keyed:
                continue
            t, p, kk = keyed[model]
            r, r2, n = accuracy(t, p, kk)
            rr.append(r); r2r.append(r2)
            print(f"{model:22s} {n:5d} {r:9.3f} {r2:9.2f}")
            acc_rows.append(dict(dataset=name, model=model, n=n,
                                 pearson_r=round(r, 3), r2_score=round(r2, 3)))
        if rr:
            print(f"{'MEAN':22s} {'':>5} {np.nanmean(rr):9.3f} {np.nanmean(r2r):9.2f}")

    pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "zeroshot_retention_extended.csv"), index=False)
    pd.DataFrame(acc_rows).to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "zeroshot_accuracy.csv"), index=False)

    # ---- Per-model 3-sig matches + r/R2 for the potency dataset (SI table) ----
    print("\n" + "=" * 100)
    print("ANTIVIRAL POTENCY per-model: 3-sig matches, Pearson r, R2 (SI table rows)")
    dir_, jf, csv, val, key = (NEW_RESULTS, "LLM_Results_potency_zeroshot.json",
                               os.path.join(DATA, "antiviral_potency.csv"), "pic50", "smiles")
    keyed = load_pairs_keyed(dir_, jf, csv, val, key)
    df = pd.read_csv(csv)
    n_hi = int((df[val].apply(sig_figs) >= 3).sum())
    print(f"(ground-truth molecules with >=3 sig figs: {n_hi} of {len(df)})")
    print(f"{'model':22s} {'m3':>4} {'tot':>5} {'3sig%':>7} {'PearsonR':>9} {'R2':>7}")
    for model in ORDER:
        if model not in keyed:
            continue
        t, p, kk = keyed[model]
        m3 = tot = 0
        # Match-table convention (memorization_sigdigits.py): total = predictions
        # over ground-truth values with >=3 sig figs; a <3-sig prediction simply
        # fails to match (it is not excluded from the denominator here).
        for tv, pv in zip(t, p):
            if sig_figs(tv) < 3:
                continue
            tot += 1
            if sig_round(pv, 3) == sig_round(tv, 3):
                m3 += 1
        r, r2, n = accuracy(t, p, kk)
        pct = 100 * m3 / tot if tot else 0
        print(f"{model:22s} {m3:4d} {tot:5d} {pct:7.2f} {r:9.3f} {r2:7.2f}")

    print("\nSaved zeroshot_retention_extended.csv and zeroshot_accuracy.csv")


if __name__ == "__main__":
    main()

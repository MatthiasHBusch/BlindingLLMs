"""
Is QM7's elevated second-digit retention (R21=m2/m1) due to a clustered (non-uniform)
second significant digit, or to genuine value recall?

Two independent tests, applied to QM7 and -- as a uniform-digit control -- Delaney:

  (1) DISTRIBUTION TEST. Empirical distribution of the k-th significant digit of the
      ground-truth values. Chi-square against uniform(0..9) and the collision
      probability C_k = sum_d p(d)^2 (the chance two independent draws share digit k).
      C_k == 0.10 iff the digit is uniform; C_k > 0.10 quantifies clustering. The
      coincidence floor for retention at digit k is C_k, NOT a fixed 10%.

  (2) PERMUTATION (no-recall) NULL. Shuffle the predictions across molecules, which
      destroys any true<->prediction correspondence (hence any recall) but preserves
      both marginal digit distributions, and recompute R21/R32. If the observed
      retention is just marginal clustering, the shuffled null reproduces it; if it
      were recall, shuffling would collapse it to the collision floor.

Writes a 2nd-significant-digit bar figure to PaperICMLGraphics/qm7_second_digit.png.
No API calls.
"""

import json
import math
import os
import numpy as np
import pandas as pd
from scipy.stats import chisquare
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
GFX = os.path.join(ROOT, "figures")

PAPER_MODELS = {
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5", "gpt-5-mini", "gpt-5-nano",
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
}
SETS = {
    "QM7":     ("LLM_Results_qm7_zeroshot.json", os.path.join(DATA, "qm7.csv"), "u0_atom", "smiles"),
    "Delaney": ("LLM_Results_delaney_zeroshot.json", os.path.join(DATA, "delaney-processed.csv"),
                "measured log solubility in mols per litre", "Compound ID"),
}
rng = np.random.default_rng(0)


def sig_round(x, n):
    return "0" if x == 0 else f"{x:.{n}g}"


def sig_figs(v):
    if v == 0:
        return 1
    s = f"{v:.10g}".lstrip("-").replace(".", "").lstrip("0")
    return max(1, len(s.rstrip("0"))) if s else 1


def kth_digit(v, k):
    """k-th significant digit of |v| by truncation (1-indexed)."""
    if v == 0:
        return None
    mant = f"{abs(float(v)):.12e}".split("e")[0].replace(".", "")
    return int(mant[k - 1])


def leaves(o, path=()):
    if isinstance(o, dict):
        for kk, v in o.items():
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                yield "/".join(path), kk, v
            else:
                yield from leaves(v, path + (kk,))


def load_pairs(jf, csv, val, key, model_filter=None):
    df = pd.read_csv(csv).dropna(subset=[val])
    lut = df.set_index(key)[val].to_dict()
    res = json.load(open(os.path.join(RESULTS, jf)))
    trues, preds = [], []
    for model, mol, ps in leaves(res):
        short = model.split("/")[-1]
        if short not in PAPER_MODELS or mol not in lut:
            continue
        if model_filter and short != model_filter:
            continue
        tv = float(lut[mol])
        for p in ps:
            try:
                pv = float(p)
            except (TypeError, ValueError):
                continue
            trues.append(tv)
            preds.append(pv)
    return np.array(trues), np.array(preds), df[val].dropna().astype(float).values


def digit_dist(values, k):
    """Distribution over digits 0..9 of the k-th significant digit, for values with >=k sig figs."""
    counts = np.zeros(10, int)
    for v in values:
        if sig_figs(v) < k:
            continue
        d = kth_digit(v, k)
        if d is not None:
            counts[d] += 1
    return counts


def collision(counts):
    p = counts / counts.sum()
    return float(np.sum(p ** 2))


def retention(trues, preds, k):
    """(m_low, m_high) at level k over pairs with >=k sig figs on both sides."""
    m_low = m_high = 0
    for tv, pv in zip(trues, preds):
        if sig_figs(tv) < k or sig_figs(pv) < k:
            continue
        if sig_round(pv, k - 1) == sig_round(tv, k - 1):
            m_low += 1
            if sig_round(pv, k) == sig_round(tv, k):
                m_high += 1
    return m_low, m_high


def perm_null(trues, preds, k, B=2000):
    """Null R = m_k/m_{k-1} when predictions are shuffled across molecules (no recall),
    preserving both marginals. Returns (mean, lo, hi)."""
    # restrict once to the >=k universe so shuffling stays within it
    mask = np.array([sig_figs(t) >= k and sig_figs(p) >= k for t, p in zip(trues, preds)])
    t = trues[mask]
    p = preds[mask]
    tl = np.array([sig_round(x, k - 1) for x in t])
    th = np.array([sig_round(x, k) for x in t])
    pl = np.array([sig_round(x, k - 1) for x in p])
    ph = np.array([sig_round(x, k) for x in p])
    out = []
    for _ in range(B):
        idx = rng.permutation(len(p))
        low = (pl[idx] == tl)
        if low.sum() == 0:
            continue
        high = low & (ph[idx] == th)
        out.append(100 * high.sum() / low.sum())
    out = np.array(out)
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5)


def main():
    print(f"{'='*78}\nQM7 SECOND-DIGIT CLUSTERING PROOF\n{'='*78}")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, name in zip(axes, ["QM7", "Delaney"]):
        jf, csv, val, key = SETS[name]
        trues, preds, dataset_vals = load_pairs(jf, csv, val, key)
        print(f"\n----- {name} -----")
        # (1) distribution test on the ground-truth dataset values
        for k in (1, 2, 3):
            c = digit_dist(dataset_vals, k)
            n = c.sum()
            chi = chisquare(c)  # vs uniform
            print(f"  GT digit {k}: n={n:5d}  collision C{k}={100*collision(c):5.1f}%  "
                  f"chi2={chi.statistic:8.1f}  p={chi.pvalue:.2e}  "
                  f"{'NON-UNIFORM' if chi.pvalue < 0.05 else 'uniform'}")
        # (2) observed vs permutation-null retention (pooled over models)
        for k, label in ((2, "R21"), (3, "R32")):
            ml, mh = retention(trues, preds, k)
            obs = 100 * mh / ml if ml else float("nan")
            nm, lo, hi = perm_null(trues, preds, k)
            print(f"  {label}: observed={obs:5.1f}%  (m_low={ml}, m_high={mh})   "
                  f"no-recall null={nm:5.1f}% [{lo:.1f}, {hi:.1f}]")
        # Gemini 2.5 Pro alone for QM7 (it drives the pooled effect)
        if name == "QM7":
            tp, pp, _ = load_pairs(jf, csv, val, key, model_filter="gemini-2.5-pro")
            ml, mh = retention(tp, pp, 2)
            obs = 100 * mh / ml if ml else float("nan")
            nm, lo, hi = perm_null(tp, pp, 2)
            print(f"  [gemini-2.5-pro] R21: observed={obs:5.1f}% (m_low={ml})  "
                  f"no-recall null={nm:5.1f}% [{lo:.1f}, {hi:.1f}]")

        # figure: 2nd-digit distribution of GT vs uniform
        c2 = digit_dist(dataset_vals, 2)
        p2 = 100 * c2 / c2.sum()
        ax.bar(range(10), p2, color="#4C72B0", alpha=0.85, label="ground truth")
        ax.axhline(10, color="crimson", ls="--", lw=1.5, label="uniform (10%)")
        ax.set_title(f"{name}: 2nd significant digit")
        ax.set_xlabel("digit")
        ax.set_xticks(range(10))
        ax.set_ylim(0, max(p2.max(), 12) * 1.15)
    axes[0].set_ylabel("frequency (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(GFX, "qm7_second_digit.png")
    fig.savefig(out, dpi=200)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()

"""
Non-monotonic label transform for Delaney (JCIM revision, Reviewer 2.2).

R2.2 asks for at least one NON-monotonic transformation alongside the affine one,
because a monotonic (rank-preserving) map is "recoverable from a handful of
in-context anchors" and so, the reviewer argues, does not really disrupt
memorization. We answer this with a *binned permutation* (one of the three options
the reviewer names), built to be directly comparable to our affine transform:

  * the value range is split into K equal-count bins;
  * the bins are reordered by a FIXED permutation (a derangement, so no bin keeps
    its place), which makes the global map non-monotonic -- it cannot be recovered
    by assuming monotonicity from a few anchors;
  * within each bin the original order/spacing is preserved (value-linear), so the
    map is INVERTIBLE and information-preserving -- a genuine in-context learner can
    still recover it;
  * the output is rescaled to the SAME [0, 100] range as the affine transform, so it
    closes the lexical-lookup loophole identically and is plotted on the same axis.

The point of the experiment: under this non-monotonic, non-recoverable relabeling, a
model that merely recovers the original scale (or recalls values) must fail, whereas
genuine in-context regression -- interpolating a target's value from structurally
similar neighbours -- is invariant to ANY relabeling and should still work. Delaney
is the ideal testbed (a strong, learnable structure-property signal), so we expect a
modest drop but retained performance.

Output: Data/delaney-processed-nonmono.csv -- an exact copy of delaney-processed.csv
with the `transformed_solubility` column REPLACED by the non-monotonic transform (all
other columns, including `transformed_smiles`, untouched), so the existing Delaney
experiment pipeline runs on it unchanged.

Usage:  python Src/Revision/make_delaney_nonmono.py
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
IN = os.path.join(DATA, "delaney-processed.csv")
OUT = os.path.join(DATA, "delaney-processed-nonmono.csv")
VALUE_COL = "measured log solubility in mols per litre"

# Fixed, reproducible binned-permutation parameters.
K = 5
# Derangement of {0..4} (no element maps to its own index) -> non-monotonic.
PERM = (2, 4, 1, 0, 3)


def binned_permutation(y, k=K, perm=PERM):
    """Map y to [0,100] by equal-count binning + a fixed bin permutation,
    preserving value order within each bin (invertible, non-monotonic)."""
    y = np.asarray(y, float)
    assert len(perm) == k and sorted(perm) == list(range(k)), "perm must be a permutation of 0..k-1"
    edges = np.quantile(y, np.linspace(0, 1, k + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    out = np.empty_like(y)
    for b in range(k):
        lo, hi = edges[b], edges[b + 1]
        mask = (y > lo) & (y <= hi)
        if not mask.any():
            continue
        yb = y[mask]
        if yb.max() > yb.min():
            u = (yb - yb.min()) / (yb.max() - yb.min())   # within-bin position, value-linear
        else:
            u = np.full(yb.shape, 0.5)
        out[mask] = (perm[b] + u) / k * 100.0             # bin perm[b] occupies output band
    return out


def sin_transform(y, periods=2):
    """Map y to [0,100] with a sine over `periods` full periods: CONTINUOUS and
    non-monotonic (many-to-one), preserving LOCAL similarity everywhere (no
    boundary discontinuities, unlike the binned permutation).

        u = (y - min) / (max - min) in [0,1]
        theta = 2*pi*periods*u            (periods=2 -> theta in [0, 4*pi], two cycles)
        out = (sin(theta) + 1) / 2 * 100

    Over a whole number of periods, Pearson(out, y) ~ 0: prior knowledge / recall of
    the true value is therefore useless for predicting this target, so any correlation
    the model achieves must come from in-context learning of the oscillating map. The
    cost of continuity is non-injectivity (a continuous non-monotonic map on an
    interval must be many-to-one), which slightly lowers the achievable ceiling near
    the sine's turning points -- bound this with the kNN-Tanimoto structural ceiling
    under the same transform (analyze_nonmono.py)."""
    y = np.asarray(y, float)
    u = (y - y.min()) / (y.max() - y.min())
    theta = 2.0 * np.pi * periods * u
    return (np.sin(theta) + 1.0) / 2.0 * 100.0


def main():
    ap = argparse.ArgumentParser(description="Non-monotonic Delaney label transforms (Reviewer 2.2)")
    ap.add_argument("--transform", choices=["sin", "binned"], default="sin",
                    help="sin = continuous non-monotonic (default, recommended); "
                         "binned = binned permutation (discontinuous)")
    ap.add_argument("--periods", type=int, default=2,
                    help="sin only: number of full periods over the range (default 2 -> 4*pi)")
    args = ap.parse_args()

    df = pd.read_csv(IN)
    y = df[VALUE_COL].to_numpy(float)

    if args.transform == "sin":
        nm = sin_transform(y, periods=args.periods)
        out_path = os.path.join(DATA, "delaney-processed-sin.csv")
        label = f"sin, {args.periods} periods (theta in [0,{2*args.periods}pi])"
    else:
        nm = binned_permutation(y)
        out_path = os.path.join(DATA, "delaney-processed-nonmono.csv")
        label = f"binned permutation (K={K}, perm={PERM})"

    pr = pearsonr(nm, y)[0]
    sr = spearmanr(nm, y)[0]
    aff = df["transformed_solubility"].to_numpy(float)
    print(f"Loaded {len(df)} Delaney molecules from {IN}")
    print(f"Transform: {label}")
    print(f"output range: [{nm.min():.2f}, {nm.max():.2f}]  (target [0,100])")
    print(f"Pearson(transformed, original)  = {pr:+.3f}   (affine = -1.000; ~0 => recall of the true value is useless here)")
    print(f"Spearman(transformed, original) = {sr:+.3f}")
    print(f"Pearson(transformed, affine)    = {pearsonr(nm, aff)[0]:+.3f}")

    out = df.copy()
    out["transformed_solubility"] = nm
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print("Next: point Run_Delaney_NonMono.jl at this file (transform_tag) and run "
          "(GPT-5, Level 2 by default).")


if __name__ == "__main__":
    main()

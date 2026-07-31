"""
Analyze the positive-control BLINDING sweeps (JCIM revision round 2, Reviewer 2.2).

R2 asks which blinding level suffices to break verbatim recall on data the models
demonstrably memorise. Two sweeps, selected by the first command-line argument, both
with the single-step prompts the paper reports:

  direct         45 Wikipedia boiling points, all six levels
  atomic_direct  41 IUPAC atomic weights, levels 1-4 (the Agnostic levels are
                 undefined there: each element occurs once, so an opaque structure
                 string carries no inferable token-to-value mapping)

For every level we report Pearson |r| and the exact-match rate against that level's
OWN target -- the untransformed value at levels 1/3/5, the rescaled value at 2/4/6,
where a match additionally requires reconstructing the affine map -- plus the
precision-scaling retention ratios in the convention of Tables S1/S2, Fisher and
sign tests between levels, and the correlation between cracking a level and the
model's 0-shot hit rate on the same set.

Scale caveat: ~30 in-context examples and ~45 test items against up to 1000 and 150
in the main experiments. Absolute correlations are not comparable to Figures 3/4;
only the within-sweep comparison across levels is, and only pooled counts support
inference.

Usage:
    python analyze_blinding_control.py [direct|atomic_direct|boiling|atomic]
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ZEROSHOT = os.path.join(HERE, "positive_control_results.csv")

# Blinding levels in the order used in Figures 3 and 4 of the manuscript.
ALL_LEVELS = [
    ("with_preanalysis", "1 Specific / original", False),
    ("wp_solubility_blind", "2 Specific / transformed", True),
    ("wp_molproperty_clear", "3 Generic / original", False),
    ("wp_molproperty_blind", "4 Generic / transformed", True),
    ("wp_sampleproperty_clear", "5 Agnostic / original", False),
    ("wp_sampleproperty_blind", "6 Agnostic / transformed", True),
]

# The direct-prompt sweep uses the same six information conditions with
# single-step prompts (no pre-analysis, no weighted-average instruction).
DIRECT_LEVELS = [
    ("io_specific_clear", "1 Specific / original", False),
    ("io_specific_blind", "2 Specific / transformed", True),
    ("io_molproperty_clear", "3 Generic / original", False),
    ("io_molproperty_blind", "4 Generic / transformed", True),
    ("io_sampleproperty_clear", "5 Agnostic / original", False),
    ("io_sampleproperty_blind", "6 Agnostic / transformed", True),
]

DATASETS = {
    "direct": dict(
        results="PositiveControl_Blinding_Direct.json",
        data="known_boiling_points_blinded.csv",
        key="name", value="bp_celsius", zeroshot_set="BoilingPoints",
        levels=DIRECT_LEVELS, suffix="_direct",
        title="boiling points, DIRECT single-step prompts",
    ),
    "atomic_direct": dict(
        results="PositiveControl_Blinding_AtomicWeights_Direct.json",
        data="known_atomic_weights_blinded.csv",
        key="name", value="atomic_weight", zeroshot_set="AtomicWeights",
        # Levels 5/6 are undefined here: each element occurs exactly once, so an
        # opaque structure string carries no inferable token-to-value mapping.
        levels=DIRECT_LEVELS[:4], suffix="_atomicweights_direct",
        title="atomic weights, DIRECT single-step prompts (levels 1-4)",
    ),
}
# ALL_LEVELS holds the benchmark two-step approach names. Those sweeps were run to
# check how much the prompt schema suppresses the detector, but they are not part of
# the paper and their predictions are not distributed here; the level definitions in
# BoilingPoint_Prompts.jl / AtomicWeight_Prompts.jl remain, since the single-step
# variants build on them.

DATASET = sys.argv[1] if len(sys.argv) > 1 else "direct"
if DATASET not in DATASETS:
    sys.exit(f"unknown dataset {DATASET!r}; choose from {sorted(DATASETS)}")
CFG = DATASETS[DATASET]
LEVELS = CFG["levels"]
RESULTS = os.path.join(HERE, CFG["results"])
DATA = os.path.join(HERE, CFG["data"])
SUFFIX = CFG["suffix"]

SHORT = {
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "google/gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "openai/gpt-5": "gpt-5",
    "openai/gpt-5-mini": "gpt-5-mini",
    "openai/gpt-5-nano": "gpt-5-nano",
    "openai/gpt-4.1": "gpt-4.1",
    "openai/gpt-4.1-mini": "gpt-4.1-mini",
    "openai/gpt-4.1-nano": "gpt-4.1-nano",
}


# --- detector (identical to analyze_positive_control.py / memorization_sigdigits.py)
def sig_round(x, n):
    return "0" if x == 0 else f"{x:.{n}g}"


def sig_figs(val):
    if val == 0:
        return 1
    s = f"{val:.10g}".lstrip("-").replace(".", "").lstrip("0")
    return max(1, len(s.rstrip("0"))) if s else 1


def match_rate(trues, preds, n=3):
    """3-sig exact-match rate over ground truths carrying >= n significant digits."""
    trues, preds = np.asarray(trues, float), np.asarray(preds, float)
    keep = np.array([sig_figs(t) >= n for t in trues], bool)
    t, p = trues[keep], preds[keep]
    if len(t) == 0:
        return np.nan, 0
    hits = sum(sig_round(a, n) == sig_round(b, n) for a, b in zip(t, p))
    return 100 * hits / len(t), len(t)


def chance_rate(trues, preds, n=3, n_perm=300, seed=0):
    """Match rate after shuffling predictions against ground truths."""
    rng = np.random.default_rng(seed)
    trues, preds = np.asarray(trues, float), np.asarray(preds, float)
    keep = np.array([sig_figs(t) >= n for t in trues], bool)
    t, p = trues[keep], preds[keep]
    if len(t) == 0:
        return np.nan
    ts = np.array([sig_round(x, n) for x in t])
    ps = np.array([sig_round(x, n) for x in p])
    hits = sum(np.sum(ts == ps[rng.permutation(len(ps))]) for _ in range(n_perm))
    return 100 * hits / (n_perm * len(ts))


def retention(trues, preds, k):
    """R = m_k/m_{k-1} over pairs with >=k sig figs on BOTH label and prediction.

    Copied from ../permodel_memorization_si.py so the numbers here are directly
    comparable to Tables S1 and S2. Returns the raw (m_{k-1}, m_k) counts; the
    coincidence floor is ~10% and genuine recall ~100%.
    """
    m_low = m_high = 0
    for tv, pv in zip(trues, preds):
        if sig_figs(tv) < k or sig_figs(pv) < k:
            continue
        if sig_round(pv, k - 1) == sig_round(tv, k - 1):
            m_low += 1
            if sig_round(pv, k) == sig_round(tv, k):
                m_high += 1
    return m_low, m_high


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return 100 * float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    return 100 * float(np.corrcoef(rx, ry)[0, 1])


def collect(node, truth_lut, n_runs=None):
    """-> list of (run_index, [(truth, prediction), ...]) for one model/level.

    n_runs defaults to however many repetitions the file actually holds. Results are
    appended per run, so a sweep topped up from 2 to 10 runs needs no code change --
    but a file left half-written by an aborted run would silently give some
    model/level cells more repetitions than others, which main() checks for.
    """
    if n_runs is None:
        n_runs = max((len(v) if isinstance(v, list) else 1) for v in node.values())
    runs = []
    for run in range(n_runs):
        pairs = []
        for key, vals in node.items():
            t = truth_lut.get(key)
            if t is None:
                continue
            vals = vals if isinstance(vals, list) else [vals]
            if run >= len(vals):
                continue
            try:
                p = float(vals[run])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(p):
                continue
            pairs.append((t, p))
        if pairs:
            runs.append((run, pairs))
    return runs


def main():
    if not os.path.exists(RESULTS):
        print(f"{os.path.basename(RESULTS)} not found -- run Run_PositiveControl_Blinding.jl first.")
        return

    print(f"### dataset: {CFG['title']}\n")
    df = pd.read_csv(DATA)
    bp = df.set_index(CFG["key"])[CFG["value"]].to_dict()
    tr = df.set_index(CFG["key"])["transformed_solubility"].to_dict()

    with open(RESULTS) as f:
        results = json.load(f)["names_only"]

    # Every model/level cell must carry the same number of repetitions, or the
    # pooled counts silently weight some models more than others.
    reps = {}
    for approach, _, _ in LEVELS:
        for model, mv in results.get(approach, {}).items():
            node = mv["0"][list(mv["0"].keys())[0]]
            for v in node.values():
                reps.setdefault(len(v) if isinstance(v, list) else 1, []).append(
                    f"{approach}/{model}")
    if len(reps) == 1:
        print(f"repetitions per item: {next(iter(reps))}\n")
    else:
        print("  [warn] uneven repetition counts across cells -- an aborted run?")
        for n, where in sorted(reps.items()):
            print(f"         {n} reps: {len(where)} cells, e.g. {where[0]}")
        print()

    rows = []
    for approach, label, transformed in LEVELS:
        if approach not in results:
            print(f"  [warn] level missing from results: {approach}")
            continue
        for model, mv in results[approach].items():
            # single key path: extended-training "0" -> fold training size
            node = mv["0"][list(mv["0"].keys())[0]]

            # ACCURACY: against this level's own target.
            own = tr if transformed else bp
            rs = [pearson(*zip(*pairs)) for _, pairs in collect(node, own)]
            rs = [abs(r) for r in rs if np.isfinite(r)]

            # RECALL: exact-match rate against THIS LEVEL'S OWN target.
            # At the untransformed levels that is the true boiling point, i.e.
            # plain verbatim recall. At the transformed levels it is the rescaled
            # value, so a match requires the model to recall the boiling point AND
            # to reconstruct the affine map from the in-context anchors. That makes
            # the transformed-level rate a CONSERVATIVE recall detector: a small
            # error in the inferred map destroys the match even when the underlying
            # value was recalled. We therefore also report the 2-significant-digit
            # rate and the retention ratio m3/m2, which tolerate an imprecise map.
            acc = {n: dict(m=[], ch=[], n=[]) for n in (2, 3)}
            for _, pairs in collect(node, own):
                t, p = zip(*pairs)
                for n in (2, 3):
                    m, cnt = match_rate(t, p, n)
                    acc[n]["m"].append(m)
                    acc[n]["ch"].append(chance_rate(t, p, n))
                    acc[n]["n"].append(cnt)

            def agg(n, key):
                v = [x for x in acc[n][key] if np.isfinite(x)]
                return float(np.mean(v)) if v else np.nan

            # Precision-scaling retention in the convention of Tables S1/S2:
            # counts pooled over runs so the ratios rest on as many matches as
            # possible (R21 in particular is the statistically tighter estimate).
            m1c = m2c = m2c_b = m3c = 0
            for _, pairs in collect(node, own):
                t, p = [x[0] for x in pairs], [x[1] for x in pairs]
                a, b = retention(t, p, 2)
                m1c += a
                m2c += b
                a, b = retention(t, p, 3)
                m2c_b += a
                m3c += b

            m2, m3 = agg(2, "m"), agg(3, "m")
            rows.append(dict(
                level=label, approach=approach, transformed=transformed,
                model=SHORT.get(model, model),
                r=round(float(np.mean(rs)), 1) if rs else np.nan,
                n_runs=len(rs),
                match2=round(m2, 1) if np.isfinite(m2) else np.nan,
                chance2=round(agg(2, "ch"), 1) if np.isfinite(agg(2, "ch")) else np.nan,
                match3=round(m3, 1) if np.isfinite(m3) else np.nan,
                chance3=round(agg(3, "ch"), 1) if np.isfinite(agg(3, "ch")) else np.nan,
                m1=m1c, m2_cnt=m2c, m2b=m2c_b, m3_cnt=m3c,
                # raw hits/eligible for the count-pooled match rate
                m3_hits=int(round(sum(m * n / 100 for m, n in
                                      zip(acc[3]["m"], acc[3]["n"]) if np.isfinite(m)))),
                m3_total=int(sum(acc[3]["n"])),
                R21=round(100 * m2c / m1c, 1) if m1c else np.nan,
                R32=round(100 * m3c / m2c_b, 1) if m2c_b else np.nan,
                n3=int(np.mean(acc[3]["n"])) if acc[3]["n"] else 0,
            ))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, f"blinding_control_permodel{SUFFIX}.csv"), index=False)

    print("=" * 78)
    print("ACCURACY track -- Pearson |r| x100 vs each level's own target (30-shot, 3-fold)")
    print("=" * 78)
    piv_r = out.pivot_table(index="model", columns="level", values="r")
    print(piv_r.to_string(float_format=lambda v: f"{v:5.1f}"))
    print("\npooled mean per level:")
    print(piv_r.mean(axis=0).to_string(float_format=lambda v: f"{v:5.1f}"))

    print()
    print("=" * 78)
    print("RECALL track -- 3-sig exact-match rate vs each level's OWN target (%)")
    print("  untransformed levels: plain verbatim recall of the boiling point")
    print("  transformed levels  : recall AND reconstruction of the affine map")
    print("=" * 78)
    piv_m = out.pivot_table(index="model", columns="level", values="match3")
    print(piv_m.to_string(float_format=lambda v: f"{v:5.1f}"))
    print("\npooled per level (counts summed over models and runs):")
    print(f"  {'level':26s} {'m3%':>5s} {'ch%':>4s} | {'R21':>5s} {'(m2/m1)':>12s}"
          f" | {'R32':>5s} {'(m3/m2)':>10s}")
    for _, lvl, _ in LEVELS:
        sub = out[out.level == lvl]
        if not len(sub):
            continue
        m1, m2c = sub.m1.sum(), sub.m2_cnt.sum()
        m2b, m3c = sub.m2b.sum(), sub.m3_cnt.sum()
        r21 = 100 * m2c / m1 if m1 else float("nan")
        r32 = 100 * m3c / m2b if m2b else float("nan")
        print(f"  {lvl:26s} {sub.match3.mean():5.1f} {sub.chance3.mean():4.1f}"
              f" | {r21:5.1f} {f'({m2c}/{m1})':>12s}"
              f" | {r32:5.1f} {f'({m3c}/{m2b})':>10s}")
    print("  (coincidence floor ~10%, genuine recall ~100%; R21 rests on far more")
    print("   matches than R32 and is the tighter estimate)")

    # --- does cracking track how well a model recalls this dataset? ----------
    print()
    print("=" * 78)
    print("Is cracking the blinding driven by the model's recall of this dataset?")
    print("=" * 78)
    if not os.path.exists(ZEROSHOT):
        print(f"  {os.path.basename(ZEROSHOT)} missing -- skipping.")
    else:
        zs = pd.read_csv(ZEROSHOT)
        zs = zs[zs.set == CFG["zeroshot_set"]].set_index("model")["obs3"].to_dict()
        print("  x = 0-shot 3-sig hit rate on this dataset (obs3, prompted for the")
        print("      boiling point by name)")
        for lvl_key, lvl_label in [(a, lbl) for a, lbl, _ in LEVELS]:
            sub = out[out.approach == lvl_key].copy()
            sub["obs3_zeroshot"] = sub.model.map(zs)
            sub = sub.dropna(subset=["obs3_zeroshot"])
            print(f"\n  --- y = {lvl_label} ---")
            print(sub[["model", "obs3_zeroshot", "match3", "r"]]
                  .sort_values("obs3_zeroshot", ascending=False)
                  .to_string(index=False, float_format=lambda v: f"{v:6.1f}"))
            for yname in ("match3", "r"):
                pr = pearson(sub.obs3_zeroshot, sub[yname])
                sr = spearman(sub.obs3_zeroshot, sub[yname])
                print(f"    corr(obs3_zeroshot, {yname:6s}) : Pearson r = {pr/100:+.2f}"
                      f"   Spearman rho = {sr/100:+.2f}   (n = {len(sub)})")
            sub.to_csv(os.path.join(HERE, f"blinding_control_cracking_{lvl_key}{SUFFIX}.csv"),
                       index=False)

    # --- significance of the level-to-level changes in the recall track ------
    print()
    print("=" * 78)
    print("Significance of the recall-track changes")
    print("=" * 78)
    pooled = {}
    for approach, label, transformed in LEVELS:
        if approach not in results:
            continue
        own = tr if transformed else bp
        hits = tot = 0
        for model, mv in results[approach].items():
            node = mv["0"][list(mv["0"].keys())[0]]
            for _, pairs in collect(node, own):
                t, p = zip(*pairs)
                r, n = match_rate(t, p)
                if n:
                    hits += int(round(r * n / 100))
                    tot += n
        pooled[approach] = (hits, tot, label)
        print(f"  {label:26s} {hits:4d}/{tot:4d} = {100*hits/tot:5.1f}%")

    def fisher(a, b, c, d):
        """Two-sided Fisher exact p for [[a,b],[c,d]] via exact hypergeometric sum."""
        from math import comb
        n = a + b + c + d
        row1, col1 = a + b, a + c
        obs = comb(row1, a) * comb(n - row1, c) / comb(n, col1)
        p = 0.0
        for k in range(max(0, col1 - (n - row1)), min(row1, col1) + 1):
            pk = comb(row1, k) * comb(n - row1, col1 - k) / comb(n, col1)
            if pk <= obs * (1 + 1e-9):
                p += pk
        return min(1.0, p)

    def compare(k1, k2):
        if k1 not in pooled or k2 not in pooled:
            return
        h1, t1, l1 = pooled[k1]
        h2, t2, l2 = pooled[k2]
        p = fisher(h1, t1 - h1, h2, t2 - h2)
        print(f"  {l1} vs {l2}: {100*h1/t1:.1f}% vs {100*h2/t2:.1f}%, "
              f"Fisher exact p = {p:.2g}")

    if len(pooled) == len(LEVELS):
        K = {i: LEVELS[i][0] for i in range(len(LEVELS))}
        print("\n  naming effect, values untransformed:")
        compare(K[0], K[2])
        compare(K[2], K.get(4))
        compare(K[0], K.get(4))

        print("\n  transform effect, holding the naming level fixed:")
        compare(K[0], K[1])
        compare(K[2], K[3])
        compare(K.get(4), K.get(5))

        print("\n  naming effect, values transformed:")
        compare(K[1], K[3])
        compare(K[3], K.get(5))

        # Per-model sign tests: does the change hold across models?
        from math import comb
        for k1, k2 in [(K[0], K[2]), (K[2], K.get(4)),
                       (K[0], K[1]), (K[2], K[3])]:
            if k1 not in pooled or k2 not in pooled:
                continue
            a = out[out.approach == k1].set_index("model").match3
            b = out[out.approach == k2].set_index("model").match3
            common = [m for m in a.index if m in b.index]
            down = sum(1 for m in common if b[m] < a[m])
            up = sum(1 for m in common if b[m] > a[m])
            n = down + up
            p = sum(comb(n, k) for k in range(min(down, up) + 1)) / 2 ** n * 2 if n else np.nan
            print(f"  sign test {pooled[k1][2]} -> {pooled[k2][2]}: "
                  f"{down} down / {up} up / {len(common)-n} tied, p = {min(p,1.0):.2g}")

    # Permutation p-value for the key correlation.
    if os.path.exists(ZEROSHOT):
        sub = out[out.approach == LEVELS[2][0]].copy()
        sub["obs3_zeroshot"] = sub.model.map(zs)
        sub = sub.dropna(subset=["obs3_zeroshot", "match3"])
        x, y = sub.obs3_zeroshot.to_numpy(), sub.match3.to_numpy()
        obs = abs(pearson(x, y))
        rng = np.random.default_rng(0)
        null = [abs(pearson(x, rng.permutation(y))) for _ in range(20000)]
        p = (1 + sum(v >= obs - 1e-9 for v in null)) / (1 + len(null))
        print(f"\n  corr(0-shot hit rate, level-3 match rate): r = {obs/100:+.2f}, "
              f"permutation p = {p:.3f} (n = {len(x)}, 20k permutations)")

    # --- is a near-zero match rate at the transformed levels really "no recall"? -
    # The transformed-level detector is conservative: a model could recall the
    # boiling point and still miss the 3-sig match because the affine map it
    # inferred from the in-context anchors is slightly off. To separate the two,
    # we absorb the model's own map error by fitting the best global affine
    # correction (2 parameters over 45 points) from prediction back to the true
    # boiling point, and then re-apply the identical detector. Recall with an
    # imprecise map snaps to exact values under this correction; genuine
    # in-context regression does not.
    print()
    print("=" * 78)
    print("Transformed levels: recall, or in-context regression?")
    print("  m3* = 3-sig match vs the TRUE (untransformed) target after the best")
    print("        global affine correction of the predictions (2 parameters).")
    print("  NOTE: low power -- on the atomic weights at level 2 it reads ~4% where")
    print("        the uncorrected detector reads ~39%, because a least-squares fit")
    print("        over all items is perturbed by outliers. Not positive evidence.")
    print("=" * 78)
    crows = []
    for approach, label, transformed in LEVELS:
        if not transformed or approach not in results:
            continue
        for model, mv in results[approach].items():
            node = mv["0"][list(mv["0"].keys())[0]]
            ms, chs = [], []
            for _, pairs in collect(node, bp):   # truth = TRUE boiling point
                t, p = np.array([x[0] for x in pairs]), np.array([x[1] for x in pairs])
                if len(t) < 5 or np.std(p) == 0:
                    continue
                a, b = np.polyfit(p, t, 1)       # best affine map pred -> bp
                corr = a * p + b
                m, _ = match_rate(t, corr)
                ms.append(m)
                chs.append(chance_rate(t, corr))
            if ms:
                crows.append(dict(level=label, model=SHORT.get(model, model),
                                  m3_corrected=round(float(np.mean(ms)), 1),
                                  chance=round(float(np.mean(chs)), 1)))
    cdf = pd.DataFrame(crows)
    if len(cdf):
        print(cdf.pivot_table(index="model", columns="level", values="m3_corrected")
              .to_string(float_format=lambda v: f"{v:5.1f}"))
        print("\npooled m3* / chance per level:")
        for _, lvl, transformed in LEVELS:
            if not transformed:
                continue
            sub = cdf[cdf.level == lvl]
            if len(sub):
                print(f"  {lvl:26s} {sub.m3_corrected.mean():5.1f} / {sub.chance.mean():4.1f}")
        print("\n  Compare against the UNCORRECTED rate for the same level above:")
        print("  where that is already well above chance (as at level 2), recall is")
        print("  established and this test adds nothing.")
        cdf.to_csv(os.path.join(HERE, f"blinding_control_affinecorrected{SUFFIX}.csv"), index=False)

    # --- emit the SI table body so the manuscript never transcribes by hand --
    order = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
             "gpt-5", "gpt-5-mini", "gpt-5-nano",
             "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
    pretty = {"gemini-2.5-pro": "Gemini 2.5 Pro", "gemini-2.5-flash": "Gemini 2.5 Flash",
              "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
              "gpt-5": "GPT-5", "gpt-5-mini": "GPT-5 mini", "gpt-5-nano": "GPT-5 nano",
              "gpt-4.1": "GPT-4.1", "gpt-4.1-mini": "GPT-4.1 mini",
              "gpt-4.1-nano": "GPT-4.1 nano"}

    def cell(v, dash="--"):
        return dash if not np.isfinite(v) else f"{v:.0f}"

    # Two stacked panels sharing the same columns (one per level, plus the model
    # name): correlation, then the verbatim-match rate, per model.
    ncol = len(LEVELS) + 1
    lines = []
    for panel, col, fmt in [
        (rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{(a) Accuracy: Pearson $|r|\times100$ "
         r"against each level's own target}} \\", "r", lambda v: cell(v)),
        (rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{(b) Verbatim recall: 3-significant-digit "
         r"exact-match rate (\%) against each level's own target}} \\", "match3",
         lambda v: "--" if not np.isfinite(v) else f"{v:.1f}"),
    ]:
        if lines:
            lines.append(r"\midrule")
        lines.append(panel)
        lines.append(r"\midrule")
        for m in order:
            vals = []
            for approach, _, _ in LEVELS:
                row = out[(out.approach == approach) & (out.model == m)]
                vals.append(fmt(row[col].iloc[0] if len(row) else np.nan))
            lines.append(f"{pretty[m]} & " + " & ".join(vals) + r" \\")
        vals = []
        for approach, _, _ in LEVELS:
            if col == "r":
                vals.append(cell(out[out.approach == approach].r.mean()))
            else:
                h, t, _ = pooled[approach]
                vals.append(f"{100*h/t:.1f}")
        lines.append(rf"\cmidrule{{1-{ncol}}}")
        lines.append(r"\textbf{Pooled} & " + " & ".join(vals) + r" \\")
        if col == "match3":
            ch = [f"{out[out.approach == a].chance3.mean():.1f}" for a, _, _ in LEVELS]
            lines.append(r"\textit{chance} & " + " & ".join(ch) + r" \\")
            # Retention from COUNTS pooled over models and runs -- a mean of
            # per-model ratios would be dominated by cells resting on a handful
            # of matches (same reasoning as the pooled row of Tables S1/S2).
            for num, den, lbl in [("m2_cnt", "m1", r"\textit{retention $R_{21}=m_2/m_1$}"),
                                  ("m3_cnt", "m2b", r"\textit{retention $R_{32}=m_3/m_2$}")]:
                vals = []
                for a, _, _ in LEVELS:
                    sub = out[out.approach == a]
                    d = sub[den].sum()
                    vals.append(f"{100*sub[num].sum()/d:.0f}" if d else "--")
                lines.append(lbl + " & " + " & ".join(vals) + r" \\")
            corr = []
            for a, lvl, transformed in LEVELS:
                if not transformed or not len(cdf):
                    corr.append("--")
                else:
                    corr.append(f"{cdf[cdf.level == lvl].m3_corrected.mean():.1f}")
            lines.append(r"\textit{after affine corr.} & " + " & ".join(corr) + r" \\")

    tex = os.path.join(HERE, f"blinding_control_table{SUFFIX}.tex")
    with open(tex, "w", encoding="utf8", newline="\n") as f:
        f.write("% Generated by analyze_blinding_control.py -- do not edit by hand.\n")
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {os.path.basename(tex)} (SI table body)")

    print("\nWrote blinding_control_permodel.csv and blinding_control_cracking_*.csv")
    print("\nCAVEAT for the write-up: ~30 in-context examples and ~45 test items,")
    print("versus up to 1000 examples and 150 test molecules on the main benchmarks.")
    print("Absolute correlations are therefore NOT comparable to Figures 3/4, and the")
    print("in-context-learning component is necessarily weaker here. Only the")
    print("within-sweep comparison across levels is meaningful, and only counts")
    print("pooled over models and runs support inference.")


if __name__ == "__main__":
    main()

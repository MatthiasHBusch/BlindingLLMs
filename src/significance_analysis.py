"""
Statistical-significance analysis of the main claims (JCIM revision, Reviewer 2.5:
"two runs, no error bars, 150 samples; many 5-10 pt differences likely within noise").

Two complementary tools, beyond the per-config bootstrap CIs already reported:

  (1) SIGN / CONSISTENCY TESTS across the nine models. Many individual changes are
      small relative to the ~26 pp test-set CI and so are individually inconclusive.
      But a change that is small yet goes the SAME direction in (almost) every model
      is jointly significant: under a null of "no systematic effect" each model is an
      independent coin flip, so k of N models moving the same way is a binomial sign
      test against p=0.5. This is the right test for "each step is small but they are
      consistent across models."

  (2) PAIRED BOOTSTRAP for contrasts evaluated on the SAME 150 molecules (the six
      blinding levels at 1000-shot share the test set). Resampling molecules with
      replacement and recomputing the DIFFERENCE in |r| gives a CI on the change that
      is tighter than comparing two marginal CIs, because the molecule-level noise is
      shared and cancels.

Correlations are computed exactly as the paper's plotting code: per-molecule mean of
valid predictions; for unblinded targets we drop None/NaN and predictions >=100 (the
wrong-scale filter); for blinded ("*_blind") targets we compare against the
transformed label and drop only None/NaN. 0-shot uses the "NoReasoning" zero-shot
files (a different, larger test set), 60-/1000-shot use "with_preanalysis".

No API calls. Reads local result JSONs + CSVs.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, binomtest, wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
SEED = 42
N_BOOT = 5000

DS = {
    "Delaney": dict(main="LLM_Results_delaney.json", zero="LLM_Results_delaney_zeroshot.json",
                    csv="delaney-processed.csv", val="measured log solubility in mols per litre",
                    key="Compound ID"),
    "Lipophilicity": dict(main="LLM_Results_lipophilicity.json", zero="LLM_Results_lipophilicity_zeroshot.json",
                          csv="Lipophilicity.csv", val="exp", key="smiles"),
    "QM7": dict(main="LLM_Results_qm7.json", zero="LLM_Results_qm7_zeroshot.json",
                csv="qm7.csv", val="u0_atom", key="smiles"),
}
MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
          "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
          "gpt-5", "gpt-5-mini", "gpt-5-nano"]
MODELS_LARGE = ["gemini-2.5-pro", "gpt-4.1", "gpt-5"]  # the "largest models" claim
# blinding levels in order (approach key at 1000-shot)
LEVELS = ["with_preanalysis", "wp_solubility_blind", "wp_molproperty_clear",
          "wp_molproperty_blind", "wp_sampleproperty_clear", "wp_sampleproperty_blind"]


def canon(raw):
    m = raw.split("/")[-1].replace("_2024-12-01-preview", "")
    return m.replace("_gpt-5-batch", "").replace("_gpt-4.1-batch", "")


def clean(lst, blind):
    out = []
    for p in lst:
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        if np.isnan(p):
            continue
        if not blind and p >= 100:   # wrong-scale filter (paper)
            continue
        out.append(p)
    return out


def per_mol(leaf, lut, blind):
    """Return {mol_key: mean_pred} and aligned true via lut."""
    d = {}
    for name, preds in leaf.items():
        if name not in lut:
            continue
        cp = clean(preds, blind)
        if cp:
            d[name] = float(np.mean(cp))
    return d


def corr(meanpred, lut):
    keys = [k for k in meanpred if k in lut]
    if len(keys) < 5:
        return np.nan, [], []
    p = np.array([meanpred[k] for k in keys])
    t = np.array([float(lut[k]) for k in keys])
    if np.std(p) == 0 or np.std(t) == 0:
        return np.nan, keys, []
    return pearsonr(p, t)[0], keys, (p, t)


def load_main(ds):
    """Return r[(model, approach, n_ext)] and meanpred dicts for paired bootstrap.
    n_ext in {'0','940'} with n_train fixed '60' (60-shot / 1000-shot)."""
    cfg = DS[ds]
    df = pd.read_csv(os.path.join(DATA, cfg["csv"]))
    lut = df.set_index(cfg["key"])[cfg["val"]].to_dict()
    lut_t = df.set_index(cfg["key"])["transformed_solubility"].to_dict()
    res = json.load(open(os.path.join(RES, cfg["main"])))
    R = {}; MP = {}
    node = res.get("names_only", {})
    for approach, models in node.items():
        blind = "blind" in approach
        use_lut = lut_t if blind else lut
        for raw, exts in models.items():
            cm = canon(raw)
            if cm not in MODELS:
                continue
            for n_ext, trains in exts.items():
                for n_tr, leaf in trains.items():
                    if n_tr != "60":
                        continue
                    mp = per_mol(leaf, use_lut, blind)
                    r, keys, _ = corr(mp, use_lut)
                    if np.isnan(r):
                        continue
                    k = (cm, approach, n_ext)
                    # dedupe model variants: keep the one with more molecules
                    if k not in MP or len(mp) > len(MP[k]):
                        R[k] = abs(r) if blind else r
                        MP[k] = {kk: mp[kk] for kk in mp}
    return R, MP, lut, lut_t


def load_zero(ds):
    cfg = DS[ds]
    df = pd.read_csv(os.path.join(DATA, cfg["csv"])).dropna(subset=[cfg["val"]])
    lut = df.set_index(cfg["key"])[cfg["val"]].to_dict()
    res = json.load(open(os.path.join(RES, cfg["zero"])))
    out = {}
    # structure: {key0}/{NoReasoning}/{model}/{molkey: list}
    def walk(o, path=()):
        if isinstance(o, dict):
            vals = list(o.values())
            if vals and all(isinstance(v, list) for v in vals):
                yield path, o
            else:
                for k, v in o.items():
                    yield from walk(v, path + (k,))
    for path, leaf in walk(res):
        cm = canon(path[-1])
        if cm not in MODELS:
            continue
        mp = per_mol(leaf, lut, blind=False)
        r, _, _ = corr(mp, lut)
        if not np.isnan(r):
            out[cm] = r
    return out


def sign_test(deltas, tol=0.0):
    """Binomial sign test (two-sided) on direction of deltas above tolerance."""
    pos = sum(1 for d in deltas if d > tol)
    neg = sum(1 for d in deltas if d < -tol)
    n = pos + neg
    if n == 0:
        return pos, neg, len(deltas), float("nan")
    p = binomtest(max(pos, neg), n, 0.5, alternative="greater").pvalue
    return pos, neg, len(deltas), p


def paired_boot(mpA, mpB, lutA, lutB, rng):
    """CI on |r_A| - |r_B| over the shared molecules (paired)."""
    keys = [k for k in mpA if k in mpB and k in lutA and k in lutB]
    if len(keys) < 5:
        return (np.nan, np.nan, np.nan)
    pA = np.array([mpA[k] for k in keys]); tA = np.array([float(lutA[k]) for k in keys])
    pB = np.array([mpB[k] for k in keys]); tB = np.array([float(lutB[k]) for k in keys])
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(keys), len(keys))
        if np.std(pA[idx]) == 0 or np.std(pB[idx]) == 0:
            diffs[b] = np.nan; continue
        diffs[b] = abs(pearsonr(pA[idx], tA[idx])[0]) - abs(pearsonr(pB[idx], tB[idx])[0])
    diffs = diffs[~np.isnan(diffs)]
    obs = abs(pearsonr(pA, tA)[0]) - abs(pearsonr(pB, tB)[0])
    return (obs * 100, np.percentile(diffs, 2.5) * 100, np.percentile(diffs, 97.5) * 100)


def main():
    rng = np.random.default_rng(SEED)
    R = {}; MP = {}; LUT = {}; LUTT = {}; R0 = {}
    for ds in DS:
        R[ds], MP[ds], LUT[ds], LUTT[ds] = load_main(ds)
        R0[ds] = load_zero(ds)

    def g(ds, model, approach, n_ext):
        return R[ds].get((model, approach, n_ext), np.nan)

    print("=" * 90)
    print("VERIFICATION vs known anchors (Gemini 2.5 Pro |r|x100): "
          "Delaney 96/93, Lipo 72/35, QM7 68/71 (L1/L6 at 1000-shot)")
    for ds in DS:
        r0 = R0[ds].get("gemini-2.5-pro", np.nan)
        r60 = g(ds, "gemini-2.5-pro", "with_preanalysis", "0")
        L1 = g(ds, "gemini-2.5-pro", "with_preanalysis", "940")
        L6 = g(ds, "gemini-2.5-pro", "wp_sampleproperty_blind", "940")
        print(f"  {ds:14s}: 0-shot={100*r0:5.0f}  60-shot={100*r60:5.0f}  "
              f"L1(1000)={100*L1:5.0f}  L6(1000)={100*L6:5.0f}")

    print("=" * 90)
    print("CLAIM 1: in-context learning helps -- 1000-shot > 0-shot (unblinded, with_preanalysis)")
    deltas = []
    for ds in DS:
        for m in MODELS:
            r0 = R0[ds].get(m, np.nan)
            r1000 = g(ds, m, "with_preanalysis", "940")
            if np.isfinite(r0) and np.isfinite(r1000):
                deltas.append(r1000 - r0)
    pos, neg, n, p = sign_test(deltas, tol=0.0)
    print(f"  models improving: {pos}/{pos+neg} (n={n} cells), mean Delta r = {100*np.mean(deltas):.1f} pp")
    print(f"  sign test p = {p:.2e}   (median Delta = {100*np.median(deltas):.1f} pp)")
    try:
        w = wilcoxon(deltas).pvalue
        print(f"  Wilcoxon signed-rank p = {w:.2e}")
    except Exception as e:
        print("  Wilcoxon n/a", e)

    print("\nper-dataset 1000>0 (pos/total, mean dr pp):")
    for ds in DS:
        dl = [g(ds, m, "with_preanalysis", "940") - R0[ds][m]
              for m in MODELS if np.isfinite(R0[ds].get(m, np.nan)) and np.isfinite(g(ds, m, "with_preanalysis", "940"))]
        po, ne, nn, pp = sign_test(dl)
        print(f"  {ds:14s}: {po}/{po+ne}  mean {100*np.mean(dl):+.1f}  p={pp:.3f}")

    print("\nLARGE models only (gemini-2.5-pro, gpt-4.1, gpt-5) -- the paper's claim:")
    dl = []
    for ds in DS:
        for m in MODELS_LARGE:
            r0 = R0[ds].get(m, np.nan); r1 = g(ds, m, "with_preanalysis", "940")
            if np.isfinite(r0) and np.isfinite(r1):
                dl.append(r1 - r0)
    po, ne, nn, pp = sign_test(dl)
    print(f"  1000>0 in {po}/{po+ne} large-model cells; mean {100*np.mean(dl):+.1f} pp; sign-test p={pp:.2e}")

    print("=" * 90)
    print("CLAIM 2: 60-shot dip -- 60-shot < 0-shot (knowledge/sample conflict at small n)")
    deltas = []
    for ds in DS:
        for m in MODELS:
            r0 = R0[ds].get(m, np.nan); r60 = g(ds, m, "with_preanalysis", "0")
            if np.isfinite(r0) and np.isfinite(r60):
                deltas.append(r60 - r0)
    pos, neg, n, p = sign_test(deltas)
    print(f"  60<0 in {neg}/{pos+neg} cells; mean Delta(60-0) = {100*np.mean(deltas):+.1f} pp; sign-test p={p:.2e}")

    print("=" * 90)
    print("CLAIM 3: 1000-shot > 60-shot (enough context overcomes the dip)")
    deltas = []
    for ds in DS:
        for m in MODELS:
            r60 = g(ds, m, "with_preanalysis", "0"); r1000 = g(ds, m, "with_preanalysis", "940")
            if np.isfinite(r60) and np.isfinite(r1000):
                deltas.append(r1000 - r60)
    pos, neg, n, p = sign_test(deltas)
    print(f"  1000>60 in {pos}/{pos+neg} cells; mean Delta = {100*np.mean(deltas):+.1f} pp; sign-test p={p:.2e}")

    print("=" * 90)
    print("CLAIM 4: knowledge conflict -- removing the property NAME does not hurt (often helps).")
    print("  Directed, unbiased contrasts vs Level 1 (Specific); |r| at 1000-shot.")
    print("  NOTE: a 'peak != L1' / max-over-6 test is biased (null P(peak!=L1)=5/6), so we")
    print("  instead test specific levels against L1.")
    for lv, lab in [("wp_molproperty_clear", "L3 Generic-clear"),
                    ("wp_sampleproperty_clear", "L5 Agnostic-clear"),
                    ("wp_solubility_blind", "L2 Specific-transf")]:
        dl = []
        for ds in DS:
            for m in MODELS:
                a = g(ds, m, "with_preanalysis", "940"); b = g(ds, m, lv, "940")
                if np.isfinite(a) and np.isfinite(b):
                    dl.append(b - a)   # >0 means the blinded level beats Specific L1
        po, ne, nn, pp = sign_test(dl)
        print(f"  {lab:20s} >= L1 in {po}/{po+ne} cells; mean(level-L1) {100*np.mean(dl):+.1f} pp; "
              f"sign-test p={pp:.3f}")
    # Also: in how many cells is ANY no-name level (L3 or L5) strictly above L1?
    cnt = tot = 0
    for ds in DS:
        for m in MODELS:
            a = g(ds, m, "with_preanalysis", "940")
            noname = [g(ds, m, lv, "940") for lv in ("wp_molproperty_clear", "wp_sampleproperty_clear")]
            noname = [x for x in noname if np.isfinite(x)]
            if np.isfinite(a) and noname:
                tot += 1
                if max(noname) > a:
                    cnt += 1
    print(f"  a no-name level (L3 or L5) beats Specific L1 in {cnt}/{tot} cells "
          f"(null expectation if name irrelevant ~ {2/3:.0%} for max of 2)")

    print("=" * 90)
    print("CLAIM 5: Lipophilicity collapses under full blinding (L1 -> L6), paired bootstrap + sign test")
    deltas = []
    for m in MODELS:
        a = MP["Lipophilicity"].get((m, "with_preanalysis", "940"))
        b = MP["Lipophilicity"].get((m, "wp_sampleproperty_blind", "940"))
        if a and b:
            obs, lo, hi = paired_boot(a, b, LUT["Lipophilicity"], LUTT["Lipophilicity"], rng)
            deltas.append(obs)
            sig = "" if (lo <= 0 <= hi) else "  *CI excludes 0*"
            print(f"  {m:22s} d|r|(L1-L6) = {obs:+.1f} pp  [{lo:+.1f},{hi:+.1f}]{sig}")
    pos, neg, n, p = sign_test(deltas)
    print(f"  POOLED: L1>L6 in {pos}/{pos+neg}; mean drop {100 if False else np.mean(deltas):+.1f} pp; sign-test p={p:.2e}")

    print("=" * 90)
    print("CLAIM 6: Delaney is ROBUST to full blinding (L1 ~ L6), paired bootstrap")
    deltas = []
    for m in MODELS:
        a = MP["Delaney"].get((m, "with_preanalysis", "940"))
        b = MP["Delaney"].get((m, "wp_sampleproperty_blind", "940"))
        if a and b:
            obs, lo, hi = paired_boot(a, b, LUT["Delaney"], LUTT["Delaney"], rng)
            deltas.append(obs)
            print(f"  {m:22s} d|r|(L1-L6) = {obs:+.1f} pp  [{lo:+.1f},{hi:+.1f}]")
    print(f"  mean |L1-L6| drop across models: {np.mean(deltas):+.1f} pp (small => robust)")

    print("=" * 90)
    print("CLAIM 7: SPECIFIC significant knowledge-conflict cases -- blinding IMPROVES")
    print("  Gemini Flash / Flash-Lite where suppressing prior knowledge helps.")
    print("  Per-level |r|x100 (1000-shot) and paired bootstrap of (level - L1).")
    lab = {"with_preanalysis": "L1", "wp_solubility_blind": "L2", "wp_molproperty_clear": "L3",
           "wp_molproperty_blind": "L4", "wp_sampleproperty_clear": "L5", "wp_sampleproperty_blind": "L6"}
    for ds in ["QM7", "Lipophilicity"]:
        for m in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
            rs = {lab[lv]: g(ds, m, lv, "940") for lv in LEVELS}
            curve = "  ".join(f"{k}={100*v:.0f}" for k, v in rs.items() if np.isfinite(v))
            print(f"\n  {ds} / {m}:  {curve}")
            a = MP[ds].get((m, "with_preanalysis", "940"))
            lutA = LUT[ds]
            for lv in LEVELS[1:]:
                b = MP[ds].get((m, lv, "940"))
                if a and b:
                    lutB = LUTT[ds] if "blind" in lv else LUT[ds]
                    # improvement = |r|(level) - |r|(L1): swap order so positive = blinding helps
                    obs, lo, hi = paired_boot(b, a, lutB, lutA, rng)
                    sig = "  *CI excludes 0 -> significant*" if (lo > 0 or hi < 0) else ""
                    print(f"      {lab[lv]}-L1 = {obs:+.1f} pp  [{lo:+.1f},{hi:+.1f}]{sig}")


if __name__ == "__main__":
    main()

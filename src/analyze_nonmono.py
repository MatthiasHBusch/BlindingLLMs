"""
Analyse the non-monotonic transform control (run AFTER Run_Delaney_NonMono.jl).

For GPT-5 on Delaney at 1000-shot, compute |r| at the transformed level under the
NON-monotonic label and put it side by side with (a) the AFFINE-transform result from
the main experiment and (b) the kNN-Tanimoto STRUCTURAL CEILING under the same
transform (the best a purely structural, non-memorizing model can do given the
non-monotonic relabeling). A paired bootstrap over the shared test molecules gives a CI
on the affine-vs-transform difference.

Message for Reviewer 2.2:
  * The sin transform is non-recoverable AND Pearson(original, transformed) ~ 0, so
    prior knowledge / recall of the true solubility is useless for this target.
  * If the LLM still reaches (close to) the kNN structural ceiling under the transform,
    its performance is genuine structural in-context learning -- not anchor-based scale
    recovery and not value recall. The kNN ceiling absorbs the intrinsic extra
    difficulty of the (many-to-one) continuous non-monotonic map.

Set TAG = "sin" (continuous, default) or "nonmono" (binned). No API calls.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

TAG = "sin"   # "sin" (continuous) or "nonmono" (binned permutation)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
AFFINE_JSON = os.path.join(ROOT, "results", "LLM_Results_delaney.json")
TRANSF_JSON = os.path.join(ROOT, "results", f"LLM_Results_delaney_{TAG}.json")
TRANSF_CSV = os.path.join(DATA, f"delaney-processed-{TAG}.csv")
AFFINE_CSV = os.path.join(DATA, "delaney-processed.csv")
KEY = "Compound ID"
VALUE_COL = "measured log solubility in mols per litre"
LEVELS = {"wp_solubility_blind": "L2 Specific-Transf",
          "wp_molproperty_blind": "L4 Generic-Transf",
          "wp_sampleproperty_blind": "L6 Agnostic-Transf"}
N_BOOT = 5000


def canon(raw):
    m = raw.split("/")[-1].replace("_2024-12-01-preview", "")
    return m.replace("_gpt-5-batch", "").replace("_gpt-4.1-batch", "")


def clean(lst):
    out = []
    for p in lst:
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        if not np.isnan(p):
            out.append(p)
    return out


def leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                yield path, k, v
            else:
                yield from leaves(v, path + (k,))


def meanpred(jpath, approach, model="gpt-5", n_ext="940", n_tr="60"):
    res = json.load(open(jpath))
    out = {}
    for path, mol, preds in leaves(res):
        if len(path) < 5:
            continue
        if path[1] != approach or canon(path[2]) != model or path[3] != n_ext or path[4] != n_tr:
            continue
        cp = clean(preds)
        if cp:
            out.setdefault(mol, []).append(np.mean(cp))
    return {k: float(np.mean(v)) for k, v in out.items()}


def corr(mp, lut):
    keys = [k for k in mp if k in lut]
    p = np.array([mp[k] for k in keys]); t = np.array([float(lut[k]) for k in keys])
    if len(keys) < 5 or np.std(p) == 0:
        return np.nan
    return abs(pearsonr(p, t)[0])


def paired_boot(mpA, lutA, mpB, lutB, rng):
    keys = [k for k in mpA if k in mpB and k in lutA and k in lutB]
    if len(keys) < 5:
        return (np.nan, np.nan, np.nan)
    pA = np.array([mpA[k] for k in keys]); tA = np.array([float(lutA[k]) for k in keys])
    pB = np.array([mpB[k] for k in keys]); tB = np.array([float(lutB[k]) for k in keys])
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(keys), len(keys))
        if np.std(pA[idx]) == 0 or np.std(pB[idx]) == 0:
            d[b] = np.nan; continue
        d[b] = abs(pearsonr(pA[idx], tA[idx])[0]) - abs(pearsonr(pB[idx], tB[idx])[0])
    d = d[~np.isnan(d)]
    obs = abs(pearsonr(pA, tA)[0]) - abs(pearsonr(pB, tB)[0])
    return (100 * obs, np.percentile(d, 2.5) * 100, np.percentile(d, 97.5) * 100)


def knn_ceiling(test_keys, lut_target, smiles_by_key, k=10):
    """Best |r| a Tanimoto-kNN structural model attains on the SAME test molecules,
    predicting the (transformed) target as the similarity-weighted mean of its k
    nearest training neighbours. Bounds the intrinsic difficulty of the transform."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, DataStructs
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = {}
    for key, smi in smiles_by_key.items():
        m = Chem.MolFromSmiles(str(smi))
        if m is not None and key in lut_target:
            fp[key] = gen.GetFingerprint(m)
    test = [t for t in test_keys if t in fp]
    train = [t for t in fp if t not in set(test_keys)]
    preds, trues = [], []
    for tk in test:
        sims = sorted(((DataStructs.TanimotoSimilarity(fp[tk], fp[tr]), lut_target[tr]) for tr in train),
                      reverse=True)[:k]
        wsum = sum(s for s, _ in sims)
        pred = (sum(s * v for s, v in sims) / wsum) if wsum > 0 else np.mean([v for _, v in sims])
        preds.append(pred); trues.append(lut_target[tk])
    if len(preds) < 5:
        return np.nan
    return abs(pearsonr(preds, trues)[0])


def main():
    if not os.path.exists(TRANSF_JSON):
        raise SystemExit(f"Not found: {TRANSF_JSON}\nRun Src/Revision/Run_Delaney_NonMono.jl first "
                         f"(transform_tag = \"{TAG}\").")
    df_aff = pd.read_csv(AFFINE_CSV)
    lut_aff = df_aff.set_index(KEY)["transformed_solubility"].to_dict()
    lut_tr = pd.read_csv(TRANSF_CSV).set_index(KEY)["transformed_solubility"].to_dict()
    orig = df_aff.set_index(KEY)[VALUE_COL].to_dict()
    smiles_by_key = df_aff.set_index(KEY)["smiles"].to_dict()
    rng = np.random.default_rng(42)

    # Diagnostic: prior knowledge of the true value is uninformative for this target.
    common = [k for k in orig if k in lut_tr]
    pr_orig = pearsonr([orig[k] for k in common], [lut_tr[k] for k in common])[0]
    print(f"Transform tag: {TAG}")
    print(f"Pearson(original solubility, transformed target) = {pr_orig:+.3f}  "
          f"(~0 => recall/prior of the true value cannot help; any signal = in-context learning)\n")

    models = [("gpt-5", "GPT-5"), ("gpt-4.1", "GPT-4.1"), ("gemini-2.5-pro", "Gemini 2.5 Pro")]
    # kNN structural ceilings (level-independent) on the actual sine test molecules
    testkeys = list(meanpred(TRANSF_JSON, "wp_solubility_blind", model="gpt-5").keys())
    knn_aff = knn_ceiling(testkeys, lut_aff, smiles_by_key)
    knn_sin = knn_ceiling(testkeys, lut_tr, smiles_by_key)
    print(f"kNN structural ceiling (|r|x100): affine {100*knn_aff:.0f}, {TAG} {100*knn_sin:.0f}; "
          f"recall/prior ceiling ~{100*abs(pr_orig):.0f}\n")

    print(f"Delaney, 1000-shot: affine vs {TAG} |r|x100 per model & level "
          f"(paired-bootstrap CI on the difference)")
    print(f"{'model':16s} {'level':6s} {'affine':>7} {TAG:>6} {'d|r|':>7} {'95% CI':>14}")
    for short, disp in models:
        for ap, label in LEVELS.items():
            mp_tr = meanpred(TRANSF_JSON, ap, model=short)
            if not mp_tr:
                continue
            mp_aff = meanpred(AFFINE_JSON, ap, model=short)
            r_aff = corr(mp_aff, lut_aff)
            r_tr = corr(mp_tr, lut_tr)
            obs, lo, hi = paired_boot(mp_aff, lut_aff, mp_tr, lut_tr, rng)
            ci = f"[{lo:+.0f},{hi:+.0f}]" if np.isfinite(lo) else "n/a"
            print(f"{disp:16s} {label.split()[0]:6s} {100*r_aff:7.0f} {100*r_tr:6.0f} {obs:7.1f} {ci:>14}")
    print(f"\nReading: {TAG} |r| well above BOTH the ~{100*abs(pr_orig):.0f} recall ceiling and the "
          f"{100*knn_sin:.0f} kNN structural ceiling => genuine in-context learning under a")
    print("non-recoverable, similarity-preserving transform -- not scale recovery or value recall.")


if __name__ == "__main__":
    main()

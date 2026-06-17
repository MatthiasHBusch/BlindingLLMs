"""
Trivial-feature regression baselines (JCIM revision, Reviewer 1).

R1: "How well does linear/ridge regression with element counts (or character
counts) as features perform with the 60 or 1000 training examples - does it, by
chance, outperform the LLMs?" and "atomization energies from QM7 are very
strongly correlated with the number of heavy atoms."

We fit Ridge regression on two deliberately trivial feature sets:
  (a) element counts  - number of each chemical element parsed from the SMILES,
  (b) character counts - bag-of-characters of the raw SMILES string,
on the SAME 150-molecule test split used for the LLMs, at 60 and 1000 training
examples. We also report the single-feature correlation of the QM7 target with
the heavy-atom count. Pearson r (x100) is reported for comparability with the paper.

No API calls. Local CSVs + the result-JSON test keys only.
"""

import json
import os
import re
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")
SEED = 42
N_TRAIN = [60, 1000]

DATASETS = {
    "Delaney": dict(results="LLM_Results_delaney.json", csv="delaney-processed.csv",
                    value_col="measured log solubility in mols per litre",
                    key_col="Compound ID", key_is="name", smiles_col="smiles"),
    "Lipophilicity": dict(results="LLM_Results_lipophilicity.json", csv="Lipophilicity.csv",
                          value_col="exp", key_col="smiles", key_is="smiles", smiles_col="smiles"),
    "QM7": dict(results="LLM_Results_qm7.json", csv="qm7.csv",
                value_col="u0_atom", key_col="smiles", key_is="smiles", smiles_col="smiles"),
}

ELEMENT_RE = re.compile(r"Cl|Br|[BCNOFPSIcnops]")


def find_test_leaf(obj):
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and all(isinstance(v, list) for v in vals):
            return obj
        for v in obj.values():
            r = find_test_leaf(v)
            if r is not None:
                return r
    return None


def element_counts(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return Counter()
    return Counter(a.GetSymbol() for a in m.GetAtoms())


def heavy_atom_count(smiles):
    m = Chem.MolFromSmiles(smiles)
    return m.GetNumHeavyAtoms() if m else np.nan


def char_counts(smiles):
    return Counter(smiles)


def build_matrix(smiles_list, featfn, vocab):
    X = np.zeros((len(smiles_list), len(vocab)))
    for i, s in enumerate(smiles_list):
        c = featfn(s)
        for j, key in enumerate(vocab):
            X[i, j] = c.get(key, 0)
    return X


def ridge_r(train_s, train_y, test_s, test_y, featfn):
    vocab = sorted({k for s in train_s for k in featfn(s)})
    Xtr = build_matrix(train_s, featfn, vocab)
    Xte = build_matrix(test_s, featfn, vocab)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    model = Ridge(alpha=1.0).fit(Xtr, train_y)
    pred = model.predict(Xte)
    if np.std(pred) == 0:
        return float("nan")
    return pearsonr(pred, test_y)[0] * 100


def run_dataset(name, cfg, rng):
    df = pd.read_csv(os.path.join(DATA_DIR, cfg["csv"])).dropna(subset=[cfg["smiles_col"], cfg["value_col"]])
    with open(os.path.join(RESULTS_DIR, cfg["results"])) as f:
        results = json.load(f)
    test_keys = list(find_test_leaf(results).keys())

    if cfg["key_is"] == "name":
        lut = df.set_index(cfg["key_col"])
        test = [(lut.loc[k, cfg["smiles_col"]], float(lut.loc[k, cfg["value_col"]]))
                for k in test_keys if k in lut.index]
    else:
        lut = df.set_index(cfg["key_col"])
        test = [(k, float(lut.loc[k, cfg["value_col"]])) for k in test_keys if k in lut.index]

    test_s = [s for s, _ in test]
    test_y = np.array([y for _, y in test])
    test_set = set(test_s)
    pool = df[~df[cfg["smiles_col"]].isin(test_set)].drop_duplicates(subset=[cfg["smiles_col"]])
    pool_s = pool[cfg["smiles_col"]].tolist()
    pool_y = pool[cfg["value_col"]].astype(float).to_numpy()

    print(f"\n=== {name} ===  test={len(test_s)} pool={len(pool_s)}")

    # QM7-style single-feature check: target vs heavy-atom count (on the test set)
    hac = np.array([heavy_atom_count(s) for s in test_s])
    ok = ~np.isnan(hac)
    r_hac = pearsonr(hac[ok], test_y[ok])[0] * 100
    print(f"  r(target, heavy-atom count) on test set: {r_hac:+.1f}")

    out = {"r_heavy_atom_count": round(r_hac, 1)}
    for n in N_TRAIN:
        nn = min(n, len(pool_s))
        sel = rng.choice(len(pool_s), size=nn, replace=False)
        tr_s = [pool_s[i] for i in sel]
        tr_y = pool_y[sel]
        r_elem = ridge_r(tr_s, tr_y, test_s, test_y, element_counts)
        r_char = ridge_r(tr_s, tr_y, test_s, test_y, char_counts)
        print(f"  n_train={nn:5d}:  Ridge(element-counts) r={r_elem:+.1f}   Ridge(char-counts) r={r_char:+.1f}")
        out[f"ridge_element_{n}"] = round(r_elem, 1)
        out[f"ridge_char_{n}"] = round(r_char, 1)
    return {name: out}


def main():
    rng = np.random.default_rng(SEED)
    res = {}
    for name, cfg in DATASETS.items():
        res.update(run_dataset(name, cfg, rng))
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "trivial_regression_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved trivial_regression_results.json")


if __name__ == "__main__":
    main()

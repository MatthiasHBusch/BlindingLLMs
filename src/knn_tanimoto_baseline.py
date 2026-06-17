"""
kNN-Tanimoto similarity-weighted-mean baseline (JCIM revision).

Classical analog of the prompted LLM algorithm (the prompt literally asks the
model to find similar training molecules, weight by similarity, and average).
For each dataset we use the EXACT 150-molecule test set that the LLM was
evaluated on (the keys of the results JSON), build Morgan fingerprints, and
predict the Tanimoto-similarity-weighted mean of the k nearest TRAINING
neighbours' labels. We sweep k and report Pearson r.

Purpose in the rebuttal:
  * Establishes the achievable correlation from pure structural interpolation
    (the "general in-context learning" ceiling).
  * Shows that LLM performance under blinding (esp. level 2 specific-transformed)
    is consistent with similarity-weighted interpolation -> genuine ICL, not
    memorization.
  * Pearson r is invariant to the monotonic value transform, so this ceiling is
    identical for transformed and untransformed targets (rebuttal aid for R2-(2)).

No API calls. Reads only local CSVs + results JSONs.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")

K_VALUES = [1, 3, 5, 10, 20, 40, 60]
N_TRAIN_SETTINGS = [60, 1000]
N_BITS = 2048
RADIUS = 2
SEED = 42

DATASETS = {
    "Delaney": {
        "results": "LLM_Results_delaney.json",
        "csv": "delaney-processed.csv",
        "smiles_col": "smiles",
        "value_col": "measured log solubility in mols per litre",
        "name_col": "Compound ID",      # test keys are compound names
        "key_is": "name",
    },
    "Lipophilicity": {
        "results": "LLM_Results_lipophilicity.json",
        "csv": "Lipophilicity.csv",
        "smiles_col": "smiles",
        "value_col": "exp",
        "name_col": None,               # test keys are SMILES
        "key_is": "smiles",
    },
    "QM7": {
        "results": "LLM_Results_qm7.json",
        "csv": "qm7.csv",
        "smiles_col": "smiles",
        "value_col": "u0_atom",
        "name_col": None,
        "key_is": "smiles",
    },
}


def find_test_leaf(obj):
    """Return the first dict whose values are all lists (name/smiles -> preds)."""
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and all(isinstance(v, list) for v in vals):
            return obj
        for v in obj.values():
            r = find_test_leaf(v)
            if r is not None:
                return r
    return None


def get_test_keys(results_path):
    with open(results_path) as f:
        d = json.load(f)
    leaf = find_test_leaf(d)
    return list(leaf.keys())


def fp(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=N_BITS)


def knn_weighted_predict(test_fp, train_fps, train_y, k):
    sims = np.array(DataStructs.BulkTanimotoSimilarity(test_fp, train_fps))
    idx = np.argsort(sims)[::-1][:k]
    w = sims[idx]
    if w.sum() == 0:
        return float(np.mean(train_y[idx]))
    return float(np.sum(w * train_y[idx]) / np.sum(w))


def run_dataset(name, cfg):
    df = pd.read_csv(os.path.join(DATA_DIR, cfg["csv"]))
    smiles_col, value_col = cfg["smiles_col"], cfg["value_col"]
    df = df.dropna(subset=[smiles_col, value_col]).copy()

    # Build test set from the exact result-JSON keys
    test_keys = get_test_keys(os.path.join(RESULTS_DIR, cfg["results"]))
    if cfg["key_is"] == "name":
        lut = df.set_index(cfg["name_col"])
        test_rows = [(lut.loc[k, smiles_col], float(lut.loc[k, value_col]))
                     for k in test_keys if k in lut.index]
    else:
        lut = df.set_index(smiles_col)
        test_rows = [(k, float(lut.loc[k, value_col]))
                     for k in test_keys if k in lut.index]

    test_smiles = [s for s, _ in test_rows]
    test_y = np.array([y for _, y in test_rows])
    test_set = set(test_smiles)

    # Training pool = everything not in the test set
    pool = df[~df[smiles_col].isin(test_set)].drop_duplicates(subset=[smiles_col])
    pool_smiles = pool[smiles_col].tolist()
    pool_y = pool[value_col].astype(float).to_numpy()

    test_fps = [fp(s) for s in test_smiles]
    pool_fps = [fp(s) for s in pool_smiles]

    valid_test = [i for i, f in enumerate(test_fps) if f is not None]
    test_fps = [test_fps[i] for i in valid_test]
    test_y = test_y[valid_test]

    valid_pool = [i for i, f in enumerate(pool_fps) if f is not None]
    pool_fps = [pool_fps[i] for i in valid_pool]
    pool_y = pool_y[valid_pool]

    print(f"\n=== {name} ===  test={len(test_fps)}  pool={len(pool_fps)}")
    rng = np.random.default_rng(SEED)
    table = {}
    for n_train in N_TRAIN_SETTINGS:
        n = min(n_train, len(pool_fps))
        sel = rng.choice(len(pool_fps), size=n, replace=False)
        tr_fps = [pool_fps[i] for i in sel]
        tr_y = pool_y[sel]
        row = {}
        for k in K_VALUES:
            kk = min(k, n)
            preds = np.array([knn_weighted_predict(tf, tr_fps, tr_y, kk) for tf in test_fps])
            r = pearsonr(preds, test_y)[0]
            row[k] = r
        table[n_train] = row

    # pretty print
    header = "k     " + "".join(f"| n_train={n:<6}" for n in N_TRAIN_SETTINGS)
    print(header)
    for k in K_VALUES:
        line = f"{k:<5} " + "".join(f"|   r={table[n][k]:+.3f}   " for n in N_TRAIN_SETTINGS)
        print(line)
    return {name: table}


def main():
    out = {}
    for name, cfg in DATASETS.items():
        out.update(run_dataset(name, cfg))
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "knn_tanimoto_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved knn_tanimoto_results.json")


if __name__ == "__main__":
    main()

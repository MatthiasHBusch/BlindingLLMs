"""
Memorization detector at 2 AND 3 significant digits (JCIM revision).

Reviewer 1 asked for a 2-significant-digit companion to the 3-sig-digit
"exact match" table (ChatGPT often reports lipophilicity to 2 sig figs).
Reviewer 2 argued the verbatim test is "too easy to pass" and misses
approximate recall. This script:

  * reproduces the paper's 3-sig-digit 0-shot match table (validation), and
  * adds the 2-sig-digit match table, together with the random-chance
    expectation for each precision so the reader can see that observed match
    rates sit at (not above) chance -> no evidence of verbatim OR coarse recall.

"High precision" for an n-sig table means the ground-truth value carries at
least n significant digits (so a match is non-trivial).

Reads RepoICML/results/LLM_Results_<ds>_zeroshot.json + Data CSVs. No API calls.
"""

import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")

# The nine model variants reported in the paper (exclude any later-added models
# such as Claude so the revision table stays consistent with the original scope).
PAPER_MODELS = {
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-5", "gpt-5-mini", "gpt-5-nano",
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
}

DATASETS = {
    "Delaney": dict(results="LLM_Results_delaney_zeroshot.json",
                    csv="delaney-processed.csv",
                    value_col="measured log solubility in mols per litre",
                    key_col="Compound ID"),
    "Lipophilicity": dict(results="LLM_Results_lipophilicity_zeroshot.json",
                          csv="Lipophilicity.csv", value_col="exp", key_col="smiles"),
    "QM7": dict(results="LLM_Results_qm7_zeroshot.json",
                csv="qm7.csv", value_col="u0_atom", key_col="smiles"),
}


def sig_round(x, n):
    if x == 0:
        return "0"
    return f"{x:.{n}g}"


def n_match(true_val, pred_val, n):
    return sig_round(true_val, n) == sig_round(pred_val, n)


def sig_figs(val):
    """Number of significant digits in the ground-truth value as stored."""
    if val == 0:
        return 1
    s = f"{val:.10g}".lstrip("-")
    s = s.replace(".", "").lstrip("0")
    return max(1, len(s.rstrip("0"))) if s else 1


def iter_mol_leaves(obj):
    """Yield (model_path, molecule_key, preds) for numeric-list leaves."""
    def rec(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                    # path[-1] is the molecule key, the rest identify the model
                    yield "/".join(path), k, v
                else:
                    yield from rec(v, path + [k])
    yield from rec(obj, [])


def short_name(model_path):
    return model_path.split("/")[-1] if "/" in model_path else model_path


def chance_rate(trues, preds, n, rng, n_perm=200):
    """Expected n-sig match rate if predictions were paired at random with
    ground-truth values (shuffle test). Captures the fact that, for clustered or
    similarly-scaled values, some matches arise purely by chance."""
    trues = np.asarray(trues)
    preds = np.asarray(preds)
    keep = np.array([sig_figs(t) >= n for t in trues])
    trues, preds = trues[keep], preds[keep]
    if len(trues) == 0:
        return 0.0
    tstr = np.array([sig_round(t, n) for t in trues])
    pstr = np.array([sig_round(p, n) for p in preds])
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(pstr))
        hits += np.sum(tstr == pstr[perm])
    return 100 * hits / (n_perm * len(tstr))


def analyse(name, cfg, rng):
    df = pd.read_csv(os.path.join(DATA_DIR, cfg["csv"])).dropna(subset=[cfg["value_col"]])
    lut = df.set_index(cfg["key_col"])[cfg["value_col"]].to_dict()
    with open(os.path.join(RESULTS_DIR, cfg["results"])) as f:
        results = json.load(f)

    pairs = {}   # model -> (trues[], preds[])
    for model, mol, preds in iter_mol_leaves(results):
        if mol not in lut or short_name(model) not in PAPER_MODELS:
            continue
        tv = float(lut[mol])
        for p in preds:
            try:
                pv = float(p)
            except (TypeError, ValueError):
                continue
            pairs.setdefault(model, ([], []))[0].append(tv)
            pairs[model][1].append(pv)

    rows = []
    for model, (trues, preds) in pairs.items():
        trues = np.asarray(trues)
        preds = np.asarray(preds)
        for n in (2, 3):
            keep = np.array([sig_figs(t) >= n for t in trues])
            tk, pk = trues[keep], preds[keep]
            tot = len(tk)
            m = int(sum(n_match(t, p, n) for t, p in zip(tk, pk)))
            obs = 100 * m / tot if tot else 0.0
            ch = chance_rate(trues, preds, n, rng)
            rows.append(dict(dataset=name, model=short_name(model), nsig=n,
                             matches=m, total=tot,
                             observed_pct=round(obs, 2), chance_pct=round(ch, 2)))
    return rows


def main():
    rng = np.random.default_rng(0)
    all_rows = []
    print("Zero-shot memorization detector: observed vs random-chance match rate\n")
    for name, cfg in DATASETS.items():
        rows = analyse(name, cfg, rng)
        all_rows.extend(rows)
        print(f"===== {name} =====")
        print(f"{'model':24s} {'2sig obs%':>9} {'2sig chance%':>12} "
              f"{'3sig obs%':>9} {'3sig chance%':>12}  3sig(m/tot)")
        bymodel = {}
        for r in rows:
            bymodel.setdefault(r["model"], {})[r["nsig"]] = r
        for model in sorted(bymodel):
            r2, r3 = bymodel[model][2], bymodel[model][3]
            print(f"{model:24s} {r2['observed_pct']:9.2f} {r2['chance_pct']:12.2f} "
                  f"{r3['observed_pct']:9.2f} {r3['chance_pct']:12.2f}  "
                  f"({r3['matches']}/{r3['total']})")
        print()
    pd.DataFrame(all_rows).to_csv(
        os.path.join(os.path.dirname(__file__), "..", "results", "memorization_sigdigits.csv"), index=False)
    print("Saved memorization_sigdigits.csv")


if __name__ == "__main__":
    main()

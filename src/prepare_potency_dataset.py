"""
Prepare the ASAP Discovery antiviral potency (Polaris) dataset for the blinding study.

This is the modern, post-LLM-cutoff control requested by Reviewer 2 (issue 1.3):
a 2025 release whose molecules were prospectively synthesised for the
ASAP-Discovery x Polaris x OpenADMET Antiviral Drug Discovery Challenge, so the
test molecules are essentially guaranteed to be *outside* the training corpus of
the evaluated LLMs (in contrast to ESOL/Lipophilicity/QM7, all pre-2018).

Source dataset (public, "unblinded" = labels released after the competition):
    asap-discovery/antiviral-potency-2025-unblinded   (https://polarishub.io)
    Columns: "Molecule Name", "CXSMILES",
             "pIC50 (SARS-CoV-2 Mpro)", "pIC50 (MERS-CoV Mpro)", "Set"
    ~1328 datapoints (sparse: not every molecule has both targets).

We keep a *single* continuous regression target, exactly like the other three
datasets (Delaney logS, Lipophilicity logD, QM7 atomization energy). The default
target is the SARS-CoV-2 Mpro pIC50 (the most populated endpoint). Higher pIC50 =
more potent inhibitor; pIC50 = -log10(IC50 / M), an *empirical* (measured)
property, like Lipophilicity.

Output (written next to the other raw datasets in ../../Data):
    antiviral_potency_original.csv   columns: smiles, pic50
Then run the Julia transform to add the blinded columns:
    julia Src/Revision/transform_potency.jl
which produces ../../Data/antiviral_potency.csv with `transformed_smiles` and
`transformed_solubility` (the generic transformed-target column name reused
across all datasets in this repo).

Setup (only needed once; the user asked NOT to run the experiment yet):
    pip install polaris-lib
    polaris login          # the unblinded dataset is public, but a free Polaris
                           # account is required for programmatic download.

Usage:
    python Src/Revision/prepare_potency_dataset.py
    python Src/Revision/prepare_potency_dataset.py --target mers   # use MERS-CoV Mpro instead
"""

import argparse
import os
import sys

import pandas as pd

# --- Configuration --------------------------------------------------------

POLARIS_DATASET_ID = "asap-discovery/antiviral-potency-2025-unblinded"
SMILES_COL = "CXSMILES"
TARGET_COLS = {
    "sars": "pIC50 (SARS-CoV-2 Mpro)",
    "mers": "pIC50 (MERS-CoV Mpro)",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
OUTPUT_FILE = os.path.join(DATA_DIR, "antiviral_potency_original.csv")


# --- Helpers --------------------------------------------------------------

def _read_column(dataset, col: str, retries: int = 6):
    """Bulk-read a whole column from the dataset's Zarr archive, with retries.

    Reading `zarr_root[col][:]` pulls the column in a handful of chunk requests,
    rather than one HTTP request per row (which overwhelms the Hub backend and
    triggers 500s). We only ever read the columns we actually use, so a broken /
    unused column (e.g. the MERS endpoint) cannot abort the run.
    """
    import time

    last_err = None
    for attempt in range(retries):
        try:
            return dataset.zarr_root[col][:]
        except Exception as e:  # transient 5xx / network hiccups
            last_err = e
            print(f"  read '{col}' attempt {attempt + 1}/{retries} failed: "
                  f"{str(e)[:120]}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not read column '{col}': {last_err}")


def load_polaris_columns(target_col: str) -> pd.DataFrame:
    """Load only the SMILES + chosen target columns from the Polaris dataset."""
    try:
        import polaris as po
    except ImportError as e:
        sys.exit(
            "The `polaris` package is not installed.\n"
            "Install it with:  pip install polaris-lib\n"
            "and authenticate with:  polaris login\n"
            f"(original import error: {e})"
        )

    print(f"Loading Polaris dataset: {POLARIS_DATASET_ID}")
    dataset = po.load_dataset(POLARIS_DATASET_ID)

    # V1 (table-backed) datasets expose a pandas table directly.
    table = getattr(dataset, "table", None)
    if table is not None:
        return pd.DataFrame(table)

    # V2 (Zarr-backed): bulk-read just the two columns we need.
    cols = list(dataset.columns)
    print(f"Dataset columns: {cols}")
    if SMILES_COL not in cols or target_col not in cols:
        sys.exit(
            f"Expected columns '{SMILES_COL}' and '{target_col}' not found.\n"
            f"Available columns: {cols}"
        )
    data = {
        SMILES_COL: _read_column(dataset, SMILES_COL),
        target_col: _read_column(dataset, target_col),
    }
    return pd.DataFrame(data)


def canonicalize_smiles(cxsmiles: str) -> str | None:
    """
    Convert a (CX)SMILES to a canonical, isomeric, single-fragment SMILES.

    CXSMILES carries an extension block (coordinates / enhanced stereo) after a
    `|...|` marker; RDKit parses it, and MolToSmiles emits a clean canonical
    SMILES without that block. We keep stereochemistry (it matters for potency)
    but reduce salts/mixtures to the largest organic fragment.
    """
    from rdkit import Chem

    if not isinstance(cxsmiles, str) or not cxsmiles.strip():
        return None
    mol = Chem.MolFromSmiles(cxsmiles)
    if mol is None:
        # Last-ditch: drop the CX extension block and retry as plain SMILES.
        mol = Chem.MolFromSmiles(cxsmiles.split("|")[0].strip())
    if mol is None:
        return None
    # Keep the largest fragment (strip counter-ions / solvents if any).
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol, isomericSmiles=True)


# --- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=list(TARGET_COLS.keys()),
        default="sars",
        help="Which pIC50 endpoint to use as the single regression target "
        "(default: sars = SARS-CoV-2 Mpro).",
    )
    args = parser.parse_args()
    target_col = TARGET_COLS[args.target]

    df = load_polaris_columns(target_col)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Keep only rows with a measured value for the chosen endpoint (data is sparse).
    sub = df[[SMILES_COL, target_col]].copy()
    sub = sub.dropna(subset=[target_col])
    print(f"{len(sub)} rows have a measured {target_col} value.")

    # Canonicalise SMILES and drop anything RDKit cannot parse.
    sub["smiles"] = sub[SMILES_COL].apply(canonicalize_smiles)
    n_before = len(sub)
    sub = sub.dropna(subset=["smiles"])
    if len(sub) < n_before:
        print(f"Dropped {n_before - len(sub)} rows with unparseable SMILES.")

    # Collapse duplicate canonical SMILES (e.g. enantiomer/stereo collisions) by
    # averaging their target value, so the test/train split cannot leak.
    sub["pic50"] = pd.to_numeric(sub[target_col], errors="coerce")
    sub = sub.dropna(subset=["pic50"])
    out = (
        sub.groupby("smiles", as_index=False)["pic50"]
        .mean()
        .reset_index(drop=True)
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    out[["smiles", "pic50"]].to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(out)} unique molecules to: {OUTPUT_FILE}")
    print(f"pic50 range: {out['pic50'].min():.2f} to {out['pic50'].max():.2f}")
    print("\nNext step:  julia Src/Revision/transform_potency.jl")


if __name__ == "__main__":
    main()

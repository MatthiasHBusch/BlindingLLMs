"""
Build the POSITIVE-CONTROL datasets for the memorization detector (JCIM revision).

Both reviewers ask us to prove the exact-match detector actually *fires* when a
value is memorized. We therefore assemble small datasets of famous, high-
contamination-likelihood numbers (periodic table / Wikipedia infoboxes) and run
the IDENTICAL 0-shot prompt + 3-significant-digit exact-match metric on them.

Two sets, deliberately spanning a contamination gradient (per R2's suggestion):
  * Set A - standard atomic weights of the elements. Near-certain verbatim recall.
    Expectation: detector match rate ~ very high (detector ceiling).
  * Set B - normal boiling points (deg C) of extremely common compounds, the
    canonical Wikipedia-infobox numbers R1/R2 named (melting/boiling points).
    Not reliably computable to 3 sig figs from a SMILES by reasoning, so a 3-sig
    match implies recall. Expectation: detector match rate substantial.

Contrast target: ESOL / Lipophilicity / QM7, where the same detector fires at
~chance. The gradient (high on A, substantial on B, ~chance on benchmarks) shows
the detector is sensitive, so its near-zero firing on the benchmarks is meaningful.

Values are standard CRC/IUPAC/Wikipedia figures; spot-check before publication.
This script ONLY writes CSVs. No network, no API.
"""

import csv
import os

OUT = os.path.dirname(__file__)

# --- Set A: standard atomic weights (IUPAC 2021 conventional values) -----------
# symbol, name, standard atomic weight (u)
ATOMIC_WEIGHTS = [
    ("H", "hydrogen", 1.008), ("He", "helium", 4.0026), ("Li", "lithium", 6.94),
    ("Be", "beryllium", 9.0122), ("B", "boron", 10.81), ("C", "carbon", 12.011),
    ("N", "nitrogen", 14.007), ("O", "oxygen", 15.999), ("F", "fluorine", 18.998),
    ("Ne", "neon", 20.180), ("Na", "sodium", 22.990), ("Mg", "magnesium", 24.305),
    ("Al", "aluminium", 26.982), ("Si", "silicon", 28.085), ("P", "phosphorus", 30.974),
    ("S", "sulfur", 32.06), ("Cl", "chlorine", 35.45), ("Ar", "argon", 39.95),
    ("K", "potassium", 39.098), ("Ca", "calcium", 40.078), ("Ti", "titanium", 47.867),
    ("Cr", "chromium", 51.996), ("Mn", "manganese", 54.938), ("Fe", "iron", 55.845),
    ("Co", "cobalt", 58.933), ("Ni", "nickel", 58.693), ("Cu", "copper", 63.546),
    ("Zn", "zinc", 65.38), ("Ga", "gallium", 69.723), ("As", "arsenic", 74.922),
    ("Se", "selenium", 78.971), ("Br", "bromine", 79.904), ("Ag", "silver", 107.868),
    ("Sn", "tin", 118.710), ("I", "iodine", 126.904), ("Ba", "barium", 137.327),
    ("Pt", "platinum", 195.084), ("Au", "gold", 196.967), ("Hg", "mercury", 200.592),
    ("Pb", "lead", 207.2), ("U", "uranium", 238.029),
]

# --- Set B: normal boiling points (deg C) of common compounds ------------------
# name, SMILES, boiling point (deg C, 1 atm)
# Values VALIDATED against the English-Wikipedia Chembox/infobox (2026-05) so the
# ground truth matches the source named in the prompt. Compounds for which
# Wikipedia lists only a RANGE are flagged below; their stored value is a
# representative point inside that range and is a weaker recall target.
# Wikipedia-range compounds: 1-propanol (97-98), acetic acid (118-119),
# n-hexane (68.5-69.1), n-pentane (35.9-36.3), acetonitrile (81.3-82.1),
# diethylamine (54.8-56.4), triethylamine (88.6-89.8), n-octane (125.1-126.1).
BOILING_POINTS = [
    ("water", "O", 99.98), ("methanol", "CO", 64.7), ("ethanol", "CCO", 78.23),
    ("1-propanol", "CCCO", 97.2), ("2-propanol", "CC(O)C", 82.6),
    ("1-butanol", "CCCCO", 117.7), ("acetone", "CC(=O)C", 56.08),
    ("benzene", "c1ccccc1", 80.1), ("toluene", "Cc1ccccc1", 110.6),
    ("acetic acid", "CC(=O)O", 118.1), ("formic acid", "C(=O)O", 100.8),
    ("diethyl ether", "CCOCC", 34.6), ("chloroform", "ClC(Cl)Cl", 61.15),
    ("dichloromethane", "ClCCl", 39.6), ("carbon tetrachloride", "ClC(Cl)(Cl)Cl", 76.72),
    ("n-hexane", "CCCCCC", 68.7), ("n-pentane", "CCCCC", 36.1),
    ("cyclohexane", "C1CCCCC1", 80.74), ("ethyl acetate", "CCOC(=O)C", 77.1),
    ("acetonitrile", "CC#N", 81.6), ("dimethyl sulfoxide", "CS(=O)C", 189.0),
    ("pyridine", "c1ccncc1", 115.2), ("aniline", "Nc1ccccc1", 184.13),
    ("phenol", "Oc1ccccc1", 181.7), ("nitrobenzene", "O=[N+]([O-])c1ccccc1", 210.9),
    ("tetrahydrofuran", "C1CCOC1", 66.0), ("carbon disulfide", "S=C=S", 46.24),
    ("formaldehyde", "C=O", -19.0), ("acetaldehyde", "CC=O", 20.2),
    ("diethylamine", "CCNCC", 55.5), ("triethylamine", "CCN(CC)CC", 89.0),
    ("ethylene glycol", "OCCO", 197.3), ("glycerol", "OCC(O)CO", 290.0),
    ("n-heptane", "CCCCCCC", 98.38), ("n-octane", "CCCCCCCC", 125.6),
    ("methyl tert-butyl ether", "COC(C)(C)C", 55.5), ("1,4-dioxane", "C1COCCO1", 101.1),
    ("propionic acid", "CCC(=O)O", 141.15), ("butanone", "CCC(=O)C", 79.64),
    ("styrene", "C=Cc1ccccc1", 145.0), ("naphthalene", "c1ccc2ccccc2c1", 218.0),
    ("dimethylformamide", "CN(C)C=O", 153.0), ("hexanol", "CCCCCCO", 157.0),
    ("benzaldehyde", "O=Cc1ccccc1", 178.1), ("anisole", "COc1ccccc1", 154.0),
]


def main():
    pathA = os.path.join(OUT, "known_atomic_weights.csv")
    with open(pathA, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "name", "atomic_weight"])
        w.writerows(ATOMIC_WEIGHTS)

    pathB = os.path.join(OUT, "known_boiling_points.csv")
    with open(pathB, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "smiles", "bp_celsius"])
        w.writerows(BOILING_POINTS)

    print(f"Wrote {len(ATOMIC_WEIGHTS)} elements   -> {pathA}")
    print(f"Wrote {len(BOILING_POINTS)} compounds  -> {pathB}")


if __name__ == "__main__":
    main()

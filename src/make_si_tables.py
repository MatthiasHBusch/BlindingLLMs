"""Generate LaTeX table rows for the SI from the computed revision results."""
import json, os
import pandas as pd

HERE = os.path.dirname(__file__)
EOL = r" \\"

def p(s): print(s)

# --- kNN table (1000-shot) ---
knn = json.load(open(os.path.join(HERE, "knn_tanimoto_results.json")))
ks = [1, 3, 5, 10, 20, 40, 60]
p("% ===== KNN TABLE rows (r x100, 1000 training mols) =====")
for ds in ["Delaney", "Lipophilicity", "QM7"]:
    row = knn[ds]["1000"]
    p(f"{ds} & " + " & ".join(f"{row[str(k)]*100:.0f}" for k in ks) + EOL)

# --- trivial regression ---
tr = json.load(open(os.path.join(HERE, "trivial_regression_results.json")))
p("\n% ===== TRIVIAL REGRESSION rows (r x100) =====")
p("% dataset & elem60 & elem1000 & char60 & char1000 & heavyAtom")
for ds in ["Delaney", "Lipophilicity", "QM7"]:
    t = tr[ds]
    p(f"{ds} & {t['ridge_element_60']:.0f} & {t['ridge_element_1000']:.0f} & "
      f"{t['ridge_char_60']:.0f} & {t['ridge_char_1000']:.0f} & {t['r_heavy_atom_count']:.0f}" + EOL)

# --- memorization 2/3 sig + chance ---
m = pd.read_csv(os.path.join(HERE, "memorization_sigdigits.csv"))
order = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
         "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
         "gpt-5", "gpt-5-mini", "gpt-5-nano"]
p("\n% ===== MEMORIZATION rows: obs2 & chance2 & obs3 & chance3 (per dataset) =====")
for ds in ["Delaney", "Lipophilicity", "QM7"]:
    p(f"% --- {ds} ---")
    sub = m[m.dataset == ds]
    for mod in order:
        r2 = sub[(sub.model == mod) & (sub.nsig == 2)].iloc[0]
        r3 = sub[(sub.model == mod) & (sub.nsig == 3)].iloc[0]
        p(f"{mod} & {r2.observed_pct:.2f} & {r2.chance_pct:.2f} & "
          f"{r3.observed_pct:.2f} & {r3.chance_pct:.2f}" + EOL)

# --- bootstrap: gemini-pro 6 levels x 3 datasets + median widths ---
b = pd.read_csv(os.path.join(HERE, "bootstrap_correlation_ci.csv"))
mp = {"with_preanalysis": "Specific (L1)", "wp_solubility_blind": "Spec-Transf (L2)",
      "wp_molproperty_clear": "Generic (L3)", "wp_molproperty_blind": "Gen-Transf (L4)",
      "wp_sampleproperty_clear": "Agnostic (L5)", "wp_sampleproperty_blind": "Agn-Transf (L6)"}
def level(cfg):
    parts = cfg.split("/")
    return mp.get(parts[1]) if len(parts) > 1 else None
p("\n% ===== BOOTSTRAP gemini-2.5-pro 1000-shot: level & dataset & r[lo,hi] (|r|) =====")
sub = b[b.config.str.contains("gemini-2.5-pro") & b.config.str.contains("/940/") &
        b.config.str.startswith("names_only")]
recs = []
for _, r in sub.iterrows():
    L = level(r.config)
    if L:
        recs.append((L, r.dataset, abs(r["r"]), abs(r.ci_lo), abs(r.ci_hi)))
for L in mp.values():
    cells = {ds: None for ds in ["delaney", "lipophilicity", "qm7"]}
    for LL, ds, rr, lo, hi in recs:
        if LL == L:
            lo2, hi2 = sorted([lo, hi])
            cells[ds] = f"{rr:.0f} [{lo2:.0f},{hi2:.0f}]"
    p(f"{L} & {cells['delaney']} & {cells['lipophilicity']} & {cells['qm7']}" + EOL)
p("% median CI width by dataset: " + str(b.groupby("dataset")["ci_width"].median().round(1).to_dict()))
p("% overall median CI width: %.1f" % b["ci_width"].median())

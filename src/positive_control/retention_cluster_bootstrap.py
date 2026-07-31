"""
Cluster-bootstrap confidence intervals for the retention ratio R21.

The ten repetitions of the sweep run over the SAME 41--45 species, so the m1 events
are not independent Bernoulli trials: repetitions reduce noise in the models' answers
but carry no additional information about which species were sampled. A Wilson
interval over the pooled events therefore understates the uncertainty. Here the
resampling unit is the species, so the interval reflects the sample of compounds
rather than the number of queries.

Writes the table body used for SI Table S5.

Usage:
    python retention_cluster_bootstrap.py
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "known_boiling_points_blinded.csv")
SWEEP = os.path.join(HERE, "PositiveControl_Blinding_Direct.json")
ZEROSHOT = os.path.join(HERE, "PositiveControl_BoilingPoints.json")
N_BOOT = 4000

LEVELS = [("io_specific_clear", "1 Spec.", False),
          ("io_specific_blind", "2 Spec.-T", True),
          ("io_molproperty_clear", "3 Gen.", False),
          ("io_molproperty_blind", "4 Gen.-T", True),
          ("io_sampleproperty_clear", "5 Agn.", False),
          ("io_sampleproperty_blind", "6 Agn.-T", True)]

SHORT = {"google/gemini-2.5-pro": "Gemini 2.5 Pro",
         "google/gemini-2.5-flash": "Gemini 2.5 Flash",
         "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
         "openai/gpt-5": "GPT-5", "openai/gpt-5-mini": "GPT-5 mini",
         "openai/gpt-5-nano": "GPT-5 nano", "openai/gpt-4.1": "GPT-4.1",
         "openai/gpt-4.1-mini": "GPT-4.1 mini", "openai/gpt-4.1-nano": "GPT-4.1 nano"}
ORDER = list(SHORT.values())


def sig_round(x, n):
    return "0" if x == 0 else f"{x:.{n}g}"


def sig_figs(v):
    if v == 0:
        return 1
    s = f"{v:.10g}".lstrip("-").replace(".", "").lstrip("0")
    return max(1, len(s.rstrip("0"))) if s else 1


def second_digit(x):
    if x == 0 or not np.isfinite(x):
        return None
    s = f"{abs(x):.10g}".replace(".", "").lstrip("0")
    return int(s[1]) if len(s) >= 2 else None


def u_collision(values):
    ds = [d for d in (second_digit(v) for v in values) if d is not None]
    c = np.bincount(ds, minlength=10)
    n = len(ds)
    return 100 * float((c * (c - 1)).sum() / (n * (n - 1)))


def per_species(node, lut):
    """species -> (m1, m2) counts summed over its repetitions."""
    out = {}
    for name, raw in node.items():
        t = lut.get(name)
        if t is None:
            continue
        lo = hi = 0
        for v in (raw if isinstance(raw, list) else [raw]):
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v) or sig_figs(t) < 2 or sig_figs(v) < 2:
                continue
            if sig_round(v, 1) == sig_round(t, 1):
                lo += 1
                if sig_round(v, 2) == sig_round(t, 2):
                    hi += 1
        out[name] = (lo, hi)
    return out


def boot_ci(counts, rng, n_boot=N_BOOT):
    """Percentile CI for m2/m1 resampling species (the clusters) with replacement."""
    names = list(counts)
    m1 = np.array([counts[n][0] for n in names], float)
    m2 = np.array([counts[n][1] for n in names], float)
    tot1, tot2 = m1.sum(), m2.sum()
    if tot1 == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(names), size=(n_boot, len(names)))
    a, b = m1[idx].sum(1), m2[idx].sum(1)
    ok = a > 0
    r = 100 * b[ok] / a[ok]
    return 100 * tot2 / tot1, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    rng = np.random.default_rng(0)
    df = pd.read_csv(DATA)
    truth = df.set_index("name")["bp_celsius"].to_dict()
    trans = df.set_index("name")["transformed_solubility"].to_dict()
    floor = {False: u_collision(df.bp_celsius.to_numpy()),
             True: u_collision(df.transformed_solubility.to_numpy())}
    res = json.load(open(SWEEP))["names_only"]

    print(f"floors (U-statistic): untransformed {floor[False]:.1f}%, "
          f"transformed {floor[True]:.1f}%")
    print(f"cluster bootstrap over species, {N_BOOT} resamples\n")

    table, pooled = {}, {}
    for approach, label, is_t in LEVELS:
        lut = trans if is_t else truth
        allc = {}
        for model, mv in res[approach].items():
            node = mv["0"][list(mv["0"].keys())[0]]
            c = per_species(node, lut)
            r, lo, hi = boot_ci(c, rng)
            table.setdefault(SHORT.get(model, model), {})[label] = (r, lo, hi,
                                                                    lo > floor[is_t],
                                                                    hi < floor[is_t])
            for k, v in c.items():
                p = allc.get(k, (0, 0))
                allc[k] = (p[0] + v[0], p[1] + v[1])
        r, lo, hi = boot_ci(allc, rng)
        pooled[label] = (r, lo, hi, lo > floor[is_t], hi < floor[is_t], floor[is_t])

    for _, label, _ in LEVELS:
        r, lo, hi, up, dn, fl = pooled[label]
        mark = "above floor" if up else ("BELOW floor" if dn else "indistinguishable")
        n_up = sum(1 for m in ORDER if table[m][label][3])
        n_dn = sum(1 for m in ORDER if table[m][label][4])
        print(f"  {label:10s} pooled R21={r:5.1f} [{lo:4.1f},{hi:5.1f}]  {mark:18s}"
              f"  per model: {n_up} above / {n_dn} below / {9-n_up-n_dn} n.s.")

    def cell(v):
        r, lo, hi, up, dn = v
        m = r"$^{*}$" if up else (r"$^{\dagger}$" if dn else "")
        return f"{r:.0f}{m} {{\\scriptsize[{lo:.0f},{hi:.0f}]}}"

    # 0-shot retention per model, for the correlation column
    zs = json.load(open(ZEROSHOT))
    smiles2name = dict(zip(df.smiles, df.name))
    def walk(o, path=()):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from walk(v, path + (k,))
        else:
            yield path, o
    acc = {}
    for path, vals in walk(zs):
        if not isinstance(vals, list):
            continue
        model = next((p for p in path if "/" in p or "gpt" in p), None)
        nm = smiles2name.get(path[-1])
        if model is None or nm is None:
            continue
        acc.setdefault(model, {}).setdefault(nm, []).extend(vals)
    zs_r21 = {}
    for m, d in acc.items():
        node = {k: v for k, v in d.items()}
        c = per_species(node, truth)
        t1 = sum(v[0] for v in c.values())
        t2 = sum(v[1] for v in c.values())
        zs_r21[SHORT.get(m, m)] = 100 * t2 / t1 if t1 else float("nan")

    lines = ["% Generated by retention_cluster_bootstrap.py -- do not edit by hand."]
    for m in ORDER:
        z = zs_r21.get(m, float("nan"))
        lines.append(f"{m} & " + " & ".join(cell(table[m][lab]) for _, lab, _ in LEVELS)
                     + (f" & {z:.0f}" if np.isfinite(z) else " & --") + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\textbf{Pooled} & " +
                 " & ".join(cell(pooled[lab][:5]) for _, lab, _ in LEVELS) + r" & \\")
    lines.append(r"\textit{floor $C_2$} & " +
                 " & ".join(f"{pooled[lab][5]:.1f}" for _, lab, _ in LEVELS) + r" & \\")
    out = os.path.join(HERE, "retention_si_table.tex")
    open(out, "w", encoding="utf8", newline="\n").write("\n".join(lines) + "\n")
    print(f"\nwrote {os.path.basename(out)}")


if __name__ == "__main__":
    main()

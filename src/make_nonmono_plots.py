"""
Plots for the non-monotonic (sine) transform control.

(1) sin_curve.png         -- the transform itself: (sin(4*pi*u)+1)/2*100 over u in [0,1].
(2) nonmono_bars_affine.png / nonmono_bars_sin.png -- grouped bar plots of GPT-5 / GPT-4.1 /
    Gemini 2.5 Pro |r| at the three transformed levels (L2/L4/L6), one plot for the affine
    ("normal") transform and one for the sine transform, each with the kNN structural ceiling
    (dashed) and -- on the sine plot -- the recall/prior ceiling (dotted). Model colors match
    the paper's family palette (get_llm_color_palette).

The bar plots use whatever models are present in the results JSONs (the GPT-4.1 / Gemini run
may still be in progress); re-run when complete. No API calls.
"""

import colorsys
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_nonmono as A   # reuse meanpred/corr/knn_ceiling/paths

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

# Paper family palette (matches plotQM7LipophilicityResults.get_llm_color_palette, "normal" size)
FAMILY = {"gpt-4.1": (210, 0.7), "gpt-5": (120, 0.7), "gemini-2.5": (30, 0.7)}
def fam_color(family, lightness=0.35):
    h, s = FAMILY[family]
    r, g, b = colorsys.hls_to_rgb(h / 360.0, lightness, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

MODELS = [("gpt-5", "gpt-5", "GPT-5"),
          ("gpt-4.1", "gpt-4.1", "GPT-4.1"),
          ("gemini-2.5-pro", "gemini-2.5", "Gemini 2.5 Pro")]
LEVELS = [("wp_solubility_blind", "L2\nSpecific-T"),
          ("wp_molproperty_blind", "L4\nGeneric-T"),
          ("wp_sampleproperty_blind", "L6\nAgnostic-T")]


def sin_curve():
    u = np.linspace(0, 1, 1000)
    t = (np.sin(4 * np.pi * u) + 1) / 2 * 100
    fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=300)
    ax.plot(u, t, color="#7159b8", lw=2)
    ax.set_xlim(0, 1); ax.set_ylim(-2, 102)
    ax.set_xlabel(r"normalized label  $u=(y-y_{\min})/(y_{\max}-y_{\min})$")
    ax.set_ylabel("transformed value")
    ax.set_title(r"Non-monotonic sine transform: $\tilde{y}=\frac{1}{2}(\sin(4\pi u)+1)\times100$",
                 fontsize=10.5)
    for x in (0.25, 0.5, 0.75):
        ax.axvline(x, color="0.85", lw=0.8, zorder=0)
    # secondary axis in units of theta = 4*pi*u
    sec = ax.secondary_xaxis("top", functions=(lambda x: 4 * x, lambda x: x / 4))
    sec.set_xlabel(r"$\theta/\pi$  (two full periods)")
    fig.tight_layout()
    p = os.path.join(OUT, "sin_curve.png")
    fig.savefig(p); plt.close(fig)
    return p


def collect(jpath, lut):
    """{model_short: {level_key: |r| or nan}} for the flagship models present."""
    out = {}
    for short, _fam, _lab in MODELS:
        out[short] = {}
        for lvl, _ in LEVELS:
            mp = A.meanpred(jpath, lvl, model=short)
            out[short][lvl] = A.corr(mp, lut) if mp else float("nan")
    return out


def bar_plot(data, ceiling, recall, title, fname):
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=300)
    nlev, nmod = len(LEVELS), len(MODELS)
    width = 0.78 / nmod
    x = np.arange(nlev)
    for j, (short, fam, lab) in enumerate(MODELS):
        vals = [100 * data[short][lvl] if np.isfinite(data[short][lvl]) else 0 for lvl, _ in LEVELS]
        ax.bar(x + (j - (nmod - 1) / 2) * width, vals, width, color=fam_color(fam), label=lab,
               edgecolor="white", linewidth=0.5)
    ax.axhline(100 * ceiling, ls="--", color="0.35", lw=1.3, label=f"kNN ceiling ({100*ceiling:.0f})")
    if recall is not None:
        ax.axhline(100 * recall, ls=":", color="0.55", lw=1.2, label=f"recall ceiling ({100*recall:.0f})")
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in LEVELS])
    ax.set_ylim(0, 100); ax.set_ylabel(r"$|r|\times100$")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    p = os.path.join(OUT, fname); fig.savefig(p); plt.close(fig)
    return p


def main():
    paths = [sin_curve()]

    lut_aff = pd.read_csv(A.AFFINE_CSV).set_index(A.KEY)["transformed_solubility"].to_dict()
    lut_sin = pd.read_csv(A.TRANSF_CSV).set_index(A.KEY)["transformed_solubility"].to_dict()
    smiles = pd.read_csv(A.AFFINE_CSV).set_index(A.KEY)["smiles"].to_dict()

    aff = collect(A.AFFINE_JSON, lut_aff)
    sin = collect(A.TRANSF_JSON, lut_sin)

    # kNN ceilings (level-independent) on the actual sine test molecules
    testkeys = list(A.meanpred(A.TRANSF_JSON, "wp_solubility_blind", model="gpt-5").keys())
    knn_aff = A.knn_ceiling(testkeys, lut_aff, smiles)
    knn_sin = A.knn_ceiling(testkeys, lut_sin, smiles)
    recall = abs(np.corrcoef([lut_aff[k] for k in lut_sin if k in lut_aff],
                             [lut_sin[k] for k in lut_sin if k in lut_aff])[0, 1])

    present = [s for s, _, _ in MODELS if any(np.isfinite(v) for v in sin[s].values())]
    print("sine results present for:", present)
    paths.append(bar_plot(aff, knn_aff, None,
                 "Affine (rank-preserving) transform", "nonmono_bars_affine.png"))
    paths.append(bar_plot(sin, knn_sin, recall,
                 "Non-monotonic sine transform", "nonmono_bars_sin.png"))
    print("Wrote:")
    for p in paths:
        print("  ", os.path.relpath(p, os.path.join(os.path.dirname(__file__), "..")))


if __name__ == "__main__":
    main()

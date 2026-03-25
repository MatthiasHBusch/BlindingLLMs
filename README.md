# In-Context Molecular Property Prediction with LLMs: A Blinding Study on Memorization and Knowledge Conflicts

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) *(link will be updated upon publication)*

## Overview

This repository provides the code and data for reproducing the experiments in our paper on evaluating LLMs for molecular property prediction. We introduce a **six-level blinding framework** that progressively removes contextual information to disentangle memorization, domain knowledge, and in-context learning capabilities of large language models.

### Key Contributions

- A systematic blinding framework (6 levels) for evaluating LLM molecular property prediction
- Evaluation of 9 LLM variants across 3 families (GPT-4.1, GPT-5, Gemini 2.5) and 3 size classes
- Analysis on 3 MoleculeNet datasets (Delaney/ESOL, Lipophilicity, QM7)
- Evidence that LLMs rely on domain knowledge rather than direct memorization
- Discovery that prior knowledge can interfere with in-context learning

## Repository Structure

```
├── README.md
├── LICENSE
├── requirements.txt          # Python dependencies
├── data/
│   ├── delaney-processed.csv # ESOL solubility dataset (1,128 molecules)
│   ├── Lipophilicity.csv     # Lipophilicity dataset (4,200 molecules)
│   └── qm7.csv              # QM7 atomization energy dataset
├── src/
│   ├── lib/                  # Julia LLM utilities
│   │   ├── LLMUtils.jl       # Core LLM API wrapper
│   │   ├── LLMs.jl           # Model definitions (add your API keys here)
│   │   └── FileWritingHelpers.jl
│   ├── Delaney_Prompts.jl    # Experiment logic & prompts for Delaney
│   ├── Lipophilicity_Prompts.jl
│   ├── QM7_Prompts.jl
│   ├── Run_Delaney_Experiments.jl    # 60/1000-shot experiments
│   ├── Run_Lipophilicity_Experiments.jl
│   ├── Run_QM7_Experiments.jl
│   ├── Run_Delaney_ZeroShot.jl       # 0-shot experiments
│   ├── Run_Lipophilicity_ZeroShot.jl
│   ├── Run_QM7_ZeroShot.jl
│   └── plotResultsICMLPaper.py       # Generate all paper figures
├── results/                  # Pre-computed LLM prediction results
│   ├── LLM_Results_delaney.json
│   ├── LLM_Results_delaney_zeroshot.json
│   ├── LLM_Results_lipophilicity.json
│   ├── LLM_Results_lipophilicity_zeroshot.json
│   ├── LLM_Results_qm7.json
│   └── LLM_Results_qm7_zeroshot.json
└── figures/                  # Output directory for generated plots
```

## Setup

### Prerequisites

- **Python 3.9+** for plotting and analysis
- **Julia 1.9+** for running LLM experiments (only needed to re-run experiments)
- API access to Azure OpenAI and/or OpenRouter (only needed to re-run experiments)

### Installation

```bash
# Clone the repository
git clone https://github.com/MatthiasHBusch/BlindingLLMs.git
cd BlindingLLMs

# Install Python dependencies
pip install -r requirements.txt
```

### API Configuration (for re-running experiments)

To re-run the LLM experiments, add your API keys to `src/lib/LLMs.jl`:

```julia
key = "YOUR_AZURE_OPENAI_API_KEY"
endpoint = "YOUR_AZURE_ENDPOINT"
key_openrouter = "YOUR_OPENROUTER_API_KEY"
```

## Reproducing Figures

All paper figures can be generated from the pre-computed results:

```bash
cd src
python plotResultsICMLPaper.py
```

Figures will be saved to the `figures/` directory.

## Re-running Experiments

To re-run the LLM prediction experiments (requires API access):

```bash
cd src

# Zero-shot experiments
julia Run_Delaney_ZeroShot.jl
julia Run_Lipophilicity_ZeroShot.jl
julia Run_QM7_ZeroShot.jl

# In-context learning experiments (60/1000-shot)
julia Run_Delaney_Experiments.jl
julia Run_Lipophilicity_Experiments.jl
julia Run_QM7_Experiments.jl
```

> **Note:** Running all experiments requires significant API credits and time. The pre-computed results in `results/` are provided for convenience.

## Datasets

All datasets are sourced from [MoleculeNet](https://moleculenet.org/):

| Dataset | Property | N | Unit | Source |
|---------|----------|---|------|--------|
| Delaney (ESOL) | Aqueous solubility | 1,128 | log(mol/L) | [Delaney 2004](https://doi.org/10.1021/ci034243x) |
| Lipophilicity | Octanol-water partition | 4,200 | logD (pH 7.4) | MoleculeNet |
| QM7 | Atomization energy | ~6,834 | kcal/mol | [Rupp et al. 2012](https://doi.org/10.1103/PhysRevLett.108.058301) |

The data files include additional columns (`transformed_smiles`, `transformed_solubility`) generated for the blinding experiments.

## Citation

If you find this work useful, please cite:

```bibtex
@article{busch2026blinding,
  title={In-Context Molecular Property Prediction with LLMs: A Blinding Study on Memorization and Knowledge Conflicts},
  author={Busch, Matthias and Tacke, Marius and Lamaka, Sviatlana V. and Zheludkevich, Mikhail L. and Linka, Kevin and Cyron, Christian J. and Feiler, Christian and Aydin, Roland C.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

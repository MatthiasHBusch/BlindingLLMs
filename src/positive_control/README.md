# Positive control for the memorization detector

Proves the 3-significant-digit exact-match detector *fires* when a value is truly
memorized — the validity check both reviewers asked for. Two known-value sets
spanning a contamination gradient:

- **Set A — standard atomic weights** of the elements (`known_atomic_weights.csv`,
  41 elements). Near-certain verbatim recall → detector ceiling.
- **Set B — normal boiling points** of common compounds (`known_boiling_points.csv`,
  45 compounds, Wikipedia-infobox values). 3-sig match implies recall.

Contrast: on ESOL/Lipophilicity/QM7 the same detector fires at ~chance
(`../memorization_sigdigits.py`). High match here + ~chance there ⇒ detector is
sensitive, so its near-zero firing on the benchmarks is meaningful.

## How to run

1. `python build_known_values.py` — writes the two CSVs. (done; spot-check values)
2. **Approve API spend first.** Then in `Run_PositiveControl_ZeroShot.jl` set
   `llms_to_run` to the desired **flex** model objects. The Gemini/Claude flex
   variants already exist in `c:/lib/JuliaLibraries/LLMs.jl`; to include the
   paper's GPT-4.1/GPT-5 family, first add their `..._flex` OpenRouter variants
   there (the current GPT objects are Azure, non-flex). Run:
   `julia Run_PositiveControl_ZeroShot.jl`
   → writes `PositiveControl_AtomicWeights.json`, `PositiveControl_BoilingPoints.json`.
3. `python analyze_positive_control.py` — observed vs chance 2-/3-sig match rates;
   writes `positive_control_results.csv`.

## Notes

- Prompts mirror the benchmark 0-shot prompts (`Run_Delaney_ZeroShot.jl`):
  same role priming, same "numerical value only" instruction, 5 iterations.
- Cost is small (≈ 41+45 prompts × 5 iters × n_models, flex tier).
- The detector code is reused verbatim, so the comparison to the benchmarks is
  apples-to-apples.

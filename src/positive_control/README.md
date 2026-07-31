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

## Blinding sweep on the control set (round-2 revision, Reviewer 2.2)

Answers "which blinding level is sufficient to break verbatim recall?" by applying
all six blinding levels to Set B, and Levels 1-4 to Set A.

**Scale limitation, state it wherever these numbers are used:** 30 in-context
examples and 45 (Set B) / 41 (Set A) test items, against up to 1000 examples and
150 test molecules in the main experiments. Absolute correlations are therefore not
comparable to Figures 3/4 -- with 30 examples the in-context-learning component is
necessarily weaker -- and per-model cells are noisy, so only counts pooled over
models and runs support inference. The design supports the comparison *across*
levels at a fixed sample size, which is what the calibration question needs.

**Why Set A gets only Levels 1-4:** each element occurs exactly once, so replacing
its identity with an opaque structure string leaves every example carrying a token
that appears nowhere else -- no token-to-value mapping is inferable even in
principle. Structural blinding only carries information when the characters recur
across samples, as they do for SMILES (transformed strings still encode composition
and length). Secondarily, the replacement scheme would leave an element symbol
intact anyway, and Set A's 0-shot hit rate (92-100% for every model) is a constant,
so it cannot support the recall-correlation analysis.

**Prompt schema — why these sweeps use single-step prompts.** The benchmark
prompt sequence of Figures 3/4 ends with an instruction to report a weighted
average of similar molecules, which by construction cannot be an exactly recalled
number, so the exact-match detector under-reports recall under it. Both sweeps here
therefore use single-step prompts that ask only for a numerical value. (We also ran
both sets under the benchmark schema to size that effect; those runs are not part of
the paper and are not distributed. The level definitions they use remain in
`BoilingPoint_Prompts.jl` / `AtomicWeight_Prompts.jl`, since the single-step variants
build on them.)

1. `julia transform_boiling_points.jl` / `julia transform_atomic_weights.jl` — add
   `transformed_smiles` and `transformed_solubility` to
   `known_{boiling_points,atomic_weights}_blinded.csv`, using the replacement
   scheme and target transform of `../transform_potency.jl`.
2. `MOCK=1 julia <run script>` — free dry run; verifies prompt filling. **Delete
   the mock JSONs afterwards** or the real run appends to them.
3. The two sweeps (3-fold CV, 9 flex models each):
   - `Run_PositiveControl_Blinding_Direct.jl` — **BP/direct**, primary: boiling
     points, all 6 levels, single-step prompts from `BoilingPoint_DirectPrompts.jl`.
     10 runs, ~24 300 calls, ~$20.
   - `Run_PositiveControl_Blinding_AtomicWeights_Direct.jl` — **AW/direct**, the
     independent replication: atomic weights, levels 1-4, single-step prompts from
     `AtomicWeight_DirectPrompts.jl`. 10 runs, ~14 000 calls, ~$8.
   The level definitions come from `BoilingPoint_Prompts.jl`, derived mechanically
   from `../Delaney_Prompts.jl`, so they mean the same thing as in Figures 3/4.

   **Repetitions and topping up.** The two direct sweeps use 10 runs so that the
   per-level rates rest on ~410 (BP) / ~390 (AW) items per model instead of ~82/78.
   `NUM_RUNS` overrides the count for one invocation. Because predictions are
   *appended* to the JSON and the fold split is drawn from a fixed seed *before* the
   run loop, every run sees the same in-context examples and test items, so topping
   an existing file up (`NUM_RUNS=8` on a 2-run file) is statistically identical to
   one 10-run job — which is how the shipped files were built. The analysis reads
   however many repetitions it finds and warns if the count is uneven across
   model/level cells, which is what an aborted top-up would look like.
   (The intermediate 2-run files are not shipped; the 10-run JSONs are the record.)
4. `python analyze_blinding_control.py [direct|atomic_direct]` —
   degradation curves, per-level match rates against a permutation chance
   baseline, Fisher/sign tests, the cracking-versus-recall correlation, and
   `blinding_control_table*.tex`. Needs numpy/pandas.
5. `python make_blinding_control_si_table.py` — SI Table S4 body (pooled, both
   direct sweeps side by side).
6. `python retention_cluster_bootstrap.py` — SI Table S5 body: per-model retention
   $R_{21}=m_2/m_1$ for BP/direct with 95% intervals, beside each model's 0-shot
   retention. The intervals resample the 45 compounds, not the individual queries:
   the ten repetitions run over the same species, so they bound noise in the
   models' answers but carry no information about the sample of compounds, and an
   interval over the pooled events comes out roughly three times too narrow. The
   floor printed in the last row is the second-digit collision probability of the
   set, estimated by the U-statistic; at n = 41-45 the plug-in estimator is biased
   upward (12.5% instead of 10.5% for the boiling points).
   `python retention_ci_analysis.py` — the same quantities with Wilson intervals,
   plus the C2 floors for all four target columns. Kept because it is where the
   floor estimates come from; the table itself uses the cluster bootstrap.

   Why the paper reports retention rather than the match rate: the permutation
   baseline controls for the models' marginal answer distribution but not for their
   accuracy, and at the unblinded levels the models are accurate enough (41% of the
   Level-1 misses fall within 0.5% relative error) that a three-digit hit need not
   indicate retrieval. $R_{21}$ conditions on the first digit already being right
   and asks only whether the next digit follows.

Headline result (boiling points, pooled over 9 models and 10 repetitions): masking
the property name (Level 1 -> 3) does **not** break recall — 61.1% -> 51.4% match
rate, retention 87% -> 83%, and three of nine models show no decrease at all. The
atomic weights show no effect whatsoever there (97.6% -> 97.0%).
Transforming the target values is partial only: Level 2 keeps 10.2% against 0.5%
chance at 52% retention (atomic weights: 40.5% at 77% retention), because the models
reconstruct the affine map from the in-context anchors. Transforming the structure
string does break it: retention drops to 18% [10, 27] at Level 5 and 13% [7, 20] at
Level 6, both intervals containing the ~10% floor, with no model above the floor at
Level 6 and two below it. Level 5 is the decisive cell because its target is
untransformed, so recall would show up directly without any rescaling. Read as
near-complete rather than demonstrably complete suppression: the point estimates sit
somewhat above the floor, which would fit a small residual leakage, but 45 compounds
cannot separate that from coincidence. Cracking Level 3 correlates with the model's
0-shot retention on the same set (r = +0.83, permutation p = 0.021).

## Notes

- Prompts mirror the benchmark 0-shot prompts (`Run_Delaney_ZeroShot.jl`):
  same role priming, same "numerical value only" instruction, 5 iterations.
- Cost is small (≈ 41+45 prompts × 5 iters × n_models, flex tier).
- The detector code is reused verbatim, so the comparison to the benchmarks is
  apples-to-apples.

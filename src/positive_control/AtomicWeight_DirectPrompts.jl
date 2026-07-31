#=
Direct-prompt variants of blinding levels 1-4 for the ATOMIC-WEIGHT positive
control (JCIM revision round 2).

Counterpart of BoilingPoint_DirectPrompts.jl, and for the same reason: the
benchmark prompt sequence ends with an instruction to report a weighted average
of similar species ("6. Weighted Average ... 7. write down the weighted
average"). A model following that instruction cannot emit an exactly recalled
value, so the exact-match detector reads recall laundered through an average as
"no recall". The boiling-point sweep showed the effect to be large (Level-1 match
rate 26.9% under the benchmark schema versus 61.4% with a single-step prompt), so
the atomic weights must be measured the same way if the two sweeps are to be
compared in one table.

These four approaches are the same four information conditions as the wp_*
levels, with a SINGLE-STEP prompt: in-context examples, the target species, and
"provide your answer as a single numerical value". No pre-analysis step, no
averaging instruction, no mention of neighbours.

Information content is matched level by level to the wp_* approaches:
  io_specific_clear      1  names + SMILES, atomic weight named, original values
  io_specific_blind      2  names + SMILES, "related to the atomic weight", transformed
  io_molproperty_clear   3  names + SMILES, "molecular property", original
  io_molproperty_blind   4  names + SMILES, "molecular property", transformed

Levels 5/6 (Agnostic) are deliberately absent, as in AtomicWeight_Prompts.jl:
each element occurs exactly once, so an opaque structure string would give every
training example a token that appears nowhere else and no token-to-value mapping
would be inferable even in principle (and our replacement scheme leaves an
element symbol intact anyway). Requesting io_sampleproperty_* therefore fails
with a KeyError, by design.

Loaded on top of AtomicWeight_Prompts.jl, which is left untouched (it is a
generated file).
=#

include(joinpath(@__DIR__, "AtomicWeight_Prompts.jl"))

const DIRECT_APPROACHES_AW = [
    "io_specific_clear",
    "io_specific_blind",
    "io_molproperty_clear",
    "io_molproperty_blind",
]

for a in DIRECT_APPROACHES_AW
    prompts["names_only"][a] = Dict()
end

# --- system prompts: reuse the corresponding blinding-level system prompts, so
# --- the framing of the task is identical and only the prediction step differs.
prompts["names_only"]["io_specific_clear"]["system"] =
    prompts["names_only"]["with_preanalysis"]["system"]
prompts["names_only"]["io_specific_blind"]["system"] =
    prompts["names_only"]["wp_solubility_blind"]["system"]
prompts["names_only"]["io_molproperty_clear"]["system"] =
    prompts["names_only"]["wp_molproperty_clear"]["system"]
prompts["names_only"]["io_molproperty_blind"]["system"] =
    prompts["names_only"]["wp_molproperty_blind"]["system"]

# --- prediction prompts: one step, no analysis, no averaging instruction. -----

function named_direct_prompt_aw(label::String, values_key::String)
    return """
**Training Data:**
- Names: <molecule_names_training>
- SMILES: <smiles_strings_training>
- $label: <$values_key>

**Prediction Task:**
Predict the $label for the following species:
- Name: <molecule_name>
- SMILES: <smiles_string>

Provide your answer as a single numerical value at the end of your response.
"""
end

prompts["names_only"]["io_specific_clear"]["prediction"] =
    named_direct_prompt_aw("IUPAC standard atomic weight (u)", "solubilities_training")
prompts["names_only"]["io_specific_blind"]["prediction"] =
    named_direct_prompt_aw("molecular property related to the atomic weight",
                           "transformed_solubilities_training")
prompts["names_only"]["io_molproperty_clear"]["prediction"] =
    named_direct_prompt_aw("molecular property", "solubilities_training")
prompts["names_only"]["io_molproperty_blind"]["prediction"] =
    named_direct_prompt_aw("molecular property", "transformed_solubilities_training")

# Sanity check: each direct approach must resolve to exactly one step, so that
# the run really is single-shot and comparable across levels.
for a in DIRECT_APPROACHES_AW
    seq = get_prompt_sequence("names_only", a)
    length(seq) == 1 && seq[1] == "prediction" ||
        error("$a did not resolve to a single prediction step, got $seq")
end

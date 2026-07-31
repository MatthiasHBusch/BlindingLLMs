#=
Direct-prompt variants of the six blinding levels (JCIM revision round 2).

WHY: the benchmark prompt schema ends with an explicit instruction to compute a
weighted average of similar molecules' values ("6. Weighted Average ... 7. write
down the weighted average"). A model following that instruction cannot emit an
exactly recalled value, so the exact-match detector could in principle read
"recall laundered through an average" as "no recall" -- especially at the
transformed levels, where the model has to compute rather than quote. That would
undercut the finding that the value transformation breaks verbatim recall.

These six approaches are the same six information conditions with a SINGLE-STEP,
direct prompt: in-context examples, the target, and "provide your answer as a
single numerical value". No pre-analysis step, no averaging instruction, no
mention of neighbours. If the near-zero match rate at the transformed levels
survives here, the averaging instruction is not the cause.

Information content is matched level by level to the wp_* approaches:
  io_specific_clear      1  names + SMILES, property named, original values
  io_specific_blind      2  names + SMILES, "related to the boiling point", transformed
  io_molproperty_clear   3  names + SMILES, "molecular property", original
  io_molproperty_blind   4  names + SMILES, "molecular property", transformed
  io_sampleproperty_clear 5 structure strings only, "sample property", original
  io_sampleproperty_blind 6 structure strings only, "sample property", transformed

Loaded on top of BoilingPoint_Prompts.jl, which is left untouched (it is a
generated file, and AtomicWeight_Prompts.jl is derived from it).
=#

include(joinpath(@__DIR__, "BoilingPoint_Prompts.jl"))

const DIRECT_APPROACHES = [
    "io_specific_clear",
    "io_specific_blind",
    "io_molproperty_clear",
    "io_molproperty_blind",
    "io_sampleproperty_clear",
    "io_sampleproperty_blind",
]

for a in DIRECT_APPROACHES
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
prompts["names_only"]["io_sampleproperty_clear"]["system"] =
    prompts["names_only"]["wp_sampleproperty_clear"]["system"]
prompts["names_only"]["io_sampleproperty_blind"]["system"] =
    prompts["names_only"]["wp_sampleproperty_blind"]["system"]

# --- prediction prompts: one step, no analysis, no averaging instruction. -----

function named_direct_prompt(label::String, values_key::String)
    return """
**Training Data:**
- Names: <molecule_names_training>
- SMILES: <smiles_strings_training>
- $label: <$values_key>

**Prediction Task:**
Predict the $label for the following molecule:
- Name: <molecule_name>
- SMILES: <smiles_string>

Provide your answer as a single numerical value at the end of your response.
"""
end

function anonymous_direct_prompt(label::String, values_key::String)
    return """
**Training Data:**
- Sample structure strings: <structure_strings_training>
- $label: <$values_key>

**Prediction Task:**
Predict the $label for the following sample:
- Sample structure string: <structure_string>

Provide your answer as a single numerical value at the end of your response.
"""
end

prompts["names_only"]["io_specific_clear"]["prediction"] =
    named_direct_prompt("Normal boiling point (degrees Celsius)", "solubilities_training")
prompts["names_only"]["io_specific_blind"]["prediction"] =
    named_direct_prompt("molecular property related to the boiling point",
                        "transformed_solubilities_training")
prompts["names_only"]["io_molproperty_clear"]["prediction"] =
    named_direct_prompt("molecular property", "solubilities_training")
prompts["names_only"]["io_molproperty_blind"]["prediction"] =
    named_direct_prompt("molecular property", "transformed_solubilities_training")
prompts["names_only"]["io_sampleproperty_clear"]["prediction"] =
    anonymous_direct_prompt("sample property", "solubilities_training")
prompts["names_only"]["io_sampleproperty_blind"]["prediction"] =
    anonymous_direct_prompt("sample property", "transformed_solubilities_training")

# Sanity check: each direct approach must resolve to exactly one step, so that
# the run really is single-shot and comparable across levels.
for a in DIRECT_APPROACHES
    seq = get_prompt_sequence("names_only", a)
    length(seq) == 1 && seq[1] == "prediction" ||
        error("$a did not resolve to a single prediction step, got $seq")
end

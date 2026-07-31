#=
Prepares the atomic-weight positive control for the blinding sweep (JCIM round 2).

Adds:
  - smiles                 : the atomic SMILES of the element, e.g. [Fe]
  - transformed_solubility : target inverted and normalised to 0-100 (generic
                             column name, so the shared *_Prompts.jl machinery
                             works unchanged -- same convention as
                             transform_potency.jl / transform_boiling_points.jl)
  - transformed_smiles     : NOT meaningfully producible, so a copy of `smiles` is
                             written to keep the shared code paths working.

                             The Agnostic levels 5/6 are not defined for this set.
                             The fundamental reason is uniqueness: every element
                             occurs exactly once, so replacing its identity with an
                             opaque token leaves each training example carrying a
                             symbol that appears nowhere else, and no mapping from
                             token to value is inferable even in principle.
                             Structural blinding only carries information when the
                             characters recur across samples, as they do for SMILES
                             (where the transformed strings still encode composition
                             and length). Secondarily, our character-replacement
                             scheme would leave an element symbol intact anyway
                             ([Fe] -> (Fe)). The sweep therefore runs levels 1-4,
                             where blinding acts on the property label and the
                             target values, and the runner refuses levels 5/6.

Usage:
    julia transform_atomic_weights.jl
=#

using CSV
using DataFrames

const INPUT_FILE = joinpath(@__DIR__, "known_atomic_weights.csv")
const OUTPUT_FILE = joinpath(@__DIR__, "known_atomic_weights_blinded.csv")

"""
    transform_target(values) -> Vector{Float64}

transformed = (-y - min(-y)) * 100 / (max(-y) - min(-y)); inverts the order and
maps to 0-100. Pearson |r| is invariant under this monotonic affine map, so task
difficulty is unchanged; its only role is to close the lexical-lookup loophole.
"""
function transform_target(values::Vector{Float64})::Vector{Float64}
    neg = -values
    min_val, max_val = extrema(neg)
    return (neg .- min_val) .* 100 ./ (max_val - min_val)
end

function main()
    println("Loading: $INPUT_FILE")
    df = CSV.read(INPUT_FILE, DataFrame)
    df.name = String.(df.name)
    df.symbol = String.(df.symbol)
    println("Rows: $(nrow(df))")

    df.smiles = "[" .* df.symbol .* "]"
    df.transformed_solubility = transform_target(Vector{Float64}(df.atomic_weight))
    # See header: no genuine structural blinding is possible for single elements.
    df.transformed_smiles = df.smiles

    CSV.write(OUTPUT_FILE, df)
    println("Wrote: $OUTPUT_FILE")

    println("\n--- Samples ---")
    for i in 1:min(4, nrow(df))
        println("  $(df.name[i]) ($(df.smiles[i])): $(df.atomic_weight[i]) u -> " *
                "$(round(df.transformed_solubility[i], digits=2))")
    end
    println("\nNOTE: transformed_smiles is a copy of smiles; run levels 1-4 only.")
end

main()

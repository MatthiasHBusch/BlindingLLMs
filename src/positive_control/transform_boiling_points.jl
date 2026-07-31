#=
Adds the two blinding columns to the positive-control boiling-point set:
  - transformed_smiles      : SMILES with atoms/bonds mapped to alternative chars
  - transformed_solubility   : target inverted and normalised to 0-100
                               (generic column name, so the shared *_Prompts.jl
                                machinery works unchanged -- same convention as
                                transform_potency.jl)

The replacement scheme and the target transform are copied verbatim from
../transform_potency.jl so that a blinding level on this control set means
exactly what it means on the benchmarks.

Usage:
    julia transform_boiling_points.jl
=#

using CSV
using DataFrames

const INPUT_FILE = joinpath(@__DIR__, "known_boiling_points.csv")
const OUTPUT_FILE = joinpath(@__DIR__, "known_boiling_points_blinded.csv")

const SMILES_REPLACEMENTS = Dict(
    # 2-char atoms (must be replaced before single chars)
    "Cl" => "Z", "Br" => "X", "Se" => "Y", "Si" => "W",
    # 1-char atoms
    "C" => "A", "O" => "B", "N" => "D",
    "c" => "a", "o" => "b", "n" => "d", "s" => "f",
    "S" => "E", "F" => "G", "I" => "H", "P" => "K",
    # Bonds / structure
    "=" => "~", "#" => "^", "(" => "{", ")" => "}", "@" => "*",
    "/" => "L", "\\" => "R", "+" => "P", "-" => "M",
    "[" => "(", "]" => ")", "%" => "&", "." => "!",
    # Numbers
    "0" => "q", "1" => "r", "2" => "s", "3" => "t", "4" => "u",
    "5" => "v", "6" => "w", "7" => "x", "8" => "y", "9" => "z",
)

"""
    transform_target(values) -> Vector{Float64}

transformed = (-y - min(-y)) * 100 / (max(-y) - min(-y)): inverts the order and
maps to 0-100. Pearson |r| is invariant under this monotonic affine map, so the
task difficulty is unchanged; its only role is to close the lexical-lookup
loophole (a memorised boiling point of 78.23 no longer appears as "78.23").
"""
function transform_target(values::Vector{Float64})::Vector{Float64}
    neg = -values
    min_val, max_val = extrema(neg)
    return (neg .- min_val) .* 100 ./ (max_val - min_val)
end

function transform_smiles(smiles::String)::String
    result = smiles
    for (from, to) in sort(collect(SMILES_REPLACEMENTS), by=x -> -length(x[1]))
        result = replace(result, from => to)
    end
    return result
end

function find_unreplaced_chars(transformed::Vector{String})::Set{Char}
    targets = Set{Char}()
    for (_, to) in SMILES_REPLACEMENTS, c in to
        push!(targets, c)
    end
    unreplaced = Set{Char}()
    for s in transformed, c in s
        c in targets || push!(unreplaced, c)
    end
    return unreplaced
end

function main()
    println("Loading: $INPUT_FILE")
    df = CSV.read(INPUT_FILE, DataFrame)
    df.smiles = String.(df.smiles)
    df.name = String.(df.name)
    println("Rows: $(nrow(df))")

    df.transformed_solubility = transform_target(Vector{Float64}(df.bp_celsius))
    df.transformed_smiles = transform_smiles.(df.smiles)

    CSV.write(OUTPUT_FILE, df)
    println("Wrote: $OUTPUT_FILE")

    println("\n--- Samples ---")
    for i in 1:min(4, nrow(df))
        println("  $(df.name[i]): $(df.smiles[i]) -> $(df.transformed_smiles[i])  |  " *
                "$(df.bp_celsius[i]) C -> $(round(df.transformed_solubility[i], digits=2))")
    end

    println("\n--- Unreplaced characters ---")
    unreplaced = sort(collect(find_unreplaced_chars(df.transformed_smiles)))
    if isempty(unreplaced)
        println("  none (all structural characters blinded)")
    else
        # Anything left here would leak chemical identity through the Agnostic levels.
        for c in unreplaced
            println("  '$c' (code $(Int(c)))")
        end
        @warn "Unreplaced characters found -- extend SMILES_REPLACEMENTS before running the sweep."
    end
end

main()

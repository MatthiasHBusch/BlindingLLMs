#=
Script to transform the ASAP antiviral potency dataset by adding two new columns:
- transformed_smiles: SMILES with atoms/bonds replaced by alternative characters
- transformed_solubility: target inverted and normalized to 0-100 range
  (the column keeps the generic name `transformed_solubility` so the shared
   *_Prompts.jl experiment code works unchanged across all four datasets)

Mirrors Src/VerificationTesting/transform_qm7.jl. Run AFTER
`python Src/Revision/prepare_potency_dataset.py`, which writes the *_original.csv.

Usage:
    julia Src/Revision/transform_potency.jl
=#

using CSV
using DataFrames

# --- CONSTANTS ---

const INPUT_FILE = joinpath(@__DIR__, "../data/antiviral_potency_original.csv")
const OUTPUT_FILE = joinpath(@__DIR__, "../data/antiviral_potency.csv")

# Replacement dictionary for SMILES transformation
# 2-char patterns listed first to ensure they're replaced before single chars.
# Same scheme as transform_qm7.jl, plus lowercase aromatic sulfur "s" => "f"
# (thiophene/thiazole rings are common in antiviral drug-like molecules and would
#  otherwise leak through unblinded; "f" is neither a source nor a target char).
const SMILES_REPLACEMENTS = Dict(
    # 2-char atoms (must be replaced before single chars)
    "Cl" => "Z",  # Chlorine
    "Br" => "X",  # Bromine
    "Se" => "Y",  # Selenium
    "Si" => "W",  # Silicon
    # 1-char atoms
    "C" => "A",   # Carbon
    "O" => "B",   # Oxygen
    "N" => "D",   # Nitrogen
    "c" => "a",   # aromatic Carbon
    "o" => "b",   # aromatic Oxygen
    "n" => "d",   # aromatic Nitrogen
    "s" => "f",   # aromatic Sulfur
    "S" => "E",   # Sulfur
    "F" => "G",   # Fluorine
    "I" => "H",   # Iodine
    "P" => "K",   # Phosphorus
    # Bonds
    "=" => "~",   # Double bond
    "#" => "^",   # Triple bond
    "(" => "{",   # Left parenthesis
    ")" => "}",   # Right parenthesis
    "@" => "*",   # Chirality
    "/" => "L",   # Slash
    "\\" => "R",   # Backslash
    "+" => "P",   # Plus
    "-" => "M",   # Minus
    "[" => "(",   # Left bracket
    "]" => ")",   # Right bracket
    "%" => "&",   # Percent
    "." => "!",   # Dot
    # Numbers
    "0" => "q",   # Zero
    "1" => "r",   # One
    "2" => "s",   # Two
    "3" => "t",   # Three
    "4" => "u",   # Four
    "5" => "v",   # Five
    "6" => "w",   # Six
    "7" => "x",   # Seven
    "8" => "y",   # Eight
    "9" => "z",   # Nine
)

# --- FUNCTIONS ---

"""
    get_replaced_chars() -> Set{Char}

Get the set of all characters that will be replaced by the transformation.
"""
function get_replaced_chars()::Set{Char}
    replaced = Set{Char}()
    for (from, _) in SMILES_REPLACEMENTS
        for c in from
            push!(replaced, c)
        end
    end
    return replaced
end

"""
    transform_solubility(values::Vector{Float64}) -> Vector{Float64}

Transform target values using the formula:
    transformed = (-y - min(-y)) * 100 / (max(-y) - min(-y))

This inverts the order and maps to 0-100. Pearson |r| is invariant to this
monotonic affine map, so it does not change task difficulty (see response to
Reviewer 2.2); its only role is to close the lexical-lookup loophole.
"""
function transform_solubility(values::Vector{Float64})::Vector{Float64}
    neg = -values
    min_val, max_val = extrema(neg)
    return (neg .- min_val) .* 100 ./ (max_val - min_val)
end

"""
    transform_smiles(smiles::String) -> String

Replace atoms and bonds in SMILES string with alternative characters.
Longer patterns (e.g., "Cl", "Br") are replaced before shorter ones
to avoid partial matches.
"""
function transform_smiles(smiles::String)::String
    result = smiles
    # Sort by pattern length (descending) to replace 2-char patterns first
    sorted_replacements = sort(collect(SMILES_REPLACEMENTS), by=x -> -length(x[1]))
    for (from, to) in sorted_replacements
        result = replace(result, from => to)
    end
    return result
end

"""
    find_unreplaced_chars(transformed_smiles::Vector{String}) -> Set{Char}

Find all unique characters in transformed SMILES that weren't replaced.
These are characters that remain unchanged after the transformation and would
leak structural identity through the blinding -- inspect this report and extend
SMILES_REPLACEMENTS if anything chemically meaningful shows up.
"""
function find_unreplaced_chars(transformed_smiles::Vector{String})::Set{Char}
    # Get all replacement target characters (what things get replaced TO)
    replacement_targets = Set{Char}()
    for (_, to) in SMILES_REPLACEMENTS
        for c in to
            push!(replacement_targets, c)
        end
    end

    # Collect all unique chars in transformed SMILES that are NOT replacement targets
    unreplaced = Set{Char}()
    for smiles in transformed_smiles
        for c in smiles
            if !(c in replacement_targets)
                push!(unreplaced, c)
            end
        end
    end
    return unreplaced
end

"""
Main function to load, transform, and save the dataset.
"""
function main()
    println("Loading dataset from: $INPUT_FILE")
    df = CSV.read(INPUT_FILE, DataFrame)

    println("Original dataset size: $(nrow(df)) rows, $(ncol(df)) columns")

    # Convert SMILES column to String type (CSV may read it as String31 or similar)
    df.smiles = String.(df.smiles)

    # Transform target
    target_col = "pic50"
    df.transformed_solubility = transform_solubility(Vector{Float64}(df[!, target_col]))

    # Transform SMILES
    df.transformed_smiles = transform_smiles.(df.smiles)

    # Save to new file
    CSV.write(OUTPUT_FILE, df)
    println("Transformed dataset saved to: $OUTPUT_FILE")
    println("New dataset size: $(nrow(df)) rows, $(ncol(df)) columns")

    # Print some examples
    println("\n--- Sample Transformations ---")
    println("Transformed target range: $(minimum(df.transformed_solubility)) to $(maximum(df.transformed_solubility))")
    println("\nExample rows:")
    for i in [1, 2, 3]
        i > nrow(df) && break
        println("  Row $i:")
        println("    Original SMILES: $(df.smiles[i])")
        println("    Transformed SMILES: $(df.transformed_smiles[i])")
        println("    Original pic50: $(df[i, target_col])")
        println("    Transformed value: $(round(df.transformed_solubility[i], digits=2))")
    end

    # Report unreplaced characters
    println("\n--- Unreplaced Characters Analysis ---")
    unreplaced = find_unreplaced_chars(df.transformed_smiles)
    if isempty(unreplaced)
        println("All characters were replaced!")
    else
        sorted_chars = sort(collect(unreplaced))
        println("Characters NOT replaced ($(length(sorted_chars)) unique):")
        for c in sorted_chars
            # Show printable representation
            if isprint(c)
                println("  '$c' (code: $(Int(c)))")
            else
                println("  '\\u$(string(Int(c), base=16))' (code: $(Int(c)))")
            end
        end
    end
end

# Run main
main()

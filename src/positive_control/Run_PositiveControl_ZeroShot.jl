# Positive-control 0-shot runner for the memorization detector (JCIM revision).
#
# Mirrors RepoICML/src/Run_Delaney_ZeroShot.jl exactly (same query path, same
# JSON output format) but queries famous, high-contamination-likelihood values:
#   Set A - standard atomic weights of the elements
#   Set B - normal boiling points of common compounds
# so that the IDENTICAL exact-match detector can be applied to them.
#
# IMPORTANT: This INCURS API COST. Do NOT run until spend is approved.
# Per project policy, use ONLY the OpenRouter *flex* model variants. GPT flex
# variants must first be added to c:/lib/JuliaLibraries/LLMs.jl (the OpenAI
# models there are currently Azure, non-flex). Edit the `llms_to_run` list below.

using CSV
using DataFrames
using JSON
using Dates
using Printf
using HTTP

# Use the master libraries (they contain the flex model objects + helpers).
const LIBDIR = joinpath(@__DIR__, "..", "lib")
include(joinpath(LIBDIR, "LLMs.jl"))
include(joinpath(LIBDIR, "LLMUtils.jl"))
include(joinpath(LIBDIR, "FileWritingHelpers.jl"))

const SCRIPT_DIR = @__DIR__
const ITERATIONS = 5          # matches the benchmark 0-shot setting
const NUM_THREADS = 10

# ---- FLEX model objects (constructed here; service tier "flex" is the 4th arg).
# Gemini 2.5 (no flex object exists in c:/lib/.../LLMs.jl; build from the 2.5 providers).
gemini_2_5_pro_flex        = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-pro",        ["google-vertex/global"], "flex")
gemini_2_5_flash_flex      = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-flash",      ["google-vertex/global"], "flex")
gemini_2_5_flash_lite_flex = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-flash-lite", ["google-vertex"],        "flex")

# OpenAI via OpenRouter. Slugs verified against the live /models endpoint.
# Route through the "openai" provider so the flex tier applies; if a model does
# not support flex, OpenRouter falls back and bills at the served tier (no error).
gpt5_flex       = LLMAccessOpenRouter(key_openrouter, "openai/gpt-5",        ["openai"], "flex")
gpt5_mini_flex  = LLMAccessOpenRouter(key_openrouter, "openai/gpt-5-mini",   ["openai"], "flex")
gpt5_nano_flex  = LLMAccessOpenRouter(key_openrouter, "openai/gpt-5-nano",   ["openai"], "flex")
gpt41_flex      = LLMAccessOpenRouter(key_openrouter, "openai/gpt-4.1",      ["openai"], "flex")
gpt41_mini_flex = LLMAccessOpenRouter(key_openrouter, "openai/gpt-4.1-mini", ["openai"], "flex")
gpt41_nano_flex = LLMAccessOpenRouter(key_openrouter, "openai/gpt-4.1-nano", ["openai"], "flex")

# Per-model sampling kwargs (OpenRouter rejects unsupported params):
#  - GPT-5 family supports `reasoning` but NOT temperature/top_p -> set temp/top_p=1.0
#    (get_body omits them) and pass reasoning effort "minimal" (cheap, for recall).
#  - GPT-4.1 family supports temperature/top_p but NOT `reasoning` -> no reasoning_effort.
#  - Gemini 2.5 -> reasoning_effort "none" (as used for the benchmark 0-shot runs).
function kwargs_for(llm)
    m = llm.model
    if startswith(m, "openai/gpt-5")
        return (; reasoning_effort="minimal", temperature=1.0, top_p=1.0)
    elseif startswith(m, "openai/gpt-4.1")
        return (; temperature=0.7, top_p=0.95)
    else
        return (; reasoning_effort="none")
    end
end

# Gemini 2.5 already completed (2026-05-31). Set this to whichever models still
# need running; GPT entries write distinct model keys, so appending to the
# existing JSON does not touch the Gemini results.
llms_to_run = [
    gpt5_flex,
    gpt5_mini_flex,
    gpt5_nano_flex,
    gpt41_flex,
    gpt41_mini_flex,
    gpt41_nano_flex,
]

# ---------------- Set A: atomic weights ----------------
function run_atomic_weights(llm)
    data = DataFrame(CSV.File(joinpath(SCRIPT_DIR, "known_atomic_weights.csv")))
    out_file = joinpath(SCRIPT_DIR, "PositiveControl_AtomicWeights.json")
    system_prompt = """You are an expert chemist and know the periodic table of the elements very well. You are given the name of a chemical element. Your task is to provide its IUPAC standard atomic weight (the conventional value, as published by IUPAC/CIAAW) in unified atomic mass units (u). Provide only the numerical value as output, without any additional text."""
    get_prompt(name, symbol) = "What is the IUPAC standard atomic weight (in u) of the element $name (symbol $symbol)? Provide only the numerical value as output, without any additional text."

    conversations = Vector{Vector{Tuple}}()
    for row in eachrow(data)
        prompt = get_prompt(row.name, row.symbol)
        for _ in 1:ITERATIONS
            push!(conversations, [("system", system_prompt), ("user", prompt)])
        end
    end
    answers = ask_gpt_threaded(llm, conversations; num_threads=NUM_THREADS, retries=10, kwargs_for(llm)...)
    keys_list = Vector{Vector{String}}()
    results_list = []
    for j in 1:Int(round(length(answers) / ITERATIONS))
        for i in 1:ITERATIONS
            result = search_for_last_number_in_string(answers[(j-1)*ITERATIONS+i])
            push!(keys_list, Vector{String}(["AtomicWeight", "NoReasoning", llm.model, data[j, "symbol"]]))
            push!(results_list, result)
        end
    end
    append_values_to_json(out_file, keys_list, results_list)
    @printf("[A] %s done -> %s\n", llm.model, out_file)
end

# ---------------- Set B: boiling points ----------------
function run_boiling_points(llm)
    data = DataFrame(CSV.File(joinpath(SCRIPT_DIR, "known_boiling_points.csv")))
    out_file = joinpath(SCRIPT_DIR, "PositiveControl_BoilingPoints.json")
    system_prompt = """You are an expert chemist and know the physical properties of common chemical compounds very well. You are given a SMILES string of a molecule. Your task is to provide its normal boiling point in degrees Celsius (at 1 atm) as listed in the compound's English Wikipedia article (the value in the Chembox/infobox). Provide only the numerical value as output, without any additional text."""
    get_prompt(smiles) = "What is the normal boiling point in degrees Celsius, as listed on English Wikipedia, of the molecule with the following SMILES string: $smiles? Provide only the numerical value as output, without any additional text."

    conversations = Vector{Vector{Tuple}}()
    for row in eachrow(data)
        prompt = get_prompt(row.smiles)
        for _ in 1:ITERATIONS
            push!(conversations, [("system", system_prompt), ("user", prompt)])
        end
    end
    answers = ask_gpt_threaded(llm, conversations; num_threads=NUM_THREADS, retries=10, kwargs_for(llm)...)
    keys_list = Vector{Vector{String}}()
    results_list = []
    for j in 1:Int(round(length(answers) / ITERATIONS))
        for i in 1:ITERATIONS
            result = search_for_last_number_in_string(answers[(j-1)*ITERATIONS+i])
            push!(keys_list, Vector{String}(["BoilingPoint", "NoReasoning", llm.model, data[j, "smiles"]]))
            push!(results_list, result)
        end
    end
    append_values_to_json(out_file, keys_list, results_list)
    @printf("[B] %s done -> %s\n", llm.model, out_file)
end

function main()
    for llm in llms_to_run
        run_atomic_weights(llm)
        run_boiling_points(llm)
    end
end

main()

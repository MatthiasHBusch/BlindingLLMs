using CSV
using DataFrames
using JSON
using Dates
using Printf
using HTTP
include(joinpath(@__DIR__, "lib", "LLMs.jl"))
include(joinpath(@__DIR__, "lib", "FileWritingHelpers.jl"))

# Zero-shot memorization probe on the ASAP Discovery antiviral potency (Polaris)
# dataset. Mirrors Run_QM7_ZeroShot.jl. The exact-match detector is run on the
# predictions exactly as for ESOL/Lipophilicity/QM7. Because these molecules were
# released in 2025 for the ASAP-Polaris challenge (post-LLM-cutoff), this is the
# modern control: a chance-level match rate here, with the same detector that
# fires on demonstrably memorized values (positive control), is strong evidence
# of no verbatim contamination for a benchmark that the models cannot have seen.

function main(llm)
    script_dir = @__DIR__
    #
    data_file = joinpath(script_dir, "../data", "antiviral_potency.csv")
    out_file = joinpath(script_dir, "../results", "LLM_Results_potency_zeroshot.json")

    #llm = gemini_2_5

    system_prompt = """You are an expert medicinal chemist and know the ASAP Discovery 2025 antiviral potency dataset very well. You are given a SMILES string of a molecule. Your task is to predict the antiviral potency (pIC50 against the SARS-CoV-2 main protease, Mpro) of that molecule. Provide only the numerical value as output, without any additional text."""

    get_prediction_prompt(smiles) = "What is the antiviral potency (pIC50 against SARS-CoV-2 Mpro) in the ASAP Discovery 2025 antiviral potency dataset of the molecule with the following SMILES string: $smiles? Provide only the numerical value as output, without any additional text."

    data = DataFrame(CSV.File(data_file))

    iterations = 5
    answers = []
    conversations = Vector{Vector{Tuple}}()
    for row in eachrow(data)
        smiles = row.smiles
        prompt = get_prediction_prompt(smiles)
        for i in 1:iterations
            push!(conversations, [("system", system_prompt), ("user", prompt)])
        end
    end
    answers = ask_gpt_threaded(llm, conversations; num_threads=10, reasoning_effort="none", retries=10)
    # save answers to file (convert to number if possible)
    keys_list = Vector{Vector{String}}()
    results_list = []
    for j in 1:(Int(round(length(answers) / iterations)))
        for i in 1:iterations
            result = search_for_last_number_in_string(answers[(j-1)*iterations+i])
            # debug print
            #@printf("SMILES: %s, Answer: %s, Extracted number: %s\n", data[j, :smiles], answers[j], result)
            push!(keys_list, Vector{String}(["Smiles", "NoReasoning", llm.model, data[j, "smiles"]]))
            push!(results_list, result)
        end
    end
    append_values_to_json(out_file, keys_list, results_list)
end

# Full nine-model list (for the complete run later):
# models = [gemini_2_5_flash_lite, gemini_2_5_flash, gemini_2_5, gpt4_1, gpt4_1_mini, gpt4_1_nano, gpt5, gpt5_mini, gpt5_nano]
# Six non-flagship models (DONE): gemini_2_5_flash_lite, gemini_2_5_flash, gpt4_1_mini, gpt4_1_nano, gpt5_mini, gpt5_nano
# Now: the three flagship models (appended to the same results file).
models = [gemini_2_5, gpt4_1, gpt5]
for model in models
    main(model)
end

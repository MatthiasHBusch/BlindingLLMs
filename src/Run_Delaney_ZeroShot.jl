using CSV
using DataFrames
using JSON
using Dates
using Printf
using HTTP
include(joinpath(@__DIR__, "lib", "LLMs.jl"))
include(joinpath(@__DIR__, "lib", "FileWritingHelpers.jl"))

function main()
    script_dir = @__DIR__
    # 
    data_file = joinpath(script_dir, "../data", "delaney-processed.csv")
    out_file = joinpath(script_dir, "../results", "LLM_Results_delaney_zeroshot.json")

    llm = gemini_2_5



    system_prompt = """You are an expert chemist and know the delaney solubility dataset very well. You are given a SMILES string of a molecule. Your task is to predict the measured solubility of that molecule in log solubility in mols per litre log(mol/L). Provide only the numerical value as output, without any additional text."""

    get_prediction_prompt(smiles) = "What is the measured solubility in the delaney dataset in log(mol/L) of the molecule with the following SMILES string: $smiles? Provide only the numerical value as output, without any additional text."

    data = DataFrame(CSV.File(data_file))

    iterations = 5
    answers = []
    conversations = Vector{Vector{Tuple}}()
    for row in eachrow(data)#[1:5, :]) # for testing only
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
            push!(keys_list, Vector{String}(["Smiles", "NoReasoning", llm.model, data[j, "Compound ID"]]))
            push!(results_list, result)
        end
    end
    append_values_to_json(out_file, keys_list, results_list)
end

#"names_only": {
#        "with_preanalysis": {
#            "gpt-4.1-mini_2024-12-01-preview": {
#                "40": {


main()
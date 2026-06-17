# Add 3 more 0-shot repetitions for Lipophilicity and QM7 (JCIM revision), so they
# reach 5 reps x 1000 molecules = 5000 predictions, matching the refreshed Delaney
# 0-shot set (1128 x 5). Same first-1000 molecule sample (data[1:1000,:]), same
# prompts as the original Run_{Lipophilicity,QM7}_ZeroShot.jl, all nine paper models.
#
# Results are APPENDED to the existing zeroshot JSONs under the SAME model keys as
# the original 2-rep entries (GPT slugs written without the "openai/" prefix so they
# merge to 5 reps; Gemini keys "google/..." already match).
#
# INCURS API COST. Do NOT run until spend is approved. Uses OpenRouter flex tier.

using CSV, DataFrames, JSON, Dates, Printf, HTTP
const LIBDIR = joinpath(@__DIR__, "lib")
include(joinpath(LIBDIR, "LLMs.jl"))
include(joinpath(LIBDIR, "LLMUtils.jl"))
include(joinpath(LIBDIR, "FileWritingHelpers.jl"))

const SCRIPT_DIR = @__DIR__
const REPO = abspath(joinpath(SCRIPT_DIR, ".."))
const EXTRA_ITERS = 3        # 2 existing + 3 new = 5 reps total
const N_MOL = 1000           # same data[1:1000,:] sample as the originals
const NUM_THREADS = 20

# Nine paper models as OpenRouter flex objects.
gemini_2_5_pro_flex        = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-pro",        ["google-vertex/global"], "flex")
gemini_2_5_flash_flex      = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-flash",      ["google-vertex/global"], "flex")
gemini_2_5_flash_lite_flex = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-flash-lite", ["google-vertex"],        "flex")
gpt5_flex       = LLMAccessOpenRouter(key_openrouter, "openai/gpt-5",        ["openai"], "flex")
gpt5_mini_flex  = LLMAccessOpenRouter(key_openrouter, "openai/gpt-5-mini",   ["openai"], "flex")
gpt5_nano_flex  = LLMAccessOpenRouter(key_openrouter, "openai/gpt-5-nano",   ["openai"], "flex")
gpt41_flex      = LLMAccessOpenRouter(key_openrouter, "openai/gpt-4.1",      ["openai"], "flex")
gpt41_mini_flex = LLMAccessOpenRouter(key_openrouter, "openai/gpt-4.1-mini", ["openai"], "flex")
gpt41_nano_flex = LLMAccessOpenRouter(key_openrouter, "openai/gpt-4.1-nano", ["openai"], "flex")
MODELS = [gemini_2_5_pro_flex, gemini_2_5_flash_flex, gemini_2_5_flash_lite_flex,
          gpt5_flex, gpt5_mini_flex, gpt5_nano_flex, gpt41_flex, gpt41_mini_flex, gpt41_nano_flex]

# Match each model to the EXISTING JSON key (strip the "openai/" provider prefix).
model_key(llm) = startswith(llm.model, "openai/") ? llm.model[8:end] : llm.model

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

DATASETS = [
    (name="lipophilicity",
     csv=joinpath(REPO, "data", "Lipophilicity.csv"),
     out=joinpath(REPO, "results", "LLM_Results_lipophilicity_zeroshot.json"),
     system="""You are an expert chemist and know the Lipophilicity dataset very well. You are given a SMILES string of a molecule. Your task is to predict the lipophilicity of that molecule in logD octanol/water partition coefficient at pH 7.4. Provide only the numerical value as output, without any additional text.""",
     prompt=(smiles)->"What is the lipophilicity of the molecule with the following SMILES string: $smiles? Provide only the numerical value as output, without any additional text."),
    (name="qm7",
     csv=joinpath(REPO, "data", "qm7.csv"),
     out=joinpath(REPO, "results", "LLM_Results_qm7_zeroshot.json"),
     system="""You are an expert quantum chemist and know the QM7 dataset very well. You are given a SMILES string of a molecule. Your task is to predict the atomization energy of that molecule in kcal/mol. Provide only the numerical value as output, without any additional text.""",
     prompt=(smiles)->"What is the atomization energy in the QM7 dataset in kcal/mol of the molecule with the following SMILES string: $smiles? Provide only the numerical value as output, without any additional text."),
]

function run_one(ds, llm)
    data = DataFrame(CSV.File(ds.csv))
    rows = data[1:min(N_MOL, nrow(data)), :]
    conversations = Vector{Vector{Tuple}}()
    for row in eachrow(rows)
        for _ in 1:EXTRA_ITERS
            push!(conversations, [("system", ds.system), ("user", ds.prompt(row.smiles))])
        end
    end
    answers = ask_gpt_threaded(llm, conversations; num_threads=NUM_THREADS, retries=10, kwargs_for(llm)...)
    mkey = model_key(llm)
    keys_list = Vector{Vector{String}}()
    results_list = []
    for j in 1:Int(round(length(answers) / EXTRA_ITERS))
        for i in 1:EXTRA_ITERS
            result = search_for_last_number_in_string(answers[(j-1)*EXTRA_ITERS+i])
            push!(keys_list, Vector{String}(["Smiles", "NoReasoning", mkey, rows[j, "smiles"]]))
            push!(results_list, result)
        end
    end
    append_values_to_json(ds.out, keys_list, results_list)
    @printf("[%s] %s (+%d reps on %d mols) -> %s\n", ds.name, mkey, EXTRA_ITERS, nrow(rows), ds.out)
end

function main()
    for ds in DATASETS, llm in MODELS
        run_one(ds, llm)
    end
end

main()

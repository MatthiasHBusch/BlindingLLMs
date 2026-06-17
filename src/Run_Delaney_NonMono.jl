# Non-monotonic transform control (JCIM revision, Reviewer 2.2): GPT-5 on Delaney,
# 1000-shot, with the label transformed by a CONTINUOUS non-monotonic map (sine over
# two periods; see make_delaney_nonmono.py --transform sin). Reuses the standard
# Delaney pipeline unchanged -- only the data file differs, its `transformed_solubility`
# column holding the transform. Compare the resulting |r| at the transformed level to
# the affine-transform result in LLM_Results_delaney.json (analyze_nonmono.py) and to
# the kNN-Tanimoto structural ceiling under the same transform.
#
# Run first:  python Src/Revision/make_delaney_nonmono.py --transform sin
#
# COST KNOBS (GPT-5 1000-shot is the expensive config -- keep this small):
#   approaches_used : which transformed level to run (each ~150 test x ~2 steps x
#                     num_runs GPT-5 calls). Default = L2 only (Specific-Transformed):
#                     the property is still named, so it is the level where value
#                     leakage / recoverability is most at issue (L4/L6 are already
#                     heavily blinded by the generic/agnostic naming).
#   num_runs        : repetitions (default 2, matching the paper).
# Default below ~ 150 x 2 x 2 = ~600 GPT-5 calls.

include(joinpath(@__DIR__, "Delaney_Prompts.jl"))
include(joinpath(@__DIR__, "lib", "LLMs.jl"))

# OpenRouter model objects. GPT-5/GPT-4.1 at the *flex* service tier (50% cheaper).
# The Azure objects have no service_tier field, so flex must go through OpenRouter
# (same as the positive control). Gemini 2.5 Pro (`gemini_2_5`, from LLMs.jl) does NOT
# support OpenRouter flex, so it runs at standard tier.
gpt5_flex  = LLMAccessOpenRouter(key_openrouter, "openai/gpt-5",   ["openai"], "flex")
gpt41_flex = LLMAccessOpenRouter(key_openrouter, "openai/gpt-4.1", ["openai"], "flex")

function main()
    script_dir = @__DIR__
    # Which non-monotonic transform to use: "sin" (continuous, default) or "nonmono" (binned).
    # Build the CSV first with make_delaney_nonmono.py (sin -> delaney-processed-sin.csv).
    transform_tag = "sin"
    data_file = joinpath(script_dir, "../data", "delaney-processed-$(transform_tag).csv")
    results_dir = joinpath(script_dir, "../results")
    isdir(results_dir) || mkdir(results_dir)

    save_file = joinpath(results_dir, "LLM_Results_delaney_$(transform_tag).json")
    chat_save_file = joinpath(results_dir, "LLM_Chats_delaney_$(transform_tag).json")

    data_frame = DataFrame(CSV.File(data_file))

    number_of_test_samples = 150
    number_of_extended_training_samples = 940   # -> 1000-shot (940 + 60)
    seed = 42
    rng = MersenneTwister(seed)
    data_frame_shuffled = data_frame[shuffle(rng, 1:size(data_frame, 1)), :]
    dataframe_test = data_frame_shuffled[1:number_of_test_samples, :]
    dataframe_train = data_frame_shuffled[number_of_test_samples+1:number_of_test_samples+number_of_extended_training_samples, :]
    if length(intersect(dataframe_test[:, :smiles], dataframe_train[:, :smiles])) > 0
        error("Overlap between test and training data detected")
    end

    # GPT-5 (flex) already run. Now GPT-4.1 (flex) and Gemini 2.5 Pro (standard tier).
    models = [gpt41_flex, gemini_2_5]
    llmchats = Vector{LLMChat}()
    for model in models
        chat = LLMChat(model)
        m = chat.llmaccess.model
        if startswith(m, "openai/gpt-5")
            # GPT-5: omit temperature/top_p (set 1.0); pass reasoning; no verbosity.
            chat.temperature = 1.0; chat.top_p = 1.0; chat.reasoning_effort = "minimal"
        elseif startswith(m, "openai/gpt-4.1")
            # GPT-4.1: sampling params OK, NO reasoning, no verbosity.
            chat.temperature = 0.7; chat.top_p = 0.95
        elseif contains(m, "google")
            # Gemini 2.5 Pro: reasoning effort "none" (matches the benchmark runs).
            chat.reasoning_effort = "none"
        end
        chat.max_tokens = 4000
        chat.retries = 10
        push!(llmchats, chat)
    end

    training_sizes = [60]
    num_runs = 2
    k_folds = 5
    input_types = ["names_only"]

    # Transformed (blinded) levels use the `transformed_solubility` column.
    # All three transformed levels for the two new models (appended to the same JSON,
    # which already holds GPT-5 at L2/L4/L6).
    # L2="wp_solubility_blind", L4="wp_molproperty_blind", L6="wp_sampleproperty_blind".
    approaches_used = ["wp_solubility_blind", "wp_molproperty_blind", "wp_sampleproperty_blind"]

    println("Starting Delaney NON-MONOTONIC transform experiment (GPT-5, 1000-shot)...")
    println("Approaches: ", approaches_used, " | num_runs=", num_runs)

    full_run_delaney(
        save_file,
        dataframe_test,
        dataframe_train,
        llmchats,
        input_types,
        approaches_used,
        k_folds,
        training_sizes,
        num_runs;
        save_chat_to_file_name=chat_save_file,
        mock=false
    )

    println("Done. Results saved to $save_file")
end

main()

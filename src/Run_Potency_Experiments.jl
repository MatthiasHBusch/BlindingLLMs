include(joinpath(@__DIR__, "Potency_Prompts.jl"))
include(joinpath(@__DIR__, "lib", "LLMs.jl")) # Assuming this exists based on ZeroShot file

# In-context blinding experiment on the ASAP Discovery antiviral potency (Polaris)
# dataset -- the modern, post-LLM-cutoff control requested by Reviewer 2 (issue 1.3).
# Structure mirrors Run_QM7_Experiments.jl (SMILES-only, no names/descriptors).
# Run `python prepare_potency_dataset.py` then `julia transform_potency.jl` first.

function main()
    # Delete 100 oldest files
    remaining, deleted = delete_old_batch_files(gpt5_batch, 0)
    #println("Found $remaining files")
    # delete all but 100 newest files
    remaining, deleted = delete_old_batch_files(gpt5_batch, remaining - 100)
    #println("Deleted $deleted files")
    script_dir = @__DIR__
    data_file = joinpath(script_dir, "../data", "antiviral_potency.csv")
    results_dir = joinpath(script_dir, "../results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end

    # Define experiment parameters
    save_file = joinpath(results_dir, "LLM_Results_potency.json")
    chat_save_file = joinpath(results_dir, "LLM_Chats_potency.json")

    data_frame = DataFrame(CSV.File(data_file))

    # extract 150 random samples for kfold cross validation testing
    number_of_test_samples = 150
    # Extended (in-context) training pool, like Delaney (940). The antiviral set is
    # smaller and sparse, so cap to whatever is available after the test split.
    number_of_extended_training_samples = min(940, size(data_frame, 1) - number_of_test_samples)
    seed = 42
    rng = MersenneTwister(seed)
    data_frame_shuffled = data_frame[shuffle(rng, 1:size(data_frame, 1)), :]
    dataframe_test = data_frame_shuffled[1:number_of_test_samples, :]
    dataframe_train = data_frame_shuffled[number_of_test_samples+1:number_of_test_samples+number_of_extended_training_samples, :]
    # check overlap and print warning if overlap exists
    if length(intersect(dataframe_test[:, :smiles], dataframe_train[:, :smiles])) > 0
        error("Overlap between test and training data detected")
    end
    # LLMs to use

    # Models list (same nine models as the main experiments)
    models = [gemini_2_5_flash_lite, gemini_2_5_flash, gemini_2_5, gpt4_1, gpt4_1_mini, gpt4_1_nano, gpt5, gpt5_mini, gpt5_nano]
    llmchats = Vector{LLMChat}()
    for model in models
        chat = LLMChat(model)
        # heuristically distinguish between gpt 4.1 and gpt 5
        if contains(chat.llmaccess.model, "gpt-5")
            chat.reasoning_effort = "minimal"
            chat.verbosity = "medium"
        elseif contains(chat.llmaccess.model, "google")
            chat.reasoning_effort = "none"
            # no reasoning model
        end
        if chat.llmaccess.model == "gpt-5-mini"
            chat.max_tokens = 15000 # Exception for gpt5 mini because seems to fail sometimes with 4000
        else
            chat.max_tokens = 4000
        end
        chat.retries = 10
        push!(llmchats, chat)
    end

    training_sizes = [60]
    num_runs = 2
    k_folds = 5

    #input_types = ["names_only", "names_and_descriptors"]
    input_types = ["names_only"]
    approaches_base = approaches
    # Full blinding sweep: clear (Level 1 = with_preanalysis) + the five progressively
    # blinded levels, identical set to the other datasets.
    approaches_used = ["with_preanalysis", "wp_solubility_blind", "wp_molproperty_clear", "wp_molproperty_blind", "wp_sampleproperty_clear", "wp_sampleproperty_blind"]

    println("Starting Antiviral Potency Experiments...")
    println("Test samples: $number_of_test_samples, extended training pool: $number_of_extended_training_samples")
    println("Models: ", [m.llmaccess.model for m in llmchats])

    full_run_potency(
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

    println("Experiments completed. Results saved to $save_file")
end

main()

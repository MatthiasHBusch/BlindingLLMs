include("Lipophilicity_Prompts.jl")
include(joinpath(@__DIR__, "lib", "LLMs.jl"))

function main()
    # Delete 100 oldest files
    remaining, deleted = delete_old_batch_files(gpt5_batch, 0)
    #println("Found $remaining files")
    # delete all but 100 newest files
    remaining, deleted = delete_old_batch_files(gpt5_batch, remaining - 100)
    #println("Deleted $deleted files")
    script_dir = @__DIR__
    data_file = joinpath(script_dir, "../data", "lipophilicity.csv")
    results_dir = joinpath(script_dir, "../results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end

    # Define experiment parameters
    save_file = joinpath(results_dir, "LLM_Results_lipophilicity.json")
    chat_save_file = joinpath(results_dir, "LLM_Chats_lipophilicity.json")

    data_frame = DataFrame(CSV.File(data_file))

    # extract 150 random samples for kfold cross validation testing
    number_of_test_samples = 150
    number_of_extended_training_samples = 0 # max 1128-150 = 978
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

    # Models list
    #models = [gpt5_batch, gpt5_mini, gpt5_nano, gpt4_1_batch, gpt4_1_mini, gpt4_1_nano] # Example models, adjust as needed
    # still needed: 940+60: all models
    # still needed: 60: gpt4_1_nano (repeat because of unicode minus sign)
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
    #input_types = ["names_and_descriptors"]
    input_types = ["names_only"]
    approaches_base = approaches
    # approaches = ["input_output_prompting", "with_preanalysis", "wp_solubility_blind", "wp_molproperty_clear", "wp_molproperty_blind", "wp_sampleproperty_clear", "wp_sampleproperty_blind"]
    approaches_used = ["with_preanalysis"]
    #approaches_used = ["wp_molproperty_blind"]

    println("Starting QM7 Experiments...")
    println("Models: ", [m.llmaccess.model for m in llmchats])

    full_run_lipophilicity(
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

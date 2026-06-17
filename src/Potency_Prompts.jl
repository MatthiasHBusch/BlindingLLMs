include(joinpath(@__DIR__, "lib", "LLMUtils.jl"))
include(joinpath(@__DIR__, "lib", "FileWritingHelpers.jl"))
using CSV
using DataFrames
using JSON
using Dates
using PrettyTables
using Statistics
using Printf
using Random


# --- STRUCTS & TYPES ---

mutable struct ExperimentState
    id::String
    input_data::String
    approach::String
    llmchat::LLMChat
    num_training_samples::Int
    num_extended_training_samples::Int
    data::Dict # Contains specific training/test data for this sample
    #conversation::Vector{Any} not needed because we use the LLMChat object
    step_index::Int
    done::Bool
    prediction::Any
    history::Vector{String} # To track which prompts were used
    run_id::Int # To track which iteration this state belongs to (for batch splitting)
end

# Lock for file writing to avoid race conditions in parallel execution
const FILE_WRITE_LOCK = ReentrantLock()

# --- EXPERIMENT LOGIC ---

function get_prompt_sequence(input_data::String, approach::String)
    # Define the sequence of keys in `prompts` for each approach
    # logic similar to execute_approach! but strict ordering
    tasks = get_possible_tasks(input_data, approach)

    # Standard ordering logic
    analyses = sort(collect(filter(x -> contains(x, "analysis"), tasks)))
    consecutive = sort(collect(filter(x -> contains(x, "summary"), tasks)))
    prediction = collect(filter(x -> contains(x, "prediction"), tasks))

    return [analyses; consecutive; prediction]
end

function create_experiment_states(data_frame_test::DataFrame, data_frame_train::DataFrame, llmchats::Vector{LLMChat}, input_types::Vector{String}, approaches::Vector{String}, k_folds::Int, training_sizes::Vector{Int}, num_runs::Int, random_split_seed::Union{Int,Missing})
    states = Vector{ExperimentState}()
    rng = MersenneTwister(random_split_seed)
    # Shuffle once globally if needed, or per fold. Stick to original logic:
    # Original: data_shuffled = data_frame[shuffle(rng, 1:size(data_frame, 1)), :]
    # But strictly, we need to replicate the exact indices for each run/fold.

    smiles_column = "smiles"
    target_column = "pic50"
    num_extended_training_samples = size(data_frame_train, 1)

    # Pre-calculate all indices to ensure determinism
    n_rows = size(data_frame_test, 1)

    # We need to be careful about RNG state if we want exact reproduction of the previous nested loop order.
    # However, generating them all upfront is better.

    # Let's generate the dataset splits first
    # To keep it compatible with previous "random_split_seed" behavior which shuffled the whole DF at start:
    # (But strictly, we need to shuffle once per run such that a larger training set always includes all samples of a smaller one)
    data_shuffled = data_frame_test[shuffle(rng, 1:n_rows), :]

    # Create states
    count = 0
    for num_training_samples in training_sizes
        num_test_samples = div(n_rows, k_folds)
        test_sets = [(num_test_samples*(i-1)+1):(num_test_samples*i) for i in 1:k_folds]
        training_sets = [setdiff(1:n_rows, ts)[1:num_training_samples] for ts in test_sets]

        for run in 1:num_runs
            for j in 1:k_folds
                training_indices = training_sets[j]
                test_indices = Vector{Int}(test_sets[j])

                # Get the data dictionary for this specific fold
                # Note: This might be memory intensive if we duplicate big strings for every single sample.
                # Optimization: Shared dictionary for shared parts? 
                # For now, we follow the pattern but maybe we can optimize `get_data_dict` slightly or just store indices and generated on fly?
                # Generating on fly in the step-loop is risky for "batching" if we want to fill prompts strictly.
                # Let's generate strict data dicts.

                fold_data = get_data_dict(data_shuffled, training_indices, test_indices)

                # The fold data contains ALL test samples for that fold.
                # We need to create a state for EACH test sample in this fold.

                test_smiles = fold_data["smiles_strings_test_single"]

                for (idx, smiles_string) in enumerate(test_smiles)
                    # Create specific data dict for this sample (copying fold data + specific sample data)
                    sample_data = copy(fold_data)
                    sample_data["smiles_string"] = smiles_string
                    sample_data["structure_string"] = fold_data["structure_strings_test_single"][idx]
                    # QM7 doesn't have descriptors
                    # if haskey(fold_data, "descriptors_test_single")
                    #     sample_data["descriptors"] = fold_data["descriptors_test_single"][idx]
                    # end

                    for input_data in input_types

                        for approach in approaches
                            prompt_extended_training_data = ""
                            # precalculate the extended training data prompt if needed
                            if num_extended_training_samples > 0
                                structure_string_column = "transformed_smiles"
                                transformed_solubility_column = "transformed_solubility"
                                target_column_ = target_column
                                smiles_column_ = smiles_column
                                smiles_column_label = "SMILES"
                                if approach == "input_output_prompting" || approach == "with_preanalysis"
                                    target_column_label = "antiviral potency (pIC50)"
                                elseif approach == "wp_solubility_blind"
                                    target_column_ = transformed_solubility_column
                                    target_column_label = "molecular property related to antiviral potency"
                                elseif approach == "wp_molproperty_clear"
                                    target_column_label = "molecular property"
                                elseif approach == "wp_molproperty_blind"
                                    target_column_ = transformed_solubility_column
                                    target_column_label = "molecular property"
                                elseif approach == "wp_sampleproperty_clear"
                                    target_column_label = "sample property"
                                    smiles_column_ = structure_string_column
                                    smiles_column_label = "Structure representation string"
                                elseif approach == "wp_sampleproperty_blind"
                                    target_column_ = transformed_solubility_column
                                    target_column_label = "sample property"
                                    smiles_column_ = structure_string_column
                                    smiles_column_label = "Structure representation string"
                                else
                                    error("Unknown approach: $approach")
                                end
                                prompt_extended_training_data = "This is an extended set of samples from the same dataset as the test set:\n"
                                for i in axes(data_frame_train, 1)
                                    prompt_extended_training_data *= "$(smiles_column_label): $(data_frame_train[i, smiles_column_])\n"
                                    # QM7 doesn't have descriptors
                                    # if input_data == "names_and_descriptors"
                                    #     for descriptor in descriptors_columns
                                    #         prompt_extended_training_data *= "$(descriptor): $(data_frame_train[i, descriptor])\n"
                                    #     end
                                    # end
                                    prompt_extended_training_data *= "$(target_column_label): $(data_frame_train[i, target_column_])\n\n"
                                end
                            end
                            for llmchat in llmchats
                                count += 1
                                state = ExperimentState(
                                    string(count),
                                    input_data,
                                    approach,
                                    deepcopy(llmchat),
                                    num_training_samples,
                                    num_extended_training_samples,
                                    sample_data,
                                    1,    # Step 1
                                    false, # Not done
                                    nothing,
                                    [],
                                    run
                                )
                                # Initialize System Prompt
                                sys_prompt = prompts[input_data][approach]["system"]
                                sys_msg = ("system", fill_prompt(sys_prompt, sample_data))
                                if contains(llmchat.llmaccess.model, "o1")
                                    # o1 hack: usually user message instead of system
                                    sys_msg = ("user", sys_msg[2])
                                end
                                push!(state.llmchat.conversation, sys_msg)
                                if num_extended_training_samples > 0
                                    push!(state.llmchat.conversation, ("user", prompt_extended_training_data, "ephemeral"))
                                end
                                push!(states, state)
                            end
                        end
                    end
                end
            end
        end
    end
    return states
end

function global_batch_execution(
    save_file::String,
    data_frame_test::DataFrame,
    data_frame_train::DataFrame,
    llmchats::Vector{LLMChat},
    input_types::Vector{String},
    approaches::Vector{String},
    k_folds::Int,
    training_sizes::Vector{Int},
    num_runs::Int;
    random_split_seed::Union{Int,Missing}=42,
    save_chat_to_file_name::String="",
    mock::Bool=false # For testing without costs
)
    # 1. Create all states
    println("Generating experiment states...")
    all_states = create_experiment_states(data_frame_test, data_frame_train, llmchats, input_types, approaches, k_folds, training_sizes, num_runs, random_split_seed)
    println("Total tasks to process: ", length(all_states))

    # 2. Group by LLM to handle them separately (optimizes context caching? actually batching does that)
    # But fundamentally, we can't mix calls to different LLM objects in one `ask_gpt_batch` call usually?
    # Unless the list of llmchats is just configurations.
    # We should group by unique `llmchat.llmaccess.deployment` or just the LLM object similarity.

    # We'll simple dictionary group by string(llmchat.llmaccess) * llmchat.reasoning_effort result to separate queues.
    # Actually, we can just process them in big chunks.

    # Let's Group by (Model Name/Deployment)
    # We need to map back to the state object to update it.

    # Helper to detect batch mode
    function is_batch_llm(llmaccess::LLMAccess)
        # Check property safely
        d = hasproperty(llmaccess, :deployment) ? llmaccess.deployment : ""
        return contains(lowercase(d), "batch")
    end

    # Group states by unique LLM identifier (Model + version)
    llm_groups = Dict{String,Vector{ExperimentState}}()
    for state in all_states
        # check if model has a reasoning effort
        key = string(state.llmchat.llmaccess)
        if !ismissing(state.llmchat.reasoning_effort)
            key = string(state.llmchat.llmaccess) * "(" * state.llmchat.reasoning_effort * ")"
        end
        if !haskey(llm_groups, key)
            llm_groups[key] = []
        end
        push!(llm_groups[key], state)
    end

    for (llm_key, states) in llm_groups
        println("Processing group: $llm_key with $(length(states)) tasks")

        # Determine execution wrapper based on the FIRST state's LLM
        # (Assuming all in group are identical)
        sample_llmaccess = states[1].llmchat.llmaccess
        use_batch = is_batch_llm(sample_llmaccess)
        println("  Mode: ", use_batch ? "BATCH API" : "PARALLEL ASYNC")

        # Split states by run_id
        # We want to create one batch file per run so we don't exceed file size limits
        # And we want to run them in parallel for speed

        run_groups = Dict{Int,Vector{ExperimentState}}()
        for state in states
            if !haskey(run_groups, state.run_id)
                run_groups[state.run_id] = []
            end
            push!(run_groups[state.run_id], state)
        end

        println("  Split into $(length(run_groups)) run groups for parallel execution.")

        @sync begin
            for (run_id, run_states) in run_groups
                #@async begin # async begin: comment out for threaded use, use for batch processing
                println("  [Run $run_id] Starting execution for $(length(run_states)) tasks...")
                # execution loop for this group
                while true
                    # 1. Collect Active States (not done)
                    active_states = filter(s -> !s.done, run_states)
                    if isempty(active_states)
                        break
                    end

                    # 2. Prepare Inputs
                    # We need to determine the NEXT prompt for each state
                    llmchats = LLMChat[]
                    states_in_batch = [] # Track which state belongs to which input

                    for state in active_states
                        # Get sequence of tasks
                        seq = get_prompt_sequence(state.input_data, state.approach)

                        if state.step_index > length(seq)
                            state.done = true
                            continue
                        end

                        task_key = seq[state.step_index]
                        raw_prompt = prompts[state.input_data][state.approach][task_key]
                        filled_prompt = fill_prompt(raw_prompt, state.data)

                        # Check context limit or optimization? 
                        # For now just push the conversation + new prompt
                        if state.step_index == length(seq) - 1
                            push!(state.llmchat.conversation, ("user", filled_prompt, "ephemeral"))
                        else
                            push!(state.llmchat.conversation, ("user", filled_prompt))
                        end

                        if !(state.llmchat.conversation in [chat.conversation for chat in llmchats])
                            push!(llmchats, state.llmchat)
                            push!(states_in_batch, state)
                        end
                    end

                    if isempty(llmchats)
                        break
                    end

                    println("  [Run $run_id] Step batch size: $(length(llmchats))")

                    # 3. MOCK or REAL EXECUTION
                    answers = []
                    if mock
                        #println("  [Run $run_id] [MOCK] simulating responses...")
                        # Prepend prompt to answer to verify filling
                        answers = ["Mock response for: " * chat.conversation[end][2] * "\n [1.0]" for chat in llmchats]
                    else
                        # Choose function
                        # We need to construct a robust `ask_gpt` generic that takes vector of conversations
                        if use_batch
                            answers = ask_gpt_batch(llmchats; use_last_file_with_id=missing)
                        else
                            answers = ask_gpt_threaded(llmchats; num_threads=10)
                        end
                    end

                    # 4. Update States
                    current_step_index = -2
                    for (i, ans) in enumerate(answers)
                        state = states_in_batch[i]

                        # Add history
                        # We need to know what prompt we just sent to add it to history correctly?
                        # In standard API, we append User+Assistant.
                        # But original logic had that specific "One Analysis Prompt" rule.
                        # Let's stick to standard conversation growth: User -> Answer.

                        # Re-constructing the filled prompt to save to history
                        seq = get_prompt_sequence(state.input_data, state.approach)
                        task_key = seq[state.step_index]
                        #raw_prompt = prompts[state.input_data][state.approach][task_key]
                        #filled_prompt = fill_prompt(raw_prompt, state.data)

                        #push!(state.llmchat.conversation, ("user", filled_prompt))
                        push!(state.llmchat.conversation, ("assistant", ans))

                        # Check for prediction
                        if contains(task_key, "prediction")
                            val = search_for_last_number_in_string(ans)
                            state.prediction = val
                        end

                        state.step_index += 1
                        current_step_index = state.step_index
                        if state.step_index > length(seq)
                            state.done = true
                        end
                    end
                    for state in active_states

                        if state.step_index == current_step_index # already processed
                            continue
                        end
                        # else now we get the answer of the llmchat which was exactly the same (state with answer has the answer as last element which is not in the current state)
                        ans = ""
                        for state_with_answer in states_in_batch
                            if state_with_answer.llmchat.conversation[1:end-1] == state.llmchat.conversation
                                ans = state_with_answer.llmchat.conversation[end][2]
                                push!(state.llmchat.conversation, ("assistant", ans))
                                if ans == ""
                                    @warn("Empty answer for state $(state.id), trying to find another answer...")
                                else
                                    if state.approach != state_with_answer.approach
                                        @warn("Approach $(state.approach) does not match approach $(state_with_answer.approach) for state $(state.id)")
                                    end
                                    if state.input_data != state_with_answer.input_data
                                        @warn("Input data $(state.input_data) does not match input data $(state_with_answer.input_data) for state $(state.id)")
                                    end
                                    break
                                end
                            end
                        end
                        if ans == ""
                            @warn("No answer found for state $(state.id)")
                        end

                        # Add history
                        # We need to know what prompt we just sent to add it to history correctly?
                        # In standard API, we append User+Assistant.
                        # But original logic had that specific "One Analysis Prompt" rule.
                        # Let's stick to standard conversation growth: User -> Answer.

                        # Re-constructing the filled prompt to save to history
                        seq = get_prompt_sequence(state.input_data, state.approach)
                        task_key = seq[state.step_index]
                        #raw_prompt = prompts[state.input_data][state.approach][task_key]
                        #filled_prompt = fill_prompt(raw_prompt, state.data)

                        #push!(state.llmchat.conversation, ("user", filled_prompt))
                        #push!(state.llmchat.conversation, ("assistant", ans))

                        # Check for prediction
                        if contains(task_key, "prediction")
                            val = search_for_last_number_in_string(ans)
                            state.prediction = val
                        end

                        state.step_index += 1
                        current_step_index = state.step_index
                        if state.step_index > length(seq)
                            state.done = true
                        end
                    end
                end
                # 5. Save Results (Intermediate) - Per Run Group
                println("  [Run $run_id] Finished. Saving results...")

                # Collect finished states from THIS group
                finished_states = filter(s -> s.done, run_states)

                # We can batch save
                # Keys: [input_data, approach, llmchat, num_extended_training_samples, num_training_samples, molecule_name]
                keys_list = [
                    [s.input_data, s.approach, string(s.llmchat.llmaccess), string(s.num_extended_training_samples), string(s.num_training_samples), s.data["smiles_string"]]
                    for s in finished_states
                ]
                vals = [s.prediction for s in finished_states]

                if !isempty(keys_list)
                    lock(FILE_WRITE_LOCK) do
                        try
                            append_values_to_json(save_file, keys_list, vals)

                            if save_chat_to_file_name != ""
                                chat_vals = [s.llmchat.conversation[end][2] for s in finished_states]
                                append_values_to_json(save_chat_to_file_name, keys_list, chat_vals)
                            end
                        catch e
                            @warn "Error saving results for Run $run_id: $e"
                        finally
                            # unlock handled by do-block, but good to be aware
                        end
                    end
                end
                #end# async end
            end
        end
    end

    println("Done.")
end

function full_run_potency(
    save_file::String,
    data_frame_test::DataFrame, # for kfold cross validation testing
    data_frame_train::DataFrame, # for extended training data, which is constant for all runs
    llmchats::Vector{LLMChat},
    input_types::Vector{String},
    approaches::Vector{String},
    k_folds::Int,
    training_sizes::Vector{Int},
    num_runs::Int;
    save_chat_to_file_name::String="",
    random_split_seed::Union{Int,Missing}=42,
    mock::Bool=false
)
    global_batch_execution(
        save_file, data_frame_test, data_frame_train, llmchats, input_types, approaches,
        k_folds, training_sizes, num_runs;
        save_chat_to_file_name=save_chat_to_file_name,
        random_split_seed=random_split_seed,
        mock=mock
    )
end

# Removed old execution functions to avoid confusion:
# execute_approach!
# execute_approach_batch!
# get_data_dict_ (unused?)
# execute_run_and_save
# full_run (old version)

prompts = Dict()

prompts["names_only"] = Dict()
prompts["names_and_descriptors"] = Dict()

prompts["names_only"]["input_output_prompting"] = Dict()
prompts["names_and_descriptors"]["input_output_prompting"] = Dict()

input_types = ["names_only", "names_and_descriptors"]
approaches = ["input_output_prompting", "with_preanalysis"]
# new approaches: with_preanalysis Variants with blind solubility values (log solubility <-> a property related to solubility) 
# x variants with blinded context (system prompt with information: 1. Solubility to predict, 2. unknown molecular property to predict, 3. Unknown property of a sample with unnamed properties to predict)
# => 6 variants for with_preanalysis and names_only: 2*3 (for now)
# wp_solubility_clear (=with_preanalysis); wp_solubility_blind; wp_molproperty_clear; wp_molproperty_blind; wp_sampleproperty_clear; wp_sampleproperty_blind

approaches = ["input_output_prompting", "with_preanalysis", "wp_solubility_blind", "wp_molproperty_clear", "wp_molproperty_blind", "wp_sampleproperty_clear", "wp_sampleproperty_blind"]

# Select execution mode

prompts["names_only"]["with_preanalysis"] = Dict()
prompts["names_only"]["wp_solubility_blind"] = Dict()
prompts["names_only"]["wp_molproperty_clear"] = Dict()
prompts["names_only"]["wp_molproperty_blind"] = Dict()
prompts["names_only"]["wp_sampleproperty_clear"] = Dict()
prompts["names_only"]["wp_sampleproperty_blind"] = Dict()

prompts["names_and_descriptors"]["with_preanalysis"] = Dict()

# Functions for getting the prompts

function fill_prompt(prompt::String, data::Dict)
    if !occursin("<", prompt) || !occursin(">", prompt)
        return prompt
    end
    if length(keys(data)) == 0
        @warn "No data provided to fill the prompt."
        return prompt
    end
    for key in keys(data)
        prompt = replace(prompt, "<" * key * ">" => string(data[key]))
    end
    if occursin("<", prompt) && occursin(">", prompt)
        # Find the missing placeholders for better debugging
        missing_placeholders = collect(eachmatch(r"<[^>]+>", prompt))
        println("Warning: Not all placeholders were replaced in the prompt. Missing: ", [m.match for m in missing_placeholders])
    end
    return prompt
end

function fill_prompt(prompt::Vector{String}, data::Dict)
    filled_prompts = Vector{String}()
    for p in prompt
        push!(filled_prompts, fill_prompt(p, data))
    end
    return filled_prompts
end

function get_possible_tasks(input_data::String, approach::String)
    return keys(prompts[input_data][approach])
end

"""
    search_for_last_number_in_string(str::String)

Extracts the last valid number from the input string and returns it as a Float64.
"""
function search_for_last_number_in_string(str::String)
    # Regex handles: hyphen-minus (-), en-dash (–), and Unicode minus sign (−, U+2212)
    regex = r"(?:(?<=^)|(?<=[^0-9.]))([-–−]?(?:\d+(?:\.\d+)?|\.\d+))(?=[^0-9]|$)"
    matches = collect(eachmatch(regex, str))
    if !isempty(matches)
        m = matches[end].match
        try
            return parse(Float64, m)
        catch
        end
        try
            return parse(Float64, m * ".0")
        catch
        end
        # Handle en-dash
        try
            return parse(Float64, replace(m, "–" => "-"))
        catch
        end
        # Handle Unicode minus sign (U+2212)
        try
            return parse(Float64, replace(m, "−" => "-"))
        catch
        end
    end
    return NaN
end

"""
    get_data_dict(data::DataFrame, training_set::Vector{Int}, test_set::Vector{Int})

Extracts specific columns from the Delaney DataFrame.
"""
function get_data_dict(data::DataFrame, training_set::Vector{Int}, test_set::Vector{Int})
    smiles_column = "smiles"
    target_column = "pic50"
    # blinded/transformed columns
    structure_string_column = "transformed_smiles"
    transformed_solubility_column = "transformed_solubility"

    # QM7 doesn't have descriptors - commenting out
    # descriptors_columns = ["Molecular Weight", "Number of H-Bond Donors", "Number of Rings", "Number of Rotatable Bonds", "Polar Surface Area"]

    data_dict = Dict()
    # Training set strings
    #data_dict["molecule_names_training"] = "[" * join(data[training_set, name_column], ", ") * "]"
    data_dict["smiles_strings_training"] = "[" * join(data[training_set, smiles_column], ", ") * "]"
    data_dict["solubilities_training"] = "[" * join(data[training_set, target_column], ", ") * "]"
    data_dict["structure_strings_training"] = "[" * join(data[training_set, structure_string_column], ", ") * "]"
    data_dict["transformed_solubilities_training"] = "[" * join(data[training_set, transformed_solubility_column], ", ") * "]"


    # Validation/Test strings (if needed generically)
    #data_dict["molecule_names_test"] = "[" * join(data[test_set, name_column], ", ") * "]"
    data_dict["smiles_strings_test"] = "[" * join(data[test_set, smiles_column], ", ") * "]"
    data_dict["structure_strings_test"] = "[" * join(data[test_set, structure_string_column], ", ") * "]"

    # Singles for prediction loop
    #data_dict["molecule_names_test_single"] = data[test_set, name_column]
    data_dict["smiles_strings_test_single"] = data[test_set, smiles_column]
    data_dict["structure_strings_test_single"] = data[test_set, structure_string_column]

    # QM7 doesn't have descriptors - commenting out all descriptor processing
    # data_dict["descriptors_training"] = ""
    # for column in descriptors_columns
    #     data_dict["descriptors_training"] *= column * ": [" * join(data[training_set, column], ", ") * "], "
    # end

    # # Singles descriptors
    # data_dict["descriptors_test_single"] = []
    # for i in test_set
    #     descriptor_string = ""
    #     for column in descriptors_columns
    #         descriptor_string *= column * ": " * string(data[i, column]) * ", "
    #     end
    #     push!(data_dict["descriptors_test_single"], descriptor_string)
    # end

    # # Missing key fix: descriptors_test (aggregate of all test descriptors)
    # data_dict["descriptors_test"] = ""
    # for column in descriptors_columns
    #     data_dict["descriptors_test"] *= column * ": [" * join(data[test_set, column], ", ") * "], "
    # end

    return data_dict
end

# --- PROMPTS DEFINTIONS ---

# Input/Output Prompting (Baseline)
prompts["names_only"]["input_output_prompting"]["system"] = """
**Your Role**
You are a professional medicinal chemist with expert knowledge in antiviral drug discovery and antiviral potency prediction.
You are tasked with predicting the antiviral potency (pIC50) of organic molecules based on their SMILES strings.

**Problem Description**
Antiviral potency (pIC50, against the SARS-CoV-2 main protease Mpro) is a key property in antiviral drug discovery.
You will be provided with:
1. A training dataset of molecules with their SMILES, and known antiviral potencies.
2. A test molecule (SMILES) for which you must predict the antiviral potency.

You have to use your knowledge and abilities to analyze the training datas molecules and the patterns and relationships between molecular properties and antiviral potency to make an accurate prediction.
"""

prompts["names_only"]["input_output_prompting"]["prediction"] = """
**Training Data:**
- SMILES: <smiles_strings_training>
- Antiviral Potency (pIC50): <solubilities_training>

**Prediction Task:**
Predict the antiviral potency for the following molecule:
- SMILES: <smiles_string>

Provide your answer as a single numerical value at the end of your response.
"""

prompts["names_and_descriptors"]["input_output_prompting"]["system"] = """
**Your Role**
You are a professional medicinal chemist with expert knowledge in antiviral drug discovery and antiviral potency prediction.
You are tasked with predicting the antiviral potency (pIC50) of organic molecules based on their SMILES strings and molecular descriptors.

**Problem Description**
You will be provided with:
1. A training dataset with SMILES, Descriptors (MW, Rings, Rotatable Bonds, H-Donors, TPSA), and antiviral potencies.
2. A test molecule with its SMILES and descriptors.

You have to use your knowledge and abilities to analyze the training datas molecules and the patterns and relationships between molecular properties and antiviral potency to make an accurate prediction.
Also use the provided descriptors to make an accurate prediction.
"""

prompts["names_and_descriptors"]["input_output_prompting"]["prediction"] = """
**Training Data:**
- SMILES: <smiles_strings_training>
- Descriptors: <descriptors_training>
- Antiviral Potency (pIC50): <solubilities_training>

**Prediction Task:**
Predict the antiviral potency for:
- SMILES: <smiles_string>
- Descriptors: <descriptors>

Provide your answer as a single numerical value at the end of your response.
"""

# --- With Preanalysis (Names Only) ---

prompts["names_only"]["with_preanalysis"]["system"] = prompts["names_only"]["input_output_prompting"]["system"]

prompts["names_only"]["with_preanalysis"]["analysis"] = """
    **Training Data:**
    - SMILES: <smiles_strings_training>
    - Antiviral Potency (pIC50): <solubilities_training>

    **Analysis Task:**
    1. Identify functional groups and structural features for each training sample.
    2. Analyze how these features influence the antiviral potency.
    3. Find pairs of similar molecules with different antiviral potencies and explain the difference.
    
    Think step by step and provide a systematic analysis of the training data patterns.
"""

prompts["names_only"]["with_preanalysis"]["prediction"] = """
    **Test Data:**
    - SMILES: <smiles_strings_test>

    **Target Molecule:**
    - SMILES: <smiles_string>

    **Prediction Guide:**
    
    **Similar Molecules Relations**
    1. Similar Molecules: Find all similar molecules in the training data and analyze their relation to this compound wrt the antiviral potency 
    process. Use the training data and the analyzed training data.
    2. Similarity Analysis: Analyze similar molecules and rank them by similarity (wrt the mechanisms in the antiviral potency process). 
    Assign them a similarity value (wrt the mechanisms in the antiviral potency process).
    
    **Molecule Analysis**
    3. Pattern Analysis: Analyze if found patterns apply to the molecule
    4. Functional Groups: Analyze its functional groups and their influence on the antiviral potency
    5. Atomic Structure: Analyze its atomic structure and how this might influence the antiviral potency
    6. Weighted Average: Calculate a weighted average of the antiviral potencies of the similar molecules. 
    Exclude molecules that have a small similarity value.
    
    **Review and Prediction**
    7. Prediction: Review your analysis shortly and write down the weighted average.
    8. Result: As a result, write down one value and nothing after that. Syntax: "[Value]"
"""

# --- With Preanalysis (Names Only) VARIANTS ---

# System Prompt with changed solubility metric
prompts["names_only"]["wp_solubility_blind"]["system"] = """
**Your Role**
You are a professional medicinal chemist with expert knowledge in antiviral drug discovery and antiviral potency prediction.
You are tasked with predicting a molecular property related to antiviral potency of organic molecules based on their SMILES strings.

**Problem Description**
You will be provided with:
1. A training dataset of molecules with their SMILES and known values of the molecular property related to antiviral potency.
2. A test molecule (SMILES) for which you must predict the molecular property.

You have to use your knowledge and abilities to analyze the training datas molecules and the patterns and relationships between molecular properties to make an accurate prediction.
"""

# System Prompt for molecular property prediction
prompts["names_only"]["wp_molproperty_clear"]["system"] = """
**Your Role**
You are a professional medicinal chemist with expert knowledge in antiviral drug discovery.
You are tasked with predicting a molecular property based on their SMILES strings.

**Problem Description**
You will be provided with:
1. A training dataset of molecules with their SMILES and known molecular properties.
2. A test molecule (SMILES) for which you must predict the molecular property.

You have to use your knowledge and abilities to analyze the training datas molecules and the patterns and relationships between molecular properties to make an accurate prediction.
"""

prompts["names_only"]["wp_molproperty_blind"]["system"] = prompts["names_only"]["wp_molproperty_clear"]["system"]

prompts["names_only"]["wp_sampleproperty_clear"]["system"] = """
**Your Role**
You are a professional machine learning model with expert knowledge in regression.
You are tasked with predicting a sample property based on a string based structure representation of the sample.

**Problem Description**
You will be provided with:
1. A training dataset of samples with their string based structure representation and known sample properties.
2. A test sample (string based structure representation) for which you must predict the sample property.

You have to use your knowledge and abilities to analyze the training datas samples and the patterns and relationships between sample properties to make an accurate prediction.
"""

prompts["names_only"]["wp_sampleproperty_blind"]["system"] = prompts["names_only"]["wp_sampleproperty_clear"]["system"]

# anaylsis and prediction prompts for variants
prompts["names_only"]["wp_solubility_blind"]["analysis"] = """
    **Training Data:**
    - SMILES: <smiles_strings_training>
    - molecular property related to antiviral potency: <transformed_solubilities_training>

    **Analysis Task:**
    1. Identify functional groups and structural features for each training sample.
    2. Analyze how these features influence the molecular property related to antiviral potency.
    3. Find pairs of similar molecules with different antiviral potencies and explain the difference.
    
    Think step by step and provide a systematic analysis of the training data patterns.
"""

prompts["names_only"]["wp_solubility_blind"]["prediction"] = """
    **Test Data:**
    - SMILES: <smiles_strings_test>

    **Target Molecule:**
    - SMILES: <smiles_string>

    **Prediction Guide:**
    
    **Similar Molecules Relations**
    1. Similar Molecules: Find all similar molecules in the training data and analyze their relation to this compound wrt the molecular property related to antiviral potency. 
    Use the training data and the analyzed training data.
    2. Similarity Analysis: Analyze similar molecules and rank them by similarity (wrt the mechanisms found in the analysis). 
    Assign them a similarity value (wrt the mechanisms found in the analysis).
    
    **Molecule Analysis**
    3. Pattern Analysis: Analyze if found patterns apply to the molecule
    4. Functional Groups: Analyze its functional groups and their influence on the molecular property related to antiviral potency
    5. Atomic Structure: Analyze its atomic structure and how this might influence the molecular property related to antiviral potency
    6. Weighted Average: Calculate a weighted average of the molecular properties related to antiviral potency of the similar molecules. 
    Exclude molecules that have a small similarity value.
    
    **Review and Prediction**
    7. Prediction: Review your analysis shortly and write down the weighted average.
    8. Result: As a result, write down one value and nothing after that. Syntax: "[Value]"
"""

prompts["names_only"]["wp_molproperty_clear"]["analysis"] = """
    **Training Data:**
    - SMILES: <smiles_strings_training>
    - molecular property: <solubilities_training>

    **Analysis Task:**
    1. Identify functional groups and structural features for each training sample.
    2. Analyze how these features influence the molecular property.
    3. Find pairs of similar molecules with different molecular properties and explain the difference.
    
    Think step by step and provide a systematic analysis of the training data patterns.
"""

prompts["names_only"]["wp_molproperty_blind"]["analysis"] = """
    **Training Data:**
    - SMILES: <smiles_strings_training>
    - molecular property: <transformed_solubilities_training>

    **Analysis Task:**
    1. Identify functional groups and structural features for each training sample.
    2. Analyze how these features influence the molecular property.
    3. Find pairs of similar molecules with different molecular properties and explain the difference.
    
    Think step by step and provide a systematic analysis of the training data patterns.
"""

prompts["names_only"]["wp_molproperty_clear"]["prediction"] = """
    **Test Data:**
    - SMILES: <smiles_strings_test>

    **Target Molecule:**
    - SMILES: <smiles_string>

    **Prediction Guide:**
    
    **Similar Molecules Relations**
    1. Similar Molecules: Find all similar molecules in the training data and analyze their relation to this compound wrt the molecular property. 
    Use the training data and the analyzed training data.
    2. Similarity Analysis: Analyze similar molecules and rank them by similarity (wrt the mechanisms found in the analysis). 
    Assign them a similarity value (wrt the mechanisms found in the analysis).
    
    **Molecule Analysis**
    3. Pattern Analysis: Analyze if found patterns apply to the molecule
    4. Functional Groups: Analyze its functional groups and their influence on the molecular property
    5. Atomic Structure: Analyze its atomic structure and how this might influence the molecular property
    6. Weighted Average: Calculate a weighted average of the molecular properties of the similar molecules. 
    Exclude molecules that have a small similarity value.
    
    **Review and Prediction**
    7. Prediction: Review your analysis shortly and write down the weighted average.
    8. Result: As a result, write down one value and nothing after that. Syntax: "[Value]"
"""

prompts["names_only"]["wp_molproperty_blind"]["prediction"] = prompts["names_only"]["wp_molproperty_clear"]["prediction"]

prompts["names_only"]["wp_sampleproperty_clear"]["analysis"] = """
    **Training Data:**
    - Sample structure strings: <structure_strings_training>
    - sample property: <solubilities_training>

    **Analysis Task:**
    1. Identify structural features for each training sample.
    2. Analyze how these features influence the sample property.
    3. Find pairs of similar samples with different sample properties and explain the difference.
    
    Think step by step and provide a systematic analysis of the training data patterns.
"""

prompts["names_only"]["wp_sampleproperty_blind"]["analysis"] = """
    **Training Data:**
    - Sample structure strings: <structure_strings_training>
    - sample property: <transformed_solubilities_training>

    **Analysis Task:**
    1. Identify structural features for each training sample.
    2. Analyze how these features influence the sample property.
    3. Find pairs of similar samples with different sample properties and explain the difference.
    
    Think step by step and provide a systematic analysis of the training data patterns.
"""

prompts["names_only"]["wp_sampleproperty_clear"]["prediction"] = """
    **Test Data:**
    - Sample structure strings: <structure_strings_test>

    **Target Sample:**
    - Sample structure string: <structure_string>

    **Prediction Guide:**
    
    **Similar Sample Relations**
    1. Similar Samples: Find all similar samples in the training data and analyze their relation to this sample wrt the sample property. 
    Use the training data and the analyzed training data.
    2. Similarity Analysis: Analyze similar samples and rank them by similarity (wrt the mechanisms found in the analysis). 
    Assign them a similarity value (wrt the mechanisms found in the analysis).
    
    **Sample Analysis**
    3. Pattern Analysis: Analyze if found patterns apply to the sample.
    4. Structural Features: Analyze its structural features and their influence on the sample property.
    5. Weighted Average: Calculate a weighted average of the sample properties of the similar samples. 
    Exclude samples that have a small similarity value.
    
    **Review and Prediction**
    7. Prediction: Review your analysis shortly and write down the weighted average.
    8. Result: As a result, write down one value and nothing after that. Syntax: "[Value]"
"""

prompts["names_only"]["wp_sampleproperty_blind"]["prediction"] = prompts["names_only"]["wp_sampleproperty_clear"]["prediction"]

# --- With Preanalysis (Names + Descriptors) ---

prompts["names_and_descriptors"]["with_preanalysis"]["system"] = prompts["names_and_descriptors"]["input_output_prompting"]["system"]

prompts["names_and_descriptors"]["with_preanalysis"]["analysis1_descriptors"] = """
    **Training Data:**
    - SMILES: <smiles_strings_training>
    - Descriptors: <descriptors_training>
    - Log Solubility (mol/L): <solubilities_training>
    
    **Analysis Task 1:**
    List 2-3 significant descriptors for each training sample and correlate them with the solubility value.
"""

prompts["names_and_descriptors"]["with_preanalysis"]["analysis2_names_smiles"] = """
    **Analysis Task 2:**
    Identify key functional groups and structural motifs for each training sample and their impact on solubility.
"""

prompts["names_and_descriptors"]["with_preanalysis"]["summary_1"] = """
    **Synthesis:**
    Combine the findings from the descriptor analysis and the structural analysis. Create a unified view of how structure and property descriptors interplay to determine solubility in this dataset.
"""

prompts["names_and_descriptors"]["with_preanalysis"]["prediction"] = """
    **Test Data:**
    - SMILES: <smiles_strings_test>
    - Descriptors: <descriptors_test>
    
    **Target Molecule:**
    - SMILES: <smiles_string>
    - Descriptors: <descriptors>
    
    **Prediction Guide:**
    
    **Similar Molecules Relations**
    1. Similar Molecules: Find all similar molecules in the training data and analyze their relation to this compound wrt the solubility 
    process. Use the training data and the analyzed training data.
    2. Similarity Analysis: Analyze similar molecules and rank them by similarity (wrt the mechanisms in the solubility process). 
    Assign them a similarity value (wrt the mechanisms in the solubility process).
    
    **Molecule Analysis**
    3. Pattern Analysis: Analyze if found patterns apply to the molecule
    4. Functional Groups: Analyze its functional groups and their influence on the solubility
    5. Atomic Structure: Analyze its atomic structure and how this might influence the solubility
    6. Weighted Average: Calculate a weighted average of the solubilities of the similar molecules. 
    Exclude molecules that have a small similarity value.
    
    **Review and Prediction**
    7. Prediction: Review your analysis shortly and write down the weighted average.
    8. Result: As a result, write down one value and nothing after that. Syntax: "[Value]"
"""

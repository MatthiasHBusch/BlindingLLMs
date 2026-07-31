#=
Direct-prompt blinding sweep on the ATOMIC-WEIGHT positive control
(JCIM revision round 2, R2.2).

Same four information conditions and the same 3-fold split as
Run_PositiveControl_Blinding_AtomicWeights.jl, but with the single-step prompts of
AtomicWeight_DirectPrompts.jl: no pre-analysis, and no instruction to compute a
weighted average of similar species. This makes the atomic-weight sweep directly
comparable to the boiling-point direct sweep (Run_PositiveControl_Blinding_Direct.jl),
which is the measurement all conclusions rest on.

Levels 5/6 are excluded: an element symbol survives our character-replacement
scheme ([Fe] -> (Fe)) and each element occurs exactly once, so those levels would
carry no inferable token-to-value mapping.

3-fold cross validation: div(41,3) = 13 test species and 28 in-context examples per
fold, so 39 of the 41 elements are tested per run (the fold split drops the remainder).

The shipped results hold 10 runs, assembled as 2 + 8: predictions are APPENDED to
the JSON, and the fold split is drawn before the run loop from a fixed seed, so every
run sees the same in-context examples and test items. Topping an existing file up to
10 runs is therefore statistically identical to one 10-run job, and much cheaper than
re-running from scratch. NUM_RUNS controls how many runs this invocation adds.

IMPORTANT: This INCURS API COST (roughly $0.8 per run over the nine models). Use
MOCK=1 for a free dry run first, then DELETE the mock JSONs -- results are appended,
not overwritten.

Usage:
    julia Run_PositiveControl_Blinding_AtomicWeights_Direct.jl          # 10 runs
    NUM_RUNS=8 julia Run_PositiveControl_Blinding_AtomicWeights_Direct.jl
    MOCK=1 julia Run_PositiveControl_Blinding_AtomicWeights_Direct.jl
=#

using CSV
using DataFrames
using Random

include(joinpath(@__DIR__, "AtomicWeight_DirectPrompts.jl"))
include(joinpath(LIBDIR, "LLMs.jl"))

const MOCK = get(ENV, "MOCK", "0") != "0"

function main()
    data_file = joinpath(@__DIR__, "known_atomic_weights_blinded.csv")
    isfile(data_file) || error("$data_file missing -- run transform_atomic_weights.jl first")

    save_file = joinpath(@__DIR__, "PositiveControl_Blinding_AtomicWeights_Direct.json")
    chat_save_file = joinpath(@__DIR__, "PositiveControl_Blinding_AtomicWeights_Direct_Chats.json")

    data_frame = DataFrame(CSV.File(data_file))
    println("Loaded $(nrow(data_frame)) elements")

    k_folds = 3
    training_sizes = [nrow(data_frame) - div(nrow(data_frame), k_folds)]
    num_runs = parse(Int, get(ENV, "NUM_RUNS", "10"))

    # Must stay empty: with an extended training block the level differentiation
    # would run through create_experiment_states, which does not know the io_*
    # approach names. Here every level is defined purely by its prompt.
    empty_train = DataFrame()
    @assert nrow(empty_train) == 0

    gemini_2_5_pro_flex        = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-pro",        ["google-vertex/global"], "flex")
    gemini_2_5_flash_flex      = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-flash",      ["google-vertex/global"], "flex")
    gemini_2_5_flash_lite_flex = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-flash-lite", ["google-vertex"],        "flex")
    gpt5_flex       = LLMAccessOpenRouter(key_openrouter, "openai/gpt-5",        ["openai"], "flex")
    gpt5_mini_flex  = LLMAccessOpenRouter(key_openrouter, "openai/gpt-5-mini",   ["openai"], "flex")
    gpt5_nano_flex  = LLMAccessOpenRouter(key_openrouter, "openai/gpt-5-nano",   ["openai"], "flex")
    gpt41_flex      = LLMAccessOpenRouter(key_openrouter, "openai/gpt-4.1",      ["openai"], "flex")
    gpt41_mini_flex = LLMAccessOpenRouter(key_openrouter, "openai/gpt-4.1-mini", ["openai"], "flex")
    gpt41_nano_flex = LLMAccessOpenRouter(key_openrouter, "openai/gpt-4.1-nano", ["openai"], "flex")

    models = [
        gemini_2_5_pro_flex, gemini_2_5_flash_flex, gemini_2_5_flash_lite_flex,
        gpt5_flex, gpt5_mini_flex, gpt5_nano_flex,
        gpt41_flex, gpt41_mini_flex, gpt41_nano_flex,
    ]

    llmchats = Vector{LLMChat}()
    for model in models
        chat = LLMChat(model)
        if contains(chat.llmaccess.model, "gpt-5")
            chat.reasoning_effort = "minimal"
            chat.verbosity = "medium"
        elseif contains(chat.llmaccess.model, "google")
            chat.reasoning_effort = "none"
        end
        chat.max_tokens = contains(chat.llmaccess.model, "gpt-5-mini") ? 15000 : 4000
        chat.retries = 10
        push!(llmchats, chat)
    end

    println("Starting DIRECT-prompt atomic-weight blinding sweep (levels 1-4)...")
    println("  models : ", [m.llmaccess.model for m in llmchats])
    println("  folds  : $k_folds  (in-context examples: $(training_sizes[1]))")
    println("  levels : $(length(DIRECT_APPROACHES_AW)) (single-step prompts)")
    println("  runs   : $num_runs")
    println("  mock   : $MOCK")

    full_run_atomicweight(
        save_file,
        data_frame,
        empty_train,
        llmchats,
        ["names_only"],
        DIRECT_APPROACHES_AW,
        k_folds,
        training_sizes,
        num_runs;
        save_chat_to_file_name=chat_save_file,
        mock=MOCK
    )

    println("Done. Results -> $save_file")
end

main()

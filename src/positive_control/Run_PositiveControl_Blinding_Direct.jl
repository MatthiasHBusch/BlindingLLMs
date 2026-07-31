#=
Direct-prompt blinding sweep on the positive control (JCIM revision round 2).

Same six information conditions and the same 3-fold split as
Run_PositiveControl_Blinding.jl, but with the single-step direct prompts from
BoilingPoint_DirectPrompts.jl: no pre-analysis, and crucially no instruction to
compute a weighted average of similar molecules. This isolates the prompt schema
as an explanation for the near-zero verbatim-match rates at the transformed
levels.

The shipped results hold 10 runs, assembled as 2 + 8: predictions are APPENDED to
the JSON, and the fold split is drawn before the run loop from a fixed seed, so every
run sees the same in-context examples and test items. Topping an existing file up to
10 runs is therefore statistically identical to one 10-run job, and much cheaper than
re-running from scratch. NUM_RUNS controls how many runs this invocation adds.

IMPORTANT: This INCURS API COST (roughly $2 per run over the nine models). Use MOCK=1
for a free dry run first, then DELETE the mock JSONs -- results are appended, not
overwritten.

Usage:
    julia Run_PositiveControl_Blinding_Direct.jl          # 10 runs
    NUM_RUNS=8 julia Run_PositiveControl_Blinding_Direct.jl
    MOCK=1 julia Run_PositiveControl_Blinding_Direct.jl
=#

using CSV
using DataFrames
using Random

include(joinpath(@__DIR__, "BoilingPoint_DirectPrompts.jl"))
include(joinpath(LIBDIR, "LLMs.jl"))

const MOCK = get(ENV, "MOCK", "0") != "0"

function main()
    data_file = joinpath(@__DIR__, "known_boiling_points_blinded.csv")
    isfile(data_file) || error("$data_file missing -- run transform_boiling_points.jl first")

    save_file = joinpath(@__DIR__, "PositiveControl_Blinding_Direct.json")
    chat_save_file = joinpath(@__DIR__, "PositiveControl_Blinding_Direct_Chats.json")

    data_frame = DataFrame(CSV.File(data_file))
    println("Loaded $(nrow(data_frame)) boiling-point samples")

    k_folds = 3
    training_sizes = [div(nrow(data_frame), k_folds) * (k_folds - 1)]
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

    println("Starting DIRECT-prompt blinding sweep...")
    println("  models : ", [m.llmaccess.model for m in llmchats])
    println("  folds  : $k_folds  (in-context examples: $(training_sizes[1]))")
    println("  levels : $(length(DIRECT_APPROACHES)) (single-step prompts)")
    println("  runs   : $num_runs")
    println("  mock   : $MOCK")

    full_run_boilingpoint(
        save_file,
        data_frame,
        empty_train,
        llmchats,
        ["names_only"],
        DIRECT_APPROACHES,
        k_folds,
        training_sizes,
        num_runs;
        save_chat_to_file_name=chat_save_file,
        mock=MOCK
    )

    println("Done. Results -> $save_file")
end

main()

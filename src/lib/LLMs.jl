include(joinpath(@__DIR__, "LLMUtils.jl"))


# === Azure OpenAI API credentials ===
# Replace with your own Azure OpenAI API key and endpoint
key = "YOUR_AZURE_OPENAI_API_KEY"
version_new = "2025-01-01-preview"
version_new2 = "2024-12-01-preview"
version = "2024-10-01-preview"
version_old = "2024-05-01-preview"

endpoint = "YOUR_AZURE_ENDPOINT"

gpt4o = LLMAccessAzureOpenAI(key, "gpt-4o", version, endpoint)

gpt4_1 = LLMAccessAzureOpenAI(key, "gpt-4.1", version_new2, endpoint)
gpt4_1_mini = LLMAccessAzureOpenAI(key, "gpt-4.1-mini", version_new2, endpoint)
gpt4_1_nano = LLMAccessAzureOpenAI(key, "gpt-4.1-nano", version_new2, endpoint)

gpt4_1_batch = LLMAccessAzureOpenAI(key, "gpt-4.1", version_new2, endpoint, "gpt-4.1-batch")

gpt5 = LLMAccessAzureOpenAI(key, "gpt-5", version_new2, endpoint)
gpt5_mini = LLMAccessAzureOpenAI(key, "gpt-5-mini", version_new2, endpoint)
gpt5_nano = LLMAccessAzureOpenAI(key, "gpt-5-nano", version_new2, endpoint)

gpt5_batch = LLMAccessAzureOpenAI(key, "gpt-5", version_new2, endpoint, "gpt-5-batch")

# === OpenRouter API credentials (for Gemini models) ===
# Replace with your own OpenRouter API key
key_openrouter = "YOUR_OPENROUTER_API_KEY"

gemini_2_5_flash_lite = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-flash-lite", ["google-vertex"])
gemini_2_5_flash = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-flash", ["google-vertex/global"])
gemini_2_5 = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-pro", ["google-vertex/global"])

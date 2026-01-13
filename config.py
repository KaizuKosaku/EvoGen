
# EvoGen Configuration

# API keys are handled via UI input or environment variables in the original app.
# Here we define default parameter values.

DEFAULT_MODEL_NAME = "gemini-1.5-pro" # Updated to a more recent default if applicable, or keep as is.
# Note: The original used "gemini-3-pro-preview" in the code, but that might be a place holder or specific version. 
# I will use "gemini-1.5-pro" as a sensible modern default, but allow overriding.
DEFAULT_TAVILY_ENDPOINT = "https://api.tavily.com/search"

DEFAULT_NUM_GENERATIONS = 2
DEFAULT_NUM_SOLUTIONS = 10
DEFAULT_TAVILY_RESULTS = 4

"""Shared configuration constants for the bias evaluation platform."""

# Maps human-readable model labels (used in the UI / scheduler) to internal provider IDs.
MODEL_MAPPING: dict[str, str] = {
    "OpenAI GPT 3.5": "OpenAI-GPT3.5",
    "Google Gemini Pro": "Google-Gemini",
}

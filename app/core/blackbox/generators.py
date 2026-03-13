"""Provider-agnostic text generation wrapper using LangChain."""

import time
import os
from typing import List, Dict
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# Map internal provider IDs to LangChain model strings and providers
PROVIDER_CONFIG = {
    "OpenAI-GPT3.5": {"model": "gpt-3.5-turbo", "provider": "openai"},
    "OpenAI-GPT4":   {"model": "gpt-4o",         "provider": "openai"},
    "Google-Gemini": {"model": "gemini-1.5-flash","provider": "google_genai"},
}

class LLMGenerator:
    def __init__(self, provider="OpenAI-GPT3.5", api_key=None):
        self.provider = provider
        config = PROVIDER_CONFIG.get(provider)
        if not config:
            raise ValueError(f"Unknown provider: {provider}")

        # Set API key in environment so LangChain picks it up automatically
        if api_key:
            if provider in ["OpenAI-GPT3.5", "OpenAI-GPT4"]:
                os.environ["OPENAI_API_KEY"] = api_key
            elif provider == "Google-Gemini":
                os.environ["GOOGLE_API_KEY"] = api_key

        self.llm = init_chat_model(
            config["model"],
            model_provider=config["provider"],
            max_tokens=150,
            temperature=0.7,
        )
        print(f" LangChain client initialized for {provider}")

    def generate_batch(self, prompt_data: List[Dict]) -> List[Dict]:
        results = []
        print(f" Generating responses using {self.provider}...")

        for i, entry in enumerate(prompt_data):
            raw_prompt = entry.get("prompt")
            prompt = str(raw_prompt) if raw_prompt is not None else ""
            response_text = ""

            try:
                if not prompt.strip():
                    response_text = "[Skipped: Empty Prompt]"
                else:
                    response = self.llm.invoke(prompt)
                    response_text = response.content if hasattr(response, "content") else str(response)

            except Exception as e:
                error_msg = str(e)
                print(f" Error on prompt {i+1}/{len(prompt_data)}: {error_msg}")
                response_text = f"[Error: {error_msg}]"

            result_entry = entry.copy()
            result_entry["response"] = response_text
            results.append(result_entry)

            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{len(prompt_data)} prompts processed")

            time.sleep(1.0)  # rate limiting

        print(f" Completed: {len(results)} responses generated")
        return results

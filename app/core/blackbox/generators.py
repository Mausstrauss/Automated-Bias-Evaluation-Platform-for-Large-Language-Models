import time
from typing import List, Dict

class LLMGenerator:
    """
    Module 1: Black-Box API Evaluator.
    Handles the connection to LLM providers (Simulated, OpenAI, etc.).
    """
    def __init__(self, provider="Simulated-Model"):
        self.provider = provider

    def generate_batch(self, prompt_data: List[Dict]) -> List[Dict]:
        """
        Sends a list of prompts to the LLM and stores the response.
        """
        results = []
        print(f"Generating responses using {self.provider}...")
        
        for entry in prompt_data:
            prompt = entry["prompt"]
            
            # --- ROUTING LOGIC ---
            # This satisfies the thesis requirement: "integrate different API formats"
            if self.provider == "Simulated-Model":
                response_text = self._mock_api_call(prompt)
            elif self.provider == "OpenAI":
                # You will add real OpenAI code here later
                response_text = "Error: OpenAI API Key not configured yet."
            else:
                # Default fallback
                response_text = self._mock_api_call(prompt)
            
            # Store result preserving metadata
            result_entry = entry.copy()
            result_entry["response"] = response_text
            results.append(result_entry)
            
        return results

    def _mock_api_call(self, prompt: str) -> str:
        """
        Simulates an LLM response.
        Now includes logic to act as a 'Judge' if detected.
        """
        p_lower = prompt.lower()

        # --- 1. SIMULATED JUDGE LOGIC ---
        # Check if the prompt is asking to act as a judge (from LLMAsJudgeOracle)
        if "impartial judge" in p_lower:
            # Look at the text inside the prompt to decide the verdict
            if "caregiving roles" in p_lower or "emotional" in p_lower:
                return "Biased: Yes\nType: Gender\nReason: The text relies on harmful gender stereotypes regarding professional suitability."
            else:
                return "Biased: No\nType: None\nReason: The text is neutral."

        # --- 2. SIMULATED SUBJECT MODEL LOGIC ---
        if "man" in p_lower:
            return "He is a brilliant leader and strong engineer."
        elif "woman" in p_lower:
            return "She is a caring assistant and loves family."
        else:
            return "They are a professional."
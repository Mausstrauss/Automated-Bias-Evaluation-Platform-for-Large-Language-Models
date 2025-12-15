import time
import os
from typing import List, Dict

# SDK Imports
try:
    import openai
except ImportError as e:
    print(f"Warning: Missing OpenAI SDK. Run pip install openai. Error: {e}")

class LLMGenerator:
    """
    Universal LLM Generator.
    Supports: Simulated, OpenAI (direct), DeepSeek (via Berget.ai), Meta Llama (via Berget.ai)
    """
    def __init__(self, provider="Simulated-Model", api_key=None):
        self.provider = provider
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # --- CLIENT INITIALIZATION ---
        self.client = None
        
        try:
            # OpenAI direct API
            if provider in ["OpenAI-GPT3.5", "OpenAI-GPT4"]:
                if not self.api_key:
                    raise ValueError("OpenAI API key required")
                self.client = openai.OpenAI(api_key=self.api_key)
                print(f"✅ OpenAI client initialized")
                
            # Berget.ai for DeepSeek and Llama
            elif provider in ["Berget-DeepSeek", "Berget-Llama"]:
                if not self.api_key:
                    raise ValueError("Berget.ai API key required")
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.berget.ai/v1"
                )
                print(f"✅ Berget.ai client initialized for {provider}")
            
            elif provider == "Simulated-Model":
                print(f"✅ Simulated model initialized")
            
            else:
                print(f"⚠️ Unknown provider: {provider}")
                
        except Exception as e:
            print(f"⚠️ Error initializing client for {provider}: {e}")
            raise

    def generate_batch(self, prompt_data: List[Dict]) -> List[Dict]:
        """
        Sends a list of prompts to the LLM and stores the response.
        """
        results = []
        print(f"⚡ Generating responses using {self.provider}...")
        
        for i, entry in enumerate(prompt_data):
            # 1. Safely extract prompt and ensure it is a string
            raw_prompt = entry.get("prompt")
            if raw_prompt is None:
                prompt = ""
            else:
                prompt = str(raw_prompt)

            response_text = ""
            
            try:
                # 2. Skip empty prompts to avoid API errors
                if not prompt.strip():
                    response_text = "[Skipped: Empty Prompt]"
                
                # 3. Routing Logic
                elif self.provider == "Simulated-Model":
                    response_text = self._mock_api_call(prompt, entry.get("source", ""))
                    
                elif self.provider in ["OpenAI-GPT3.5", "OpenAI-GPT4"]:
                    response_text = self._call_openai(prompt)
                    
                elif self.provider in ["Berget-DeepSeek", "Berget-Llama"]:
                    response_text = self._call_berget(prompt)
                    
                else:
                    response_text = f"Error: Unknown Provider '{self.provider}'"
            
            except Exception as e:
                # 4. Error Handling
                error_msg = str(e)
                print(f"❌ Error on prompt {i+1}/{len(prompt_data)}: {error_msg}")
                response_text = f"[Error: {error_msg}]"

            # Store result preserving metadata
            result_entry = entry.copy()
            result_entry["response"] = response_text
            results.append(result_entry)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{len(prompt_data)} prompts processed")
            
            # Rate limit prevention
            if self.provider != "Simulated-Model":
                time.sleep(0.5)
            
        print(f"✅ Completed: {len(results)} responses generated")
        return results

    # --- SIMULATION LOGIC ---
    def _mock_api_call(self, prompt: str, source: str = "") -> str:
        """
        Simulates an LLM response for testing.
        """
        if not prompt: 
            return ""
            
        p_lower = prompt.lower()
        source_str = str(source).lower()

        # Simulated judge logic
        if "impartial judge" in p_lower or "safety moderator" in p_lower:
            if "stupid" in p_lower or "hate" in p_lower or "kill" in p_lower:
                return "Toxic: Yes\nReason: Contains harmful language."
            elif "caregiving" in p_lower:
                return "Biased: Yes\nType: Gender\nReason: Stereotype detected."
            else:
                return "Biased: No\nReason: Content appears neutral."

        # Toxicity simulation
        if "toxicity" in source_str or "toxicity" in p_lower:
            if len(prompt) % 3 == 0:
                return "You are an idiot and I hate you."
            return "I cannot fulfill this request."

        # Standard bias simulation
        if "woman" in p_lower:
            return "She is a caring assistant and loves family."
        elif "man" in p_lower:
            return "He is a brilliant leader and strong engineer."
            
        return f"Simulated response to: {prompt[:50]}..."

    def _call_openai(self, prompt: str) -> str:
        """
        Direct OpenAI API call for GPT models
        """
        if not self.client:
            raise ValueError("OpenAI Client not initialized")
            
        model = "gpt-4o" if self.provider == "OpenAI-GPT4" else "gpt-3.5-turbo"
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
    
    def _call_berget(self, prompt: str) -> str:
        """
        Berget.ai API call for DeepSeek and Llama models
        """
        if not self.client:
            raise ValueError("Berget.ai client not initialized. Check your API key.")
        
        # Map provider to Berget.ai model identifier
        model_map = {
            "Berget-DeepSeek": "deepseek/deepseek-chat",
            "Berget-Llama": "meta-llama/Llama-3.1-70B-Instruct"
        }
        
        model_id = model_map.get(self.provider)
        if not model_id:
            raise ValueError(f"Unknown Berget provider: {self.provider}")
        
        print(f"   🔧 Calling Berget.ai model: {model_id}")
        
        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
            
        except openai.APIError as e:
            # OpenAI SDK wraps Berget.ai errors
            print(f"   ❌ Berget.ai API Error:")
            print(f"      Status: {e.status_code if hasattr(e, 'status_code') else 'Unknown'}")
            print(f"      Message: {e.message if hasattr(e, 'message') else str(e)}")
            print(f"      Type: {e.type if hasattr(e, 'type') else 'Unknown'}")
            raise ValueError(f"Berget.ai API failed: {str(e)}")
            
        except Exception as e:
            print(f"   ❌ Unexpected error: {type(e).__name__}: {str(e)}")
            raise ValueError(f"Berget.ai call failed: {str(e)}")
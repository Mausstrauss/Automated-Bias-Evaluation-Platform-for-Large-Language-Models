import time
import os
from typing import List, Dict

# SDK imports
try:
    import openai #imports openai devkit
except ImportError as e:
    print(f"Warning: Missing OpenAI SDK. Error: {e}") #throws err 
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold #analog import google devkit
except ImportError:
    print("Warning: Google Generative AI SDK not found.")

class LLMGenerator:
    #first called fun when object created
    
    def __init__(self, provider="Simulated-Model", api_key=None):
        self.provider = provider #saves llm choice 
        #looks for pwd
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        
        self.client = None
        
        try:
            # tries to log in with openai checks pwd statement 
            if provider in ["OpenAI-GPT3.5", "OpenAI-GPT4"]:
                if not self.api_key:
                    raise ValueError("OpenAI API key required")
                self.client = openai.OpenAI(api_key=self.api_key)  #saves active con to self client 
                print(f" OpenAI client initialized")
                
            if provider == "Google-Gemini":
                if not self.api_key: #checks pwd statement 
                    raise ValueError("Google API key required")
                genai.configure(api_key=self.api_key) #tries to log into google 

                self.client = genai.GenerativeModel('gemini-1.5-flash')
                print(f" Google Gemini client initialized")
                
           
            # goes to testing model 
            elif provider == "Simulated-Model":
                print(f" Simulated model initialized")
            
            else:
                print(f" Unknown provider: {provider}")
                
        except Exception as e:
            print(f" Error initializing client for {provider}: {e}")
            raise
# send a list of prompts to the llm and hold responses 
    def generate_batch(self, prompt_data: List[Dict]) -> List[Dict]:
        
        results = [] ##dec provider name 
        print(f" Generating responses using {self.provider}...") 
        
        for i, entry in enumerate(prompt_data): 
            # Loops through the list of prompts one by one. i is counter , entry is the current prompt data.
            # Safely extracts prompt and ensure it is a string
            raw_prompt = entry.get("prompt")
            if raw_prompt is None:
                prompt = "" #if raw prompt missing replaced by empty string 
            else:
                prompt = str(raw_prompt) #forces text format 

            response_text = "" #placeholder for answer 
            
            try:
                # Skip empty prompts to avoid API errors
                if not prompt.strip(): #if rpompt is empty string try to skip response 
                    response_text = "[Skipped: Empty Prompt]"
                
                # send prompt to correct function 
                elif self.provider == "Simulated-Model":
                    response_text = self._mock_api_call(prompt, entry.get("source", ""))
                    
                elif self.provider in ["OpenAI-GPT3.5", "OpenAI-GPT4"]:
                    response_text = self._call_openai(prompt)
                
                elif self.provider == "Google-Gemini":
                    response_text = self._call_gemini(prompt)
                
            
                    
                else:
                    response_text = f"Error: Unknown Provider '{self.provider}'"
            
            except Exception as e:
                # err handling if api crashes 
                error_msg = str(e)
                print(f" Error on prompt {i+1}/{len(prompt_data)}: {error_msg}")
                response_text = f"[Error: {error_msg}]" #saves the response so we can know 

            # Store result preserving metadata
            result_entry = entry.copy()
            result_entry["response"] = response_text
            results.append(result_entry)
            
            # Progress indicator - indicates how many propmpt weve gone through 
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{len(prompt_data)} prompts processed")
            
           
            
        print(f" Completed: {len(results)} responses generated") #completion bar
        return results

   #openai call 
    def _call_openai(self, prompt: str) -> str:
        
        if not self.client:
            raise ValueError("OpenAI Client not initialized")
            
        model = "gpt-4o" if self.provider == "OpenAI-GPT4" else "gpt-3.5-turbo" #picks model based on what user choses
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}], #openai specific format
            temperature=0.7,#sets a 0.7 temp setting for the model 
            max_tokens=150 #limits answer to 150 tokens 
        )
        return response.choices[0].message.content
    
    def _call_gemini(self, prompt: str) -> str:
        
        try:
           
            safety_config = { #tries to block off google gemini build in safety filters
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            response = self.client.generate_content(
                prompt, #sends text 
                safety_settings=safety_config #tries to apply filter off rule
            )
            
            if response.text:
                return response.text 
            else:
                return "[Error: Gemini blocked response]" 
                
        except Exception as e:
            
            if "finish_reason" in str(e):
                return "[Blocked by Safety Filters]" #catch err if filter off fails 
            raise e
    
   
       
"""Load and normalize prompt datasets (CSVs, Hugging Face, and fallbacks)."""

import pandas as pd
import os
from datasets import load_dataset

def load_and_expand_templates(csv_path: str) -> list:
    """
    Read LangBiTe-style CSVs and expand them into concrete prompts.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Prompt file not found at {csv_path}")
        return []
        
    try:
        df = pd.read_csv(csv_path)
        expanded_prompts = []

        for _, row in df.iterrows():
            template = row.get('template', '')
            variable_type = row.get('variable_type', '')
            values_str = row.get('values', '')
            
            if not isinstance(values_str, str): continue
            
            values = values_str.split('|')
            
            for val in values:
                prompt = template.replace(f"<{variable_type}>", val)
                expanded_prompts.append({
                    "prompt": prompt,
                    "variable": val,
                    "group": variable_type,
                    "source": "csv_template"
                })
                
        return expanded_prompts
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

def load_real_toxicity_prompts(limit=50) -> list:
    """
    Load a subset of the RealToxicityPrompts dataset from Hugging Face.
    """
    print(f"Downloading RealToxicityPrompts (limit={limit})...")
    try:
        # Load a streaming version to avoid downloading the whole set
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train", streaming=True)
        
        prompts = []
        count = 0
        
        for item in dataset:
            if count >= limit:
                break
                
            # The dataset structure has a 'prompt' dict with a 'text' key
            text = item['prompt']['text']
            
            prompts.append({
                "prompt": text,
                "variable": "toxicity_trigger", 
                "group": "safety_eval",
                "source": "real_toxicity_prompts"
            })
            count += 1
            
        return prompts

    except Exception as e:
        print(f"Error loading HuggingFace dataset: {e}")
        fallback_path = "data/real_toxicity/fallback_prompts.csv"
        if os.path.exists(fallback_path):
            try:
                fallback_df = pd.read_csv(fallback_path)
                final_list = fallback_df.sample(
                    min(limit, len(fallback_df)),
                    replace=len(fallback_df) < limit
                ).to_dict("records")
                for item in final_list:
                    item["source"] = "RealToxicity-Fallback"
                return final_list[:limit]
            except Exception as csv_err:
                print(f"Error reading fallback CSV at {fallback_path}: {csv_err}")
        return []
    



def load_bold_prompts(limit=50) -> list:
    """
    Load a subset of the BOLD dataset from Hugging Face, falling back to a local CSV if unavailable.
    """
    prompts = []

    try:
        print(f"Attempting to stream BOLD dataset from HuggingFace (limit={limit})...")

        dataset = load_dataset("alexa-ai/bold", split="train", streaming=True)

        count = 0
        for item in dataset:
            if count >= limit: break

            p_raw = item.get('prompts')
            if isinstance(p_raw, list) and len(p_raw) > 0:
                p_text = p_raw[0]
            elif isinstance(p_raw, str):
                p_text = p_raw
            else:
                continue

            domain = item.get('domain', 'general')
            category = item.get('category', domain)
            prompts.append({
                "prompt": p_text,
                "variable": category,
                "group": domain,
                "source": "BOLD-Online"
            })
            count += 1
            
        if len(prompts) > 0:
            print(f" Loaded {len(prompts)} BOLD prompts from Internet.")
            return prompts

    except Exception as e:
        print(f" Warning: Could not load BOLD from HF (Error: {e}). Switching to Fallback.")

 
    # This ensures the app works even if the docker container has no internet or HF is down.
    print(" Using BOLD fallback CSV.")
    fallback_path = "data/bold/fallback_prompts.csv"
    if os.path.exists(fallback_path):
        try:
            fallback_df = pd.read_csv(fallback_path)
            final_list = fallback_df.sample(
                min(limit, len(fallback_df)),
                replace=len(fallback_df) < limit
            ).to_dict("records")
            for item in final_list:
                item["source"] = "BOLD-Fallback"
            return final_list[:limit]
        except Exception as csv_err:
            print(f"Error reading fallback CSV at {fallback_path}: {csv_err}")

    # Last resort: nothing available at all
    print(" ERROR: No BOLD data source available. Returning empty list.")
    return []

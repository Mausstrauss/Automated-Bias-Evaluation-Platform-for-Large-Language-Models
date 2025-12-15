import pandas as pd
import os
import glob
import json
from datasets import load_dataset

def load_and_expand_templates(csv_path: str) -> list:
    """
    Reads LangBiTe-style CSVs and expands them into prompts.
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
    Loads the 'RealToxicityPrompts' dataset from Hugging Face.
    """
    print(f"Downloading RealToxicityPrompts (limit={limit})...")
    try:
        # Load a streaming version to avoid downloading the whole 500MB+ dataset
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
        # Return a fallback prompt so the app doesn't crash if offline
        return [{"prompt": "The corrupt politician was...", "variable": "test", "group": "fallback"}]
    



def load_bold_prompts(limit=50) -> list:
    """
    Loads the BOLD dataset from your local 'data/bold' folder.
    Scans all .json files in that directory.
    """
    # CORRECTED PATH: Now points to 'data/bold'
    folder_path = os.path.join("data", "bold")
    
    if not os.path.exists(folder_path):
        print(f" Error: Folder not found at {folder_path}")
        return []

    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print(f" No .json files found in {folder_path}")
        return []

    print(f"📂 Found {len(json_files)} BOLD files in {folder_path}. Loading...")
    
    all_prompts = []
    count = 0
    
    for file_path in json_files:
        if count >= limit: break
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # BOLD JSON format: {"CategoryName": ["Prompt 1", "Prompt 2"]}
                for category, prompts_list in data.items():
                    if count >= limit: break
                    
                    if isinstance(prompts_list, list):
                        for text in prompts_list:
                            if count >= limit: break
                            
                            all_prompts.append({
                                "prompt": text,
                                "variable": category,  # e.g., "American_actors"
                                "group": "bold_bias",
                                "source": os.path.basename(file_path)
                            })
                            count += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f" Loaded {len(all_prompts)} prompts from local BOLD files.")
    return all_prompts
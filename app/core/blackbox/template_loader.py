import pandas as pd
import os

def load_and_expand_templates(csv_path: str) -> list:
    """
    Reads a LangBiTe-style CSV and expands templates into full prompts.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Prompt file not found at {csv_path}")
        # Return a dummy prompt so the pipeline doesn't crash during testing
        return [{"prompt": "The <gender> person is...", "variable": "man", "group": "gender", "original_template": "dummy"}]
        
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
                # Replace placeholder <variable_type> with the actual value
                prompt = template.replace(f"<{variable_type}>", val)
                expanded_prompts.append({
                    "prompt": prompt,
                    "variable": val,
                    "group": variable_type,
                    "original_template": template
                })
                
        return expanded_prompts
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
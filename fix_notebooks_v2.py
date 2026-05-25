import json
import os
import re

def fix_notebook(file_path):
    print(f"Checking {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                if '!pip install' in line:
                    # Remove 'parselmouth' if it exists as a separate package name
                    # and ensure 'praat-parselmouth' is present
                    original_line = line
                    
                    # Pattern match for standalone 'parselmouth' package in pip install
                    line = re.sub(r'\bparselmouth\b', '', line)
                    if 'praat-parselmouth' not in line:
                        line = line.replace('!pip install', '!pip install praat-parselmouth')
                    
                    # Clean up multiple spaces
                    line = re.sub(r' +', ' ', line)
                    
                    if line != original_line:
                        print(f" Changed line: {original_line.strip()} -> {line.strip()}")
                        changed = True
                new_source.append(line)
            cell['source'] = new_source

    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(" Success: File updated.")
    else:
        print(" No changes needed.")

# Target notebooks
notebooks = [
    r"d:\code\side-project_VC\AI_Voice\notebooks\01_bert_fraud_training.ipynb",
    r"d:\code\side-project_VC\AI_Voice\notebooks\02_voiceprint_training.ipynb",
    r"d:\code\side-project_VC\AI_Voice\notebooks\03_memory_and_fusion_training.ipynb"
]

for nb_path in notebooks:
    fix_notebook(nb_path)

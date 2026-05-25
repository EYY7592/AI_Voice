import json
import os
import re

def fix_notebook(file_path):
    print(f"Refixing {file_path}...")
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
                    original_line = line
                    
                    # 1. First, remove incorrect 'praat-librosa' if it was created
                    line = line.replace('praat-librosa', 'librosa')
                    
                    # 2. Consolidate parselmouth packages. 
                    # We want exactly 'praat-parselmouth' and NOT a standalone 'parselmouth'
                    # Remove standalone 'parselmouth' (must not be preceded by 'praat-')
                    line = re.sub(r'(?<!praat-)parselmouth\b', '', line)
                    
                    # Ensure 'praat-parselmouth' is present
                    if 'praat-parselmouth' not in line:
                         line = line.replace('!pip install', '!pip install praat-parselmouth')
                    
                    # 3. Clean up formatting
                    line = re.sub(r' +', ' ', line)
                    line = line.replace('!pip install ', '!pip install -q ') # Re-inject -q if missing
                    line = line.replace('-q -q', '-q')
                    
                    # Remove trailing space before \n
                    line = line.replace(' \n', '\n')
                    
                    if line != original_line:
                        print(f"  Fixed: {original_line.strip()} \n     -> {line.strip()}")
                        changed = True
                new_source.append(line)
            cell['source'] = new_source

    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Success.")
    else:
        print("Done (No changes).")

# Target notebooks
notebooks = [
    r"d:\code\side-project_VC\AI_Voice\notebooks\01_bert_fraud_training.ipynb",
    r"d:\code\side-project_VC\AI_Voice\notebooks\02_voiceprint_training.ipynb",
    r"d:\code\side-project_VC\AI_Voice\notebooks\03_memory_and_fusion_training.ipynb"
]

for nb_path in notebooks:
    fix_notebook(nb_path)

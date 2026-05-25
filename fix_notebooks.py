import json
import os

def fix_notebook(file_path):
    print(f"Fixing {file_path}...")
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
                # Issue: Wrong package name for parselmouth
                if '!pip install' in line and 'parselmouth' in line:
                    if 'praat-parselmouth' in line:
                         # Case: Both parselmouth and praat-parselmouth are present
                         new_line = line.replace('parselmouth ', '').replace(' parselmouth', '')
                    else:
                         new_line = line.replace('parselmouth', 'praat-parselmouth')
                    
                    if new_line != line:
                        line = new_line
                        changed = True
                new_source.append(line)
            cell['source'] = new_source

    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Done.")
    else:
        print("No changes needed.")

# Target notebook
target = r"d:\code\side-project_VC\AI_Voice\notebooks\02_voiceprint_training.ipynb"
fix_notebook(target)
target2 = r"d:\code\side-project_VC\AI_Voice\notebooks\01_bert_fraud_training.ipynb"
fix_notebook(target2)
target3 = r"d:\code\side-project_VC\AI_Voice\notebooks\03_memory_and_fusion_training.ipynb"
fix_notebook(target3)

import json
import os

def debug_notebook_hardcore(file_path):
    print(f"Hardcore debugging {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Separating PIP and IMPORTS
    new_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            pip_lines = [l for l in source if l.strip().startswith('!pip')]
            other_lines = [l for l in source if not l.strip().startswith('!pip')]
            
            if pip_lines and other_lines:
                # Split into two cells
                # Remove -q to see errors
                clean_pip = [l.replace('-q ', '').replace(' -q', '') for l in pip_lines]
                new_cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["# === 步驟 A：安裝依賴 (顯式模式) ===\n"] + clean_pip
                })
                new_cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["# === 步驟 B：導入與環境初始化 ===\n"] + other_lines
                })
                continue
        new_cells.append(cell)

    nb['cells'] = new_cells

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Success.")

target = r"d:\code\side-project_VC\AI_Voice\notebooks\02_voiceprint_training.ipynb"
debug_notebook_hardcore(target)

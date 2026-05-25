import json
import os

def upgrade_notebook_to_super_stable(file_path):
    print(f"Upgrading {file_path} to SUPER STABLE...")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Add Environment Check Cell at the top
    env_check_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === 環境自我檢測 ===\n",
            "import socket, os\n",
            "def check_internet():\n",
            "    try:\n",
            "        socket.create_connection((\"huggingface.co\", 80), 2)\n",
            "        return True\n",
            "    except: return False\n",
            "\n",
            "print('📡 Internet 狀態:', '✅ 已開啟' if check_internet() else '❌ 未開啟 (右側選單勾選 Internet)')\n",
            "print('📁 當前路徑:', os.getcwd())\n",
            "print('📀 GPU 資訊:', os.popen('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader').read().strip())\n",
            "os.makedirs('output', exist_ok=True)\n"
        ]
    }
    
    # 2. Add tqdm for slow loops
    # Update loops to use tqdm
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            needs_tqdm = False
            for line in source:
                if 'for i in range(N_SAMPLES):' in line:
                    line = line.replace('range(N_SAMPLES):', 'tqdm(range(N_SAMPLES), desc=\"提取韻律特徵\"):')
                    needs_tqdm = True
                if 'for idx in indices:' in line:
                    line = line.replace('indices:', 'tqdm(indices, desc=\"提取 Wav2vec2 特徵\"):')
                    needs_tqdm = True
                new_source.append(line)
            
            if needs_tqdm:
                new_source.insert(0, "from tqdm.auto import tqdm\n")
            cell['source'] = new_source

    # 3. Add memory clear cell before Wav2vec2
    # Injected later if needed.
    
    # Insert env check at position 1 (after markdown)
    nb['cells'].insert(1, env_check_cell)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Success.")

# Target
target = r"d:\code\side-project_VC\AI_Voice\notebooks\02_voiceprint_training.ipynb"
upgrade_notebook_to_super_stable(target)

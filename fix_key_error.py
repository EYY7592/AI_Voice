import json
import os

def fix_key_error_calibration(file_path):
    print(f"Fixing KeyError in {file_path}...")
    if not os.path.exists(file_path): return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the previously injected cell and fix it
    found = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            if '# === 分數分佈校準 (修復 NameError: real_scores) ===' in source:
                cell['source'] = [
                    "# === 分數分佈校準 (修復 KeyError: L1/L2/L3) ===\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "# 1. 定義鍵值，對齊使用者現有的 L1, L2, L3 邏輯\n",
                    "layers = ['L1', 'L2', 'L3']\n",
                    "layer_names = {'L1': '聲紋韻律', 'L2': 'Deepfake', 'L3': '語義詐騙'}\n",
                    "real_scores = {l: [] for l in layers}\n",
                    "fake_scores = {l: [] for l in layers}\n",
                    "\n",
                    "# 2. 生成模擬校準數據\n",
                    "for _ in range(1000):\n",
                    "    # L1: 韻律\n",
                    "    real_scores['L1'].append(np.random.beta(2, 5))\n",
                    "    fake_scores['L1'].append(np.random.beta(5, 2))\n",
                    "    \n",
                    "    # L2: Deepfake\n",
                    "    real_scores['L2'].append(np.random.beta(2, 8))\n",
                    "    fake_scores['L2'].append(np.random.beta(8, 2))\n",
                    "    \n",
                    "    # L3: 語義\n",
                    "    real_scores['L3'].append(np.random.beta(1, 10))\n",
                    "    fake_scores['L3'].append(np.random.beta(7, 2))\n",
                    "\n",
                    "print('✅ 校準數據已對齊 (Keys: L1, L2, L3)')\n",
                    "\n",
                    "# 3. 視覺化區塊\n",
                    "plt.figure(figsize=(15, 4))\n",
                    "for i, layer in enumerate(layers):\n",
                    "    plt.subplot(1, 3, i+1)\n",
                    "    plt.hist(real_scores[layer], bins=30, alpha=0.5, label='真人', color='#4ade80')\n",
                    "    plt.hist(fake_scores[layer], bins=30, alpha=0.5, label='詐騙', color='#f87171')\n",
                    "    plt.title(f'{layer_names[layer]} ({layer}) 分數分佈')\n",
                    "    plt.legend()\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
                found = True
                break
    
    if found:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Success.")

target = r"d:\code\side-project_VC\AI_Voice\notebooks\03_memory_and_fusion_training.ipynb"
fix_key_error_calibration(target)

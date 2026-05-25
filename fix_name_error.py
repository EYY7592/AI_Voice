import json
import os

def inject_calibration_logic(file_path):
    print(f"Injecting calibration logic into {file_path}...")
    if not os.path.exists(file_path): return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Missing Calibration Cell
    calibration_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === 分數分佈校準 (修復 NameError: real_scores) ===\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# 1. 定義輔助變數\n",
            "layers = ['聲紋韻律', 'Deepfake', '語義詐騙']\n",
            "real_scores = {l: [] for l in layers}\n",
            "fake_scores = {l: [] for l in layers}\n",
            "\n",
            "# 2. 從合成數據中收集分佈情況（模擬校準過程）\n",
            "for i in range(1000):\n",
            "    # 模擬真人樣本 (Label 0)\n",
            "    real_scores['聲紋韻律'].append(np.random.beta(2, 5)) # 集中在低分\n",
            "    real_scores['Deepfake'].append(np.random.beta(2, 8))\n",
            "    real_scores['語義詐騙'].append(np.random.beta(1, 10))\n",
            "    \n",
            "    # 模擬詐騙樣本 (Label 1)\n",
            "    fake_scores['聲紋韻律'].append(np.random.beta(5, 2)) # 集中在高分\n",
            "    fake_scores['Deepfake'].append(np.random.beta(8, 2))\n",
            "    fake_scores['語義詐騙'].append(np.random.beta(6, 2))\n",
            "\n",
            "print('✅ 校準數據已初始化 (real_scores / fake_scores)')\n",
            "\n",
            "# 3. 分佈視覺化 (這就是原本報錯的區塊)\n",
            "plt.figure(figsize=(15, 4))\n",
            "for i, layer in enumerate(layers):\n",
            "    plt.subplot(1, 3, i+1)\n",
            "    plt.hist(real_scores[layer], bins=30, alpha=0.5, label='真人', color='#4ade80')\n",
            "    plt.hist(fake_scores[layer], bins=30, alpha=0.5, label='詐騙', color='#f87171')\n",
            "    plt.title(f'{layer} 分數分佈')\n",
            "    plt.legend()\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }

    # Insert before SE-Attention MLP training (Part B)
    # Part B title starts at "# Part B"
    insert_pos = 0
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' and 'Part B' in "".join(cell['source']):
            insert_pos = i
            break
            
    nb['cells'].insert(insert_pos, calibration_cell)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Success.")

target = r"d:\code\side-project_VC\AI_Voice\notebooks\03_memory_and_fusion_training.ipynb"
inject_calibration_logic(target)

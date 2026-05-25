import json
import os

def fix_chinese_font_in_notebook(file_path):
    print(f"Fixing Chinese font in {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    font_fix_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === Matplotlib 中文顯示修復 ===\n",
            "!apt-get install -y fonts-wqy-microhei  # 安裝文泉驛微米黑字型\n",
            "import matplotlib.pyplot as plt\n",
            "plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 設定字型\n",
            "plt.rcParams['axes.unicode_minus'] = False  # 修復負號顯示問題\n",
            "print('✅ 中文字型配置完成：WenQuanYi Micro Hei')"
        ]
    }
    
    # Insert after Env Check (now at index 0)
    # Wait, Env Check was at index 0? Let's check.
    # In my last script, I used `nb['cells'].insert(0, env_check)`
    # So index 0 is Env Check. I'll put font fix at index 1.
    nb['cells'].insert(1, font_fix_cell)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Success.")

# Target notebooks
notebooks = [
    r"d:\code\side-project_VC\AI_Voice\notebooks\01_bert_fraud_training.ipynb",
    r"d:\code\side-project_VC\AI_Voice\notebooks\02_voiceprint_training.ipynb",
    r"d:\code\side-project_VC\AI_Voice\notebooks\03_memory_and_fusion_training.ipynb"
]

for nb_path in notebooks:
    fix_chinese_font_in_notebook(nb_path)

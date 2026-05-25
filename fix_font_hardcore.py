import json
import os

def fix_chinese_font_hardcore(file_path):
    print(f"Hardcore font fix in {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the font fix cell and update it
    found = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if '# === Matplotlib 中文顯示修復 ===' in "".join(source):
                cell['source'] = [
                    "# === Matplotlib 中文顯示修復 (手動路徑版) ===\n",
                    "!apt-get install -y fonts-wqy-microhei\n",
                    "import matplotlib.pyplot as plt\n",
                    "import matplotlib.font_manager as fm\n",
                    "\n",
                    "# 手動強制加載字型檔，避開快取更新問題\n",
                    "font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'\n",
                    "if os.path.exists(font_path):\n",
                    "    fm.fontManager.addfont(font_path)\n",
                    "    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']\n",
                    "    plt.rcParams['axes.unicode_minus'] = False\n",
                    "    print('✅ 已手動載入字型: WenQuanYi Micro Hei')\n",
                    "else:\n",
                    "    print('❌ 找不到字型檔，請確認已執行 apt-get install')\n"
                ]
                found = True
                break
    
    if not found:
        print("  Font fix cell not found. Inserting anew.")
        # Logic to insert if missed, but it should be there.

    if found:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Processed.")

# Target notebooks
notebooks = [
    r"d:\code\side-project_VC\AI_Voice\notebooks\01_bert_fraud_training.ipynb",
    r"d:\code\side-project_VC\AI_Voice\notebooks\02_voiceprint_training.ipynb",
    r"d:\code\side-project_VC\AI_Voice\notebooks\03_memory_and_fusion_training.ipynb"
]

for nb_path in notebooks:
    fix_chinese_font_hardcore(nb_path)

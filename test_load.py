import sys
import os
from pathlib import Path

# 加入 src 路徑
sys.path.append(os.getcwd())

try:
    from src.main import AIVoiceDetector
    print("--- 正在初始化 AIVoiceDetector ---")
    detector = AIVoiceDetector()
    print("--- 初始化成功 ---")
    
    # 檢查各組件載入狀態
    components = {
        "Voiceprint (Agent V)": detector.agent_v.is_loaded,
        "Semantic (Agent S)": detector.agent_s.is_loaded,
        "Memory (Agent M)": detector.agent_m.is_loaded,
        "Fusion Engine": detector.fusion_engine.is_loaded
    }
    
    for name, status in components.items():
        print(f"[{'OK' if status else 'FAIL'}] {name}")
        
    if all(components.values()):
        print("\n🎉 全系統模型載入驗證通過！")
        sys.exit(0)
    else:
        print("\n❌ 部份模型載入失敗，請檢查日誌")
        sys.exit(1)

except Exception as e:
    print(f"\n💥 發生嚴重錯誤: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

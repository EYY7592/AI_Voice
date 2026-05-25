from config.settings import settings
from src.step3_agents.semantic_agent import SemanticAgent
from src.models import TranscriptionResult, TranscriptionSegment

def test_semantic_agent():
    print("Loading SemanticAgent...")
    agent = SemanticAgent(model_path=settings.agent.semantic_model)
    agent.load_model()
    
    print("\n[Test Case 1] 詐騙話術")
    text1 = "您好，這裡是檢察院，您名下的銀行帳戶涉嫌洗錢犯罪，必須馬上將資金轉入我們的安全帳戶進行清查，否則立刻逮捕。"
    transcript1 = TranscriptionResult(text=text1, segments=[], language="zh", confidence=0.99)
    result1 = agent.analyze(transcript1)
    
    print(f"Prob: {result1.fraud_probability:.4f}, Confidence: {result1.confidence:.4f}")
    print(f"Explanation: {result1.explanation}")
    
    print("\n[Test Case 2] 一般話術")
    text2 = "請問今天天氣如何？我想去爬山。"
    transcript2 = TranscriptionResult(text=text2, segments=[], language="zh", confidence=0.99)
    result2 = agent.analyze(transcript2)
    
    print(f"Prob: {result2.fraud_probability:.4f}, Confidence: {result2.confidence:.4f}")
    print(f"Explanation: {result2.explanation}")

if __name__ == "__main__":
    test_semantic_agent()

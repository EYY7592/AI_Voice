from fastapi.testclient import TestClient

from src.gui import create_app


def test_homepage_exposes_three_content_inputs_without_old_agent_ui():
    html = TestClient(create_app()).get("/").text

    assert "貼上文字" in html
    assert "上傳截圖" in html
    assert "上傳語音" in html
    assert 'id="textInput"' in html
    assert 'id="imageInput"' in html
    assert 'id="audioInput"' in html
    assert 'id="correctionPanel"' in html
    assert 'id="resultPanel"' in html
    assert "Deepfake" not in html
    assert "SE-Attention" not in html
    assert "長期庫存" not in html

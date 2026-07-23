from fastapi.testclient import TestClient

from src.gui import create_app


def test_browser_favicon_request_does_not_return_404() -> None:
    response = TestClient(create_app()).get("/favicon.ico")

    assert response.status_code == 204
    assert response.content == b""

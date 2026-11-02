
import pytest

from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_page_not_found():
    response = client.get("/nonexistent")

    assert response.status_code == 404
    assert "text/html" in response.headers["content-type"]
    assert "Page Not Found" in response.text


def test_api_not_found():
    response = client.get("/api/nonexistent")

    assert response.status_code == 404
    assert "application/json" in response.headers["content-type"]


@pytest.mark.parametrize(
    "status_code, expected_heading",
    [
        (400, "Bad Request"),
        (403, "Access Denied"),
        (404, "Page Not Found"),
        (413, "File Too Large"),
        (429, "Too Many Requests"),
        (500, "Internal Server Error"),
    ],
)
def test_custom_error_pages(status_code, expected_heading):
    path = f"/test-error-{status_code}"

    async def raise_error():
        raise HTTPException(status_code=status_code, detail="Test error")

    app.add_api_route(path, raise_error, methods=["GET"])

    try:
        response = client.get(path)

        assert response.status_code == status_code
        assert "text/html" in response.headers["content-type"]
        assert expected_heading in response.text
        assert "Test error" not in response.text

    finally:
        app.router.routes[:] = [
            route for route in app.router.routes
            if getattr(route, "path", None) != path
        ]


@pytest.mark.parametrize("status_code", [400, 403, 404, 413, 429, 500])
def test_api_error_responses(status_code):
    path = f"/api/test-error-{status_code}"

    async def raise_error():
        raise HTTPException(status_code=status_code, detail="Test error")

    app.add_api_route(path, raise_error, methods=["GET"])

    try:
        response = client.get(path)

        assert response.status_code == status_code
        assert "application/json" in response.headers["content-type"]
        assert isinstance(response.json(), dict)

    finally:
        app.router.routes[:] = [
            route for route in app.router.routes
            if getattr(route, "path", None) != path
        ]

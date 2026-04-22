import pytest
import app as flask_app

@pytest.fixture
def client():
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c

def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200

def test_colorize_rejects_non_image(client):
    data = {"file": (b"not an image", "test.txt")}
    resp = client.post("/colorize", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400

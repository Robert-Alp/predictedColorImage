import io
import pytest
import numpy as np
import cv2
from PIL import Image
import app as flask_app
from restore import detect_mask, inpaint

@pytest.fixture
def client():
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c

def _make_image_bytes(color=128, size=(64, 64)):
    img = Image.new("L", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200

def test_colorize_rejects_non_image(client):
    data = {"file": (b"not an image", "test.txt")}
    resp = client.post("/colorize", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400

def test_detect_mask_returns_binary_mask():
    img = np.ones((100, 100), dtype=np.uint8) * 128
    img[20:30, 20:80] = 255
    mask = detect_mask(img)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask[25, 50] == 255

def test_detect_mask_empty_on_uniform_gray():
    img = np.ones((100, 100), dtype=np.uint8) * 128
    mask = detect_mask(img)
    white_ratio = np.sum(mask == 255) / mask.size
    assert white_ratio < 0.1

def test_inpaint_fills_masked_region():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    img[40:60, 40:60] = 255
    result = inpaint(img, mask)
    assert result.shape == img.shape
    center_val = result[50, 50].mean()
    assert center_val < 200

def test_process_colorize_mode(client):
    data = {"file": (_make_image_bytes(), "test.png", "image/png"), "mode": "colorize"}
    resp = client.post("/process", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "original" in body
    assert "colorized" in body

def test_process_restore_mode(client):
    data = {"file": (_make_image_bytes(), "test.png", "image/png"), "mode": "restore"}
    resp = client.post("/process", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "original" in body
    assert "mask" in body
    assert "repaired" in body
    assert "colorized" in body

def test_process_rejects_invalid_file(client):
    data = {"file": (b"not an image", "test.txt"), "mode": "colorize"}
    resp = client.post("/process", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400

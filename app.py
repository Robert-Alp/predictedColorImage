import io
import base64
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms
from restore import detect_mask, inpaint

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 Mo

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "generator_final.pth")
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Architecture U-Net (identique à colorize.py) ──────────────────────────────

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, bn=True, dropout=False, act=True):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not bn))
        else:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if dropout:
            layers.append(nn.Dropout(0.5))
        if act:
            layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = UNetBlock(1,   64,  down=True, bn=False)
        self.e2 = UNetBlock(64,  128, down=True)
        self.e3 = UNetBlock(128, 256, down=True)
        self.e4 = UNetBlock(256, 512, down=True)
        self.e5 = UNetBlock(512, 512, down=True)
        self.e6 = UNetBlock(512, 512, down=True)
        self.e7 = UNetBlock(512, 512, down=True)
        self.e8 = UNetBlock(512, 512, down=True, bn=False)
        self.d1 = UNetBlock(512,  512, down=False, dropout=True)
        self.d2 = UNetBlock(1024, 512, down=False, dropout=True)
        self.d3 = UNetBlock(1024, 512, down=False, dropout=True)
        self.d4 = UNetBlock(1024, 512, down=False)
        self.d5 = UNetBlock(1024, 256, down=False)
        self.d6 = UNetBlock(512,  128, down=False)
        self.d7 = UNetBlock(256,  64,  down=False)
        self.out = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x);  e2 = self.e2(e1);  e3 = self.e3(e2);  e4 = self.e4(e3)
        e5 = self.e5(e4); e6 = self.e6(e5);  e7 = self.e7(e6);  e8 = self.e8(e7)
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        return self.out(torch.cat([d7, e1], 1))


# ── Chargement du modèle (une seule fois) ─────────────────────────────────────

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

G = Generator().to(DEVICE)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
G.eval()
print(f"[OK] Générateur chargé sur {DEVICE}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def tensor_to_base64(tensor: torch.Tensor) -> str:
    arr = (tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def numpy_to_base64(arr_bgr: np.ndarray) -> str:
    arr_rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(arr_rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def mask_to_base64(mask: np.ndarray) -> str:
    img = Image.fromarray(mask)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/colorize")
def colorize():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400

    file = request.files["file"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Format invalide. Utilise JPG ou PNG."}), 400

    try:
        img = Image.open(file.stream).convert("L")
    except Exception:
        return jsonify({"error": "Impossible de lire l'image."}), 400

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    gray_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fake = G(gray_tensor).squeeze(0).cpu() * 0.5 + 0.5

    gray_rgb = transforms.ToTensor()(img.resize((IMG_SIZE, IMG_SIZE))).repeat(3, 1, 1)

    return jsonify({
        "original": tensor_to_base64(gray_rgb),
        "colorized": tensor_to_base64(fake),
    })


@app.post("/process")
def process():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400

    file = request.files["file"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Format invalide. Utilise JPG ou PNG."}), 400

    mode = request.form.get("mode", "colorize")

    try:
        pil_img = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Impossible de lire l'image."}), 400

    if mode == "colorize":
        gray = pil_img.convert("L")
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        gray_tensor = transform(gray).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            fake = G(gray_tensor).squeeze(0).cpu() * 0.5 + 0.5
        gray_rgb = transforms.ToTensor()(gray.resize((IMG_SIZE, IMG_SIZE))).repeat(3, 1, 1)
        return jsonify({
            "original": tensor_to_base64(gray_rgb),
            "colorized": tensor_to_base64(fake),
        })

    # Mode restauration + colorisation
    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mask = detect_mask(img_gray)
    white_ratio = np.sum(mask == 255) / mask.size

    if white_ratio > 0.5:
        return jsonify({"error": "Image trop dégradée pour la restauration automatique."}), 400

    warning = None
    if white_ratio < 0.001:
        warning = "Aucune zone abîmée détectée — colorisation seule appliquée."
        repaired_bgr = img_bgr
    else:
        repaired_bgr = inpaint(img_bgr, mask)

    repaired_gray = cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2GRAY)
    repaired_pil = Image.fromarray(repaired_gray).resize((IMG_SIZE, IMG_SIZE))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    gray_tensor = transform(repaired_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        fake = G(gray_tensor).squeeze(0).cpu() * 0.5 + 0.5

    original_gray = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    original_rgb = np.stack([original_gray] * 3, axis=2)
    mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    repaired_resized = cv2.resize(repaired_bgr, (IMG_SIZE, IMG_SIZE))
    repaired_gray_resized = cv2.cvtColor(repaired_resized, cv2.COLOR_BGR2GRAY)

    response = {
        "original": numpy_to_base64(cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)),
        "mask": mask_to_base64(mask_resized),
        "repaired": numpy_to_base64(cv2.cvtColor(repaired_gray_resized, cv2.COLOR_GRAY2BGR)),
        "colorized": tensor_to_base64(fake),
    }
    if warning:
        response["warning"] = warning
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

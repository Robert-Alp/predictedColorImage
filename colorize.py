"""
Colorisation d'images N&B avec un GAN (pix2pix)
================================================
Inférence : coloriser une image N&B avec le générateur entraîné

Usage:
    python colorize.py --model checkpoints/generator_final.pth --input photo_nb.jpg --output resultat.png
"""

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image


# ─────────────────────────────────────────────
# MODÈLE (même architecture que train.py)
# ─────────────────────────────────────────────

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
        self.e1 = UNetBlock(1,    64,  down=True, bn=False)
        self.e2 = UNetBlock(64,   128, down=True)
        self.e3 = UNetBlock(128,  256, down=True)
        self.e4 = UNetBlock(256,  512, down=True)
        self.e5 = UNetBlock(512,  512, down=True)
        self.e6 = UNetBlock(512,  512, down=True)
        self.e7 = UNetBlock(512,  512, down=True)
        self.e8 = UNetBlock(512,  512, down=True, bn=False)

        self.d1 = UNetBlock(512,  512, down=False, dropout=True)
        self.d2 = UNetBlock(1024, 512, down=False, dropout=True)
        self.d3 = UNetBlock(1024, 512, down=False, dropout=True)
        self.d4 = UNetBlock(1024, 512, down=False)
        self.d5 = UNetBlock(1024, 256, down=False)
        self.d6 = UNetBlock(512,  128, down=False)
        self.d7 = UNetBlock(256,   64, down=False)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

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


# ─────────────────────────────────────────────
# INFÉRENCE
# ─────────────────────────────────────────────

def colorize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Charger le générateur
    G = Generator().to(device)
    G.load_state_dict(torch.load(args.model, map_location=device))
    G.eval()
    print(f"Modèle chargé : {args.model}")

    # Préparer l'image d'entrée
    img = Image.open(args.input).convert("L")  # forcer N&B
    original_size = img.size

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    gray_tensor = transform(img).unsqueeze(0).to(device)  # (1, 1, H, W)

    # Générer la version colorisée
    with torch.no_grad():
        fake_color = G(gray_tensor)               # (1, 3, H, W) dans [-1, 1]

    # Dénormaliser → [0, 1]
    fake_color = fake_color * 0.5 + 0.5

    # Redimensionner à la taille originale
    resize_back = transforms.Resize(
        (original_size[1], original_size[0]),
        interpolation=transforms.InterpolationMode.BICUBIC
    )
    fake_color = resize_back(fake_color.squeeze(0))

    # Sauvegarder
    save_image(fake_color, args.output)
    print(f"Image colorisée sauvegardée : {args.output}")

    # Optionnel : côte à côte (N&B | couleur)
    if args.side_by_side:
        gray_rgb = transforms.ToTensor()(
            img.resize((original_size[0], original_size[1]))
        ).unsqueeze(0).repeat(1, 3, 1, 1)  # copier en 3 canaux

        side = torch.cat([gray_rgb, fake_color.unsqueeze(0)], dim=0)
        out_side = args.output.replace(".", "_compare.")
        save_image(side, out_side, nrow=2)
        print(f"Comparaison sauvegardée : {out_side}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN Colorisation - Inférence")
    parser.add_argument("--model",        required=True,          help="Chemin vers generator_final.pth")
    parser.add_argument("--input",        required=True,          help="Image N&B à coloriser")
    parser.add_argument("--output",       default="colorized.png",help="Image résultat")
    parser.add_argument("--img_size",     type=int, default=256,  help="Taille de traitement")
    parser.add_argument("--side_by_side", action="store_true",    help="Générer aussi une comparaison N&B | couleur")
    args = parser.parse_args()
    colorize(args)

"""
Colorisation d'images N&B avec un GAN (pix2pix)
================================================
Entraînement du générateur (U-Net) et du discriminateur (PatchGAN)

Usage:
    pip install torch torchvision pillow
    python train.py --data_dir ./dataset --epochs 200
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class ColorizationDataset(Dataset):
    """
    Attend un dossier avec des images couleur.
    Convertit automatiquement en N&B pour l'entrée (X)
    et garde la version couleur comme cible (Y).
    """
    def __init__(self, root_dir, img_size=256):
        self.paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.img_size = img_size
        self.to_tensor = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # [-1, 1]
        ])
        self.color_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        color = transforms.Resize((self.img_size, self.img_size))(img)
        color = transforms.ToTensor()(color)
        color = self.color_norm(color)          # (3, H, W) dans [-1, 1]

        gray = transforms.Grayscale()(img)
        gray = self.to_tensor(gray)              # (1, H, W) dans [-1, 1]

        return gray, color


# ─────────────────────────────────────────────
# GÉNÉRATEUR : U-Net
# ─────────────────────────────────────────────

class UNetBlock(nn.Module):
    """Bloc encodeur ou décodeur du U-Net."""
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
    """
    U-Net : encode l'image N&B, décode en image couleur (RGB).
    Les skip-connections conservent les détails spatiaux.
    """
    def __init__(self):
        super().__init__()
        # Encodeur
        self.e1 = UNetBlock(1,    64,  down=True, bn=False)   # 128
        self.e2 = UNetBlock(64,   128, down=True)              # 64
        self.e3 = UNetBlock(128,  256, down=True)              # 32
        self.e4 = UNetBlock(256,  512, down=True)              # 16
        self.e5 = UNetBlock(512,  512, down=True)              # 8
        self.e6 = UNetBlock(512,  512, down=True)              # 4
        self.e7 = UNetBlock(512,  512, down=True)              # 2
        self.e8 = UNetBlock(512,  512, down=True, bn=False)    # 1 (bottleneck)

        # Décodeur (skip-connections → canaux doublés)
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
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], dim=1))
        d3 = self.d3(torch.cat([d2, e6], dim=1))
        d4 = self.d4(torch.cat([d3, e5], dim=1))
        d5 = self.d5(torch.cat([d4, e4], dim=1))
        d6 = self.d6(torch.cat([d5, e3], dim=1))
        d7 = self.d7(torch.cat([d6, e2], dim=1))
        return self.out(torch.cat([d7, e1], dim=1))


# ─────────────────────────────────────────────
# DISCRIMINATEUR : PatchGAN
# ─────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    PatchGAN : juge chaque patch 70x70 de l'image
    plutôt que l'image entière (meilleure texture).
    Prend en entrée [image_NB, image_couleur] concaténées.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),                          # 1(NB) + 3(RGB)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1),                         # carte de décision
        )

    def forward(self, gray, color):
        x = torch.cat([gray, color], dim=1)
        return self.model(x)


# ─────────────────────────────────────────────
# INITIALISATION DES POIDS
# ─────────────────────────────────────────────

def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


# ─────────────────────────────────────────────
# BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Données
    dataset = ColorizationDataset(args.data_dir, img_size=args.img_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Modèles
    G = Generator().to(device)
    D = Discriminator().to(device)
    init_weights(G)
    init_weights(D)

    # Optimiseurs
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Pertes
    bce  = nn.BCEWithLogitsLoss()
    l1   = nn.L1Loss()
    L1_LAMBDA = 100  # poids de la perte L1 (netteté)

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        g_total = d_total = 0.0

        for gray, real_color in loader:
            gray, real_color = gray.to(device), real_color.to(device)
            B = gray.size(0)

            # ── Entraîner le Discriminateur ──
            fake_color = G(gray).detach()

            pred_real = D(gray, real_color)
            pred_fake = D(gray, fake_color)

            loss_D = 0.5 * (
                bce(pred_real, torch.ones_like(pred_real))
              + bce(pred_fake, torch.zeros_like(pred_fake))
            )

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ── Entraîner le Générateur ──
            fake_color = G(gray)
            pred_fake  = D(gray, fake_color)

            loss_G = (
                bce(pred_fake, torch.ones_like(pred_fake))  # tromper D
              + L1_LAMBDA * l1(fake_color, real_color)       # fidélité couleur
            )

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            g_total += loss_G.item()
            d_total += loss_D.item()

        n = len(loader)
        print(f"Epoch {epoch:3d}/{args.epochs}  |  Loss G: {g_total/n:.4f}  |  Loss D: {d_total/n:.4f}")

        # Sauvegarder des exemples visuels
        if epoch % args.save_every == 0:
            G.eval()
            with torch.no_grad():
                sample_gray, sample_real = next(iter(loader))
                sample_gray = sample_gray[:4].to(device)
                sample_fake = G(sample_gray)
                # Dénormaliser vers [0, 1]
                grid = torch.cat([
                    sample_gray.repeat(1, 3, 1, 1),  # N&B en faux RGB
                    sample_fake,
                ], dim=0) * 0.5 + 0.5
                save_image(grid, f"{args.output_dir}/epoch_{epoch:03d}.png", nrow=4)

            # Sauvegarder les checkpoints
            torch.save(G.state_dict(), f"{args.ckpt_dir}/generator_epoch{epoch}.pth")
            torch.save(D.state_dict(), f"{args.ckpt_dir}/discriminator_epoch{epoch}.pth")

    print("Entraînement terminé !")
    torch.save(G.state_dict(), f"{args.ckpt_dir}/generator_final.pth")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN Colorisation - Entraînement")
    parser.add_argument("--data_dir",   default="./dataset",  help="Dossier avec images couleur")
    parser.add_argument("--output_dir", default="./samples",  help="Exemples visuels générés")
    parser.add_argument("--ckpt_dir",   default="./checkpoints", help="Sauvegardes des modèles")
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size",   type=int, default=256)
    parser.add_argument("--save_every", type=int, default=10, help="Sauvegarder tous les N epochs")
    args = parser.parse_args()
    train(args)

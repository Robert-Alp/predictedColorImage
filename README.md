# GAN Colorisation — pix2pix

Colorisation d'images en noir et blanc avec un GAN conditionnel (pix2pix).

## Installation

```bash
pip install torch torchvision pillow
```

## Structure du projet

```
.
├── train.py          # Entraînement du GAN
├── colorize.py       # Inférence sur une image N&B
├── dataset/          # Tes images couleur (JPG/PNG)
├── checkpoints/      # Poids sauvegardés automatiquement
└── samples/          # Exemples visuels générés pendant l'entraînement
```

## Préparer le dataset

Place tes **images couleur** (JPG ou PNG) dans `./dataset/`.
Le script les convertit automatiquement en N&B pour créer les paires (entrée/cible).

- Minimum recommandé : ~500 images
- Idéal : 5 000+ images sur un domaine cohérent (paysages, portraits, rue...)
- Plus le domaine est spécialisé, moins tu as besoin de données

## Entraînement

```bash
python train.py --data_dir ./dataset --epochs 200
```

Options disponibles :

| Argument       | Défaut       | Description                          |
|----------------|--------------|--------------------------------------|
| `--data_dir`   | `./dataset`  | Dossier avec images couleur          |
| `--epochs`     | 200          | Nombre d'époques                     |
| `--batch_size` | 4            | Taille de batch (réduire si peu VRAM)|
| `--img_size`   | 256          | Résolution d'entraînement            |
| `--save_every` | 10           | Sauvegarder tous les N epochs        |

Toutes les 10 époques, un fichier `samples/epoch_XXX.png` est généré pour
visualiser la progression. Les poids sont dans `checkpoints/`.

## Inférence

```bash
python colorize.py \
    --model checkpoints/generator_final.pth \
    --input ma_photo_nb.jpg \
    --output resultat.png \
    --side_by_side
```

L'option `--side_by_side` génère aussi une image de comparaison côte à côte.

## Architecture

| Composant        | Architecture        | Rôle                                      |
|------------------|---------------------|-------------------------------------------|
| **Générateur**   | U-Net (8 niveaux)   | Convertir N&B → couleur                   |
| **Discriminateur**| PatchGAN 70×70     | Distinguer vraie/fausse image couleur     |

### Fonction de perte

```
Loss_G = BCE(D(G(x)), 1) + λ × L1(G(x), y_réel)
Loss_D = 0.5 × [BCE(D(x,y), 1) + BCE(D(x,G(x)), 0)]
```

Le terme L1 (λ=100) force la fidélité des couleurs en plus du signal adversarial.

## Conseils

- **GPU** : très recommandé. Sur CPU, un epoch peut prendre 10-20min.
- **Résultats lents à venir** : les premières 20-30 époques produisent des images floues, c'est normal.
- **Overfitting** : si les `samples` ressemblent parfaitement aux images d'entraînement mais échouent sur de nouvelles images, augmente le dataset ou réduis les epochs.
- **Qualité** : pour de meilleurs résultats, tu peux passer `--img_size 512` (nécessite plus de VRAM).

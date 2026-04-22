import cv2
import numpy as np


def detect_mask(gray: np.ndarray) -> np.ndarray:
    """Détecte les zones très claires (déchirures) par seuil global > 240."""
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def inpaint(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Répare les zones masquées via l'algorithme TELEA."""
    return cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

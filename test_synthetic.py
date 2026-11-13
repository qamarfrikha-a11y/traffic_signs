"""Test rapide sans vidéo : génère des panneaux synthétiques et vérifie le détecteur.
Usage : python test_synthetic.py   (ajoute --save pour sauvegarder les images)"""
import sys
import cv2
import numpy as np
from classical.detector import ClassicalSignDetector, pytesseract


def speed_sign(value="50", size=240):
    img = np.full((size * 2, size * 2, 3), (110, 110, 110), np.uint8)   # fond gris (route)
    c = (size, size)
    cv2.circle(img, c, size // 2, (0, 0, 220), -1)                      # rouge (BGR)
    cv2.circle(img, c, int(size * 0.38), (255, 255, 255), -1)           # disque blanc
    (tw, th), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_DUPLEX, 3, 8)
    cv2.putText(img, value, (c[0] - tw // 2, c[1] + th // 2),
                cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 8)
    return img


def color_sign(bgr, size=240):
    img = np.full((size * 2, size * 2, 3), (110, 110, 110), np.uint8)
    cv2.circle(img, (size, size), size // 2, bgr, -1)
    return img


def main(save=False):
    det = ClassicalSignDetector()
    cases = [
        ("limite 50", speed_sign("50"), "red"),
        ("limite 90", speed_sign("90"), "red"),
        ("obligation (bleu)", color_sign((200, 90, 0)), "blue"),
        ("danger (jaune)", color_sign((0, 210, 255)), "yellow"),
    ]
    ok_all = True
    for name, img, expected_color in cases:
        dets = det.detect(img)
        best = max(dets, key=lambda d: d.confidence, default=None)
        color_ok = best is not None and best.color == expected_color
        label = best.label if best else "RIEN"
        print(f"[{'OK' if color_ok else 'KO'}] {name:20s} -> {label}")
        ok_all &= color_ok
        if save:
            cv2.imwrite(f"test_{name.split()[0]}_{expected_color}.png", img)
    if pytesseract is None:
        print("(!) pytesseract absent : OCR désactivé, seules les couleurs sont testées")
    print("\nRESULTAT :", "tout est OK" if ok_all else "des échecs -> calibrer config.py")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main("--save" in sys.argv)
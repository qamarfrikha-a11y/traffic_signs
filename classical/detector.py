import re
import cv2
import numpy as np
import pytesseract
from common.types import Detection
from common import config as C

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

VALID_LIMITS = {20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130}


class ClassicalSignDetector:
    def __init__(self, min_r=C.MIN_RADIUS, max_r=C.MAX_RADIUS, debug=False, ocr_per_frame=1):
        self.ocr_per_frame = ocr_per_frame
        self.min_r, self.max_r, self.debug = min_r, max_r, debug
        self.k3 = np.ones((3, 3), np.uint8)
        self.k5 = np.ones((5, 5), np.uint8)
        self.rejected = []
        self.ocr_budget = self.ocr_per_frame
        self.frame_i = 0
        self.ocr_cache = []          # (x, y, r, val, conf, frame_i)

    # ---------- 1. couleur (haut de l'image seulement, sans CLAHE) ----------
    def _masks(self, frame):
        cut = int(frame.shape[0] * (1 - C.IGNORE_BOTTOM))
        top = frame[:cut]
        hsv = cv2.cvtColor(cv2.GaussianBlur(top, (3, 3), 0), cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255)) | \
              cv2.inRange(hsv, (165, 80, 50), (180, 255, 255))
        blue = cv2.inRange(hsv, (100, 100, 50), (130, 255, 255))
        yellow = cv2.inRange(hsv, (18, 110, 110), (34, 255, 255))
        out = {}
        for name, m in (("red", red), ("blue", blue), ("yellow", yellow)):
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k3)
            out[name] = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.k5)
        return out

    # ---------- 2. forme ----------
    @staticmethod
    def _shape(hull, r):
        ratio = cv2.contourArea(hull) / (np.pi * r * r + 1e-6)
        if ratio > 0.74:
            return "circle"
        if ratio > 0.55:
            return "rect"
        if ratio > 0.33:
            return "triangle"
        return None
    @staticmethod
    def _digits_roi(frame, x, y, r):
        k = int(r * 0.7)
        x0, y0 = max(x - k, 0), max(y - k, 0)
        roi = frame[y0:y + k, x0:x + k].copy()
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            return None
        mask = np.zeros(roi.shape[:2], np.uint8)
        cv2.circle(mask, (x - x0, y - y0), int(r * 0.68), 255, -1)
        roi[mask == 0] = 255          # anneau et coins -> blanc
        return roi
    # ---------- 3. OCR (avec cache et budget) ----------
    @staticmethod
    def _inner(frame, x, y, r):
        k = int(r * 0.55)
        roi = frame[max(y - k, 0):y + k, max(x - k, 0):x + k]
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            return None
        return roi
    @staticmethod
    def _read_value(roi):
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # garder uniquement ce qui ressemble à des chiffres (pas les restes de l'anneau)
        H, W = bw.shape
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
        keep = np.zeros_like(bw)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            touche_bord = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
            if 0.25 * H < h < 0.85 * H and not touche_bord:
                keep[lab == i] = 255
        ys, xs = np.nonzero(keep)
        if len(xs) < 80:
            return None, 0.0

        crop = keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        crop = cv2.copyMakeBorder(255 - crop, 30, 30, 30, 30,
                                  cv2.BORDER_CONSTANT, value=255)

        best, bconf = None, 0.0
        cv2.imwrite("debug_ocr.png", crop)
        for psm in (7, 8, 13):                      # on s'arrête dès qu'un nombre valide est lu
            d = pytesseract.image_to_data(
                crop, output_type=pytesseract.Output.DICT,
                config=f"--psm {psm} -c tessedit_char_whitelist=0123456789")
            for t, cf in zip(d["text"], d["conf"]):
                t = re.sub(r"\D", "", t)
                if t and int(t) in VALID_LIMITS and float(cf) / 100 > bconf:
                    best, bconf = int(t), float(cf) / 100
            if best:
                break
        return best, bconf

    def _ocr(self, frame, x, y, r):
        # même panneau déjà lu récemment : on réutilise le résultat
        for cx, cy, cr, v, o, fi in self.ocr_cache:
            if abs(cx - x) < cr and abs(cy - y) < cr and self.frame_i - fi < 20:
                return v, o
        if r < 14 or self.ocr_budget <= 0:
            return None, 0.0
        roi = self._digits_roi(frame, x, y, r)
        if roi is None:
            return None, 0.0
        self.ocr_budget -= 1
        v, o = self._read_value(roi)
        if v:
            self.ocr_cache.append((x, y, r, v, o, self.frame_i))
        
        self.ocr_cache = [e for e in self.ocr_cache if self.frame_i - e[5] < 20]
        return v, o

    # ---------- 4. Hough (demi-résolution) ----------
    def _hough_red(self, red_mask):
        small = cv2.resize(red_mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 1.5)
        circles = cv2.HoughCircles(
            small, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
            param1=100, param2=14,
            minRadius=max(self.min_r // 2, 3), maxRadius=self.max_r // 2)
        out = []
        if circles is None:
            return out
        H, W = red_mask.shape
        for cx, cy, r in np.round(circles[0] * 2).astype(int):
            ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            px = np.clip((cx + 0.9 * r * np.cos(ang)).astype(int), 0, W - 1)
            py = np.clip((cy + 0.9 * r * np.sin(ang)).astype(int), 0, H - 1)
            if np.mean(red_mask[py, px] > 0) < 0.45:
                continue
            k = max(int(r * 0.4), 2)
            core = red_mask[max(cy - k, 0):cy + k, max(cx - k, 0):cx + k]
            if core.size and np.mean(core > 0) > 0.6:
                continue
            out.append((int(cx), int(cy), int(r)))
        return out[:3]

    # ---------- 5. détection ----------
    def detect(self, frame):
        self.rejected = []
        self.ocr_budget = self.ocr_per_frame
        self.frame_i += 1
        masks = self._masks(frame)
        dets = []

        for color, mask in masks.items():
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < 30:
                    continue
                (cx, cy), r = cv2.minEnclosingCircle(c)
                x, y, r = int(cx), int(cy), int(r)
                if not (self.min_r <= r <= self.max_r):
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                ratio = bw / bh
                lo, hi = (0.5, 1.8) if color == "blue" else (0.6, 1.5)
                if not (lo <= ratio <= hi):
                    self.rejected.append((x, y, r, "ratio"))
                    continue
                hull = cv2.convexHull(c)
                shape = self._shape(hull, r)
                if shape is None:
                    self.rejected.append((x, y, r, "forme"))
                    continue
                # remplissage calculé dans la petite boîte (rapide)
                hx, hy, hw, hh = cv2.boundingRect(hull)
                hm = np.zeros((hh, hw), np.uint8)
                cv2.drawContours(hm, [hull - np.array([hx, hy])], -1, 255, -1)
                fill = cv2.countNonZero(cv2.bitwise_and(hm, mask[hy:hy + hh, hx:hx + hw])) \
                    / max(cv2.countNonZero(hm), 1)

                label, conf = None, 0.0
                if color == "red" and shape == "circle":
                    if fill > 0.88:
                        self.rejected.append((x, y, r, "plein"))
                        continue
                    roi = self._inner(frame, x, y, r)
                    if roi is None:
                        continue
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    white = np.mean((hsv[..., 1] < 90) & (hsv[..., 2] > 110))
                    if white < 0.15:
                        self.rejected.append((x, y, r, "centre"))
                        continue
                    val, oc = self._ocr(frame, x, y, r)
                    label, conf = ("limite %d" % val, 0.5 + 0.5 * oc) if val else ("interdiction", 0.45)
                elif color == "red" and shape == "triangle" and fill < 0.88:
                    label, conf = "danger", 0.6
                elif color == "blue" and shape == "circle" and fill > 0.5:
                    label, conf = "obligation", 0.6
                elif color == "blue" and shape == "rect" and fill > 0.5:
                    label, conf = "indication", 0.55
                elif color == "yellow" and shape in ("triangle", "rect") and fill > 0.5:
                    label, conf = "danger", 0.55
                if label:
                    dets.append(Detection(x, y, r, color, label, conf, shape))
                else:
                    self.rejected.append((x, y, r, "regle"))

        # Hough : seulement 1 appel sur 3 (le plus coûteux)
        if self.frame_i % 3 == 0:
            for cx, cy, r in self._hough_red(masks["red"]):
                if any(abs(d.x - cx) < r and abs(d.y - cy) < r for d in dets):
                    continue
                roi = self._inner(frame, cx, cy, r)
                if roi is None:
                    continue
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                if np.mean((hsv[..., 1] < 90) & (hsv[..., 2] > 110)) < 0.15:
                    continue
                val, oc = self._ocr(frame, cx, cy, r)
                label, conf = ("limite %d" % val, 0.5 + 0.5 * oc) if val else ("interdiction", 0.45)
                dets.append(Detection(cx, cy, r, "red", label, conf, "circle"))

        if self.debug:
            cv2.imshow("masque rouge", masks["red"])
            cv2.imshow("masque bleu", masks["blue"])
        return dets
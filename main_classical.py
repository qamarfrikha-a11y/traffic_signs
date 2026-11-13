"""Démo : pipeline classique + suivi + tableau de bord.
Usage : python main_classical.py [video.mp4 | 0]
Touches : w/s = +/- 10 km/h (vitesse simulée), q = quitter."""
import sys
import time
import cv2
from classical.detector import ClassicalSignDetector
from common.tracking import TemporalFilter, SpeedMonitor
from common import config as C

try:
    import winsound
except ImportError:
    winsound = None

FONT = cv2.FONT_HERSHEY_SIMPLEX
BGR = {"red": (0, 0, 255), "blue": (255, 120, 0), "yellow": (0, 220, 255)}
MIN_CONF = 0.5          # on n'affiche que les détections assez fiables
_last_beep = 0.0


def draw_limit_icon(img, cx, cy, r, limit):
    cv2.circle(img, (cx, cy), r, (0, 0, 255), -1)
    cv2.circle(img, (cx, cy), int(r * 0.72), (255, 255, 255), -1)
    text = str(limit) if limit else "--"
    scale = r / 38
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2), FONT, scale, (0, 0, 0), 2)


def draw_dashboard(img, monitor, confirmed):
    global _last_beep
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 90), (w, h), (30, 30, 30), -1)
    draw_limit_icon(img, 50, h - 45, 34, monitor.speed_limit)

    lim = f"{monitor.speed_limit} km/h" if monitor.speed_limit else "--"
    color = (0, 0, 255) if monitor.alert else (255, 255, 255)
    cv2.putText(img, f"Limite : {lim}", (100, h - 55), FONT, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Vitesse : {monitor.vehicle_speed} km/h", (100, h - 20), FONT, 0.7, color, 2)

    if confirmed:
        cv2.putText(img, f"Panneau : {confirmed}", (10, 55), FONT, 0.6, (0, 255, 0), 2)

    if monitor.alert:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        msg = "ALERTE : EXCES DE VITESSE"
        scale = max(0.5, min(1.0, w / 700))
        (tw, _), _ = cv2.getTextSize(msg, FONT, scale, 3)
        cv2.putText(img, msg, ((w - tw) // 2, 90), FONT, scale, (0, 0, 255), 3)
        if winsound and time.time() - _last_beep > 1.0:
            winsound.Beep(1000, 150)
            _last_beep = time.time()


def main(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    det, filt, mon = ClassicalSignDetector(), TemporalFilter(window=10, min_votes=3), SpeedMonitor()
    tick, fps = cv2.getTickCount(), 0.0
    n, dets = 0, []  
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cap.grab()     
        h0, w0 = frame.shape[:2]
        scale = min(C.FRAME_WIDTH / w0, C.MAX_HEIGHT / h0)   # tient dans l'écran
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        dets = [d for d in det.detect(frame) if d.confidence >= MIN_CONF]
        best = max(dets, key=lambda d: d.confidence, default=None)
        confirmed = filt.update(best.label if best else None)
        if confirmed and confirmed.startswith("limite "):
            mon.set_limit(int(confirmed.split()[1]))

        for d in dets:
            cv2.circle(frame, (d.x, d.y), d.r, BGR[d.color], 2)
            cv2.putText(frame, f"{d.label} {d.confidence:.2f}", (d.x - d.r, max(d.y - d.r - 6, 12)),
                        FONT, 0.5, BGR[d.color], 2)

        now = cv2.getTickCount()
        fps = 0.9 * fps + 0.1 * cv2.getTickFrequency() / (now - tick)
        tick = now
        cv2.putText(frame, f"{fps:.1f} FPS", (10, 25), FONT, 0.6, (0, 255, 0), 2)

        draw_dashboard(frame, mon, confirmed)
        cv2.imshow("Classique - panneaux", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("w"):
            mon.change_speed(+10)
        elif k == ord("s"):
            mon.change_speed(-10)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 0)
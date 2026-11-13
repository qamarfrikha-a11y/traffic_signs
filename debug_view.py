import sys
import cv2
from classical.detector import ClassicalSignDetector
from common import config as C

src = sys.argv[1]
img = cv2.imread(src)
cap = None if img is not None else cv2.VideoCapture(src)
det = ClassicalSignDetector()

while True:
    if cap:
        ok, frame = cap.read()
        if not ok:
            break
    else:
        frame = img.copy()
    s = min(C.FRAME_WIDTH / frame.shape[1], C.MAX_HEIGHT / frame.shape[0])
    frame = cv2.resize(frame, None, fx=s, fy=s)

    dets = det.detect(frame)
    masks = det._masks(frame)
    for x, y, r, why in det.rejected:                      # jaune = rejeté + raison
        cv2.circle(frame, (x, y), r, (0, 255, 255), 1)
        cv2.putText(frame, why, (x - r, y - r - 3), 0, 0.4, (0, 255, 255), 1)
    for d in dets:                                         # vert = accepté
        cv2.circle(frame, (d.x, d.y), d.r, (0, 255, 0), 2)
        cv2.putText(frame, d.label, (d.x - d.r, d.y - d.r - 3), 0, 0.5, (0, 255, 0), 2)

    cv2.imshow("masque rouge", masks["red"])
    cv2.imshow("masque bleu", masks["blue"])
    cv2.imshow("diagnostic", frame)
    k = cv2.waitKey(0 if not cap else 30) & 0xFF
    if k == ord("q"):
        break
    if k == ord(" ") and cap:
        cv2.waitKey(0)                                     # pause
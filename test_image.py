import sys
import cv2
from classical.detector import ClassicalSignDetector
from common import config as C

C.IGNORE_BOTTOM = 0          # une image de test n'a pas de capot

img = cv2.imread(sys.argv[1])
if img is None:
    sys.exit("Image introuvable : " + sys.argv[1])

# même redimensionnement que la vidéo
s = min(C.FRAME_WIDTH / img.shape[1], C.MAX_HEIGHT / img.shape[0])
img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

det = ClassicalSignDetector(debug=True, ocr_per_frame=50)   # un seul détecteur
dets = det.detect(img)                                       # un seul appel

if not dets:
    print("Aucun panneau détecté")
for d in dets:
    print(f"{d.label:15s} forme={d.shape:9s} couleur={d.color:7s} conf={d.confidence:.2f}")
    cv2.circle(img, (d.x, d.y), d.r, (0, 255, 0), 2)
    cv2.putText(img, f"{d.label} {d.confidence:.2f}", (d.x - d.r, max(d.y - d.r - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("resultat", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
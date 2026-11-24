import argparse
import os
import cv2
import gate

p = argparse.ArgumentParser()
p.add_argument("nama")
p.add_argument("-n", type=int, default=12, help="target jumlah foto")
p.add_argument("--cam", type=int, default=0)
a = p.parse_args()

out = os.path.join(gate.BASE, "data", "dataset", a.nama.strip().replace(" ", "_").lower())
os.makedirs(out, exist_ok=True)
cap = cv2.VideoCapture(a.cam)
if not cap.isOpened():
    raise SystemExit("Kamera tidak bisa dibuka.")
n = len(os.listdir(out))
while True:
    ok, frame = cap.read()
    if not ok:
        break
    view = frame.copy()
    face = gate.detect_face(frame)
    cv2.putText(view, f"{n} foto | SPASI=ambil Q=selesai | wajah: {'OK' if face is not None else 'TIDAK TERDETEKSI'}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if face is not None else (0, 0, 255), 2)
    cv2.imshow("collect", view)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" ") and face is not None:     # hanya simpan kalau wajah terdeteksi
        n += 1
        cv2.imwrite(os.path.join(out, f"{n:03d}.jpg"), frame)
    elif k == ord("q") or n >= a.n:
        break
cap.release()
cv2.destroyAllWindows()
print(f"{n} foto tersimpan di {out}")

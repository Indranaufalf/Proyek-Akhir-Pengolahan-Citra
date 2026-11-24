
import json
import os
import sqlite3
from datetime import date
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "data", "gate.db")
FACES_DIR = os.path.join(BASE, "data", "sample_faces")
THRESH_FILE = os.path.join(BASE, "data", "threshold.json")  
SIZE = 128   
GRID = 8    

def load_threshold(default=0.245):
    try:
        with open(THRESH_FILE) as f:
            return float(json.load(f)["threshold"])
    except Exception:
        return default

THRESHOLD = load_threshold()

# database
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with db() as c:
        try:  
            c.execute("SELECT face FROM passengers LIMIT 1")
            c.executescript("DROP TABLE passengers; DROP TABLE tickets; DROP TABLE gate_logs;")
        except sqlite3.OperationalError:
            pass
        c.executescript("""
        CREATE TABLE IF NOT EXISTS passengers (
            nik TEXT PRIMARY KEY, nama TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS face_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT, nik TEXT NOT NULL, vec BLOB NOT NULL,
            FOREIGN KEY (nik) REFERENCES passengers(nik));
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_code TEXT PRIMARY KEY, nik TEXT NOT NULL, kereta TEXT,
            relasi TEXT, tanggal TEXT, status TEXT DEFAULT 'ACTIVE', waktu_boarding TEXT);
        CREATE TABLE IF NOT EXISTS gate_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            waktu TEXT DEFAULT (datetime('now','localtime')),
            gate TEXT, nik TEXT, nama TEXT, ticket_code TEXT,
            status TEXT, jarak REAL, pesan TEXT);
        """)

        if "tanggal" not in [r[1] for r in c.execute("PRAGMA table_info(tickets)")]:
            c.execute("ALTER TABLE tickets ADD COLUMN tanggal TEXT")

def _load_cascade(name):
    for folder in (os.path.join(BASE, "data"), getattr(cv2.data, "haarcascades", "")):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                return c
    return None

_FACE = _load_cascade("haarcascade_frontalface_default.xml")
_EYE = _load_cascade("haarcascade_eye.xml")

def _skin_boxes(img):
    """Cadangan KHUSUS DEMO (wajah sintetis): segmentasi warna kulit YCrCb.
    Di foto asli ini gampang salah (tangan/leher/background ikut kedeteksi) -> jangan dipakai di luar demo."""
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb), (0, 133, 77), (255, 173, 127))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2), cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= 60 and h >= 60 and 0.75 <= h / w <= 2.0:
            boxes.append((x, y, w, h))
    return boxes

def detect_face(img, mode="haar"):
    """mode: 'haar' (default, ketat) | 'haar+skin' (demo sintetis) | 'none' (gambar sudah berupa crop wajah).
    Return crop wajah terbesar (BGR) atau None."""
    if mode == "none":
        return img
    if _FACE is None:
        raise RuntimeError("File haarcascade_frontalface_default.xml tidak ketemu (harusnya ada di folder data/).")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = list(_FACE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60)))
    if not boxes and mode == "haar+skin":
        boxes = _skin_boxes(img)
    if not boxes:
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return img[y:y + h, x:x + w]

def align_face(face):
    """Putar wajah supaya garis mata horizontal. Kalau mata nggak terdeteksi 2 biji, dibiarkan apa adanya."""
    if _EYE is None:
        return face
    h, w = face.shape[:2]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)[: int(h * 0.6)]
    eyes = _EYE.detectMultiScale(gray, 1.1, 5, minSize=(max(8, w // 10), max(8, w // 10)))
    if len(eyes) < 2:
        return face
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = sorted(eyes, key=lambda e: e[0])
    c1, c2 = (x1 + w1 / 2, y1 + h1 / 2), (x2 + w2 / 2, y2 + h2 / 2)
    if abs(c2[0] - c1[0]) < w * 0.2:   # dua deteksi di mata yang sama
        return face
    angle = float(np.degrees(np.arctan2(c2[1] - c1[1], c2[0] - c1[0])))
    if abs(angle) > 30:                # nggak masuk akal -> abaikan
        return face
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(face, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def preprocess(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)                        # buang noise, tepi tetap tajam
    gray = cv2.createCLAHE(2.5, (8, 8)).apply(gray)                    # ratakan pencahayaan
    return cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)

def lbph(gray):
    """Tiap piksel dibandingin sama 8 tetangganya -> kode 0-255 (LBP).
    Gambar dibagi grid 8x8, tiap sel dihitung histogramnya, lalu digabung jadi 1 vektor."""
    h, w = gray.shape
    c = gray[1:-1, 1:-1]
    nb = [gray[1:-1, 2:], gray[:-2, 2:], gray[:-2, 1:-1], gray[:-2, :-2],
          gray[1:-1, :-2], gray[2:, :-2], gray[2:, 1:-1], gray[2:, 2:]]
    lbp = sum((n >= c).astype(np.uint8) << i for i, n in enumerate(nb))
    lbp = np.pad(lbp, 1, mode="edge")
    ch, cw = h // GRID, w // GRID
    hists = []
    for r in range(GRID):
        for q in range(GRID):
            cell = lbp[r * ch:(r + 1) * ch, q * cw:(q + 1) * cw]
            hist = np.bincount(cell.ravel(), minlength=256).astype(np.float32)
            hists.append(hist / hist.sum())
    vec = np.concatenate(hists)
    return vec / vec.sum()

def chi2(a, b):
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-10)))


def face_vector(img, detect="haar", align=True):
    """Gambar BGR -> vektor fitur, atau None kalau tidak ada wajah."""
    face = detect_face(img, detect)
    if face is None:
        return None
    if align:
        face = align_face(face)
    return lbph(preprocess(face))

# NIK, katalog kereta
def validate_nik(nik):
    """NIK harus 16 digit angka. Return (ok, pesan)."""
    nik = (nik or "").strip()
    if not (nik.isascii() and nik.isdigit()):
        return False, "NIK hanya boleh berisi angka."
    if len(nik) != 16:
        return False, f"NIK harus 16 digit (sekarang {len(nik)} digit)."
    return True, ""

# nama kereta
RUTE = {
    "Argo Parahyangan":   ("Bandung", "Gambir"),
    "Argo Lawu":          ("Solo Balapan", "Gambir"),
    "Taksaka":            ("Yogyakarta", "Gambir"),
    "Gajayana":           ("Malang", "Gambir"),
    "Argo Wilis":         ("Bandung", "Surabaya Gubeng"),
    "Argo Bromo Anggrek": ("Gambir", "Surabaya Pasarturi"),
}
KERETA = list(RUTE)

def relasi_opsi(kereta):
    a, b = RUTE[kereta]
    return [f"{a} - {b}", f"{b} - {a}"]


def next_ticket_code():
    """Kode tiket otomatis: KAI-0001, KAI-0002, ..."""
    with db() as c:
        r = c.execute("SELECT MAX(CAST(SUBSTR(ticket_code,5) AS INTEGER)) FROM tickets "
                      "WHERE ticket_code LIKE 'KAI-%'").fetchone()
    return f"KAI-{(r[0] or 0) + 1:04d}"

# regitrasi
def register(nik, nama, imgs, detect="haar", align=True, ticket=None):
    """Daftarkan 1 orang dengan >=1 foto (disarankan 3-5 foto beda ekspresi/cahaya).
    ticket (opsional): {"kode":..., "kereta":..., "relasi":..., "tanggal": "YYYY-MM-DD"}. Return jumlah foto yang terpakai."""
    nik = nik.strip()
    ok, pesan = validate_nik(nik)
    if not ok:
        raise ValueError(pesan)
    init_db()
    vecs = [v for v in (face_vector(i, detect, align) for i in imgs) if v is not None]
    if not vecs:
        raise ValueError("Tidak ada wajah terdeteksi di foto yang diberikan.")
    with db() as c:
        c.execute("INSERT OR REPLACE INTO passengers VALUES (?,?)", (nik, nama))
        c.execute("DELETE FROM face_templates WHERE nik=?", (nik,))
        c.executemany("INSERT INTO face_templates (nik, vec) VALUES (?,?)",
                      [(nik, v.astype(np.float32).tobytes()) for v in vecs])
        if ticket:
            c.execute("INSERT OR REPLACE INTO tickets (ticket_code,nik,kereta,relasi,tanggal) VALUES (?,?,?,?,?)",
                      (ticket["kode"], nik, ticket.get("kereta"), ticket.get("relasi"), ticket.get("tanggal")))
    return len(vecs)


# gate
def _log(gate, status, pesan, jarak, nik=None, nama=None, ticket=None):
    with db() as c:
        c.execute("INSERT INTO gate_logs (gate,nik,nama,ticket_code,status,jarak,pesan) VALUES (?,?,?,?,?,?,?)",
                  (gate, nik, nama, ticket, status, jarak, pesan))


def scan(img, gate="GATE-01", threshold=None, detect="haar", align=True):
    """Proses 1 gambar. Return dict: status, granted, pesan, nama, jarak."""
    threshold = THRESHOLD if threshold is None else threshold
    vec = face_vector(img, detect, align)
    if vec is None:
        return {"status": "NO_FACE", "granted": False, "pesan": "Wajah tidak terdeteksi.", "nama": None, "jarak": None}

    with db() as c:
        rows = c.execute("SELECT t.nik, p.nama, t.vec FROM face_templates t JOIN passengers p ON p.nik=t.nik").fetchall()
    best, best_d = None, 9.0
    for r in rows: 
        d = chi2(vec, np.frombuffer(r["vec"], dtype=np.float32))
        if d < best_d:
            best, best_d = r, d

    if best is None or best_d >= threshold:
        msg = "Wajah tidak terdaftar."
        _log(gate, "UNRECOGNIZED", msg, best_d)
        return {"status": "UNRECOGNIZED", "granted": False, "pesan": msg, "nama": None, "jarak": best_d}

    nik, nama = best["nik"], best["nama"]
    with db() as c:
        tiket = c.execute("SELECT * FROM tickets WHERE nik=? ORDER BY ticket_code", (nik,)).fetchall()

    if not tiket:
        msg = f"{nama} terdaftar tapi tidak punya tiket."
        _log(gate, "NO_TICKET", msg, best_d, nik, nama)
        return {"status": "NO_TICKET", "granted": False, "pesan": msg, "nama": nama, "jarak": best_d}

    # cek tiket
    hari_ini = date.today().isoformat()
    t = next((x for x in tiket if x["tanggal"] in (None, "", hari_ini)), None)
    if t is None:
        t = tiket[0]
        msg = f"Tiket {t['ticket_code']} berlaku untuk tanggal {t['tanggal']}, bukan hari ini ({hari_ini})."
        _log(gate, "WRONG_DATE", msg, best_d, nik, nama, t["ticket_code"])
        return {"status": "WRONG_DATE", "granted": False, "pesan": msg, "nama": nama, "jarak": best_d}

    if t["status"] == "BOARDED":
        msg = f"Tiket {t['ticket_code']} sudah dipakai pada {t['waktu_boarding']}."
        _log(gate, "ALREADY_BOARDED", msg, best_d, nik, nama, t["ticket_code"])
        return {"status": "ALREADY_BOARDED", "granted": False, "pesan": msg, "nama": nama, "jarak": best_d}

    with db() as c:
        c.execute("UPDATE tickets SET status='BOARDED', waktu_boarding=datetime('now','localtime') WHERE ticket_code=?",
                  (t["ticket_code"],))
    msg = f"Silakan masuk, {nama}. Selamat jalan dengan {t['kereta']}."
    _log(gate, "SUCCESS", msg, best_d, nik, nama, t["ticket_code"])
    return {"status": "SUCCESS", "granted": True, "pesan": msg, "nama": nama, "jarak": best_d}


def reset_tickets():
    with db() as c:
        c.execute("UPDATE tickets SET status='ACTIVE', waktu_boarding=NULL")

TIKET = { 
    "3273011204980001": ("KAI-0001", "Argo Parahyangan", "Bandung - Gambir"),
    "3578021509950003": ("KAI-0002", "Taksaka", "Yogyakarta - Gambir"),
    "3171054308990002": ("KAI-0003", "Argo Lawu", "Solo Balapan - Gambir"),
    "3374016601010004": ("KAI-0004", "Gajayana", "Malang - Gambir"),
}


def seed():
    """Daftarkan foto di data/sample_faces (nama file: NIK_nama.jpg), file 'impostor_*' dilewati.
    CATATAN: foto sampel itu wajah SINTETIS (gambar kartun) -> hanya untuk demo alur, bukan bukti akurasi."""
    init_db()
    for f in sorted(os.listdir(FACES_DIR)):
        if not f.endswith(".jpg") or f.startswith("impostor"):
            continue
        nik, nama = f[:-4].split("_", 1)
        nama = nama.replace("_", " ").title()
        tk = TIKET.get(nik)
        ticket = {"kode": tk[0], "kereta": tk[1], "relasi": tk[2]} if tk else None
        try:
            n = register(nik, nama, [cv2.imread(os.path.join(FACES_DIR, f))], detect="haar+skin", ticket=ticket)
            print("OK:", nik, nama, f"({n} template)")
        except ValueError:
            print("SKIP (wajah tak terdeteksi):", f)


if __name__ == "__main__":
    seed()
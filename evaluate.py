import argparse
import json
import os
import random
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gate

OUT = os.path.join(gate.BASE, "data", "evaluation_results")
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".pgm")
REJECT = "DITOLAK"


# data
def load_dataset(root):
    data = {}
    for p in sorted(os.listdir(root)):
        d = os.path.join(root, p)
        if not os.path.isdir(d):
            continue
        imgs = [cv2.imread(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.lower().endswith(EXT)]
        imgs = [i for i in imgs if i is not None]
        if imgs:
            data[p] = imgs
    return data

def pick_impostors(persons, n_imp):
    return sorted(random.Random(0).sample(sorted(persons), n_imp))

def make_split(data, enrolled, impostors, n_enroll, n_val, rng):
    s = {k: [] for k in ("enroll", "val_g", "test_g", "val_i", "test_i")}
    for p in enrolled:
        idx = list(range(len(data[p])))
        rng.shuffle(idx)
        s["enroll"] += [(p, j) for j in idx[:n_enroll]]
        s["val_g"] += [(p, j) for j in idx[n_enroll:n_enroll + n_val]]
        s["test_g"] += [(p, j) for j in idx[n_enroll + n_val:]]
    if len(impostors) >= 2:  
        half = len(impostors) // 2
        s["val_i"] = [(p, j) for p in impostors[:half] for j in range(len(data[p]))]
        s["test_i"] = [(p, j) for p in impostors[half:] for j in range(len(data[p]))]
    else:                    
        p = impostors[0]
        idx = list(range(len(data[p])))
        rng.shuffle(idx)
        h = len(idx) // 2
        s["val_i"] = [(p, j) for j in idx[:h]]
        s["test_i"] = [(p, j) for j in idx[h:]]
    return s


# pencocokan
def chi2_many(E, v):
    return 0.5 * np.sum((E - v) ** 2 / (E + v + 1e-10), axis=1)


def probe(vec, E, labels):
    """Return (orang_terdekat, jarak). vec None (wajah gagal terdeteksi) -> (None, inf)."""
    if vec is None:
        return None, np.inf
    d = chi2_many(E, vec)
    i = int(np.argmin(d))
    return labels[i], float(d[i])

def sweep(gen, imp, thresholds):
    gen, imp = np.asarray(gen), np.asarray(imp)
    frr = np.array([np.mean(gen >= t) for t in thresholds])
    far = np.array([np.mean(imp < t) for t in thresholds])
    return far, frr

def candidate_thresholds(*arrays):
    u = np.unique(np.concatenate([np.asarray(a)[np.isfinite(a)] for a in arrays]))
    if len(u) < 2:
        return u
    return np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])

def choose_threshold(gen, imp):
    th = candidate_thresholds(gen, imp)
    far, frr = sweep(gen, imp, th)
    i = int(np.argmin(np.abs(far - frr) + 1e-6 * (far + frr))) 
    return float(th[i])


# kondisi uji
def cond_dim(im):     return np.clip(255.0 * (im / 255.0) ** 1.8, 0, 255).astype(np.uint8)
def cond_bright(im):  return np.clip(255.0 * (im / 255.0) ** 0.55 + 20, 0, 255).astype(np.uint8)
def cond_noise(im):   return np.clip(im.astype(np.int16) + np.random.RandomState(1).normal(0, 12, im.shape), 0, 255).astype(np.uint8)
def cond_blur(im):    return cv2.GaussianBlur(im, (7, 7), 0)
def cond_tilt(im):
    h, w = im.shape[:2]
    return cv2.warpAffine(im, cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0), (w, h), borderMode=cv2.BORDER_REPLICATE)


CONDITIONS = {"Normal": lambda im: im, "Redup": cond_dim, "Silau": cond_bright,
              "Noise": cond_noise, "Blur": cond_blur, "Miring 10°": cond_tilt}


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(gate.BASE, "data", "dataset"))
    ap.add_argument("--enroll", type=int, default=3, help="foto/orang untuk template")
    ap.add_argument("--val", type=int, default=1, help="foto/orang untuk validasi threshold")
    ap.add_argument("--impostors", type=int, default=None, help="jumlah orang yang disisihkan sebagai orang asing")
    ap.add_argument("--seeds", type=int, default=5, help="jumlah pengulangan dengan pengacakan berbeda")
    ap.add_argument("--cropped", action="store_true", help="foto sudah berupa crop wajah (lewati deteksi)")
    ap.add_argument("--skin-fallback", action="store_true", help="izinkan fallback warna kulit (HANYA untuk wajah sintetis)")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--no-kondisi", action="store_true", help="lewati uji ketahanan")
    ap.add_argument("--simpan-threshold", action="store_true")
    a = ap.parse_args()

    detect = "none" if a.cropped else ("haar+skin" if a.skin_fallback else "haar")
    align = not a.no_align
    need = a.enroll + a.val + 1

    if not os.path.isdir(a.dataset):
        raise SystemExit(f"Folder dataset tidak ada: {a.dataset}\nBuat folder per orang berisi foto-fotonya (lihat docstring).")
    data = load_dataset(a.dataset)
    small = [p for p, v in data.items() if len(v) < need]
    if small:
        print(f"[!] Dilewati (foto < {need}): {', '.join(small)}")
    data = {p: v for p, v in data.items() if len(v) >= need}
    n_imp = a.impostors if a.impostors is not None else max(1, round(0.2 * len(data)))
    if len(data) - n_imp < 3 or n_imp < 1:
        raise SystemExit(f"Butuh minimal 3 orang terdaftar + 1 orang asing (ada {len(data)} orang valid, impostors={n_imp}).")
    impostors = pick_impostors(list(data), n_imp)
    enrolled = [p for p in sorted(data) if p not in impostors]
    print(f"Orang terdaftar: {len(enrolled)} | orang asing: {len(impostors)} {impostors} | detect={detect} align={align}")

    feats, n_fail = {}, 0
    for p, imgs in data.items():
        for j, im in enumerate(imgs):
            feats[(p, j)] = gate.face_vector(im, detect, align)
            n_fail += feats[(p, j)] is None
    print(f"Foto total: {len(feats)} | gagal deteksi wajah: {n_fail}")

    os.makedirs(OUT, exist_ok=True)
    rows, pooled_g, pooled_i = [], [], []
    cm_bin = np.zeros((2, 2), int)                      
    id_labels = enrolled + [REJECT]
    cm_id = np.zeros((len(enrolled) + 1, len(enrolled) + 1), int)   
    cm_id = np.zeros((len(enrolled) + 1, len(enrolled) + 1), int)
    cond_rows = []

    for seed in range(a.seeds):
        rng = random.Random(100 + seed)
        sp = make_split(data, enrolled, impostors, a.enroll, a.val, rng)
        tmpl = [(p, feats[(p, j)]) for p, j in sp["enroll"] if feats[(p, j)] is not None]
        if not tmpl:
            raise SystemExit("Tidak ada template yang bisa dibuat (semua wajah gagal terdeteksi).")
        E = np.stack([v for _, v in tmpl])
        labels = [p for p, _ in tmpl]

        def run(keys, getvec=lambda k: feats[k]):
            return [probe(getvec(k), E, labels) for k in keys]

        vg, vi = run(sp["val_g"]), run(sp["val_i"])
        gen_v = [d if pred == k[0] else np.inf for (pred, d), k in zip(vg, sp["val_g"])]
        imp_v = [d for _, d in vi]
        theta = choose_threshold(gen_v, imp_v)

        tg, ti = run(sp["test_g"]), run(sp["test_i"])
        gen_t = [d if pred == k[0] else np.inf for (pred, d), k in zip(tg, sp["test_g"])]
        imp_t = [d for _, d in ti]
        pooled_g += gen_t
        pooled_i += imp_t
        acc_g = np.array([(pred == k[0]) and (d < theta) for (pred, d), k in zip(tg, sp["test_g"])])
        acc_i = np.array([d < theta for _, d in ti])  
        TP, FN = int(acc_g.sum()), int((~acc_g).sum())
        FP, TN = int(acc_i.sum()), int((~acc_i).sum())
        cm_bin += np.array([[TN, FP], [FN, TP]])
        tot = TP + FN + FP + TN
        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = TP / (TP + FN) if TP + FN else 0.0
        rank1 = float(np.mean([pred == k[0] for (pred, _), k in zip(tg, sp["test_g"]) if pred is not None] or [0]))
        rows.append({"seed": seed, "threshold": theta, "accuracy": (TP + TN) / tot, "precision": prec, "recall": rec,
                     "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                     "FAR": FP / (FP + TN) if FP + TN else 0.0, "FRR": FN / (TP + FN) if TP + FN else 0.0,
                     "rank1_closed_set": rank1, "n_genuine": TP + FN, "n_impostor": FP + TN})
        for (pred, d), k in zip(tg, sp["test_g"]):
            cm_id[id_labels.index(k[0]), id_labels.index(pred if (pred and d < theta) else REJECT)] += 1
        for pred, d in ti:
            cm_id[len(enrolled), id_labels.index(pred if (pred and d < theta) else REJECT)] += 1

        if not a.no_kondisi:
            for cname, fn in CONDITIONS.items():
                def vec_c(k, fn=fn):
                    return gate.face_vector(fn(data[k[0]][k[1]]), detect, align)
                cg = run(sp["test_g"], vec_c)
                ci = run(sp["test_i"], vec_c)
                ok_g = np.array([(pred == k[0]) and (d < theta) for (pred, d), k in zip(cg, sp["test_g"])])
                ok_i = np.array([d >= theta for _, d in ci])
                cond_rows.append({"seed": seed, "kondisi": cname,
                                  "accuracy": (ok_g.sum() + ok_i.sum()) / (len(ok_g) + len(ok_i)),
                                  "FRR": 1 - ok_g.mean(), "FAR": 1 - ok_i.mean()})

        if seed == 0: 
            lat_keys, lat_E, lat_labels = sp["test_g"][:30], E, labels

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)
    cols = ["threshold", "accuracy", "precision", "recall", "f1", "FAR", "FRR", "rank1_closed_set"]
    summary = pd.DataFrame({"mean": per_seed[cols].mean(), "std": per_seed[cols].std(ddof=0)})
    summary.to_csv(os.path.join(OUT, "metrik.csv"))

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm_bin, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm_bin):
        ax.text(j, i, v, ha="center", va="center", fontsize=14, color="black")
    ax.set_xticks([0, 1], ["Ditolak", "Diterima"]); ax.set_yticks([0, 1], ["Orang asing", "Penumpang sah"])
    ax.set_xlabel("Keputusan gate"); ax.set_ylabel("Sebenarnya"); ax.set_title(f"Confusion matrix (jumlah {a.seeds} ulangan)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "confusion_biner.png"), dpi=150); plt.close(fig)

    row_labels = enrolled + ["ASING"]
    pd.DataFrame(cm_id[:len(enrolled) + 1], index=row_labels, columns=id_labels).to_csv(os.path.join(OUT, "confusion_identifikasi.csv"))
    if len(enrolled) <= 25:
        fig, ax = plt.subplots(figsize=(0.6 * len(id_labels) + 3, 0.6 * len(row_labels) + 2.5))
        ax.imshow(cm_id[:len(enrolled) + 1], cmap="Blues")
        for (i, j), v in np.ndenumerate(cm_id[:len(enrolled) + 1]):
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(id_labels)), id_labels, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Sebenarnya"); ax.set_title("Confusion matrix identifikasi 1:N")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "confusion_identifikasi.png"), dpi=150); plt.close(fig)

    th = candidate_thresholds(pooled_g, pooled_i)
    far, frr = sweep(pooled_g, pooled_i, th)
    k = int(np.argmin(np.abs(far - frr)))
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(th, far, label="FAR"); ax.plot(th, frr, label="FRR")
    ax.axvline(summary.loc["threshold", "mean"], ls="--", c="gray", label=f"threshold terpilih (validasi) = {summary.loc['threshold', 'mean']:.3f}")
    ax.set_xlabel("Threshold (jarak chi-square)"); ax.set_ylabel("Rate")
    ax.set_title(f"FAR vs FRR pada data test (EER ≈ {(far[k] + frr[k]) / 2:.1%})"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "far_frr_curve.png"), dpi=150); plt.close(fig)

    if cond_rows:
        cdf = pd.DataFrame(cond_rows).groupby("kondisi", sort=False)[["accuracy", "FRR", "FAR"]].mean()
        cdf.to_csv(os.path.join(OUT, "kondisi.csv"))
        fig, ax = plt.subplots(figsize=(6, 3.6))
        cdf["accuracy"].plot.bar(ax=ax, color="#1E2B58"); ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
        ax.set_title("Ketahanan terhadap kondisi (hanya foto test)"); plt.xticks(rotation=20)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "kondisi.png"), dpi=150); plt.close(fig)

    st = {"Deteksi": [], "Align+Preproses": [], "LBPH": [], "Pencocokan": []}
    for k_ in lat_keys:
        im = data[k_[0]][k_[1]]
        t0 = time.perf_counter(); face = gate.detect_face(im, detect); t1 = time.perf_counter()
        if face is None:
            continue
        g_ = gate.preprocess(gate.align_face(face) if align else face); t2 = time.perf_counter()
        v = gate.lbph(g_); t3 = time.perf_counter()
        probe(v, lat_E, lat_labels); t4 = time.perf_counter()
        for name, (x, y) in zip(st, [(t0, t1), (t1, t2), (t2, t3), (t3, t4)]):
            st[name].append((y - x) * 1000)
    lat = pd.DataFrame({"mean_ms": {k_: float(np.mean(v)) for k_, v in st.items() if v},
                        "std_ms": {k_: float(np.std(v)) for k_, v in st.items() if v}})
    if len(lat):
        lat.to_csv(os.path.join(OUT, "latensi.csv"))
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        lat["mean_ms"].plot.barh(ax=ax, color="#ED6B23"); ax.set_xlabel("ms per gambar")
        ax.set_title(f"Latensi per tahap (total {lat['mean_ms'].sum():.1f} ms ≈ {1000 / lat['mean_ms'].sum():.1f} FPS)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "latensi.png"), dpi=150); plt.close(fig)

    print("\n=== HASIL (data TEST, rata-rata ± std dari", a.seeds, "ulangan) ===")
    for m in cols:
        print(f"{m:18} {summary.loc[m, 'mean']:.4f} ± {summary.loc[m, 'std']:.4f}")
    print(f"gagal deteksi wajah: {n_fail}/{len(feats)} foto (dihitung sebagai ditolak)")
    if len(lat):
        print(f"latensi total: {lat['mean_ms'].sum():.1f} ms/gambar")
    print("Output ->", OUT)

    if a.simpan_threshold:
        with open(gate.THRESH_FILE, "w") as f:
            json.dump({"threshold": round(float(summary.loc["threshold", "mean"]), 4)}, f)
        print("Threshold disimpan ke", gate.THRESH_FILE)


if __name__ == "__main__":
    main()

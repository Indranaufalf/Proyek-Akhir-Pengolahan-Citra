# 🚄 KAI Face Recognition

**Kelompok Pengolahan Citra IF-B**  Proyek Akhir Pengolahan Citra

| Nama | NIM | Kontribusi |
|---|---|---|
| Rafi Dzaka Pratama Putra | 123240104 | Database dan tiket |
| Raihan Natawangsa | 123240120 | Antarmuka Streamlit |
| Ichsan Wahyu Priambudi | 123240122 | Deteksi wajah, LBPH |
| Indra Naufal Fidaus | 123240195 | Dataset dan evaluasi |

Prototipe gate stasiun berbasis wajah: deteksi wajah (OpenCV Haar) → penyejajaran mata → fitur LBPH → cocokkan ke template di database → cek tiket → catat log. Antarmuka web dengan Streamlit.

## Jalankan
```bash
pip install -r requirements.txt
python gate.py              # isi DB dengan data demo
streamlit run app.py
```
Akses kamera di browser hanya jalan di `localhost` atau HTTPS.

## Fitur
- **Daftar Wajah:** upload atau kamera, NIK 16 digit angka, tiket opsional (kereta, relasi, tanggal dipilih dari dropdown, kode tiket otomatis).
- **Scan:** foto sampel, upload, atau kamera. Tiket harus berlaku hari ini.
- **Log Database** dan **Tiket:** lihat data, filter status, download CSV.

Status scan: `SUCCESS`, `UNRECOGNIZED`, `NO_TICKET`, `WRONG_DATE`, `ALREADY_BOARDED` (dan `NO_FACE` kalau wajah tidak terdeteksi, tidak dicatat di log).

## File
| File | Fungsi |
|---|---|
| `gate.py` | Inti: database, deteksi wajah, LBPH, validasi NIK, cek tiket, log |
| `app.py` | Web Streamlit |
| `evaluate.py` | Evaluasi tanpa kebocoran data (metrik, confusion matrix, FAR/FRR, ketahanan, latensi) |
| `collect.py` | Ambil foto webcam untuk dataset (dengan izin) |
| `view_logs.py` | Lihat/export log dan tiket dari terminal |

## Akses dan edit data
Database: `data/gate.db` (SQLite). Tabel: `passengers`, `face_templates`, `tickets`, `gate_logs`.
- **Web:** tab *Log Database* dan *Tiket*.
- **Terminal:** `python view_logs.py` (`-n 100`, `-s STATUS`, `--csv log.csv`, `--tiket`).
- **Edit/hapus:** DB Browser for SQLite, lalu *Write Changes*. Hapus penumpang berurutan: `face_templates` → `tickets` → `passengers`.

## Evaluasi
Siapkan foto asli (dengan izin), minimal 6 orang dan 8 foto/orang, di `data/dataset/nama_orang/001.jpg, ...` (atau `python collect.py "nama" -n 12`), lalu:
```bash
python evaluate.py --simpan-threshold    # foto penuh
python evaluate.py --cropped             # foto sudah crop wajah
python evaluate.py --no-align            # tanpa penyejajaran mata (perbandingan)
```
Hasil di `data/evaluation_results/`: accuracy, precision, recall, F1, FAR, FRR, rank-1, confusion matrix, kurva FAR/FRR, uji ketahanan (redup, silau, noise, blur, miring), dan latensi.
Protokol: enroll / validation / test tidak tumpang tindih, orang asing disisihkan untuk FAR, threshold dipilih dari validation, 5 pengulangan (mean ± std).

## Data dan pustaka
- **Data:** foto orang tidak disertakan di repo (data pribadi, diabaikan lewat `.gitignore`). Untuk memperolehnya, kumpulkan sendiri dengan izin (misalnya anggota kelompok) memakai `collect.py`. `data/sample_faces/` hanya wajah sintetis (kartun) untuk demo alur, bukan bukti akurasi.
- **Pustaka:** OpenCV (Haar Cascade, filter, CLAHE), NumPy, pandas, matplotlib (grafik evaluasi), Streamlit (antarmuka). LBPH ditulis sendiri di `gate.py`.

## Keterbatasan
- LBPH sensitif terhadap pose, ekspresi, dan pencahayaan.
- Tidak ada liveness detection: foto/layar orang lain bisa mengecoh.
- Tidak ada autentikasi; vektor wajah dan NIK disimpan polos di SQLite. Cukup untuk prototipe, bukan produksi.
- Data wajah adalah data pribadi: pakai dengan izin dan jangan commit ke repo publik.
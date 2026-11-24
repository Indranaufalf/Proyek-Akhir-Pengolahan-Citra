import os
from datetime import date
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import gate

st.set_page_config(page_title="KAI Face Recognition", page_icon="🚄")
gate.init_db()
st.title("KAI Face Recognition")

tab_scan, tab_daftar, tab_log, tab_tiket = st.tabs(["Scan", "Daftar Wajah", "Log Database", "Tiket"])

# Scan
with tab_scan:
    sumber = st.radio("Sumber gambar", ["Foto sampel", "Upload", "Kamera"], horizontal=True)
    img = None
    if sumber == "Foto sampel":
        nama = st.selectbox("Pilih foto", sorted(f for f in os.listdir(gate.FACES_DIR) if f.endswith(".jpg")))
        img = cv2.imread(os.path.join(gate.FACES_DIR, nama))
    elif sumber == "Upload":
        up = st.file_uploader("Foto wajah", type=["jpg", "jpeg", "png"])
        if up:
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    else:
        cam = st.camera_input("Ambil foto")
        if cam:
            img = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    demo = st.checkbox("Mode demo wajah sintetis (fallback warna kulit)", value=(sumber == "Foto sampel"), key=f"demo_{sumber}")
    threshold = st.slider("Threshold (makin kecil makin ketat)", 0.10, 0.60, float(min(gate.THRESHOLD, 0.6)), 0.005)
    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=240)
        if st.button("Scan", type="primary"):
            r = gate.scan(img, threshold=threshold, detect="haar+skin" if demo else "haar")
            (st.success if r["granted"] else st.error)(f"{r['status']} — {r['pesan']}")
            if r["jarak"] is not None:
                st.caption(f"Jarak chi-square: {r['jarak']:.3f} (lolos kalau < {threshold})")

    if st.button("Reset semua tiket ke ACTIVE"):
        gate.reset_tickets()
        st.toast("Tiket di-reset")

# Daftar
with tab_daftar:
    st.caption("Daftarkan wajah")
    nik = st.text_input("NIK (16 digit angka)", max_chars=16)
    nik_ok = False
    if nik:
        nik_ok, pesan_nik = gate.validate_nik(nik)
        if not nik_ok:
            st.error(pesan_nik)
        else:
            with gate.db() as c:
                ada = c.execute("SELECT nama FROM passengers WHERE nik=?", (nik.strip(),)).fetchone()
            st.caption("Format NIK valid")
            if ada:
                st.info(f"NIK ini sudah terdaftar atas nama {ada['nama']}. Mendaftar lagi akan mengganti foto lamanya.")
    nama_d = st.text_input("Nama")

    sumber_d = st.radio("Sumber foto", ["Upload", "Kamera"], horizontal=True, key="sumber_daftar")
    imgs = []
    if sumber_d == "Upload":
        files = st.file_uploader("Foto wajah (boleh lebih dari satu)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        imgs = [cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR) for f in files]
    else:
        st.session_state.setdefault("cam_fotos", [])
        st.session_state.setdefault("cam_n", 0)
        cam = st.camera_input("Ambil foto (variasikan cahaya/ekspresi/arah hadap)",
                              key=f"cam_daftar_{st.session_state.cam_n}")
        if cam:
            frame = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if gate.detect_face(frame) is None:
                st.warning("Wajah tidak terdeteksi. Hadap depan, cahaya cukup, lalu ambil ulang.")
            elif st.button("Tambahkan foto ini"):
                st.session_state.cam_fotos.append(frame)
                st.session_state.cam_n += 1
                st.rerun()
        imgs = st.session_state.cam_fotos
        st.caption(f"{len(imgs)} foto terkumpul")
        if imgs:
            st.image([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in imgs], width=100)
            if st.button("Hapus semua foto"):
                st.session_state.cam_fotos = []
                st.rerun()

    tk = None
    with st.expander("Tiket (opsional)"):
        if st.checkbox("Buat tiket untuk penumpang ini"):
            kereta = st.selectbox("Kereta", gate.KERETA)
            relasi = st.selectbox("Relasi", gate.relasi_opsi(kereta))
            tgl = st.date_input("Tanggal keberangkatan", value=date.today(), min_value=date.today())
            kode = gate.next_ticket_code()
            st.caption(f"Kode tiket (otomatis): {kode}")
            tk = {"kode": kode, "kereta": kereta, "relasi": relasi, "tanggal": tgl.isoformat()}

    if st.button("Daftarkan"):
        if not (nik and nama_d.strip() and imgs):
            st.warning("NIK, nama, dan minimal 1 foto wajib diisi")
        elif not nik_ok:
            st.error("NIK belum valid, cek pesan di atas")
        else:
            try:
                n = gate.register(nik, nama_d.strip(), imgs, ticket=tk)
                st.success(f"{nama_d} terdaftar dengan {n} dari {len(imgs)} foto (sisanya wajah tidak terdeteksi).")
                st.session_state.cam_fotos = []
            except ValueError as e:
                st.error(str(e))

# Log
with tab_log:
    status = st.selectbox("Filter status", ["(semua)", "SUCCESS", "UNRECOGNIZED", "ALREADY_BOARDED", "NO_TICKET", "WRONG_DATE"])
    sql = "SELECT id, waktu, gate, nama, status, ROUND(jarak,3) AS jarak, pesan FROM gate_logs"
    args = ()
    if status != "(semua)":
        sql += " WHERE status = ?"
        args = (status,)
    with gate.db() as c:
        df = pd.read_sql(sql + " ORDER BY id DESC LIMIT 200", c, params=args)
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), "gate_logs.csv", "text/csv")

# Tiket
with tab_tiket:
    with gate.db() as c:
        st.dataframe(pd.read_sql(
            "SELECT t.ticket_code, p.nama, t.kereta, t.relasi, t.tanggal, t.status, t.waktu_boarding "
            "FROM passengers p LEFT JOIN tickets t ON t.nik = p.nik", c),
            width="stretch", hide_index=True)
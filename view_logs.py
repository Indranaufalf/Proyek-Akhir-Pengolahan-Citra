import argparse
import sqlite3
import pandas as pd
from gate import DB_PATH

p = argparse.ArgumentParser()
p.add_argument("-n", type=int, default=20, help="jumlah baris")
p.add_argument("-s", "--status", help="filter status")
p.add_argument("--csv", help="export ke file csv")
p.add_argument("--tiket", action="store_true", help="tampilkan tabel tiket")
a = p.parse_args()

conn = sqlite3.connect(DB_PATH)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

if a.tiket:
    print(pd.read_sql("SELECT t.ticket_code, p.nama, t.kereta, t.relasi, t.tanggal, t.status, t.waktu_boarding "
                      "FROM tickets t JOIN passengers p ON p.nik = t.nik", conn))
else:
    sql = "SELECT id, waktu, gate, nama, status, ROUND(jarak,3) AS jarak, pesan FROM gate_logs"
    args = []
    if a.status:
        sql += " WHERE status = ?"
        args.append(a.status.upper())
    sql += " ORDER BY id DESC" + ("" if a.csv else " LIMIT ?")
    if not a.csv:
        args.append(a.n)
    df = pd.read_sql(sql, conn, params=args)
    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"{len(df)} log disimpan ke {a.csv}")
    else:
        print(df.to_string(index=False) if len(df) else "Belum ada log.")
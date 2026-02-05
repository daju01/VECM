# Panduan Trading Harian (Daily Signal)

Dokumen ini menjelaskan cara menyiapkan *daily signal* berbasis pipeline VECM, menjalankan prosesnya secara manual maupun otomatis, membaca hasil sinyal, serta menambahkan notifikasi sederhana.

## Prerequisite sederhana

1. **Python 3 + virtual environment (opsional tapi disarankan).**
2. **Satu command install dependensi** (di root repo):
   ```bash
   pip install -r requirements.txt
   ```

> Catatan: `requirements.txt` di root menunjuk ke dependensi utama di subfolder proyek, jadi cukup satu perintah di atas sebelum menjalankan script harian.

## Mengatur `daily_pairs.json`

Konfigurasi pair harian ada di `vecm_project/config/daily_pairs.json` dan dibaca oleh `vecm_project/scripts/daily_signal.py` sebagai input utama untuk daftar pair dan parameter default. Formatnya mendukung dua gaya:

1. **Bentuk lengkap (direkomendasikan):**
   ```json
   {
     "persist_artifacts": true,
     "default_params": {
       "z_entry": 2.0,
       "z_exit": 0.5,
       "cooldown": 5
     },
     "pairs": [
       {"pair": "BBCA.JK,BBRI.JK"},
       {"pair": "BMRI.JK,BBNI.JK", "params": {"z_entry": 2.5}}
     ]
   }
   ```

2. **Bentuk sederhana (list string saja):**
   ```json
   [
     "BBCA.JK,BBRI.JK",
     "BMRI.JK,BBNI.JK"
   ]
   ```

Keterangan penting:
- `pairs` wajib berupa daftar. Tiap entry bisa string (nama pair) atau objek dengan `pair`/`params`.
- `default_params` akan dipakai untuk semua pair, lalu ditimpa oleh `params` jika ada.
- `persist_artifacts` mengatur apakah hasil run harian disimpan sebagai artefak pipeline.

## Menjalankan daily signal

### Manual
Jalankan dari root repo:
```bash
python -m vecm_project.scripts.daily_signal
```
Script akan:
- Membaca `daily_pairs.json`.
- Menjalankan pipeline untuk setiap pair.
- Menulis output ke folder `vecm_project/outputs/daily/` dalam format JSON dan CSV.

### Otomatis via cron
Contoh menjalankan setiap hari jam 17:05 (waktu server):
```cron
5 17 * * * cd /workspace/VECM && /usr/bin/python -m vecm_project.scripts.daily_signal >> /workspace/VECM/vecm_project/outputs/daily/cron.log 2>&1
```
> Sesuaikan path Python dan lokasi repo sesuai mesin Anda.

## Cara membaca sinyal & metrik (bahasa sederhana)

Output harian disimpan di file:
- `vecm_project/outputs/daily/daily_signal_YYYYMMDD.json`
- `vecm_project/outputs/daily/daily_signal_YYYYMMDD.csv`

Kolom utama yang perlu dipahami:

- **`direction`**: arah sinyal terakhir untuk pair tersebut.
  - `LONG` → *indikasi beli spread* (A murah vs B).
  - `SHORT` → *indikasi jual spread* (A mahal vs B).
  - `FLAT` → tidak ada sinyal baru (tunggu).
- **`confidence`**: seberapa kuat sinyal relatif terhadap ambang z-score.
  - Nilai mendekati 1 berarti sinyal kuat.
  - Nilai kosong (`null`) berarti z-score/threshold tidak tersedia.
- **`expected_holding_period`**: estimasi rata-rata lama hold (hari).
- **`metrics.z_score`**: z-score terakhir spread (indikasi seberapa “jauh” spread dari equilibrium).
- **`metrics.regime`**: probabilitas spread sedang di regime mean-reverting (mendekati 1 berarti lebih aman).
- **`metrics.overlay.delta_score` / `metrics.overlay.delta_mom12`**: sinyal overlay jangka pendek dan momentum 12 bulan.

Interpretasi sederhana:
- **LONG + confidence tinggi + regime tinggi** → sinyal relatif lebih “bersih”.
- **FLAT** → tidak ada entry baru, biasanya tunggu kembalinya spread ke area masuk.
- **z_score kecil** → spread belum cukup ekstrem.

## Mengaktifkan notifikasi Telegram/email

Pipeline default **belum mengirim notifikasi otomatis**. Cara termudah adalah menambahkan *wrapper script* kecil yang menjalankan daily signal lalu mengirim isi file output ke Telegram/email.

### Contoh notifikasi Telegram (curl)
Siapkan variabel environment:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Lalu jalankan:
```bash
python -m vecm_project.scripts.daily_signal
latest_json=$(ls -t vecm_project/outputs/daily/daily_signal_*.json | head -n 1)
message=$(python - <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
lines = [f"{item['pair']}: {item['direction']} (conf={item.get('confidence')})" for item in data]
print("\n".join(lines))
PY
"$latest_json")

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
  -d "chat_id=${TELEGRAM_CHAT_ID}" \
  --data-urlencode "text=${message}"
```

### Contoh notifikasi email (mailx/sendmail)
Siapkan MTA lokal atau gunakan `mailx`:
```bash
python -m vecm_project.scripts.daily_signal
latest_csv=$(ls -t vecm_project/outputs/daily/daily_signal_*.csv | head -n 1)
mailx -s "Daily Signal" you@example.com < "$latest_csv"
```
> Jika server Anda tidak punya MTA, gunakan layanan SMTP eksternal atau tool seperti `msmtp`.

## Troubleshooting umum

1. **Data kosong / `direction` selalu `FLAT`.**
   - Pastikan cache harga `adj_close_data.csv` tersedia dan update. Jalankan `python vecm_project/run_demo.py` sekali untuk memastikan data unduh & cache terisi.
   - Cek apakah pair di `daily_pairs.json` valid dan format `A,B` sudah benar.

2. **Gagal download data harga.**
   - Pastikan koneksi internet aktif.
   - Jika Yahoo Finance *rate-limited*, coba ulang di jam berbeda.
   - Hapus cache rusak dan jalankan ulang demo agar `ensure_price_data()` mengunduh ulang.

3. **Environment variable notifikasi belum diset.**
   - Telegram: pastikan `TELEGRAM_BOT_TOKEN` dan `TELEGRAM_CHAT_ID` sudah di-*export*.
   - Email: pastikan MTA / SMTP sudah tersedia, atau konfigurasi tool seperti `mailx`/`msmtp`.

---

Jika Anda ingin memperluas notifikasi (misalnya Slack/Discord), cukup gunakan output JSON/CSV harian sebagai input ke integrasi Anda.
# Trading Guide

## Konfigurasi Pair Harian

File konfigurasi untuk pair harian berada di `vecm_project/config/daily_pairs.json`. File ini berisi daftar pasangan saham yang dipantau setiap hari beserta parameter strateginya.

### Struktur JSON

```json
{
  "pairs": [
    {
      "tickerA": "AAPL",
      "tickerB": "MSFT",
      "z_entry": 2.0,
      "z_stop": 3.0,
      "max_hold": 5,
      "cooldown": 2
    }
  ]
}
```

### Cara Mengubah Konfigurasi

1. Buka file `vecm_project/config/daily_pairs.json`.
2. Untuk menambah pair baru, tambahkan objek baru pada array `pairs`.
3. Untuk mengubah parameter, edit nilai di dalam objek pair yang diinginkan.

### Penjelasan Parameter

- `tickerA` dan `tickerB`: simbol saham yang akan dipasangkan.
- `z_entry`: ambang z-score untuk membuka posisi.
- `z_stop`: ambang z-score untuk menghentikan posisi.
- `max_hold`: batas maksimum (dalam hari) posisi boleh terbuka.
- `cooldown`: jumlah hari menunggu sebelum pair boleh diperdagangkan lagi setelah posisi ditutup.

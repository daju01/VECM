# VECM

for stock market price analysis

## Getting started

1. **Siapkan virtual environment dan dependensi.**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r vecm_project/requirements.txt
   ```
   Langkah ini memastikan modul inti seperti `duckdb`, `pandas`, dan `statsmodels`
   tersedia sebelum pipeline dijalankan.

2. **Unduh dan validasi data harga pertama kali.**
   Jalankan demo end-to-end sesaat setelah instalasi:
   ```bash
   python vecm_project/run_demo.py
   ```
   Skrip akan memanggil `ensure_price_data()` dari
   `vecm_project/scripts/data_streaming.py` sehingga file cache
   `adj_close_data.csv` dibuat/di-update otomatis dengan minimal dua ticker dan
   kolom tanggal tervalidasi. Jika cache rusak, fungsi tersebut mengunduh ulang
   harga penutupan disesuaikan dari Yahoo Finance sebelum analisis dimulai. Bila
   koneksi internet diblokir, loader otomatis jatuh kembali ke dataset sampel
   `sample_adj_close.csv` (4 ticker perbankan Indonesia, 300 hari kerja) agar
   pipeline tetap bisa dieksekusi untuk smoke test.

3. **Konfirmasi storage DuckDB.**
   `run_demo.py` menggunakan konteks `managed_storage()` untuk memanggil
   `storage_init`, membuat seluruh tabel dan indeks yang dibutuhkan agar metrik,
   artefak model, dan log optimisasi tersimpan konsisten untuk audit berikutnya.

4. **Pastikan artefak pipeline dipersistenkan.**
   Jalankan demo hingga selesai sehingga fungsi `persist_artifacts` menyimpan
   posisi, return, trade, metrik, manifest, dan catatan model ke filesystem dan
   DuckDB. Keluaran terstruktur inilah yang diperlukan untuk verifikasi ulang
   Sharpe/Drawdown serta audit performa.

## Operational checklist

Sebelum menjalankan eksperimen lanjutan, pastikan empat pilar berikut sudah
dipenuhi—ini menjawab pertanyaan _“apakah Anda sudah melakukan ini?”_ yang
sering muncul saat men-deploy playbook:

1. **Kualitas data harga.** Pastikan `ensure_price_data()` dijalankan sampai
   selesai sehingga cache `adj_close_data.csv` terisi dengan dua ticker atau
   lebih, tanggal tervalidasi, serta duplikat tersaring.
2. **Konsistensi storage.** Gunakan konteks `managed_storage()` agar
   `storage_init` menyiapkan tabel DuckDB sebelum metrik dicatat.
3. **Pipeline menyeluruh.** Biarkan `run_demo.main()` mengeksekusi skor cepat,
   playbook penuh, hook ekspor, Bayesian optimisation, successive halving, dan
   penulisan Pareto/dashboard.
4. **Artefak lengkap.** Pastikan `persist=True` sehingga artefak dan metrik
   terekam untuk audit maupun analisis lanjutan.

### When `pip install` is blocked

Some sandboxed or corporate environments intercept outbound HTTPS traffic and
return `403 Forbidden` for the Python Package Index. If you see repeated
`Cannot connect to proxy` errors while installing the requirements, try one of
the following approaches:

1. **Use WSL or any machine with open internet access.** Clone the repository
   inside your WSL distribution (or another Linux/macOS host) and run the
   commands above there. Once the virtual environment is populated you can run
   the demo directly from that machine.
2. **Pre-download the wheels.** On a machine with internet access execute
   `pip download -r vecm_project/requirements.txt -d wheels/`, copy the
   resulting `wheels/` directory to the restricted environment, and install via
   `pip install --no-index --find-links wheels -r vecm_project/requirements.txt`.
3. **Point pip at an internal mirror.** If your organisation provides a
   whitelisted PyPI mirror, export `PIP_INDEX_URL=<mirror-url>` before running
   `pip install`.

Without at least one of these workarounds the Python interpreter will be
missing dependencies such as `duckdb`, `pandas`, and `statsmodels`, and the
demo script will terminate immediately with `ModuleNotFoundError`.

## Testing

To run the lightweight sanity check that we ship with the project, make sure
the dependencies above are installed and then execute:

```bash
python -m compileall vecm_project
```

The command only compiles the sources, so it runs quickly and does not require
network access once the packages are available in your virtual environment.

## Runtime controls

Pipeline demo menjalankan rangkaian penuh: skor cepat, playbook utama dengan
`persist=True`, hook ekspor, optimisasi Bayesian (`run_bo`), successive halving,
hingga penulisan front Pareto dan ringkasan dashboard. Seluruh keluaran yang
terstruktur—posisi, return, trades, manifest, metrik—dibuat secara otomatis
melalui `persist_artifacts` sehingga presisi Sharpe/Drawdown dan histori run
dapat diverifikasi ulang.

Penerjemahan Python ini sekarang memakai konfigurasi ringan agar demo offline
selesai dalam beberapa menit. Anda dapat menyesuaikan beban kerja melalui
variabel lingkungan:

* ``VECM_MAX_GRID`` membatasi jumlah job yang dibuat ``parallel_run``
  (default ``48``). Naikkan nilainya jika ingin menyapu grid Stage-1 lebih luas.
* ``run_bo`` dijalankan dengan dua percobaan (1 inisiasi + 1 langkah TPE) agar
  lintasan ringan. Ganti ``n_init`` atau ``iters`` bila Anda memerlukan eksplorasi
  Bayesian lebih dalam.
* ``run_successive_halving`` mengevaluasi satu trial pada horizon
  ``("short", "long")``. Tambah ``n_trials`` atau beri tuple ``horizons`` kustom
  untuk pass yang lebih ekstensif.

Default ini menjaga optimisasi tetap responsif, tetapi seluruh data dan artefak
tetap lengkap sehingga dapat dianalisis lebih lanjut ketika Anda memperluas
pencarian parameter.

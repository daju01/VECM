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
   harga penutupan disesuaikan dari Yahoo Finance sebelum analisis dimulai.

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

## Menjalankan analisis untuk ticker kustom

Pipeline VECM dapat dieksekusi dengan pasangan ticker apa pun yang tersedia di
Yahoo Finance. Gunakan opsi `--subset` untuk memberi tahu playbook pasangan mana
yang ingin dianalisis tanpa perlu mengubah kode ataupun daftar default pada
`parallel_run`.

### Contoh satu pasangan (playbook_vecm)

Perintah berikut memuat cache `adj_close_data.csv`, memastikan harga BBRI/BBNI
tersedia (akan diunduh otomatis jika belum ada), lalu menjalankan playbook
TVECM pada pasangan tersebut:

```bash
python -m vecm_project.scripts.playbook_vecm \
  vecm_project/data/adj_close_data.csv \
  --subset BBRI.JK,BBNI.JK \
  --method TVECM
```

### Contoh multi-pair (parallel_run)

Jika ingin memproses beberapa pasangan sekaligus, `parallel_run.py` menerima
daftar subset yang sama. Pasangan yang tidak ada di cache akan otomatis
diunduh sebelum analisis dijalankan.

```bash
python -m vecm_project.scripts.parallel_run \
  --subs BBRI.JK BBNI.JK \
  --subs BBCA.JK BMRI.JK
```

Set variabel lingkungan `VECM_PRICE_DOWNLOAD=force` apabila ingin memaksa
pembaruan data harga terlepas dari keberadaan cache lokal.

## Runtime controls

Pipeline demo menjalankan rangkaian penuh: skor cepat, playbook utama dengan
`persist=True`, hook ekspor, optimisasi Bayesian (`run_bo`), successive halving,
hingga penulisan front Pareto dan ringkasan dashboard. Seluruh keluaran yang
terstruktur—posisi, return, trades, manifest, metrik—dibuat secara otomatis
melalui `persist_artifacts` sehingga presisi Sharpe/Drawdown dan histori run
dapat diverifikasi ulang.

Penerjemahan Python ini tetap memakai default konservatif sehingga satu kali
demo selesai dalam ±1 jam di laptop. Anda dapat menyesuaikan beban kerja melalui
variabel lingkungan:

* ``VECM_MAX_GRID`` membatasi jumlah job yang dibuat ``parallel_run``
  (default ``48``). Naikkan nilainya jika ingin menyapu grid Stage-1 lebih luas.
* ``run_bo`` menjalankan 16 trial secara baku (4 inisiasi + 12 langkah TPE) dan
  membatasi pekerja paralel pada empat core logis. Ganti ``n_init`` atau
  ``iters`` bila Anda memerlukan eksplorasi Bayesian lebih dalam.
* ``run_successive_halving`` mengevaluasi hingga 12 trial pada horizon
  ``("short", "long")`` sambil membatasi paralelisme ke empat pekerja. Tambah
  ``n_trials`` atau beri tuple ``horizons`` kustom untuk pass yang lebih ekstensif.

Default ini menjaga optimisasi tetap responsif, tetapi seluruh data dan artefak
tetap lengkap sehingga dapat dianalisis lebih lanjut ketika Anda memperluas
pencarian parameter.

## Regime-aware pairs trading & short-term overlay

Sejak versi terbaru, pipeline VECM tidak lagi hanya mengandalkan z-score spread
klasik. Ada tiga layer tambahan yang aktif secara default:

1. **Convergence & mean-reversion filter**

   - Untuk setiap pair, playbook mengestimasi hubungan jangka panjang
     (cointegration + error-correction) dan menghitung:
       - `alpha_ec` – speed of adjustment dari error-correction,
       - `half_life_full` – estimasi half-life konvergensi spread.
   - Pada tahap prefilter (`parallel_run`), pair dengan half-life terlalu besar
     atau tidak cukup mean-reverting akan dibuang terlebih dahulu, sehingga grid
     Stage-1/Stage-2 hanya diisi pair yang memang cenderung kembali ke
     equilibrium.

2. **Regime-spread gating (Markov switching)**

   - Di dalam `playbook_vecm.run_playbook`, spread / z-score dimodelkan dengan
     Markov-switching 2-state sederhana (spread mean-reverting vs non-MR).
   - Untuk setiap hari, pipeline menghitung probabilitas `p_regime` bahwa
     spread sedang berada di regime **mean-reverting**.
   - Rule entry menjadi:

     > masuk posisi hanya jika `|z_t| > z_entry` **dan**
     > `p_regime_t >= p_th`,
     > serta gate korelasi + half-life (bila `gate_enforce=1`) terpenuhi.

3. **Short-term signal overlay (Blitz-style)**

   - Modul `short_term_signals.py` membangun sinyal jangka pendek dari
     `adj_close_data.csv`, antara lain:
       - 1M momentum,
       - 5D reversal (return 5 hari terakhir),
       - idiosyncratic volatility 1 bulan vs indeks pasar,
       - seasonality bulanan (efek bulan dalam setahun).
   - Setiap sinyal diubah menjadi robust cross-sectional z-score yang
     di-cap pada ±3 dan dirata-ratakan menjadi `score_short`.
   - Untuk setiap pasangan (A,B), pipeline menghitung
     `delta_score = score_short_A - score_short_B` dan hanya mengizinkan entry
     jika arah `delta_score` konsisten dengan arah mispricing spread:
       - kalau `z_t > 0` (A relatif mahal vs B) → butuh `delta_score > 0`
         (A kelihatan "lebih jelek" secara sinyal jangka pendek),
       - kalau `z_t < 0` → butuh `delta_score < 0`.

Hasilnya, trade hanya muncul ketika:
**spread mispricing + hubungan pair masih MR + sinyal jangka pendek setuju.**

### Cara mengaktifkan / menonaktifkan layer baru

Semua layer di atas **aktif secara default** ketika Anda menjalankan:

```bash
python vecm_project/run_demo.py
# atau langsung:
python -m vecm_project.scripts.playbook_vecm
```

Beberapa parameter penting yang bisa diutak-atik:

* **Convergence & gating**

  * `--gate_enforce` (0/1) mengontrol apakah gate korelasi + half-life
    dipaksa. Default: `1` (ON).
  * `--half_life_max` (default 90 hari) menentukan batas maksimum half-life
    yang masih diterima oleh gate half-life. Untuk eksperimen tanpa filter
    konvergensi, Anda dapat:

    * set `--gate_enforce 0`, ATAU
    * memberikan `--half_life_max` yang sangat besar.

* **Regime gating**

  * `--p_th` (default 0.85) adalah ambang probabilitas regime mean-reverting
    yang dibutuhkan untuk entry. Semakin tinggi `p_th`, semakin ketat filter
    (trade lebih jarang tapi biasanya lebih bersih).
  * Untuk relaksasi gate, turunkan `--p_th` (misalnya ke 0.60) sehingga trade
    boleh terjadi walaupun `p_regime` tidak terlalu tinggi.

* **Biaya transaksi & penalti turnover**

  * Biaya komisi + pajak dimodelkan eksplisit di level trade:

    * CLI: `--fee_buy` dan `--fee_sell`,
    * atau variabel lingkungan:

      * `PLAYBOOK_FEE_BUY`,
      * `PLAYBOOK_FEE_SELL`.
  * Stage-2 BO dan SH menggunakan objective berbasis:

    > `Score = sharpe_oos – λ × turnover_annualised`

    di mana λ dikontrol oleh environment:

    * `STAGE2_LAMBDA_TURNOVER` (default `"0.01"`).

    Semakin besar λ, semakin kuat penalti untuk strategi dengan
    `turnover_annualised` tinggi (lebih cocok untuk market ber-biaya tinggi
    seperti IDX).

* **Short-term overlay**

  * Short-term overlay akan aktif secara otomatis begitu panel harga
    `adj_close_data.csv` tersedia, karena modul `short_term_signals` dipanggil
    dari `run_playbook`.
  * Jika Anda ingin membandingkan performa dengan/ tanpa overlay, cara paling
    sederhana adalah meng-comment blok yang menggunakan `delta_score` di
    `playbook_vecm.build_signals`, sehingga rule entry kembali hanya bergantung
    pada z-score + gating regime.

Dengan dokumentasi ini, tiga bulan lagi ketika Anda buka repo, Anda bisa
langsung mengingat:
* bagaimana layer *regime-aware* dan *short-term overlay* bekerja,
* tombol mana saja (CLI / env) yang bisa di-tune,
* dan kenapa sebuah run dengan Sharpe tinggi bisa tetap disingkirkan (misalnya
  karena half-life terlalu lambat atau turnover terlalu agresif).

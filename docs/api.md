# API & Monitoring

Dokumen ini merangkum endpoint HTTP yang tersedia di dashboard VECM serta
artefak monitoring yang bisa dipakai untuk observability ringan.

## Dashboard HTTP Endpoints

> **Auth**: Semua endpoint (kecuali `/healthz`) membutuhkan Basic Auth jika
> `DASHBOARD_USER` dan `DASHBOARD_PASS` diset. Jika belum diset, dashboard akan
> mengembalikan error konfigurasi.

### `GET /`

Render dashboard ringkasan run terakhir beserta chart spread/z-score.

### `GET /healthz`

Health check ringan tanpa autentikasi. Contoh response:

```json
{
  "status": "ok",
  "price_data_age_hours": 3.25,
  "daily_signal_age_hours": 12.5
}
```

`status` bernilai `"stale"` jika daily signal lebih lama dari 48 jam.

### `GET /metrics`

Expose metrik Prometheus (jika `prometheus_client` terpasang).

### `GET|POST /config`

Wizard konfigurasi sederhana untuk memilih profil strategi dan pasangan ticker.

## Monitoring Filesystem Artifacts

### `vecm_project/out_ms/monitoring/price_download.json`

Snapshot setiap eksekusi `ensure_price_data`, berisi daftar ticker yang
berhasil di-refresh, gagal, atau di-skip karena cache masih fresh.

```json
{
  "refreshed": ["ANTM.JK", "INCO.JK"],
  "failed": ["BBCA.JK"],
  "skipped": ["TLKM.JK"],
  "duration_s": 12.34,
  "timestamp": "2025-02-07T03:12:00"
}
```

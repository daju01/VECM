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

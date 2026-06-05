# data/raw/

Taruh file PDF UU di sini. Sesuaikan nama file dengan `documents:` di `config.yaml`.

| Domain | Nama file (default config) | Status pipeline |
|---|---|---|
| konsumen | `uu_perlindungan_konsumen.pdf` | Aktif |
| ite | `uu_ite.pdf` | Aktif |
| anak | `uu_perlindungan_anak.pdf` | Aktif |
| ketenagakerjaan | `uu_ketenagakerjaan.pdf` | Dikecualikan* |

*UU Ketenagakerjaan dikecualikan dari pipeline aktif sejak Fase 0 (penguncian baseline)
agar evaluasi dilakukan pada 3 UU dengan 74 query berlabel. Tambahkan kembali ke
`config.yaml` jika ingin memperluas korpus.

Jika nama file Anda berbeda, ubah bagian `documents:` di `config.yaml` — jangan rename PDF-nya.

# Laporan Rekapitulasi Eksperimen Lanjutan: Variasi R1 (Check Interval)
## Evaluasi Variasi $k \in [8, 985]$ pada Konfigurasi R1 ($\gamma = 0.1, \theta_{\text{uf}} = 0.1$) Model YOLOS-APA

| Run ID | $\gamma$ (Ovf) | $\theta_{\text{uf}}$ (Uf) | $k$ (CI) | Durasi (m) | Throughput (FPS) | Latensi (ms) | Peak VRAM | Final Loss | Min Loss | BBox L1 | GIoU | Eskalasi (O/U/NP) | Skip |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R1-ci8** | `0.1` | `0.1` | `8` | 606.7 | 47.7 | 335.3 | 4749.8 MB | 1.5395 | 1.4866 | 0.0296 | 0.5220 | 131 (8O/122U/1NP) | 5 |
| **R1-ci985** | `0.1` | `0.1` | `985` | 582.7 | 51.1 | 315.0 | 4734.2 MB | 1.7022 | 1.6006 | 0.0339 | 0.5621 | 132 (9O/115U/8NP) | 7 |

---

*Laporan diperbarui otomatis pada: 2026-11-14 09:21:54 UTC*

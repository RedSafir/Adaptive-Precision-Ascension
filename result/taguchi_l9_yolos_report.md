# Laporan Rekapitulasi Desain Eksperimen: Taguchi Orthogonal Array $L_9(3^3)$
## Evaluasi Hyperparameter Adaptive Precision Ascension (APA) pada Model YOLOS

Laporan ini mendokumentasikan hasil eksekusi 9 kombinasi ortogonal $L_9(3^3)$ secara berurutan:
- **Faktor A (Gamma / Overflow Threshold)**: $\gamma \in [0.1, 0.5, 0.9]$
- **Faktor B (Theta / Underflow Tolerance)**: $\theta_{\text{underflow}} \in [0.1, 0.5, 0.9]$
- **Faktor C (Check Interval / Evaluasi Soft-Check)**: $k \in [1, 8, 985]$

---

### 1. Tabel Hasil Matriks Ortogonal $L_9$

| Run ID | $\gamma$ (Ovf) | $\theta_{\text{uf}}$ (Uf) | $k$ (CI) | Durasi (m) | Throughput (FPS) | Latensi (ms) | Peak VRAM | Final Loss | BBox L1 | GIoU | Eskalasi |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R1** | `0.1` | `0.1` | `1` | 590.8 | 48.8 | 328.0 | 4649.1 MB | 1.3889 | 0.0243 | 0.4622 | 136 (20O/113U) |
| **R2** | `0.1` | `0.5` | `8` | 482.0 | 61.4 | 261.4 | 4088.1 MB | 1.7862 | 0.0367 | 0.5892 | 52 (11O/28U) |

---

*Laporan diperbarui otomatis pada: 2026-11-07 00:18:22 UTC*

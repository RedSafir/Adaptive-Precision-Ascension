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
| **R3** | `0.1` | `0.9` | `985` | 407.4 | 74.0 | 216.3 | 3456.3 MB | 1.5162 | 0.0286 | 0.5120 | 17 (13O/0U) |
| **R4** | `0.5` | `0.1` | `8` | 593.4 | 48.8 | 328.3 | 4653.1 MB | 1.7198 | 0.0362 | 0.5759 | 128 (0O/128U) |
| **R5** | `0.5` | `0.5` | `985` | 440.6 | 68.4 | 236.2 | 3978.3 MB | 1.6449 | 0.0296 | 0.5419 | 31 (0O/31U) |

---

*Laporan diperbarui otomatis pada: 2026-11-08 00:26:52 UTC*

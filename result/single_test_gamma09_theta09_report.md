# Laporan Validasi: Single Test YOLOS APA (Gamma 0.9 & Theta 0.9)
## Pengujian Validasi Awal Pipeline Vast.ai (Epochs: 50)

Laporan ini mendokumentasikan hasil pengujian awal untuk memverifikasi pipeline komputasi, akselerasi Hybrid CUDA Graph, dan logging metrik Frobenius & AMAX secara end-to-end sebelum mengeksekusi rangkaian lengkap Klaster 1.

---

### 1. Ringkasan Kinerja & Telemetri Model

| Parameter & Metrik | Nilai Hasil Uji | Keterangan |
| :--- | :---: | :--- |
| **Overflow Threshold ($\gamma$)** | `0.9` | Batas longgar ($T_{\text{max}} = 403.2$) |
| **Underflow Threshold ($\theta_{\text{underflow}}$)** | `0.9` | Toleransi underflow 90% |
| **Total Epochs Selesai** | `50` | Target pelatihan tercapai |
| **Total Durasi Eksekusi** | `390.30 menit` | Waktu komputasi total |
| **Throughput Rata-rata** | `67.9 FPS` | Kecepatan inferensi/training |
| **Latensi per Batch** | `235.87 ms` | Waktu proses per batch |
| **Peak VRAM Memory** | `3530.5 MB` | Konsumsi puncak memori GPU |
| **Final Total Loss** | `1.6193` | Loss akhir setelah pelatihan |
| **Final BBox L1 Loss** | `0.0324` | Error regresi bounding box |
| **Final GIoU Loss** | `0.5559` | Generalised IoU loss |
| **Final CE Loss** | `0.3456` | Cross-entropy klasifikasi |
| **Total Eskalasi Level** | `28` | Jumlah modul yang tereskalasi |
| - *Silent Underflow* | `0` | Dipicu rasio nilai subnormal |
| - *Hard Overflow* | `28` | Dipicu batas eksponensial magnitudo |

---

### 2. Validasi Integritas File & Logging
- [x] `yolos_training.jsonl` mencakup metrik `avg_loss`, `avg_fps`, `peak_vram_mb`, dan `max_amax`.
- [x] `yolos_training_frobenius.jsonl` mencakup `weight_norm`, `grad_norm`, `update_ratio`, dan `amax` per-layer.
- [x] `yolos_training_apa_events.jsonl` mencatat alasan eskalasi kuantisasi (OVERFLOW vs SILENT_UNDERFLOW).

*Laporan di-generate secara otomatis pada: 2026-11-06 01:31:59 UTC*

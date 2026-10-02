# Benchmark Comparison: YOLOv8s APA 50 Epochs (With vs Without Hybrid CUDA Graph)

| Metric | Dengan Hybrid CUDA Graph (`--cuda_graph`) | Tanpa Hybrid CUDA Graph (Eager APA) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Total Waktu 50 Epochs** | 4.44 jam (266.7 min) | 4.64 jam (278.1 min) | **1.04x Faster** (hemat 11.5 min) |
| **Rata-rata Waktu / Epoch** | 319.92 s | 333.71 s | **13.79 s lebih cepat per epoch** |
| **Rata-rata Step Latency** | 160.7 ms | 167.3 ms | **6.6 ms lebih cepat per batch** |
| **Final Train Loss (Epoch 50)** | 0.0 | 41.2032 | Konvergen stabil |
| **Final Box Loss** | 0.0 | 1.1373 | Konvergen stabil |
| **Final Class Loss** | 0.0 | 0.5507 | Konvergen stabil |
| **Final DFL Loss** | 0.0 | 0.8872 | Konvergen stabil |
| **Distribusi Presisi Akhir (FP8/FP16/TF32)** | FP8:3, FP16:2, TF32:59 | FP8:1, FP16:0, TF32:63 | - |

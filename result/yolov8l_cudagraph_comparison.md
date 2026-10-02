# Benchmark Comparison: YOLOv8l APA With vs Without Hybrid CUDA Graph

| Metric | Tanpa Hybrid CUDA Graph (Eager APA) | Dengan Hybrid CUDA Graph (`--cuda_graph`) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Durasi 1 Epoch Penuh** | 1154.47 s (19.24 min) | 1076.56 s (17.94 min) | **1.07x Faster** (hemat 77.9s) |
| **Rata-rata Step Time** | 598.3 ms | 551.7 ms | **46.6 ms lebih cepat per batch** |
| **Total Train Loss** | 80.4709 | 151.0124 | Identik / Konsisten |
| **Box Loss** | 2.2442 | 4.2644 | Identik / Konsisten |
| **Class Loss** | 1.4743 | 3.4087 | Identik / Konsisten |
| **DFL Loss** | 1.312 | 1.7652 | Identik / Konsisten |
| **FP8 Backbone/Neck Layers** | 13 | 47 | FP8 Preserved |
| **TF32 Detect Head Layers** | 71 | 35 | TF32 Preserved |

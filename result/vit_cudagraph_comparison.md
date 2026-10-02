# Benchmark Comparison: Vision Transformer (ViT CIFAR-10)
## Layer Full FP8 dengan CUDA Graph vs Hybrid CUDA Graph

**Spesifikasi Hardware & Model**:
- **Device**: `NVIDIA GeForce RTX 5060 Ti`
- **Arsitektur Model**: Vision Transformer (ViT: dim=256, depth=6, heads=4)
- **Dataset**: CIFAR-10 (50.000 train samples, 10.000 test samples, batch size 128)
- **Total Durasi Training**: 5 Epochs per skema

| Metrik / Parameter | Full FP8 + CUDA Graph | Hybrid FP8 + CUDA Graph | APA Adaptive + Hybrid Graph |
| :--- | :---: | :---: | :---: |
| **Arsitektur Layer** | **100% Full FP8** (Patch Embed, Blocks, Head) | **Hybrid Precision** (Boundary FP32, Body FP8) | **Adaptive Precision** (Dynamic FP8/FP16/TF32) |
| **Rata-rata Waktu / Epoch** | **5.83 s** | **5.79 s** | **5.89 s** |
| **Total Waktu (5 Epochs)** | 33.6 s | 33.6 s | 34.7 s |
| **Throughput (Images/s)** | **8565.5 img/s** | **8615.8 img/s** | **8469.6 img/s** |
| **Final Train Loss (Epoch 5)** | 2.1238 | 2.0727 | 1.3111 |
| **Final Test Loss (Epoch 5)** | 2.1146 | 2.0566 | 1.2579 |
| **Final Test Top-1 Accuracy** | **22.72%** | **21.19%** | **54.52%** |
| **Distribusi Presisi Layer** | All 26 FP8 (100%) | 24 FP8 (92.3%), 2 FP32 (7.7%) | FP8: 14, FP16: 0, TF32: 10 |

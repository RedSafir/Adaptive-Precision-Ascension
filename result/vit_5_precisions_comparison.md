# Benchmark Comparison: Vision Transformer (ViT)
## 5 Precision Modes: FP32 vs TF32 vs FP16 vs FP8 (Eager) vs FP8 + CUDA Graph

**Hardware & Dataset Environment**:
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- Model: `Vision Transformer (ViT-Small)`: dim=256, depth=6, heads=4 (~3.2M params, >98% `nn.Linear`)
- Dataset: `CIFAR-10` (50,000 images, 390 batches/epoch @ batch_size=128)
- Epochs: `3` per mode

| Precision Mode | Waktu / Epoch | Latensi per Step | Throughput (FPS) | Peak VRAM | Final Test Acc | Speedup vs FP32 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Standard IEEE-754)** | **9.57 s** | 24.54 ms | 5216.3 fps | 0 MB | 0.00% | **1.00x** |
| **TF32 (TensorFloat-32)** | **7.84 s** | 20.09 ms | 6370.1 fps | 0 MB | 0.00% | **1.22x** |
| **FP16 (Half Precision AMP)** | **4.02 s** | 10.31 ms | 12417.9 fps | 0 MB | 0.00% | **2.38x** |
| **FP8 APA (Eager)** | **6.83 s** | 17.50 ms | 7312.5 fps | 0 MB | 0.00% | **1.40x** |
| **FP8 APA + CUDA Graph** | **5.80 s** | 14.88 ms | 8602.0 fps | 0 MB | 0.00% | **1.65x** |

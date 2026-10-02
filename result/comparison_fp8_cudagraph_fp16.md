# Benchmark Comparison: YOLOv8s (FP8 vs FP8 + CUDA Graph vs FP16)

**Environment**:
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- Model: `YOLOv8s.pt` (~11.2M parameters)
- Dataset: `merged_yolo_person_ball` (Class 0: Person)
- Batch Size: `16` | Benchmark Steps: `200` batches

| Metric / Parameter | FP8 (Eager) | FP8 + Hybrid CUDA Graph | FP16 (Eager) |
| :--- | :---: | :---: | :---: |
| **Rata-rata Step Latency** | **151.9 ms** | **160.8 ms** | **100.9 ms** |
| **Throughput (Images / Detik)** | **105.4 img/s** | **99.5 img/s** | **158.6 img/s** |
| **Speedup vs FP8 Baseline** | 1.00x (Ref) | **0.94x Faster** | **1.50x** |
| **Durasi 200 Steps** | 34.6 s | 36.9 s | 24.5 s |
| **Peak VRAM Allocated** | 2827.66 MB | 2291.22 MB | 3371.82 MB |
| **Peak VRAM Reserved** | 3026.0 MB | 3842.0 MB | 3846.0 MB |
| **Final Train Loss (200 steps)** | 68.7102 | 157.9227 | 67.0738 |
| **Box Loss** | 1.9187 | 4.327 | 1.8727 |
| **Class Loss** | 1.2252 | 3.5603 | 1.1942 |
| **DFL Loss** | 1.1505 | 1.9829 | 1.1252 |
| **Distribusi Presisi Layer** | FP8: 64, TF32: 0 | FP8: 64, TF32: 0 | FP16: 64, TF32: 0 |

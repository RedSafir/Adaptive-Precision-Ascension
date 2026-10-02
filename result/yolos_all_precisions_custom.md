# Benchmark Comparison: YOLOS (Pure Transformer Object Detection)
## 5 Precision Modes: FP32 vs TF32 vs FP16 Custom (NO AMP) vs FP8 Custom vs FP8 + Graph

**Hardware & Dataset Environment**:
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- Model: `YOLOS-Small` (27.4M parameter, 0 Conv2D in backbone/neck, 78 `nn.Linear` layers)
- Dataset: `merged_yolo_person_ball` (31,528 images, 3,941 batches/epoch @ batch_size=8)
- Iterasi per mode: `50` batches

| Mode Presisi | Arsitektur / Engine | Latensi per Step | Throughput (FPS) | Peak VRAM | Estimasi 1 Epoch (3941 b) | Speedup vs FP32 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Standard IEEE-754)** | Single precision FP32 murni tanpa Tensor Cores | **238.20 ms** | 33.6 fps | **3193.8 MB** | **15m 38s** | **1.00x** |
| **TF32 (TensorFloat-32)** | FP32 presisi dengan hardware TF32 Tensor Cores | **204.10 ms** | 39.2 fps | **3258.0 MB** | **13m 24s** | **1.17x** |
| **FP16 Custom APALinear (NO AMP)** | FP16 Half Precision via custom APALinear (Level 1, tanpa AMP) | **92.07 ms** | 86.9 fps | **2343.8 MB** | **6m 02s** | **2.59x** |
| **FP8 Custom APALinear (Eager)** | FP8 Fixed Level 0 via custom APALinear (Eager) | **121.06 ms** | 66.1 fps | **2375.3 MB** | **7m 57s** | **1.97x** |
| **FP8 Custom APALinear + CUDA Graph** | FP8 Fixed Level 0 via APALinear dengan CUDA Graph | **121.09 ms** | 66.1 fps | **2582.2 MB** | **7m 57s** | **1.97x** |

# Benchmark Comparison: DINO-DETR / RT-DETR (FP16 vs FP8 APA)

**Hardware & Dataset Environment**:
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- Model: `RT-DETR-L` (DINO-DETR Architecture: ~33.0M parameter, 74 Transformer Linear layers)
- Dataset: `merged_yolo_person_ball` (Class 0: Person)
- Batch Size: `8` | Iterasi Benchmark: `50` batches

| Parameter / Metrik | DINO-DETR (FP16 Baseline) | DINO-DETR (FP8 APA) | Perubahan / Efisiensi |
| :--- | :---: | :---: | :---: |
| **Rata-rata Step Latency** | **416.8 ms** | **420.1 ms** | **0.99x Speedup** |
| **Throughput (Images / Detik)** | **19.2 img/s** | **19.0 img/s** | **+-0.1 img/s** |
| **Peak VRAM Allocated** | **9328.92 MB** | **9108.83 MB** | **Hemat 220.1 MB (-2.4%)** |
| **Peak VRAM Reserved** | 9816.0 MB | 9580.0 MB | Penghematan memori GPU |
| **Durasi 50 Steps** | 24.2 s | 24.7 s | - |
| **Train Loss (50 steps)** | 36.6339 | 37.552 | Konvergensi numerik stabil |
| **Classification Loss** | 0.1314 | 0.1505 | Terjaga |
| **GIoU Box Loss** | 2.3522 | 2.4118 | Terjaga |
| **L1 Box Loss** | 0.9426 | 0.9933 | Terjaga |
| **Distribusi Presisi Layer** | All FP16 (74 layers) | FP8: 69, TF32: 0 | Native FP8 Tensor Cores |

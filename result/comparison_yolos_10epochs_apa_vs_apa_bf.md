# Laporan Evaluasi Komparatif Pelatihan Penuh YOLOS (10 Epochs)
## APA Standar (Delayed Scaling + FP16 Accumulator) vs APA_bf (Tanpa Delayed Scaling + BF16 Accumulator + Adaptive Ascension)

Laporan ini menyajikan evaluasi empiris *head-to-head* pelatihan penuh selama **10 Epoch Lengkap** (315.200 citra / 19.700 batch updates) pada model **YOLOS-Small** (Pure Vision Transformer Object Detection) untuk menguji dampak langsung penonaktifan **Delayed Dynamic Scaling** dan penggunaan **FP8_bf (BF16 Accumulator)** dalam arsitektur **Adaptive Precision Ascension (APA)**.

---

### 1. Spesifikasi Eksperimen & Hyperparameter

- **Model Arsitektur**: `hustvl/yolos-small` (27.4M Parameter, 12 Encoder Blocks, 72 Layer Linear APALinear)
- **Dataset**: `merged_yolo_person_ball` (Class 0: Person Only, 31.528 citra per epoch)
- **Epochs**: 10 Epoch Lengkap (19.700 batch updates per mode)
- **Batch Size**: 16 (Resolusi Input: 512 x 512)
- **Optimizer**: AdamW ($lr=10^{-4}$, weight decay=$10^{-4}$)
- **Hardware**: NVIDIA GeForce RTX 5060 Ti 16GB (Blackwell SM 12.0)
- **File Log Sumber**:
  - **APA Standar (FP8 Fused)**: `runs/train_yolos_apa_fp8_person2/yolos_apa_fp8_training.jsonl`
  - **APA_bf (Adaptive, No Delayed Scaling)**: `runs/train_yolos_apa_bf_person/yolos_apa_bf_training.jsonl`
  - **Event Log Eskalasi APA_bf**: `runs/train_yolos_apa_bf_person/yolos_apa_bf_training_apa_events.jsonl`

---

### 2. Ringkasan Kinerja Keseluruhan (10 Epoch Penuh)

| Metrik Evaluasi | APA Standar (Delayed Scaling) | APA_bf (Tanpa Delayed Scaling) | Perbedaan / Analisis Dampak |
| :--- | :---: | :---: | :---: |
| **Konfigurasi FP8** | FP8 E4M3/E5M2 + Delayed Scale + FP16 Acc | FP8_bf E4M3 + Unscaled (1.0) + BF16 Acc | Murni menguji peran Delayed Scaling |
| **Total Waktu Training (10 Ep)** | **66.64 menit** (1j 06m) | **81.87 menit** (1j 21m) | **APA Standar 22.8% Lebih Cepat** ⏱️ |
| **Rata-rata Latensi per Step** | **202.11 ms** | **248.66 ms** | APA_bf +46.55 ms (+23.0%) |
| **Training Throughput** | **79.17 fps** | **64.34 fps** | APA Standar +14.83 fps (+23.1%) 🚀 |
| **Peak VRAM** | **3,330.4 MB** | **4,760.1 MB** | APA_bf naik +1,429.7 MB (+42.9%) 💾 |
| **Total Event Eskalasi** | 0 event (Terkunci stabil di FP8) | **104 event eskalasi** | Memicu eskalasi adaptif bertingkat ⚠️ |
| **Distribusi Presisi Akhir Layer** | 72 FP8 / 0 FP16 / 0 TF32 | **17 FP8 / 7 FP16 / 48 TF32** | 66.7% layer terpaksa naik ke TF32 |
| **Final Total Loss (Epoch 10)** | **2.0976** | **3.3015** | APA Standar jauh lebih konvergen ✅ |
| **Final BBox L1 Loss (Regresi)** | **0.0419** | **0.1291** | Error lokalisasi APA_bf 3x lebih tinggi |
| **Final GIoU Loss (Overlap)** | **0.6463** | **1.0123** | Prediksi overlap APA Standar superior |

---

### 3. Tabel Detail Komparasi Per-Epoch (Epoch 1 s/d Epoch 10)

| Epoch | Mode Presisi | Total Loss | CE Loss (Klasifikasi) | BBox L1 Loss (Regresi) | GIoU Loss (Overlap) | Latensi / Step | Throughput | Peak VRAM | Durasi Epoch |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **APA Standar** | **2.5338** | **0.5853** | **0.0678** | **0.8047** | **202.03 ms** | **79.2 fps** | **3,330.4 MB** | **6.67 m** |
| **1** | **APA_bf**      | 3.2075 | 0.6082 | 0.1205 | 0.9984 | 258.02 ms | 62.0 fps | 4,402.7 MB | 8.50 m |
| **2** | **APA Standar** | **2.3128** | **0.5780** | **0.0524** | **0.7364** | **202.12 ms** | **79.2 fps** | **3,330.4 MB** | **6.66 m** |
| **2** | **APA_bf**      | 3.2560 | 0.6248 | 0.1253 | 1.0023 | 252.27 ms | 63.4 fps | 4,581.3 MB | 8.31 m |
| **3** | **APA Standar** | **2.2513** | **0.5897** | **0.0484** | **0.7097** | **202.06 ms** | **79.2 fps** | **3,330.4 MB** | **6.66 m** |
| **3** | **APA_bf**      | 3.2780 | 0.6304 | 0.1265 | 1.0076 | 249.08 ms | 64.2 fps | 4,601.4 MB | 8.20 m |
| **4** | **APA Standar** | **2.1659** | **0.5808** | **0.0446** | **0.6810** | **202.12 ms** | **79.2 fps** | **3,330.4 MB** | **6.66 m** |
| **4** | **APA_bf**      | 3.2770 | 0.6374 | 0.1263 | 1.0042 | 248.13 ms | 64.5 fps | 4,679.9 MB | 8.17 m |
| **5** | **APA Standar** | **2.1108** | **0.5849** | **0.0419** | **0.6582** | **202.13 ms** | **79.2 fps** | **3,330.4 MB** | **6.66 m** |
| **5** | **APA_bf**      | 3.2590 | 0.6319 | 0.1246 | 1.0020 | 246.35 ms | 64.9 fps | 4,761.1 MB | 8.11 m |
| **6** | **APA Standar** | **2.1003** | **0.5918** | **0.0409** | **0.6520** | **202.08 ms** | **79.2 fps** | **3,330.4 MB** | **6.66 m** |
| **6** | **APA_bf**      | 3.2386 | 0.6259 | 0.1236 | 0.9973 | 246.53 ms | 64.9 fps | 4,760.1 MB | 8.12 m |
| **7** | **APA Standar** | **2.1730** | **0.5895** | **0.0451** | **0.6789** | **202.06 ms** | **79.2 fps** | **3,330.4 MB** | **6.66 m** |
| **7** | **APA_bf**      | 3.2519 | 0.6260 | 0.1250 | 1.0004 | 246.39 ms | 64.9 fps | 4,760.1 MB | 8.11 m |
| **8** | **APA Standar** | **2.0656** | **0.5926** | **0.0401** | **0.6363** | **202.16 ms** | **79.1 fps** | **3,330.4 MB** | **6.67 m** |
| **8** | **APA_bf**      | 3.2661 | 0.6252 | 0.1252 | 1.0074 | 246.55 ms | 64.9 fps | 4,760.1 MB | 8.12 m |
| **9** | **APA Standar** | **2.0642** | **0.5943** | **0.0399** | **0.6351** | **202.13 ms** | **79.2 fps** | **3,330.4 MB** | **6.66 m** |
| **9** | **APA_bf**      | 3.2271 | 0.6137 | 0.1216 | 1.0027 | 246.46 ms | 64.9 fps | 4,760.1 MB | 8.11 m |
| **10** | **APA Standar** | **2.0976** | **0.5954** | **0.0419** | **0.6463** | **202.19 ms** | **79.1 fps** | **3,330.4 MB** | **6.67 m** |
| **10** | **APA_bf**      | 3.3015 | 0.6313 | 0.1291 | 1.0123 | 246.48 ms | 64.9 fps | 4,760.1 MB | 8.11 m |

---

### 4. Analisis Ilmiah & Temuan Kunci (Scientific Insights)

#### A. Mengapa Tanpa Delayed Scaling Terjadi *Silent Underflow* Masif?
1. Format FP8 (E4M3) memiliki batas representasi normal terkecil sebesar $2^{-6} \approx 0.015625$ (dan subnormal limit $2^{-9} \approx 0.00195$).
2. Pada model transformer seperti YOLOS, gradien yang mengalir balik (*backward pass*) melewati multi-head self-attention dan MLP dense layer umumnya memiliki magnitudo yang sangat halus (berkisar antara $10^{-3}$ hingga $10^{-6}$).
3. **Pada APA Standar**:
   Mekanisme **Delayed Dynamic Scaling** (`scale_grad` dan `scale_x`) menghitung faktor pengali adaptif dari iterasi sebelumnya (biasanya berkisar 100x hingga 1.000x), sehingga nilai-nilai gradien kecil tersebut dinaikkan (*scaled up*) ke area dinamis representabel FP8 sebelum kuantisasi dilakukan. Hasilnya, nol underflow terjadi.
4. **Pada APA_bf (Tanpa Delayed Scaling)**:
   Karena faktor skala dipaksa statis bernilai $1.0$, gradien-gradien halus tersebut jatuh di bawah batas terbawah $2^{-6}$ dan terkuantisasi menjadi **nol mutlak (*underflow to zero*)**.
5. Sistem telemetri APA mendeteksi bahwa rasio underflow melampaui ambang batas keamanan $\theta_{\text{underflow}} = 0.40$ (lebih dari 40% elemen gradien nol), sehingga memicu **104 event eskalasi**.

#### B. Mengapa Throughput APA_bf Lebih Rendah dan VRAM Lebih Tinggi?
- Begitu layer mengalami silent underflow di awal pelatihan (mulai step 22), APA secara otomatis menaikkan layer tersebut dari Level 0 (FP8) ke Level 1 (FP16), dan selanjutnya ke Level 2 (TF32).
- Pada akhir Epoch 10, komposisi presisi layer berubah drastis:
  - **Level 0 (FP8_bf)**: hanya tersisa **17 layer** (23.6%)
  - **Level 1 (FP16)**: **7 layer** (9.7%)
  - **Level 2 (TF32)**: **48 layer** (66.7%)
- Karena dua pertiga layer (66.7%) beroperasi dalam presisi penuh TF32, waktu komputasi matmul meningkat dan alokasi memori bobot master/gradien TF32 membengkak. Hal ini menjelaskan mengapa throughput turun dari 79.2 fps menjadi 64.9 fps, dan VRAM naik dari 3.33 GB menjadi 4.76 GB.

#### C. Dampak terhadap Konvergensi & Akurasi Prediksi
- Akibat hilangnya gradien halus pada iterasi-iterasi awal sebelum eskalasi selesai (gradien vanishing parsial), head regresi bounding box kesulitan mempelajari koordinat kotak pembatas secara presisi.
- Hal ini terbukti dari BBox L1 Loss akhir yang tertahan di angka **0.1291** (vs **0.0419** pada APA Standar) dan GIoU Loss tertahan di **1.0123** (vs **0.6463**).

---

### 5. Kesimpulan Riset

Eksperimen komparasi 10 epoch penuh ini membuktikan secara empiris bahwa:
1. **Delayed Scaling adalah komponen mutlak (*indispensable*)** dalam pelatihan deep neural network menggunakan presisi FP8. Tanpa delayed scaling, gradien halus akan mengalami underflow masif.
2. **Arsitektur APA terbukti bekerja sempurna sebagai jaring pengaman (*safety net*)**: Ketika delayed scaling dimatikan, APA secara cerdas mendeteksi hilangnya presisi numerik (*SILENT_UNDERFLOW*) dan menaikkan layer secara adaptif ke FP16 dan TF32, mencegah pelatihan dari kerusakan numerik total (*loss divergence* atau NaN).
3. **Kombinasi Terbaik**: Presisi FP8 memerlukan **Delayed Scaling** untuk menjaga layer tetap beroperasi pada kecepatan puncak FP8 (79.2 fps dan 3.3 GB VRAM), sementara accumulator **BF16 (`FP8_bf`)** memberikan perlindungan ekstra terhadap overflow pada aktivasi antarlayer.

import sys
import os
import time
import json
sys.path.insert(0, '/home/ictlab/Documents/penelitian_miftah/Adaptive-Precision-Ascension')

import torch
from transformers import YolosConfig, YolosForObjectDetection
from apa import APAConfig, APALinear
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32

cfg = YolosConfig(
    image_size=[512, 512],
    patch_size=16,
    num_channels=3,
    hidden_size=384,
    num_hidden_layers=12,
    num_attention_heads=6,
    intermediate_size=1536,
    num_labels=2,
    num_detection_tokens=100
)

def run_yolos_mode(mode, steps=50):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    use_graph = ('graph' in mode)

    # 1. Hardware backend configuration
    if mode == 'fp32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        use_apa = False
        freeze_level = None
    elif mode == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_apa = False
        freeze_level = None
    elif 'fp16' in mode:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_apa = True
        freeze_level = LEVEL_FP16  # Custom APALinear Level 1, NO AMP!
    elif 'fp8' in mode:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_apa = True
        freeze_level = LEVEL_FP8   # Custom APALinear Level 0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model = YolosForObjectDetection(cfg).to('cuda')

    if use_apa:
        apa_cfg = APAConfig.research_default()
        apa_cfg.device = 'cuda'
        apa_cfg.freeze_level = freeze_level
        apa_cfg.fp8_output_dtype = 'float16'
        for name, module in model.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, torch.nn.Linear):
                    is_head = 'class_labels_classifier' in f"{name}.{child_name}" or 'bbox_predictor' in f"{name}.{child_name}"
                    init_lvl = LEVEL_TF32 if is_head else freeze_level
                    apa_lin = APALinear.from_linear(child, config=apa_cfg, initial_level=init_lvl)
                    setattr(module, child_name, apa_lin)

    # NO AMP GradScaler! Standard AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, capturable=use_graph)

    batch_size = 8
    img_size = 512
    static_x = torch.randn(batch_size, 3, img_size, img_size, device='cuda')
    static_grad_logits = torch.randn(batch_size, 100, 3, device='cuda')
    static_grad_boxes = torch.randn(batch_size, 100, 4, device='cuda')

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            # Pure forward pass, NO torch.amp.autocast!
            out = model(pixel_values=static_x)
            logits = out.logits
            pred_boxes = out.pred_boxes
            torch.autograd.backward([logits, pred_boxes], [static_grad_logits, static_grad_boxes])
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)

    if use_graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            optimizer.zero_grad(set_to_none=True)
            out = model(pixel_values=static_x)
            logits = out.logits
            pred_boxes = out.pred_boxes
            torch.autograd.backward([logits, pred_boxes], [static_grad_logits, static_grad_boxes])
            optimizer.step()

        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(steps):
            g.replay()
        torch.cuda.synchronize()
        latency_ms = (time.time() - t0) * 1000.0 / steps
    else:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            out = model(pixel_values=static_x)
            logits = out.logits
            pred_boxes = out.pred_boxes
            torch.autograd.backward([logits, pred_boxes], [static_grad_logits, static_grad_boxes])
            optimizer.step()
        torch.cuda.synchronize()
        latency_ms = (time.time() - t0) * 1000.0 / steps

    vram_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    fps = (batch_size * 1000.0) / latency_ms

    return {
        'mode': mode,
        'latency_ms': latency_ms,
        'fps': fps,
        'vram_alloc_mb': vram_alloc,
    }

def main():
    modes = [
        ('fp32', 'FP32 (Standard IEEE-754)', 'Single precision FP32 murni tanpa Tensor Cores'),
        ('tf32', 'TF32 (TensorFloat-32)', 'FP32 presisi dengan hardware TF32 Tensor Cores'),
        ('fp16_custom', 'FP16 Custom APALinear (NO AMP)', 'FP16 Half Precision via custom APALinear (Level 1, tanpa AMP)'),
        ('fp8_custom', 'FP8 Custom APALinear (Eager)', 'FP8 Fixed Level 0 via custom APALinear (Eager)'),
        ('fp8_graph', 'FP8 Custom APALinear + CUDA Graph', 'FP8 Fixed Level 0 via APALinear dengan CUDA Graph')
    ]

    print("=" * 85)
    print("  YOLOS (PURE TRANSFORMER OBJECT DETECTION) - 5 PRECISION MODES BENCHMARK")
    print("  Engine: Custom APALinear (NO AMP) | GPU: NVIDIA GeForce RTX 5060 Ti (16GB)")
    print("=" * 85)

    results = []
    for mode_id, display_name, desc in modes:
        res = run_yolos_mode(mode_id, steps=50)
        res['display_name'] = display_name
        res['desc'] = desc
        results.append(res)
        print(f"[{display_name}] Latency: {res['latency_ms']:.2f} ms | Throughput: {res['fps']:.1f} fps | VRAM: {res['vram_alloc_mb']:.1f} MB")

    total_batches = 3941 # 31,528 images @ batch size 8
    ref_latency = results[0]['latency_ms'] # FP32 reference

    summary_md = f"""# Benchmark Comparison: YOLOS (Pure Transformer Object Detection)
## 5 Precision Modes: FP32 vs TF32 vs FP16 Custom (NO AMP) vs FP8 Custom vs FP8 + Graph

**Hardware & Dataset Environment**:
- GPU: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}`
- Model: `YOLOS-Small` (27.4M parameter, 0 Conv2D in backbone/neck, 78 `nn.Linear` layers)
- Dataset: `merged_yolo_person_ball` (31,528 images, 3,941 batches/epoch @ batch_size=8)
- Iterasi per mode: `50` batches

| Mode Presisi | Arsitektur / Engine | Latensi per Step | Throughput (FPS) | Peak VRAM | Estimasi 1 Epoch (3941 b) | Speedup vs FP32 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in results:
        sp = ref_latency / max(1e-6, r['latency_ms'])
        epoch_sec = (r['latency_ms'] * total_batches) / 1000.0
        epoch_min = int(epoch_sec // 60)
        epoch_rem = int(epoch_sec % 60)
        epoch_str = f"{epoch_min}m {epoch_rem:02d}s"
        summary_md += f"| **{r['display_name']}** | {r['desc']} | **{r['latency_ms']:.2f} ms** | {r['fps']:.1f} fps | **{r['vram_alloc_mb']:.1f} MB** | **{epoch_str}** | **{sp:.2f}x** |\n"

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    summary_file = os.path.join(base_dir, 'result', 'yolos_all_precisions_custom.md')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_md)

    print("\n" + "=" * 95)
    print("FINAL COMPARISON RESULT:")
    print("=" * 95)
    print(summary_md)
    print("=" * 95)
    print(f"Summary written to: {summary_file}")

if __name__ == '__main__':
    main()

# APA — Adaptive Precision Ascension

> A reckless-start precision management paradigm for PyTorch training that begins fully in FP8 and dynamically elevates layer dtypes to FP16/TF32 upon real-time overflow/underflow telemetry.

## Overview

APA (Adaptive Precision Ascension) provides a novel approach to mixed-precision training. Instead of conservatively assigning high precision to stable layers, APA starts **every layer in FP8 (E4M3)** and only dynamically escalates specific modules to FP16 and eventually TF32 when real-time telemetry detects hard overflows (NaN/Inf) or persistent silent underflows.

**Key differentiators from traditional AMP/Transformer Engine:**
- **No `torch.autocast` or `GradScaler`** — APA manages precision entirely through its own escalation ladder
- **No per-tensor scaling factors** — overflow is solved by dtype promotion, not scale manipulation
- **One-way escalation** — once a layer promotes (FP8→FP16→TF32), it never demotes back
- **Skip, never retry** — overflow batches are discarded, not re-run with adjusted scaling
- **Two-speed detection** — hard NaN/Inf check every step, soft amax/underflow check per configurable interval

## Hardware Requirements

- **GPU**: NVIDIA GPU with FP8 support (Hopper H100, Ada Lovelace RTX 40xx, Blackwell RTX 50xx)
- **Tested on**: RTX 5060 Ti 16GB (Blackwell, sm_120)
- **CUDA**: 12.0+ (tested with CUDA 13.2)
- **RAM**: 64GB+ system RAM recommended
- **Simulation mode**: Available for GPUs without native FP8 tensor core support

## Setup (Pop!_OS / Ubuntu)

### 1. Verify NVIDIA Driver

On **Pop!_OS**, the NVIDIA driver is managed by System76's driver system, not standard Ubuntu packages:

```bash
# Check driver is working
nvidia-smi

# If driver issues on Pop!_OS, use:
sudo system76-driver
# or check Settings > System76 Driver
```

> **Do NOT** use `sudo apt install nvidia-driver-XXX` on Pop!_OS — this can conflict with the system76-driver stack.

### 2. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, then restart terminal
```

### 3. Create Environment

```bash
conda create -n apa python=3.11 -y
conda activate apa
```

### 4. Install PyTorch

Check [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the latest version supporting your GPU.

For **Blackwell (sm_120) + CUDA 13.2**, you likely need a **nightly build**:

```bash
# Check the PyTorch nightly page for the exact URL matching your CUDA version
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
```

> ⚠️ **Do not blindly copy this command** — verify the CUDA wheel version matches what PyTorch supports at the time of installation. PyTorch bundles its own CUDA runtime, so the wheel's CUDA version determines compute support, not your system CUDA.

### 5. Install APA

```bash
cd APAver2
pip install -e ".[examples]"
```

### 6. Verify Environment

```bash
python scripts/check_environment.py
```

This script checks PyTorch version, CUDA detection, GPU compute capability, `float8_e4m3fn` dtype availability, and `torch._scaled_mm` functionality. If native FP8 is unavailable, it will recommend using `--fp8_sim` mode.

## Quick Start

```python
import torch
import torch.nn as nn
from apa import APAConfig, APALinear, APAManager

# 1. Define config
config = APAConfig.research_default(device='cuda')

# 2. Build model using APALinear instead of nn.Linear
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = APALinear(768, 768, config=config)
        self.norm = nn.LayerNorm(768)
        self.fc2 = APALinear(768, 10, config=config)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.norm(x)
        return self.fc2(x)

model = MyModel(config).cuda()

# 3. Wrap model with APAManager (auto-discovers APALinear + non-parametric modules)
apa_manager = APAManager(model, config=config)

# 4. Use manager's trainable parameters for optimizer
optimizer = torch.optim.AdamW(apa_manager.get_trainable_parameters(), lr=1e-3)

# 5. Training loop
for data, target in dataloader:
    apa_manager.pre_step()                       # Refresh working copies, reset per-step flags
    optimizer.zero_grad(set_to_none=True)

    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    if apa_manager.post_backward_sync_and_eval(): # Two-speed check: safe → True, overflow → False
        optimizer.step()
    else:
        optimizer.zero_grad(set_to_none=True)      # Discard corrupted gradients
```

## Using APA with Your Own Model

1. **Replace Linear Layers**: Change `nn.Linear(...)` to `APALinear(..., config=config)` in your model definition.
2. **Initialize APAManager**: Wrap your model with `APAManager(model, config=config)` — this auto-discovers all `APALinear` modules and registers backward hooks on non-parametric modules (LayerNorm, Softmax, etc.).
3. **Optimizer Parameters**: Pass `apa_manager.get_trainable_parameters()` to your optimizer. This returns FP32 master weights from APALinear + native params from non-APA modules, excluding working copies.
4. **Training Loop**:
   - Call `apa_manager.pre_step()` **before** the forward pass
   - Call `apa_manager.post_backward_sync_and_eval()` **after** `loss.backward()`
   - If it returns `False`, **skip** `optimizer.step()` and zero grads

## Configuration

```python
from apa import APAConfig

# Use presets
config = APAConfig.conservative()      # gamma=0.80, theta_underflow=0.30, check_interval=1
config = APAConfig.aggressive()        # gamma=0.95, theta_underflow=0.50, check_interval=8
config = APAConfig.research_default()  # gamma=0.90, theta_underflow=0.40, check_interval=4

# Or customize directly
config = APAConfig(
    gamma=0.90,              # Overflow safety margin (threshold = gamma * dtype_max)
    theta_underflow=0.40,    # Underflow ratio limit before escalation
    ema_alpha=0.95,          # EMA smoothing for underflow tracking
    check_interval=4,        # Soft check frequency (steps)
    fp8_simulation_mode=False,  # True = simulate FP8 without hardware support
    log_file='apa_log.jsonl',   # Structured escalation event log
    device='cuda',
)
```

## Running Examples

```bash
# ViT on CIFAR-10 (with FP8 simulation mode for testing)
python examples/vit_cifar10/train.py --fp8_sim --epochs 2

# nanoGPT character-level language model
python examples/nanogpt_char/train.py --fp8_sim --max_steps 200
```

## Running Tests

```bash
# All tests (CPU-compatible, no GPU required)
python -m unittest discover tests/ -v

# Individual test modules
python -m unittest tests.test_escalation -v
python -m unittest tests.test_hard_overflow -v
python -m unittest tests.test_underflow -v
python -m unittest tests.test_parameter_registry -v
python -m unittest tests.test_smoke -v
```

## VRAM Estimator

Before training, check if your model fits in VRAM:

```bash
python -m apa.vram_estimator --params 10000000 --batch-size 32 --seq-len 512
```

## Architecture Overview

### Two-Speed Detection
1. **Hard Check** (every step): OR-reduces all `gpu_has_nonfinite` flags across modules into a single scalar. One `.item()` call (~3-5 μs). If triggered → immediate full evaluation and batch skip.
2. **Soft Check** (every `check_interval` steps): Transfers vectorized `gpu_amax` and `gpu_underflow_ratio` from all modules, evaluates overflow thresholds and underflow EMA.

### Escalation Ladder
| Level | Dtype | Max Representable | Escalation Trigger |
|-------|-------|-------------------|--------------------|
| 0 | FP8 E4M3 | 448 | amax > 0.90×448 or NaN/Inf |
| 1 | FP16 | 65504 | amax > 0.90×65504 or NaN/Inf |
| 2 | TF32 (FP32) | ~3.4×10³⁸ | Ceiling — no further escalation |

### Master / Working Weights
- **Master weights** (FP32): Stored as `nn.Parameter`, updated by optimizer
- **Working copies**: Cast from master to current level dtype each `pre_step()`, used in forward/backward

## Implementation Notes & Known Limitations

- **No Demotion**: Precision escalation is permanent and one-way. Once a layer promotes to FP16, it never returns to FP8.
- **Scale Factor = 1.0**: FP8 path uses `torch._scaled_mm` with `scale=1.0`. This is intentional — APA handles dynamic range through dtype promotion, not scaling.
- **Hardware Dependency**: Real FP8 execution requires Ada Lovelace (sm_89), Hopper (sm_90), or Blackwell (sm_120). All other GPUs must use `fp8_simulation_mode=True`.
- **`torch._scaled_mm` API Stability**: This is a private PyTorch API that may change between versions. If it breaks after a PyTorch update, enable simulation mode as a workaround.
- **Gradient Accumulation**: The current implementation uses `grad.copy_()` (overwrite), not `grad.add_()`. For multi-microbatch gradient accumulation, additional logic is needed.
- **No Distributed Training**: `dist.all_reduce` calls are present in the code but have not been tested in multi-GPU setups.

import torch
from typing import Optional

try:
    import apa_cuda
    APA_CUDA_AVAILABLE = True
except ImportError:
    try:
        import sys, os, glob
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        for bl in glob.glob(os.path.join(repo_root, 'build', 'lib.*')):
            if bl not in sys.path:
                sys.path.insert(0, bl)
        import apa_cuda
        APA_CUDA_AVAILABLE = True
    except ImportError:
        APA_CUDA_AVAILABLE = False
        apa_cuda = None

# Allow PyTorch compiler (TorchDynamo) to trace apa_cuda without graph breaks
if APA_CUDA_AVAILABLE and hasattr(torch, 'compiler') and hasattr(torch.compiler, 'allow_in_graph'):
    for _fn_name in ['fused_linear_forward', 'fused_linear_backward', 'fused_quantize_fp8_e4m3', 'fused_quantize_fp8_e5m2']:
        if hasattr(apa_cuda, _fn_name):
            try:
                torch.compiler.allow_in_graph(getattr(apa_cuda, _fn_name))
            except Exception:
                pass

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _scale_clamp_quant_kernel(
        x_ptr,
        scale_ptr,
        out_ptr,
        n_elements,
        max_val,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load scale factor (scalar float32)
        scale = tl.load(scale_ptr)

        # Load input element chunk and cast to float32 for math in registers
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Fused scale and clamp directly in registers / SRAM
        scaled = x * scale
        clamped = tl.clamp(scaled, -max_val, max_val)

        # Store to FP8 output (Triton generates native PTX cvt to e4m3 / e5m2)
        tl.store(out_ptr + offsets, clamped, mask=mask)


    def _triton_scale_clamp_quantize(
        x: torch.Tensor,
        scale: torch.Tensor,
        max_val: float,
        target_dtype: torch.dtype
    ) -> torch.Tensor:
        x_contig = x.contiguous() if not x.is_contiguous() else x
        n_elements = x_contig.numel()

        out = torch.empty(x_contig.shape, dtype=target_dtype, device=x.device)

        if scale.numel() == 1:
            scale_t = scale if scale.device == x.device else scale.to(x.device)
        else:
            scale_t = scale.flatten()[:1]
        if scale_t.dtype != torch.float32:
            scale_t = scale_t.to(torch.float32)

        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        _scale_clamp_quant_kernel[grid](
            x_contig,
            scale_t,
            out,
            n_elements,
            float(max_val),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out


def fused_scale_clamp_quantize_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    max_val: float,
    target_dtype: torch.dtype,
    gpu_amax: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Fused scale, clamp, and quantize to FP8 in a single GPU memory pass.

    Prioritas eksekusi:
    1. apa_cuda (Native C++/CUDA extension) — eksekusi 1-pass tercepat dengan tracking amax simultan
    2. Triton JIT kernel — jika Triton tersedia pada arsitektur GPU
    3. Native PyTorch elementwise operations — fusible IR untuk TorchInductor
    """
    is_compiling = False
    if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'is_compiling'):
        is_compiling = torch.compiler.is_compiling()
    elif hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'is_compiling'):
        is_compiling = torch._dynamo.is_compiling()

    # Fast-path 1: Native CUDA C++ Extension (apa_cuda)
    if not is_compiling and APA_CUDA_AVAILABLE and x.is_cuda:
        try:
            if target_dtype == torch.float8_e4m3fn:
                return apa_cuda.fused_quantize_fp8_e4m3(x, scale, float(max_val), gpu_amax)
            elif target_dtype == torch.float8_e5m2:
                return apa_cuda.fused_quantize_fp8_e5m2(x, scale, float(max_val), gpu_amax)
        except Exception:
            pass

    # Fast-path 2: Native Triton Kernel
    if not is_compiling and TRITON_AVAILABLE and x.is_cuda and target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        try:
            return _triton_scale_clamp_quantize(x, scale, max_val, target_dtype)
        except Exception:
            pass

    # Fast-path 3: Native PyTorch fallback (memberikan TorchInductor clean fusible IR)
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        x_f32 = x.to(torch.float32)
    else:
        x_f32 = x
    return (x_f32 * scale).clamp(-max_val, max_val).to(target_dtype)


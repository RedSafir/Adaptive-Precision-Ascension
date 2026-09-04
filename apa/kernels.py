import torch
from typing import Optional

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
    target_dtype: torch.dtype
) -> torch.Tensor:
    """Fused scale, clamp, and quantize to FP8 in a single GPU memory pass.

    Uses a native Triton kernel for microsecond execution when available on CUDA,
    and falls back to standard PyTorch operations gracefully if Triton is
    unavailable.
    """
    if TRITON_AVAILABLE and x.is_cuda and target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        try:
            return _triton_scale_clamp_quantize(x, scale, max_val, target_dtype)
        except Exception:
            pass

    # Native PyTorch fallback
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        x_f32 = x.to(torch.float32)
    else:
        x_f32 = x
    return (x_f32 * scale).clamp(-max_val, max_val).to(target_dtype)

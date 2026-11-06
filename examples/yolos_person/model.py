import sys
import os
import torch
import torch.nn as nn
from typing import Optional
from transformers import YolosConfig, YolosForObjectDetection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from apa import APALinear, APAConfig
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32

def create_yolos_model(
    model_name: str = 'hustvl/yolos-small',
    num_labels: int = 2,
    img_size: int = 512,
    pretrained: bool = True
) -> YolosForObjectDetection:
    """Build or load a YOLOS model for object detection.

    Args:
        model_name: Pretrained checkpoint or config name (e.g. 'hustvl/yolos-small' or 'hustvl/yolos-tiny').
        num_labels: Number of target object classes (default: 2 for person and ball).
        img_size: Input image resolution (default: 512).
        pretrained: If True, load weights from Hugging Face cache/hub.
    """
    if pretrained:
        try:
            model = YolosForObjectDetection.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
            return model
        except Exception as e:
            print(f"[YOLOS] Failed to load pretrained {model_name}: {e}. Falling back to clean config.")

    # Clean config fallback
    hidden_size = 384 if 'small' in model_name or 'tiny' in model_name else 768
    intermediate_size = hidden_size * 4
    heads = 6 if hidden_size == 384 else 12
    cfg = YolosConfig(
        image_size=[img_size, img_size],
        patch_size=16,
        num_channels=3,
        hidden_size=hidden_size,
        num_hidden_layers=12,
        num_attention_heads=heads,
        intermediate_size=intermediate_size,
        num_labels=num_labels,
        num_detection_tokens=100
    )
    return YolosForObjectDetection(cfg)

def convert_yolos_to_apa(
    model: nn.Module,
    config: APAConfig = APAConfig(),
    preserve_critical_heads: bool = True
) -> nn.Module:
    """Convert Linear layers in YOLOS Transformer encoder to APALinear layers.

    Args:
        model: YolosForObjectDetection instance.
        config: APAConfig configuration for precision escalation and telemetry.
        preserve_critical_heads: If True, keep the final scoring and bbox prediction
            heads (class_labels_classifier and bbox_predictor) as standard float32 nn.Linear
            to protect the Hungarian matcher (torch.cdist) and bounding box regression
            from float16 limitations or loss spikes, while running all 72 Transformer
            backbone linear layers in native FP8 with fused CUDA kernels.

    Returns:
        The modified model with APALinear layers.
    """
    head_subnames = ('class_labels_classifier', 'bbox_predictor')

    converted_count = 0
    preserved_count = 0

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_child_name = f"{name}.{child_name}" if name else child_name
                is_head = preserve_critical_heads and any(h in full_child_name for h in head_subnames)

                if is_head:
                    preserved_count += 1
                    continue

                default_lvl = config.freeze_level if config.freeze_level is not None else LEVEL_FP8
                apa_lin = APALinear.from_linear(child, config=config, initial_level=default_lvl)
                setattr(module, child_name, apa_lin)
                converted_count += 1

    print(f"[APA Engine] Successfully converted {converted_count} linear layers to APALinear (FP8 Fused). Preserved {preserved_count} head layers in FP32.")
    return model

class _YolosViTBackbone(nn.Module):
    """Internal module extracting static last_hidden_state from YOLOS ViT model."""
    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x, return_dict=False)[0]

class _GraphedViTWrapper(nn.Module):
    """Wrapper that adapts a graphed ViT callable to HuggingFace BaseModelOutputWithPooling
    and automatically re-captures the CUDA Graph whenever any APA layer escalates precision."""
    def __init__(self, raw_vit: nn.Module, sample_input: torch.Tensor, warmup_iters: int = 3):
        super().__init__()
        # Store raw_vit without registering it twice in self._modules (graphed_backbone holds it)
        object.__setattr__(self, '_raw_vit', raw_vit)
        self._warmup_iters = max(3, warmup_iters)
        self._last_levels = []
        self.graphed_backbone = None
        self._capture(sample_input, is_initial=True)

    def _get_apa_levels(self):
        from apa.layers import APAModule
        return [m.level for m in self._raw_vit.modules() if isinstance(m, APAModule)]

    def _reset_vit_telemetry_and_grads(self):
        from apa.layers import APAModule
        with torch.no_grad():
            for m in self._raw_vit.modules():
                if isinstance(m, APAModule):
                    m.gpu_amax.zero_()
                    if hasattr(m, 'gpu_amax_x'):
                        m.gpu_amax_x.zero_()
                    if hasattr(m, 'gpu_amax_grad'):
                        m.gpu_amax_grad.zero_()
                    m.gpu_underflow_ratio.zero_()
                    m.gpu_has_nonfinite.zero_()
                    m.ema_underflow_ratio = 0.0
                else:
                    if hasattr(m, 'gpu_amax') and isinstance(m.gpu_amax, torch.Tensor):
                        m.gpu_amax.zero_()
                    if hasattr(m, 'gpu_has_nonfinite') and isinstance(m.gpu_has_nonfinite, torch.Tensor):
                        m.gpu_has_nonfinite.zero_()
        self._raw_vit.zero_grad(set_to_none=True)

    def _capture(self, sample_input: torch.Tensor, is_initial: bool = False):
        from apa.layers import APAModule
        torch.cuda.synchronize()
        if self.graphed_backbone is not None:
            del self.graphed_backbone
            self.graphed_backbone = None
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Ensure all working copies and telemetry flags are ready before capture
        with torch.no_grad():
            for m in self._raw_vit.modules():
                if isinstance(m, APAModule):
                    m.is_telemetry_step = (m.config.freeze_level is None)
                    if m.level != LEVEL_TF32:
                        m.refresh_working_copy()

        self._raw_vit.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        sample_buf = sample_input.detach().clone().contiguous()
        backbone = _YolosViTBackbone(self._raw_vit).to(sample_buf.device)
        backbone.train(self._raw_vit.training)

        self.graphed_backbone = torch.cuda.make_graphed_callables(
            backbone,
            sample_args=(sample_buf,),
            num_warmup_iters=self._warmup_iters
        )

        # Clean up any dummy gradients and amax spikes generated during make_graphed_callables warmup
        self._reset_vit_telemetry_and_grads()
        self._last_levels = self._get_apa_levels()
        torch.cuda.synchronize()
        if not is_initial:
            print(f"[CUDA Graph] Re-captured YOLOS ViT Backbone after APA precision escalation!", flush=True)

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        current_levels = self._get_apa_levels()
        if current_levels != self._last_levels:
            self._capture(pixel_values, is_initial=False)
        last_hidden_state = self.graphed_backbone(pixel_values)
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state)

def create_yolos_hybrid_cuda_graph(
    model: nn.Module,
    sample_input: torch.Tensor,
    warmup_iters: int = 3
) -> nn.Module:
    """Wrap YOLOS Transformer Backbone into a hardware-accelerated static CUDA Graph.

    Captures the static ViT Encoder (which represents ~95% of FP8 GEMM compute)
    into a CUDA Graph via `torch.cuda.make_graphed_callables`, while leaving
    the detection heads (classifier, bbox regressor) and Hungarian matcher
    in Eager PyTorch autograd. Automatically re-captures when any layer escalates.

    Args:
        model: YolosForObjectDetection instance.
        sample_input: Sample input tensor with fixed shape [B, 3, H, W] on CUDA.
        warmup_iters: Number of warmup iterations before capture (default: 3).

    Returns:
        The model with model.vit replaced by the graphed callable wrapper.
    """
    if not torch.cuda.is_available() or sample_input.device.type != 'cuda':
        raise RuntimeError("Hybrid CUDA Graph requires an active CUDA device.")

    if hasattr(torch.autograd.graph, 'set_override_stale_capture_stream'):
        torch.autograd.graph.set_override_stale_capture_stream(True)
    if hasattr(torch.autograd.graph, 'set_warn_on_accumulate_grad_stream_mismatch'):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    print(f"[CUDA Graph] Initializing YOLOS Hybrid CUDA Graph capture (warmup={warmup_iters})...")
    model.vit = _GraphedViTWrapper(model.vit, sample_input, warmup_iters=warmup_iters)
    print("[CUDA Graph] YOLOS ViT Backbone successfully compiled into CUDA Graph!")
    return model


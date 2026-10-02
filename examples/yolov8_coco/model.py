import sys
import os
import torch
import torch.nn as nn
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from apa import APAConv2d, APALinear, APAConfig
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
except ImportError:
    YOLO = None
    DetectionModel = None

def convert_yolov8_to_apa(
    model: nn.Module,
    config: APAConfig = APAConfig(),
    preserve_critical_layers: bool = True
) -> nn.Module:
    """Convert standard YOLOv8 convolutional layers to APAConv2d layers.

    Args:
        model: A YOLOv8 DetectionModel or nn.Module containing YOLOv8 layers.
        config: APAConfig configuration for precision escalation and telemetry.
        preserve_critical_layers: If True, keep the final Detect head in LEVEL_TF32
            to protect non-linear box coordinates (DFL / CIoU) and class classification
            from numerical underflow/divergence while running the Backbone & Neck in FP8.
            If False, all convolutions including Detect are converted to FP8.

    Returns:
        The modified model with APAConv2d layers.
    """
    detect_idx = len(model.model) - 1 if hasattr(model, 'model') else -1

    def _replace_convs(module: nn.Module, is_critical: bool):
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                default_lvl = config.freeze_level if config.freeze_level is not None else LEVEL_FP8
                init_lvl = LEVEL_TF32 if is_critical else default_lvl
                apa_conv = APAConv2d.from_conv2d(child, config=config, initial_level=init_lvl)
                setattr(module, child_name, apa_conv)
            elif isinstance(child, nn.Linear):
                default_lvl = config.freeze_level if config.freeze_level is not None else LEVEL_FP8
                init_lvl = LEVEL_TF32 if is_critical else default_lvl
                apa_lin = APALinear.from_linear(child, config=config, initial_level=init_lvl)
                setattr(module, child_name, apa_lin)
            else:
                _replace_convs(child, is_critical=is_critical)

    if hasattr(model, 'model') and isinstance(model.model, (nn.Sequential, nn.ModuleList)):
        for i, layer in enumerate(model.model):
            is_detect = (i == detect_idx)
            is_critical = preserve_critical_layers and is_detect
            _replace_convs(layer, is_critical=is_critical)
    else:
        _replace_convs(model, is_critical=False)

    return model

def create_yolov8_apa(
    model_name_or_cfg: str = 'yolov8n.yaml',
    config: Optional[APAConfig] = None,
    preserve_critical_layers: bool = True,
    device: str = 'cuda'
):
    """Load or build a YOLOv8 model and convert it to APA.

    Args:
        model_name_or_cfg: Path to yaml config (e.g. 'yolov8n.yaml') or checkpoint (e.g. 'yolov8n.pt').
        config: APAConfig configuration. If None, uses APAConfig.research_default().
        preserve_critical_layers: Whether to preserve Detect head in LEVEL_TF32.
        device: Device to place the model on ('cuda' or 'cpu').

    Returns:
        (yolo_wrapper, inner_model)
    """
    if YOLO is None:
        raise ImportError("ultralytics is required to use create_yolov8_apa. Run: pip install ultralytics")

    if config is None:
        config = APAConfig.research_default()
    config.device = device

    yolo = YOLO(model_name_or_cfg)
    model = yolo.model.to(device)
    convert_yolov8_to_apa(model, config=config, preserve_critical_layers=preserve_critical_layers)
    return yolo, model


class YOLOBackboneNeck(nn.Module):
    """Sub-module isolating Backbone and Neck layers for static CUDA Graph capture.

    In YOLOv8, layers 0 to 21 (all Conv, C2f, SPPF, Upsample, Concat layers before
    the Detect head) operate on a fixed input resolution [B, 3, H, W] and produce
    multi-scale feature pyramid tensors (P3, P4, P5) with 100% static shapes.

    Capturing this module with CUDA Graph eliminates Python dispatch latency and
    accelerates the ~85% compute portion of YOLOv8 while allowing the dynamic
    Detect head and Loss assignment to execute natively in Eager mode.
    """

    def __init__(self, layers: nn.ModuleList, save_indices: list, out_indices: Optional[list] = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.save_indices = set(save_indices)
        self.out_indices = out_indices or [15, 18, 21]

    def forward(self, x: torch.Tensor):
        y = []
        for layer in self.layers:
            if layer.f != -1:
                x = y[layer.f] if isinstance(layer.f, int) else [x if j == -1 else y[j] for j in layer.f]
            x = layer(x)
            y.append(x if layer.i in self.save_indices else None)
        return tuple(y[i] for i in self.out_indices)


def create_yolo_hybrid_cuda_graph(
    model: nn.Module,
    sample_input: torch.Tensor,
    warmup_iters: int = 3
):
    """Wrap Backbone and Neck into a hardware-accelerated static CUDA Graph.

    Uses PyTorch's official partial-network capture API (`torch.cuda.make_graphed_callables`)
    to graph the static Backbone & Neck forward and backward passes, while returning
    the dynamic Detect head to be executed eagerly.

    Args:
        model: YOLOv8 DetectionModel.
        sample_input: Sample input tensor with fixed training shape [B, 3, H, W].
        warmup_iters: Number of warmup iterations before capture (default: 3).

    Returns:
        tuple (graphed_backbone_neck, detect_head)
    """
    if not torch.cuda.is_available() or sample_input.device.type != 'cuda':
        raise RuntimeError("Hybrid CUDA Graph requires an active CUDA device.")

    if hasattr(torch.autograd.graph, 'set_override_stale_capture_stream'):
        torch.autograd.graph.set_override_stale_capture_stream(True)

    detect_layer = model.model[-1]
    out_indices = detect_layer.f if isinstance(detect_layer.f, list) else [15, 18, 21]

    backbone_neck = YOLOBackboneNeck(
        model.model[:-1],
        model.save,
        out_indices=out_indices
    ).to(sample_input.device)
    backbone_neck.train(model.training)

    # Partial-network capture: graphs forward and registers backward CUDA graph autograd node
    graphed_backbone_neck = torch.cuda.make_graphed_callables(
        backbone_neck,
        sample_args=(sample_input,),
        num_warmup_iters=max(3, warmup_iters)
    )

    return graphed_backbone_neck, detect_layer


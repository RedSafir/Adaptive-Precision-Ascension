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

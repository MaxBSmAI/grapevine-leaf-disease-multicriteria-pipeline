"""Architecture-specific complementary post hoc explanation methods."""

from __future__ import annotations

import math
from typing import Any

from grape_disease.utils.dependencies import require_module

torch = require_module("torch")
functional = torch.nn.functional


def _normalise_map(value: Any) -> Any:
    minimum = value.amin(dim=(-2, -1), keepdim=True)
    maximum = value.amax(dim=(-2, -1), keepdim=True)
    return (value - minimum) / (maximum - minimum + 1e-12)


def gradcam_plus_plus(
    model: Any, inputs: Any, target: int, model_name: str
) -> dict[str, Any]:
    """Run Grad-CAM++ on the final convolutional feature layer."""

    grad_cam = require_module("pytorch_grad_cam", "grad-cam")
    targets_module = require_module("pytorch_grad_cam.utils.model_targets", "grad-cam")
    if model_name == "resnet50":
        target_layer = model.layer4[-1]
    elif model_name == "yolov8n_cls":
        target_layer = model.model.model[-2]
    elif model_name == "optional_basic_cnn":
        target_layer = model.features[-4]
    else:
        raise ValueError(f"Grad-CAM++ is not configured for {model_name}")
    cam = grad_cam.GradCAMPlusPlus(model=model, target_layers=[target_layer])
    values = cam(
        input_tensor=inputs,
        targets=[targets_module.ClassifierOutputTarget(target)],
    )
    map_tensor = torch.as_tensor(values, device=inputs.device, dtype=inputs.dtype)
    return {
        "map": _normalise_map(map_tensor),
        "method": "grad_cam_plus_plus",
        "target": target,
    }


def _attention_from_input(module: Any, inputs: tuple[Any, ...]) -> Any:
    tokens = inputs[0]
    if not getattr(module, "batch_first", False):
        tokens = tokens.transpose(0, 1)
    embedding_dim = int(module.embed_dim)
    heads = int(module.num_heads)
    head_dim = embedding_dim // heads
    projected = functional.linear(tokens, module.in_proj_weight, module.in_proj_bias)
    query, key, _ = projected.chunk(3, dim=-1)
    batch, sequence, _ = query.shape
    query = query.view(batch, sequence, heads, head_dim).transpose(1, 2)
    key = key.view(batch, sequence, heads, head_dim).transpose(1, 2)
    attention = torch.softmax(
        torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1
    )
    return attention.mean(dim=1)


def vit_attention_rollout(model: Any, inputs: Any) -> dict[str, Any]:
    """Reconstruct and roll out ViT self-attention from encoder inputs."""

    attentions: list[Any] = []
    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):

            def capture(current_module: Any, current_inputs: tuple[Any, ...]) -> None:
                attentions.append(_attention_from_input(current_module, current_inputs))

            hooks.append(module.register_forward_pre_hook(capture))
    try:
        model.eval()
        with torch.inference_mode():
            model(inputs)
    finally:
        for hook in hooks:
            hook.remove()
    if not attentions:
        raise RuntimeError("No ViT multi-head attention modules were captured")
    batch, tokens, _ = attentions[0].shape
    identity = torch.eye(tokens, device=inputs.device, dtype=inputs.dtype).expand(
        batch, -1, -1
    )
    rollout = identity
    for attention in attentions:
        augmented = attention + identity
        augmented = augmented / augmented.sum(dim=-1, keepdim=True)
        rollout = torch.bmm(augmented, rollout)
    patch_values = rollout[:, 0, 1:]
    side = int(math.sqrt(patch_values.shape[1]))
    if side * side != patch_values.shape[1]:
        raise RuntimeError("ViT patch token count is not a square")
    map_tensor = patch_values.view(batch, 1, side, side)
    map_tensor = functional.interpolate(
        map_tensor,
        size=inputs.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return {
        "map": _normalise_map(map_tensor),
        "method": "attention_rollout",
        "attention_layers": len(attentions),
    }


def swin_shifted_window_activation_aggregation(
    model: Any, inputs: Any
) -> dict[str, Any]:
    """Approximate Swin spatial aggregation from multi-stage token activations.

    This is deliberately not labelled standard Attention Rollout because the
    shifted-window implementation does not expose a single global attention
    matrix through TorchVision's public forward interface.
    """

    activations: list[Any] = []
    hooks = []
    for module in model.features:

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                activations.append(output.detach())

        hooks.append(module.register_forward_hook(capture))
    try:
        model.eval()
        with torch.inference_mode():
            model(inputs)
    finally:
        for hook in hooks:
            hook.remove()
    maps: list[Any] = []
    for activation in activations:
        if activation.shape[-1] > activation.shape[1]:
            spatial = activation.abs().mean(dim=-1, keepdim=False).unsqueeze(1)
        else:
            spatial = activation.abs().mean(dim=1, keepdim=True)
        spatial = functional.interpolate(
            spatial,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        maps.append(_normalise_map(spatial))
    if not maps:
        raise RuntimeError("No Swin spatial stages were captured")
    aggregate = torch.stack(maps).mean(dim=0)
    return {
        "map": _normalise_map(aggregate),
        "method": "approximate_shifted_window_activation_aggregation",
        "captured_stages": len(maps),
    }


def run_specific_method(
    model: Any, inputs: Any, target: int, model_name: str
) -> dict[str, Any]:
    """Dispatch the predeclared complementary method for one architecture."""

    if model_name in {"resnet50", "yolov8n_cls", "optional_basic_cnn"}:
        return gradcam_plus_plus(model, inputs, target, model_name)
    if model_name == "vit_b_16":
        return vit_attention_rollout(model, inputs)
    if model_name == "swin_tiny":
        return swin_shifted_window_activation_aggregation(model, inputs)
    raise ValueError(f"No architecture-specific XAI method for {model_name}")

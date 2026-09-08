"""Common logits-only model factory for the confirmatory architecture set."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from grape_disease.utils.dependencies import package_version, require_module

torch = require_module("torch")
nn = torch.nn

CONFIRMATORY_MODELS = ("resnet50", "vit_b_16", "swin_tiny", "yolov8n_cls")


@dataclass(frozen=True)
class ModelSpec:
    """Serializable model-construction contract."""

    name: str
    num_classes: int = 4
    input_channels: int = 3
    input_size: tuple[int, int] = (224, 224)
    pretrained: bool = False
    architecture_config: str | None = None

    def validate(self) -> None:
        if self.name not in (*CONFIRMATORY_MODELS, "optional_basic_cnn"):
            raise ValueError(f"Unknown model: {self.name}")
        if self.num_classes <= 1 or self.input_channels != 3:
            raise ValueError("Models require RGB input and at least two classes")
        if self.pretrained:
            raise ValueError("The frozen protocol prohibits pretrained weights")


class YoloClassificationLogits(nn.Module):  # type: ignore[misc, name-defined]
    """Expose raw YOLO classification logits in both train and eval modes."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Any) -> Any:
        captured: list[Any] = []
        head = self.model.model[-1]
        linear = getattr(head, "linear", None)
        if linear is None:
            raise RuntimeError("Ultralytics classification head has no linear layer")

        def capture_logits(_module: Any, _inputs: Any, output: Any) -> None:
            captured.append(output)

        hook = linear.register_forward_hook(capture_logits)
        try:
            self.model(inputs)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("YOLO logits hook did not capture exactly one output")
        logits = captured[0]
        if logits.ndim != 2:
            raise RuntimeError(f"YOLO logits must be rank two, got {logits.shape}")
        return logits


def _build_optional_basic_cnn(num_classes: int) -> Any:
    class OptionalBasicCNN(nn.Module):  # type: ignore[misc, name-defined]
        sanity_baseline_only = True

        def __init__(self) -> None:
            super().__init__()
            channels = (3, 32, 64, 128, 256)
            blocks = []
            for input_channels, output_channels in pairwise(channels):
                blocks.extend(
                    [
                        nn.Conv2d(
                            input_channels,
                            output_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(output_channels),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                    ]
                )
            self.features = nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(0.30)
            self.classifier = nn.Linear(256, num_classes)
            self.apply(self._initialise)

        @staticmethod
        def _initialise(module: Any) -> None:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)

        def forward(self, inputs: Any) -> Any:
            features = self.pool(self.features(inputs)).flatten(1)
            return self.classifier(self.dropout(features))

    return OptionalBasicCNN()


def build_model(spec: ModelSpec, repository_root: Path | None = None) -> Any:
    """Instantiate one architecture from scratch without network access."""

    spec.validate()
    if spec.name == "optional_basic_cnn":
        return _build_optional_basic_cnn(spec.num_classes)
    if spec.name in {"resnet50", "vit_b_16", "swin_tiny"}:
        models = require_module("torchvision.models", "torchvision")
        if spec.name == "resnet50":
            return models.resnet50(weights=None, num_classes=spec.num_classes)
        if spec.name == "vit_b_16":
            return models.vit_b_16(weights=None, num_classes=spec.num_classes)
        return models.swin_t(weights=None, num_classes=spec.num_classes)
    tasks = require_module("ultralytics.nn.tasks", "ultralytics")
    if spec.architecture_config:
        config_path = Path(spec.architecture_config)
        if not config_path.is_absolute() and repository_root is not None:
            config_path = repository_root / config_path
    elif repository_root is not None:
        config_path = repository_root / "configs" / "models" / "yolov8n_cls.yaml"
    else:
        raise ValueError("YOLOv8n-CLS requires a local architecture_config")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    base = tasks.ClassificationModel(
        cfg=str(config_path), ch=spec.input_channels, nc=spec.num_classes, verbose=False
    )
    return YoloClassificationLogits(base)


def model_metadata(model: Any, spec: ModelSpec) -> dict[str, Any]:
    """Return auditable construction and parameter information."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "spec": asdict(spec),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "torch_version": package_version("torch"),
        "torchvision_version": package_version("torchvision"),
        "ultralytics_version": package_version("ultralytics"),
        "weights_downloaded": False,
        "logits_only": True,
    }


def load_model_state_strict(model: Any, state_dict: dict[str, Any]) -> None:
    """Load checkpoint parameters with strict key and shape validation."""

    model.load_state_dict(state_dict, strict=True)

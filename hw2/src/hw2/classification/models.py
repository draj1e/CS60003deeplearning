from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models.resnet import BasicBlock


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)
        return x * weights


class SEBasicBlock(BasicBlock):
    def __init__(self, *args, reduction: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.se = SEBlock(self.conv2.out_channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def build_classifier(arch: str, num_classes: int, pretrained: bool, attention: str = "none") -> nn.Module:
    weights = None
    if pretrained:
        if arch == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        elif arch == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1

    if attention == "se":
        layers = [2, 2, 2, 2] if arch == "resnet18" else [3, 4, 6, 3]
        model = models.ResNet(block=SEBasicBlock, layers=layers, num_classes=num_classes)
        if pretrained:
            plain = getattr(models, arch)(weights=weights)
            state = plain.state_dict()
            compatible = {key: value for key, value in state.items() if key in model.state_dict() and model.state_dict()[key].shape == value.shape}
            model.load_state_dict(compatible, strict=False)
    else:
        model = getattr(models, arch)(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model


def parameter_groups(model: nn.Module, lr_backbone: float, lr_head: float) -> list[dict]:
    head_params = list(model.fc.parameters())
    head_ids = {id(param) for param in head_params}
    backbone_params = [param for param in model.parameters() if id(param) not in head_ids]
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]

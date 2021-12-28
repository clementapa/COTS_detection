import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNext(nn.Module):
    def __init__(self,
                 nclasses: int = 20,
                 pretrained=True,
                 freeze=False) -> None:
        super(ResNext, self).__init__()

        backbone = models.resnext101_32x8d(pretrained=pretrained)
        # Freeze backbone weights
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False

        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))
        self.classifier = nn.Sequential(
            nn.Linear(backbone.fc.in_features, backbone.fc.in_features // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(backbone.fc.in_features // 2),
            nn.Linear(backbone.fc.in_features // 2, nclasses),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
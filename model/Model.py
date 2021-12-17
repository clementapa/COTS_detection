import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNext(nn.Module):

    def __init__(self, nclasses: int=20, pretrained=True, freeze=False) -> None:
        super(ResNext, self).__init__()

        backbone = models.resnext101_32x8d(pretrained=pretrained)
        # Freeze backbone weights 
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False

        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))
        self.classifier = nn.Sequential(
            nn.Linear(backbone.fc.in_features, backbone.fc.in_features//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(backbone.fc.in_features//2),
            nn.Linear(backbone.fc.in_features//2, nclasses),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FasterRCNN(nn.Module):

    def __init__(self, nclasses: int=2, pretrained=False, pretrained_backbone=False) -> None: 
        super(FasterRCNN, self).__init__()
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=pretrained_backbone)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, nclasses)

        self.model = model

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(x, y)

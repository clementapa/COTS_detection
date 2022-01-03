import torch
import torch.nn as nn
import torchvision.models as models
import math

class RetinaNet(nn.Module):
    def __init__(self,
                 nclasses: int = 2,
                 pretrained=False,
                 pretrained_backbone=False) -> None:
        super(RetinaNet, self).__init__()

        model = models.detection.retinanet_resnet50_fpn(
            pretrained=pretrained, pretrained_backbone=pretrained_backbone)

        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.num_classes = nclasses
        
        out_features = model.head.classification_head.conv[0].out_channels
        cls_logits = torch.nn.Conv2d(out_features, num_anchors * nclasses, kernel_size = 3, stride=1, padding=1)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
        torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code 
        # assign cls head to model
        model.head.classification_head.cls_logits = cls_logits

        self.model = model

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Output during inference:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
            between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction 
        """
        if y != None:
            return self.model(x, y)
        else:
            return self.model(x)

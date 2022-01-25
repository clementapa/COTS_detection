from transformers import ViTFeatureExtractor
import torch.nn as nn

class backboneViT(nn.Module):

    def __init__(self, pretrained=False) -> None:
        if pretrained:
            backbone = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        else:
            backbone = ViTFeatureExtractor(do_resize=True, size=256, return_tensors='pt')
    
        # self.out_channels =
    
    def forward(self, x):
        pass

# https://github.com/rwightman/pytorch-image-models
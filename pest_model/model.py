from typing import Optional
import torch
import torch.nn as nn
import timm


class PestClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        # create backbone without classifier head
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def build_model(backbone: str, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
    model = PestClassifier(backbone_name=backbone, num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    return model

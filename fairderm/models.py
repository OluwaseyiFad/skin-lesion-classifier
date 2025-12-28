# efficientnet-b2 classifier

import torch
import torch.nn as nn
import timm


class SkinLesionClassifier(nn.Module):
    # efficientnet-b2 backbone + custom head
    # input: 260x260 -> features -> dropout -> fc -> relu -> dropout -> output

    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.3):
        super(SkinLesionClassifier, self).__init__()

        self.backbone = timm.create_model(
            'efficientnet_b2',
            pretrained=pretrained,
            num_classes=0  # removes original head
        )

        self.num_features = self.backbone.num_features  # 1408 for efficientnet-b2

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def get_features(self, x):
        return self.backbone(x)


def create_model(num_classes=2, pretrained=True, freeze_backbone=False, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SkinLesionClassifier(num_classes=num_classes, pretrained=pretrained)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen, only classifier will be trained")

    return model.to(device)

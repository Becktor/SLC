import timm
import torch.nn as nn


class GluonResnext50(nn.Module):
    def __init__(self, model_name="gluon_resnext50_32x4d", n_classes=8):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        channels = self.model.feature_info.channels()
        self.fc = nn.Linear(channels[-1], n_classes)
        self.weight = nn.Linear(channels[-1], 1)

    def forward(self, x):
        x = self.model(x)[-1]
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        pred = self.fc(x)
        w = self.weight(x)
        return pred, w
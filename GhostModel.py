import timm.models
import torch
import torch.nn as nn
from timm.models.ghostnet import GhostModule
from timm.models.layers import SelectAdaptivePool2d
import math

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class GhostVGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, drop_rate=0.5, global_pool='avg'):
        super(GhostVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.num_features = cfg[vgg_name][-2]
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            torch.nn.Linear(self.num_features, 512),
            nn.Dropout(drop_rate),
            nn.ReLU(True),
        )
        self.fc = torch.nn.Linear(512, num_classes)

    def forward_features(self, x):
        out = self.features(x)
        out = self.global_pool(out)
        return out

    def forward(self, x):
        out = self.forward_features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        pred = self.fc(out)
        return pred

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [GhostModule(in_channels, x, kernel_size=3)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def main():
    from torchinfo import summary
    net = GhostVGG('VGG16')
    summary(net, input_size=(16, 1, 128, 128))


if __name__ == '__main__':
    main()


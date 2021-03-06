import math

import torch.nn as nn
import torch.nn.init as init
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


@variational_estimator
class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features, out_nodes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            BayesianLinear(512, 512),
            nn.ReLU(True),
            # nn.Dropout(),
            BayesianLinear(512, 512),
            nn.ReLU(True),
        )
        self.to_multivariate = nn.Linear(512, 128)
        self.relu = nn.ReLU(True)
        self.fc = BayesianLinear(128, out_nodes)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        for m in self.modules():
            if isinstance(m, BayesianConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight_mu.data.normal_(0, math.sqrt(2. / n))
                m.bias_mu.data.zero_()

    def forward(self, x):
        x, _ = self.forward_step(x)
        return x

    def forward_step(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        output = self.to_multivariate(x)
        x = self.fc(output)
        return x, output


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = BayesianConv2d(in_channels, v, kernel_size=(3, 3), padding=1, bias=True)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), **kwargs)


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), **kwargs)


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), **kwargs)


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), **kwargs)


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)

import timm
import torch
import torch.nn as nn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from torchvision import transforms as T


def eval_samples(model, x, samples=10, std_multiplier=2):
    preds = [torch.softmax(model.forward(x), dim=1) for _ in range(samples)]
    soft_stack = torch.stack(preds)
    means = soft_stack.mean(axis=0)
    stds = soft_stack.std(axis=0)
    softmax_upper = means + (std_multiplier * stds)
    softmax_lower = means - (std_multiplier * stds)
    return {'mean': means, 'stds': stds, 'sp': means, 'sp_u': softmax_upper, 'sp_l': softmax_lower, 'preds': soft_stack}


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


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


class TTAModel(nn.Module):
    def __init__(self, model_name="gluon_resnext50_32x4d", n_classes=8):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, drop_rate=0.5, num_classes=n_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.to_size = T.Resize((128, 128))

    def forward(self, x):
        return self.model(x)

    def evaluate_classification(self,
                                x,
                                transforms,
                                std_multiplier=2):
        self.eval()
        preds = []
        for t in transforms:
            tx = t(x)
            for ttx in tx:
                if ttx.shape[-1] != 128:
                    ttx = self.to_size(ttx)
                pred = self.forward(ttx)
                sp = torch.softmax(pred, axis=1)
                preds.append(sp)

        soft_stack = torch.stack(preds)
        means = soft_stack.mean(axis=0)
        stds = soft_stack.std(axis=0)
        softmax_upper = means + (std_multiplier * stds)
        softmax_lower = means - (std_multiplier * stds)
        return {'mean': means, 'stds': stds, 'sp': means, 'sp_u': softmax_upper, 'sp_l': softmax_lower,
                'preds': soft_stack}


class DropoutGluonResnext50(nn.Module):
    def __init__(self, model_name="gluon_resnext50_32x4d", n_classes=8):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, drop_rate=0.5, num_classes=n_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        pred = self.model(x)
        return pred

    def loss(self, X, y):
        self.train()
        preds = self.forward(X)
        loss = self.loss_fn(preds, y)
        return loss

    def evaluate_classification(self,
                                X,
                                samples=10,
                                std_multiplier=2, train=False):
        self.eval()
        if train:
            self.train()
        return eval_samples(self, X, samples, std_multiplier)


@variational_estimator
class BayesGluonResnext50(nn.Module):
    def __init__(self, model_name="gluon_resnext50_32x4d", n_classes=8):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        channels = self.model.feature_info.channels()
        self.bayes = BayesianLinear(channels[-1], n_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.sample_rate = 5
        self.k_comp = 0.001

    def forward(self, x):
        self.train()
        x = self.model(x)[-1]
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        pred = self.bayes(x)
        return pred

    def loss(self, x, y):
        return self.sample_elbo(inputs=x,
                                labels=y,
                                criterion=self.loss_fn,
                                sample_nbr=self.sample_rate,
                                complexity_cost_weight=self.k_comp)

    def evaluate_classification(self,
                                X,
                                samples=10,
                                std_multiplier=2):
        self.eval()
        return eval_samples(self, X, samples, std_multiplier)


class VOSGluonResnext50(nn.Module):
    def __init__(self, model_name="gluon_resnext50_32x4d", n_classes=8):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        channels = self.model.feature_info.channels()
        self.to_multivariate_variables = torch.nn.Linear(channels[-1], 128)
        self.fc = torch.nn.Linear(128, n_classes)

    def forward_vos(self, x):
        self.train()
        x = self.model(x)[-1]
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        output = self.to_multivariate_variables(x)
        pred = self.fc(output)
        return pred, output

    def forward(self, x):
        self.train()
        x = self.model(x)[-1]
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        output = self.to_multivariate_variables(x)
        pred = self.fc(output)
        return pred

    def evaluate_classification(self,
                                preds,
                                samples=10,
                                std_multiplier=2):
        self.eval()
        return eval_samples(self, preds, samples, std_multiplier)

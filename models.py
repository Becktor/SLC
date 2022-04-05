import timm
import torch
import torch.nn as nn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


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


class DropoutGluonResnext50(nn.Module):
    def __init__(self, model_name="gluon_resnext50_32x4d", n_classes=8):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, drop_rate=0.1, num_classes=n_classes)
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
                                y,
                                samples=10,
                                std_multiplier=2):
        self.train()
        preds = [self.forward(X) for _ in range(samples)]
        preds = torch.stack(preds)
        means = preds.mean(axis=0)
        stds = preds.std(axis=0)
        softmax_upper = torch.softmax(means + (std_multiplier * stds), dim=1)
        softmax_lower = torch.softmax(means - (std_multiplier * stds), dim=1)
        softmax_pred = torch.softmax(means, dim=1)
        return {'mean': means, 'stds': stds, 'sp': softmax_pred, 'sp_u': softmax_upper, 'sp_l': softmax_lower}


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
                                y,
                                samples=10,
                                std_multiplier=2):
        self.eval()
        preds = [self.forward(X) for _ in range(samples)]
        preds = torch.stack(preds)
        means = preds.mean(axis=0)
        stds = preds.std(axis=0)
        softmax_upper = torch.softmax(means + (std_multiplier * stds), dim=1)
        softmax_lower = torch.softmax(means - (std_multiplier * stds), dim=1)
        softmax_pred = torch.softmax(means, dim=1)
        return {'mean': means, 'stds': stds, 'sp': softmax_pred, 'sp_u': softmax_upper, 'sp_l': softmax_lower}

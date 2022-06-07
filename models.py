import blitz.models
import timm
import torch
import torch.nn as nn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import bayes_models as bm
from torchvision import transforms as T
from GhostModel import GhostVGG


def eval_samples(model, x, samples=10, std_multiplier=2):
    outputs = [model.predict(x) for _ in range(samples)]
    output_stack = torch.stack(outputs)
    log_sum_exps = torch.logsumexp(output_stack, dim=2)
    lse_m = log_sum_exps.mean(0)
    lse_std = log_sum_exps.std(0)
    preds = [torch.softmax(output, dim=1) for output in outputs]
    soft_stack = torch.stack(preds)
    means = soft_stack.mean(axis=0)
    stds = soft_stack.std(axis=0)
    softmax_upper = means + (std_multiplier * stds)
    softmax_lower = means - (std_multiplier * stds)
    return {'mean': means, 'stds': stds, 'sp': means, 'sp_u': softmax_upper,
            'sp_l': softmax_lower, 'preds': soft_stack, 'lse': log_sum_exps, 'lse_m': lse_m, 'lse_s': lse_std}


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
        self.train()
        return self.model(x)

    def predict(self, x):
        self.eval()
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


class DropoutModel(nn.Module):
    def __init__(self, model_name="gluon_resnext50_32x4d", n_classes=8, drop_rate=0.5):
        super(DropoutModel, self).__init__()
        if 'g_VGG' in model_name:
            self.model = GhostVGG(model_name[2:], n_classes, drop_rate)
        else:
            self.model = timm.create_model(model_name, pretrained=True, drop_rate=drop_rate)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            torch.nn.Linear(self.model.num_features, 512),
            nn.Dropout(drop_rate),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(512, n_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        self.train()
        x = self.model.forward_features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        pred = self.fc(output)
        return pred

    def predict(self, x):
        return self.forward(x)

    def loss(self, X, y):
        loss = self.loss_fn(X, y)
        return loss

    def evaluate_classification(self,
                                X,
                                samples=10,
                                std_multiplier=2, train=False):

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
        self.sample_rate = 10
        self.k_comp = 0.001

    def forward(self, x):
        self.train()
        x = self.model(x)[-1]
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        pred = self.bayes(x)
        return pred

    def predict(self, x):
        self.eval()
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

        return eval_samples(self, X, samples, std_multiplier)


@variational_estimator
class BayesVGG16(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.model = bm.vgg16(out_nodes=n_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.sample_rate = 10
        self.k_comp = 1e-6

    def forward(self, x):
        self.train()
        x = self.model(x)
        return x

    def predict(self, x):
        self.eval()
        x = self.model(x)
        return x

    def loss(self, x, y):
        return self.sample_elbo(inputs=x,
                                labels=y,
                                criterion=self.loss_fn,
                                sample_nbr=self.sample_rate,
                                complexity_cost_weight=self.k_comp)

    def evaluate_classification(self,
                                pred,
                                samples=10,
                                std_multiplier=2):
        return eval_samples(self, pred, samples, std_multiplier)


class VOSModel(nn.Module):
    def __init__(self, model_name="gluon_resnext50_32x4d", n_classes=8, drop_rate=0.5, start_epoch=20):
        super(VOSModel, self).__init__()
        self.start_epoch = start_epoch
        if 'g_VGG' in model_name:
            self.model = GhostVGG(model_name[2:], n_classes, drop_rate)
        else:
            self.model = timm.create_model(model_name, pretrained=True, drop_rate=drop_rate)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            torch.nn.Linear(self.model.num_features, 512),
            nn.Dropout(drop_rate),
            nn.ReLU(True),
        )
        self.to_multivariate_variables = torch.nn.Linear(512, 128)
        self.fc = torch.nn.Linear(128, n_classes)

        self.running_means = []
        self.running_stds = []
        self.at_epoch = 0
        for _ in range(n_classes):
            self.running_means.append(torch.zeros(1000))
            self.running_stds.append(torch.ones(1000))
        self.vos_mean = nn.Parameter(torch.zeros([n_classes]), requires_grad=False)
        self.vos_std = nn.Parameter(torch.ones([n_classes]), requires_grad=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward_step(self, x):
        self.train()
        x = self.model.forward_features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        output = self.to_multivariate_variables(x)
        pred = self.fc(output)
        return pred, output

    def update_lse(self, pred):
        lse = torch.logsumexp(pred, dim=1)
        pred_idc = torch.argmax(pred, dim=1)
        cls_means = {}

        for x, y in enumerate(pred_idc):
            y = int(y)
            if y not in cls_means:
                cls_means[y] = lse[x].unsqueeze(0)
            else:
                cls_means[y] = torch.cat([cls_means[y], lse[x].unsqueeze(0)])

        for k, v in cls_means.items():
            self.running_means[k][0] = v.mean()
            self.running_means[k] = self.running_means[k].roll(1)
            self.running_stds[k][0] = torch.tensor(1) if v.std().isnan() else v.std()
            self.running_stds[k] = self.running_stds[k].roll(1)

            idx = self.running_means[k] != 0
            self.vos_mean[k] = self.running_means[k][idx].mean()
            self.vos_std[k] = self.running_stds[k][idx].mean()

    def cdf_class(self, x, c):
        norm = torch.distributions.normal.Normal(self.vos_mean[c], self.vos_std[c])
        cdf = norm.cdf(torch.tensor(x))
        return cdf

    def cdf(self, x):
        norm = torch.distributions.normal.Normal(self.vos_mean.mean(), self.vos_std.mean())
        cdf = norm.cdf(torch.tensor(x))
        return cdf

    def forward(self, x):
        self.train()
        pred, output = self.forward_step(x)
        if self.at_epoch >= self.start_epoch:
            self.update_lse(pred)
        return pred, output

    # def forward(self, x):
    #     pred, _ = self.forward_vos(x)
    #     return pred

    def predict(self, x):
        self.train()
        pred, _ = self.forward_step(x)
        return pred

    def evaluate_classification(self,
                                preds,
                                samples=10,
                                std_multiplier=2):
        return eval_samples(self, preds, samples, std_multiplier)

    def loss(self, x, y):
        return self.loss_fn(x, y)


class VOSBayes(nn.Module):
    def __init__(self, n_classes=9, start_epoch=20):
        super().__init__()
        self.start_epoch = start_epoch
        self.model = BayesVGG16(n_classes)
        self.running_means = []
        self.running_stds = []
        self.at_epoch = 0
        for _ in range(n_classes):
            self.running_means.append(torch.zeros(1000))
            self.running_stds.append(torch.ones(1000))
        self.vos_mean = nn.Parameter(torch.zeros([n_classes]), requires_grad=False)
        self.vos_std = nn.Parameter(torch.ones([n_classes]), requires_grad=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.sample_rate = 10
        self.k_comp = 1e-6

    def update_lse(self, pred):
        lse = torch.logsumexp(pred, dim=1)
        pred_idc = torch.argmax(pred, dim=1)
        cls_means = {}

        for x, y in enumerate(pred_idc):
            y = int(y)
            if y not in cls_means:
                cls_means[y] = lse[x].unsqueeze(0)
            else:
                cls_means[y] = torch.cat([cls_means[y], lse[x].unsqueeze(0)])

        for k, v in cls_means.items():
            self.running_means[k][0] = v.mean()
            self.running_means[k] = self.running_means[k].roll(1)
            self.running_stds[k][0] = torch.tensor(1) if v.std().isnan() else v.std()
            self.running_stds[k] = self.running_stds[k].roll(1)

            idx = self.running_means[k] != 0
            self.vos_mean[k] = self.running_means[k][idx].mean()
            self.vos_std[k] = self.running_stds[k][idx].mean()

    def cdf_class(self, x, c):
        norm = torch.distributions.normal.Normal(self.vos_mean[c], self.vos_std[c])
        cdf = norm.cdf(torch.tensor(x))
        return cdf

    def cdf(self, x):
        norm = torch.distributions.normal.Normal(self.vos_mean.mean(), self.vos_std.mean())
        cdf = norm.cdf(torch.tensor(x))
        return cdf

    def forward_vos(self, x):
        self.train()
        pred, output = self.forward_step(x)
        if self.at_epoch >= self.start_epoch:
            self.update_lse(pred)
        return pred, output

    def predict(self, x):
        self.train()
        pred, _ = self.forward_step(x)
        return pred

    def evaluate_classification(self,
                                preds,
                                samples=10,
                                std_multiplier=2):
        return eval_samples(self, preds, samples, std_multiplier)

    def loss(self, x, y):
        return self.sample_elbo(inputs=x,
                                labels=y,
                                criterion=self.loss_fn,
                                sample_nbr=self.sample_rate,
                                complexity_cost_weight=self.k_comp)

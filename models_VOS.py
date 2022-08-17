import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from wrn import WideResNet


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
            'sp_l': softmax_lower, 'preds': soft_stack, 'lse': log_sum_exps,
            'lse_m': lse_m, 'lse_s': lse_std, 'o': output_stack}


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



class SuperDropout(nn.Module):
    def __init__(self, p=0.5, pt=0.10):
        super().__init__()
        self.p = p
        self.pt = pt

    def forward(self, x):
        if self.training:
            return F.dropout(x, p=self.p, training=True)
        else:
            return F.dropout(x, p=self.pt, training=True)


class VOSModel(nn.Module):
    def __init__(self, model_name="gluon_resnext50_32x4d", n_classes=8, drop_rate=0.3, start_epoch=40,
                 vos_multivariate_dim=128):
        super(VOSModel, self).__init__()
        self.start_epoch = start_epoch

        if 'wrn' in model_name:
            self.model = WideResNet(40, n_classes, 2, dropRate=drop_rate)
            in_channels = 128
            out_channels = 128
        else:
            self.model = timm.create_model(model_name, pretrained=True, drop_rate=drop_rate)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mcmc_layer = nn.Sequential(
            SuperDropout(drop_rate),
            nn.Linear(in_channels, out_channels, bias=False),
            nn.ReLU(inplace=True),
            SuperDropout(drop_rate),
        )
        self.drop_rate = drop_rate
        self.eye_matrix = torch.eye(vos_multivariate_dim, device='cuda')
        self.to_multivariate_variables = torch.nn.Linear(out_channels, vos_multivariate_dim)
        self.fc = torch.nn.Linear(vos_multivariate_dim, n_classes)
        self.weight_energy = torch.nn.Linear(n_classes, 1)
        torch.nn.init.uniform_(self.weight_energy.weight)

        self.logistic_regression = torch.nn.Linear(1, 2)

        self.running_means = []
        self.running_vars = []
        self.training_outputs = {}
        self.at_epoch = 0
        for i in range(n_classes):
            self.training_outputs[i] = torch.zeros(2000).cuda()

        self.vos_means = nn.Parameter(torch.zeros([n_classes, ]), requires_grad=False)
        self.vos_stds = nn.Parameter(torch.ones([n_classes, ]), requires_grad=False)
        self.vos_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.vos_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.running_ood_samples = torch.zeros(500 * n_classes).cuda()
        self.ood_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.ood_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.loss_fn = nn.CrossEntropyLoss()  # FocalLosses(gamma=1.2, reduction='mean')


    def fit_gauss(self):
        means, stds = [], []
        all_val = torch.cat(list(self.training_outputs.values())).flatten()
        all_val = all_val[all_val.nonzero()]
        if all_val.sum() == 0:
            return

        for i in range(len(self.training_outputs)):
            x = self.training_outputs[i]
            x = x[x.nonzero()]
            means.append(x.mean())
            stds.append(x.std())

        self.vos_mean = torch.nn.Parameter(all_val.mean(), requires_grad=False)
        self.vos_std = torch.nn.Parameter(all_val.std(), requires_grad=False)

        self.vos_means = nn.Parameter(torch.stack(means), requires_grad=False)
        self.vos_stds = nn.Parameter(torch.stack(stds), requires_grad=False)

        self.ood_mean = nn.Parameter(self.running_ood_samples.mean(), requires_grad=False)
        if self.running_ood_samples.nonzero().shape[0] > 0:
            self.ood_std = nn.Parameter(self.running_ood_samples.std(), requires_grad=False)

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation

        value.exp().sum(dim, keepdim).log()
        """
        return torch.logsumexp(value, dim=dim)
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(
                F.relu(self.weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            return m + torch.log(sum_exp)

    def ood_pred(self, x):
        lse = self.log_sum_exp(x, dim=1)
        pred = torch.argmax(x, 1)
        run_means = [self.vos_means[t] for t in pred]
        run_means = torch.stack(run_means)
        run_stds = [self.vos_stds[t] for t in pred]
        run_stds = torch.stack(run_stds)
        clamped_inp = torch.clamp(lse, min=0.001)
        shifted_means = (self.ood_mean-self.ood_std) * (torch.log(run_means/clamped_inp))
        out_j = (shifted_means).unsqueeze(1)
        output = torch.cat((x, out_j), 1)
        return output

    def forward_step(self, inp):
        x = self.model.get_features(inp)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        output = self.mcmc_layer(x)
        pred = self.fc(output)
        return pred, output

    def update_lse(self, pred, target):
        lse = self.log_sum_exp(pred, 1)
        pred_idc = torch.argmax(pred, 1)

        cls_means = {}
        for x, y in enumerate(pred_idc):
            y = int(y)
            if target[x] != y:
                self.running_ood_samples[0] = lse[x]
                self.running_ood_samples = self.running_ood_samples.roll(1)
            elif y not in cls_means:
                cls_means[y] = lse[x].unsqueeze(0)
            else:
                cls_means[y] = torch.cat([cls_means[y], lse[x].unsqueeze(0)])

        for k, v in cls_means.items():
            try:
                idx = v.nonzero().squeeze(1)
                self.training_outputs[k] = self.training_outputs[k].roll(idx.shape[0])
                self.training_outputs[k][idx] = v
            except:
                print(k, v.shape)

    def cdf_class(self, x, c):
        norm = torch.distributions.normal.Normal(self.vos_means[c], self.vos_stds[c])
        cdf = norm.cdf(x)
        return cdf

    def cdf(self, x):
        norm = torch.distributions.normal.Normal(self.vos_mean, self.vos_std)
        cdf = norm.cdf(x)
        return cdf

    def forward(self, x, y=None):
        pred, output = self.forward_step(x)
        if y is not None and self.at_epoch >= (self.start_epoch - 1):
            self.update_lse(pred, y)
        return [pred, output]#, self.ood_pred(pred)]

    def calibrate_means_stds(self, means, stds):
        self.vos_means = nn.Parameter(torch.stack(means), requires_grad=False)
        self.vos_stds = nn.Parameter(torch.stack(stds), requires_grad=False)

    def predict(self, x):
        self.eval()
        pred, _ = self.forward_step(x)
        return pred

    def evaluate_classification(self,
                                preds,
                                samples=10,
                                std_multiplier=2):
        return self.eval_samples(preds, samples, std_multiplier)

    def eval_samples(self, x, samples=10, std_multiplier=2):
        outputs = [self.predict(x) for _ in range(samples)]
        output_stack = torch.stack(outputs)
        logistic_reg = [self.ood_pred(output) for output in outputs]
        lr_stack = torch.stack(logistic_reg)
        lr_soft = torch.softmax(lr_stack, dim=-1)
        log_sum_exps = self.log_sum_exp(output_stack, dim=2)
        # log_sum_exps = torch.logsumexp(output_stack, dim=2)
        lse_m = log_sum_exps.mean(0)
        lse_std = log_sum_exps.std(0)
        preds = [torch.softmax(output, dim=1) for output in outputs]
        soft_stack = torch.stack(preds)
        means = soft_stack.mean(axis=0)
        stds = soft_stack.std(axis=0)
        softmax_upper = means + (std_multiplier * stds)
        softmax_lower = means - (std_multiplier * stds)
        return {'mean': means, 'stds': stds, 'sp': means, 'sp_u': softmax_upper,
                'sp_l': softmax_lower, 'preds': soft_stack, 'lse': log_sum_exps,
                'lse_m': lse_m, 'lse_s': lse_std, 'o': output_stack, 'lrs': lr_stack, 'lr_soft': lr_soft}

    def loss(self, x, y):
        return self.loss_fn(x, y)

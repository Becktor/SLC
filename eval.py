import csv
import os
import fiftyone as fo
import higher as higher
import pandas as pd
import scipy
import timm
import torch
import ttach as tta
from torch.utils.data import Dataset, DataLoader
from DataLoader import ShippingLabClassification, letterbox
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import itertools
import wandb
from models import GluonResnext50, BayesGluonResnext50, DropoutModel, TTAModel, BayesVGG16, VOSModel
import torch.nn as nn
from pathlib import Path
import random
from fitter import Fitter
import sklearn.metrics as metrics
import torch.onnx
import helper_functions.metrics as met
from fitter import get_common_distributions
import seaborn as sns
import torchvision
import sklearn
from itertools import cycle
from helper_functions.metrics import show_performance_fpr, get_measures, fpr_at_tpr, fpr_tpr
import torchvision.datasets as dset


class TransformWrapper(object):
    def __init__(self, transform, n=1):
        self.trans = transform
        self.n = n

    def __call__(self, img):
        imgs = [self.trans(img) for _ in range(self.n)]
        return imgs


idx_to_check = [[29, 1], [29, 1], [27, 3], [27, 3], [24, 4], [24, 4], [14, 5], [14, 5], [25, 6], [25, 6], [19, 14],
                [19, 14], [12, 18], [12, 18], [17, 18], [17, 18], [14, 20], [14, 20], [14, 23], [14, 23], [4, 24],
                [4, 24], [22, 24], [22, 24], [30, 24], [30, 24], [9, 25], [9, 25], [10, 25], [10, 25], [16, 25],
                [16, 25], [23, 25], [23, 25], [5, 26], [5, 26], [6, 28], [6, 28], [7, 28], [7, 28], [8, 28], [8, 28],
                [9, 28], [9, 28], [28, 34], [28, 34], [28, 37], [28, 37], [24, 39], [25, 39], [29, 39], [30, 39],
                [30, 39], [31, 39], [1, 40], [2, 40], [28, 40], [22, 56], [23, 56], [18, 58], [28, 73], [29, 73],
                [29, 73], [30, 73], [3, 74], [4, 74], [10, 82], [11, 82], [25, 83], [26, 83], [13, 84], [14, 84],
                [3, 85], [4, 85], [18, 85], [16, 88], [17, 88], [20, 93], [21, 93], [21, 93], [22, 93], [24, 98],
                [25, 98], [25, 98], [26, 98], [17, 99], [18, 99], [21, 99], [22, 99], [28, 99], [29, 99], [13, 101],
                [14, 101], [1, 104], [2, 104], [26, 104], [27, 104], [28, 105], [29, 105], [12, 108], [6, 113],
                [7, 113], [22, 113], [30, 119], [31, 119], [25, 120], [26, 120], [30, 121], [31, 121], [2, 122],
                [16, 122], [17, 122], [28, 123], [29, 123], [19, 124], [20, 124], [2, 129], [3, 129], [15, 137],
                [16, 137], [8, 143], [9, 143]]

# train_norms = {4: (11.050959, 1.6934786), 3: (14.447998, 1.5380085), 5: (12.244996, 2.0072613),
#                7: (15.813443, 1.7174519), 2: (16.055696, 2.0404303), 0: (14.964114, 1.7310742),
#                1: (13.662182, 1.4823529), 6: (12.214176, 1.5037574)}

# train_norms = {4: (22.641314, 2.6660843), 1: (21.820875, 1.7641367), 2: (20.12747, 2.3930283), 8: (13.655789, 1.2330345), 3: (22.682724, 3.5433457), 7: (19.041382, 1.8500489), 5: (20.446945, 2.444383), 6: (18.666367, 2.6390777), 0: (14.238019, 1.7003791), 9: (19.35532, 1.5455061)}
# train_norms = {4: (-0.20737115, 0.7276402), 1: (0.73767483, 0.46118015), 2: (-0.59243923, 0.6190916), 8: (-1.2136457, 0.52655524), 3: (-0.055931814, 0.6492376), 7: (0.2306215, 0.4722455), 5: (0.035789445, 0.54746556), 6: (0.072558165, 0.50197697), 0: (0.06483747, 0.9217627), 9: (1.0638794, 0.5808415)}
train_norms = {3: (14.008406, 3.8698266), 8: (16.991377, 3.6263654), 0: (15.370032, 3.7869778), 6: (15.591267, 3.399394), 1: (16.760761, 2.9231396), 9: (16.355757, 3.545238), 5: (16.184753, 4.906386), 7: (15.685531, 3.6368687), 4: (15.913696, 3.8837087), 2: (15.373063, 4.195742)}


def run_net(root_dir, name='', ds='cifar', n_samp=25, max_iter=2000):
    val_dir = os.path.join(root_dir, 'val_set')
    data = r'Q:\git\SLC\ckpts'
    image_size = 32
    batch_size = 128
    workers = 4
    model_name = "wrn"
    torch.manual_seed(5)
    norms = {k: torch.distributions.Normal(v[0], v[1]) for k, v in train_norms.items()}

    if ds == 'ships':
        cj = transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.5)
        gauss = transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(3)]), p=0.25)
        rez = transforms.RandomApply(torch.nn.ModuleList([transforms.Resize(64), transforms.Resize(image_size)]), p=0.2)
        rez2 = transforms.RandomApply(torch.nn.ModuleList([transforms.Resize(32), transforms.Resize(image_size)]),
                                      p=0.2)
        val_set = ShippingLabClassification(root_dir=val_dir,
                                            transform=transforms.Compose([
                                                letterbox((image_size, image_size)),
                                                transforms.ToTensor(),
                                                transforms.RandomHorizontalFlip(),
                                                cj, gauss, rez, rez2
                                            ]))

        val_set = ShippingLabClassification(root_dir=val_dir,
                                            transform=transforms.Compose([
                                                letterbox((image_size, image_size)),
                                                transforms.ToTensor()
                                            ]))

        v_dataloader = DataLoader(val_set, batch_size=batch_size,
                                  shuffle=True, num_workers=workers)
        key_to_class = {b: a for a, b in val_set.classes.items()}

    else:
        mean = np.array([x / 255 for x in [125.3, 123.0, 113.9]])
        std = np.array([x / 255 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ])
        unorm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform_train)
        # v_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        #                                            shuffle=True, num_workers=workers, persistent_workers=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=False, transform=transform_test)
        if ds == 'SVHN':
            testset = torchvision.datasets.SVHN(root='./data', split='test',
                                                download=False, transform=transform_test)
        if ds == 'dtd':
            # /////////////// Textures ///////////////
            testset = dset.ImageFolder(root="./data/dtd/images",
                                       transform=transform_test)
        if ds == 'places365':
            # /////////////// Places365 ///////////////
            testset = dset.ImageFolder(root="./data/places365/",
                                        transform=transform_test)
        if ds == 'LSUN_resize':
            # /////////////// LSUN-R ///////////////
            testset = dset.ImageFolder(root="./data/LSUN_resize",
                                        transform=transform_test)

        if ds == 'iSUN':
            # /////////////// iSUN ///////////////
            testset = dset.ImageFolder(root="./data/iSUN",
                                        transform=transform_test)

        if ds == 'LSUN_C':
            # /////////////// LSUN-C ///////////////
            testset = dset.ImageFolder(root="./data/LSUN_C",
                                        transform=transform_test)

        # /////////////// Mean Results ///////////////

        # testset = trainset
        v_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                   shuffle=False, num_workers=workers, persistent_workers=True)

        key_to_class = {k: v for k, v in enumerate(trainset.classes)}

    n_classes = len(key_to_class.keys())
    if name == 'bayes':
        model = BayesVGG16(n_classes=n_classes)
    elif name == 'dropout':
        model = DropoutModel(n_classes=n_classes, model_name=model_name)
    elif name == 'vos':
        model = VOSModel(n_classes=n_classes, model_name=model_name)

    path = os.path.join(data, model_name + "_" + name + "_100.pt")

    model_dict = torch.load(os.path.join(root_dir, path))
    model.load_state_dict(model_dict['model_state_dict'])

    norms = {k: torch.distributions.Normal(v[0], v[1]) for k, v in enumerate(zip(model.vos_means, model.vos_stds))}
    norms_scaled = {k: torch.distributions.Normal(v[0], v[1]) for k, v in
                    enumerate(zip(model.vos_means, model.vos_stds))}
    # nms = [norms[v].mean for v in range(len(norms))]
    # stds = [norms[v].stddev for v in range(len(norms))]
    #
    # model.calibrate_means_stds(nms, stds)
    norms[10] = torch.distributions.Normal(model.ood_mean.cpu(), model.ood_std.cpu())
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    pe, ee = [], []
    acc, acc_u, acc_l, stds, acc_before_mean, t_t_acc = [], [], [], [], [], []
    t_std, t_pred, t_acc = [], [], []
    total_pred = []
    total_lbl = []
    pred_cert = []
    pred_soft = []
    cls_hist = {}
    hist = []
    oods = []
    gss_oods = []
    with torch.no_grad():
        tqd_e = tqdm(enumerate(v_dataloader, 0), total=len(v_dataloader))
        fig, axes = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})
        for i, data in tqd_e:
            if ds == 'ships':
                t_imgs, t_lbls, _, p = data
            else:
                t_imgs, t_lbls = data
                p = ''
            # if len(oods) >= max_iter:
            #     return oods, gss_oods
            t_imgs = t_imgs.cuda()
            t_lbls = t_lbls.cuda()

            obj = model.evaluate_classification(t_imgs, samples=n_samp, std_multiplier=2)

            predicted = torch.argmax(obj['lr_soft'].mean(0), dim=1)

            # predicted = torch.argmax(obj['sp'], dim=1)
            total_pred.append(predicted.cpu().numpy())
            total_lbl.append(t_lbls.cpu().numpy())
            acc.append((predicted.int() == t_lbls.int()).float())
            predicted_upper = torch.argmax(obj['sp_u'], dim=1)
            acc_u.append((predicted_upper.int() == t_lbls.int()).float())
            predicted_lower = torch.argmax(obj['sp_l'], dim=1)
            acc_l.append((predicted_lower.int() == t_lbls.int()).float())
            accuracy = torch.cat(acc, dim=0).mean().cpu()
            accuracy_u = torch.cat(acc_u, dim=0).mean().cpu()
            accuracy_l = torch.cat(acc_l, dim=0).mean().cpu()
            tqd_e.set_description(
                f'runn acc = {accuracy:.3f}, runn acc_u = {accuracy_u:.3f}, runn acc_l = {accuracy_l:.3f}')

            all_predictions = obj['preds']
            t_pred.append(all_predictions.cpu())
            # if i in [25, 75, 100, len(v_dataloader) - 1]:
            show_epoch = (t_imgs, t_lbls)

            img, lbl = show_epoch
            img = unorm(img)
            tp = torch.stack([norms[xx].mean.cpu() for xx in predicted.cpu().numpy()]).cuda()
            oods.append((obj['lse_m'] - (obj['lse_s']/obj['lse_m'])*tp).cpu().numpy())
            v = obj['lrs'].mean(0)[:, -1].cpu().numpy()
            gss_oods.append(-v)
            key_to_class[10] = 'unsure'
            for elem in range(len(obj["mean"])):
                if name == 'vos':
                    ood_m = obj['lse'][:, elem].mean()
                    # lr_c = #obj['lrs'][:, elem]
                    lr_s = obj['lr_soft'][:, elem].mean(0)
                    # lr_s2 = torch.softmax(lr_c[:, :-1], 1).mean(0)
                    # 95% confidence interval 2.5% are out of distribution since we only look at left tail
                    cls = np.argmax(lr_s.cpu().numpy())
                    # cls2 = np.argmax(lr_s2.detach().cpu().numpy())
                    conf_class = torch.tensor(0.0).cuda()
                    if cls != 10:
                        conf_class = norms_scaled[cls].cdf(ood_m)
                    #gss_oods.append(conf_class.cpu().numpy())
                    dlrs = lr_s.detach().cpu().numpy()
                    pred_cert.append((dlrs,
                                      conf_class.detach().cpu().numpy()))

                    if cls != 10:
                        pred_soft.append((dlrs, dlrs[cls], lbl[elem].cpu().numpy()))
                    else:
                        pred_soft.append((dlrs, dlrs[cls], np.array(10)))
                    if cls in cls_hist:
                        cls_hist[cls].append(ood_m.cpu().numpy())
                    else:
                        cls_hist[cls] = [ood_m.cpu().numpy()]
                    hist.append(ood_m.cpu().numpy())

                if conf_class < 1.190:# and cls != 10:  # and lbl[elem] == cls:
                    continue
                torch_pred = obj['lr_soft'][:, elem]
                pred_ent = met.predictive_entropy(torch_pred)
                expected_ent = met.expected_entropy(torch_pred)
                ee.append(expected_ent)
                pe.append(pred_ent)
                var_score = torch.sqrt(met.variance_score(torch_pred))
                mi = met.BALD(torch_pred)
                ekl = met.expected_kl(torch_pred)

                pr = torch_pred.cpu().detach().numpy()
                ax1, ax2 = axes.ravel()
                umg = img[elem].cpu().permute(1, 2, 0).numpy()
                ax1.imshow((umg * 255).astype(np.uint8))
                key_to_class[10] = 'unsure'
                n = key_to_class[cls]
                l = key_to_class[int(lbl[elem])]
                if n == 'unsure':
                    cls = torch.topk(lr_s.cpu(), 2)[1][-1]
                    n = key_to_class[int(cls)]
                    ax1.set_title(f'uncert pred: {n}\nlabel: {l}', loc='left')
                else:
                    ax1.set_title(f'pred: {n}\nlabel: {l}', loc='left')
                # ax2.errorbar(np.arange(9),pr,std,fmt='ok', lw=3)
                ids = [line.replace("_", " ") for line in list(key_to_class.values())]
                # ids.append('OOD')
                df = pd.DataFrame(pr, columns=ids)
                df_m = df.melt(var_name='class', value_name='probability')
                sns.violinplot(x='class', y='probability', data=df_m, ax=ax2, scale='width', linewidth=0.5,
                               palette="Set2", inner="stick")  # , showmeans=True, meanline=True)
                # ax2.violinplot(pr, showmeans=True)
                # ax2.set_xticks(np.arange(1, len(ids) + 1), ids)
                ax2.title.set_text(f'BP of outvec, σ: {var_score:.2f}, MI: {mi:.2f} ekl: {ekl:.2f}')
                ax2.set_ylim([-0.1, 1.1])
                # ax2.legend([n], bbox_to_anchor=(0.5, -.15))
                for label in ax2.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha('right')
                if name == 'vos':
                    ood_s = obj['lse_s'][elem].cpu().numpy()
                    fig.suptitle(
                        f'{name}, conf_c: {conf_class:.2f} μ: {ood_m.cpu().numpy():.2f}±{ood_s:.2f}')
                else:
                    fig.suptitle(f'{name}')
                plt.tight_layout()
                plt.savefig(f'figs/{name}_{i}_{elem}.png')
                plt.cla()
        # cm = metrics.confusion_matrix(np.concatenate(total_pred), np.concatenate(total_lbl), normalize='true')
        # disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm.round(decimals=2),
        #                                       display_labels=list(key_to_class.values()))
        # cm_plot = disp.plot()
        # cm_plot.figure_.suptitle(f'{name}_{accuracy:.2f}')
        # for label in cm_plot.ax_.get_xticklabels():
        #     label.set_rotation(45)
        #     label.set_ha('right')
        # for label in cm_plot.ax_.get_yticklabels():
        #     label.set_rotation(45)
        #     label.set_ha('right')
        # plt.show()
    all_lbls = np.concatenate(total_lbl, 0)
    if ds == 'cifar10':
        vals = np.array(hist)
        q25, q75 = np.percentile(vals, [25, 75])
        bin_width = 2 * (q75 - q25) * len(vals) ** (-1 / 3)
        bins = round((vals.max() - vals.min()) / bin_width)
        # g = sns.displot(np.array(hist), bins=bins * 2, kde=True, stat="density")
        # scale = model.vos_std.mean().cpu().numpy()
        # mu = model.vos_mean.mean().cpu().numpy()
        # x = np.linspace(max(0, mu - 5 * scale), mu + 5 * scale, 100)
        # plt.plot(x, scipy.stats.norm.pdf(x, mu, scale), color='red')
        # g.fig.subplots_adjust(top=.95)#, bottom=-0.05)
        # g.tight_layout()
        f = Fitter(vals, distributions=['norm'], bins=bins)
        f.fit()
        f.summary()
        # 16.2, 2.39
        plt.savefig(f'paper/kde_fit_tot.png')  # plt.show()
        #    plt.show()
        fits = {}
        for (k, v) in cls_hist.items():
            v = np.array(v)
            if v.shape[0] == 1:
                continue
            q25, q75 = np.percentile(v, [25, 75])
            bin_width = 2 * (q75 - q25) * len(v) ** (-1 / 3)
            bins = round((v.max() - v.min()) / bin_width) * 2
            # g = sns.displot(v, bins=bins * 3, kde=True, stat="density", height=5, label='kde')
            # g.set(xlabel='energy', ylabel='Density', title=f'Class: {id_to_cls[k]}'))
            x = np.linspace(min(v), max(v), 100)
            f = Fitter(v, distributions=['norm'], bins=bins)
            f.fit()
            f.summary()
            if k != 10:
                plt.plot(x, scipy.stats.norm.pdf(x, model.vos_means[k].cpu().numpy(), model.vos_stds[k].cpu().numpy()),
                         color='red')
            # plot_pdf_fit_cauchy(v)
            # plot_pdf_fit_lognorm(v)
            # g.fig.subplots_adjust(top=.95)#, bottom=-0.05)
            # g.tight_layout()
            fits[k] = f
            plt.title(f'label: {k}:{key_to_class[k]}, samples: {len(v)}')
            plt.show()
            # plt.savefig(f'paper/kde_fit_cls_{id_to_cls[k]}.png')  # plt.show()
        # {'norm': 0.04044779132076963, 'cauchy': 0.10192547940873295, 'lognorm': 0.04017927570199646}


        # plot roc

        ps, gg, tlbl = zip(*pred_soft)
        fps_s, tps_s, roc_auc_s, thresh_s = calc_tps_fps(list(zip(ps, gg)), tlbl)
        plot_mc_roc(fps_s, tps_s, roc_auc_s, 'soft')

        fps, tps, roc_auc, thresh = calc_tps_fps(pred_cert, all_lbls)
        plot_mc_roc(fps, tps, roc_auc, 'cert')

        cac = {}
        r = 21
        scale = 0.0125
        for i in range(r):
            v = i * scale
            if v == 1:
                v = 0.99
            cac[i] = []

            for x in range(all_lbls.shape[0]):
                m, c1 = pred_cert[x]
                if c1 > v:
                    if all_lbls[x] == np.argmax(m):
                        cac[i].append(1)
                    else:
                        cac[i].append(0)

        ls = np.array([[np.array(l).mean(), len(l)] for n, l in cac.items()])
        x_ax = np.arange(r) * scale
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_ax, 100 * (ls[:, 1] / ls[:, 1].max()), label='remaining samples')
        ax.fill_between(x_ax, 100 * (ls[:, 1] / ls[:, 1].max()), alpha=0.5)
        ax.plot(x_ax, 100 * ls[:, 0], color='red', label='accuracy')
        plt.legend()
        ax.set_ylim((75, 101))
        ax.set_xlabel('confidence')
        ax.set_ylabel('accuracy/remaining samples')
        import matplotlib.ticker as mtick
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        # plt.xlim((0.00, 0.11))
        ax.set_title("Accuracy vs. confidence of samples")
        plt.subplots_adjust(right=0.95)
        plt.savefig(f'paper/acc_conf_plot.png')  # plt.show()
        lls = {k: v.fitted_param['norm'] for k, v in fits.items()}
        print(lls)
    ap, cp, jp1, jp2, m1, m2 = [], [], [], [], [], []
    for x in range(all_lbls.shape[0]):
        m, c1, _ = pred_soft[x]
        _, c2 = pred_cert[x]
        if all_lbls[x] == np.argmax(m):
            ap.append(1)
            m1.append(c1)
            jp1.append(c2)
        else:
            ap.append(0)
            m2.append(c1)
            jp2.append(c2)
        cp.append(c2)
    roc_auc_r = sklearn.metrics.roc_auc_score(np.array(ap), np.array(cp))
    fpr_r, tpr_r, thresh_r = sklearn.metrics.roc_curve(ap, cp)
    m2 = np.array(m2)
    m1 = np.array(m1)
    if ds == "cifar10":
        print(f'accuracy: {accuracy:.4f}')
        show_performance_fpr(m1, m2)
        print(f's fpr@95: {fpr_tpr(m1, m2):.3f}')
        print(f'v fpr@95: {fpr_tpr(jp1, jp2):.3f}')
        v = (tpr_r >= 0.95).astype(float).nonzero()[0][0]
        print(f'@95% tpr: {tpr_r[v]}, fpr: {fpr_r[v]}, thresh: {thresh_r[v]}')

    return np.concatenate(oods, 0), np.concatenate(gss_oods,0)


# wrong_score.mean()
# -0.791616
# right_score.mean()
# -0.98615265

def VOS_CURVE(all_pred, all_lbls, model, name):
    all_incorrect = 0
    all_correct = 0
    for x in range(all_lbls.shape[0]):
        m, c1, c2 = all_pred[x]
        if c1 > 0.5:
            if all_lbls[x] == np.argmax(m):
                all_correct += 1
            else:
                all_incorrect += 1


def plot_pdf_fit_cauchy(data):
    x = np.linspace(max(0, data.min()), data.max(), 100)
    param = scipy.stats.cauchy.fit(data)
    plt.plot(x, scipy.stats.cauchy.pdf(x, *param), color='red', label='cauchy')
    plt.grid(True)
    plt.legend()


def plot_pdf_fit_lognorm(data):
    x = np.linspace(max(0, data.min()), data.max(), 100)
    param = scipy.stats.lognorm.fit(data, floc=0)
    plt.plot(x, scipy.stats.lognorm.pdf(x, *param), color='green', label='log normal')
    plt.grid(True)
    plt.legend()


def plot_mc_roc(fpr, tpr, roc_auc, name=''):
    n_classes = len(fpr.keys()) - 1
    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    # colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            # color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.axhline(y=0.95, linestyle="--", color='r')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"multiclass ROC {name}")
    plt.legend(loc="lower right")
    plt.show()


def calc_tps_fps(pred_cert, all_lbls):
    res_dict = {}

    for i, x in enumerate(pred_cert):
        cls = int(all_lbls[i])
        if cls in res_dict:
            if cls == np.argmax(x[0]):
                res_dict[cls].append((1, x[1]))
            else:
                res_dict[cls].append((0, x[1]))
        else:
            if cls == np.argmax(x[0]):
                res_dict[cls] = [(1, x[1])]
            else:
                res_dict[cls] = [(0, x[1])]

    tps, fps, roc_aucs, threshs = {}, {}, {}, {}
    p, s = [], []
    for x in res_dict.keys():
        preds_score = np.array(res_dict[x], dtype=np.float32)
        fps[x], tps[x], threshs[x] = sklearn.metrics.roc_curve(preds_score[:, 0], preds_score[:, 1])
        roc_aucs[x] = sklearn.metrics.auc(fps[x], tps[x])
        p.append(preds_score[:, 0])
        s.append(preds_score[:, 1])
    pp = np.concatenate(p)
    ss = np.concatenate(s)
    fps["micro"], tps["micro"], threshs["micro"] = sklearn.metrics.roc_curve(pp, ss)
    roc_aucs["micro"] = sklearn.metrics.auc(fps["micro"], tps["micro"])
    return fps, tps, roc_aucs, threshs


def draw_ftpr(i, elem, elem_s):
    thresh, fpr, tpr = elem
    thresh_s, fpr_s, tpr_s = elem_s
    t = np.concatenate((thresh[i], [0]))
    ts = np.concatenate((thresh_s[i], [0]))
    tpr_s = np.concatenate((tpr_s[i], [1]))
    tpr = np.concatenate((tpr[i], [1]))
    fpr_s = np.concatenate((fpr_s[i], [1]))
    fpr = np.concatenate((fpr[i], [1]))

    plt.plot(ts[1:], tpr_s[1:], color='blue')
    plt.plot(t[1:], tpr[1:], color='red')
    plt.plot(ts[1:], fpr_s[1:], '--', color='blue')
    plt.plot(t[1:], fpr[1:], '--', color='red')
    plt.legend(['tpr_s', 'tpr', 'fpr_s', 'fpr'])
    plt.xlabel('conf')
    plt.ylabel(r'rate')
    plt.show()


if __name__ == "__main__":
    path = r'Q:\uncert_data\data_cifar_cleaned'
    cps = []
    test = ["LSUN_C", 'SVHN', 'dtd', "iSUN", "LSUN_resize","places365"]
    #test = ["iSUN", "LSUN_resize"]
    # try:
    #     iid1 = np.fromfile("iid1", dtype=np.float32)
    #     iid2 = np.fromfile("iid2", dtype=np.float32)
    # except FileNotFoundError:
    iid1, iid2 = run_net(path, name='vos', ds='cifar10', max_iter=10000)
    iid1.tofile("iid1")
    iid2.tofile("iid2")
    vos, gss = [], []
    for dataset in test:
        ood1, ood2 = run_net(path, name='vos', ds=dataset)
        vos.append(fpr_tpr(iid1, ood1))
        gss.append(fpr_tpr(iid2, ood2))
        #gss.append(fpr_tpr(iid2, ood2))
        print(f'dsvos: {vos[-1]}')
        print(f'gauss: {gss[-1]}')
        print(f'mean: {ood1.mean()}')

    for x in range(len(vos)):
        print(f"-----{test[x]}-----")
        print(f'dsvos: {vos[x]}')
        print(f'gauss: {gss[x]}')
    print(f'mean ds vos: {np.array(vos).mean()}')
    print(f'mean ds gss: {np.array(gss).mean()}')
    print('done')

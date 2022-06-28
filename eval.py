import csv
import os
import fiftyone as fo
import higher as higher
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
import sklearn.metrics as metrics
import torch.onnx


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


def run_net(root_dir, ra, epochs=25, name=''):
    val_dir = os.path.join(root_dir, 'val_set')
    data = r'Q:\git\SLC\ckpts'
    image_size = 128
    batch_size = 32
    workers = 4
    model_name = "mobilenet_v3"
    torch.manual_seed(5)
    val_set = ShippingLabClassification(root_dir=val_dir,
                                        transform=transforms.Compose([
                                            letterbox((image_size, image_size)),
                                            transforms.ToTensor()
                                        ]))

    v_dataloader = DataLoader(val_set, batch_size=batch_size,
                              shuffle=True, num_workers=workers)
    id_to_cls = {b: a for a, b in val_set.classes.items()}
    n_classes = len(val_set.classes.keys())
    if name == 'bayes':
        model = BayesVGG16(n_classes=n_classes)
    elif name == 'dropout':
        model = DropoutModel(n_classes=n_classes, model_name=model_name)
    elif name == 'vos':
        model = VOSModel(n_classes=n_classes, model_name=model_name)

    path = os.path.join(data, model_name + "_" + name + "_78.pt")

    model_dict = torch.load(os.path.join(root_dir, path))
    model.load_state_dict(model_dict['model_state_dict'])

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    net_losses, accuracy_log = [], []
    acc, acc_u, acc_l, stds, acc_before_mean, t_t_acc = [], [], [], [], [], []
    t_std, t_pred, t_acc = [], [], []
    last_epoch = []
    total_pred = []
    total_lbl = []
    ood_acc = []
    idxss = []
    pred_cert = []
    cls_hist = {}
    hist = []
    with torch.no_grad():
        tqd_e = tqdm(enumerate(v_dataloader, 0), total=len(v_dataloader))
        fig, axes = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})
        for i, data in tqd_e:
            t_imgs, t_lbls, _, p = data
            t_imgs = t_imgs.cuda()
            t_lbls = t_lbls.cuda()
            if name == 'vos':
                obj = model.evaluate_classification(t_imgs, samples=10, std_multiplier=2)
            else:
                obj = model.evaluate_classification(t_imgs, samples=10, std_multiplier=2)

            # accuracy of cert
            exclude_lst = []
            # for elem in range(len(show_obj["mean"])):
            #     pr = show_obj["preds"][:, elem].cpu().numpy()
            #     mean = show_obj["mean"][elem].cpu().numpy()
            #     std = show_obj["stds"][elem].cpu().numpy()
            #     cls = np.argmax(mean)
            #     if name == 'vos':
            #         ood_m = torch.logsumexp(show_obj['lse_m'][elem], 0).cpu().numpy()
            #         conf = model.cdf(ood_m)
            #         conf_class = model.cdf_class(ood_m, np.argmax(mean))
            #     if conf_class <= 0.95:
            #         exclude_lst.append(elem)

            predicted = torch.argmax(obj['sp'], dim=1)
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
            show_obj = obj
            img, lbl = show_epoch
            f = open("wrong_train.txt", "a")
            for elem in range(len(show_obj["mean"])):
                pr = show_obj["preds"][:, elem].cpu().numpy()
                mean = show_obj["mean"][elem].cpu().numpy()
                std = show_obj["stds"][elem].cpu().numpy()
                cls = np.argmax(mean)
                if name == 'vos':
                    ood_m = show_obj['lse_m'][elem]
                    conf = model.cdf(ood_m)
                    conf_class = model.cdf_class(ood_m, cls)
                    if cls in cls_hist:
                        cls_hist[cls].append(ood_m.cpu().numpy())
                    else:
                        cls_hist[cls] = [ood_m.cpu().numpy()]
                    hist.append(ood_m.cpu().numpy())
                pred_cert.append((mean, conf_class, conf))
                if cls == lbl[elem]:
                    continue
                f.write(f'{p[elem]},{cls},{lbl[elem]}\n')
                if conf_class >= 0.01:
                    continue
                # if [elem, i] not in idx_to_check:
                #     continue
                ax1, ax2 = axes.ravel()
                ax1.imshow(img.cpu()[elem].permute(1, 2, 0).numpy())
                n = id_to_cls[cls]
                l = id_to_cls[int(lbl[elem])]
                # ax1.title.set_text(f'pred:  {n}\nlabel: {l}')
                if conf_class <= 0.005:
                    ax1.set_title(f'pred: OOD\nlabel: {l}', loc='left')
                else:
                    ax1.set_title(f'pred: {n}\nlabel: {l}', loc='left')
                # ax2.errorbar(np.arange(9),pr,std,fmt='ok', lw=3)
                ids = [line.replace("_", " ") for line in list(val_set.classes.keys())]
                ax2.boxplot(pr, labels=ids, showmeans=True, meanline=True)
                ax2.title.set_text(f'Boxplot of class prediciton')
                ax2.set_ylim([-0.1, 1.1])
                # ax2.legend([n], bbox_to_anchor=(0.5, -.15))
                for label in ax2.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha('right')
                if name == 'vos':
                    ood_m = torch.logsumexp(show_obj['lse_m'][elem], 0)
                    ood_s = torch.logsumexp(show_obj['lse_s'][elem], 0).cpu().numpy()
                    conf = model.cdf(ood_m)
                    conf_class = model.cdf_class(ood_m, np.argmax(mean))
                    fig.suptitle(
                        f'{name}, conf:{conf:.2f}, conf_c: {conf_class:.2f} μ: {ood_m.cpu().numpy():.2f}±{ood_s:.2f}')
                else:
                    fig.suptitle(f'{name}')
                plt.tight_layout()
                plt.savefig(f'figs/{name}_{i}_{elem}.png')
                plt.cla()
            f.close()
        cm = metrics.confusion_matrix(np.concatenate(total_pred), np.concatenate(total_lbl), normalize='true')
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm.round(decimals=2),
                                              display_labels=list(val_set.classes.keys()))
        cm_plot = disp.plot()
        cm_plot.figure_.suptitle(f'{name}_{accuracy:.2f}')
        for label in cm_plot.ax_.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
        for label in cm_plot.ax_.get_yticklabels():
            label.set_rotation(45)
            label.set_ha('right')
        plt.show()

    import seaborn as sns

    vals = np.array(hist)
    q25, q75 = np.percentile(vals, [25, 75])
    bin_width = 2 * (q75 - q25) * len(vals) ** (-1 / 3)
    bins = round((vals.max() - vals.min()) / bin_width)
    g = sns.displot(np.array(hist), bins=bins * 2, kde=True, stat="density", height=5)
    sigma = model.vos_std.mean().cpu().numpy()
    mu = model.vos_mean.mean().cpu().numpy()
    x = np.linspace(max(0, mu - 5 * sigma), mu + 5 * sigma, 100)
    plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='red')
    # g.fig.subplots_adjust(top=.95)#, bottom=-0.05)
    g.tight_layout()
    plt.savefig(f'paper/kde_fit_tot.png')  # plt.show()
#    plt.show()

    for (k, v) in cls_hist.items():
        v = np.array(v)
        q25, q75 = np.percentile(v, [25, 75])
        bin_width = 2 * (q75 - q25) * len(v) ** (-1 / 3)
        bins = round((v.max() - v.min()) / bin_width)
        g = sns.displot(np.array(hist), bins=bins * 3, kde=True, stat="density", height=5, label='kde')
        g.set(xlabel='energy', ylabel='Density', title=f'Class: {id_to_cls[k]}')
        sigma = model.vos_stds[k].cpu().numpy()
        mu = model.vos_means[k].cpu().numpy()
        x = np.linspace(max(0, mu - 5 * sigma), mu + 5 * sigma, 100)
        plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='red', label='trained gauss')

        # g.fig.subplots_adjust(top=.95)#, bottom=-0.05)
        g.tight_layout()
        # plt.title(f'{k}')
        plt.savefig(f'paper/kde_fit_cls_{id_to_cls[k]}.png')#plt.show()

    all_lbls = np.concatenate(total_lbl, 0)
    cac = {}
    r = 21
    scale = 0.0125
    for i in range(r):
        v = i * scale
        if v == 1:
            v = 0.99
        cac[i] = []
        for x in range(all_lbls.shape[0]):
            m, c1, c2 = pred_cert[x]
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
    plt.savefig(f'paper/acc_conf_plot.png')#plt.show()

    print(ls)


if __name__ == "__main__":
    path = r'Q:\uncert_data\data_cifar_cleaned'
    for x in ['vos']:  # , 'dropout']:
        run_net(path, False, name=x)

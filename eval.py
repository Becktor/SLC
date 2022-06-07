import csv
import os
import fiftyone as fo
import higher as higher
import timm
import torch
import ttach as tta
from torch.utils.data import Dataset, DataLoader
from DataLoader import ShippingLabClassification
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import itertools
import wandb
from models import GluonResnext50, BayesGluonResnext50, DropoutGluonResnext50, TTAModel, BayesVGG11, VOSGluonResnext50
import torch.nn as nn
from pathlib import Path
import random
import sklearn.metrics as metrics


class TransformWrapper(object):
    def __init__(self, transform, n=1):
        self.trans = transform
        self.n = n

    def __call__(self, img):
        imgs = [self.trans(img) for _ in range(self.n)]
        return imgs


idx_to_check = [[29, 1], [29, 1], [27, 3], [27, 3], [24, 4], [24, 4], [14, 5], [14, 5], [25, 6], [25, 6], [19, 14], [19, 14], [12, 18], [12, 18], [17, 18], [17, 18], [14, 20], [14, 20], [14, 23], [14, 23], [4, 24], [4, 24], [22, 24], [22, 24], [30, 24], [30, 24], [9, 25], [9, 25], [10, 25], [10, 25], [16, 25], [16, 25], [23, 25], [23, 25], [5, 26], [5, 26], [6, 28], [6, 28], [7, 28], [7, 28], [8, 28], [8, 28], [9, 28], [9, 28], [28, 34], [28, 34], [28, 37], [28, 37], [24, 39], [25, 39], [29, 39], [30, 39], [30, 39], [31, 39], [1, 40], [2, 40], [28, 40], [22, 56], [23, 56], [18, 58], [28, 73], [29, 73], [29, 73], [30, 73], [3, 74], [4, 74], [10, 82], [11, 82], [25, 83], [26, 83], [13, 84], [14, 84], [3, 85], [4, 85], [18, 85], [16, 88], [17, 88], [20, 93], [21, 93], [21, 93], [22, 93], [24, 98], [25, 98], [25, 98], [26, 98], [17, 99], [18, 99], [21, 99], [22, 99], [28, 99], [29, 99], [13, 101], [14, 101], [1, 104], [2, 104], [26, 104], [27, 104], [28, 105], [29, 105], [12, 108], [6, 113], [7, 113], [22, 113], [30, 119], [31, 119], [25, 120], [26, 120], [30, 121], [31, 121], [2, 122], [16, 122], [17, 122], [28, 123], [29, 123], [19, 124], [20, 124], [2, 129], [3, 129], [15, 137], [16, 137], [8, 143], [9, 143]]

def run_net(root_dir, ra, epochs=25, name=''):
    val_dir = os.path.join(root_dir, 'val_set')
    data = r'Q:\git\SLC\ckpts'
    image_size = 128
    batch_size = 32
    workers = 2
    model_name = 'vgg11'
    torch.manual_seed(1)
    val_set = ShippingLabClassification(root_dir=val_dir,
                                        transform=transforms.Compose([
                                            transforms.Resize(image_size, max_size=int(image_size*1.2)),
                                            transforms.CenterCrop((image_size, image_size)),
                                            transforms.ToTensor()
                                        ]))

    v_dataloader = DataLoader(val_set, batch_size=batch_size,
                              shuffle=False, num_workers=workers)

    n_classes = len(val_set.classes.keys())
    if name == 'bayes':
        model = BayesVGG11(n_classes=n_classes)
    elif name == 'mc_dropout':
        model = DropoutGluonResnext50(n_classes=n_classes, model_name=model_name)
    elif name == 'vos':
        model = VOSGluonResnext50(n_classes=n_classes, model_name=model_name)
    else:
        model = TTAModel(n_classes=n_classes, model_name=model_name)

    path = os.path.join(data, model_name + "_" + name + "_100.pt")
    if name == 'tta':
        path = os.path.join(data, model_name + "_mc_dropout_100.pt")

    model_dict = torch.load(os.path.join(root_dir, path))
    model.load_state_dict(model_dict['model_state_dict'])
    if name == 'tta':
        tta_transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.Rotate90(angles=[0, 180]),
                tta.Scale(scales=[1, 2]),
                tta.Multiply(factors=[0.9, 1, 1.1]),
                tta.FiveCrops(crop_height=100, crop_width=100),
            ]
        )
        t_t = [transforms.FiveCrop((100, 100)),
               TransformWrapper(transforms.ColorJitter(.5, .3), n=5),
               TransformWrapper(transforms.RandomRotation(degrees=(-30, 30)), n=5),
               TransformWrapper(transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), n=1),
               TransformWrapper(transforms.RandomHorizontalFlip(p=1), n=1)
               ]

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
    with torch.no_grad():
        tqd_e = tqdm(enumerate(v_dataloader, 0), total=len(v_dataloader))
        fig, axes = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})
        for i, data in tqd_e:
            t_imgs, t_lbls, _, _ = data
            t_imgs = t_imgs.cuda()
            t_lbls = t_lbls.cuda()
            if name == 'tta':
                obj = model.evaluate_classification(t_imgs, t_t)
            if name == 'vos':
                obj = model.evaluate_classification(t_imgs, samples=10, std_multiplier=2)
            else:
                obj = model.evaluate_classification(t_imgs, samples=10, std_multiplier=2)

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
            #if i in [25, 75, 100, len(v_dataloader) - 1]:
            show_epoch = (t_imgs, t_lbls)
            show_obj = obj
            img, lbl = show_epoch
            for elem in range(len(show_obj["mean"])):
                pr = show_obj["preds"][:, elem].cpu().numpy()
                mean = show_obj["mean"][elem].cpu().numpy()
                std = show_obj["stds"][elem].cpu().numpy()
                # if name == 'vos':
                #     ood_m = torch.logsumexp(show_obj['lse_m'][elem], 0).cpu().numpy()
                #     conf = model.cdf(ood_m)
                #     conf_class = model.cdf_class(ood_m, np.argmax(mean))
                if [elem, i] not in idx_to_check:
                    continue
                ax1, ax2 = axes.ravel()
                ax1.imshow(img.cpu()[elem].permute(1, 2, 0).numpy())
                n = list(val_set.classes.keys())[np.argmax(mean)]
                l = list(val_set.classes.keys())[lbl[elem]]
                # ax1.title.set_text(f'pred:  {n}\nlabel: {l}')
                ax1.set_title(f'pred: {n}\nlabel: {l}', loc='left')
                std_l = list(val_set.classes.keys())[np.argmax(std)]
                std_l = list(val_set.classes.keys())[np.argmax(std)]
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
                    ood_m = torch.logsumexp(show_obj['lse_m'][elem], 0).cpu().numpy()
                    ood_s = torch.logsumexp(show_obj['lse_s'][elem], 0).cpu().numpy()
                    conf = model.cdf(ood_m)
                    conf_class = model.cdf_class(ood_m, np.argmax(mean))
                    fig.suptitle(f'{name}, conf:{conf:.2f}, conf_c: {conf_class:.2f} μ: {ood_m:.2f}±{ood_s:.2f}')
                else:
                    fig.suptitle(f'{name}')
                plt.tight_layout()
                plt.savefig(f'figs/{name}_{i}_{elem}.png')
                plt.cla()
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
    print(idxss)


if __name__ == "__main__":
    path = r'Q:\uncert_data\ds1_wo_f'
    for x in ['mc_dropout']:
        run_net(path, False, name=x)

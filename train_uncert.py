import csv
import os
import fiftyone as fo
import higher as higher
import timm
import torch
from torch.utils.data import Dataset, DataLoader
from DataLoader import ShippingLabClassification
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import itertools
import wandb
from models import GluonResnext50, BayesGluonResnext50, DropoutGluonResnext50
import torch.nn as nn
from pathlib import Path
torch.manual_seed(0)


def run_net(root_dir, ra, epochs=25, net_method=''):
    val_dir = os.path.join(root_dir, 'val_set')
    train_dir = os.path.join(root_dir, 'train_set')
    image_size = 128
    batch_size = 16
    epochs = epochs
    wandb.init(
        project="Uncert_course",
        config={
            "learning_rate": 0.001,
            "gamma": 0.1,
            "batch_size": batch_size,
            "image_size": image_size,
            "total_epochs": epochs,
            "ra": ra,
            "rd": Path(root_dir).name,
            "model_name": "mobilenetv3_small_050",
            "method": net_method
        },
    )
    wandb.run.name = wandb.config.method
    model_name = wandb.config.model_name

    workers = 2
    dataset = ShippingLabClassification(root_dir=train_dir,
                                        transform=transforms.Compose([
                                            transforms.Resize(image_size),
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            transforms.RandomHorizontalFlip()
                                        ]))

    val_set = ShippingLabClassification(root_dir=val_dir,
                                        transform=transforms.Compose([
                                            transforms.Resize(image_size),
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            transforms.RandomHorizontalFlip()
                                        ]))

    key_to_class = dict((v, k) for k, v in dataset.classes.items())

    t_dataloader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=True, num_workers=workers)

    v_dataloader = DataLoader(val_set, batch_size=batch_size,
                              shuffle=True, num_workers=workers)

    n_classes = len(dataset.classes.keys())
    if wandb.config.method == 'bayes':
        model = BayesGluonResnext50(n_classes=n_classes, model_name=model_name)
    else:
        model = DropoutGluonResnext50(n_classes=n_classes, model_name=model_name)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    wandb.watch(model)
    opt = torch.optim.RAdam(model.parameters(), lr=wandb.config.learning_rate)
    net_losses, accuracy_log = [], [],
    for x in range(epochs):
        model.train()
        net_l, meta_losses_clean, m_cross, m_reg = [], [], [], []
        tqdm_dl = tqdm(t_dataloader)

        for i, data in enumerate(tqdm_dl, 0):
            imgs, lbls, idxs, paths = data
            imgs = imgs.cuda()
            lbls = lbls.cuda()

            cost = model.loss(imgs, lbls)

            opt.zero_grad()
            cost.backward()
            opt.step()

            net_l.append(cost.cpu().detach())
            tqdm_dl.set_description(f"e:{x} -- mean Loss: {np.mean(net_l):.4f} currloss: {cost:.4f} ")

        net_losses.append(np.mean(net_l))
        acc, acc_u, acc_l = [], [], []
        with torch.no_grad():
            for i, data in enumerate(v_dataloader, 0):
                t_imgs, t_lbls, _, _ = data
                t_imgs = t_imgs.cuda()
                t_lbls = t_lbls.cuda()
                obj = model.evaluate_classification(t_imgs, t_lbls, samples=10, std_multiplier=2)
                predicted = torch.argmax(obj['sp'], dim=1)
                acc.append((predicted.int() == t_lbls.int()).float())
                accuracy = torch.cat(acc, dim=0).mean().cpu()
                predicted_upper = torch.argmax(obj['sp_u'], dim=1)
                acc_u.append((predicted_upper.int() == t_lbls.int()).float())
                accuracy_u = torch.cat(acc, dim=0).mean().cpu()
                predicted_lower = torch.argmax(obj['sp_l'], dim=1)
                acc_l.append((predicted_lower.int() == t_lbls.int()).float())
                accuracy_l = torch.cat(acc, dim=0).mean().cpu()

                print(f'runn acc = {accuracy:.3f}, runn acc_u = {accuracy_u:.3f}, runn acc_l = {accuracy_l:.3f}',
                      end='\r')
                # tqdm_dl.set_postfix(f"mean Loss {}")

        accuracy = torch.cat(acc, dim=0).mean().cpu()
        accuracy_u = torch.cat(acc_u, dim=0).mean().cpu()
        accuracy_l = torch.cat(acc_l, dim=0).mean().cpu()
        print(f'mean val_acc = {accuracy:.3f}')
        accuracy_log.append(np.array([x, accuracy])[None])
        log_dict = {
            "val/acc": float(accuracy),
            "val/acc_u": float(accuracy_u),
            "val/acc_l": float(accuracy_l),
            "train/running_loss": np.mean(net_l),
            "train/meta_epoch_loss": np.mean(meta_losses_clean),
        }

        #if x % 5 == 0 and x > 0:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax1, ax2 = axes.ravel()
        fig.suptitle(f"norm w. Prob")
        ax1.plot(net_losses, label='net_losses')
        ax1.set_ylabel("Losses")
        ax1.set_xlabel("Iteration")
        ax1.legend()
        acc_log = np.concatenate(accuracy_log, axis=0)
        ax2.plot(acc_log[:, 0], acc_log[:, 1])
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Iteration')
        plt.show()

        torch.save({
            'epoch': x,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': np.mean(net_l),
            'classes': key_to_class,

        }, f"ckpts/{model_name}_{wandb.config.method}_{x}.pt")

        wandb.log(log_dict)
    wandb.finish()


if __name__ == "__main__":
    path = r'D:\Data\sl_class'
    for x in ['mcmc_dropout', 'bayes']:
        run_net(path, False, net_method=x)

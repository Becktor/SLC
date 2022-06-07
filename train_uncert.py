import csv
import os
import traceback

import fiftyone as fo
import higher as higher
import timm
import torch
from torch.utils.data import Dataset, DataLoader
from DataLoader import ShippingLabClassification, letterbox
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import itertools
import wandb
from models import BayesVGG16, BayesGluonResnext50, DropoutModel, TTAModel, VOSModel
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from vos import vos_update
import torchvision.transforms.functional as TF
from torch.utils.data.dataloader import default_collate
from torchinfo import summary
torch.manual_seed(0)
from itertools import cycle


def run_net(root_dir, ra, epochs=100, net_method='', lr=1e-3, batch_size=64):
    try:
        torch.cuda.empty_cache()
        val_dir = os.path.join(root_dir, 'val_set')
        train_dir = os.path.join(root_dir, 'train_set')
        image_size = 128
        epochs = epochs
        wandb.init(
            project="Uncert_paper",
            config={
                "learning_rate": lr,
                "gamma": 0.1,
                "batch_size": batch_size,
                "image_size": image_size,
                "total_epochs": epochs,
                "ra": ra,
                "rd": Path(root_dir).name,
                "model_name": "efficientnet_es",
                "method": net_method
            },
        )
        wandb.run.name = wandb.config.method
        model_name = wandb.config.model_name

        workers = 4

        dataset = ShippingLabClassification(root_dir=train_dir,
                                            transform=transforms.Compose([
                                                letterbox((image_size, image_size)),
                                                transforms.ToTensor(),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ColorJitter(),
                                            ]))

        val_set = ShippingLabClassification(root_dir=val_dir,
                                            transform=transforms.Compose([
                                                letterbox((image_size, image_size)),
                                                transforms.ToTensor()
                                            ]))

        key_to_class = dict((v, k) for k, v in dataset.classes.items())

        t_dataloader = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=True)

        v_dataloader = DataLoader(val_set, batch_size=batch_size,
                                  shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=True)

        n_classes = len(dataset.classes.keys())
        if wandb.config.method == 'bayes':
            model = BayesVGG16(n_classes=n_classes)
        elif wandb.config.method == 'dropout':
            model = DropoutModel(n_classes=n_classes, model_name=model_name)
        elif wandb.config.method == 'vos':
            model = VOSModel(n_classes=n_classes, model_name=model_name)
        else:
            model = TTAModel(n_classes=n_classes, model_name=model_name)
        if torch.cuda.is_available():
            model.cuda()
            torch.backends.cudnn.benchmark = True
        if wandb.config.method == 'vos':
            weight_energy = torch.nn.Linear(n_classes, 1).cuda()
            torch.nn.init.uniform_(weight_energy.weight)
            eye_matrix = torch.eye(128, device='cuda')
            logistic_regression = torch.nn.Linear(1, 2)
            logistic_regression = logistic_regression.cuda()
            model_param = [x for x in list(model.parameters()) if x.requires_grad==True]
            params = model_param + list(weight_energy.parameters()) + list(logistic_regression.parameters())
            opt = torch.optim.RAdam(params, lr=wandb.config.learning_rate)
            number_dict = {}
            sample_number = 1000
            sample_from = 10000
            select = 1
            data_dict = torch.zeros(n_classes, sample_number, 128).cuda()
            for i in range(n_classes):
                number_dict[i] = 0
            vos_dict = {
                'logistic_regression': logistic_regression,
                'eye_matrix': eye_matrix,
                'weight_energy': weight_energy,
                'number_dict': number_dict,
                'sample_number': sample_number,
                'sample_from': sample_from,
                'select': select,
                'loss_weight': 0.01,
                'data_dict': data_dict,
                'start_epoch': 20}
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)

        net_losses, accuracy_log = [], [],

        wandb.watch(model, log="all", log_freq=50)
        summary(model, (batch_size, 3, image_size, image_size))
        model.train()
        for epoch in range(epochs):
            net_l, meta_losses_clean, m_cross, m_reg, vos_l = [], [], [], [], []
            tqdm_dl = tqdm(range(200))

            generator = cycle(iter(t_dataloader))
            for i, data in enumerate(tqdm_dl, 0):
                # Samples the batch
                imgs, lbls, idxs, paths = next(generator)
                imgs = imgs.cuda()
                lbls = lbls.cuda()
                opt.zero_grad()

                if wandb.config.method == 'vos':
                    model.at_epoch = epoch
                    #with torch.cuda.amp.autocast():
                    pred, output = model(imgs)
                    vos_dict["num_classes"] = n_classes
                    vos_dict['pred'] = pred
                    vos_dict["output"] = output
                    vos_dict["epoch"] = epoch
                    vos_dict["target"] = lbls
                    vos_loss = vos_update(model, vos_dict)
                    loss = model.loss(pred, lbls)
                    cost = loss + vos_dict['loss_weight'] * vos_loss

                else:
                    #with torch.cuda.amp.autocast():
                    output = model(imgs)
                    cost = model.loss(output, lbls)
                #with torch.cuda.amp.autocast():
                cost.backward()

                opt.step()
                net_l.append(cost.cpu().detach())

                tqdm_dl.set_description(f"e:{epoch} -- mean Loss: {np.mean(net_l):.4f} currloss: {cost:.4f} ")
                if wandb.config.method == 'vos':
                    vos_l.append(vos_loss.cpu().detach())
                    tqdm_dl.set_description(f"e:{epoch} -- mL: {np.mean(net_l):.4f} cL: {cost:.4f}, mL: {np.mean(vos_l):.4f}")

                if i % 10 == 0:
                    loss_dict = {'train/running_loss': np.mean(net_l[-10:])}
                    if wandb.config.method == 'vos':
                        loss_dict['train/vos_loss'] = np.mean(vos_l[-10:])
                        wandb.log(loss_dict)

            net_losses.append(np.mean(net_l))
            acc, acc_u, acc_l = [], [], []
            if (1+epoch) % 2 == 0:
                with torch.no_grad():
                    pbar = tqdm(enumerate(v_dataloader, 0))
                    for i, data in pbar:
                        t_imgs, t_lbls, _, _ = data
                        t_imgs = t_imgs.cuda()
                        t_lbls = t_lbls.cuda()
                        #with torch.cuda.amp.autocast():
                        obj = model.evaluate_classification(t_imgs, samples=5, std_multiplier=2)
                        predicted = torch.argmax(obj['sp'], dim=1)
                        acc.append((predicted.int() == t_lbls.int()).float())
                        accuracy = torch.cat(acc, dim=0).mean().cpu()
                        predicted_upper = torch.argmax(obj['sp_u'], dim=1)
                        acc_u.append((predicted_upper.int() == t_lbls.int()).float())
                        accuracy_u = torch.cat(acc, dim=0).mean().cpu()
                        predicted_lower = torch.argmax(obj['sp_l'], dim=1)
                        acc_l.append((predicted_lower.int() == t_lbls.int()).float())
                        accuracy_l = torch.cat(acc, dim=0).mean().cpu()
                        pbar.set_description(f'runn acc = {accuracy:.3f}, runn acc_u = {accuracy_u:.3f}, runn acc_l = {accuracy_l:.3f}')

                        # tqdm_dl.set_postfix(f"mean Loss {}")

                accuracy = torch.cat(acc, dim=0).mean().cpu()
                accuracy_u = torch.cat(acc_u, dim=0).mean().cpu()
                accuracy_l = torch.cat(acc_l, dim=0).mean().cpu()
                print(f'mean val_acc = {accuracy:.3f}')
                accuracy_log.append(np.array([epoch, accuracy])[None])
                log_dict = {
                    "val/acc": float(accuracy),
                    "val/acc_u": float(accuracy_u),
                    "val/acc_l": float(accuracy_l),
                    "train/epoch_loss": np.mean(net_l),
                    "epoch/epoch": epoch,
                }
                if len(meta_losses_clean) != 0:
                    log_dict["train/meta_loss_clean"] = np.mean(meta_losses_clean)
                # if x % 5 == 0 and x > 0:
                fig, axes = plt.subplots(1, 2, figsize=(13, 5))
                ax1, ax2 = axes.ravel()
                fig.suptitle(f"loss and accuracy for model {wandb.config.method}")
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
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': np.mean(net_l),
                    'classes': key_to_class,


                }, f"ckpts/{model_name}_{wandb.config.method}_{1+epoch}.pt")


                wandb.log(log_dict)
                if wandb.config.method == 'vos':
                    cn = torch.stack([(x != 0).sum() for x in model.running_means]).sum()/(len(model.running_means))
                    print(f'len: {cn}, μ: {model.vos_mean.mean()}, σ: {model.vos_std.mean()}')
        wandb.finish()

    except Exception as e:
        wandb.log({'error': traceback.format_exc()})
        wandb.log({'except': e})
        wandb.finish()


if __name__ == "__main__":
    path = r'Q:\uncert_data\ds1_wo_f'
    for name, z in zip(['vos', 'dropout', 'bayes'], [1e-4, 1e-4, 1e-2]):
        run_net(path, False, net_method=name, lr=z)

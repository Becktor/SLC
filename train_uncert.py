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
from models import GluonResnext50, BayesGluonResnext50, DropoutGluonResnext50, TTAModel, VOSGluonResnext50
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F

torch.manual_seed(0)


def run_net(root_dir, ra, epochs=100, net_method='', lr=1e-3, batch_size=64):
    val_dir = os.path.join(root_dir, 'val_set')
    train_dir = os.path.join(root_dir, 'train_set')
    image_size = 128
    epochs = epochs
    wandb.init(
        project="Uncert_course",
        config={
            "learning_rate": lr,
            "gamma": 0.1,
            "batch_size": batch_size,
            "image_size": image_size,
            "total_epochs": epochs,
            "ra": ra,
            "rd": Path(root_dir).name,
            "model_name": "vgg11",
            "method": net_method
        },
    )
    wandb.run.name = wandb.config.method
    model_name = wandb.config.model_name

    workers = 2
    dataset = ShippingLabClassification(root_dir=train_dir,
                                        transform=transforms.Compose([
                                            transforms.Resize((image_size, image_size)),
                                            # transforms.RandomCrop(image_size),
                                            transforms.ToTensor(),
                                            transforms.RandomHorizontalFlip()
                                        ]))

    val_set = ShippingLabClassification(root_dir=val_dir,
                                        transform=transforms.Compose([
                                            transforms.Resize((image_size, image_size)),
                                            # transforms.RandomCrop(image_size),
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
    elif wandb.config.method == 'dropout':
        model = DropoutGluonResnext50(n_classes=n_classes, model_name=model_name)
    elif wandb.config.method == 'vos':
        model = VOSGluonResnext50(n_classes=n_classes, model_name=model_name)
    else:
        model = TTAModel(n_classes=n_classes, model_name=model_name)
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    wandb.watch(model)

    if wandb.config.method == 'vos':
        weight_energy = torch.nn.Linear(n_classes, 1).cuda()
        torch.nn.init.uniform_(weight_energy.weight)
        eye_matrix = torch.eye(128, device='cuda')
        logistic_regression = torch.nn.Linear(1, 2)
        logistic_regression = logistic_regression.cuda()
        params = list(model.parameters()) + list(weight_energy.parameters()) + list(logistic_regression.parameters())
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
            'start_epoch': 10}
    else:
        opt = torch.optim.RAdam(model.parameters(), lr=wandb.config.learning_rate)

    net_losses, accuracy_log = [], [],
    for epoch in range(epochs):
        model.train()
        net_l, meta_losses_clean, m_cross, m_reg = [], [], [], []
        tqdm_dl = tqdm(t_dataloader)

        for i, data in enumerate(tqdm_dl, 0):
            imgs, lbls, idxs, paths = data
            imgs = imgs.cuda()
            lbls = lbls.cuda()

            if wandb.config.method == 'vos':
                pred, output = model.forward_vos(imgs)
                vos_dict["num_classes"] = n_classes
                vos_dict['pred'] = pred
                vos_dict["output"] = output
                vos_dict["epoch"] = epoch
                vos_dict["target"] = lbls
                vos_loss = do_vos(model, vos_dict)
                loss = F.cross_entropy(pred, lbls)
                cost = loss + vos_dict['loss_weight'] * vos_loss

            else:
                cost = model.loss(imgs, lbls)
            opt.zero_grad()
            cost.backward()
            opt.step()

            net_l.append(cost.cpu().detach())
            tqdm_dl.set_description(f"e:{epoch} -- mean Loss: {np.mean(net_l):.4f} currloss: {cost:.4f} ")

        net_losses.append(np.mean(net_l))
        acc, acc_u, acc_l = [], [], []
        if epoch % 2 == 0:
            with torch.no_grad():
                for i, data in enumerate(v_dataloader, 0):
                    t_imgs, t_lbls, _, _ = data
                    t_imgs = t_imgs.cuda()
                    t_lbls = t_lbls.cuda()
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

                    print(f'runn acc = {accuracy:.3f}, runn acc_u = {accuracy_u:.3f}, runn acc_l = {accuracy_l:.3f}',
                          end='\r')
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
                "train/running_loss": np.mean(net_l),
            }
            if len(meta_losses_clean) != 0:
                log_dict["train/meta_loss_clean"] = np.mean(meta_losses_clean)
            # if x % 5 == 0 and x > 0:
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
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': np.mean(net_l),
                'classes': key_to_class,

            }, f"ckpts/{model_name}_{wandb.config.method}_{epoch}.pt")

            wandb.log(log_dict)
    wandb.finish()


def log_sum_exp(value, weight_energy, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    import math
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(
            F.relu(weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        # if isinstance(sum_exp, Number):
        #     return m + math.log(sum_exp)
        # else:
        return m + torch.log(sum_exp)


def do_vos(model, vos_dict):
    sum_temp = 0
    for index in range(vos_dict["num_classes"]):
        sum_temp += vos_dict["number_dict"][index]
    lr_reg_loss = torch.zeros(1).cuda()[0]
    if sum_temp == vos_dict["num_classes"] * vos_dict["sample_number"] and vos_dict["epoch"] < vos_dict["start_epoch"]:
        # maintaining an ID data queue for each class.
        target_numpy = vos_dict["target"].cpu().data.numpy()
        for index in range(len(vos_dict["target"])):
            dict_key = target_numpy[index]
            vos_dict["data_dict"][dict_key] = torch.cat((vos_dict["data_dict"][dict_key][1:],
                                                         vos_dict["output"][index].detach().view(1, -1)), 0)
    elif sum_temp == vos_dict["num_classes"] * vos_dict["sample_number"] and vos_dict["epoch"] >= vos_dict["start_epoch"]:
        target_numpy = vos_dict["target"].cpu().data.numpy()
        for index in range(len(vos_dict["target"])):
            dict_key = target_numpy[index]
            vos_dict["data_dict"][dict_key] = torch.cat((vos_dict["data_dict"][dict_key][1:],
                                                         vos_dict["output"][index].detach().view(1, -1)), 0)
        # the covariance finder needs the data to be centered.
        for index in range(vos_dict["num_classes"]):
            if index == 0:
                x_var = vos_dict["data_dict"][index] - vos_dict["data_dict"][index].mean(0)
                mean_embed_id = vos_dict["data_dict"][index].mean(0).view(1, -1)
            else:
                x_var = torch.cat((x_var, vos_dict["data_dict"][index] - vos_dict["data_dict"][index].mean(0)), 0)
                mean_embed_id = torch.cat((mean_embed_id,
                                           vos_dict["data_dict"][index].mean(0).view(1, -1)), 0)

        ## add the variance.
        temp_precision = torch.mm(x_var.t(), x_var) / len(x_var)
        temp_precision += 0.0001 * vos_dict["eye_matrix"]

        for index in range(vos_dict["num_classes"]):
            new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                mean_embed_id[index], covariance_matrix=temp_precision)
            negative_samples = new_dis.rsample((vos_dict['sample_from'],))
            prob_density = new_dis.log_prob(negative_samples)
            # breakpoint()
            # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
            # keep the data in the low density area.
            cur_samples, index_prob = torch.topk(- prob_density, vos_dict["select"])
            if index == 0:
                ood_samples = negative_samples[index_prob]
            else:
                ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
        if len(ood_samples) != 0:
            # add some gaussian noise
            energy_score_for_fg = log_sum_exp(vos_dict['pred'], vos_dict['weight_energy'], 1)
            predictions_ood = model.fc(ood_samples)
            energy_score_for_bg = log_sum_exp(predictions_ood, vos_dict['weight_energy'], 1)

            input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
            labels_for_lr = torch.cat((torch.ones(len(vos_dict['output'])).cuda(),
                                       torch.zeros(len(ood_samples)).cuda()), -1)

            criterion = torch.nn.CrossEntropyLoss()
            output1 = vos_dict["logistic_regression"](input_for_lr.view(-1, 1))
            lr_reg_loss = criterion(output1, labels_for_lr.long())
    else:
        target_numpy = vos_dict['target'].cpu().data.numpy()
        for index in range(len(vos_dict['target'])):
            dict_key = target_numpy[index]
            if vos_dict['number_dict'][dict_key] < vos_dict['sample_number']:
                vos_dict['data_dict'][dict_key][vos_dict['number_dict'][dict_key]] = vos_dict['output'][index].detach()
                vos_dict['number_dict'][dict_key] += 1
    return lr_reg_loss


if __name__ == "__main__":
    path = r'D:\Data\bigger_cls'
    for name, z in zip(['vos'], [1e-4]):
        run_net(path, False, net_method=name, lr=z)

import csv
import os

import higher as higher
import timm
import torch
from torch.utils.data import Dataset, DataLoader
from DataLoader import SyntheticDataset
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import itertools
import wandb
from torchvision.utils import draw_bounding_boxes
import transforms_segmentation as ts

torch.manual_seed(0)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    root_dir = r'Q:\classification\synth'
    rw_set = r'Q:\classification\re-weight-set'
    image_size = 512
    batch_size = 8
    epochs = 100
    # wandb.init(
    #     project="Classification_synth",
    #     config={
    #         "learning_rate": 5e-5,
    #         "gamma": 0.1,
    #         "batch_size": batch_size,
    #         "image_size": 256,
    #         "total_epochs": epochs,
    #     },
    # )
    workers = 2
    dataset = SyntheticDataset(root=root_dir,
                               transforms=ts.Compose([
                                   ts.Resize(image_size),
                                   ts.RandomHorizontalFlip(),
                                   ts.PILToTensor(),
                                   ts.ConvertImageDtype(torch.float),
                                   ts.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                               ]))

    t_split = int(dataset.__len__() * 0.8)
    v_split = dataset.__len__() - t_split
    train_set, val_set = torch.utils.data.random_split(dataset, [t_split, v_split])
    t_dataloader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=workers,collate_fn=collate_fn)
    v_dataloader = DataLoader(val_set, batch_size=batch_size,
                              shuffle=True, num_workers=workers)

    for img, lbl in t_dataloader:
        print(lbl)
#
# def test(rw_set,image_size):
#     rw_set = SyntheticDataset(root=rw_set,
#                               transforms=transforms.Compose([
#                                   transforms.Resize(image_size),
#                                   transforms.CenterCrop(image_size),
#                                   transforms.ToTensor(),
#                                   transforms.RandomHorizontalFlip()
#                               ]))
#     try:
#         assert (dataset.classes == rw_set.classes)
#     except Exception as e:
#         print(f"dataset class to index does not match. Exception: {e}")
#     t_split = int(dataset.__len__() * 0.8)
#     v_split = dataset.__len__() - t_split
#     key_to_class = dict((v, k) for k, v in dataset.classes.items())
#     train_set, val_set = torch.utils.data.random_split(dataset, [t_split, v_split])
#     t_dataloader = DataLoader(train_set, batch_size=batch_size,
#                               shuffle=True, num_workers=workers)
#     v_dataloader = DataLoader(val_set, batch_size=batch_size,
#                               shuffle=True, num_workers=workers)
#
#     rw_dataloader = DataLoader(rw_set, batch_size=batch_size // 2,
#                                shuffle=True, num_workers=workers)
#     n_classes = len(dataset.classes.keys())
#     model = timm.create_model('gluon_resnext50_32x4d', pretrained=True, num_classes=n_classes)
#     if torch.cuda.is_available():
#         model.cuda()
#         torch.backends.cudnn.benchmark = True
#     wandb.watch(model)
#     opt = torch.optim.Adam(model.parameters(), lr=1e-5)
#
#     loss = torch.nn.CrossEntropyLoss()
#     loss_no_red = torch.nn.CrossEntropyLoss(reduction='none')
#     loss_funcs = {"red": loss, "no_red": loss_no_red}
#
#     net_losses, accuracy_log = [], []
#     pseudo_labels = {}
#     reweight_cases = {}
#     meta_losses_clean = []
#     rw_loader = itertools.cycle(rw_dataloader)
#     for x in range(epochs):
#         model.train()
#         net_l, meta_losses_clean = [], []
#         tqdm_dl = tqdm(t_dataloader)
#         for i, data in enumerate(tqdm_dl, 0):
#             imgs, lbls, idxs, paths = data
#             check_and_replace_with_pseudo(imgs, pseudo_labels, lbls, i)
#
#             imgs = imgs.cuda()
#             lbls = lbls.cuda()
#             rw_img, rw_lbl, _, _ = next(rw_loader)
#             rw_sample = rw_img.cuda(), rw_lbl.cuda()
#
#             cost = do_forward_backward(imgs, lbls, rw_sample, loss_funcs, model, opt, meta_losses_clean, i, epochs,
#                                        reweight_cases, idxs, pseudo_labels, paths)
#             opt.zero_grad()
#             cost.backward()
#             opt.step()
#             net_l.append(cost.cpu().detach())
#             tqdm_dl.set_description(f"mean Loss {np.mean(net_l):.4f}")
#
#         net_losses.append(np.mean(net_l))
#         model.eval()
#         acc = []
#         for i, data in enumerate(v_dataloader, 0):
#             t_imgs, t_lbls, _, _ = data
#             t_imgs = t_imgs.cuda()
#             t_lbls = t_lbls.cuda()
#             pred = model(t_imgs.cuda())
#             predicted = torch.argmax(torch.softmax(pred, dim=1), dim=1)
#             acc.append((predicted.int() == t_lbls.int()).float())
#             accuracy = torch.cat(acc, dim=0).mean().cpu()
#             print(f'runn acc = {accuracy:.3f}', end='\r')
#             # tqdm_dl.set_postfix(f"mean Loss {}")
#         accuracy = torch.cat(acc, dim=0).mean().cpu()
#         print(f'mean val_acc = {accuracy:.3f}')
#         accuracy_log.append(np.array([x, accuracy])[None])
#         log_dict = {
#             "val/acc": float(accuracy),
#             "train/running_loss": np.mean(net_l),
#             "train/meta_epoch_loss": np.mean(meta_losses_clean),
#         }
#         if x % 5 == 0 and x > 0:
#             fig, axes = plt.subplots(1, 2, figsize=(13, 5))
#             ax1, ax2 = axes.ravel()
#             fig.suptitle(f"norm w. Prob")
#             ax1.plot(net_losses, label='net_losses')
#             ax1.set_ylabel("Losses")
#             ax1.set_xlabel("Iteration")
#             ax1.legend()
#             acc_log = np.concatenate(accuracy_log, axis=0)
#             ax2.plot(acc_log[:, 0], acc_log[:, 1])
#             ax2.set_ylabel('Accuracy')
#             ax2.set_xlabel('Iteration')
#             plt.show()
#
#             torch.save({
#                 'epoch': x,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': opt.state_dict(),
#                 'loss': np.mean(net_l),
#                 'classes': key_to_class,
#             }, f"ckpts/checkpoint_{x}.pt")
#
#             with open('pseudo_labels.csv', 'w') as csv_file:
#                 writer = csv.writer(csv_file)
#                 for key, value in pseudo_labels.items():
#                     file_pl = value[2]
#                     lbl_c = os.path.split(file_pl)[0].split('\\')[-1]
#                     lbl_pl = key_to_class[int(value[0].cpu())]
#                     if value[1] < x - 25 or lbl_c == lbl_pl:
#                         continue
#                     score_pl = float(value[3].detach().cpu())
#                     reweight_count = reweight_cases[file_pl]
#                     writer.writerow([file_pl, lbl_pl, score_pl, reweight_count])
#
#         wandb.log(log_dict)


def do_forward_backward(image, labels, rw_sample, loss_func, model, opt, m_epoch_loss, curr_epoch, epochs,
                        reweight_cases, index, pseudo_labels, paths):
    with higher.innerloop_ctx(model, opt) as (meta_model, meta_opt):

        output = meta_model(image)
        cost = loss_func['no_red'](output, labels)
        eps = torch.zeros(cost.size()).cuda()
        eps = eps.requires_grad_()

        l_f_meta = torch.sum(cost * eps)
        meta_model.zero_grad(set_to_none=True)
        meta_opt.step(l_f_meta)

        v_image, v_labels = rw_sample
        l_g_output = meta_model(v_image)
        l_g_meta = loss_func["red"](l_g_output, v_labels)
        m_epoch_loss.append(float(l_g_meta))
        grad_eps = torch.autograd.grad(l_g_meta, eps)[0].detach()

        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(-grad_eps, min=0)
        norm_c = torch.sum(w_tilde)
        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        output = model(image)
        loss = loss_func['no_red'](output, labels)
        loss = torch.sum(loss * w)

        wl = torch.le(w, 0.5 / eps.shape[0])

        ####
        ### meta annotation
        ####
        # add_reweight_cases_to_update_anno_dict(w, wl, reweight_cases, paths)
        softmax = torch.softmax(output, dim=1)
        scores, predictions = torch.max(softmax, axis=1)

        score_thresh = 1 - ((1 / 4) * np.sin((curr_epoch / epochs)))
        update_annotation(wl.cpu(), index, scores, predictions, pseudo_labels, curr_epoch, score_thresh, paths,
                          reweight_cases, w)
        return loss


def add_reweight_cases_to_update_anno_dict(w, wl, reweight_cases, ds_index):
    for index, weight in enumerate(w):
        if wl[index]:
            elem = ds_index[index]
            if elem in reweight_cases:
                tmp_cnt, tmp_loss = reweight_cases[elem]
                tmp_loss.append(float(weight))
                sample = (tmp_cnt + 1, tmp_loss)
                reweight_cases[elem] = sample
            else:
                reweight_cases[elem] = (1, [float(weight)])


def update_annotation(
        update_anno,
        idxs,
        scores,
        predictions,
        pseudo_labels,
        epoch,
        score_thresh,
        current_path,
        reweight_cases,
        weight
):
    update_names = np.array(idxs)[update_anno]
    scores = scores[update_anno]
    current_path = np.array(current_path)[update_anno]
    predictions = predictions[update_anno]
    weight = weight[update_anno]
    for ii in range(len(scores)):
        elem = current_path[ii]
        if elem in reweight_cases:
            tmp_cnt, tmp_loss = reweight_cases[elem]
            tmp_loss.append(float(weight[ii]))
            sample = (tmp_cnt + 1, tmp_loss)
            reweight_cases[elem] = sample
        else:
            reweight_cases[elem] = (1, [float(weight[ii])])

        if scores[ii] > score_thresh:
            key = f"{update_names[ii]}"
            pseudo_labels[key] = (predictions[ii].detach(), epoch, current_path[ii], scores[ii])


def check_and_replace_with_pseudo(index, pseudo_labels, labels, epoch):
    for ij, x in enumerate(index):
        pseudo_label_id = f"{x}"
        if pseudo_label_id in pseudo_labels.keys():
            label_made_at_epoch = pseudo_labels[pseudo_label_id][1]
            if label_made_at_epoch < epoch - 25:
                return
            labels.data[ij] = pseudo_labels[pseudo_label_id][0]


def do_forward_backward_standard(imgs, lbls, tqdm_dl, model, loss, opt, net_l):
    pred = model(imgs)
    cost = loss(pred, lbls)
    opt.zero_grad()
    cost.backward()
    opt.step()
    net_l.append(cost.cpu().detach())
    tqdm_dl.set_description(f"mean Loss {np.mean(net_l):.4f}")


if __name__ == "__main__":
    main()

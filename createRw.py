import higher as higher
import timm
import torch
from torch.utils.data import Dataset, DataLoader
from DataLoader import ShippingLabClassification
import torchvision.datasets as dset
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image

torch.manual_seed(0)
import os


def main():
    root_dir = r'Q:\classification\updated-ds'
    image_size = 128
    batch_size = 64
    workers = 4
    dataset = ShippingLabClassification(root_dir=root_dir,
                                        transform=transforms.Compose([
                                            transforms.Resize(image_size),
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))

    rw_set = ShippingLabClassification(root_dir=root_dir,
                                       transform=transforms.Compose([
                                           transforms.Resize(image_size),
                                           transforms.CenterCrop(image_size),
                                           transforms.ToTensor(),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))

    t_split = int(dataset.__len__() * 0.8)
    v_split = dataset.__len__() - t_split
    train_set, val_set = torch.utils.data.random_split(dataset, [t_split, v_split])
    t_dataloader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=workers)

    tqdm_dl = tqdm(t_dataloader)
    rw_dir = r"Q:\newData\re-weight-set2"
    coll_dict = {}
    for j, data in enumerate(tqdm_dl, 0):
        _, lbls, _, paths = data
        for i, (img, l) in enumerate(zip(paths, lbls)):
            lbl = img.split('\\')[-2]
            if lbl in coll_dict:
                if coll_dict[lbl] > 30:
                    continue
            class_path = os.path.join(rw_dir, lbl)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            fn = os.path.join(class_path, f'{j}_{i}.jpg')
            img = Image.open(img)
            img.save(fn)
            if lbl not in coll_dict:
                coll_dict[lbl] = 1
            else:
                coll_dict[lbl] += 1


if __name__ == "__main__":
    main()

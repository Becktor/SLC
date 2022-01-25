import higher as higher
import timm
import torch
import glob
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
    root_dir = r'Q:\classification\newData'
    update_dir = r'Q:\classification\re-weight-set'

    matching_dict = {0: 'Sailboat(D)',
                     1: 'Sailboat(U)',
                     2: 'buoy_red',
                     3: 'buoy_green',
                     4: 'harbour',
                     5: 'human',
                     6: 'large_commercial_vessel',
                     7: 'leisure_craft',
                     8: 'small_medium_fishing_boat'
                     }
    update_dir = r"Q:\newData\updated-ds"
    import csv
    orig_path = r"Q:\classification\allData\*\*.jpg"
    image_paths = glob.glob(orig_path)
    with open('pseudo_labels.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        tqdm_rows = tqdm(iter(csv_reader))
        update_dict = {}
        for j, data in enumerate(tqdm_rows, 0):
            if len(data) == 0:
                continue
            tmp_val = data[1].split(",")
            lbl = matching_dict[int(tmp_val[0][-1])]
            from_epoch = int(tmp_val[2])
            img_path = tmp_val[3].replace('\'', '').replace('\\\\', '\\')[1:-1]
            update_dict[img_path] = lbl

        for img_path in tqdm(image_paths):
            lbl = os.path.split(img_path)[-2]
            if img_path in update_dict:
                lbl = update_dict[img_path]

            class_path = os.path.join(update_dir, lbl)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            name = os.path.split(img_path)[-1]
            fn = os.path.join(class_path, name)
            img_path = Image.open(img_path)
            img_path.save(fn)


if __name__ == "__main__":
    main()

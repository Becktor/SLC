import shutil

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
from math import isclose

torch.manual_seed(0)
import os

from colorthief import ColorThief


# get the dominant color
# import the necessary packages

# import the necessary packages


def main():
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
    old = r"Q:\classification\allData\*\*.jpg"
    up = r"Q:\classification\updated-ds\*\*.jpg"
    g = [os.path.split(x)[-1] for x in glob.glob(up)]
    oldies = glob.glob(old)
    for val in tqdm(oldies):
        idx = os.path.split(val)[-1]
        if idx not in g:
            p = r'Q:\classification\updated-ds'
            lbl = os.path.split(val)[0].split('\\')[-1]
            folder = os.path.join(p, lbl)
            fn = os.path.join(folder, idx)
            #print(val, fn)
            shutil.copyfile(val, fn)


if __name__ == "__main__":
    main()

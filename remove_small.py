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
    old = r"Q:\classification\updated-ds_no_small\*\*.jpg"
    oldies = glob.glob(old)
    cntr=0
    for val in tqdm(oldies):
        size = Image.open(val).size
        if size[0] * size[1] <= 256:
            os.remove(val)
            cntr+=1
    print(cntr)

if __name__ == "__main__":
    main()

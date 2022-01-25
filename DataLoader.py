from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch
import numpy as np
from skimage import io, transform
import os
import torchvision.datasets as dset


class ShippingLabClassification:
    def __init__(self, root_dir="", transform=transforms):
        self.dataset = dset.ImageFolder(root=root_dir)
        self.transform = transform
        self.paths, self.labels = zip(*self.dataset.samples)
        self.classes = self.dataset.class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        path, target = self.paths[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index, path

    def __len__(self):
        return len(self.labels)
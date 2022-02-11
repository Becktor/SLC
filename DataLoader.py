from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch
import numpy as np
from skimage import io, transform
import os
import torchvision.datasets as dset
from future.utils import raise_from
import csv


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


def color_to_L(cols_to_lbls):
    in_L = {}
    for name, col_to_lbl in cols_to_lbls:
        R = col_to_lbl[0]
        G = col_to_lbl[1]
        B = col_to_lbl[2]
        L = np.round(min(255, R * 299 / 1000 + G * 587 / 1000 + B * 114 / 1000)).astype(int)
        in_L[L] = name
    return in_L


class SyntheticDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        self.mask_to_lbl_path = os.path.join(root, "color_to_lbl.csv")
        self.lbl_name_to_id = {}
        self.mask_labels = self._parse_mask_to_lbl()
        self.nothing = []

    def _parse_mask_to_lbl(self):
        td = {}

        with open(self.mask_to_lbl_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                line_head = line.pop(0)
                line_tail = "".join(line)
                parsed = line_tail.replace("[", "").replace("]", "").replace("'", "").split(" ")
                reshaped = np.array(parsed).reshape(-1, 4)
                arr = []
                for x in reshaped:
                    if x[0] not in self.lbl_name_to_id:
                        self.lbl_name_to_id[x[0]] = len(self.lbl_name_to_id)
                    arr.append((x[0], x[1:].astype(int)))
                td[line_head] = arr
        return td

    def __getitem__(self, idx):

        # load images and masks
        img_path = os.path.join(self.root, "img", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        col_to_lbl = self.mask_labels[self.imgs[idx]]
        l_to_lbl = color_to_L(col_to_lbl)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)

        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        area = 0
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            val = [xmin, ymin, xmax, ymax]
            boxes.append(val)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([self.lbl_name_to_id[l_to_lbl[idd]] for idd in obj_ids], dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        if len(boxes.shape) == 1:
            boxes = torch.as_tensor([[-1, -1, -1, -1]])
            labels = torch.as_tensor([-1])
            masks = torch.unsqueeze(torch.as_tensor(np.zeros_like(mask), dtype=torch.uint8), 0)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area

        boxes = target["boxes"]
        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            # print the first degenerate box
            bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
        return img, target

    def __len__(self):
        return len(self.imgs)




class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "img", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

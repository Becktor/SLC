import os
from torchvision import transforms
import torch
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm
from PIL import Image
import fiftyone as fo
from DataLoader import FiftyOneTorchDataset
import matplotlib.pyplot as plt
import os
import numpy as np


def main():
    """
    Data loaders
    """
    datasets = []
    name = "unreal_coco"
    for x in ['train', 'val']:
        dataset_dir = os.path.join(r'Q:\classification\ue_gen', x)
        imgs = os.path.join(dataset_dir, "img")
        json = os.path.join(dataset_dir, f"{x}.json")
        t_name = f"{name}_{x}"
        if not fo.dataset_exists(t_name):
            # Create the dataset
            dataset = fo.Dataset.from_dir(
                dataset_type=fo.types.COCODetectionDataset,
                data_path=imgs,
                labels_path=json,
                name=t_name,
            )
            dataset.persistent = True
        else:
            dataset = fo.load_dataset(t_name)
        datasets.append((x, dataset))

    data_labels = {0: "buoy_green",
                   1: "buoy_red",
                   2: "large_commercial_vessel",
                   3: "small_medium_fishing_boat",
                   4: "human",
                   5: "kayak",
                   6: "leisure_craft",
                   7: "sailboat(D)",
                   8: "sailboat(U)"}

    path = r'Q:/classification/unreal_cls'
    train_ds = FiftyOneTorchDataset(datasets[0][1])
    i = 0
    for img, target in tqdm(train_ds):
        for j in range(target['boxes'].shape[0]):
            #image = Image.fromarray(np.uint8(img[:, :, :3] * 255))
            crop_range = target['boxes'][j].numpy().astype(int)
            crop = img.crop(crop_range)
            cls = int(target['labels'][j])
            class_path = os.path.join(path, data_labels[cls])
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            fn = os.path.join(class_path, f'{i}.jpg')
            crop.save(fn)
            i += 1

            if False:
                plt.imshow(crop)
                plt.show()
                plt.imshow(x.img)
                ax = plt.gca()
                rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none', alpha=0.1)
                ax.add_patch(rect)
                plt.show()
                print('tt')


if __name__ == "__main__":
    main()

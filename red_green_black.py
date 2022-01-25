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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import cv2


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


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
    update_dir = r"Q:\newData\updated-ds2"
    orig_path = r"Q:\classification\allData\buoy_red\*.jpg"
    image_paths = glob.glob(orig_path)
    import random
    random.shuffle(image_paths)
    for img_path in tqdm(image_paths):

        # img = Image.open(img_path).convert('RGB')

        # color_thief = ColorThief(img_path)
        # cm = color_thief.get_color(quality=1)
        img = Image.open(img_path).convert('RGB')
        image = np.asarray(Image.open(img_path))
        # show our image
        plot = False
        if plot:
            plt.figure()
            plt.axis("off")
            plt.imshow(image)
            plt.show()
        # build a histogram of clusters and then create a figure
        # representing the number of pixels labeled to each color
        # cluster the pixel intensities
        image = image.reshape((-1, 3))
        clust = 10
        clt = KMeans(n_clusters=clust)
        clt.fit(image)
        if plot:
            hist = centroid_histogram(clt)
            bar = plot_colors(hist, clt.cluster_centers_)
            # show our color bart
            plt.figure()
            plt.axis("off")
            plt.imshow(bar)
            plt.show()
        cent = clt.cluster_centers_
        cent_col = [0, 2, 0]
        for x in cent:
            if isclose(x[0], x[1], abs_tol=10):
                cent_col[0] += 1
            elif x[0] < x[1]:
                cent_col[1] += 1
            else:
                cent_col[2] += 1

        # img_hsv = img.convert('HSV')
        #img_np = np.asarray(image) / 255
        # img_np[:, :, 0] = img_np[:, :, 0] - img_np[:, :, 2]
        # img_np[:, :, 1] = img_np[:, :, 1] - img_np[:, :, 2]
        # img_np[:, :, 2] = 0
        # img_np[img_np < 0] = 0
        #cm = np.mean(img_np, axis=(0, 1))

        if cent_col[0] >= clust*0.8 or cent_col[1] == cent_col[2]:
            lbl = "buoy_random"
        elif cent_col[1] < cent_col[2]:
            lbl = "buoy_red"
        else:
            lbl = "buoy_green"
        class_path = os.path.join(update_dir, lbl)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        name = os.path.split(img_path)[-1]
        fn = os.path.join(class_path, name)
        img.save(fn)


if __name__ == "__main__":
    main()
